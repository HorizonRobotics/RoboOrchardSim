# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Bounded producer-consumer worker for shared MCAP recording."""

from __future__ import annotations
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from robo_orchard_sim.ext.envs.managers.record.record_controller import (
    ManualRecordControllerCfg,
)
from robo_orchard_sim.ext.envs.managers.record.record_manager import (
    RecordManager,
    RecordManagerCfg,
)
from robo_orchard_sim.ext.envs.managers.record.recording_snapshot import (
    RecordingFrameSnapshot,
)
from robo_orchard_sim.ext.envs.managers.record.recording_statistics import (
    RecordingStatistics,
    RecordingStatisticsSnapshot,
)


@dataclass(slots=True)
class _CommandResult:
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _StartCommand:
    prefix: str
    step_count: int
    start_time: datetime
    episode_user_data: dict[str, object]
    result: _CommandResult


@dataclass(frozen=True, slots=True)
class _FrameCommand:
    snapshot: RecordingFrameSnapshot
    slot_id: int


@dataclass(frozen=True, slots=True)
class _StopCommand:
    path: Path
    episode_user_data: dict[str, object] | None
    result: _CommandResult


@dataclass(frozen=True, slots=True)
class _ResetTimelineCommand:
    current_time: datetime
    step_count: int


@dataclass(frozen=True, slots=True)
class _ShutdownCommand:
    result: _CommandResult


_WorkerCommand = (
    _StartCommand
    | _FrameCommand
    | _StopCommand
    | _ResetTimelineCommand
    | _ShutdownCommand
)


class RecordingWorker:
    """Own one RecordManager and MCAP writer on a background thread."""

    def __init__(
        self,
        manager_cfg: RecordManagerCfg,
        *,
        step_dt: float,
        queue_capacity: int,
        num_envs: int = 1,
        command_timeout: float = 5.0,
    ) -> None:
        self._manager_cfg = manager_cfg
        self._step_dt = step_dt
        self._queue_capacity = queue_capacity
        self._num_envs = num_envs
        self._command_timeout = command_timeout
        self._commands: queue.Queue[_WorkerCommand] = queue.Queue()
        self._free_frame_slots: queue.Queue[int] = queue.Queue(
            maxsize=queue_capacity
        )
        for slot_id in range(queue_capacity):
            self._free_frame_slots.put(slot_id)
        self._state_lock = threading.Lock()
        self._pending_frames = 0
        self._state: Literal["idle", "recording", "stopping", "closed"] = (
            "idle"
        )
        self._failure: RuntimeError | None = None
        self._stop_result: _CommandResult | None = None
        self._statistics = RecordingStatistics()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="robo-orchard-mcap-recorder",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(command_timeout):
            raise RuntimeError("MCAP recording worker failed to initialize.")
        self.raise_if_failed()

    @property
    def recording(self) -> bool:
        """Return whether the producer currently accepts frame snapshots."""
        self._refresh_state()
        with self._state_lock:
            return self._state == "recording"

    @property
    def stopping(self) -> bool:
        """Return whether an episode is draining and finalizing."""
        self._refresh_state()
        with self._state_lock:
            return self._state == "stopping"

    def start(
        self,
        *,
        prefix: str,
        step_count: int,
        start_time: datetime,
        episode_user_data: dict[str, object],
    ) -> bool:
        """Start an episode after the worker acknowledges writer creation."""
        self._refresh_state()
        self.raise_if_failed()
        with self._state_lock:
            if self._state != "idle":
                return False
        self._statistics.reset()
        result = _CommandResult()
        self._commands.put(
            _StartCommand(
                prefix=prefix,
                step_count=step_count,
                start_time=start_time,
                episode_user_data=episode_user_data,
                result=result,
            )
        )
        self._wait_for_result(result, "start recording")
        with self._state_lock:
            self._state = "recording"
        return True

    def try_enqueue(self, snapshot: RecordingFrameSnapshot) -> bool:
        """Enqueue a frame without blocking the Isaac simulation thread."""
        slot_id = self.reserve_frame()
        if slot_id is None:
            return False
        self.enqueue_reserved(snapshot, slot_id)
        return True

    def reserve_frame(self) -> int | None:
        """Reserve one bounded queue slot before building a large snapshot."""
        self._refresh_state()
        self.raise_if_failed()
        with self._state_lock:
            if self._state != "recording":
                return None
        try:
            return self._free_frame_slots.get_nowait()
        except queue.Empty:
            self._statistics.record_queue_drop()
            return None

    def enqueue_reserved(
        self,
        snapshot: RecordingFrameSnapshot,
        slot_id: int,
    ) -> None:
        """Enqueue a completed snapshot using an earlier reservation."""
        with self._state_lock:
            self._pending_frames += 1
            depth = self._pending_frames
        self._statistics.record_enqueued(
            depth,
            snapshot.build_duration_ms,
        )
        self._commands.put(_FrameCommand(snapshot=snapshot, slot_id=slot_id))

    def cancel_reservation(self, slot_id: int) -> None:
        """Release a slot when snapshot construction cannot complete."""
        self._free_frame_slots.put_nowait(slot_id)

    def request_stop(
        self,
        path: Path,
        episode_user_data: dict[str, object] | None = None,
    ) -> bool:
        """Stop accepting frames and enqueue a FIFO finalization barrier."""
        self._refresh_state()
        self.raise_if_failed()
        with self._state_lock:
            if self._state != "recording":
                return False
            self._state = "stopping"
        result = _CommandResult()
        self._stop_result = result
        self._commands.put(
            _StopCommand(
                path=path,
                episode_user_data=episode_user_data,
                result=result,
            )
        )
        return True

    def reset_timeline(self, current_time: datetime, step_count: int) -> None:
        """Reset episode time after any earlier stop barrier is processed."""
        self.raise_if_failed()
        self._commands.put(
            _ResetTimelineCommand(
                current_time=current_time,
                step_count=step_count,
            )
        )

    def record_incomplete_camera(self) -> None:
        """Count one sample skipped before it reached the queue."""
        self._statistics.record_incomplete_camera()

    def statistics(self) -> RecordingStatisticsSnapshot:
        """Return the latest producer and worker counters."""
        return self._statistics.snapshot()

    def raise_if_failed(self) -> None:
        """Raise a worker failure on the Isaac thread."""
        with self._state_lock:
            failure = self._failure
        if failure is not None:
            raise failure

    def close(self, flush_timeout: float) -> None:
        """Finalize an active episode and terminate the worker."""
        self._refresh_state()
        with self._state_lock:
            state = self._state
        if state == "recording":
            raise RuntimeError(
                "RecordingWorker.close() requires request_stop() first."
            )
        if state == "stopping" and self._stop_result is not None:
            if not self._stop_result.done.wait(flush_timeout):
                raise TimeoutError(
                    "Timed out while flushing asynchronous MCAP frames."
                )
            self._refresh_state()
        self.raise_if_failed()
        result = _CommandResult()
        self._commands.put(_ShutdownCommand(result=result))
        self._wait_for_result(result, "shut down recording worker")
        self._thread.join(timeout=self._command_timeout)
        if self._thread.is_alive():
            raise RuntimeError("MCAP recording worker did not exit.")
        with self._state_lock:
            self._state = "closed"

    def _refresh_state(self) -> None:
        stop_result = self._stop_result
        if stop_result is None or not stop_result.done.is_set():
            return
        if stop_result.error is not None:
            self._remember_failure(stop_result.error)
        with self._state_lock:
            if self._state == "stopping":
                self._state = "idle"
        self._stop_result = None

    def _wait_for_result(
        self,
        result: _CommandResult,
        operation: str,
    ) -> None:
        if not result.done.wait(self._command_timeout):
            raise TimeoutError(f"Timed out trying to {operation}.")
        if result.error is not None:
            raise RuntimeError(f"Unable to {operation}.") from result.error
        self.raise_if_failed()

    def _run(self) -> None:
        env = SimpleNamespace(
            num_envs=self._num_envs,
            step_dt=self._step_dt,
            step_count=0,
        )
        manager: RecordManager | None = None
        current_result: _CommandResult | None = None
        try:
            manager_cfg = RecordManagerCfg(
                file_path=self._manager_cfg.file_path,
                terms=self._manager_cfg.terms,
                controller=ManualRecordControllerCfg(),
            )
            manager = RecordManager(manager_cfg, env)
            manager.record_post_reset(None, datetime.now())
            self._ready.set()
            while True:
                command = self._commands.get()
                current_result = getattr(command, "result", None)
                if isinstance(command, _StartCommand):
                    env.step_count = command.step_count
                    manager.record_post_reset(None, command.start_time)
                    if not manager.start_record(
                        prefix=command.prefix,
                        start_time=command.start_time,
                    ):
                        raise RuntimeError("RecordManager rejected start.")
                    manager.set_episode_user_data(command.episode_user_data)
                    command.result.done.set()
                    current_result = None
                elif isinstance(command, _FrameCommand):
                    started = time.perf_counter()
                    try:
                        for event in command.snapshot.ready_events:
                            event.synchronize()
                        env.step_count = command.snapshot.recording_step
                        if command.snapshot.step_user_data is not None:
                            manager.set_step_user_data(
                                command.snapshot.step_user_data
                            )
                        if command.snapshot.episode_user_data is not None:
                            manager.set_episode_user_data(
                                command.snapshot.episode_user_data
                            )
                        observation = command.snapshot.build_observation()
                        if command.snapshot.event == "post_reset":
                            manager.record_post_reset(
                                observation,
                                command.snapshot.timestamp,
                            )
                        else:
                            manager.record_step(
                                observation,
                                command.snapshot.timestamp,
                            )
                        self._statistics.record_written(
                            (time.perf_counter() - started) * 1000.0
                        )
                    finally:
                        self._release_frame_slot(command.slot_id)
                elif isinstance(command, _StopCommand):
                    if command.episode_user_data is not None:
                        manager.set_episode_user_data(
                            command.episode_user_data
                        )
                    manager.record_pre_reset()
                    print(
                        f"Recording stopped: {command.path}",
                        flush=True,
                    )
                    self._print_statistics()
                    command.result.done.set()
                    current_result = None
                elif isinstance(command, _ResetTimelineCommand):
                    env.step_count = command.step_count
                    manager.record_post_reset(None, command.current_time)
                elif isinstance(command, _ShutdownCommand):
                    manager.close()
                    command.result.done.set()
                    current_result = None
                    return
        except BaseException as exc:
            if manager is not None:
                try:
                    manager.close()
                except BaseException:
                    pass
            self._remember_failure(exc)
            self._ready.set()
            if current_result is not None:
                current_result.error = exc
                current_result.done.set()
            self._signal_pending_results(exc)

    def _release_frame_slot(self, slot_id: int) -> None:
        with self._state_lock:
            self._pending_frames -= 1
        self._free_frame_slots.put_nowait(slot_id)

    def _signal_pending_results(self, error: BaseException) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            if isinstance(command, _FrameCommand):
                self._release_frame_slot(command.slot_id)
                continue
            result = getattr(command, "result", None)
            if result is not None:
                result.error = error
                result.done.set()

    def _remember_failure(self, error: BaseException) -> None:
        details = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        failure = RuntimeError(
            "Asynchronous MCAP recording worker failed:\n" + details
        )
        with self._state_lock:
            if self._failure is None:
                self._failure = failure

    def _print_statistics(self) -> None:
        stats = self._statistics.snapshot()
        print(
            "Async recording statistics: "
            f"enqueued={stats.enqueued_frames}, "
            f"written={stats.written_frames}, "
            f"queue_drops={stats.dropped_queue_frames}, "
            f"incomplete_cameras={stats.incomplete_camera_frames}, "
            f"max_queue={stats.max_queue_depth}/{self._queue_capacity}, "
            f"snapshot_ms={stats.average_snapshot_ms:.1f}, "
            f"worker_ms={stats.average_worker_ms:.1f}",
            flush=True,
        )
