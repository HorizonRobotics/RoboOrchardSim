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

"""Record manager with bounded CPU snapshots and background MCAP writing."""

from __future__ import annotations
import copy
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generic, Literal, Sequence

from robo_orchard_core.envs.managers.manager_base import (
    EnvType_co,
    ManagerBase,
)

from robo_orchard_sim.ext.envs.managers.record.cpu_snapshot import (
    CpuSnapshotCopies,
)
from robo_orchard_sim.ext.envs.managers.record.record_controller import (
    ManualRecordController,
    RecordController,
)
from robo_orchard_sim.ext.envs.managers.record.record_manager import (
    RecordManagerCfg,
    RecordTermCfgType_co,
)
from robo_orchard_sim.ext.envs.managers.record.recording_snapshot import (
    RecordingFrameSnapshot,
)
from robo_orchard_sim.ext.envs.managers.record.recording_statistics import (
    RecordingStatisticsSnapshot,
)
from robo_orchard_sim.ext.envs.managers.record.recording_worker import (
    RecordingWorker,
)
from robo_orchard_sim.utils.config import ClassType_co


class ParallelRecordManager(
    ManagerBase[EnvType_co, "ParallelRecordManagerCfg"]
):
    """Keep RecordManager behavior while encoding and writing off-thread."""

    def __init__(self, cfg: "ParallelRecordManagerCfg", env: EnvType_co):
        super().__init__(cfg, env)
        self._controller: RecordController = cfg.controller(env=env)
        self._worker = RecordingWorker(
            RecordManagerCfg(
                file_path=cfg.file_path,
                terms=cfg.terms,
            ),
            step_dt=float(env.step_dt),
            queue_capacity=cfg.queue_capacity,
            num_envs=int(env.num_envs),
        )
        self._snapshot_buffers = [{} for _ in range(cfg.queue_capacity)]
        self._episode = 0
        self._running = False
        self._last_obs: dict[str, Any] | None = None
        self._step_user_data: dict[str, Any] = {}
        self._episode_user_data: dict[str, Any] = {}
        self._episode_start_time: datetime | None = None
        self._active_prefix: str | None = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        del env_ids
        self._last_obs = None
        self._step_user_data = {}
        self._episode_user_data = {}

    def record_post_reset(
        self,
        obs: dict[str, Any] | None,
        cur_time: datetime,
    ) -> None:
        self._last_obs = obs
        self._episode_start_time = cur_time
        decision = self._controller.on_post_reset(obs)
        if decision.start and not self._running:
            self._start_recording(start_time=cur_time)
        if self._running and obs is not None:
            self._enqueue_snapshot(
                obs,
                event="post_reset",
                step_count=0,
                timestamp=cur_time,
            )
        if decision.stop and self._running:
            self._stop_recording()

    def record_step(self, obs: dict[str, Any] | None) -> None:
        self._last_obs = obs
        decision = self._controller.on_post_step(obs)
        if decision.start and not self._running:
            self._start_recording(
                start_time=self._get_episode_datetime(self._env.step_count)
            )
        if self._running and obs is not None:
            self._enqueue_snapshot(
                obs,
                event="step",
                step_count=self._env.step_count,
                timestamp=self._get_episode_datetime(self._env.step_count),
            )
        self._step_user_data = {}
        if decision.stop and self._running:
            self._stop_recording()

    def record_pre_reset(self) -> None:
        decision = self._controller.on_pre_reset(self._last_obs)
        if decision.start and not self._running:
            self._start_recording(
                prefix="pre_reset",
                start_time=self._get_episode_datetime(self._env.step_count),
            )
        if decision.stop and self._running:
            self._stop_recording()

    def set_step_user_data(self, data: dict[str, Any]) -> None:
        self._step_user_data = copy.deepcopy(data)

    def update_step_user_data(self, data: dict[str, Any]) -> None:
        self._step_user_data.update(copy.deepcopy(data))

    def set_episode_user_data(self, data: dict[str, Any]) -> None:
        self._episode_user_data = copy.deepcopy(data)

    def update_episode_user_data(self, data: dict[str, Any]) -> None:
        self._episode_user_data.update(copy.deepcopy(data))

    def start_record(self, *, prefix: str | None = None) -> bool:
        """Start manual recording unless active output is still finalizing."""
        if not isinstance(self._controller, ManualRecordController):
            raise RuntimeError(
                "ParallelRecordManager.start_record() requires "
                "ManualRecordController."
            )
        if self._running:
            return False
        if not self._start_recording(
            prefix=prefix or "",
            start_time=self._get_episode_datetime(self._env.step_count),
            skip_if_finalizing=True,
        ):
            return False
        self._controller.on_manual_start()
        return True

    def stop_record(self) -> bool:
        """Manually enqueue the recording finalization barrier."""
        if not isinstance(self._controller, ManualRecordController):
            raise RuntimeError(
                "ParallelRecordManager.stop_record() requires "
                "ManualRecordController."
            )
        if not self._running:
            return False
        self._stop_recording()
        return True

    def close(self) -> None:
        if self._running:
            self._stop_recording()
        self._worker.close(self.cfg.flush_timeout)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def statistics(self) -> RecordingStatisticsSnapshot:
        """Return the latest asynchronous recording counters."""
        return self._worker.statistics()

    def _start_recording(
        self,
        prefix: str = "",
        start_time: datetime | None = None,
        skip_if_finalizing: bool = False,
    ) -> bool:
        if start_time is None:
            start_time = self._get_episode_datetime(self._env.step_count)
        if skip_if_finalizing and self._worker.stopping:
            return False
        active_prefix = prefix or f"episode{self._episode}"
        started = self._worker.start(
            prefix=active_prefix,
            step_count=self._env.step_count,
            start_time=start_time,
            episode_user_data=copy.deepcopy(self._episode_user_data),
        )
        if not started:
            raise RuntimeError("Asynchronous recording worker rejected start.")
        self._running = True
        self._active_prefix = active_prefix
        return True

    def _stop_recording(self) -> None:
        active_prefix = self._active_prefix
        if active_prefix is None:
            raise RuntimeError("Recording prefix is unavailable during stop.")
        path = Path(self.cfg.file_path) / active_prefix / "env0_data.mcap"
        if not self._worker.request_stop(
            path,
            episode_user_data=copy.deepcopy(self._episode_user_data),
        ):
            raise RuntimeError("Asynchronous recording worker rejected stop.")
        self._running = False
        self._active_prefix = None
        self._step_user_data = {}
        self._episode_user_data = {}
        self._episode += 1

    def _enqueue_snapshot(
        self,
        obs: dict[str, Any],
        *,
        event: Literal["post_reset", "step"],
        step_count: int,
        timestamp: datetime,
    ) -> None:
        slot_id = self._worker.reserve_frame()
        if slot_id is None:
            return
        started = time.perf_counter()
        try:
            copies = CpuSnapshotCopies(self._snapshot_buffers[slot_id])
            observation = copies.clone(obs, "observation")
            ready_events = copies.finalize()
        except BaseException:
            self._worker.cancel_reservation(slot_id)
            raise
        snapshot = RecordingFrameSnapshot(
            recording_step=step_count,
            timestamp=timestamp,
            observation=observation,
            observation_builder=None,
            ready_events=ready_events,
            build_duration_ms=(time.perf_counter() - started) * 1000.0,
            event=event,
            step_user_data=copy.deepcopy(self._step_user_data),
            episode_user_data=copy.deepcopy(self._episode_user_data),
        )
        self._worker.enqueue_reserved(snapshot, slot_id)

    def _get_episode_datetime(self, step_count: int) -> datetime:
        if self._episode_start_time is None:
            raise RuntimeError("Episode start time has not been initialized.")
        return self._episode_start_time + timedelta(
            seconds=step_count * float(self._env.step_dt)
        )


class ParallelRecordManagerCfg(
    RecordManagerCfg[RecordTermCfgType_co],
    Generic[RecordTermCfgType_co],
):
    """Configuration for the shared asynchronous recording backend."""

    class_type: ClassType_co[ParallelRecordManager] = ParallelRecordManager
    queue_capacity: int = 4
    flush_timeout: float = 30.0
