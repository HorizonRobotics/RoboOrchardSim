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

"""Tests for the shared parallel MCAP recording backend."""

import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from google.protobuf.timestamp_pb2 import Timestamp
from mcap_protobuf.reader import read_protobuf_messages

from robo_orchard_sim.ext.envs.managers.record import (
    ManualRecordControllerCfg,
    ParallelRecordManager,
    ParallelRecordManagerCfg,
    RecordManagerCfg,
    RecordTermBase,
    RecordTermBaseCfg,
)
from robo_orchard_sim.ext.envs.managers.record.mcap import (
    McapDictTermCfg,
    McapJointsTermCfg,
)
from robo_orchard_sim.ext.envs.managers.record.recording_snapshot import (
    RecordingFrameSnapshot,
)
from robo_orchard_sim.ext.envs.managers.record.recording_worker import (
    RecordingWorker,
)
from robo_orchard_sim.utils.config import ClassType_co


class _BlockingRecordTerm(
    RecordTermBase[Any, "_BlockingRecordTermCfg", dict[str, list]]
):
    entered = threading.Event()
    release = threading.Event()

    def __call__(
        self,
        data: dict[str, Any | dict[str, Any]],
        ts: Timestamp,
    ) -> dict[str, list]:
        del data, ts
        self.entered.set()
        if not self.release.wait(timeout=5.0):
            raise TimeoutError("Timed out waiting to release blocking term.")
        return {}

    def reset(self, env_ids=None) -> None:
        del env_ids


class _BlockingRecordTermCfg(RecordTermBaseCfg[_BlockingRecordTerm]):
    class_type: ClassType_co[_BlockingRecordTerm] = _BlockingRecordTerm


def _manager_cfg(output_dir: Path) -> RecordManagerCfg:
    return RecordManagerCfg(
        file_path=str(output_dir),
        terms={
            "joints": McapJointsTermCfg(
                topic="/robot/joint_states",
                fps=30.0,
                position_key="/robot/joint_position",
                joint_name_prefix="joint",
            ),
            "meta": McapDictTermCfg(
                topic="/meta_data",
                fps=1.0,
                key="episode/meta_dict",
                record_mode="once",
            ),
        },
        controller=ManualRecordControllerCfg(),
    )


def _snapshot(sequence_id: int) -> RecordingFrameSnapshot:
    return RecordingFrameSnapshot(
        recording_step=sequence_id * 2,
        timestamp=datetime.now(),
        observation={
            "/robot": {
                "joint_position": torch.tensor([[float(sequence_id), 2.0]])
            }
        },
        observation_builder=None,
        ready_events=(),
        build_duration_ms=0.25,
    )


def _env(step_dt: float = 1.0 / 60.0):
    return type(
        "Env",
        (),
        {
            "num_envs": 1,
            "step_dt": step_dt,
            "step_count": 0,
        },
    )()


def test_recording_worker_normal_stop_writes_valid_mcap(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "records"
    path = output_dir / "episode" / "env0_data.mcap"
    worker = RecordingWorker(
        _manager_cfg(output_dir),
        step_dt=1.0 / 60.0,
        queue_capacity=4,
    )

    assert worker.start(
        prefix="episode",
        step_count=0,
        start_time=datetime.now(),
        episode_user_data={"meta_dict": {"source": "test"}},
    )
    assert worker.try_enqueue(_snapshot(1))
    assert worker.request_stop(
        path,
        episode_user_data={
            "meta_dict": {
                "task_success": 1.0,
                "task_progress": 1.0,
            }
        },
    )
    worker.close(flush_timeout=5.0)

    messages = list(read_protobuf_messages(path))
    topics = {message.topic for message in messages}
    assert topics == {"/robot/joint_states", "/meta_data"}
    metadata = next(
        message.proto_msg
        for message in messages
        if message.topic == "/meta_data"
    )
    assert dict(metadata)["task_success"] == 1.0
    assert dict(metadata)["task_progress"] == 1.0


def test_recording_worker_queue_overflow_drops_complete_frame(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "records"
    path = output_dir / "episode" / "env0_data.mcap"
    worker = RecordingWorker(
        _manager_cfg(output_dir),
        step_dt=1.0 / 60.0,
        queue_capacity=1,
    )

    assert worker.start(
        prefix="episode",
        step_count=0,
        start_time=datetime.now(),
        episode_user_data={"meta_dict": {"source": "test"}},
    )
    accepted = [worker.try_enqueue(_snapshot(index)) for index in range(32)]
    assert worker.request_stop(path)
    worker.close(flush_timeout=5.0)

    stats = worker.statistics()
    assert not all(accepted)
    assert stats.dropped_queue_frames == accepted.count(False)


def test_parallel_record_manager_manual_episode_writes_shared_terms(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "records"
    env = _env()
    manager = ParallelRecordManager(
        ParallelRecordManagerCfg(
            file_path=str(output_dir),
            terms=_manager_cfg(output_dir).terms,
            controller=ManualRecordControllerCfg(),
            queue_capacity=2,
        ),
        env,
    )
    start_time = datetime.now()

    manager.record_post_reset(_snapshot(0).observation, start_time)
    manager.set_episode_user_data({"meta_dict": {"source": "parallel"}})
    assert manager.start_record(prefix="episode")
    env.step_count = 1
    manager.record_step(_snapshot(1).observation)
    manager.update_episode_user_data({"meta_dict": {"task_success": 1.0}})
    manager.record_pre_reset()
    manager.close()

    path = output_dir / "episode" / "env0_data.mcap"
    messages = list(read_protobuf_messages(path))
    topics = {message.topic for message in messages}
    assert topics == {"/robot/joint_states", "/meta_data"}


def test_parallel_record_manager_manual_start_after_steps_uses_single_offset(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "records"
    env = _env(step_dt=0.1)
    manager = ParallelRecordManager(
        ParallelRecordManagerCfg(
            file_path=str(output_dir),
            terms=_manager_cfg(output_dir).terms,
            controller=ManualRecordControllerCfg(),
            queue_capacity=2,
        ),
        env,
    )
    episode_start = datetime(2026, 1, 1, 12, 0, 0)
    env.step_count = 3

    manager.record_post_reset(_snapshot(0).observation, episode_start)
    manager.set_episode_user_data({"meta_dict": {"source": "parallel"}})
    assert manager.start_record(prefix="episode")
    assert manager.stop_record()
    manager.close()

    metadata = next(
        message
        for message in read_protobuf_messages(
            output_dir / "episode" / "env0_data.mcap"
        )
        if message.topic == "/meta_data"
    )
    assert metadata.log_time == episode_start + timedelta(seconds=0.3)


def test_parallel_record_manager_start_while_finalizing_returns_false(
    tmp_path: Path,
) -> None:
    _BlockingRecordTerm.entered.clear()
    _BlockingRecordTerm.release.clear()
    output_dir = tmp_path / "records"
    env = _env()
    manager = ParallelRecordManager(
        ParallelRecordManagerCfg(
            file_path=str(output_dir),
            terms={
                "blocking": _BlockingRecordTermCfg(
                    topic="/blocking",
                    fps=60.0,
                )
            },
            controller=ManualRecordControllerCfg(),
            queue_capacity=2,
        ),
        env,
    )

    try:
        manager.record_post_reset({}, datetime.now())
        assert manager.start_record(prefix="episode1")
        env.step_count = 1
        manager.record_step({})
        assert _BlockingRecordTerm.entered.wait(timeout=2.0)
        assert manager.stop_record()

        assert not manager.start_record(prefix="episode2")

        _BlockingRecordTerm.release.set()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if manager.start_record(prefix="episode2"):
                break
            time.sleep(0.01)
        assert manager.running
    finally:
        _BlockingRecordTerm.release.set()
        if manager.running:
            manager.stop_record()
        manager.close()
