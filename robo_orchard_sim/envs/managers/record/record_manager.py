# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
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
from __future__ import annotations
import copy
import os
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generic, Mapping, Protocol, Sequence

from google.protobuf.timestamp_pb2 import Timestamp
from mcap_protobuf.writer import Writer
from robo_orchard_core.envs.managers.manager_base import (
    EnvType_co,
    ManagerBase,
    ManagerBaseCfg,
)
from typing_extensions import TypeAlias, TypeVar

from robo_orchard_sim.envs.managers.record.record_controller import (
    NoOpRecordControllerCfg,
    RecordController,
    RecordControllerCfg,
)
from robo_orchard_sim.envs.managers.record.record_term_base import (
    RecordTermBase,
    RecordTermBaseCfg,
)
from robo_orchard_sim.utils.config import ClassType_co


class RecordMessage(Protocol):
    def write_message(self, mcap_writer: Writer, topic: str) -> None: ...


RecordTermCfgType_co = TypeVar(
    "RecordTermCfgType_co",
    bound=RecordTermBaseCfg,
    covariant=True,
    default=RecordTermBaseCfg,
)

MsgsType: TypeAlias = dict[
    str, list[RecordMessage] | list[list[RecordMessage]]
]


class McapRecorder:
    def __init__(self, file_path: str, recorder_id: int):
        self._file_path = file_path
        self._file = None
        self._writer = None
        self._running = False
        self._id = recorder_id

    def start(self) -> None:
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
        self._file = open(self._file_path, "wb")
        self._writer = Writer(self._file)
        self._running = True

    def end(self) -> None:
        if self._writer is not None:
            self._writer.finish()
            self._writer = None
        if self._file is not None:
            self._file.close()
            self._file = None
        self._running = False

    @property
    def writer(self) -> Writer:
        if self._writer is None:
            raise RuntimeError("MCAP writer has not been started.")
        return self._writer

    @property
    def id(self) -> int:
        return self._id


class RecordManager(ManagerBase[EnvType_co, "RecordManagerCfg"]):
    def __init__(self, cfg: "RecordManagerCfg", env: EnvType_co):
        super().__init__(cfg, env)
        self._cfg = cfg
        self._env = env
        self._term_cfgs: Mapping[str, RecordTermBaseCfg] = self.cfg.terms
        self._terms: Dict[str, RecordTermBase] = self.cfg.create_terms(env)
        self._controller: RecordController = self.cfg.controller(env=env)
        self._mcap_writers: list[McapRecorder] = []
        self._episode = 0
        self._running = False
        self._last_obs: dict[str, Any] | None = None
        self._step_user_data: dict[str, Any] = {}
        self._episode_user_data: dict[str, Any] = {}
        self._episode_start_time: datetime | None = None
        self._record_start_time: datetime | None = None
        self._record_step_count = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._episode = 0
        self._running = False
        self._last_obs = None
        self._step_user_data = {}
        self._episode_user_data = {}
        self._episode_start_time = None
        self._record_start_time = None
        self._record_step_count = 0
        for term in self._terms.values():
            term.reset(env_ids)

    def record_post_reset(
        self, obs: dict[str, Any] | None, cur_time: datetime
    ) -> None:
        self._last_obs = copy.deepcopy(obs) if obs is not None else None
        self._episode_start_time = cur_time
        decision = self._controller.on_post_reset(obs)
        if decision.start and not self._running:
            self._start_recording(start_time=cur_time)
        if self._running:
            self._record_terms(
                self._merge_record_input(obs),
                record_modes=("post_reset",),
                timestamp_fn=lambda _term_cfg: self._get_episode_timestamp(
                    step_count=0
                ),
            )
        if decision.stop and self._running:
            self._stop_recording()

    def record_step(self, obs: dict[str, Any] | None) -> None:
        self._last_obs = copy.deepcopy(obs) if obs is not None else None
        decision = self._controller.on_post_step(obs)
        if decision.start and not self._running:
            self._start_recording(
                start_time=self._get_episode_datetime(self._env.step_count)
            )
        if self._running:
            self._record_terms(
                self._merge_record_input(obs),
                record_modes=("step",),
                timestamp_fn=lambda _term_cfg: self._get_episode_timestamp(
                    self._env.step_count
                ),
                step_filtered=True,
            )
            self._record_step_count += 1
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
        if self._running and self._episode_start_time is not None:
            self._record_terms(
                self._merge_record_input(self._last_obs),
                record_modes=("pre_reset", "once"),
                timestamp_fn=self._get_pre_reset_timestamp,
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

    def _merge_record_input(
        self, obs: dict[str, Any] | None
    ) -> dict[str, Any | dict[str, Any]]:
        data: dict[str, Any | dict[str, Any]] = {}
        if obs is not None:
            data.update(copy.deepcopy(obs))
        data["step"] = copy.deepcopy(self._step_user_data)
        data["episode"] = copy.deepcopy(self._episode_user_data)
        return data

    def _record_terms(
        self,
        data: dict[str, Any | dict[str, Any]],
        *,
        record_modes: tuple[str, ...],
        timestamp_fn: Callable[[RecordTermBaseCfg], Timestamp],
        step_filtered: bool = False,
    ) -> None:
        for term_name, term in self._terms.items():
            term_cfg = self._term_cfgs[term_name]
            if term_cfg.record_mode not in record_modes:
                continue
            if step_filtered and not self._should_record_step(term_cfg.fps):
                continue
            try:
                msgs = term(data, timestamp_fn(term_cfg))
            except KeyError:
                continue
            if msgs:
                self._record_msg(msgs)

    def _should_record_step(self, fps: float) -> bool:
        step_dt = getattr(self._env, "step_dt", None)
        if step_dt in (None, 0):
            return True
        sample = max(1, int(round(1 / (fps * step_dt))))
        return self._record_step_count % sample == 0

    def _start_recording(
        self,
        prefix: str = "",
        start_time: datetime | None = None,
    ) -> None:
        self._record_step_count = 0
        self._step_user_data = {}
        self._episode_user_data = {}
        self._record_start_time = start_time
        self._init_mcap_writer(prefix=prefix)
        for recorder in self._mcap_writers:
            recorder.start()
        self._running = True

    def _stop_recording(self) -> None:
        for recorder in self._mcap_writers:
            recorder.end()
        self._running = False
        self._record_step_count = 0
        self._step_user_data = {}
        self._episode_user_data = {}
        self._record_start_time = None
        self._episode += 1

    def close(self) -> None:
        if self._running:
            self._stop_recording()

    def _record_msg(self, msgs: MsgsType) -> None:
        for key, values in msgs.items():
            for msg, mcap_writer in zip(
                values, self._mcap_writers, strict=False
            ):
                if isinstance(msg, list):
                    for nested_msg in msg:
                        nested_msg.write_message(mcap_writer.writer, key)
                else:
                    msg.write_message(mcap_writer.writer, key)

    def _init_mcap_writer(self, prefix: str = "") -> None:
        self._mcap_writers.clear()
        env_ids = range(self._env.num_envs)
        record_prefix = prefix or f"episode{self._episode}"
        for env_id in env_ids:
            filename = os.path.join(
                self._cfg.file_path,
                record_prefix,
                f"env{env_id}_data.mcap",
            )
            self._mcap_writers.append(McapRecorder(filename, env_id))

    def _to_proto_timestamp(self, cur_time: datetime) -> Timestamp:
        timestamp_proto = Timestamp()
        timestamp_proto.FromDatetime(cur_time)
        return timestamp_proto

    def _get_episode_datetime(self, step_count: int) -> datetime:
        if self._episode_start_time is None:
            raise RuntimeError("Episode start time has not been initialized.")
        step_dt = getattr(self._env, "step_dt", None)
        if step_dt is None:
            raise RuntimeError(
                "Env step_dt is required for record timestamps."
            )
        return self._episode_start_time + timedelta(
            seconds=step_count * step_dt
        )

    def _get_episode_timestamp(self, step_count: int) -> Timestamp:
        return self._to_proto_timestamp(self._get_episode_datetime(step_count))

    def _get_pre_reset_timestamp(
        self,
        term_cfg: RecordTermBaseCfg,
    ) -> Timestamp:
        if term_cfg.record_mode == "once":
            if self._record_start_time is None:
                raise RuntimeError(
                    "Record start time has not been initialized."
                )
            return self._to_proto_timestamp(self._record_start_time)
        return self._get_episode_timestamp(self._env.step_count)

    @property
    def running(self) -> bool:
        return self._running


class RecordManagerCfg(
    ManagerBaseCfg[RecordManager], Generic[RecordTermCfgType_co]
):
    class_type: ClassType_co[RecordManager] = RecordManager

    file_path: str
    terms: Mapping[str, RecordTermCfgType_co]
    controller: RecordControllerCfg = NoOpRecordControllerCfg()

    def create_terms(self, env: Any) -> Dict[str, RecordTermBase]:
        return {
            term_name: term_cfg(env=env)
            for term_name, term_cfg in self.terms.items()
        }
