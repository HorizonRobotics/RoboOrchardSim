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
from abc import ABCMeta, abstractmethod
from typing import Any, Generic, Literal, Sequence

from robo_orchard_core.envs.managers.manager_base import EnvType_co
from robo_orchard_core.envs.managers.manager_term_base import (
    ManagerTermBase,
    ManagerTermBaseCfg,
)
from typing_extensions import TypeVar

RecordTermType_co = TypeVar(
    "RecordTermType_co", bound="RecordTermBase", covariant=True
)
RecordTermCfgType_co = TypeVar(
    "RecordTermCfgType_co", bound="RecordTermBaseCfg", covariant=True
)
MsgType = TypeVar("MsgType", default=Any)


class RecordTermBase(
    ManagerTermBase[EnvType_co, RecordTermCfgType_co],
    Generic[EnvType_co, RecordTermCfgType_co, MsgType],
    metaclass=ABCMeta,
):
    def __init__(self, cfg: RecordTermCfgType_co, env: EnvType_co):
        super().__init__(cfg, env)

    @abstractmethod
    def __call__(
        self, data: dict[str, Any | dict[str, Any]], ts: Any
    ) -> MsgType:
        raise NotImplementedError

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        raise NotImplementedError

    def _parse_data_from_dict(
        self, data: dict[str, Any | dict[str, Any]], key: str
    ) -> dict[str, Any]:
        def split_str(string: str) -> list[str]:
            raw = string.split("/")

            parts = []
            for seg in raw:
                if seg == "":
                    if not parts or not parts[-1].startswith("/"):
                        parts.append("/")
                else:
                    if parts and parts[-1] == "/":
                        parts[-1] = "/" + seg
                    else:
                        parts.append(seg)
            return parts

        res = {}
        current = data
        for k in split_str(key):
            if k not in current:
                raise KeyError(
                    f"Key {k} not found in the dictionary while parsing "
                    f"path {key}. Available keys: {list(current.keys())}"
                )
            current = current[k]
        res[key] = current
        return res


class RecordTermBaseCfg(
    ManagerTermBaseCfg[RecordTermType_co],
    Generic[RecordTermType_co],
):
    topic: str
    fps: float
    record_mode: Literal["step", "post_reset", "pre_reset", "once"] = "step"
