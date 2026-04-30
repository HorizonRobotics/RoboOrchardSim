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

from typing import Any, Sequence

from google.protobuf.struct_pb2 import Struct
from google.protobuf.timestamp_pb2 import Timestamp

from robo_orchard_sim.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.envs.managers.record import (
    RecordTermBase,
    RecordTermBaseCfg,
)
from robo_orchard_sim.envs.managers.record.mcap.message import Message
from robo_orchard_sim.utils.config import ClassType_co

ReturnType = dict[str, list[Message]]


class McapDictTerm(
    RecordTermBase[IsaacEnvType_co, "McapDictTermCfg", ReturnType]
):
    def __init__(self, cfg: "McapDictTermCfg", env: IsaacEnvType_co):
        super().__init__(cfg, env)
        self._cfg = cfg
        self._env = env

    def __call__(
        self, data: dict[str, Any | dict[str, Any]], ts: Timestamp
    ) -> ReturnType:
        """The implementation of the record term.

        All subclasses should implement this method to return the observation.

        """
        dict_data = self._parse_data_from_dict(data, self._cfg.key)[
            self._cfg.key
        ]

        if isinstance(dict_data, dict):
            dict_data = [dict_data]

        if len(dict_data) != self._env.num_envs:
            raise ValueError(
                f"The length of dict_data {len(dict_data)} is not equal to "
                f"num_envs {self._env.num_envs}."
            )

        env_msgs = []
        for i in range(self._env.num_envs):
            struct_message = Struct()
            data = dict_data[i]

            if not isinstance(data, dict):
                raise ValueError(f"Data for env {i} is not a dict: {data}.")

            struct_message.update(dict_data[i])

            msg = Message(
                data=struct_message,
                log_time=ts,
                pub_time=ts,
            )
            env_msgs.append(msg)

        msgs = {self._cfg.topic: env_msgs}
        return msgs

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the observation term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        """
        pass


class McapDictTermCfg(RecordTermBaseCfg):
    """The configuration class for the McapImageTerm.

    Args:
        input data should be list[dict] or dict
        class_type: The class type of the record term.
        env_type: The environment type.
        env_ids: The environment ids. Defaults to None, in which case
            all environments are considered.

    """

    class_type: ClassType_co[McapDictTerm] = McapDictTerm

    key: str
