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

import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp
from robo_orchard_schemas.sensor_msgs.JointState_pb2 import (
    JointState,
    MultiJointStateStamped,
)

from robo_orchard_sim.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.envs.managers.record import (
    RecordTermBase,
    RecordTermBaseCfg,
)
from robo_orchard_sim.envs.managers.record.mcap.message import Message
from robo_orchard_sim.utils.config import ClassType_co

ReturnType = dict[str, list[Message]]


class McapJointsTerm(
    RecordTermBase[IsaacEnvType_co, "McapJointsTermCfg", ReturnType]
):
    def __init__(self, cfg: "McapJointsTermCfg", env: IsaacEnvType_co):
        super().__init__(cfg, env)
        self._cfg = cfg
        self.name = cfg.joint_name_prefix
        self.joint_id_offset = cfg.joint_id_offset

    def __call__(
        self, data: dict[str, Any | dict[str, Any]], ts: Timestamp
    ) -> ReturnType:
        """The implementation of the record term.

        All subclasses should implement this method to return the observation.

        """
        joint_pos = self._try_parse_joint_array(data, self._cfg.position_key)
        joint_vel = self._try_parse_joint_array(data, self._cfg.velocity_key)
        joint_eff = self._try_parse_joint_array(data, self._cfg.effort_key)

        reference = next(
            (
                array
                for array in (joint_pos, joint_vel, joint_eff)
                if array is not None
            ),
            None,
        )
        if reference is None:
            return {}

        if joint_pos is None:
            joint_pos = np.zeros_like(reference)
        if joint_vel is None:
            joint_vel = np.zeros_like(reference)
        if joint_eff is None:
            joint_eff = np.zeros_like(reference)

        # Ensure dimensions of joint_pos, joint_vel, and joint_eff are equal
        if not (joint_pos.shape == joint_vel.shape == joint_eff.shape):
            raise ValueError(
                "Dimensions of joint_pos, joint_vel, and joint_eff "
                "are not equal."
            )

        if joint_pos.ndim != 2:
            raise ValueError(
                "Joint data must have shape (num_instances, num_joints)."
            )

        if self._cfg.joint_ids is not None:
            joint_ids = self._cfg.joint_ids
        else:
            joint_ids = list(range(joint_pos.shape[1]))

        joint_pos = joint_pos[:, joint_ids]
        joint_vel = joint_vel[:, joint_ids]
        joint_eff = joint_eff[:, joint_ids]

        msgs: ReturnType = {}
        batch_size, joint_num = joint_pos.shape

        batch_msgs: list[Message] = []
        for batch_id in range(batch_size):
            states = []
            for joint_offset, selected_joint_id in enumerate(joint_ids):
                states.append(
                    JointState(
                        name=self.name
                        + f"{self.joint_id_offset + selected_joint_id + 1}",
                        position=joint_pos[batch_id, joint_offset],
                        velocity=joint_vel[batch_id, joint_offset],
                        effort=joint_eff[batch_id, joint_offset],
                    )
                )
            batch_msgs.append(
                Message(
                    data=MultiJointStateStamped(
                        timestamp=ts,
                        states=states,
                    ),
                    log_time=ts,
                    pub_time=ts,
                )
            )

        msgs[self._cfg.topic] = batch_msgs

        return msgs

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the observation term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        """
        pass

    def _try_parse_joint_array(
        self,
        data: dict[str, Any | dict[str, Any]],
        key: str,
    ) -> np.ndarray | None:
        try:
            value = self._parse_data_from_dict(data, key)[key]
        except KeyError:
            return None
        return value.cpu().numpy()


class McapJointsTermCfg(RecordTermBaseCfg):
    """Configuration class for McapJointsTerm."""

    class_type: ClassType_co[McapJointsTerm] = McapJointsTerm

    position_key: str
    """The key for the joint position data in the input data dictionary."""

    velocity_key: str = "/"
    """The key for the joint velocity data in the input data dictionary.
        If not provided, the joint velocity will be set to zero."""

    effort_key: str = "/"
    """The key for the joint effort data in the input data dictionary.
        If not provided, the joint effort will be set to zero."""

    joint_name_prefix: str = "joint"
    """The prefix for the joint names in the output messages."""

    joint_ids: Sequence[int] | None = None
    """The list of joint ids to record. If None, all joints will be
        recorded."""

    joint_id_offset: int = 0
    """The offset to add to joint IDs for naming and topic generation."""
