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

from robo_orchard_sim.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.envs.managers.actions.articulation.joint_base import (
    ArticulationJointActionTerm,
    ArticulationJointActionTermCfg,
)
from robo_orchard_sim.utils.config import ClassType_co

__all__ = [
    "ArticulationJointVelocityActionTerm",
    "ArticulationJointVelocityActionTermCfg",
]


class ArticulationJointVelocityActionTerm(
    ArticulationJointActionTerm[
        IsaacEnvType_co, "ArticulationJointVelocityActionTermCfg"
    ],
):
    def __init__(
        self,
        cfg: "ArticulationJointVelocityActionTermCfg",
        env: IsaacEnvType_co,
    ):
        super().__init__(cfg, env)
        # use default joint velocity as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_vel[
                :, self.joint_ids
            ].clone()

    def apply(self):
        # set position targets
        self._asset.set_joint_velocity_target(
            self.processed_actions, joint_ids=self.joint_ids
        )


class ArticulationJointVelocityActionTermCfg(ArticulationJointActionTermCfg):
    class_type: ClassType_co[ArticulationJointVelocityActionTerm] = (
        ArticulationJointVelocityActionTerm
    )

    use_default_offset: bool = True
    """Whether to use default joint velocities configured in the
    articulation asset as offset. Defaults to True.

    This overrides the settings from :attr:`offset` if set to True.
    """
