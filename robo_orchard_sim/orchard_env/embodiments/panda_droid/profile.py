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

"""Embodiment profile metadata for the Panda Droid robot."""

from __future__ import annotations

import numpy as np

from robo_orchard_sim.ext.models.assets.asset_cfg import ORCHARD_ASSET
from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ManipulatorProfile,
    RobotInfoCfg,
)
from robo_orchard_sim.orchard_env.embodiments.franka_panda.profile import (
    _FRANKA_PLANNER,
)

__all__ = [
    "PANDA_DROID_ROBOT_INFO_CFGS",
]


_PANDA_DROID_GRIPPER_OPEN_VAL = [0.0]
_PANDA_DROID_GRIPPER_CLOSE_VAL = [0.7854]

_PANDA_DROID_STANDARD_TCP_TO_ROBOT_EE = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.12],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

_PANDA_DROID_PLANNER = _FRANKA_PLANNER.model_copy(deep=True)
_PANDA_DROID_PLANNER.robot.kinematics.urdf_path = (
    f"{ORCHARD_ASSET}/ROBOTS/FRANKA/franka_droid.urdf"
)


PANDA_DROID_ROBOT_INFO_CFGS = {
    "main_arm": RobotInfoCfg(
        robot_name="robots/panda_droid",
        manipulator_name="main_arm",
        gripper_open_val=_PANDA_DROID_GRIPPER_OPEN_VAL,
        gripper_close_val=_PANDA_DROID_GRIPPER_CLOSE_VAL,
        t_standard_tcp_to_robot_ee=_PANDA_DROID_STANDARD_TCP_TO_ROBOT_EE,
        manipulator_profile=ManipulatorProfile(
            arm_joint_names=("panda_joint[1-7]",),
            gripper_joint_names=("finger_joint",),
            body_names=(
                "panda_link0",
                "panda_link1",
                "panda_link2",
                "panda_link3",
                "panda_link4",
                "panda_link5",
                "panda_link6",
                "panda_link7",
                "base_link",
            ),
            base_body_name="panda_link0",
            ee_body_name="base_link",
        ),
        planner=_PANDA_DROID_PLANNER,
    ),
}
