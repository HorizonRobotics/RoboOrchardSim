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

"""Embodiment profile metadata for the dual-arm Piper robot."""

from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ManipulatorProfile,
    RobotInfoCfg,
)

DUALARM_PIPER_ROBOT_INFO_CFGS = {
    "left_arm": RobotInfoCfg(
        robot_name="robots/dualarm_piper",
        manipulator_name="left_arm",
        manipulator_profile=ManipulatorProfile(
            arm_joint_names=("left_joint[1-6]",),
            gripper_joint_names=("left_joint7", "left_joint8"),
            body_names=(
                "left_base_link",
                "left_link1",
                "left_link2",
                "left_link3",
                "left_link4",
                "left_link5",
                "left_link6",
            ),
            base_body_name="left_base_link",
            ee_body_name="left_link6",
        ),
    ),
    "right_arm": RobotInfoCfg(
        robot_name="robots/dualarm_piper",
        manipulator_name="right_arm",
        manipulator_profile=ManipulatorProfile(
            arm_joint_names=("right_joint[1-6]",),
            gripper_joint_names=("right_joint7", "right_joint8"),
            body_names=(
                "right_base_link",
                "right_link1",
                "right_link2",
                "right_link3",
                "right_link4",
                "right_link5",
                "right_link6",
            ),
            base_body_name="right_base_link",
            ee_body_name="right_link6",
        ),
    ),
}
