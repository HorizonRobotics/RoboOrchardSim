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

_PANDA_DROID_ASSET_ROOT = f"{ORCHARD_ASSET}/ROBOTS/FRANKA"
_PANDA_DROID_USD_PATH = (
    f"{_PANDA_DROID_ASSET_ROOT}/franka_panda_robotiq_flange.usd"
)
_PANDA_DROID_USD_ROBOT_ROOT = "/panda"

_PANDA_DROID_GRIPPER_OPEN_VAL = [0.0]
_PANDA_DROID_GRIPPER_CLOSE_VAL = [0.7854]

_PANDA_DROID_STANDARD_TCP_TO_ROBOT_EE = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.1],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# _PANDA_DROID_PLANNER = ArticulationJointCuroboTrajPlannerCfg(
#     device="cuda:0",
#     robot=RobotCfg(
#         kinematics=RobotKinematicsCfg(
#             usd_path=_PANDA_DROID_USD_PATH,
#             usd_robot_root=_PANDA_DROID_USD_ROBOT_ROOT,
#             use_usd_kinematics=True,
#             asset_root_path=_PANDA_DROID_ASSET_ROOT,
#             base_link="panda_link0",
#             ee_link="base_link",
#             collision_sphere_buffer=0.005,
#             lock_joints=None,
#             self_collision_ignore={
#                 "panda_link0": ["panda_link1", "panda_link2"],
#                 "panda_link1": [
#                     "panda_link2",
#                     "panda_link3",
#                     "panda_link4",
#                 ],
#                 "panda_link2": ["panda_link3", "panda_link4"],
#                 "panda_link3": ["panda_link4", "panda_link6"],
#                 "panda_link4": [
#                     "panda_link5",
#                     "panda_link6",
#                     "panda_link7",
#                 ],
#                 "panda_link5": [
#                     "panda_link6",
#                     "panda_link7",
#                     "base_link",
#                 ],
#                 "panda_link6": [
#                     "panda_link7",
#                     "base_link",
#                 ],
#                 "panda_link7": [
#                     "base_link",
#                 ],
#                 "base_link": [
#                     "left_outer_knuckle",
#                     "right_outer_knuckle",
#                 ],
#                 "left_outer_knuckle": ["left_outer_finger"],
#                 "right_outer_knuckle": ["right_outer_finger"],
#             },
#             self_collision_buffer={
#                 "panda_link0": 0.1,
#                 "panda_link1": 0.05,
#                 "panda_link2": 0.0,
#                 "panda_link3": 0.0,
#                 "panda_link4": 0.0,
#                 "panda_link5": 0.0,
#                 "panda_link6": 0.0,
#                 "panda_link7": 0.0,
#                 "base_link": 0.02,
#                 "left_outer_knuckle": 0.01,
#                 "left_outer_finger": 0.01,
#                 "left_inner_finger": 0.01,
#                 "left_inner_knuckle": 0.01,
#                 "right_outer_knuckle": 0.01,
#                 "right_outer_finger": 0.01,
#                 "right_inner_finger": 0.01,
#                 "right_inner_knuckle": 0.01,
#             },
#             use_global_cumul=True,
#             cspace=CSpaceCfg(
#                 joint_names=[
#                     "panda_joint1",
#                     "panda_joint2",
#                     "panda_joint3",
#                     "panda_joint4",
#                     "panda_joint5",
#                     "panda_joint6",
#                     "panda_joint7",
#                 ],
#                 retract_config=torch.tensor(
#                     [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]
#                 ),
#                 null_space_weight=torch.tensor(
#                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#                 ),
#                 cspace_distance_weight=torch.tensor(
#                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#                 ),
#                 max_jerk=500.0,
#                 max_acceleration=15.0,
#                 position_limit_clip=0.0,
#             ),
#         )
#     ),
#     motion_gen=DEFAULT_MOTION_GEN_CFG,
#     motion_gen_plan=DEFAULT_MOTION_GEN_PLAN_CFG,
#     ik_solver=DEFAULT_IK_SOLVER_CFG,
# )


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
        planner=_FRANKA_PLANNER,
    ),
}
