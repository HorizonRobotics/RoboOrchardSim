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

"""Embodiment profile metadata for the Franka Panda robot."""

from __future__ import annotations

import numpy as np
import torch

from robo_orchard_sim.controllers.curobo_planner.curobo import (
    ArticulationJointCuroboTrajPlannerCfg,
    CSpaceCfg,
    RobotCfg,
    RobotKinematicsCfg,
)
from robo_orchard_sim.ext.models.assets.asset_cfg import (
    NV_ISAACLAB_DIR,
    ORCHARD_ASSET,
)
from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ManipulatorProfile,
    RobotInfoCfg,
)
from robo_orchard_sim.orchard_env.embodiments.planner_cfg import (
    DEFAULT_IK_SOLVER_CFG,
    DEFAULT_MOTION_GEN_CFG,
    DEFAULT_MOTION_GEN_PLAN_CFG,
)

__all__ = [
    "FRANKA_PANDA_ROBOT_INFO_CFGS",
]

_FRANKA_GRIPPER_OPEN_VAL = [0.04, 0.04]
_FRANKA_GRIPPER_CLOSE_VAL = [0.0, 0.0]

_FRANKA_STANDARD_TCP_TO_ROBOT_EE = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.1],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


_FRANKA_PLANNER = ArticulationJointCuroboTrajPlannerCfg(
    device="cuda:0",
    robot=RobotCfg(
        kinematics=RobotKinematicsCfg(
            usd_path=f"{NV_ISAACLAB_DIR}/Robots/FrankaEmika/panda_instanceable.usd",  # noqa: E501
            usd_robot_root=f"{NV_ISAACLAB_DIR}/Robots/FrankaEmika/",
            urdf_path=f"{ORCHARD_ASSET}/ROBOTS/FRANKA/franka_panda.urdf",
            asset_root_path=f"{ORCHARD_ASSET}",
            base_link="panda_link0",
            ee_link="panda_hand",
            collision_sphere_buffer=0.005,
            collision_link_names=[
                "panda_link0",
                "panda_link1",
                "panda_link2",
                "panda_link3",
                "panda_link4",
                "panda_link5",
                "panda_link6",
                "panda_link7",
                "panda_hand",
                "panda_leftfinger",
                "panda_rightfinger",
                "attached_object",
            ],
            lock_joints={
                "panda_finger_joint1": 0.08,
                "panda_finger_joint2": 0.08,
            },
            extra_collision_spheres={"attached_object": 50},
            collision_spheres="/3rdparty/curobo/src/curobo/content/configs/robot/spheres/franka_mesh.yml",  # noqa: E501
            self_collision_ignore={
                "panda_link0": ["panda_link1", "panda_link2"],
                "panda_link1": [
                    "panda_link2",
                    "panda_link3",
                    "panda_link4",
                ],
                "panda_link2": ["panda_link3", "panda_link4"],
                "panda_link3": ["panda_link4", "panda_link6"],
                "panda_link4": [
                    "panda_link5",
                    "panda_link6",
                    "panda_link7",
                    "panda_link8",
                ],
                "panda_link5": [
                    "panda_link6",
                    "panda_link7",
                    "panda_hand",
                    "panda_leftfinger",
                    "panda_rightfinger",
                ],
                "panda_link6": [
                    "panda_link7",
                    "panda_hand",
                    "attached_object",
                    "panda_leftfinger",
                    "panda_rightfinger",
                ],
                "panda_link7": [
                    "panda_hand",
                    "attached_object",
                    "panda_leftfinger",
                    "panda_rightfinger",
                ],
                "panda_hand": [
                    "panda_leftfinger",
                    "panda_rightfinger",
                    "attached_object",
                ],
                "panda_leftfinger": [
                    "panda_rightfinger",
                    "attached_object",
                ],
                "panda_rightfinger": [
                    "panda_leftfinger",
                    "attached_object",
                ],
            },
            self_collision_buffer={
                "panda_link0": 0.1,
                "panda_link1": 0.05,
                "panda_link2": 0.0,
                "panda_link3": 0.0,
                "panda_link4": 0.0,
                "panda_link5": 0.0,
                "panda_link6": 0.0,
                "panda_link7": 0.0,
                "panda_hand": 0.02,
                "panda_leftfinger": 0.01,
                "panda_rightfinger": 0.01,
                "attached_object": 0.0,
            },
            extra_links={
                "attached_object": {
                    "parent_link_name": "panda_hand",
                    "link_name": "attached_object",
                    "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
                    "joint_type": "FIXED",
                    "joint_name": "attach_joint",
                }
            },
            use_global_cumul=True,
            cspace=CSpaceCfg(
                joint_names=[
                    "panda_joint1",
                    "panda_joint2",
                    "panda_joint3",
                    "panda_joint4",
                    "panda_joint5",
                    "panda_joint6",
                    "panda_joint7",
                    "panda_finger_joint1",
                    "panda_finger_joint2",
                ],
                retract_config=torch.tensor(
                    [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0, 0.04, 0.04]
                ),
                null_space_weight=torch.tensor(
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                ),
                cspace_distance_weight=torch.tensor(
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                ),
                max_jerk=500.0,
                max_acceleration=15.0,
                position_limit_clip=0.0,
            ),
        )
    ),
    motion_gen=DEFAULT_MOTION_GEN_CFG,
    motion_gen_plan=DEFAULT_MOTION_GEN_PLAN_CFG,
    ik_solver=DEFAULT_IK_SOLVER_CFG,
)


FRANKA_PANDA_ROBOT_INFO_CFGS = {
    "main_arm": RobotInfoCfg(
        robot_name="robots/franka_panda",
        manipulator_name="main_arm",
        gripper_open_val=_FRANKA_GRIPPER_OPEN_VAL,
        gripper_close_val=_FRANKA_GRIPPER_CLOSE_VAL,
        t_standard_tcp_to_robot_ee=_FRANKA_STANDARD_TCP_TO_ROBOT_EE,
        manipulator_profile=ManipulatorProfile(
            arm_joint_names=("panda_joint[1-7]",),
            gripper_joint_names=(
                "panda_finger_joint1",
                "panda_finger_joint2",
            ),
            body_names=(
                "panda_link0",
                "panda_link1",
                "panda_link2",
                "panda_link3",
                "panda_link4",
                "panda_link5",
                "panda_link6",
                "panda_link7",
                "panda_hand",
            ),
            base_body_name="panda_link0",
            ee_body_name="panda_hand",
        ),
        planner=_FRANKA_PLANNER,
    ),
}
