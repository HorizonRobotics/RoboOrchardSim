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

import numpy as np
import torch

from robo_orchard_sim.controllers.curobo_planner.curobo import (
    ArticulationJointCuroboTrajPlannerCfg,
    CSpaceCfg,
    IKSolverCfg,
    MotionGenCfg,
    MotionGenPlanConfig,
    RobotCfg,
    RobotKinematicsCfg,
)
from robo_orchard_sim.models.assets.asset_cfg import ORCHARD_ASSET
from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ManipulatorProfile,
    RobotInfoCfg,
)

_PIPER_GRIPPER_OPEN_VAL = [0.05, -0.05]
_PIPER_GRIPPER_CLOSE_VAL = [0.0, 0.0]
_PIPER_ASSET_ROOT = f"{ORCHARD_ASSET}/ROBOTS/dualarm_piper"
_PIPER_URDF_PATH = (
    f"{_PIPER_ASSET_ROOT}/"
    "piper_description_dualarm_new_textured_large_stroke.urdf"
)

# Transformation matrix from standard TCP (tool center point) to robot end-effector (EE) frame. # noqa: E501
# -0.11 is the offset from standard TCP to link6 # noqa: E501
_PIPER_STANDARD_TCP_TO_ROBOT_EE = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.11],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

_PIPER_PLANNER = ArticulationJointCuroboTrajPlannerCfg(
    device="cuda:0",
    robot=RobotCfg(
        kinematics=RobotKinematicsCfg(
            urdf_path=_PIPER_URDF_PATH,
            asset_root_path=_PIPER_ASSET_ROOT,
            base_link="left_base_link",
            ee_link="left_link6",
            collision_sphere_buffer=0.005,
            self_collision_ignore={
                "left_link2": ["left_link3", "left_link1"],
                "left_link3": ["left_link4"],
                "left_link4": ["left_link5"],
                "left_link5": ["left_link6"],
            },
            self_collision_buffer={
                "left_link2": 0,
                "left_link3": 0,
                "left_link4": 0,
                "left_link5": 0,
                "left_link6": 0,
            },
            use_global_cumul=True,
            cspace=CSpaceCfg(
                joint_names=[
                    "left_joint1",
                    "left_joint2",
                    "left_joint3",
                    "left_joint4",
                    "left_joint5",
                    "left_joint6",
                ],
                retract_config=torch.tensor([0.0, 2.2, -2.0, 0.0, 0.0, 0.0]),
                null_space_weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                cspace_distance_weight=torch.tensor(
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                ),
                max_jerk=500.0,
                max_acceleration=15.0,
                position_limit_clip=0.0,
            ),
        )
    ),
    motion_gen=MotionGenCfg(
        interpolation_dt=1 / 30,
        collision_activation_distance=0.02,
        interpolation_steps=5000,
        num_ik_seeds=30,
        num_trajopt_seeds=12,
        grad_trajopt_iters=50,
        evaluate_interpolated_trajectory=True,
        trajopt_tsteps=32,
        use_cuda_graph=True,
        num_graph_seeds=12,
        self_collision_check=True,
        maximum_trajectory_time=15,
        jerk_scale=1.0,
        finetune_dt_scale=0.98,
        collision_cache=dict(
            obb=30,
            mesh=10,
        ),
    ),
    motion_gen_plan=MotionGenPlanConfig(
        enable_graph=False,
        enable_opt=True,
        use_nn_ik_seed=True,
        need_graph_success=True,
        max_attempts=4,
        timeout=10.0,
        enable_graph_attempt=2,
        partial_ik_opt=True,
        success_ratio=1.0,
        fail_on_invalid_query=True,
        enable_finetune_trajopt=True,
        parallel_finetune=True,
    ),
    ik_solver=IKSolverCfg(
        num_seeds=50,
        use_cuda_graph=True,
        self_collision_check=False,
        self_collision_opt=False,
        cuda_grasp_batch_size=1280,
    ),
)


DUALARM_PIPER_ROBOT_INFO_CFGS = {
    "left_arm": RobotInfoCfg(
        robot_name="robots/dualarm_piper",
        manipulator_name="left_arm",
        gripper_open_val=_PIPER_GRIPPER_OPEN_VAL,
        gripper_close_val=_PIPER_GRIPPER_CLOSE_VAL,
        t_standard_tcp_to_robot_ee=_PIPER_STANDARD_TCP_TO_ROBOT_EE,
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
        planner=_PIPER_PLANNER,
    ),
    "right_arm": RobotInfoCfg(
        robot_name="robots/dualarm_piper",
        manipulator_name="right_arm",
        gripper_open_val=_PIPER_GRIPPER_OPEN_VAL,
        gripper_close_val=_PIPER_GRIPPER_CLOSE_VAL,
        t_standard_tcp_to_robot_ee=_PIPER_STANDARD_TCP_TO_ROBOT_EE,
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
        # since left and right arms are symmetric, we can reuse the same planner config # noqa: E501
        planner=_PIPER_PLANNER,
    ),
}
