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

"""Shared planner defaults for embodiment profiles."""

from __future__ import annotations

from robo_orchard_sim.controllers.curobo_planner.curobo import (
    IKSolverCfg,
    MotionGenCfg,
    MotionGenPlanConfig,
)

__all__ = [
    "DEFAULT_IK_SOLVER_CFG",
    "DEFAULT_MOTION_GEN_CFG",
    "DEFAULT_MOTION_GEN_PLAN_CFG",
]


DEFAULT_MOTION_GEN_CFG = MotionGenCfg(
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
    collision_cache={
        "obb": 30,
        "mesh": 10,
    },
)


DEFAULT_MOTION_GEN_PLAN_CFG = MotionGenPlanConfig(
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
)


DEFAULT_IK_SOLVER_CFG = IKSolverCfg(
    num_seeds=50,
    use_cuda_graph=True,
    self_collision_check=False,
    self_collision_opt=False,
    cuda_grasp_batch_size=1280,
)
