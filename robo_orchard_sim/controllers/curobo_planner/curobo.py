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
import logging
import os
from typing import TYPE_CHECKING, Any, List, Literal, Mapping

import numpy as np
import torch
import yaml
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Mesh, Sphere, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState as CuroboJointState
from curobo.util_file import get_world_configs_path
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenResult,
)
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassType_co,
    Config,
    TorchTensor,
)

from robo_orchard_sim.controllers.curobo_planner.curobo_ext import (
    MotionGenEx as MotionGen,
)
from robo_orchard_sim.controllers.curobo_planner.mixin import (
    ArticulationJointTrajPlannerMixin,
    CannotFindTrajectoryError,
    IkResult,
    JointStateTrajetory,
)

if TYPE_CHECKING:
    from curobo.util.usd_helper import UsdHelper


logger = logging.getLogger(__file__)


class ArticulationJointCuroboTrajPlanner(ArticulationJointTrajPlannerMixin):
    def __init__(
        self, cfg: "ArticulationJointCuroboTrajPlannerCfg", env_nums: int = 1
    ) -> None:
        self.cfg = cfg

        self.tensor_args = TensorDeviceType(
            device=torch.device(self.cfg.device), dtype=torch.float32
        )

        self._last_world_config = self.get_world_config("collision_table.yml")
        self.world_cfg_list = []
        for _i in range(env_nums):
            self.world_cfg_list.append(self._last_world_config)

        self._usd_helper = None

        robot_config = RobotConfig.from_dict(
            self.cfg.robot.to_dict(), tensor_args=self.tensor_args
        )

        motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg=robot_config,
            world_model=self.world_cfg_list,
            tensor_args=self.tensor_args,
            interpolation_dt=self.cfg.motion_gen.interpolation_dt,
            collision_activation_distance=self.cfg.motion_gen.collision_activation_distance,
            interpolation_steps=self.cfg.motion_gen.interpolation_steps,
            num_ik_seeds=self.cfg.motion_gen.num_ik_seeds,
            num_trajopt_seeds=self.cfg.motion_gen.num_trajopt_seeds,
            grad_trajopt_iters=self.cfg.motion_gen.grad_trajopt_iters,
            evaluate_interpolated_trajectory=self.cfg.motion_gen.evaluate_interpolated_trajectory,
            trajopt_tsteps=self.cfg.motion_gen.trajopt_tsteps,
            use_cuda_graph=self.cfg.motion_gen.use_cuda_graph,
            num_graph_seeds=self.cfg.motion_gen.num_graph_seeds,
            collision_checker_type=CollisionCheckerType.MESH,
            self_collision_check=self.cfg.motion_gen.self_collision_check,
            maximum_trajectory_time=self.cfg.motion_gen.maximum_trajectory_time,
            jerk_scale=self.cfg.motion_gen.jerk_scale,
            finetune_dt_scale=self.cfg.motion_gen.finetune_dt_scale,
            collision_cache=self.cfg.motion_gen.collision_cache,
        )
        self.planner = MotionGen(motion_gen_cfg)

        # self.planner.world_coll_checker.world_model.save_world_as_mesh(
        #     "collision_base.obj"
        # )

        self.planner_cfg = MotionGenPlanConfig(
            enable_graph=self.cfg.motion_gen_plan.enable_graph,
            enable_opt=self.cfg.motion_gen_plan.enable_opt,
            use_nn_ik_seed=self.cfg.motion_gen_plan.use_nn_ik_seed,
            need_graph_success=self.cfg.motion_gen_plan.need_graph_success,
            max_attempts=self.cfg.motion_gen_plan.max_attempts,
            timeout=self.cfg.motion_gen_plan.timeout,
            enable_graph_attempt=self.cfg.motion_gen_plan.enable_graph_attempt,
            partial_ik_opt=self.cfg.motion_gen_plan.partial_ik_opt,
            success_ratio=self.cfg.motion_gen_plan.success_ratio,
            fail_on_invalid_query=self.cfg.motion_gen_plan.fail_on_invalid_query,
            enable_finetune_trajopt=self.cfg.motion_gen_plan.enable_finetune_trajopt,
            parallel_finetune=self.cfg.motion_gen_plan.parallel_finetune,
            num_ik_seeds=self.cfg.motion_gen_plan.num_ik_seeds,
            num_graph_seeds=self.cfg.motion_gen_plan.num_graph_seeds,
            num_trajopt_seeds=self.cfg.motion_gen_plan.num_trajopt_seeds,
        )
        self.kinematics_model = CudaRobotModel(
            motion_gen_cfg.robot_cfg.kinematics
        )

        # ik_solver
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg=robot_config,
            world_model=None,
            tensor_args=self.tensor_args,
            num_seeds=self.cfg.ik_solver.num_seeds,
            position_threshold=self.cfg.ik_solver.position_threshold,
            rotation_threshold=self.cfg.ik_solver.rotation_threshold,
            world_coll_checker=None,
            collision_checker_type=self.cfg.ik_solver.collision_checker_type,
            self_collision_check=self.cfg.ik_solver.self_collision_check,
            self_collision_opt=self.cfg.ik_solver.self_collision_opt,
            grad_iters=self.cfg.ik_solver.grad_iters,
            use_particle_opt=self.cfg.ik_solver.use_particle_opt,
            use_cuda_graph=self.cfg.ik_solver.use_cuda_graph,
            collision_cache=self.cfg.ik_solver.collision_cache,
            n_collision_envs=self.cfg.ik_solver.n_collision_envs,
            ee_link_name=self.cfg.ik_solver.ee_link_name,
            use_es=self.cfg.ik_solver.use_es,
            es_learning_rate=self.cfg.ik_solver.es_learning_rate,
            use_fixed_samples=self.cfg.ik_solver.use_fixed_samples,
            store_debug=self.cfg.ik_solver.store_debug,
            regularization=self.cfg.ik_solver.regularization,
            collision_activation_distance=self.cfg.ik_solver.collision_activation_distance,
            high_precision=self.cfg.ik_solver.high_precision,
            project_pose_to_goal_frame=self.cfg.ik_solver.project_pose_to_goal_frame,
            # seed = self.cfg.seed,
        )

        # create ik solver
        self.ik_solver = IKSolver(ik_config)

        self.reset()

    def close(self) -> None:
        """Release large planner objects before dropping the planner cache."""
        self.planner = None
        self.ik_solver = None
        self.kinematics_model = None
        self._usd_helper = None
        self.world_cfg_list.clear()

    def _get_default_world_config(self) -> WorldConfig:
        filename = os.path.join(get_world_configs_path(), "collision_base.yml")
        with open(filename, "r") as f:
            world_config = WorldConfig.from_dict(yaml.safe_load(f))
        return world_config

    def get_world_config(self, file_name) -> WorldConfig:
        """Get the current world configuration."""
        import robo_orchard_sim.controllers.curobo_planner as base_config_path

        package_path = os.path.dirname(base_config_path.__file__)
        filename = os.path.join(package_path, file_name)
        with open(filename, "r") as f:
            world_config = WorldConfig.from_dict(yaml.safe_load(f))
        return world_config

    @property
    def usd_helper(self) -> "UsdHelper":
        from curobo.util.usd_helper import UsdHelper

        if self._usd_helper is None:
            self._usd_helper = UsdHelper()
        return self._usd_helper

    @property
    def joint_names(self) -> List[str]:
        return self.cfg.robot.kinematics.cspace.joint_names

    @property
    def ee_link_name(self) -> str:
        return self.cfg.robot.kinematics.ee_link

    def plan_to_target_ee_pose(
        self,
        start_joint_positions: torch.Tensor,
        target_poses: torch.Tensor,
        mode: Literal["AvoidObs", "Simple"] = "Simple",
    ) -> JointStateTrajetory:
        """Plans a trajectory to move the end-effector to the target poses.

        Args:
            start_joint_positions (torch.Tensor): A tensor of shape (BATCH,
                NUM_JOINTS) representing the starting joint positions.
            target_poses (torch.Tensor): A tensor of shape (BATCH, 7)
                representing the target poses of the end-effector. Each pose
                is defined as [x, y, z, qx, qy, qz, qw].

        Returns:
            JointStateTrajetory: The planned trajectory containing joint
                positions, velocities, and success indices.
        """
        if start_joint_positions.ndim != 2 or target_poses.ndim != 2:
            raise ValueError(
                "start_joint_positions and target_poses must be 2-dimensional tensors"  # noqa: E501
            )

        if start_joint_positions.shape[0] != target_poses.shape[0]:
            raise ValueError(
                "start_joint_positions and target_poses must have the same batch size"  # noqa: E501
            )

        start_joint_positions = CuroboJointState.from_position(
            position=self.tensor_args.to_device(start_joint_positions),
            joint_names=self.joint_names[: start_joint_positions.shape[1]],
        )
        target_poses = Pose(
            position=self.tensor_args.to_device(target_poses[:, :3]),
            quaternion=self.tensor_args.to_device(
                target_poses[:, 3:][:, [3, 0, 1, 2]]
            ),
        )

        # IF batch is 1 use plan_single, else use plan_batch
        batch = start_joint_positions.position.shape[0]
        if batch < 2:
            # use plan_single
            _valid_flag = self._check_joint_validation(start_joint_positions)

            if not _valid_flag:
                raise CannotFindTrajectoryError("START JOINT OUT OF LIMITS!")

            raw_ret: MotionGenResult = self.planner.plan_single(
                start_joint_positions, target_poses, self.planner_cfg
            )

            # raw_ret.success is a boolean tensor, filter out all
            # elements == True
            if raw_ret.success[raw_ret.success].numel() == 0:
                raise CannotFindTrajectoryError(raw_ret.status)

            interpolated_raw_ret = raw_ret.get_interpolated_plan()

            ret = JointStateTrajetory(
                positions=interpolated_raw_ret.position,
                velocities=interpolated_raw_ret.velocity,
                indices=raw_ret.success,
            )

        else:
            # use plan_batch
            self._clamp_joint(start_joint_positions)
            if mode == "Simple":
                raw_ret: MotionGenResult = self.planner.plan_batch(
                    start_joint_positions, target_poses, self.planner_cfg
                )
            else:
                raw_ret: MotionGenResult = self.planner.plan_batch_env(
                    start_joint_positions, target_poses, self.planner_cfg
                )

            if raw_ret.success[raw_ret.success].numel() == 0:
                raise CannotFindTrajectoryError(raw_ret.status)

            # transfer result to JointStateTrajetory
            ret = self._transfer_batch_trajs(raw_ret)

        return ret

    def plan_to_target_joint_positions(
        self,
        start_joint_positions: torch.Tensor,
        target_joint_positions: torch.Tensor,
    ) -> JointStateTrajetory:
        """Plan to target joint positions.

        Plans a trajectory from the start joint positions to the target joint
        positions.

        Args:
            start_joint_positions (torch.Tensor): A tensor of shape (BATCH,
                NUM_JOINTS) representing the starting joint positions.
            target_joint_positions (torch.Tensor): A tensor of shape (BATCH,
                NUM_JOINTS) representing the target joint positions.

        Returns:
            JointStateTrajetory: The planned trajectory containing joint
                positions, velocities, and success indices.

        """
        if start_joint_positions.ndim != 2 or target_joint_positions.ndim != 2:
            raise ValueError(
                "start_joint_positions and target_poses must be "
                "2-dimensional tensors"
            )

        if start_joint_positions.shape[0] != target_joint_positions.shape[0]:
            raise ValueError(
                "start_joint_positions and target_poses must have the same "
                "batch size"
            )

        start_joint_positions = CuroboJointState.from_position(
            position=self.tensor_args.to_device(start_joint_positions),
            joint_names=self.joint_names[: start_joint_positions.shape[1]],
        )
        target_joint_positions = CuroboJointState.from_position(
            position=self.tensor_args.to_device(target_joint_positions),
            joint_names=self.joint_names[: target_joint_positions.shape[1]],
        )

        _valid_flag = self._check_joint_validation(start_joint_positions)

        if not _valid_flag:
            raise CannotFindTrajectoryError("START JOINT OUT OF LIMITS!")

        raw_ret: MotionGenResult = self.planner.plan_single_js(
            start_joint_positions, target_joint_positions, self.planner_cfg
        )

        # raw_ret.success is a boolean tensor, filter out all elements == True
        if raw_ret.success[raw_ret.success].numel() == 0:
            raise CannotFindTrajectoryError(raw_ret.status)

        interpolated_raw_ret = raw_ret.get_interpolated_plan()

        ret = JointStateTrajetory(
            positions=interpolated_raw_ret.position,
            velocities=interpolated_raw_ret.velocity,
            indices=raw_ret.success,
        )

        return ret

    def _transfer_batch_trajs(
        self, raw_ret: MotionGenResult
    ) -> JointStateTrajetory:
        trajs_positons: List[TorchTensor] = []
        trajs_velocities: List[TorchTensor] = []

        interpolated_raw_ret = raw_ret.get_paths()

        for it in interpolated_raw_ret:
            trajs_positons.append(it.position)
            trajs_velocities.append(it.velocity)

        ret = JointStateTrajetory(
            positions=trajs_positons,
            velocities=trajs_velocities,
            indices=raw_ret.success,
        )

        return ret

    def fk(
        self,
        joint_angles: torch.Tensor,
        w_first: bool = False,
    ) -> torch.Tensor:
        """Calculate forward kinematics for the robot end-effector.

        Computes the pose of the robot's end-effector.
        given the joint angles using the robot's kinematic model.

        Args:
            joint_angles (torch.Tensor): Joint angles tensor with shape
                [BATCH_SIZE, NUM_JOINTS]. Each row represents a different
                robot configuration, and each column represents a joint angle
                in radians.
            w_first (bool, optional): Quaternion format flag.

        Returns:
            torch.Tensor: End-effector pose tensor with shape [BATCH_SIZE, 7].
                Each row contains [x, y, z, qx, qy, qz, qw] where:
                - [x, y, z]: Position of the end-effector in meters
                - [qx, qy, qz, qw]: Quaternion orientation of the end-effector
                The quaternion format depends on the w_first parameter:
                - If w_first=False: [x, y, z, w] (default)
                - If w_first=True: [w, x, y, z]
        """

        # Compute Forward Kinematics
        if not joint_angles.is_contiguous():
            joint_angles = joint_angles.contiguous()
        fk_result = self.kinematics_model.get_state(joint_angles)
        # print(f"Forward Kinematics Result:\n{fk_result}\n")

        end_effector_position = fk_result.ee_position

        if w_first:
            end_effector_orientation = fk_result.ee_quaternion
        else:
            end_effector_orientation = fk_result.ee_quaternion[:, [1, 2, 3, 0]]

        # Concatenate end_effector_position and end_effector_orientation_wxyz
        end_effector_pose = torch.cat(
            (end_effector_position, end_effector_orientation), dim=1
        )

        return end_effector_pose

    def update_world_from_isaacsim(
        self,
        world,
        reference_prim_paths: List[str],
        only_paths: List[List[str]] | None = None,
        ignore_paths: List[List[str]] | None = None,
        only_substrings: List[List[str]] | None = None,
        ignore_substrings: List[List[str]] | None = None,
        reset: bool = True,
    ) -> None:
        """Update the internal world configuration from an IsaacSim USD stage.

            Retrieves obstacles from the given USD stage and updates
            the planner's world configuration for each environment,
            using the specified filtering criteria and reference prims.

        Args:
            world: The IsaacSim world object containing the USD stage.
            reference_prim_paths (List[str]): List of USD prim paths,
                one per environment, whose transforms are used as the
                coordinate frame for subsequent operations.
            only_paths (List[List[str]], optional): For each environment,
                a list of path prefixes. If provided, only prims whose
                paths start with one of these prefixes will be processed.
            ignore_paths (List[List[str]], optional): For each environment,
                a list of path prefixes. If provided, prims whose paths
                start with one of these prefixes will be skipped.
            only_substrings (List[List[str]], optional):
                For each environment, a list of substrings. If provided,
                only prims whose paths contain one of these substrings will
                be processed.
            ignore_substrings (List[List[str]], optional):
                For each environment, a list of substrings. If provided,
                prims whose paths contain one of these substrings will be
                skipped.
            reset (bool, optional): If True (default), resets the planner
                after updating the world configuration.If False,
                updates the planner's world in-place.
        """
        self.usd_helper.load_stage(world.stage)

        env_nums = len(reference_prim_paths)
        self.world_cfg_list = []

        for env_idx in range(env_nums):
            only_path = only_paths[env_idx] if only_paths is not None else None
            ignore_path = (
                ignore_paths[env_idx] if ignore_paths is not None else None
            )
            only_substring = (
                only_substrings[env_idx]
                if only_substrings is not None
                else None
            )
            ignore_substring = (
                ignore_substrings[env_idx]
                if ignore_substrings is not None
                else None
            )
            reference_prim_path = (
                reference_prim_paths[env_idx]
                if reference_prim_paths is not None
                else None
            )

            world_cfg = self.usd_helper.get_obstacles_from_stage(
                only_paths=only_path,
                ignore_paths=ignore_path,
                only_substring=only_substring,
                ignore_substring=ignore_substring,
                reference_prim_path=reference_prim_path,
            ).get_collision_check_world()
            self.world_cfg_list.append(world_cfg)
        if reset:
            self.reset()
        else:
            for i, world_config in enumerate(self.world_cfg_list):
                print(
                    f"Updated env{i} with {len(world_config.objects)} ",
                    "obstacles.",
                )
            self.planner.update_world(self.world_cfg_list)

    def update_world_from_pointcloud(
        self, point_cloud: np.ndarray, reset: bool = True
    ) -> None:
        if len(point_cloud.shape) != 2 or point_cloud.shape[1] != 3:
            raise ValueError(
                "point_cloud shape should be [N, 3], where N is the number of points"  # noqa: E501
            )
        self._last_world_config = WorldConfig(
            mesh=[Mesh.from_pointcloud(point_cloud)]
        )
        if reset:
            self.reset()
        else:
            self.planner.update_world(self._last_world_config)

    def attach_obj(
        self,
        joint_positions: torch.Tensor,
        obj_names: List[str],
        surface_sphere_radius=0.05,
    ) -> None:
        """Attach the object to the robot.

        Only support single env for now
        """

        joint_state = CuroboJointState.from_position(
            position=self.tensor_args.to_device(joint_positions),
            joint_names=self.joint_names[: joint_positions.shape[1]],
        )

        # kinematics_cfg = self.planner.robot_cfg.kinematics.kinematics_config
        # spheres_before = kinematics_cfg.link_spheres.cpu().numpy().copy()
        # print("spheres before attaching:", spheres_before)

        self.planner.attach_objects_to_robot(
            joint_state=joint_state,
            object_names=obj_names,
            surface_sphere_radius=surface_sphere_radius,
            world_objects_pose_offset=Pose.from_list(
                [0, 0, 0.01, 1, 0, 0, 0], self.tensor_args
            ),
        )

        # spheres_after = kinematics_cfg.link_spheres.cpu().numpy().copy()
        # print("spheres after attaching:", spheres_after)

    def get_robot_sphers(self, joint_positions: torch.Tensor) -> List[Sphere]:
        """Get the robot as spheres."""

        sph_list = self.planner.kinematics.get_robot_as_spheres(
            joint_positions
        )
        return sph_list

    def detach_obj(self):
        self.planner.detach_object_from_robot()

    def reset(self) -> None:
        self.planner.clear_world_cache()
        self.planner.reset(reset_seed=False)
        self.planner.update_world(self.world_cfg_list)

    def _clamp_joint(self, start_joint_angles):
        jl = self.kinematics_model.get_joint_limits()
        lower_limit = jl.position[0, :]
        upper_limit = jl.position[1, :]

        cur_joint = start_joint_angles.position

        is_below_lower_soft = torch.any(cur_joint < lower_limit)
        is_above_upper_soft = torch.any(cur_joint > upper_limit)

        if is_below_lower_soft or is_above_upper_soft:
            np.set_printoptions(precision=15, suppress=True)
            clamped_joint = torch.clamp(
                cur_joint, min=lower_limit, max=upper_limit
            )
            # print(f"  Clamped Joints:  {clamped_joint.cpu().numpy()}")
            print("[WARNNING] Clamped Joints")
            start_joint_angles.position = clamped_joint
            np.set_printoptions(precision=8, suppress=False)

            return True
        else:
            return False

    def _check_joint_validation(
        self, start_joint_angles, tolerance=1e-3
    ) -> bool:
        jl = self.kinematics_model.get_joint_limits()
        lower_limit = jl.position[0, :]
        upper_limit = jl.position[1, :]

        cur_joint = start_joint_angles[0].position

        tolerance = 1e-3
        is_below_lower = torch.any(cur_joint < lower_limit - tolerance)
        is_above_upper = torch.any(cur_joint > upper_limit + tolerance)

        if is_below_lower or is_above_upper:
            np.set_printoptions(precision=15, suppress=True)
            print(
                f"Current joint angles {cur_joint.cpu().numpy()} "
                f"out of limits!\n"
                f"Joint lower limits: {lower_limit.cpu().numpy()}\n"
                f"Joint upper limits: {upper_limit.cpu().numpy()}\n"
            )

            np.set_printoptions(precision=8, suppress=False)
            return False
        else:
            is_below_lower_soft = torch.any(cur_joint < lower_limit)
            is_above_upper_soft = torch.any(cur_joint > upper_limit)

            if is_below_lower_soft or is_above_upper_soft:
                np.set_printoptions(precision=15, suppress=True)
                clamped_joint = torch.max(
                    torch.min(cur_joint, upper_limit), lower_limit
                ).unsqueeze(0)
                # print(f"  Clamped Joints:  {clamped_joint[0].cpu().numpy()}")
                start_joint_angles.position = clamped_joint
                np.set_printoptions(precision=8, suppress=False)
            return True

    def filter_poses_with_IK(
        self, candidate_poses: torch.Tensor, current_joint_angles: torch.Tensor
    ) -> torch.Tensor:
        """Filter candidate poses using Curobo's IKSolver.

        Args:
            candidate_poses (torch.Tensor): Candidate poses with
                shape [B, N, 7].
            current_joint_angles (torch.Tensor): Current robot joint angles
                with shape [B, dof].

        Returns:
            best_pose(torch.Tensor): Filtered best poses with shape [B, 7].
                If no reachable solution exists in a batch, the corresponding
                row will contain NaN values.
            best_indices(torch.Tensor): Best pose indices with shape[B]
        """
        if (
            candidate_poses.dim() != 3
            or candidate_poses.shape[0] != current_joint_angles.shape[0]
        ):
            raise ValueError(
                "Input dimension mismatch! candidate_poses should "
                "be [B, N, 7], current_joint_angles should be [B, dof]"
            )

        batch_size, num_poses, _ = candidate_poses.shape

        # Call IKSolver's solve_batch method for batch solving
        ik_result = self.graph_ik(candidate_poses)
        success_mask = ik_result.success
        solutions = ik_result.solution

        # Calculate L2 squared distance in joint space (motion cost)
        # current_joint_angles: [B,dof] -> [B,1,dof]
        joint_diff = solutions - current_joint_angles.unsqueeze(1)
        costs = torch.sum(joint_diff**2, dim=-1)  # Shape: [B, N]

        # Set cost of unreachable solutions to infinity
        costs[~success_mask] = float("inf")

        # Find index of minimum cost solution for each batch
        min_cost_indices = torch.argmin(costs, dim=1)  # Shape: [B]

        # 4. Select best poses
        # Use torch.gather to select best poses from candidate_poses
        best_pose_indices = min_cost_indices.view(batch_size, 1, 1).expand(
            -1, 1, 7
        )
        best_poses = torch.gather(
            candidate_poses, 1, best_pose_indices
        ).squeeze(1)

        # Mark batches where no solution is found among all N candidates
        any_success = torch.any(success_mask, dim=1)
        best_poses[~any_success] = float("nan")

        # For batches with no solution, set index to -1
        best_indices = min_cost_indices.clone()
        best_indices[~any_success] = -1

        return best_poses, best_indices

    def ik(self, candidate_poses: torch.Tensor, use_batch: bool = True):
        """Filter candidate poses using Curobo's IKSolver.

        Args:
            candidate_poses (torch.Tensor): Candidate poses with
                shape [B, N, 7]. or [B, 7]

        Returns:
            ik_result (torch.Tensor): Filtered best poses with shape [B, 7].
                If no reachable solution exists in a batch, the corresponding
                row will contain NaN values.
            indices (torch.Tensor): success indices
        """

        if candidate_poses.dim() == 2 and not use_batch:
            mode = "use_goalset"
        elif candidate_poses.dim() == 2 and use_batch:
            mode = "use_batch"
        elif candidate_poses.dim() == 3:
            mode = "use_batch_goalset"
        else:
            raise ValueError(
                "Input dimension mismatch! candidate_poses should be"
                " [B, N, 7] or [B, 7]",
            )

        goal_poses = Pose(
            position=candidate_poses[..., :3],
            quaternion=candidate_poses[..., 3:],
        )

        if mode == "use_goalset":
            ik_result = self.ik_solver.solve_goalset(goal_poses)
        elif mode == "use_batch":
            ik_result = self.ik_solver.solve_batch(goal_poses)
        else:
            ik_result = self.ik_solver.solve_batch_goalset(goal_poses)

        return ik_result

    def graph_ik(self, candidate_poses: torch.Tensor):
        if candidate_poses.dim() == 2:
            B, _ = candidate_poses.shape
            N = 1
            flat_poses = candidate_poses
        elif candidate_poses.dim() == 3:
            B, N, _ = candidate_poses.shape
            flat_poses = candidate_poses.view(B * N, -1)
            # flat_poses = candidate_poses
        else:
            raise ValueError(
                "Input dimension mismatch! candidate_poses should be"
                " [B, N, 7] or [B, 7]",
            )

        if self.cfg.ik_solver.use_cuda_graph:
            max_batch = self.cfg.ik_solver.cuda_grasp_batch_size
            original_total = B * N

            # Determine effective N (candidates per batch) to fit in max_batch
            # We prioritize keeping B (batch size) over N (candidates)
            if original_total > max_batch:
                print(
                    f"[WARNING]: Use CUDA graph IK with max batching "
                    f"{max_batch}.",
                    f"Original total poses {original_total} may be truncated.",
                    "You can increase cuda_grasp_batch_size param",
                )
                if B > max_batch:
                    # process as many batches as possible (1 candidate each).
                    effective_N = 1
                    processed_B = max_batch
                    raise ValueError(
                        f"Batch size B={B} exceeds max_batch={max_batch}. "
                        "Please reduce batch size, "
                        "or you can increase cuda_grasp_batch_size "
                        "in ik_solver config.",
                    )
                else:
                    effective_N = max_batch // B
                    processed_B = B

                # Prepare input for solver by slicing candidates
                if candidate_poses.dim() == 3:
                    # Slice (B, N, 7) -> (processed_B, effective_N, 7)
                    poses_to_solve = candidate_poses[
                        :processed_B, :effective_N, :
                    ].reshape(-1, 7)
                else:
                    # Slice (B, 7) -> (processed_B, 7)
                    poses_to_solve = candidate_poses[:processed_B]
            else:
                effective_N = N
                processed_B = B
                poses_to_solve = flat_poses

            # Pad to max_batch if needed
            current_size = poses_to_solve.shape[0]
            if current_size < max_batch:
                padding_size = max_batch - current_size
                padding = poses_to_solve[-1:].repeat(padding_size, 1)
                poses_to_solve = torch.cat([poses_to_solve, padding], dim=0)

            # Solve
            goal_poses = Pose(
                position=poses_to_solve[..., :3],
                quaternion=poses_to_solve[..., 3:],
            )
            ik_result = self.ik_solver.solve_batch(goal_poses)

            # If truncated or padded, need to reconstruct the results
            # to match the caller's expectation of (B * N) results
            if (
                original_total != max_batch
                or processed_B != B
                or effective_N != N
            ):
                valid_computed_size = processed_B * effective_N

                # Extract valid results from the solver output
                valid_success = ik_result.success[:valid_computed_size]
                valid_solution = ik_result.solution[:valid_computed_size]

                # Create full buffers (default to failure/zeros)
                full_success = torch.zeros(
                    (B, N),
                    dtype=valid_success.dtype,
                    device=valid_success.device,
                )
                full_solution = torch.zeros(
                    (B, N, valid_solution.shape[-1]),
                    dtype=valid_solution.dtype,
                    device=valid_solution.device,
                )

                valid_success_view = valid_success.view(
                    processed_B, effective_N
                )
                valid_solution_view = valid_solution.view(
                    processed_B, effective_N, -1
                )

                # Assign computed results back to their original positions.
                # This works for both [B, 7] (N == 1) and [B, N, 7] inputs.
                full_success[:processed_B, :effective_N] = valid_success_view
                full_solution[:processed_B, :effective_N, :] = (
                    valid_solution_view
                )

                # Update ik_result object
                ik_result.success = full_success
                ik_result.solution = full_solution

            else:
                ik_result.success = ik_result.success.view(B, N)
                ik_result.solution = ik_result.solution.view(B, N, -1)
        else:
            goal_poses = Pose(
                position=flat_poses[..., :3], quaternion=flat_poses[..., 3:]
            )
            ik_result = self.ik_solver.solve_batch(goal_poses)
            ik_result.success = ik_result.success.view(B, N)
            ik_result.solution = ik_result.solution.view(B, N, -1)

        result = IkResult(
            goal_poses=candidate_poses,
            solution=ik_result.solution,
            success=ik_result.success,
            # goalset_index=ik_result.goalset_index,
        )

        return result


class CSpaceCfg(Config):
    joint_names: List[str]

    retract_config: TorchTensor | None = None

    cspace_distance_weight: TorchTensor | None = None

    null_space_weight: TorchTensor | None = None

    max_acceleration: float | List[float] = 10.0

    max_jerk: float | List[float] = 500.0

    velocity_scale: float | List[float] = 1.0

    acceleration_scale: float | List[float] = 1.0

    jerk_scale: float | List[float] = 1.0

    position_limit_clip: float | List[float] = 0.0


class RobotKinematicsCfg(Config):
    base_link: str

    ee_link: str

    link_names: List[str] | None = None

    collision_link_names: List[str] | None = None

    collision_sphere_buffer: float | Mapping[str, float] = 0.0

    compute_jacobian: bool = False

    self_collision_buffer: Mapping[str, float] | None = None

    self_collision_ignore: Mapping[str, List[str]] | None = None

    use_global_cumul: bool = True

    asset_root_path: str = ""

    mesh_link_names: List[str] | None = None

    load_link_names_with_mesh: bool = False

    urdf_path: str | None = None

    usd_path: str | None = None

    usd_robot_root: str | None = None

    isaac_usd_path: str | None = None

    use_usd_kinematics: bool = False

    usd_flip_joints: List[str] | None = None

    usd_flip_joint_limits: List[str] | None = None

    lock_joints: Mapping[str, float] | None = None

    add_object_link: bool = False

    use_external_assets: bool = False

    external_asset_path: str | None = None

    external_robot_configs_path: str | None = None

    extra_collision_spheres: Mapping[str, int] | None = None

    cspace: CSpaceCfg | None = None

    load_meshes: bool = False

    collision_spheres: str | None = None

    extra_links: dict[str, Any | dict[str, Any]] | None = None


class RobotCfg(Config):
    kinematics: RobotKinematicsCfg


class MotionGenCfg(Config):
    interpolation_dt: float = 0.02

    interpolation_steps: int = 5000

    collision_activation_distance: float | None = None

    num_ik_seeds: int = 32

    num_graph_seeds: int = 4

    num_trajopt_seeds: int = 4

    num_batch_ik_seeds: int = 32

    num_batch_trajopt_seeds: int = 1

    num_trajopt_noisy_seeds: int = 1

    position_threshold: float = 0.005

    rotation_threshold: float = 0.05

    cspace_threshold: float = 0.05

    grad_trajopt_iters: int | None = None

    evaluate_interpolated_trajectory: bool = True

    trajopt_tsteps: int = 32

    use_cuda_graph: bool = True

    self_collision_check: bool = True

    maximum_trajectory_time: float | None = None

    jerk_scale: float | List[float] | None = None

    finetune_dt_scale: float = 0.9

    collision_cache: Mapping[str, int] | None = None


class IKSolverCfg(Config):
    num_seeds: int = 100

    position_threshold: float = 0.005

    rotation_threshold: float = 0.05

    # base_cfg_file: str = 'base_cfg.yml',

    # particle_file: str = 'particle_ik.yml',

    # gradient_file: str = 'gradient_ik_autotune.yml',

    use_cuda_graph: bool = True

    cuda_grasp_batch_size: int = 144

    self_collision_check: bool = True

    self_collision_opt: bool = True

    grad_iters: int | None = None

    use_particle_opt: bool = True

    collision_checker_type: CollisionCheckerType | None = (
        CollisionCheckerType.MESH
    )

    sync_cuda_time: bool | None = None

    use_gradient_descent: bool = False

    collision_cache: dict[str, int] | None = None

    n_collision_envs: int | None = None

    ee_link_name: str | None = None

    use_es: bool | None = None

    es_learning_rate: float | None = 0.1

    use_fixed_samples: bool | None = None

    store_debug: bool = False

    regularization: bool = True

    collision_activation_distance: float | None = None

    high_precision: bool = False

    project_pose_to_goal_frame: bool = True


class ArticulationJointCuroboTrajPlannerCfg(
    ClassConfig[ArticulationJointCuroboTrajPlanner]
):
    motion_gen: MotionGenCfg

    motion_gen_plan: MotionGenPlanConfig

    ik_solver: IKSolverCfg

    robot: RobotCfg

    class_type: ClassType_co[ArticulationJointCuroboTrajPlanner] = (
        ArticulationJointCuroboTrajPlanner
    )

    device: str = "cuda:0"
