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

"""Pick executor for object and direct target grasp planning."""

from __future__ import annotations
from typing import Any, Literal

import robo_orchard_core.utils.math as math_utils
import torch

from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ResolvedManipulatorProfile,
)
from robo_orchard_sim.tasks.trajs_gen.base_executor import (
    BaseExecutor,
    BaseExecutorCfg,
    ObjectInfo,
    Trajectories,
    _ManipulatorRuntimeState,
)
from robo_orchard_sim.tasks.trajs_gen.manipulator_resolver import (
    ManipulatorBindingContext,
)
from robo_orchard_sim.tasks.trajs_gen.pose_generator import (
    MotionPose,
    MoveByDisplacementCfg,
    PoseGenerationContext,
    PoseGeneratorCfg,
)
from robo_orchard_sim.utils.config import ClassType_co
from robo_orchard_sim.utils.env_utils import PoseAugmentor

GraspMode = Literal["Top-down", "Horizontal", "Default"]


class PickExecutor(BaseExecutor):
    """Plan pre-grasp, grasp, and gripper-close trajectories."""

    cfg: "PickExecutorCfg"

    ROTATION_MATRIX = (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )

    def __init__(self, cfg: "PickExecutorCfg") -> None:
        super().__init__(cfg)
        self._target_pose_gen = (
            cfg.target() if cfg.target is not None else None
        )
        pre_grasp_cfg = cfg.pre_grasp or MoveByDisplacementCfg(
            distance=-0.02,
            direction="z",
            frame="gripper",
        )
        self._pre_grasp_pose_gen = pre_grasp_cfg()

    def plan(
        self,
        env: Any,
        context: ManipulatorBindingContext,
    ) -> Trajectories:
        """Resolve robot info and plan a full pick action."""
        # self._validate_target_source()
        resolved = self.cfg.resolve_manipulator_info(env, context=context)
        # self._validate_resolved_manipulator(resolved)
        self.last_resolved_manipulator = resolved

        runtime_state = self.build_runtime_state(
            env=env,
            resolved=resolved,
            frame="env",
        )
        grasp_pose = self._resolve_grasp_pose(
            env=env,
            resolved=resolved,
            runtime_state=runtime_state,
        )
        debug_target_pose = self.build_debug_target_pose(
            env=env,
            runtime_state=runtime_state,
            pose_robot_base=grasp_pose,
            name="pick_grasp",
        )
        pre_grasp_pose = self._generate_pre_grasp_pose(
            resolved=resolved,
            runtime_state=runtime_state,
            grasp_pose=grasp_pose,
        )
        trajs_mode = self._resolve_trajs_mode()

        open_val = self._resolve_gripper_value(resolved, state="OPEN")
        close_val = self._resolve_gripper_value(resolved, state="CLOSED")
        traj1, success_flag1 = self.gen_to_target_pose_trajs(
            planner=resolved.planner,
            start_joint_positions=runtime_state.current_joint_positions,
            target_pose=self._convert_standard_ee_to_robot_ee(
                resolved,
                pre_grasp_pose,
            ),
            gripper_val=open_val,
            mode=trajs_mode,
        )
        traj2, success_flag2 = self.gen_to_target_pose_trajs(
            planner=resolved.planner,
            start_joint_positions=self.get_last_trajs(traj1)[
                :, : len(resolved.joint_ids)
            ],
            target_pose=self._convert_standard_ee_to_robot_ee(
                resolved,
                grasp_pose,
            ),
            gripper_val=open_val,
            mode=trajs_mode,
        )
        close_trajs = self.gen_gripper_trajs(
            current_joint_positions=self.get_last_trajs(traj2)[
                :, : len(resolved.joint_ids)
            ],
            start_gripper_val=open_val,
            end_gripper_val=close_val,
            length=self.cfg.close_gripper_steps,
        )
        trajectories = self._merge_trajs(traj1, traj2, close_trajs)
        success = bool(torch.all(success_flag1 & success_flag2).item())
        return Trajectories(
            trajectories=trajectories,
            success=success,
            resolved_manipulator=resolved,
            debug_target_poses=(debug_target_pose,),
        )

    # get grasp pose
    def _resolve_grasp_pose(
        self,
        env: Any,
        resolved: ResolvedManipulatorProfile,
        runtime_state: _ManipulatorRuntimeState,
    ) -> torch.Tensor:
        # Get target standard pose in robot base frame.
        if (
            self._target_pose_gen is not None
            and self.cfg.pick_object_info is None
        ):
            context = PoseGenerationContext(
                robot_base_pose_w=runtime_state.robot_base_pose_w,
                current_joint_pos=runtime_state.current_joint_positions,
                ee_pose_w=runtime_state.ee_pose_w,
                executor=self,
            )
            target = self._target_pose_gen.generate(context)
            if target.type != "pose":
                raise ValueError(
                    "PickExecutorCfg.target must generate a pose target."
                )
            target_pose = target.data.to(
                runtime_state.current_joint_positions.device
            )
        elif (
            self._target_pose_gen is None
            and self.cfg.pick_object_info is not None
        ):
            # pose in standard ee frame
            object_pose = self._plan_object(
                env=env,
                resolved=resolved,
                runtime_state=runtime_state,
            )
            target_pose = self._convert_pose_to_robot_base(
                runtime_state,
                object_pose,
            )
        elif (
            self._target_pose_gen is None and self.cfg.pick_object_info is None
        ):
            raise ValueError(
                "Either target_pose or object_info must be provided for "
                "the pick action."
            )
        else:
            raise ValueError(
                "Only one of target_pose or object_info should be provided "
                "for the pick action."
            )
        return target_pose

    def _plan_object(
        self,
        env: Any,
        resolved: ResolvedManipulatorProfile,
        runtime_state: _ManipulatorRuntimeState,
    ) -> torch.Tensor:
        """Plan the object to be picked.

        return standard_ee_pose in evn frame.
        """
        name = self.cfg.pick_object_info.name
        mode = self.cfg.pick_object_info.mode
        action = self.cfg.pick_object_info.action
        part = self.cfg.pick_object_info.part

        # Only support default grasp mode for now
        grasp_mode = self.cfg.grasp_mode

        grasp_part = env.scene[name].get_element_pose(  # noqa: E501
            mode=mode, action=action, part=part
        )  # (nums_env, num_parts, ...)

        grasp_pos = grasp_part.pos
        ENV_NUMS, PART, _ = grasp_pos.shape
        # trans to env frame
        grasp_pos -= (
            env.scene.env_origins[:].unsqueeze(1).expand(ENV_NUMS, PART, -1)
        )
        grasp_quat = grasp_part.quat

        # trans to ee standard frame
        rotation_transform = torch.tensor(self.ROTATION_MATRIX).to(
            grasp_pos.device
        )

        rot_quat = math_utils.matrix_to_quaternion(rotation_transform)
        grasp_quat = math_utils.quaternion_multiply(grasp_quat, rot_quat)

        grasp_pose_origin = torch.cat((grasp_pos, grasp_quat), dim=-1).to(
            grasp_quat.device
        )

        if grasp_mode == "Default":
            result = self._pose_filter(
                resolved,
                runtime_state,
                grasp_pose_origin,
            )

        elif grasp_mode == "Top-down":
            result = self._multi_pose_filter(
                resolved,
                runtime_state,
                grasp_pose_origin,
                reference_axis_world=torch.tensor([0.0, 0.0, -1.0]),
                angle_range_deg=(0, 45),
                pose_augument_config={
                    "x": (0, 180, 2),
                    "y": (-45, 45, 5),
                    "z": (0, 180, 2),
                },
            )

        elif grasp_mode == "Horizontal":
            result = self._multi_pose_filter(
                resolved,
                runtime_state,
                grasp_pose_origin,
                reference_axis_world=torch.tensor([0.0, 0.0, -1.0]),
                angle_range_deg=(60, 90),
                pose_augument_config={
                    "x": (0, 180, 2),
                    "y": (-45, 45, 5),
                },
            )
        else:
            raise ValueError(f"Unsupported grasp mode: {grasp_mode}")

        return result

    # gen pre greps pose

    def _generate_pre_grasp_pose(
        self,
        resolved: ResolvedManipulatorProfile,
        runtime_state: _ManipulatorRuntimeState,
        grasp_pose: torch.Tensor,
    ) -> torch.Tensor:
        robot_2_ee = self._convert_standard_ee_to_robot_ee(
            resolved, grasp_pose
        )

        ik_result = resolved.planner.graph_ik(robot_2_ee)
        joint_positions = ik_result.solution
        if isinstance(joint_positions, torch.Tensor):
            joint_positions = joint_positions.to(grasp_pose.device)
        else:
            joint_positions = torch.tensor(joint_positions).to(
                grasp_pose.device
            )
        joint_positions = joint_positions.view(
            runtime_state.current_joint_positions.shape
        )

        ee_pos_w, ee_quat_w = math_utils.frame_transform_combine(
            runtime_state.robot_base_pose_w[..., :3],
            runtime_state.robot_base_pose_w[..., 3:],
            robot_2_ee[..., :3],
            robot_2_ee[..., 3:],
        )
        ee_pose_w = torch.cat((ee_pos_w, ee_quat_w), dim=-1)

        context = PoseGenerationContext(
            robot_base_pose_w=runtime_state.robot_base_pose_w,
            ee_pose_w=ee_pose_w,
            current_joint_pos=joint_positions,
            executor=self,
        )
        pre_grasp_pose: MotionPose = self._pre_grasp_pose_gen.generate(context)

        if pre_grasp_pose.type == "joint":
            fk_pose = resolved.planner.fk(pre_grasp_pose.data, w_first=True)
            return self._convert_robot_ee_to_standard_ee(resolved, fk_pose)
        if pre_grasp_pose.type == "pose":
            return pre_grasp_pose.data.to(grasp_pose.device)
        raise ValueError(
            "PickExecutorCfg.pre_grasp must generate pose or joint targets."
        )

    def _convert_standard_ee_to_robot_ee(
        self,
        resolved: ResolvedManipulatorProfile,
        standard_ee_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Convert the target pose from standard EE to robot EE."""
        t_standard_ee_to_robot_ee = self._standard_to_robot_ee_transform(
            resolved,
            device=standard_ee_pose.device,
            dtype=standard_ee_pose.dtype,
        )

        std_pos = standard_ee_pose[..., :3]
        std_quat = standard_ee_pose[..., 3:]

        T_pos = t_standard_ee_to_robot_ee[:3, 3]
        T_quat = math_utils.matrix_to_quaternion(
            t_standard_ee_to_robot_ee[:3, :3]
        )

        pos, quat = math_utils.frame_transform_combine(
            std_pos, std_quat, T_pos, T_quat
        )

        robot_ee_pose = torch.cat((pos, quat), dim=-1).to(
            standard_ee_pose.device
        )

        return robot_ee_pose

    def _convert_robot_ee_to_standard_ee(
        self,
        resolved: ResolvedManipulatorProfile,
        robot_ee_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Convert the target pose from robot EE to standard EE."""
        # Get the transformation from standard EE to robot EE
        t_std_to_robot_ee_homo = self._standard_to_robot_ee_transform(
            resolved,
            device=robot_ee_pose.device,
            dtype=robot_ee_pose.dtype,
        )

        # Calculate the inverse transformation (from robot EE to standard EE)
        # The inverse of a homogeneous matrix [R | p] is [R.T | -R.T @ p]
        t_robot_to_std_ee_rot = t_std_to_robot_ee_homo[:3, :3].T
        t_robot_to_std_ee_pos = (
            -t_robot_to_std_ee_rot @ t_std_to_robot_ee_homo[:3, 3]
        )

        # Decompose the inverse transformation into position and quaternion
        T_inv_pos = t_robot_to_std_ee_pos
        T_inv_quat = math_utils.matrix_to_quaternion(t_robot_to_std_ee_rot)

        # Apply the inverse transformation
        # standard_ee_pose = robot_ee_pose * T_robot_to_standard
        robot_pos = robot_ee_pose[..., :3]
        robot_quat = robot_ee_pose[..., 3:]

        pos, quat = math_utils.frame_transform_combine(
            robot_pos, robot_quat, T_inv_pos, T_inv_quat
        )

        standard_ee_pose = torch.cat((pos, quat), dim=-1).to(
            robot_ee_pose.device
        )

        return standard_ee_pose

    def _standard_to_robot_ee_transform(
        self,
        resolved: ResolvedManipulatorProfile,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        transform_obj: Any = resolved.t_standard_tcp_to_robot_ee
        if transform_obj is None:
            return torch.eye(4, dtype=dtype, device=device)
        if hasattr(transform_obj, "to_homogeneous_matrix"):
            transform_obj = transform_obj.to_homogeneous_matrix()
        return torch.as_tensor(transform_obj, dtype=dtype, device=device)

    def _convert_pose_to_robot_base(
        self,
        runtime_state: _ManipulatorRuntimeState,
        target_pose: torch.Tensor,
    ) -> torch.Tensor:
        w_robot_base_pos = runtime_state.robot_base_pose_w[:, :3]
        w_robot_base_quat = runtime_state.robot_base_pose_w[:, 3:]

        if target_pose.dim() == 3:  # (B, N, 7)
            B, N = target_pose.shape[0], target_pose.shape[1]
            w_robot_base_pos = w_robot_base_pos.unsqueeze(1).expand(B, N, 3)
            w_robot_base_quat = w_robot_base_quat.unsqueeze(1).expand(B, N, 4)
            target_pos = target_pose[..., :3]
            target_quat = target_pose[..., 3:]
        else:  # (B, 7)
            target_pos = target_pose[:, :3]
            target_quat = target_pose[:, 3:]

        robot_2_grasp_pos, robot_2_grasp_quat = (
            math_utils.frame_transform_subtract(
                w_robot_base_pos,
                w_robot_base_quat,
                target_pos,
                target_quat,
            )
        )

        robot_2_grasp_pose = torch.cat(
            (robot_2_grasp_pos, robot_2_grasp_quat), dim=-1
        ).to(target_pose.device)

        return robot_2_grasp_pose

    def _pose_filter(
        self,
        resolved: ResolvedManipulatorProfile,
        runtime_state: _ManipulatorRuntimeState,
        origin_poses: torch.Tensor,
    ) -> torch.Tensor:
        """origin_poses: (B, 7) or (B, N, 7) tensor in env frame."""
        # step1: pre process
        if origin_poses.dim() == 2:  # (B, 7)
            B = origin_poses.shape[0]
            ori_pose = origin_poses
        elif origin_poses.dim() == 3:  # (B, N, 7)
            B, N = origin_poses.shape[0], origin_poses.shape[1]
            ori_pose = origin_poses.view(B * N, 7)
        else:
            raise ValueError(
                f"Invalid origin_poses dim: {origin_poses.dim()}, "
                "should be 2 or 3."
            )

        grasp_poses_all = PoseAugmentor.augment_single_axis(
            ori_pose, "y", (-30, 30), 9
        )
        grasp_poses_all = grasp_poses_all.view(B, -1, 7)

        # trans grasp to robot base and robot origin ee
        std_ee_in_robot_base = self._convert_pose_to_robot_base(
            runtime_state, grasp_poses_all
        )

        # trans to real robot ee
        robot_ee_pose = self._convert_standard_ee_to_robot_ee(
            resolved,
            std_ee_in_robot_base,
        )

        final_pose, best_id = resolved.planner.filter_poses_with_IK(
            robot_ee_pose, runtime_state.current_joint_positions
        )
        print(f"PICK FINAL POSE:{final_pose}, BEST ID:{best_id}")

        failed_mask = best_id == -1
        if failed_mask.any():
            failed_batch_ids = torch.where(failed_mask)[0]
            print(
                f"Env [{failed_batch_ids.tolist()}] Ik failed, "
                "Use default pose"
            )

        best_id = best_id.to(device=grasp_poses_all.device)
        safe_best_id = best_id.clamp(min=0)
        batch_indices = torch.arange(
            grasp_poses_all.shape[0],
            device=grasp_poses_all.device,
        )
        selected = grasp_poses_all[batch_indices, safe_best_id]
        fallback = grasp_poses_all[:, 0]
        result = torch.where(failed_mask.unsqueeze(-1), fallback, selected)

        return result

    def _multi_pose_filter(
        self,
        resolved: ResolvedManipulatorProfile,
        runtime_state: _ManipulatorRuntimeState,
        origin_poses: torch.Tensor,
        reference_axis_world: torch.Tensor,
        angle_range_deg: tuple[float, float],
        pose_augument_config: dict[str, tuple[float, float, int]],
    ) -> torch.Tensor:
        """origin_poses: (B, 7) or (B, N, 7) tensor in env frame."""
        # step1: pre process
        if origin_poses.dim() == 2:  # (B, 7)
            B = origin_poses.shape[0]
            ori_pose = origin_poses
        elif origin_poses.dim() == 3:  # (B, N, 7)
            B, N = origin_poses.shape[0], origin_poses.shape[1]
            ori_pose = origin_poses.view(B * N, 7)
        else:
            raise ValueError(
                f"Invalid origin_poses dim: {origin_poses.dim()}, "
                "should be 2 or 3."
            )

        augment_pose_env = PoseAugmentor.augment_multi_axis(
            ori_pose, pose_augument_config
        )
        augment_pose_env = augment_pose_env.view(B, -1, 7)

        # step3: filter pose by angle
        top_down_pose, top_down_mask = self._ang_filter(
            augment_pose_env,
            "z",
            reference_axis_world,
            angle_range_deg,
        )  # (B, N, 7)

        # step4: filter with IK and min joint cost
        # trans grasp to robot base and robot origin ee
        std_ee_in_robot_base = self._convert_pose_to_robot_base(
            runtime_state, top_down_pose
        )

        # trans to real robot ee
        robot_ee_pose = self._convert_standard_ee_to_robot_ee(
            resolved,
            std_ee_in_robot_base,
        )

        final_pose, best_id = resolved.planner.filter_poses_with_IK(
            robot_ee_pose, runtime_state.current_joint_positions
        )
        print(f"PICK FINAL POSE:{final_pose}, BEST ID:{best_id}")

        failed_mask = best_id == -1
        if failed_mask.any():
            failed_batch_ids = torch.where(failed_mask)[0]
            print(
                f"Env [{failed_batch_ids.tolist()}] Ik failed, "
                "Use default pose"
            )

        best_id = best_id.to(device=augment_pose_env.device)
        safe_best_id = best_id.clamp(min=0)
        batch_indices = torch.arange(
            augment_pose_env.shape[0],
            device=augment_pose_env.device,
        )
        selected = augment_pose_env[batch_indices, safe_best_id]
        fallback = augment_pose_env[:, 0]
        result = torch.where(failed_mask.unsqueeze(-1), fallback, selected)
        has_angle_candidate = torch.any(top_down_mask, dim=1)
        result = torch.where(
            has_angle_candidate.unsqueeze(-1),
            result,
            fallback,
        )

        return result

    def _ang_filter(
        self,
        origin_poses: torch.Tensor,
        axis: str,
        reference_axis_world: torch.Tensor,
        angle_range_deg: tuple[float, float],
    ) -> torch.Tensor:
        # 1. --- Input Validation and Preparation ---
        if origin_poses.dim() != 3 or origin_poses.shape[-1] != 7:
            raise ValueError(
                f"Expected origin_poses to have shape (B, N, 7),"
                f"but got {origin_poses.shape}"
            )

        device = origin_poses.device

        # Ensure axes are normalized 3D vectors on the correct device
        if axis == "x":
            axis_in_pose_frame = torch.tensor([1.0, 0.0, 0.0], device=device)
        elif axis == "y":
            axis_in_pose_frame = torch.tensor([0.0, 1.0, 0.0], device=device)
        elif axis == "z":
            axis_in_pose_frame = torch.tensor([0.0, 0.0, 1.0], device=device)

        reference_axis_world = reference_axis_world.to(device).float()
        reference_axis_world = math_utils.normalize(reference_axis_world)

        min_angle_rad = torch.deg2rad(
            torch.tensor(angle_range_deg[0], device=device)
        )
        max_angle_rad = torch.deg2rad(
            torch.tensor(angle_range_deg[1], device=device)
        )

        quats = origin_poses[..., 3:]  # Shape: (B, N, 4)

        rotated_axis_world = math_utils.quaternion_apply_point(
            quats, axis_in_pose_frame
        )

        #  Calculate the Angle ---
        dot_product = torch.sum(
            rotated_axis_world * reference_axis_world, dim=-1
        )
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angles_rad = torch.acos(dot_product)

        mask = (angles_rad >= min_angle_rad) & (angles_rad <= max_angle_rad)

        valid_pose = origin_poses * mask.unsqueeze(-1)  # (num_envs, M, 7)
        return valid_pose, mask

    def _identity_pose_like(self, pose: torch.Tensor) -> torch.Tensor:
        identity = torch.zeros(
            pose.shape[0],
            7,
            dtype=pose.dtype,
            device=pose.device,
        )
        identity[:, 3] = 1.0
        return identity


class PickExecutorCfg(BaseExecutorCfg):
    """Configuration for :class:`PickExecutor`."""

    class_type: ClassType_co[PickExecutor] = PickExecutor
    action_type: str = "pick"

    target: PoseGeneratorCfg | None = None
    pick_object_info: ObjectInfo | None = None
    pre_grasp: PoseGeneratorCfg | None = None
    grasp_mode: GraspMode = "Default"
    close_gripper_steps: int = 20
