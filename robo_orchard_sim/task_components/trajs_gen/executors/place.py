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

"""Place executors for object-to-object placement planning."""

from __future__ import annotations
import re
from typing import Any, Literal

import robo_orchard_core.utils.math as math_utils
import torch

from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ResolvedManipulatorProfile,
)
from robo_orchard_sim.task_components.trajs_gen.base_executor import (
    BaseExecutor,
    BaseExecutorCfg,
    ObjectInfo,
    Trajectories,
    _ManipulatorRuntimeState,
)
from robo_orchard_sim.task_components.trajs_gen.manipulator_resolver import (
    ManipulatorBindingContext,
)
from robo_orchard_sim.task_components.trajs_gen.pose_generator import (
    MotionPose,
    MoveByDisplacementCfg,
    PoseGenerationContext,
    PoseGeneratorCfg,
)
from robo_orchard_sim.task_components.validators.utils import (
    is_object_center_in_obb,
)
from robo_orchard_sim.utils.config import ClassType_co
from robo_orchard_sim.utils.env_utils import PoseAugmentor

PlaceConstrain = Literal["align_dir_axis", "free", "align", "follow_ee"]


class PlaceExecutor(BaseExecutor):
    """Plan pre-place, place, and gripper-open trajectories."""

    cfg: "PlaceExecutorCfg"

    def __init__(self, cfg: "PlaceExecutorCfg") -> None:
        super().__init__(cfg)
        pre_place_cfg = cfg.pre_place_cfg or MoveByDisplacementCfg(
            distance=-0.02,
            direction="z",
            frame="gripper",
        )
        self._pre_place_pose_gen = pre_place_cfg()

    def plan(
        self,
        env: Any,
        context: ManipulatorBindingContext,
    ) -> Trajectories:
        """Resolve robot info and plan a full place action."""
        resolved = self.cfg.resolve_manipulator_info(env, context=context)
        self.last_resolved_manipulator = resolved

        runtime_state = self.build_runtime_state(
            env=env,
            resolved=resolved,
            frame="world",
        )

        target_pose = None
        if (
            self.cfg.pick_object_info is not None
            and self.cfg.place_object_info is not None
        ):
            target_pose = self._generate_place_target(
                env=env,
                resolved=resolved,
                runtime_state=runtime_state,
            )
        else:
            raise ValueError(
                "pick_object_info and place_object_info should be provided"
            )

        robot_base_pose_env = runtime_state.robot_base_pose_w.clone()
        robot_base_pose_env[:, :3] -= env.scene.env_origins[:]

        robot_2_place_pose = self._convert_pose_to_robot_base(
            robot_base_pose_env,
            target_pose,
        )
        debug_target_pose = self.build_debug_target_pose(
            env=env,
            runtime_state=_ManipulatorRuntimeState(
                frame="env",
                current_joint_positions=runtime_state.current_joint_positions,
                robot_base_pose_w=robot_base_pose_env,
                ee_pose_w=runtime_state.ee_pose_w,
            ),
            pose_robot_base=robot_2_place_pose,
            name="place_target",
        )

        robot_2_pre_place_pose = self._calculate_pre_place(
            resolved=resolved,
            target_pose_w=target_pose,
            robot_base_pose_w=robot_base_pose_env,
            current_joint_positions=runtime_state.current_joint_positions,
        )
        trajs_mode = self._resolve_trajs_mode()

        open_val = self._resolve_gripper_value(resolved, state="OPEN")
        close_val = self._resolve_gripper_value(resolved, state="CLOSED")
        traj1, success_flag1 = self.gen_to_target_pose_trajs(
            planner=resolved.planner,
            start_joint_positions=runtime_state.current_joint_positions,
            target_pose=robot_2_pre_place_pose,
            gripper_val=close_val,
            mode=trajs_mode,
        )

        if not torch.all(success_flag1).item():
            print("[PlaceExecutor] Pre-place traj generation failed.")

        traj2, success_flag2 = self.gen_to_target_pose_trajs(
            planner=resolved.planner,
            start_joint_positions=self.get_last_trajs(traj1)[
                :, : len(resolved.joint_ids)
            ],
            target_pose=robot_2_place_pose,
            gripper_val=close_val,
            mode=trajs_mode,
        )

        if not torch.all(success_flag2).item():
            print("[PlaceExecutor] Place traj generation failed.")

        open_traj = self.gen_gripper_trajs(
            current_joint_positions=self.get_last_trajs(traj2)[
                :, : len(resolved.joint_ids)
            ],
            start_gripper_val=close_val,
            end_gripper_val=open_val,
            length=20,
        )

        trajectories = self._merge_trajs(traj1, traj2, open_traj)
        success = bool(torch.all(success_flag1 & success_flag2).item())

        return Trajectories(
            trajectories=trajectories,
            success=success,
            resolved_manipulator=resolved,
            debug_target_poses=(debug_target_pose,),
        )

    def check_success(self, env: Any) -> bool:
        """Check whether the picked object center lies in the place OBB."""
        if (
            self.cfg.pick_object_info is None
            or self.cfg.place_object_info is None
        ):
            raise ValueError(
                "pick_object_info and place_object_info should be provided"
            )
        pick_object_info = self.cfg.pick_object_info
        place_object_info = self.cfg.place_object_info

        scene = env.scene
        place_pose_array = (
            scene[place_object_info.name]
            .data.root_state_w[:, :7]
            .cpu()
            .numpy()
        )

        path = scene[place_object_info.name].cfg.prim_path
        prim_path = re.sub(r"\.\*", "0", path)

        pick_actor_center = (
            scene[pick_object_info.name].data.root_pos_w[0].cpu().numpy()
        )

        return is_object_center_in_obb(
            scene.stage,
            prim_path,
            place_pose_array,
            pick_actor_center,
        )

    def _generate_place_target(
        self,
        env: Any,
        resolved: ResolvedManipulatorProfile,
        runtime_state: _ManipulatorRuntimeState,
    ) -> torch.Tensor:
        """Return place target pose in env frame."""
        if (
            self.cfg.pick_object_info is None
            or self.cfg.place_object_info is None
        ):
            raise ValueError(
                "pick_object_info and place_object_info should be provided"
            )
        pick_object_info = self.cfg.pick_object_info
        place_object_info = self.cfg.place_object_info

        pick_name = pick_object_info.name
        pick_part = env.scene[pick_name].get_element_pose(
            mode=pick_object_info.mode,
            action=pick_object_info.action,
            part=pick_object_info.part,
            id=[[0]],
        )

        place_name = place_object_info.name
        place_part = env.scene[place_name].get_element_pose(
            mode=place_object_info.mode,
            action=place_object_info.action,
            part=place_object_info.part,
            id=[[0]],
        )

        place_pose_w = (
            torch.cat((place_part.pos, place_part.quat), dim=-1)
            .to(pick_part.pos.device)
            .squeeze(1)
        )
        pick_pose_w = (
            torch.cat((pick_part.pos, pick_part.quat), dim=-1)
            .to(pick_part.pos.device)
            .squeeze(1)
        )
        place_part_direction_w = place_part.get_axis("x").squeeze(1)

        if self.cfg.constrain == "follow_ee":
            pick_pose_w = self._convert_ee_to_pick_pose(
                resolved,
                runtime_state.ee_pose_w,
            )

        current_world_to_ee_pose = runtime_state.ee_pose_w

        if self.cfg.constrain in ("free", "follow_ee"):
            pose_with_z_aug_w = _gen_multi_pose_with_axis(
                place_pose_w,
                place_part_direction_w,
                9,
            )

            batch_size, pose_count, _ = pose_with_z_aug_w.shape
            pose_with_z_aug_w = pose_with_z_aug_w.view(-1, 7)

            rotation_config: dict[str, tuple[float, float, int]] = {
                "x": (-30.0, 30.0, 5),
                "y": (-40.0, 40.0, 5),
                "z": (-30.0, 30.0, 5),
            }

            multi_augmented = PoseAugmentor.augment_multi_axis(
                pose_with_z_aug_w,
                rotation_config,
            )
            augmented_count = multi_augmented.shape[1]
            multi_augmented = (
                multi_augmented.contiguous()
                .view(batch_size, pose_count, augmented_count, 7)
                .reshape(batch_size, pose_count * augmented_count, 7)
            )

            pose_candidate_ee_w = self._get_ee_from_pick_and_place(
                pick_pose_w.unsqueeze(1),
                multi_augmented,
                current_world_to_ee_pose.unsqueeze(1),
            )

            target_place_pose, _ = self._ik_filter_closest_pose(
                resolved=resolved,
                pose_candidate=pose_candidate_ee_w,
                ref_pose=current_world_to_ee_pose,
                robot_base_pose=runtime_state.robot_base_pose_w,
                fill_with_original_pose=False,
            )

            target_place_pose[..., :3] -= env.scene.env_origins[:]

            return target_place_pose

        if self.cfg.constrain == "align":
            rotation_config: dict[str, tuple[float, float, int]] = {
                "x": (-20.0, 20.0, 4),
                "y": (-10.0, 10.0, 4),
                "z": (-20.0, 20.0, 4),
            }

            multi_augmented = PoseAugmentor.augment_multi_axis(
                place_pose_w,
                rotation_config,
            )
            multi_poses = torch.cat(
                [place_pose_w.unsqueeze(1), multi_augmented],
                dim=1,
            )

            multi_poses_w = self._get_ee_from_pick_and_place(
                pick_pose_w.unsqueeze(1),
                multi_poses,
                current_world_to_ee_pose.unsqueeze(1),
            )

            original_pose_w = multi_poses_w[:, 0, :]
            place_part_pose_w = multi_poses_w[:, 1:, :]

            place_target_pose, _ = self._ik_filter_closest_pose(
                resolved=resolved,
                pose_candidate=place_part_pose_w,
                ref_pose=original_pose_w,
                robot_base_pose=runtime_state.robot_base_pose_w,
            )
            place_target_pose[..., :3] -= env.scene.env_origins[:]

            ori_place_part_pose = place_part_pose_w[:, 0, :].clone()
            ori_place_part_pose[..., :3] -= env.scene.env_origins[:]

            return place_target_pose

        pose_with_z_aug_w = _gen_multi_pose_with_axis(
            place_pose_w,
            place_part_direction_w,
            9,
        )

        multi_poses = torch.cat(
            [place_pose_w.unsqueeze(1), pose_with_z_aug_w],
            dim=1,
        )

        multi_poses_w = self._get_ee_from_pick_and_place(
            pick_pose_w.unsqueeze(1),
            multi_poses,
            current_world_to_ee_pose.unsqueeze(1),
        )

        original_pose_w = multi_poses_w[:, 0, :]
        place_part_pose_w = multi_poses_w[:, 1:, :]

        place_target_pose, _ = self._ik_filter_closest_pose(
            resolved=resolved,
            pose_candidate=place_part_pose_w,
            ref_pose=original_pose_w,
            robot_base_pose=runtime_state.robot_base_pose_w,
        )
        place_target_pose[..., :3] -= env.scene.env_origins[:]

        ori_place_part_pose = place_part_pose_w[:, 0, :].clone()
        ori_place_part_pose[..., :3] -= env.scene.env_origins[:]

        return place_target_pose

    def _ik_filter_closest_pose(
        self,
        resolved: ResolvedManipulatorProfile,
        pose_candidate: torch.Tensor,
        ref_pose: torch.Tensor,
        robot_base_pose: torch.Tensor,
        pos_weight: float = 0.5,
        rot_weight: float = 0.5,
        fill_with_original_pose: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select candidate poses via IK and distance to a reference pose."""
        batch_size, _, _ = pose_candidate.shape

        cand_robot = self._convert_pose_to_robot_base(
            robot_base_pose,
            pose_candidate,
        )
        ik_results = resolved.planner.graph_ik(cand_robot)
        success_mask = ik_results.success

        ref_robot = self._convert_pose_to_robot_base(robot_base_pose, ref_pose)

        ref_pos = ref_robot[:, :3].unsqueeze(1)
        cand_pos = pose_candidate[:, :, :3]
        pos_diff = cand_pos - ref_pos
        position_costs = torch.sum(pos_diff**2, dim=-1)

        ref_quat = ref_pose[:, 3:].unsqueeze(1)
        cand_quat = pose_candidate[:, :, 3:]
        dot_product = torch.sum(ref_quat * cand_quat, dim=-1)
        rotation_costs = 1.0 - torch.abs(dot_product)

        costs = pos_weight * position_costs + rot_weight * rotation_costs

        costs_masked = costs.clone()
        costs_masked[~success_mask] = float("inf")

        _, best_success_idx = torch.min(costs_masked, dim=1)

        any_success = torch.any(success_mask, dim=1)
        best_poses = torch.empty(
            batch_size,
            7,
            device=pose_candidate.device,
            dtype=pose_candidate.dtype,
        )

        successful_batches = torch.where(any_success)[0]
        if successful_batches.numel() > 0:
            batch_idx = successful_batches
            cand_idx = best_success_idx[batch_idx]
            best_poses[batch_idx] = pose_candidate[batch_idx, cand_idx, :]

        failed_batches = torch.where(~any_success)[0]
        if failed_batches.numel() > 0:
            if fill_with_original_pose:
                best_poses[failed_batches] = ref_pose[failed_batches]
            else:
                failed_costs = costs[failed_batches]
                _, nearest_idx = torch.min(failed_costs, dim=1)
                best_poses[failed_batches] = pose_candidate[
                    failed_batches,
                    nearest_idx,
                    :,
                ]
            print(
                f"[IK Filter] All IK failed in batches {failed_batches.tolist()} "  # noqa: E501
                f"{'-> use ref_pose' if fill_with_original_pose else '-> use nearest candidate'}."  # noqa: E501
            )

        print(
            f"Best Pose ={best_poses}\n",
            f"Best_success_idx={(best_success_idx.tolist(),)}"
            f"Failed_batches={failed_batches.tolist()}",
        )

        return best_poses, failed_batches

    def _convert_pose_to_robot_base(
        self,
        robot_base_pose_w: torch.Tensor,
        target_pose: torch.Tensor,
    ) -> torch.Tensor:
        w_robot_base_pos = robot_base_pose_w[:, :3]
        w_robot_base_quat = robot_base_pose_w[:, 3:]

        if target_pose.dim() == 3:
            batch_size, pose_count = target_pose.shape[0], target_pose.shape[1]
            w_robot_base_pos = w_robot_base_pos.unsqueeze(1).expand(
                batch_size,
                pose_count,
                3,
            )
            w_robot_base_quat = w_robot_base_quat.unsqueeze(1).expand(
                batch_size,
                pose_count,
                4,
            )
            target_pos = target_pose[..., :3]
            target_quat = target_pose[..., 3:]
        else:
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
            (robot_2_grasp_pos, robot_2_grasp_quat),
            dim=-1,
        ).to(target_pose.device)

        return robot_2_grasp_pose

    def _calculate_pre_place(
        self,
        resolved: ResolvedManipulatorProfile,
        target_pose_w: torch.Tensor,
        robot_base_pose_w: torch.Tensor,
        current_joint_positions: torch.Tensor,
    ) -> torch.Tensor:
        robot_2_ee_pos, robot_2_ee_quat = math_utils.frame_transform_subtract(
            robot_base_pose_w[..., :3],
            robot_base_pose_w[..., 3:],
            target_pose_w[..., :3],
            target_pose_w[..., 3:],
        )

        robot_2_ee = torch.cat([robot_2_ee_pos, robot_2_ee_quat], dim=-1)

        ik_result = resolved.planner.graph_ik(robot_2_ee)
        joint_positions = ik_result.solution

        if isinstance(joint_positions, torch.Tensor):
            joint_positions = joint_positions.to(target_pose_w.device)
        else:
            joint_positions = torch.tensor(joint_positions).to(
                target_pose_w.device
            )
        joint_positions = joint_positions.view(current_joint_positions.shape)

        context = PoseGenerationContext(
            robot_base_pose_w=robot_base_pose_w,
            ee_pose_w=target_pose_w,
            current_joint_pos=joint_positions,
            executor=self,
        )

        pre_grasp_pose: MotionPose = self._pre_place_pose_gen.generate(context)

        if pre_grasp_pose.type == "joint":
            fk_pose = resolved.planner.fk(pre_grasp_pose.data, w_first=True)
            return fk_pose
        return pre_grasp_pose.data

    def _get_ee_from_pick_and_place(
        self,
        pick_pose: torch.Tensor,
        place_pose: torch.Tensor,
        world_to_ee_pose: torch.Tensor,
    ) -> torch.Tensor:
        t_tmp_pos, t_tmp_quat = math_utils.frame_transform_subtract(
            pick_pose[..., :3],
            pick_pose[..., 3:],
            world_to_ee_pose[..., :3],
            world_to_ee_pose[..., 3:],
        )
        place_target_pos, place_target_quat = (
            math_utils.frame_transform_combine(
                place_pose[..., :3],
                place_pose[..., 3:],
                t_tmp_pos,
                t_tmp_quat,
            )
        )

        ee_pose_w = torch.cat(
            (place_target_pos, place_target_quat),
            dim=-1,
        ).to(pick_pose.device)

        return ee_pose_w

    def _convert_ee_to_pick_pose(
        self,
        resolved: ResolvedManipulatorProfile,
        world_to_ee_pose: torch.Tensor,
    ) -> torch.Tensor:
        transform_obj: Any = resolved.t_standard_tcp_to_robot_ee
        if transform_obj is None:
            transform_obj = torch.eye(
                4,
                device=world_to_ee_pose.device,
                dtype=torch.float32,
            )
        elif hasattr(transform_obj, "to_homogeneous_matrix"):
            transform_obj = transform_obj.to_homogeneous_matrix()
        t_std_to_robot_ee_homo = torch.as_tensor(
            transform_obj,
            device=world_to_ee_pose.device,
            dtype=torch.float32,
        )

        t_pick_to_std = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).to(world_to_ee_pose.device)

        t_pick_to_ee = t_pick_to_std @ t_std_to_robot_ee_homo
        t_ee_to_pick = torch.inverse(t_pick_to_ee)

        t_ee_to_pick_rot = t_ee_to_pick[:3, :3]
        t_ee_to_pick_pos = t_ee_to_pick[:3, 3]

        T_inv_pos = t_ee_to_pick_pos
        T_inv_quat = math_utils.matrix_to_quaternion(t_ee_to_pick_rot)

        world_to_ee_pos = world_to_ee_pose[..., :3]
        world_to_ee_quat = world_to_ee_pose[..., 3:]

        pick_pos, pick_quat = math_utils.frame_transform_combine(
            world_to_ee_pos,
            world_to_ee_quat,
            T_inv_pos,
            T_inv_quat,
        )

        pick_pose = torch.cat((pick_pos, pick_quat), dim=-1).to(
            world_to_ee_pose.device
        )

        return pick_pose


class PrePlaceExecutor(PlaceExecutor):
    """Plan only the pre-place trajectory segment."""

    cfg: "PrePlaceExecutorCfg"

    def plan(
        self,
        env: Any,
        context: ManipulatorBindingContext,
    ) -> Trajectories:
        """Resolve robot info and plan only to the pre-place pose."""
        resolved = self.cfg.resolve_manipulator_info(env, context=context)
        self.last_resolved_manipulator = resolved

        runtime_state = self.build_runtime_state(
            env=env,
            resolved=resolved,
            frame="world",
        )

        target_pose = None
        if (
            self.cfg.pick_object_info is not None
            and self.cfg.place_object_info is not None
        ):
            target_pose = self._generate_place_target(
                env=env,
                resolved=resolved,
                runtime_state=runtime_state,
            )
        else:
            raise ValueError(
                "pick_object_info and place_object_info should be provided"
            )

        robot_base_pose_env = runtime_state.robot_base_pose_w.clone()
        robot_base_pose_env[:, :3] -= env.scene.env_origins[:]

        robot_2_pre_place_pose = self._calculate_pre_place(
            resolved=resolved,
            target_pose_w=target_pose,
            robot_base_pose_w=robot_base_pose_env,
            current_joint_positions=runtime_state.current_joint_positions,
        )

        debug_target_pose = self.build_debug_target_pose(
            env=env,
            runtime_state=_ManipulatorRuntimeState(
                frame="env",
                current_joint_positions=runtime_state.current_joint_positions,
                robot_base_pose_w=robot_base_pose_env,
                ee_pose_w=runtime_state.ee_pose_w,
            ),
            pose_robot_base=robot_2_pre_place_pose,
            name="pre_place_target",
        )

        trajs_mode = self._resolve_trajs_mode()

        close_val = self._resolve_gripper_value(resolved, state="CLOSED")
        trajectories, success_flag = self.gen_to_target_pose_trajs(
            planner=resolved.planner,
            start_joint_positions=runtime_state.current_joint_positions,
            target_pose=robot_2_pre_place_pose,
            gripper_val=close_val,
            mode=trajs_mode,
        )
        success = bool(torch.all(success_flag).item())
        if not success:
            print("[PrePlaceExecutor] Pre-place traj generation failed.")

        return Trajectories(
            trajectories=trajectories,
            success=success,
            resolved_manipulator=resolved,
            debug_target_poses=(debug_target_pose,),
        )


class PlaceExecutorCfg(BaseExecutorCfg):
    """Configuration for :class:`PlaceExecutor`."""

    class_type: ClassType_co[PlaceExecutor] = PlaceExecutor
    action_type: str = "place"

    pick_object_info: ObjectInfo | None = None
    place_object_info: ObjectInfo | None = None
    pre_place_cfg: PoseGeneratorCfg | None = None
    constrain: PlaceConstrain = "align_dir_axis"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.constrain not in (
            "align_dir_axis",
            "free",
            "align",
            "follow_ee",
        ):
            raise ValueError(
                f"Invalid constrain value: '{self.constrain}'. Must be one of "
                "'align_dir_axis', 'free', 'align', or 'follow_ee'."
            )


class PrePlaceExecutorCfg(PlaceExecutorCfg):
    """Configuration for :class:`PrePlaceExecutor`."""

    class_type: ClassType_co[PrePlaceExecutor] = PrePlaceExecutor
    action_type: str = "pre_place"


def _gen_multi_pose_with_axis(
    base_pose: torch.Tensor,
    axis_vector: torch.Tensor,
    num_poses: int = 4,
) -> torch.Tensor:
    """Generate poses around a given axis vector."""
    batch_size = base_pose.shape[0]
    device = base_pose.device

    base_pos = base_pose[:, :3]
    base_quat = base_pose[:, 3:]

    axis_vector = math_utils.normalize(axis_vector)

    angles = torch.linspace(0, 2 * torch.pi, num_poses + 1, device=device)[:-1]

    half_angles = angles / 2.0
    cos_half = torch.cos(half_angles)
    sin_half = torch.sin(half_angles)

    rotation_quats = torch.zeros(
        num_poses,
        4,
        device=device,
        dtype=base_pose.dtype,
    )
    rotation_quats[:, 0] = cos_half
    rotation_quats_xyz = sin_half.view(
        1,
        num_poses,
        1,
    ) * axis_vector.unsqueeze(1)

    rotated_quats = torch.zeros(
        batch_size,
        num_poses,
        4,
        device=device,
        dtype=base_pose.dtype,
    )
    rotated_quats[:, :, 0] = cos_half
    rotated_quats[:, :, 1:] = rotation_quats_xyz

    final_quats = math_utils.quaternion_multiply(
        rotated_quats,
        base_quat.unsqueeze(1),
    )

    result_poses = torch.cat(
        (base_pos.unsqueeze(1).expand(-1, num_poses, -1), final_quats),
        dim=-1,
    )

    return result_poses
