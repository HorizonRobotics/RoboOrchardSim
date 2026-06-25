"""Pose generator helpers used by trajectory executors."""

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

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import robo_orchard_core.utils.math as math_utils
import torch
from robo_orchard_core.utils.config import ClassConfig

from robo_orchard_sim.utils.config import ClassType_co

if TYPE_CHECKING:
    from robo_orchard_sim.task_components.trajs_gen.base_executor import (
        BaseExecutor,
    )


# -------------data type----------------------
@dataclass
class PoseGenerationContext:
    robot_base_pose_w: torch.Tensor
    """robot base pose in world frame, shape (B, 7)"""
    ee_pose_w: torch.Tensor
    """end-effector pose in world frame, shape (B, 7)"""
    current_joint_pos: torch.Tensor
    """current joint positions, shape (B, num_joints)"""
    executor: BaseExecutor
    """the executor that uses this context"""
    # default_joint_pos: torch.Tensor | None = None
    """default joint positions, shape (B, num_joints)"""


@dataclass
class MotionPose(metaclass=ABCMeta):
    type: Literal["pose", "joint"]
    data: torch.Tensor
    """ type pose, data is (B, 7) tensor representing [x, y, z, qw, qx, qy, qz]
    in robot frame
        type joint, data is (B, num_joints) tensor representing joint positions
    """


class PoseGenerator(metaclass=ABCMeta):
    cfg: "PoseGeneratorCfg"

    def __init__(self, cfg: "PoseGeneratorCfg") -> None:
        self.cfg = cfg

    @abstractmethod
    def generate(self, context: PoseGenerationContext) -> MotionPose:
        raise NotImplementedError

    def _trans_to_robot_base(
        self, pose_w: torch.Tensor, robot_base_pose_w: torch.Tensor
    ) -> torch.Tensor:
        """Transform a pose from world frame to robot base frame."""
        robot_2_target_pos, robot_2_target_quat = (
            math_utils.frame_transform_subtract(
                robot_base_pose_w[..., :3],
                robot_base_pose_w[..., 3:],
                pose_w[..., :3],
                pose_w[..., 3:],
            )
        )

        robot_2_target_pose = torch.cat(
            (robot_2_target_pos, robot_2_target_quat), dim=-1
        ).to(pose_w.device)

        return robot_2_target_pose


def _validate_joint_value_count(
    joint_id_idxs: list[int],
    joint_values: list[float],
    cfg_name: str,
    value_name: str,
) -> None:
    if len(joint_values) != len(joint_id_idxs):
        raise ValueError(
            f"{cfg_name} must have the same number of joint values as "
            f"joint indices: got {len(joint_values)} {value_name} values "
            f"for {len(joint_id_idxs)} joint indices."
        )


class FixedPoseGenerator(PoseGenerator):
    cfg: "FixedPoseCfg"

    def __init__(self, cfg: "FixedPoseCfg") -> None:
        super().__init__(cfg)

        if len(cfg.pose) != 7:
            raise ValueError(
                "FixedPoseCfg pose must be of shape (x,y,z,qw,qx,qy,qz)"
            )

    def generate(self, context: PoseGenerationContext) -> MotionPose:
        B = context.robot_base_pose_w.shape[0]
        device = context.robot_base_pose_w.device
        pose = torch.tensor(self.cfg.pose, device=device).repeat(B, 1)

        if self.cfg.frame == "world":
            target_pose = self._trans_to_robot_base(
                pose, context.robot_base_pose_w
            )
        elif self.cfg.frame == "robot_base":
            target_pose = pose
        else:
            raise ValueError(f"Invalid frame: {self.cfg.frame}")

        return MotionPose(type="pose", data=target_pose)


class FixedJointGenerator(PoseGenerator):
    cfg: "FixedJointCfg"

    def __init__(self, cfg: "FixedJointCfg") -> None:
        super().__init__(cfg)
        _validate_joint_value_count(
            joint_id_idxs=cfg.joint_id_idxs,
            joint_values=cfg.joints,
            cfg_name="FixedJointCfg",
            value_name="joint",
        )

    def generate(self, context: PoseGenerationContext) -> MotionPose:
        device = context.current_joint_pos.device
        target_joint_positions = context.current_joint_pos.clone()

        target_joint_positions[:, self.cfg.joint_id_idxs] = torch.tensor(
            self.cfg.joints
        ).to(device)

        return MotionPose(type="joint", data=target_joint_positions)


class MoveByDisplacementPoseGenerator(PoseGenerator):
    cfg: "MoveByDisplacementCfg"

    def __init__(self, cfg: "MoveByDisplacementCfg") -> None:
        super().__init__(cfg)

        if cfg.direction not in ["x", "y", "z"]:
            raise ValueError(f"Invalid direction: {cfg.direction}")
        if cfg.frame not in ["world", "gripper"]:
            raise ValueError(f"Invalid frame: {cfg.frame}")

    def generate(self, context: PoseGenerationContext) -> MotionPose:
        device = context.robot_base_pose_w.device
        ee_pose_w = context.ee_pose_w.clone()

        if self.cfg.frame == "world":
            offset = self._gen_offset_from_direction(
                self.cfg.direction, self.cfg.distance
            ).to(device)
            target_pose_w = ee_pose_w
            target_pose_w[:, :3] += offset

        elif self.cfg.frame == "gripper":
            rotation_matrix = math_utils.quaternion_to_matrix(ee_pose_w[:, 3:])

            if self.cfg.direction == "x":
                axis = rotation_matrix[:, :, 0]  # Shape: (batch_size, 3)
            elif self.cfg.direction == "y":
                axis = rotation_matrix[:, :, 1]
            elif self.cfg.direction == "z":
                axis = rotation_matrix[:, :, 2]

            target_pose_w = ee_pose_w
            target_pose_w[:, :3] += axis * self.cfg.distance
        else:
            raise ValueError(f"Invalid frame: {self.cfg.frame}")

        target_pose = self._trans_to_robot_base(
            target_pose_w, context.robot_base_pose_w
        )

        return MotionPose(type="pose", data=target_pose)

    def _gen_offset_from_direction(
        self, direction: str, distance: float
    ) -> torch.Tensor:
        if direction == "x":
            offset = torch.tensor([1.0, 0.0, 0.0]) * distance
        elif direction == "y":
            offset = torch.tensor([0.0, 1.0, 0.0]) * distance
        elif direction == "z":
            offset = torch.tensor([0.0, 0.0, 1.0]) * distance
        else:
            raise ValueError(f"Invalid direction: {direction}")
        return offset


class MoveByJointOffsetPoseGenerator(PoseGenerator):
    cfg: "MoveByJointOffsetCfg"

    def __init__(self, cfg: "MoveByJointOffsetCfg") -> None:
        super().__init__(cfg)
        _validate_joint_value_count(
            joint_id_idxs=cfg.joint_id_idxs,
            joint_values=cfg.joint_offsets,
            cfg_name="MoveByJointOffsetCfg",
            value_name="joint offset",
        )

    def generate(self, context: PoseGenerationContext) -> MotionPose:
        device = context.robot_base_pose_w.device
        current_joint_pos = context.current_joint_pos.clone()

        offset = torch.tensor(self.cfg.joint_offsets).to(device).unsqueeze(0)

        target_joint_positions = current_joint_pos
        target_joint_positions[:, self.cfg.joint_id_idxs] += offset

        return MotionPose(type="joint", data=target_joint_positions)


# class MoveToDefaultPoseGenerator(PoseGenerator):
#     cfg: "MoveToDefaultCfg"

#     def __init__(self, cfg: "MoveToDefaultCfg") -> None:
#         super().__init__(cfg)

#     def generate(self, context: PoseGenerationContext) -> MotionPose:
#         device = context.robot_base_pose_w.device
#         B, DOF = context.current_joint_pos.shape

#         if context.default_joint_pos is not None:
#             default_joint_pos = context.default_joint_pos.to(device)
#         else:
#             default_joint_pos = context.executor.cfg.default_joint_positions
#             if not isinstance(default_joint_pos, torch.Tensor):
#                 default_joint_pos = torch.tensor(
#                     default_joint_pos, device=device
#                 )[:DOF]

#             default_joint_pos = default_joint_pos.unsqueeze(0).repeat(B, 1)

#         return MotionPose(type="joint", data=default_joint_pos)


class MoveByRandOffsetPoseGenerator(PoseGenerator):
    cfg: "MoveByRandOffsetCfg"

    def __init__(self, cfg: "MoveByRandOffsetCfg") -> None:
        super().__init__(cfg)

    def generate(self, context: PoseGenerationContext) -> MotionPose:
        import random

        device = context.robot_base_pose_w.device
        ee_pose_w = context.ee_pose_w.clone()
        rotation_matrix = math_utils.quaternion_to_matrix(ee_pose_w[:, 3:])

        if self.cfg.trans_directions:
            trans_offset = []
            for dir_, range_ in zip(
                self.cfg.trans_directions, self.cfg.trans_ranges, strict=False
            ):
                if dir_ not in ["x", "y", "z"]:
                    raise ValueError(f"Invalid direction: {dir_}")
                if len(range_) != 2:
                    raise ValueError(
                        f"Range must be of length 2, got {len(range_)}"
                    )
                if dir_ == "x":
                    axis = rotation_matrix[:, :, 0]
                elif dir_ == "y":
                    axis = rotation_matrix[:, :, 1]
                elif dir_ == "z":
                    axis = rotation_matrix[:, :, 2]
                rand_offset = random.uniform(*range_)
                trans_offset.append(rand_offset)
                ee_pose_w[:, :3] += axis * rand_offset
            print(
                f"Random translation for {self.cfg.trans_directions}: {trans_offset}"  # noqa: E501
            )

        if self.cfg.rot_directions:
            euler_angles = torch.zeros(3).to(device)
            for rot_, range_ in zip(
                self.cfg.rot_directions, self.cfg.rot_ranges, strict=False
            ):
                if rot_ not in ["x", "y", "z"]:
                    raise ValueError(f"Invalid rotation direction: {rot_}")
                if len(range_) != 2:
                    raise ValueError(
                        f"Rotation range must be 2, got {len(range_)}"
                    )
                if rot_ == "x":
                    euler_angles[0] = random.uniform(*range_)
                elif rot_ == "y":
                    euler_angles[1] = random.uniform(*range_)
                elif rot_ == "z":
                    euler_angles[2] = random.uniform(*range_)
            print(
                f"Random rotation for {self.cfg.rot_directions}: {euler_angles.tolist()[: len(self.cfg.rot_directions)]}"  # noqa: E501
            )

            delta_matrix = math_utils.euler_angles_to_matrix(
                euler_angles, "XYZ"
            )
            delta_quaternions = (
                math_utils.matrix_to_quaternion(delta_matrix)
                .unsqueeze(0)
                .repeat(ee_pose_w.shape[0], 1)
            )
            ee_pose_w[:, 3:] = math_utils.quaternion_multiply(
                ee_pose_w[:, 3:], delta_quaternions
            )

        target_pose = self._trans_to_robot_base(
            ee_pose_w, context.robot_base_pose_w
        )
        return MotionPose(type="pose", data=target_pose)


class PoseGeneratorCfg(ClassConfig[PoseGenerator]):
    class_type: ClassType_co[PoseGenerator] = PoseGenerator
    type: str = ""

    def __call__(self, **kwargs: Any) -> PoseGenerator:
        return self.class_type(self, **kwargs)


class FixedPoseCfg(PoseGeneratorCfg):
    class_type: ClassType_co[FixedPoseGenerator] = FixedPoseGenerator

    pose: list[float]
    frame: Literal["world", "robot_base"] = "world"
    type: str = "FixedPoseCfg"


class FixedJointCfg(PoseGeneratorCfg):
    class_type: ClassType_co[FixedJointGenerator] = FixedJointGenerator

    joints: list[float]
    joint_id_idxs: list[int]
    type: str = "FixedJointCfg"


class MoveByDisplacementCfg(PoseGeneratorCfg):
    # TODO: define what is gripper frame
    class_type: ClassType_co[MoveByDisplacementPoseGenerator] = (
        MoveByDisplacementPoseGenerator
    )

    distance: float
    frame: Literal["world", "gripper"] = "gripper"
    direction: Literal["x", "y", "z"] = "z"
    type: str = "MoveByDisplacementCfg"


# class MoveToDefaultCfg(PoseGeneratorCfg):
#     class_type: ClassType_co[MoveToDefaultPoseGenerator] = (
#         MoveToDefaultPoseGenerator
#     )
#     type: str = "MoveToDefault"


class MoveByJointOffsetCfg(PoseGeneratorCfg):
    class_type: ClassType_co[MoveByJointOffsetPoseGenerator] = (
        MoveByJointOffsetPoseGenerator
    )

    joint_id_idxs: list[int]
    joint_offsets: list[float]
    type: str = "MoveByJointOffsetCfg"


class MoveByRandOffsetCfg(PoseGeneratorCfg):
    class_type: ClassType_co[MoveByRandOffsetPoseGenerator] = (
        MoveByRandOffsetPoseGenerator
    )

    trans_directions: list[Literal["x", "y", "z"]] | None = None
    trans_ranges: list[list[float]] | None = None
    rot_directions: list[Literal["x", "y", "z"]] | None = None
    rot_ranges: list[list[float]] | None = None
    type: str = "MoveByRandOffsetCfg"
