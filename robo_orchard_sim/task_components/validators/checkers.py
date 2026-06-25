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

import re
from typing import Sequence

import numpy as np
import torch

from robo_orchard_sim.task_components.validators.base import (
    GripperRange,
    ValidatorActor,
)

# Fraction of the close->open joint travel required to count as "open".
DEFAULT_GRIPPER_OPEN_RATIO = 0.8

# Keeps the boundary inclusive despite float rounding (0.04/0.05 < 0.8).
_OPEN_RATIO_TOL = 1e-6


def _gripper_joint_is_open(
    value: float, gripper_range: GripperRange, open_ratio: float
) -> bool:
    """Return whether a joint position is open enough.

    Direction-agnostic: open_val may be greater or less than close_val. A
    degenerate range (open == close) cannot gate and counts as open.
    """
    span = gripper_range.open_val - gripper_range.close_val
    if span == 0:
        return True
    ratio = (value - gripper_range.close_val) / span
    return ratio >= open_ratio - _OPEN_RATIO_TOL


class CheckerBase:
    """Base checker with a unified boolean check interface."""

    def check(self, env, env_idx: int = 0) -> bool:
        """Run checker logic and return whether the condition is met."""
        raise NotImplementedError

    def __call__(self, env, env_idx: int = 0) -> bool:
        """Allow checker instances to be called like functions."""
        return self.check(env, env_idx=env_idx)


class ActorBoundChecker(CheckerBase):
    """Base checker for logic that depends on a validator actor."""

    def __init__(self, actor: ValidatorActor):
        self.actor = actor
        self.actor_name = actor.name

    def _get_init_state(self) -> np.ndarray:
        if self.actor.init_state is None:
            raise ValueError(
                f"ValidatorActor '{self.actor_name}' init_state is not set."
            )
        return self.actor.init_state

    def _get_final_state(self) -> np.ndarray:
        if self.actor.final_state is None:
            raise ValueError(
                f"ValidatorActor '{self.actor_name}' final_state is not set."
            )
        return self.actor.final_state


def _resolve_env_prim_path(scene_asset, env_idx: int) -> str:
    """Resolve an asset's configured prim path for one environment."""
    prim_path = scene_asset.cfg.prim_path
    return re.sub(r"env_\.\*", f"env_{env_idx}", prim_path, count=1)


class ReachChecker(CheckerBase):
    """Check whether the target object is within reach of any EE link."""

    def __init__(
        self,
        actor_name: str,
        threshold: float = 0.05,
        robot_name: str = "",
        ee_links: Sequence[str] = (),
    ):
        if not robot_name:
            raise ValueError("robot_name is required for reach checker.")
        if not ee_links:
            raise ValueError("ee_links is required for reach checker.")

        self.actor_name = actor_name
        self.threshold = threshold
        self.robot_name = robot_name
        self.ee_links = tuple(ee_links)

    def check(self, env, env_idx: int = 0) -> bool:
        """Return True if any configured EE is closer than the threshold."""
        actor_pos = env.scene[self.actor_name].data.root_pos_w[env_idx]
        robot = env.scene[self.robot_name]
        body_positions = robot.data.body_com_pos_w[env_idx]
        ee_indices = []
        for ee_link_name in self.ee_links:
            body_ids, _ = robot.find_bodies(ee_link_name)
            if len(body_ids) == 0:
                raise ValueError(
                    f"EE link '{ee_link_name}' not found "
                    f"in robot '{self.robot_name}'."
                )
            ee_indices.extend(body_ids)

        for idx in ee_indices:
            dist = torch.norm(actor_pos - body_positions[idx])
            if dist < self.threshold:
                return True
        return False


class LiftChecker(ActorBoundChecker):
    """Check whether the target object is lifted above initial height."""

    def __init__(
        self,
        actor: ValidatorActor,
        threshold: float = 0.05,
    ):
        super().__init__(actor=actor)
        self.threshold = threshold

    def check(self, env, env_idx: int = 0) -> bool:
        """Return True if height gain exceeds the configured threshold."""
        actor_data = env.scene[self.actor_name].data
        actor_pos = actor_data.root_pos_w[env_idx]
        init_height = self._get_init_state()[env_idx, 2]
        return (actor_pos[2] - init_height).item() > self.threshold


class WithinXYChecker(CheckerBase):
    """Check whether actor1 center lies inside actor2 OBB on XY plane."""

    def __init__(
        self,
        actor1: str,
        actor2: str,
        robot_name: str = "",
        gripper_joints: Sequence[GripperRange] = (),
        require_gripper_open: bool = False,
        open_ratio: float = DEFAULT_GRIPPER_OPEN_RATIO,
    ):
        self.actor1 = actor1
        self.actor2 = actor2
        if require_gripper_open and not robot_name:
            raise ValueError("robot_name is required for within XY checker.")
        if require_gripper_open and not gripper_joints:
            raise ValueError(
                "gripper_joints is required for within XY checker."
            )

        self.gripper_checker = (
            BothGripperOpenChecker(
                robot_name=robot_name,
                gripper_joints=gripper_joints,
                open_ratio=open_ratio,
            )
            if require_gripper_open
            else None
        )

    def check(self, env, env_idx: int = 0) -> bool:
        """Run optional gripper gate, then evaluate XY containment."""
        if self.gripper_checker is not None and not self.gripper_checker(
            env, env_idx=env_idx
        ):
            return False

        from robo_orchard_sim.task_components.validators.utils import (
            is_object_center_in_obb,
        )

        actor1_pos = (
            env.scene[self.actor1].data.root_pos_w[env_idx].cpu().numpy()
        )
        actor2_pose_array = (
            env.scene[self.actor2].data.root_state_w[:, :7].cpu().numpy()
        )
        actor2_prim_path = _resolve_env_prim_path(
            env.scene[self.actor2], env_idx=env_idx
        )
        return is_object_center_in_obb(
            env.scene.stage,
            actor2_prim_path,
            actor2_pose_array,
            actor1_pos,
            idx_env=env_idx,
        )


class AlignmentXYChecker(CheckerBase):
    """Check whether two objects are aligned in XY within tolerance."""

    def __init__(
        self,
        actor1: str,
        actor2: str,
        robot_name: str = "",
        gripper_joints: Sequence[GripperRange] = (),
        eps: tuple[float, float] = (0.02, 0.02),
        require_gripper_open: bool = False,
        open_ratio: float = DEFAULT_GRIPPER_OPEN_RATIO,
    ):
        self.actor1 = actor1
        self.actor2 = actor2
        self.eps = np.array(eps)
        if require_gripper_open and not robot_name:
            raise ValueError(
                "robot_name is required for alignment XY checker."
            )
        if require_gripper_open and not gripper_joints:
            raise ValueError(
                "gripper_joints is required for alignment XY checker."
            )
        self.gripper_checker = (
            BothGripperOpenChecker(
                robot_name=robot_name,
                gripper_joints=gripper_joints,
                open_ratio=open_ratio,
            )
            if require_gripper_open
            else None
        )

    def check(self, env, env_idx: int = 0) -> bool:
        """Run optional gripper gate, then evaluate XY alignment."""
        if self.gripper_checker is not None and not self.gripper_checker(
            env, env_idx=env_idx
        ):
            return False

        actor1_pos = (
            env.scene[self.actor1].data.root_pos_w[env_idx].cpu().numpy()
        )
        actor2_pos = (
            env.scene[self.actor2].data.root_pos_w[env_idx].cpu().numpy()
        )
        return np.all(abs(actor1_pos[:2] - actor2_pos[:2]) < self.eps)


class AlignmentXYZChecker(CheckerBase):
    """Check whether actor2 matches actor1 with a target height offset.

    The checker treats actor1 as the XY reference and expects actor2 to be at
    ``(actor1.x, actor1.y, actor1.z + target_height_offset)`` within ``eps``.
    """

    def __init__(
        self,
        actor1: str,
        actor2: str,
        robot_name: str = "",
        gripper_joints: Sequence[GripperRange] = (),
        eps: tuple[float, float, float] = (0.025, 0.025, 0.0120),
        target_height_offset: float = 0.04,
        require_gripper_open: bool = False,
        open_ratio: float = DEFAULT_GRIPPER_OPEN_RATIO,
    ):
        self.actor1 = actor1
        self.actor2 = actor2
        self.eps = np.array(eps)
        self.target_height_offset = target_height_offset
        if require_gripper_open and not robot_name:
            raise ValueError(
                "robot_name is required for alignment XYZ checker."
            )
        if require_gripper_open and not gripper_joints:
            raise ValueError(
                "gripper_joints is required for alignment XYZ checker."
            )
        self.gripper_checker = (
            BothGripperOpenChecker(
                robot_name=robot_name,
                gripper_joints=gripper_joints,
                open_ratio=open_ratio,
            )
            if require_gripper_open
            else None
        )

    def check(self, env, env_idx: int = 0) -> bool:
        """Run optional gripper gate, then evaluate XYZ alignment."""
        if self.gripper_checker is not None and not self.gripper_checker(
            env, env_idx=env_idx
        ):
            return False

        actor1_pos = (
            env.scene[self.actor1].data.root_pos_w[env_idx].cpu().numpy()
        )
        actor2_pos = (
            env.scene[self.actor2].data.root_pos_w[env_idx].cpu().numpy()
        )
        target_pos = np.array(
            actor1_pos[:2].tolist()
            + [actor1_pos[2] + self.target_height_offset]
        )
        return np.all(abs(actor2_pos - target_pos) < self.eps)


class GripperOpenChecker(CheckerBase):
    """Check whether a single gripper joint is open enough."""

    def __init__(
        self,
        robot_name: str = "",
        gripper_joint: GripperRange | None = None,
        open_ratio: float = DEFAULT_GRIPPER_OPEN_RATIO,
    ):
        if not robot_name:
            raise ValueError("robot_name is required for gripper checker.")
        if gripper_joint is None:
            raise ValueError("gripper_joint is required for gripper checker.")
        self.robot_name = robot_name
        self.gripper_joint = gripper_joint
        self.open_ratio = open_ratio

    def check(self, env, env_idx: int = 0) -> bool:
        """Return True when the gripper joint has opened past the ratio."""
        robot = env.scene[self.robot_name]
        joint_ids, _ = robot.find_joints(self.gripper_joint.name)
        if len(joint_ids) == 0:
            raise ValueError(
                f"Gripper joint '{self.gripper_joint.name}' "
                f"not found in robot '{self.robot_name}'."
            )
        value = robot.data.joint_pos[env_idx][joint_ids[0]].item()
        return _gripper_joint_is_open(
            value, self.gripper_joint, self.open_ratio
        )


class BothGripperOpenChecker(CheckerBase):
    """Check whether all configured gripper joints are open enough."""

    def __init__(
        self,
        robot_name: str = "",
        gripper_joints: Sequence[GripperRange] = (),
        open_ratio: float = DEFAULT_GRIPPER_OPEN_RATIO,
    ):
        if not robot_name:
            raise ValueError(
                "robot_name is required for both gripper checker."
            )
        if len(gripper_joints) == 0:
            raise ValueError(
                "gripper_joints is required for both gripper checker."
            )
        self.robot_name = robot_name
        self.gripper_joints = tuple(gripper_joints)
        self.open_ratio = open_ratio

    def check(self, env, env_idx: int = 0) -> bool:
        """Return True only if every configured gripper joint is open."""
        robot = env.scene[self.robot_name]
        for gripper_range in self.gripper_joints:
            joint_ids, _ = robot.find_joints(gripper_range.name)
            if len(joint_ids) == 0:
                raise ValueError(
                    f"Gripper joint '{gripper_range.name}' "
                    f"not found in robot '{self.robot_name}'."
                )
            value = robot.data.joint_pos[env_idx][joint_ids[0]].item()
            if not _gripper_joint_is_open(
                value, gripper_range, self.open_ratio
            ):
                return False
        return True


def reach(
    actor_name,
    threshold=0.05,
    robot_name: str = "",
    ee_links: Sequence[str] = (),
):
    """Create a reach checker for object and robot identifiers."""
    return ReachChecker(
        actor_name=actor_name,
        threshold=threshold,
        robot_name=robot_name,
        ee_links=ee_links,
    )


def lift(
    actor: ValidatorActor,
    threshold: float = 0.05,
):
    """Create a lift checker bound to one validator actor."""
    return LiftChecker(
        actor=actor,
        threshold=threshold,
    )


def is_within_xy(
    actor1,
    actor2,
    require_gripper_open: bool = False,
    robot_name: str = "",
    gripper_joints: Sequence[GripperRange] = (),
    open_ratio: float = DEFAULT_GRIPPER_OPEN_RATIO,
):
    """Create an XY containment checker for object identifiers."""
    return WithinXYChecker(
        actor1=actor1,
        actor2=actor2,
        robot_name=robot_name,
        gripper_joints=gripper_joints,
        require_gripper_open=require_gripper_open,
        open_ratio=open_ratio,
    )


def is_alignment_xy(
    actor1,
    actor2,
    eps=(0.02, 0.02),
    require_gripper_open: bool = False,
    robot_name: str = "",
    gripper_joints: Sequence[GripperRange] = (),
    open_ratio: float = DEFAULT_GRIPPER_OPEN_RATIO,
):
    """Create an XY alignment checker for object identifiers."""
    return AlignmentXYChecker(
        actor1=actor1,
        actor2=actor2,
        robot_name=robot_name,
        gripper_joints=gripper_joints,
        eps=eps,
        require_gripper_open=require_gripper_open,
        open_ratio=open_ratio,
    )


def is_alignment_xyz(
    actor1,
    actor2,
    eps=(0.025, 0.025, 0.0120),
    target_height_offset: float = 0.04,
    require_gripper_open: bool = False,
    robot_name: str = "",
    gripper_joints: Sequence[GripperRange] = (),
    open_ratio: float = DEFAULT_GRIPPER_OPEN_RATIO,
):
    """Create an XYZ alignment checker with actor identifiers.

    ``target_height_offset`` is added to actor1's current Z to produce actor2's
    expected target height.
    """
    return AlignmentXYZChecker(
        actor1=actor1,
        actor2=actor2,
        robot_name=robot_name,
        gripper_joints=gripper_joints,
        eps=eps,
        target_height_offset=target_height_offset,
        require_gripper_open=require_gripper_open,
        open_ratio=open_ratio,
    )


def is_gripper_open(
    gripper_joint: GripperRange,
    robot_name: str = "",
    open_ratio: float = DEFAULT_GRIPPER_OPEN_RATIO,
):
    """Create a single-joint gripper-open checker."""
    return GripperOpenChecker(
        robot_name=robot_name,
        gripper_joint=gripper_joint,
        open_ratio=open_ratio,
    )


def is_both_gripper_open(
    gripper_joints: Sequence[GripperRange] = (),
    robot_name: str = "",
    open_ratio: float = DEFAULT_GRIPPER_OPEN_RATIO,
):
    """Create a checker for opening all gripper joints on one robot."""
    return BothGripperOpenChecker(
        robot_name=robot_name,
        gripper_joints=gripper_joints,
        open_ratio=open_ratio,
    )
