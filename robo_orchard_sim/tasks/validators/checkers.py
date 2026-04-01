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
from typing import Literal, Sequence

import numpy as np
import torch


class CheckerBase:
    """Base checker with a unified boolean check interface."""

    def check(self, env, env_idx: int = 0) -> bool:
        """Run checker logic and return whether the condition is met."""
        raise NotImplementedError

    def __call__(self, env, env_idx: int = 0) -> bool:
        """Allow checker instances to be called like functions."""
        return self.check(env, env_idx=env_idx)


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


class LiftChecker(CheckerBase):
    """Check whether the target object is lifted above default height."""

    def __init__(
        self,
        actor_name: str,
        threshold: float = 0.05,
        default_height: float | None = None,
    ):
        self.actor_name = actor_name
        self.threshold = threshold
        self.default_height = default_height

    def check(self, env, env_idx: int = 0) -> bool:
        """Return True if height gain exceeds the configured threshold."""
        actor_data = env.scene[self.actor_name].data
        actor_pos = actor_data.root_pos_w[env_idx]
        default_height = self.default_height
        if default_height is None:
            default_height = actor_data.default_root_state[env_idx, 2]
        return (actor_pos[2] - default_height).item() > self.threshold


class WithinXYChecker(CheckerBase):
    """Check whether actor1 center lies inside actor2 OBB on XY plane."""

    def __init__(
        self,
        actor1: str,
        actor2: str,
        robot_name: str = "",
        gripper_links: Sequence[str] = (),
        open_gripper_threshold: float | None = None,
    ):
        self.actor1 = actor1
        self.actor2 = actor2
        if open_gripper_threshold is not None and not robot_name:
            raise ValueError("robot_name is required for within XY checker.")
        if open_gripper_threshold is not None and not gripper_links:
            raise ValueError(
                "gripper_links is required for within XY checker."
            )

        self.gripper_checker = (
            BothGripperOpenChecker(
                robot_name=robot_name,
                gripper_links=gripper_links,
                open_gripper_threshold=open_gripper_threshold,
            )
            if open_gripper_threshold is not None
            else None
        )

    def check(self, env, env_idx: int = 0) -> bool:
        """Run optional gripper gate, then evaluate XY containment."""
        if self.gripper_checker is not None and not self.gripper_checker(
            env, env_idx=env_idx
        ):
            return False

        from robo_orchard_sim.tasks.validators.utils import (
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
        gripper_links: Sequence[str] = (),
        eps: tuple[float, float] = (0.02, 0.02),
        open_gripper_threshold: float | None = None,
    ):
        self.actor1 = actor1
        self.actor2 = actor2
        self.eps = np.array(eps)
        if open_gripper_threshold is not None and not robot_name:
            raise ValueError(
                "robot_name is required for alignment XY checker."
            )
        if open_gripper_threshold is not None and not gripper_links:
            raise ValueError(
                "gripper_links is required for alignment XY checker."
            )
        self.gripper_checker = (
            BothGripperOpenChecker(
                robot_name=robot_name,
                gripper_links=gripper_links,
                open_gripper_threshold=open_gripper_threshold,
            )
            if open_gripper_threshold is not None
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
        gripper_links: Sequence[str] = (),
        eps: tuple[float, float, float] = (0.025, 0.025, 0.0120),
        target_height_offset: float = 0.04,
        open_gripper_threshold: float | None = None,
    ):
        self.actor1 = actor1
        self.actor2 = actor2
        self.eps = np.array(eps)
        self.target_height_offset = target_height_offset
        if open_gripper_threshold is not None and not robot_name:
            raise ValueError(
                "robot_name is required for alignment XYZ checker."
            )
        if open_gripper_threshold is not None and not gripper_links:
            raise ValueError(
                "gripper_links is required for alignment XYZ checker."
            )
        self.gripper_checker = (
            BothGripperOpenChecker(
                robot_name=robot_name,
                gripper_links=gripper_links,
                open_gripper_threshold=open_gripper_threshold,
            )
            if open_gripper_threshold is not None
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
        gripper_link: str = "",
        open_gripper_threshold: float = 0.04,
    ):
        if not robot_name:
            raise ValueError("robot_name is required for gripper checker.")
        if not gripper_link:
            raise ValueError("gripper_link is required for gripper checker.")
        self.robot_name = robot_name
        self.gripper_link = gripper_link
        self.open_gripper_threshold = open_gripper_threshold

    def check(self, env, env_idx: int = 0) -> bool:
        """Return True when gripper joint position reaches threshold."""
        robot = env.scene[self.robot_name]
        joint_ids, _ = robot.find_joints(self.gripper_link)
        if len(joint_ids) == 0:
            raise ValueError(
                f"Gripper link '{self.gripper_link}' "
                f"not found in robot '{self.robot_name}'."
            )
        gripper_value = robot.data.joint_pos[env_idx][joint_ids[0]].item()
        return bool(gripper_value >= self.open_gripper_threshold)


class BothGripperOpenChecker(CheckerBase):
    """Check whether all configured gripper joints are open enough."""

    def __init__(
        self,
        robot_name: str = "",
        gripper_links: Sequence[str] = (),
        open_gripper_threshold: float = 0.04,
    ):
        if not robot_name:
            raise ValueError(
                "robot_name is required for both gripper checker."
            )
        if len(gripper_links) == 0:
            raise ValueError(
                "gripper_links is required for both gripper checker."
            )
        self.robot_name = robot_name
        self.gripper_links = tuple(gripper_links)
        self.open_gripper_threshold = open_gripper_threshold

    def check(self, env, env_idx: int = 0) -> bool:
        """Return True only if every configured gripper joint is open."""
        robot = env.scene[self.robot_name]
        for gripper_link in self.gripper_links:
            joint_ids, _ = robot.find_joints(gripper_link)
            if len(joint_ids) == 0:
                raise ValueError(
                    f"Gripper link '{gripper_link}' "
                    f"not found in robot '{self.robot_name}'."
                )
            gripper_value = robot.data.joint_pos[env_idx][joint_ids[0]].item()
            if gripper_value < self.open_gripper_threshold:
                return False
        return True


def reach(
    actor_name,
    threshold=0.05,
    robot_name: str = "robots/dualarm_piper",
    ee_links: Sequence[str] = ("left_link6", "right_link6"),
):
    """Create a reach checker for object and robot identifiers."""
    return ReachChecker(
        actor_name=actor_name,
        threshold=threshold,
        robot_name=robot_name,
        ee_links=ee_links,
    )


def lift(actor_name, threshold=0.05, default_height=None):
    """Create a lift checker for one object identifier."""
    return LiftChecker(
        actor_name=actor_name,
        threshold=threshold,
        default_height=default_height,
    )


def is_within_xy(
    actor1,
    actor2,
    open_gripper_threshold=None,
    robot_name: str = "robots/dualarm_piper",
    gripper_links: Sequence[str] = ("left_joint7", "right_joint7"),
):
    """Create an XY containment checker for object identifiers."""
    return WithinXYChecker(
        actor1=actor1,
        actor2=actor2,
        robot_name=robot_name,
        gripper_links=gripper_links,
        open_gripper_threshold=open_gripper_threshold,
    )


def is_alignment_xy(
    actor1,
    actor2,
    eps=(0.02, 0.02),
    open_gripper_threshold=None,
    robot_name: str = "robots/dualarm_piper",
    gripper_links: Sequence[str] = ("left_joint7", "right_joint7"),
):
    """Create an XY alignment checker for object identifiers."""
    return AlignmentXYChecker(
        actor1=actor1,
        actor2=actor2,
        robot_name=robot_name,
        gripper_links=gripper_links,
        eps=eps,
        open_gripper_threshold=open_gripper_threshold,
    )


def is_alignment_xyz(
    actor1,
    actor2,
    eps=(0.025, 0.025, 0.0120),
    target_height_offset: float = 0.04,
    open_gripper_threshold=None,
    robot_name: str = "robots/dualarm_piper",
    gripper_links: Sequence[str] = ("left_joint7", "right_joint7"),
):
    """Create an XYZ alignment checker with actor identifiers.

    ``target_height_offset`` is added to actor1's current Z to produce actor2's
    expected target height.
    """
    return AlignmentXYZChecker(
        actor1=actor1,
        actor2=actor2,
        robot_name=robot_name,
        gripper_links=gripper_links,
        eps=eps,
        target_height_offset=target_height_offset,
        open_gripper_threshold=open_gripper_threshold,
    )


def is_gripper_open(
    arm: Literal["left", "right"],
    open_gripper_threshold: float,
    robot_name: str = "robots/dualarm_piper",
    gripper_link: str | None = None,
):
    """Create a single-arm gripper-open checker.

    The gripper joint name is inferred from `arm` if not provided.
    """
    resolved_gripper_link = gripper_link or (
        "left_joint7" if arm == "left" else "right_joint7"
    )
    return GripperOpenChecker(
        robot_name=robot_name,
        gripper_link=resolved_gripper_link,
        open_gripper_threshold=open_gripper_threshold,
    )


def is_both_gripper_open(
    open_gripper_threshold: float,
    robot_name: str = "robots/dualarm_piper",
    gripper_links: Sequence[str] = ("left_joint7", "right_joint7"),
):
    """Create a checker for opening both grippers on one robot identifier."""
    return BothGripperOpenChecker(
        robot_name=robot_name,
        gripper_links=gripper_links,
        open_gripper_threshold=open_gripper_threshold,
    )
