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

"""Submodule for articulation asset."""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import robo_orchard_core.utils.math as math_utils
import torch
from isaaclab.assets.rigid_object import RigidObject as _RigidObject

import robo_orchard_sim.utils.env_utils as env_utils
from robo_orchard_sim.cfg_wrappers.assets_cfg import (
    RigidObjectCfg as _RigidObjectCfg,
    SpawnerCfgType_co,
)
from robo_orchard_sim.cfg_wrappers.sim.spawners import (
    UsdFileCfg,
)
from robo_orchard_sim.utils.config import ClassType_co

__all__ = ["InteractivePoseData", "RigidObject", "RigidObjectCfg"]


@dataclass
class InteractivePoseData:
    """Public interactive pose output in world frame."""

    pos: torch.Tensor
    quat: torch.Tensor

    def get_axis(self, axis: Literal["x", "y", "z"]) -> torch.Tensor:
        """Return the specified local axis represented in world frame.

        Args:
            axis: Target axis in local frame.

        Returns:
            A tensor with shape (..., 3) that represents the selected axis in
            world coordinates.
        """
        if axis == "x":
            basis = torch.tensor([1.0, 0.0, 0.0], device=self.pos.device)
        elif axis == "y":
            basis = torch.tensor([0.0, 1.0, 0.0], device=self.pos.device)
        elif axis == "z":
            basis = torch.tensor([0.0, 0.0, 1.0], device=self.pos.device)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be x, y, or z.")
        basis = basis.to(self.quat.dtype)
        return math_utils.quaternion_apply_point(self.quat, basis)

    @property
    def xyz_w(self) -> torch.Tensor:
        """Backward-compatible alias for world position."""
        return self.pos

    @property
    def quat_w(self) -> torch.Tensor:
        """Backward-compatible alias for world quaternion."""
        return self.quat

    @property
    def direction_w(self) -> torch.Tensor:
        """Backward-compatible alias for world x-axis direction."""
        return self.get_axis("x")


@dataclass(frozen=True)
class _InteractivePoseKey:
    """Internal metadata for an annotated interaction pose."""

    part: str
    pose_id: int


@dataclass
class _InteractivePoseBatch:
    """Internal tensorized interaction pose data in object local frame."""

    xyz: torch.Tensor
    direction: torch.Tensor
    ref_frame: torch.Tensor
    keys: list[_InteractivePoseKey]


class RigidObject(_RigidObject):
    """Extended RigidObject class for RoboOrchard.

    This class inherits from isaaclab's RigidObject and provides additional
    functionalities for handling interactive object elements and their
    poses in the simulation.

    """

    def __init__(self, cfg: RigidObjectCfg):
        super().__init__(cfg)
        self._element_path = cfg.object_elements_path
        self._interactive_pose: dict = {}

        # get obj scale from cfg
        self.scale = (1.0, 1.0, 1.0)
        if hasattr(self.cfg, "spawn"):
            if isinstance(self.cfg.spawn, UsdFileCfg):
                self.scale = self.cfg.spawn.scale

        if self._element_path is not None:
            self._interactive_pose = self._read_interactive_pose_from_json(
                self._element_path
            )

    def get_element_pose(
        self,
        mode: Literal["active", "passive"],
        action: str,
        part: str | list[str],
        id: list[list[int]] | None = None,
    ) -> InteractivePoseData:
        """Get the pose information of a specified part during interaction.

        Args:
            mode (Literal["active", "passive"]): Interaction mode, either
            "active" or "passive".
            action (str): Type of interaction action(like "pick", "place"...)
            part: A list of part names to retrieve poses for.
            id: An optional list of lists of indices to select specific poses.
                Its length must match the length of `part`. Defaults to None.

        Returns:
            An `InteractivePoseData` object in world frame. The output tensors
            have shape (num_envs, num_selected_poses, ...).
        """
        local_batch = self._prepare_interactive_pose_batch(
            mode=mode,
            action=action,
            part=part,
            id=id,
        )
        return self._to_interactive_pose_data(
            local_batch=local_batch,
            root_pos_w=self.data.root_pos_w,
            root_quat_w=self.data.root_quat_w,
        )

    def _prepare_interactive_pose_batch(
        self,
        mode: Literal["active", "passive"],
        action: str,
        part: str | list[str],
        id: list[list[int]] | None,
    ) -> _InteractivePoseBatch:
        """Load and tensorize annotated interactive poses from JSON config."""
        if self._element_path is None:
            raise ValueError(
                "object elements path is None, please set it in the config"
            )

        if isinstance(part, str):
            part = [part]

        if id is not None and len(id) != len(part):
            raise ValueError(
                f"Part count ({len(part)}) does not match "
                f"id count ({len(id)})."
            )

        device = self.data.root_pos_w.device
        dtype = self.data.root_pos_w.dtype
        num_envs = self.data.root_pos_w.shape[0]

        interaction = self._interactive_pose.get("interaction", {})
        mode_data = interaction.get(mode, {})
        action_data = mode_data.get(action, {})

        xyz_list: list[torch.Tensor] = []
        direction_list: list[torch.Tensor] = []
        ref_list: list[torch.Tensor] = []
        keys: list[_InteractivePoseKey] = []

        scale_tensor = torch.tensor(self.scale, device=device, dtype=dtype)
        for part_idx, part_name in enumerate(part):
            if part_name not in action_data:
                raise ValueError(
                    f"Part {part_name} not found in interactive pose data."
                )

            part_poses: list[dict[str, Any]] = action_data[part_name]
            selected_ids = (
                list(range(len(part_poses))) if id is None else id[part_idx]
            )

            for pose_id, pose in enumerate(part_poses):
                if pose_id not in selected_ids:
                    continue

                xyz = (
                    torch.tensor(pose["xyz"], device=device, dtype=dtype)
                    * scale_tensor
                ).repeat(num_envs, 1)
                direction = (
                    torch.tensor(pose["direction"], device=device, dtype=dtype)
                    * scale_tensor
                ).repeat(num_envs, 1)
                ref_frame = (
                    torch.tensor(pose["ref_frame"], device=device, dtype=dtype)
                    * scale_tensor
                ).repeat(num_envs, 1)

                xyz_list.append(xyz)
                direction_list.append(direction)
                ref_list.append(ref_frame)
                keys.append(
                    _InteractivePoseKey(part=part_name, pose_id=pose_id)
                )

        if not xyz_list:
            raise ValueError(
                f"No interactive poses were selected for "
                f"mode={mode}, action={action}, part={part}."
            )

        return _InteractivePoseBatch(
            xyz=torch.stack(xyz_list, dim=1),
            direction=torch.stack(direction_list, dim=1),
            ref_frame=torch.stack(ref_list, dim=1),
            keys=keys,
        )

    def _to_interactive_pose_data(
        self,
        local_batch: _InteractivePoseBatch,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> InteractivePoseData:
        """Convert local annotated poses to public world-frame output."""
        num_envs, num_pose, _ = local_batch.xyz.shape
        xyz = local_batch.xyz.view(-1, 3)
        direction = math_utils.normalize(local_batch.direction.view(-1, 3))
        ref_frame = math_utils.normalize(local_batch.ref_frame.view(-1, 3))

        root_pos_w = torch.repeat_interleave(root_pos_w, num_pose, dim=0)
        root_quat_w = torch.repeat_interleave(root_quat_w, num_pose, dim=0)
        local_quat = env_utils.constract_quat_with_vec(direction, ref_frame)

        pos_w = (
            math_utils.quaternion_apply_point(root_quat_w, xyz) + root_pos_w
        )
        quat_w = math_utils.quaternion_multiply(root_quat_w, local_quat)
        return InteractivePoseData(
            pos=pos_w.view(num_envs, num_pose, 3),
            quat=quat_w.view(num_envs, num_pose, 4),
        )

    def _read_interactive_pose_from_json(self, file_path: str) -> dict:
        """Read the interactive pose from json file."""
        if not file_path or not os.path.isfile(file_path):
            print("file path is None")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _align_quat_x_to_vector(
        self, v: torch.Tensor, angle: float = 0.0
    ) -> torch.Tensor:
        """Align the X axis of a quaternion to a given vector.

        Args:
            v: (..., 3) shape tensor, arbitrary batch of vectors
            angle: rotation angle (in radians) around the aligned
            X axis, for controlling y/z direction
        Returns:
            q: (..., 4) shape unit quaternion, such that
            q*(1,0,0)*q^{-1} = v/|v|,
            and then rotated by angle around new X axis
        """

        # 1. Normalize the target vector
        v_norm = math_utils.normalize(v)

        # 2. Compute rotation axis u = e_x × v_norm
        ex = torch.tensor([1.0, 0.0, 0.0], device=v.device, dtype=v.dtype)

        axis = torch.cross(ex.expand_as(v_norm), v_norm, dim=-1)
        axis = math_utils.normalize(axis, dim=-1)  # (..., 3)

        # 3. Compute rotation angle θ = arccos(e_x · v_norm)
        cos_theta = torch.clamp(
            torch.sum(ex * v_norm, dim=-1, keepdim=True), -1.0, 1.0
        )
        theta = torch.acos(cos_theta)  # (..., 1)

        # 4. Construct quaternion for X alignment
        half_theta = theta * 0.5
        w = torch.cos(half_theta)  # (..., 1)
        sin_ht = torch.sin(half_theta)  # (..., 1)
        xyz = axis * sin_ht  # (..., 3)

        q = torch.cat([w, xyz], dim=-1)  # (..., 4)
        q_align = math_utils.normalize(q)

        if angle == 0.0:
            return q_align

        half_angle = angle * 0.5
        q_roll = torch.zeros_like(q_align)
        q_roll[..., 0] = torch.cos(torch.tensor(half_angle))
        q_roll[..., 1] = torch.sin(torch.tensor(half_angle))
        # q_roll[..., 2] = 0
        # q_roll[..., 3] = 0

        # 6. Compose the two rotations: first align, then roll
        q_final = math_utils.quaternion_multiply(q_align, q_roll)

        return math_utils.normalize(q_final)


class RigidObjectCfg(_RigidObjectCfg[SpawnerCfgType_co, RigidObject]):
    """Configuration class template for all rigid objects in RoboOrchard."""

    class_type: ClassType_co[RigidObject] = RigidObject

    object_elements_path: str | None = None
    """Object interactive pose file path."""
