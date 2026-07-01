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

"""Shared gripper policy encoding helpers for policy adapters."""

from __future__ import annotations
from typing import Literal

import numpy as np
import torch

GripperPolicyRepresentation = Literal["first_joint", "all_joints"]
GripperDecodeCoupling = Literal["symmetric", "mirrored", "identity"]


def gripper_policy_dim(
    *,
    joint_count: int,
    representation: GripperPolicyRepresentation,
) -> int:
    """Return how many policy dimensions represent the gripper."""
    _validate_joint_count(joint_count)
    if representation == "first_joint":
        return 1
    if representation == "all_joints":
        return joint_count
    raise ValueError(
        f"Unsupported gripper policy representation {representation!r}."
    )


def gripper_positions_to_policy_torch(
    gripper_position: torch.Tensor,
    *,
    gripper_policy_representation: GripperPolicyRepresentation,
    gripper_policy_scale: float,
) -> torch.Tensor:
    """Encode physical gripper joint positions for policy consumption."""
    _validate_policy_scale(gripper_policy_scale)
    gripper_policy_dim(
        joint_count=gripper_position.shape[-1],
        representation=gripper_policy_representation,
    )
    if gripper_policy_representation == "first_joint":
        return gripper_position[..., :1] * gripper_policy_scale
    return gripper_position * gripper_policy_scale


def gripper_positions_to_policy_numpy(
    gripper_position: np.ndarray,
    *,
    gripper_policy_representation: GripperPolicyRepresentation,
    gripper_policy_scale: float,
) -> np.ndarray:
    """Encode physical gripper joint positions for policy consumption."""
    if gripper_position.ndim != 1:
        raise ValueError(
            "Policy adapters expect one gripper position vector, got "
            f"shape {gripper_position.shape}."
        )
    _validate_policy_scale(gripper_policy_scale)
    gripper_policy_dim(
        joint_count=gripper_position.size,
        representation=gripper_policy_representation,
    )
    if gripper_policy_representation == "first_joint":
        selected = gripper_position[:1]
    else:
        selected = gripper_position
    return np.asarray(selected * gripper_policy_scale, dtype=np.float32)


def policy_to_gripper_positions_torch(
    gripper_policy: torch.Tensor,
    *,
    gripper_policy_representation: GripperPolicyRepresentation,
    gripper_decode_coupling: GripperDecodeCoupling,
    gripper_policy_scale: float,
    joint_count: int,
) -> torch.Tensor:
    """Decode one policy gripper action into physical joint controls."""
    _validate_policy_scale(gripper_policy_scale)
    expected_dim = gripper_policy_dim(
        joint_count=joint_count,
        representation=gripper_policy_representation,
    )
    physical = gripper_policy.reshape(-1) / gripper_policy_scale
    if physical.numel() != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} gripper policy values, "
            f"got {physical.numel()}."
        )
    if gripper_policy_representation == "all_joints":
        if gripper_decode_coupling != "identity":
            raise ValueError(
                "all_joints gripper policy representation requires "
                "identity decode coupling."
            )
        return physical.reshape(1, joint_count)

    value = physical[0]
    if gripper_decode_coupling == "identity":
        if joint_count != 1:
            raise ValueError(
                "identity decode coupling with first_joint representation "
                "requires exactly one gripper joint."
            )
        return value.reshape(1, 1)
    if gripper_decode_coupling == "symmetric":
        return value.repeat(joint_count).reshape(1, joint_count)
    if gripper_decode_coupling == "mirrored":
        values = torch.cat(
            (
                value.reshape(1),
                -value.repeat(joint_count - 1),
            )
        )
        return values.reshape(1, joint_count)
    raise ValueError(
        f"Unsupported gripper decode coupling {gripper_decode_coupling!r}."
    )


def _validate_joint_count(joint_count: int) -> None:
    if joint_count < 1:
        raise ValueError(
            "Policy gripper encoding requires at least one gripper joint, "
            f"got {joint_count}."
        )


def _validate_policy_scale(scale: float) -> None:
    if scale == 0.0:
        raise ValueError("gripper_policy_scale must be nonzero.")
