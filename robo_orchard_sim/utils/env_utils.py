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

import math
import random
import warnings
from typing import List, Literal

import robo_orchard_core.utils.math as math_utils
import torch
from scipy.spatial.transform import Rotation as R


def sample_poses(
    pose_range_: dict[str, tuple[float, float]],
    mode: Literal["scattered", "orderly", "stacked"],
    len: int,
) -> List[List[float]]:
    """Generate a list of object poses in three modes.

    Generate a specified number of object poses based on the given pose range
    and mode. Supports three modes:
    - "scattered": randomly distributed
    - "orderly": arranged in order
    - "stacked": stacked

    Args:
        pose_range_ (dict[str, tuple[float, float]]): A dictionary of pose
            ranges, including the ranges for "x", "y", "z", "roll",
            "pitch", and "yaw".
        mode (Literal["scattered", "orderly", "stacked"]): The mode used
            to generate poses.
        len (int): The number of poses to generate.

    Returns:
        List[List[float]]: A list of generated poses, where each pose includes
            the position (x, y, z) and the quaternion (qw, qx, qy, qz).
    """
    range_list = [
        pose_range_.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    pose_range = dict(
        zip(["x", "y", "z", "roll", "pitch", "yaw"], range_list, strict=False)
    )

    poses = []
    if mode == "scattered":
        for _ in range(len):
            x = random.uniform(*pose_range["x"])
            y = random.uniform(*pose_range["y"])
            z = random.uniform(*pose_range["z"])
            roll = random.uniform(*pose_range["roll"])
            pitch = random.uniform(*pose_range["pitch"])
            yaw = random.uniform(*pose_range["yaw"])
            quat = R.from_euler("xyz", [roll, pitch, yaw]).as_quat(
                scalar_first=True
            )
            poses.append([x, y, z, *quat])
    elif mode == "orderly":
        col = round(math.sqrt(len))
        row = col if len <= col**2 else col + 1
        x_step = (pose_range["x"][1] - pose_range["x"][0]) / (row - 1)
        y_step = (pose_range["y"][1] - pose_range["y"][0]) / (col - 1)
        for i in range(len):
            x = pose_range["x"][0] + (i // col) * x_step
            y = pose_range["y"][0] + (i % col) * y_step
            z = random.uniform(*pose_range["z"])
            roll = random.uniform(*pose_range["roll"])
            pitch = random.uniform(*pose_range["pitch"])
            yaw = random.uniform(*pose_range["yaw"])
            quat = R.from_euler("xyz", [roll, pitch, yaw]).as_quat(
                scalar_first=True
            )
            poses.append([x, y, z, *quat])
    elif mode == "stacked":
        for i in range(len):
            x_mid = (
                pose_range["x"][0]
                + (pose_range["x"][1] - pose_range["x"][0]) / 2.0
            )
            y_mid = (
                pose_range["y"][0]
                + (pose_range["y"][1] - pose_range["y"][0]) / 2.0
            )
            x = random.uniform(x_mid - 0.02, x_mid + 0.02)
            y = random.uniform(y_mid - 0.02, y_mid + 0.02)
            z = pose_range["z"][0] + i * (
                pose_range["z"][1] - pose_range["z"][0]
            ) / (len - 1)
            roll = random.uniform(*pose_range["roll"])
            pitch = random.uniform(*pose_range["pitch"])
            yaw = random.uniform(*pose_range["yaw"])
            quat = R.from_euler("xyz", [roll, pitch, yaw]).as_quat(
                scalar_first=True
            )
            poses.append([x, y, z, *quat])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return poses


def _generate_perpendicular_vector(
    v1: torch.Tensor, tol: float = 1e-8
) -> torch.Tensor:
    """Gen perpendicular vector automatically."""

    if torch.any(torch.norm(v1, dim=-1) < tol):
        raise ValueError(
            "Input vector is approximately zero, "
            "cannot generate perpendicular vector"
        )

    v1_norm = math_utils.normalize(v1)

    # Find the component with the smallest absolute value in v1
    abs_components = torch.abs(v1_norm)
    min_component_idx = torch.argmin(abs_components, dim=-1)

    batch_size = v1.shape[0] if v1.dim() > 1 else 1
    candidate = torch.zeros_like(v1)

    if v1.dim() > 1:
        candidate[torch.arange(batch_size), min_component_idx] = 1.0
    else:
        candidate[min_component_idx] = 1.0

    perpendicular = torch.cross(v1_norm, candidate, dim=-1)

    return math_utils.normalize(perpendicular)


def constract_quat_with_vec(
    v1: torch.Tensor,
    v2: torch.Tensor | None = None,
    cmd: str = "xy",
    tol: float = 1e-8,
    perpendicular_tol: float = 1e-2,
) -> torch.Tensor:
    """Construct a quaternion from two vectors.

    Args:
        v1: (3,) or (N,3) tensor, first direction vector
        v2: (3,) or (N,3) tensor, second direction vector (not parallel to v1).
            If None, a perpendicular vector will be automatically generated.
            For accurate quaternion construction, v2 should be provided and
            preferably perpendicular to v1. If not perpendicular, the function
            will use Gram-Schmidt orthogonalization to construct orthogonal
            vectors
        cmd: str, two characters from 'x', 'y', 'z', e.g. "xy", "yz", "zx".
            The first char is the axis for v1, the second for v2.

    Returns:
        quat: (4,) or (N,4) quaternion, in wxyz order
    """
    if not (len(cmd) == 2 and all(c in "xyz" for c in cmd)):
        raise ValueError("cmd must be two of 'x','y','z'")

    v2_was_none = v2 is None
    if v2 is None:
        v2 = _generate_perpendicular_vector(v1, tol=tol)

    axes = {}

    # check if input vectors validation
    if torch.any(torch.norm(v1, dim=1) < tol):
        raise ValueError(
            "v1 is approximately a zero vector (||v1|| < tol). ",
            "Cannot construct an orthonormal basis",
        )

    if torch.any(torch.norm(v2, dim=1) < tol):
        raise ValueError(
            "v2 is approximately a zero vector (||v2|| < tol). ",
            "Cannot construct an orthonormal basis",
        )

    # Normalize input vectors
    v1_norm = math_utils.normalize(v1)
    v2_norm = math_utils.normalize(v2)

    cross12 = torch.cross(v1_norm, v2_norm, dim=-1)
    # norm_cross12 = math_utils.normalize(cross12)
    if torch.any(torch.norm(cross12, dim=1) < tol):
        raise ValueError(
            "v1 and v2 are approximately colinear. ",
            "Cannot construct the second orthogonal direction.",
        )

    # Check if vectors are perpendicular and issue warning if not
    if not v2_was_none:  # Only check if v2 was provided by user
        dot_product = (v1_norm * v2_norm).sum(-1)
        abs_dot_product = torch.abs(dot_product)

        if torch.any(abs_dot_product > perpendicular_tol):
            # Find the maximum deviation from perpendicularity
            dot_product_max = dot_product[torch.argmax(abs_dot_product)]
            actual_angle_rad = torch.acos(
                torch.clamp(dot_product_max, -1.0, 1.0)
            )
            actual_angle_deg = torch.rad2deg(actual_angle_rad).item()

            warnings.warn(
                f"Input vectors v1 and v2 are not perpendicular. "
                f"Actual angle between vectors: {actual_angle_deg:.2f}°. "
                f"Gram-Schmidt orthogonalization will be applied to ensure "
                "orthogonality.",
                UserWarning,
                stacklevel=2,
            )

    v2_proj = v2_norm - (v2_norm * v1_norm).sum(-1, keepdim=True) * v1_norm
    v2_norm = math_utils.normalize(v2_proj)

    # Assign axes according to cmd
    axes[cmd[0]] = v1_norm
    axes[cmd[1]] = v2_norm
    # The remaining axis
    remaining_axis = ({"x", "y", "z"} - set(cmd)).pop()

    if "x" not in axes:
        v3_candidate = torch.cross(axes["y"], axes["z"], dim=-1)
        axes[remaining_axis] = math_utils.normalize(v3_candidate)
    elif "y" not in axes:
        v3_candidate = torch.cross(axes["z"], axes["x"], dim=-1)
        axes[remaining_axis] = math_utils.normalize(v3_candidate)
    elif "z" not in axes:
        v3_candidate = torch.cross(axes["x"], axes["y"], dim=-1)
        axes[remaining_axis] = math_utils.normalize(v3_candidate)

    # Stack axes in x, y, z order to form rotation matrix
    rotmat = torch.stack(
        [axes["x"], axes["y"], axes["z"]], dim=-1
    )  # (..., 3, 3)
    quat = math_utils.matrix_to_quaternion(rotmat)  # Should return wxyz order
    return quat
