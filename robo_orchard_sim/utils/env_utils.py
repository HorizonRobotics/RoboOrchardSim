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
from typing import Dict, List, Literal, Tuple

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


class PoseAugmentor:
    """Simplified Pose Augmentor for multi-axis rotation combinations.

    Based on the single-axis augment_grasp_poses function
    """

    @staticmethod
    def augment_single_axis(
        grasp_poses: torch.Tensor,
        axis: str,
        ang_range: Tuple[float, float],
        num_poses: int,
    ) -> torch.Tensor:
        """Single axis augmentation (wrapper of the original function).

        Args:
            grasp_poses: Input poses [B, 7] - (x, y, z, w, qx, qy, qz)
            axis: Rotation axis ('x', 'y', or 'z')
            ang_range: Angle range (min_deg, max_deg)
            num_poses: Number of poses to generate

        Returns:
            Augmented poses [B, num_poses, 7]
        """
        if grasp_poses.dim() != 2 or grasp_poses.shape[1] != 7:
            raise ValueError("The shape of input grasp_poses must be [B, 7]")
        if axis not in ["x", "y", "z"]:
            raise ValueError("Parameter axis must be one of 'x', 'y', or 'z'")
        if not isinstance(ang_range, (tuple, list)) or len(ang_range) != 2:
            raise ValueError(
                "Parameter ang_range must be a tuple or list containing two "
                "elements"
            )
        if num_poses < 1:
            raise ValueError(
                "Parameter num_poses must be greater than or equal to 1"
            )

        device = grasp_poses.device

        # 1. Separate position and original quaternion
        positions = grasp_poses[:, :3]  # Shape: [B, 3]
        original_quats = grasp_poses[:, 3:]  # Shape: [B, 4]

        # 2. Equally space num_poses angles within the [ang_min, ang_max]
        ang_min, ang_max = ang_range
        # Generate angles using torch.linspace and convert to radians
        angles_rad = torch.deg2rad(
            torch.linspace(ang_min, ang_max, num_poses, device=device)
        )

        # 3. Construct rotation quaternions based on the specified axis
        half_angles = angles_rad / 2.0
        cos_half = torch.cos(half_angles)
        sin_half = torch.sin(half_angles)

        # Initialize rotation quaternion tensor
        rot_quats = torch.zeros(num_poses, 4, device=device)
        rot_quats[:, 0] = cos_half

        # Fill rotation components based on axis
        if axis == "x":
            rot_quats[:, 1] = sin_half
        elif axis == "y":
            rot_quats[:, 2] = sin_half
        else:  # axis == 'z'
            rot_quats[:, 3] = sin_half

        # 4. Prepare and execute batch quaternion multiplication
        # Expand original quaternions and rotation quaternions for broadcasting
        # original_quats: [B, 4] -> [B, 1, 4]
        # rot_quats:      [N, 4] -> [1, N, 4] (N=num_poses)
        expanded_orig_quats = original_quats.unsqueeze(1)

        # Perform multiplication: q_new = q_orig * q_rot
        # Broadcasting mechanism will automatically handle: [B, N, 4]
        new_quats = math_utils.quaternion_multiply(
            expanded_orig_quats, rot_quats.unsqueeze(0)
        )

        # Normalize quaternions to avoid accumulated errors
        new_quats = new_quats / torch.linalg.norm(
            new_quats, dim=-1, keepdim=True
        )

        # 5. Combine new poses
        # Expand position information to match new quaternions: [B, N, 3]
        expanded_positions = positions.unsqueeze(1).expand(-1, num_poses, -1)

        augmented_poses = torch.cat((expanded_positions, new_quats), dim=-1)

        return augmented_poses

    @staticmethod
    def augment_multi_axis(
        grasp_poses: torch.Tensor,
        rotation_config: Dict[str, Tuple[float, float, int]],
    ) -> torch.Tensor:
        """Multi-axis rotation augmentation.

        Args:
            grasp_poses: Input poses [B, 7] - (x, y, z, w, qx, qy, qz)
            rotation_config: Dict with axis config, e.g.:
                {'x': (min_deg, max_deg, num_poses), 'z': (0, 360, 12)}

        Returns:
            Augmented poses [B, N, 7] where N is the product of all num_poses
        """
        if grasp_poses.dim() != 2 or grasp_poses.shape[1] != 7:
            raise ValueError("Input grasp_poses must have shape [B, 7]")

        # Generate all angle combinations
        axis_angles = {}
        for axis, (min_deg, max_deg, num_poses) in rotation_config.items():
            if axis not in ["x", "y", "z"]:
                raise ValueError(f"Invalid axis: {axis}")
            angles = torch.linspace(
                min_deg, max_deg, num_poses, device=grasp_poses.device
            )
            axis_angles[axis] = angles

        # Create all combinations
        axes = ["x", "y", "z"]
        angle_lists = []
        for axis in axes:
            if axis in axis_angles:
                angle_lists.append(axis_angles[axis])
            else:
                # No rotation for this axis
                angle_lists.append(
                    torch.tensor([0.0], device=grasp_poses.device)
                )

        # Generate all combinations using meshgrid
        angle_combinations = torch.meshgrid(*angle_lists, indexing="ij")
        x_angles = angle_combinations[0].flatten()  # All x angles
        y_angles = angle_combinations[1].flatten()  # All y angles
        z_angles = angle_combinations[2].flatten()  # All z angles

        # Apply all rotation combinations
        result_poses = PoseAugmentor._apply_rotation_combinations(
            grasp_poses, x_angles, y_angles, z_angles
        )

        return result_poses

    @staticmethod
    def augment_z_axis(
        grasp_poses: torch.Tensor,
        step_deg: float = 30.0,
        full_rotation: bool = True,
    ) -> torch.Tensor:
        """Z-axis full rotation augmentation (most common case).

        Args:
            grasp_poses: Input poses [B, 7]
            step_deg: Angle step in degrees
            full_rotation: Whether to do full 360 degree rotation

        Returns:
            Augmented poses [B, N, 7]
        """
        if full_rotation:
            num_poses = int(360 / step_deg)
            ang_range = (0, 360 - step_deg)  # Avoid duplicate at 0 and 360
        else:
            num_poses = int(360 / step_deg) + 1
            ang_range = (0, 360)

        return PoseAugmentor.augment_single_axis(
            grasp_poses, "z", ang_range, num_poses
        )

    @staticmethod
    def _apply_rotation_combinations(
        grasp_poses: torch.Tensor,
        x_angles: torch.Tensor,
        y_angles: torch.Tensor,
        z_angles: torch.Tensor,
    ) -> torch.Tensor:
        """Apply multiple rotation combinations efficiently.

        Args:
            grasp_poses: Original poses [B, 7]
            x_angles, y_angles, z_angles: Angle arrays [N]

        Returns:
            Augmented poses [B, N, 7]
        """
        device = grasp_poses.device
        N = len(x_angles)

        # Extract positions and quaternions
        positions = grasp_poses[:, :3]  # [B, 3]
        original_quats = grasp_poses[:, 3:]  # [B, 4] - (w, qx, qy, qz)

        # Create combined rotation quaternions
        combined_rot_quats = (
            PoseAugmentor._create_combined_rotation_quaternions(
                x_angles, y_angles, z_angles, device
            )
        )  # [N, 4]

        # Expand for batch multiplication
        expanded_orig_quats = original_quats.unsqueeze(1)  # [B, 1, 4]
        expanded_rot_quats = combined_rot_quats.unsqueeze(0)  # [1, N, 4]

        # Apply rotations: q_new = q_orig * q_rot
        new_quats = math_utils.quaternion_multiply(
            expanded_orig_quats, expanded_rot_quats
        )  # [B, N, 4]

        # Normalize quaternions
        new_quats = new_quats / torch.linalg.norm(
            new_quats, dim=-1, keepdim=True
        )

        # Expand positions
        expanded_positions = positions.unsqueeze(1).expand(
            -1, N, -1
        )  # [B, N, 3]

        # Combine results
        augmented_poses = torch.cat((expanded_positions, new_quats), dim=-1)

        return augmented_poses

    @staticmethod
    def _create_combined_rotation_quaternions(
        x_angles: torch.Tensor,
        y_angles: torch.Tensor,
        z_angles: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Create combined rotation quaternions for XYZ rotations.

        Args:
            x_angles, y_angles, z_angles: Rotation angles in degrees [N]
            device: Device to create tensors on

        Returns:
            Combined rotation quaternions [N, 4] - (w, qx, qy, qz)
        """

        # Convert to radians and get half angles
        x_rad = torch.deg2rad(x_angles) / 2.0
        y_rad = torch.deg2rad(y_angles) / 2.0
        z_rad = torch.deg2rad(z_angles) / 2.0

        # Create individual axis quaternions
        cx, sx = torch.cos(x_rad), torch.sin(x_rad)
        cy, sy = torch.cos(y_rad), torch.sin(y_rad)
        cz, sz = torch.cos(z_rad), torch.sin(z_rad)

        # X-axis rotation quaternions
        qx = torch.stack(
            [cx, sx, torch.zeros_like(cx), torch.zeros_like(cx)], dim=1
        )
        # Y-axis rotation quaternions
        qy = torch.stack(
            [cy, torch.zeros_like(cy), sy, torch.zeros_like(cy)], dim=1
        )
        # Z-axis rotation quaternions
        qz = torch.stack(
            [cz, torch.zeros_like(cz), torch.zeros_like(cz), sz], dim=1
        )

        # Combine rotations: q_combined = qz * qy * qx (ZYX order)
        temp = math_utils.quaternion_multiply(qz, qy)
        combined = math_utils.quaternion_multiply(temp, qx)

        return combined
