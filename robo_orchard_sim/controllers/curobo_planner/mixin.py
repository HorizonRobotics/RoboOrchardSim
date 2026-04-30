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

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List

import torch

__all__ = [
    "CannotFindTrajectoryError",
    "JointStateTrajetory",
    "ArticulationJointTrajPlannerMixin",
]


class CannotFindTrajectoryError(Exception):
    """Custom exception raised when a trajectory cannot be found."""

    pass


@dataclass
class IkResult:
    goal_poses: torch.Tensor
    """Shape: [BATCH , 7] or [BATCH, N, 7]"""

    success: torch.Tensor
    """Shape: [BATCH,  N]"""

    solution: torch.Tensor
    """Shape: [BATCH, N, NUM_JOINTS]"""

    goalset_index: torch.Tensor | None = None

    def __post_init__(self):
        if self.goal_poses.ndim == 2:
            # expend to [BATCH, N, 7]
            self.goal_poses = self.goal_poses.unsqueeze(1)


@dataclass
class JointStateTrajetory:
    """JointStateTrajetory.

    A data structure representing the joint state trajectory of a robotic
    system.

    Attributes:
        positions (torch.Tensor | List[torch.Tensor]):
            Represents the joint positions over time.
            - If a torch.Tensor is provided, it must have a shape of
            [BATCH, TRAJECTORY_LENGTH, NUM_JOINTS].
            - If a list is provided, it must contain tensors of shape
            [TRAJECTORY_LENGTH, NUM_JOINTS].

        indices (torch.Tensor):
            A 1D tensor representing the indices of the joints for each batch.
            Shape: [BATCH].

        velocities (torch.Tensor | List[torch.Tensor] | None):
            Represents the joint velocities over time.
            - If a torch.Tensor is provided, it must have the same shape
            as `positions`.
            - If a list is provided, it must contain tensors with the same
            shape as the corresponding elements in `positions`.
            - If None, velocities are not provided.

    Methods:
        __post_init__():
            Validates the input data during initialization to ensure that
            `positions`, `indices`, and `velocities` meet the required shapes
            and types.

    Raises:
        ValueError:
            If the input data does not meet the required shapes or types, such
            as mismatched batch sizes, invalid tensor dimensions, or
            inconsistent shapes between `positions` and `velocities`.
    """

    positions: torch.Tensor | List[torch.Tensor]
    indices: torch.Tensor
    velocities: torch.Tensor | List[torch.Tensor] | None

    def __post_init__(self):
        """Validates the shape of the input tensors during initialization.

        Raises:
            ValueError: If the tensor dimensions or shapes do not meet
            the requirements.
        """
        if self.indices.ndim != 1:
            raise ValueError(
                "indices should be a 1D tensor with shape [BATCH]"
            )

        # Validates whether positions is 3D Tensor or List[Tensor]
        if isinstance(self.positions, torch.Tensor):
            if self.positions.ndim == 2:
                self.positions = self.positions.unsqueeze(0)  # expend to 3D
            if self.positions.ndim != 3:
                raise ValueError(
                    "positions should be a 3D tensor with shape [BATCH, "
                    "TRAJECTORY_LENGTH, NUM_JOINTS], "
                    f"but got shape = {self.positions.shape}."
                )
        elif isinstance(self.positions, list):
            if not all(isinstance(p, torch.Tensor) for p in self.positions):
                raise ValueError(
                    "All elements in positions list must be torch.Tensor."
                )
            if not all(p.ndim == 2 for p in self.positions):
                raise ValueError(
                    "Each tensor in positions list must have shape "
                    "[TRAJECTORY_LENGTH, NUM_JOINTS]."
                )
        else:
            raise ValueError(
                "positions must be either a 3D torch.Tensor or a "
                "list of 2D torch.Tensor."
            )

        # Validates whether velocities is None, 3D Tensor or List[Tensor]
        if self.velocities is not None:
            if isinstance(self.velocities, torch.Tensor):
                if self.velocities.ndim == 2:
                    self.velocities = self.velocities.unsqueeze(
                        0
                    )  # expend to 3D
                if self.velocities.ndim != 3:
                    raise ValueError(
                        "velocities must be a 3D tensor with the same shape as positions, "  # noqa: E501
                        f"but got shape = {self.velocities.shape}."
                    )
            elif isinstance(self.velocities, list):
                if not all(
                    isinstance(v, torch.Tensor) for v in self.velocities
                ):
                    raise ValueError(
                        "All elements in velocities list must be torch.Tensor."
                    )
                if not all(v.ndim == 2 for v in self.velocities):
                    raise ValueError(
                        "Each tensor in velocities list must have shape [TRAJECTORY_LENGTH, NUM_JOINTS]."  # noqa: E501
                    )
            else:
                raise ValueError(
                    "velocities must be either None, a 3D torch.Tensor, or a list of 2D torch.Tensor."  # noqa: E501
                )

        # Validates the shape of positions and velocities
        if isinstance(self.positions, torch.Tensor) and isinstance(
            self.velocities, torch.Tensor
        ):
            if self.positions.shape != self.velocities.shape:
                raise ValueError(
                    "velocities must have the same shape as positions if provided, "  # noqa: E501
                    f"but got positions shape = {self.positions.shape}, "
                    f"velocities shape = {self.velocities.shape}."
                )
            elif isinstance(self.positions, list) and isinstance(
                self.velocities, list
            ):
                if len(self.positions) != len(self.velocities):
                    raise ValueError(
                        "positions and velocities lists must have the same "
                        "BATCH length if both are provided."
                    )

        # Validates whether batch size of positions match indices
        if isinstance(self.positions, torch.Tensor):
            if self.positions.shape[0] != self.indices.shape[0]:
                raise ValueError(
                    "The batch size of positions must match the size of indices, "  # noqa: E501
                    f"but got positions batch size = {self.positions.shape[0]}, "  # noqa: E501
                    f"indices size = {self.indices.shape[0]}."
                )
        elif isinstance(self.positions, list):
            if len(self.positions) != self.indices.shape[0]:
                raise ValueError(
                    "The length of positions list must match the size of indices, "  # noqa: E501
                    f"but got positions length = {len(self.positions)}, "
                    f"indices size = {self.indices.shape[0]}."
                )

    def get_last_positions(self) -> torch.Tensor:
        """Retrieves the last position for each batch from `positions`.

        Returns:
            torch.Tensor: A tensor of shape [BATCH, NUM_JOINTS] containing
                the last position for each batch.
        """
        if isinstance(self.positions, torch.Tensor):
            # If positions is a 3D tensor, directly slice the last time step
            return self.positions[:, -1, :]
        elif isinstance(self.positions, list):
            # If positions is a list of tensors, extract the last row
            # from each tensor
            return torch.stack([p[-1] for p in self.positions], dim=0)
        else:
            raise ValueError(
                "positions must be either a torch.Tensor or a list of torch.Tensor."  # noqa: E501
            )


class ArticulationJointTrajPlannerMixin(metaclass=ABCMeta):
    @abstractmethod
    def plan_to_target_ee_pose(
        self, start_joint_positions: torch.Tensor, target_poses: torch.Tensor
    ) -> JointStateTrajetory:
        """Plans a trajectory to reach the specified end-effector pose.

        Args:
            start_joint_positions (torch.Tensor): A 1D tensor representing the
                starting joint positions. Shape: [BATCH, NUM_JOINTS].
            target_poses (torch.Tensor): A 2D tensor representing the target
                poses for the end-effector.
                Shape: [BATCH, 7] where 6 represents [x, y, z, qx, qy, qz, qw].

        Returns:
            JointStateTrajetory: The planned joint positions, indices, and
                velocities.

        Raises:
            CannotFindTrajectoryError: If no valid trajectory can be planned.
        """

    @abstractmethod
    def plan_to_target_joint_positions(
        self,
        start_joint_positions: torch.Tensor,
        target_joint_positions: torch.Tensor,
    ) -> JointStateTrajetory:
        """Plans a trajectory to reach the specified joint positions.

        Args:
            start_joint_positions (torch.Tensor): A 1D tensor representing the
                starting joint positions. Shape: [BATCH, NUM_JOINTS].
            target_joint_positions (torch.Tensor): A 2D tensor representing
                the target joint positions. Shape: [BATCH, NUM_JOINTS].

        Returns:
            JointStateTrajetory: The planned joint positions, indices, and
                velocities.

        Raises:
            CannotFindTrajectoryError: If no valid trajectory can be planned.
        """
