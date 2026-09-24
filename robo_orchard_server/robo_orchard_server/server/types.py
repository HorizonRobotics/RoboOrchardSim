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

"""Public dictionary contracts for canonical observations and joint targets."""

from __future__ import annotations
from typing import TypedDict

import numpy as np


class Pose(TypedDict):
    """Position [B, 3] and scalar-first quaternion [B, 4]."""

    xyz: np.ndarray
    quat: np.ndarray


class CameraObservation(TypedDict, total=False):
    """Only configured fields are present; arrays retain their source shape."""

    rgb: np.ndarray
    depth: np.ndarray
    intrinsic_matrices: np.ndarray
    pose: Pose


class _ManipulatorRequired(TypedDict):
    joint_position: np.ndarray


class ManipulatorObservation(_ManipulatorRequired, total=False):
    """Joint positions plus optional fields supplied by the binding schema."""

    gripper_position: np.ndarray
    ee_pose: np.ndarray
    base_pose: np.ndarray


class ManipulatorLayout(TypedDict):
    """Joint metadata; gripper settings are passed through without applying."""

    slot: str
    arm_joint_names: list[str]
    gripper_joint_names: list[str]
    gripper_policy_representation: str
    gripper_decode_coupling: str
    gripper_policy_scale: float


class ActionLayout(TypedDict):
    """Robot-specific layout supplied by the evaluator."""

    embodiment_type: str
    schema_version: str
    manipulator_order: list[str]
    manipulators: dict[str, ManipulatorLayout]


class Observation(TypedDict):
    """The plain dictionary passed to the user's adapter."""

    instruction: str | None
    cameras: dict[str, CameraObservation]
    manipulators: dict[str, ManipulatorObservation]
    action_layout: ActionLayout


class JointAction(TypedDict):
    """Physical target positions for all joints, in named column order."""

    joint_names: list[str]
    values: np.ndarray  # float32, [B, J]
