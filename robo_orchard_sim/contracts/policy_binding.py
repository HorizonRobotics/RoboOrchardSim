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

"""Schema types for binding raw observations to canonical policy inputs."""

from __future__ import annotations
from typing import Any, Literal

from pydantic import BaseModel, Field


class CameraBinding(BaseModel):
    """Bind one canonical camera slot to one raw camera observation term."""

    obs_term: str
    rgb: bool = True
    depth: bool = False
    intrinsic: bool = False
    pose: bool = False


class ManipulatorBinding(BaseModel):
    """Bind one canonical manipulator slot to raw robot observation keys."""

    joint_position_obs_key: str
    gripper_position_obs_key: str | None = None
    ee_pose_obs_key: str | None = None
    base_pose_obs_key: str | None = None
    arm_joint_name_specs: tuple[str, ...]
    gripper_joint_name_specs: tuple[str, ...] = ()
    gripper_policy_representation: Literal["first_joint", "all_joints"] = (
        "first_joint"
    )
    gripper_decode_coupling: Literal[
        "symmetric",
        "mirrored",
        "identity",
    ] = "symmetric"
    gripper_policy_scale: float = 1.0


class PolicyBindingSchema(BaseModel):
    """Describe one embodiment's raw-to-canonical observation bindings."""

    schema_version: str
    embodiment_type: str
    camera_slots: dict[str, CameraBinding] = Field(default_factory=dict)
    manipulator_slots: dict[str, ManipulatorBinding] = Field(
        default_factory=dict
    )


class PolicyRequirement(BaseModel):
    """Describe the broad canonical input contract a policy requires."""

    required_camera_modalities: tuple[str, ...] = ()
    min_camera_count: int | None = None
    min_manipulator_count: int | None = None
    require_instruction: bool = False


class CanonicalPolicyInput(BaseModel):
    """Canonical policy input assembled from raw environment observations."""

    instruction: str | None = None
    cameras: dict[str, dict[str, Any]] = Field(default_factory=dict)
    manipulators: dict[str, dict[str, Any]] = Field(default_factory=dict)
    action_layout: Any | None = None
