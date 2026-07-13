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

"""Canonicalize raw observations for policy consumption."""

from __future__ import annotations
from typing import Any

from robo_orchard_sim.contracts.policy_binding import (
    CanonicalPolicyInput,
    PolicyBindingSchema,
    PolicyRequirement,
)
from robo_orchard_sim.policy.action_layout import compile_action_layout


def validate_policy_compatibility(
    *,
    canonical: CanonicalPolicyInput,
    requirement: PolicyRequirement,
) -> None:
    """Validate broad policy requirements against canonical policy input."""
    if requirement.require_instruction and not canonical.instruction:
        raise ValueError("Policy requires instruction, but none was provided.")

    if requirement.min_camera_count is not None:
        camera_count = len(canonical.cameras)
        if camera_count < requirement.min_camera_count:
            raise ValueError(
                "Policy requires at least "
                f"{requirement.min_camera_count} cameras, but canonical "
                f"input contains {camera_count}."
            )

    if requirement.min_manipulator_count is not None:
        manipulator_count = len(canonical.manipulators)
        if manipulator_count < requirement.min_manipulator_count:
            raise ValueError(
                "Policy requires at least "
                f"{requirement.min_manipulator_count} manipulators, but "
                f"canonical input contains {manipulator_count}."
            )

    missing_by_slot: dict[str, tuple[str, ...]] = {}
    for slot, camera_obs in canonical.cameras.items():
        missing = tuple(
            modality
            for modality in requirement.required_camera_modalities
            if not _canonical_camera_has_modality(camera_obs, modality)
        )
        if missing:
            missing_by_slot[slot] = missing

    if missing_by_slot:
        raise ValueError(
            "Policy camera modality requirements are not satisfied: "
            f"{missing_by_slot}."
        )


def _canonical_camera_has_modality(
    camera_obs: dict[str, Any],
    modality: str,
) -> bool:
    """Return whether one canonical camera observation provides a modality."""
    if modality in ("rgb", "depth"):
        return modality in camera_obs
    if modality == "intrinsic":
        rgb_sensor = camera_obs.get("rgb")
        return getattr(rgb_sensor, "intrinsic_matrices", None) is not None
    if modality == "pose":
        rgb_sensor = camera_obs.get("rgb")
        return getattr(rgb_sensor, "pose", None) is not None
    raise ValueError(f"Unsupported camera modality: {modality!r}")


def canonicalize_observations(
    *,
    observations: dict[str, Any],
    instruction: str | None,
    schema: PolicyBindingSchema,
) -> CanonicalPolicyInput:
    """Map raw simulator observations into canonical policy slots."""
    cameras: dict[str, dict[str, Any]] = {}
    if schema.camera_slots:
        if "/camera" not in observations:
            raise ValueError(
                "Policy binding schema declares cameras, but observations "
                "has no /camera group."
            )
        for slot, binding in schema.camera_slots.items():
            if binding.obs_term not in observations["/camera"]:
                raise ValueError(
                    f"Camera slot {slot!r} requires raw observation term "
                    f"{binding.obs_term!r}, but it is missing."
                )
            camera_obs = observations["/camera"][binding.obs_term]
            slot_obs: dict[str, Any] = {}
            if binding.rgb:
                if "rgb" not in camera_obs:
                    raise ValueError(
                        f"Camera slot {slot!r} requires rgb from raw term "
                        f"{binding.obs_term!r}, but it is missing."
                    )
                slot_obs["rgb"] = camera_obs["rgb"]
                _validate_camera_metadata(
                    slot=slot,
                    rgb_sensor=slot_obs["rgb"],
                    require_intrinsic=binding.intrinsic,
                    require_pose=binding.pose,
                )
            if binding.depth:
                if "depth" not in camera_obs:
                    raise ValueError(
                        f"Camera slot {slot!r} requires depth from raw term "
                        f"{binding.obs_term!r}, but it is missing."
                    )
                slot_obs["depth"] = camera_obs["depth"]
            cameras[slot] = slot_obs

    manipulators: dict[str, dict[str, Any]] = {}
    if "/robot" in observations:
        robot_obs = observations["/robot"]
        for slot, binding in schema.manipulator_slots.items():
            joint_position = robot_obs[binding.joint_position_obs_key]
            slot_obs = {"joint_position": joint_position}
            if binding.gripper_position_obs_key is not None:
                slot_obs["gripper_position"] = robot_obs[
                    binding.gripper_position_obs_key
                ]
            if binding.ee_pose_obs_key is not None:
                slot_obs["ee_pose"] = robot_obs[binding.ee_pose_obs_key]
            if binding.base_pose_obs_key is not None:
                slot_obs["base_pose"] = robot_obs[binding.base_pose_obs_key]
            manipulators[slot] = slot_obs

    return CanonicalPolicyInput(
        instruction=instruction,
        cameras=cameras,
        manipulators=manipulators,
        action_layout=compile_action_layout(schema),
    )


def _validate_camera_metadata(
    *,
    slot: str,
    rgb_sensor: Any,
    require_intrinsic: bool,
    require_pose: bool,
) -> None:
    if (
        require_intrinsic
        and getattr(rgb_sensor, "intrinsic_matrices", None) is None
    ):
        raise ValueError(f"Camera slot {slot!r} requires intrinsic metadata.")
    if require_pose and getattr(rgb_sensor, "pose", None) is None:
        raise ValueError(f"Camera slot {slot!r} requires pose metadata.")
