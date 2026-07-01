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

"""Compiled policy runtime layout derived from one binding schema."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from robo_orchard_sim.contracts.joint_command import (
    resolve_joint_name_specs,
)
from robo_orchard_sim.contracts.policy_binding import (
    ManipulatorBinding,
    PolicyBindingSchema,
)
from robo_orchard_sim.policy.gripper_codec import (
    gripper_policy_dim,
    gripper_positions_to_policy_numpy,
)


@dataclass(frozen=True)
class ManipulatorActionSpec:
    """Normalized manipulator layout for policy-side compilation."""

    slot: str
    arm_joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]
    gripper_policy_representation: str
    gripper_decode_coupling: str
    gripper_policy_scale: float

    @property
    def arm_dim(self) -> int:
        """Return the number of arm joints represented in model state."""
        return len(self.arm_joint_names)

    @property
    def model_dim(self) -> int:
        """Return this manipulator's model action/state dimension."""
        return self.arm_dim + self.gripper_policy_dim

    @property
    def gripper_policy_dim(self) -> int:
        """Return the number of policy dimensions used by the gripper."""
        if not self.gripper_joint_names:
            return 0
        return gripper_policy_dim(
            joint_count=len(self.gripper_joint_names),
            representation=self.gripper_policy_representation,
        )

    def extract_gripper_policy(
        self,
        manipulator_obs: dict[str, Any],
        *,
        joint_position: np.ndarray,
    ) -> np.ndarray:
        """Extract this manipulator's policy gripper representation."""
        if "gripper_position" in manipulator_obs:
            gripper_position = (
                manipulator_obs["gripper_position"][0].detach().cpu().numpy()
            )
        else:
            gripper_position = joint_position[self.arm_dim :]
        return gripper_positions_to_policy_numpy(
            gripper_position,
            gripper_policy_representation=self.gripper_policy_representation,
            gripper_policy_scale=self.gripper_policy_scale,
        )


@dataclass(frozen=True)
class CompiledActionLayout:
    """Policy-side runtime layout for canonical manipulator actions."""

    embodiment_type: str
    schema_version: str
    manipulators: dict[str, ManipulatorActionSpec]
    manipulator_order: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        """Serialize the compiled layout for remote transport."""
        return {
            "embodiment_type": self.embodiment_type,
            "schema_version": self.schema_version,
            "manipulator_order": list(self.manipulator_order),
            "manipulators": {
                slot: {
                    "slot": manip.slot,
                    "arm_joint_names": list(manip.arm_joint_names),
                    "gripper_joint_names": list(manip.gripper_joint_names),
                    "gripper_policy_representation": (
                        manip.gripper_policy_representation
                    ),
                    "gripper_decode_coupling": (manip.gripper_decode_coupling),
                    "gripper_policy_scale": manip.gripper_policy_scale,
                }
                for slot, manip in self.manipulators.items()
            },
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
    ) -> "CompiledActionLayout":
        """Deserialize one compiled layout payload."""
        manipulators = {
            slot: ManipulatorActionSpec(
                slot=data["slot"],
                arm_joint_names=tuple(data["arm_joint_names"]),
                gripper_joint_names=tuple(data["gripper_joint_names"]),
                gripper_policy_representation=data[
                    "gripper_policy_representation"
                ],
                gripper_decode_coupling=data["gripper_decode_coupling"],
                gripper_policy_scale=data["gripper_policy_scale"],
            )
            for slot, data in payload["manipulators"].items()
        }
        return cls(
            embodiment_type=payload["embodiment_type"],
            schema_version=payload["schema_version"],
            manipulators=manipulators,
            manipulator_order=tuple(payload["manipulator_order"]),
        )


def compile_action_layout(
    schema: PolicyBindingSchema,
) -> CompiledActionLayout:
    """Compile one binding schema into a policy-friendly manipulator layout."""
    manipulators = {
        slot: _compile_manipulator_layout(slot, binding)
        for slot, binding in schema.manipulator_slots.items()
    }
    return CompiledActionLayout(
        embodiment_type=schema.embodiment_type,
        schema_version=schema.schema_version,
        manipulators=manipulators,
        manipulator_order=tuple(schema.manipulator_slots),
    )


def validate_action_layout_compatibility(
    *,
    manipulator_observations: Mapping[str, Any],
    layout: CompiledActionLayout,
    context: str,
) -> None:
    """Validate that canonical manipulator data matches compiled layout."""
    observed_slots = set(manipulator_observations)
    layout_slots = set(layout.manipulators)
    missing = tuple(
        slot for slot in layout.manipulator_order if slot not in observed_slots
    )
    if missing:
        raise ValueError(
            f"{context} missing manipulator slots required by layout: "
            f"{missing}."
        )
    unexpected = tuple(sorted(observed_slots - layout_slots))
    if unexpected:
        raise ValueError(
            f"{context} unexpected manipulator slots not declared by layout: "
            f"{unexpected}."
        )


def _compile_manipulator_layout(
    slot: str,
    binding: ManipulatorBinding,
) -> ManipulatorActionSpec:
    arm_joint_names = resolve_joint_name_specs(binding.arm_joint_name_specs)
    if not arm_joint_names:
        raise ValueError(
            f"Manipulator slot {slot!r} must define arm_joint_name_specs."
        )
    gripper_joint_names = resolve_joint_name_specs(
        binding.gripper_joint_name_specs
    )
    _validate_gripper_policy_config(
        slot=slot,
        gripper_joint_count=len(gripper_joint_names),
        gripper_policy_representation=binding.gripper_policy_representation,
        gripper_decode_coupling=binding.gripper_decode_coupling,
        gripper_policy_scale=binding.gripper_policy_scale,
    )
    return ManipulatorActionSpec(
        slot=slot,
        arm_joint_names=arm_joint_names,
        gripper_joint_names=gripper_joint_names,
        gripper_policy_representation=binding.gripper_policy_representation,
        gripper_decode_coupling=binding.gripper_decode_coupling,
        gripper_policy_scale=binding.gripper_policy_scale,
    )


def _validate_gripper_policy_config(
    *,
    slot: str,
    gripper_joint_count: int,
    gripper_policy_representation: str,
    gripper_decode_coupling: str,
    gripper_policy_scale: float,
) -> None:
    if gripper_joint_count == 0:
        return
    if gripper_policy_scale == 0.0:
        raise ValueError(
            f"Manipulator slot {slot!r} gripper_policy_scale must be nonzero."
        )
    if gripper_policy_representation == "all_joints":
        if gripper_decode_coupling != "identity":
            raise ValueError(
                f"Manipulator slot {slot!r} uses all_joints gripper policy "
                "representation, which requires identity decode coupling."
            )
        return
    if gripper_policy_representation == "first_joint":
        if gripper_decode_coupling == "identity" and gripper_joint_count != 1:
            raise ValueError(
                f"Manipulator slot {slot!r} uses identity decode coupling "
                "with first_joint representation, which requires exactly one "
                "gripper joint."
            )
        return
    raise ValueError(
        f"Manipulator slot {slot!r} has unsupported gripper policy "
        f"representation {gripper_policy_representation!r}."
    )
