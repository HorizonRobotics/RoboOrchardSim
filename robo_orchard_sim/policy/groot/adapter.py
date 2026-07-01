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

# INTERNAL

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from robo_orchard_sim.contracts.joint_command import UnifiedJointCommand
from robo_orchard_sim.contracts.policy_binding import CanonicalPolicyInput
from robo_orchard_sim.policy.action_layout import (
    CompiledActionLayout,
    ManipulatorActionSpec,
    validate_action_layout_compatibility,
)
from robo_orchard_sim.policy.gripper_codec import (
    policy_to_gripper_positions_torch,
)

GrootAction = UnifiedJointCommand

# GR00T video key -> canonical camera slot (AnyMove dual-arm default).
DEFAULT_VIDEO_MAP = {
    "static_camera": "base",
    "left_hand_camera": "left_wrist",
    "right_hand_camera": "right_wrist",
}
DEFAULT_LANGUAGE_KEY = "annotation.human.task_description"


@dataclass(frozen=True)
class GrootArmSpec:
    """Map one canonical manipulator slot to GR00T arm/gripper keys."""

    manipulator_slot: str
    arm_key: str
    gripper_key: str
    arm_relative: bool = False


# AnyMove/GR00T returns absolute joint targets.
DEFAULT_ARM_SPECS = (
    GrootArmSpec("left_arm", "left_arm", "left_gripper"),
    GrootArmSpec("right_arm", "right_arm", "right_gripper"),
)


class GrootAdapter:
    """Convert canonical observations to GR00T inputs and actions back."""

    def __init__(
        self,
        video_map: dict[str, str] | None = None,
        arm_specs: tuple[GrootArmSpec, ...] | None = None,
        language_key: str = DEFAULT_LANGUAGE_KEY,
        default_instruction: str | None = None,
    ) -> None:
        self._video_map = dict(video_map or DEFAULT_VIDEO_MAP)
        self._arm_specs = tuple(arm_specs or DEFAULT_ARM_SPECS)
        self._language_key = language_key
        self._default_instruction = default_instruction

    def build_model_input(self, obs: CanonicalPolicyInput) -> dict[str, Any]:
        """Build the GR00T native ``{video, state, language}`` payload."""
        layout = self._require_action_layout(obs)
        validate_action_layout_compatibility(
            manipulator_observations=obs.manipulators,
            layout=layout,
            context="GR00T observation",
        )
        instruction = self._require_instruction(obs)
        return {
            "video": self._build_video(obs),
            "state": self._build_state(obs, layout=layout),
            "language": {self._language_key: [[instruction]]},
        }

    def build_action_sequence(
        self,
        action: dict[str, Any],
        obs: CanonicalPolicyInput,
        *,
        device: torch.device | str,
        open_loop_horizon: int | None = None,
    ) -> list[GrootAction]:
        """Decode a GR00T action chunk into per-step joint commands."""
        layout = self._require_action_layout(obs)
        arm_chunks = {}
        grip_chunks = {}
        current_arms = {}
        horizon: int | None = None
        for spec in self._arm_specs:
            manip = layout.manipulators[spec.manipulator_slot]
            arm = self._chunk(action[spec.arm_key], device=device)
            arm_chunks[spec.manipulator_slot] = arm
            current_arms[spec.manipulator_slot] = self._current_arm(
                obs, manipulator=manip, device=device
            )
            if manip.gripper_joint_names:
                grip_chunks[spec.manipulator_slot] = self._chunk(
                    action[spec.gripper_key], device=device
                )
            horizon = (
                arm.shape[0] if horizon is None else min(horizon, arm.shape[0])
            )
        if horizon is None:
            raise ValueError("GR00T adapter requires at least one arm spec.")
        if open_loop_horizon is not None:
            horizon = min(horizon, open_loop_horizon)

        sequence: list[GrootAction] = []
        for step in range(horizon):
            commands: list[UnifiedJointCommand] = []
            for spec in self._arm_specs:
                manip = layout.manipulators[spec.manipulator_slot]
                arm_step = arm_chunks[spec.manipulator_slot][step]
                if spec.arm_relative:
                    arm_step = current_arms[spec.manipulator_slot] + arm_step
                commands.append(
                    UnifiedJointCommand(
                        values=arm_step.reshape(1, -1),
                        joint_names=manip.arm_joint_names,
                    )
                )
                if manip.gripper_joint_names:
                    commands.append(
                        UnifiedJointCommand(
                            values=policy_to_gripper_positions_torch(
                                grip_chunks[spec.manipulator_slot][step],
                                gripper_policy_representation=(
                                    manip.gripper_policy_representation
                                ),
                                gripper_decode_coupling=(
                                    manip.gripper_decode_coupling
                                ),
                                gripper_policy_scale=manip.gripper_policy_scale,
                                joint_count=len(manip.gripper_joint_names),
                            ),
                            joint_names=manip.gripper_joint_names,
                        )
                    )
            sequence.append(UnifiedJointCommand.merge(*commands))
        return sequence

    def _build_video(self, obs: CanonicalPolicyInput) -> dict[str, np.ndarray]:
        video = {}
        for groot_key, slot in self._video_map.items():
            if slot not in obs.cameras:
                raise ValueError(
                    f"GR00T requires canonical camera slot {slot!r} for "
                    f"video key {groot_key!r}."
                )
            rgb_sensor = obs.cameras[slot]["rgb"]
            frame = rgb_sensor.sensor_data[0].detach().cpu().numpy()
            video[groot_key] = self._as_rgb_uint8(frame)[None, None, ...]
        return video

    def _build_state(
        self,
        obs: CanonicalPolicyInput,
        *,
        layout: CompiledActionLayout,
    ) -> dict[str, np.ndarray]:
        state = {}
        for spec in self._arm_specs:
            manip = layout.manipulators[spec.manipulator_slot]
            manipulator_obs = obs.manipulators[spec.manipulator_slot]
            joint_position = (
                manipulator_obs["joint_position"][0].detach().cpu().numpy()
            )
            if joint_position.shape[0] < manip.arm_dim:
                raise ValueError(
                    f"Manipulator {manip.slot!r} joint_position has "
                    f"{joint_position.shape[0]} dims, expected at least "
                    f"{manip.arm_dim}."
                )
            arm = joint_position[: manip.arm_dim].astype(np.float32)
            state[spec.arm_key] = arm[None, None, :]
            if manip.gripper_joint_names:
                gripper = manip.extract_gripper_policy(
                    manipulator_obs,
                    joint_position=joint_position,
                ).astype(np.float32)
                state[spec.gripper_key] = gripper[None, None, :]
        return state

    @staticmethod
    def _current_arm(
        obs: CanonicalPolicyInput,
        *,
        manipulator: ManipulatorActionSpec,
        device: torch.device | str,
    ) -> torch.Tensor:
        joint_position = obs.manipulators[manipulator.slot]["joint_position"]
        return joint_position[0, : manipulator.arm_dim].to(
            device=device, dtype=torch.float32
        )

    @staticmethod
    def _chunk(
        values: np.ndarray | torch.Tensor,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        if isinstance(values, np.ndarray) and not values.flags.writeable:
            values = values.copy()
        tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
        if tensor.ndim == 3:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError(
                "GR00T action chunk must have shape [horizon, dim] or "
                f"[batch, horizon, dim], got {tuple(tensor.shape)}."
            )
        return tensor

    def _require_instruction(self, obs: CanonicalPolicyInput) -> str:
        instruction = obs.instruction or self._default_instruction
        if not instruction:
            raise ValueError(
                "GR00T observation requires an instruction (set it on the "
                "task or via the policy config default_instruction)."
            )
        return instruction

    @staticmethod
    def _require_action_layout(
        obs: CanonicalPolicyInput,
    ) -> CompiledActionLayout:
        layout = obs.action_layout
        if not isinstance(layout, CompiledActionLayout):
            raise ValueError(
                "GR00T observation requires a compiled action layout"
            )
        return layout

    @staticmethod
    def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image
