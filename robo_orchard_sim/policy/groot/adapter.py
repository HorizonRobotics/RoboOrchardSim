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
import robo_orchard_core.utils.math as math_utils
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

# Neutral end-effector pose: zero position + identity rotation (XYZ + ROT6D).
_IDENTITY_EEF_9D = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
_IDENTITY_QUAT = torch.tensor([[1.0, 0.0, 0.0, 0.0]])


@dataclass(frozen=True)
class GrootArmSpec:
    """Map one canonical manipulator slot to GR00T arm/gripper keys."""

    manipulator_slot: str
    arm_key: str
    gripper_key: str
    arm_relative: bool = False
    eef_key: str | None = None


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
        eef_ee_to_tcp_offset: tuple[float, float, float] | None = None,
    ) -> None:
        self._video_map = dict(video_map or DEFAULT_VIDEO_MAP)
        self._arm_specs = tuple(arm_specs or DEFAULT_ARM_SPECS)
        self._language_key = language_key
        self._default_instruction = default_instruction
        self._eef_tcp_offset = (
            torch.tensor([list(eef_ee_to_tcp_offset)], dtype=torch.float32)
            if eef_ee_to_tcp_offset is not None
            else None
        )

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
            frame = obs.cameras[slot]["rgb"].sensor_data[0].detach().cpu()
            # [B=1, T=1, H, W, 3]
            video[groot_key] = self._as_rgb_uint8(frame.numpy())[None, None]
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
            if spec.eef_key is not None:
                state[spec.eef_key] = self._build_eef(manipulator_obs)[
                    None, None, :
                ]
        return state

    def _build_eef(self, manipulator_obs: dict[str, Any]) -> np.ndarray:
        """Return the 9D end-effector state in the robot base frame.

        The pose is XYZ + first-two-rows ROT6D; falls back to an identity
        pose when the ee/base poses are unavailable.
        """
        ee_pose = manipulator_obs.get("ee_pose")
        base_pose = manipulator_obs.get("base_pose")
        if ee_pose is None or base_pose is None:
            return _IDENTITY_EEF_9D.copy()
        # Compute on CPU: the poses may be on cuda while the offset/identity
        # constants are on CPU, and the result is serialized to numpy anyway.
        ee = (
            torch.as_tensor(ee_pose, dtype=torch.float32)
            .reshape(-1, 7)[0:1]
            .detach()
            .cpu()
        )
        base = (
            torch.as_tensor(base_pose, dtype=torch.float32)
            .reshape(-1, 7)[0:1]
            .detach()
            .cpu()
        )
        ee_pos, ee_quat = ee[:, :3], ee[:, 3:]
        if self._eef_tcp_offset is not None:
            ee_pos, ee_quat = math_utils.frame_transform_combine(
                ee_pos, ee_quat, self._eef_tcp_offset, _IDENTITY_QUAT
            )
        pos_b, quat_b = math_utils.frame_transform_subtract(
            base[:, :3], base[:, 3:], ee_pos, ee_quat
        )
        rot = math_utils.quaternion_to_matrix(quat_b[0])
        eef = torch.cat([pos_b[0], rot[0], rot[1]])
        eef_np = eef.detach().cpu().numpy().astype(np.float32)
        return eef_np

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
