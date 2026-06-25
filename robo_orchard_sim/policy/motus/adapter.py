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

"""Adapt RoboOrchard canonical policy data to Motus server inputs."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from robo_orchard_sim.contracts.joint_command import UnifiedJointCommand
from robo_orchard_sim.contracts.policy_binding import CanonicalPolicyInput
from robo_orchard_sim.policy.action_layout import (
    CompiledActionLayout,
    validate_action_layout_compatibility,
)
from robo_orchard_sim.policy.gripper_codec import (
    policy_to_gripper_positions_torch,
)

DEFAULT_CAMERA_MAPPING = {
    "static_camera": "base",
    "left_hand_camera": "left_wrist",
    "right_hand_camera": "right_wrist",
}


@dataclass(frozen=True)
class _MotusCameraSpec:
    position: str
    request_key: str


_MOTUS_CAMERA_SPECS = (
    _MotusCameraSpec(
        position="static_camera",
        request_key="observation/exterior_image_0_left",
    ),
    _MotusCameraSpec(
        position="left_hand_camera",
        request_key="observation/exterior_image_1_left",
    ),
    _MotusCameraSpec(
        position="right_hand_camera",
        request_key="observation/wrist_image_left",
    ),
)


class MotusAdapter:
    """Build Motus requests and decode Motus action chunks."""

    def __init__(
        self,
        camera_mapping: dict[str, str] | None = None,
    ) -> None:
        self._camera_mapping = dict(camera_mapping or DEFAULT_CAMERA_MAPPING)
        self._validate_camera_mapping()

    def build_request(
        self,
        obs: CanonicalPolicyInput,
        *,
        session_id: str,
    ) -> dict[str, Any]:
        layout = self._require_action_layout(obs)
        validate_action_layout_compatibility(
            manipulator_observations=obs.manipulators,
            layout=layout,
            context="Motus observation",
        )
        request = self._build_camera_request(obs)
        joint_position, gripper_position = self._build_state(obs, layout)
        request.update(
            {
                "observation/joint_position": joint_position,
                "observation/gripper_position": gripper_position,
                "prompt": obs.instruction or "",
                "session_id": session_id,
            }
        )
        return request

    def build_action_sequence(
        self,
        actions: np.ndarray | torch.Tensor,
        obs: CanonicalPolicyInput,
        *,
        device: torch.device | str,
        valid_action_step: int | None = None,
    ) -> list[UnifiedJointCommand]:
        layout = self._require_action_layout(obs)
        action_tensor = torch.as_tensor(
            actions,
            dtype=torch.float32,
            device=device,
        )
        if action_tensor.ndim != 2:
            raise ValueError("Motus actions must be a 2D action chunk.")

        expected_dim = self._expected_action_dim(layout)
        if action_tensor.shape[1] != expected_dim:
            raise ValueError(
                "Expected Motus actions with "
                f"{expected_dim} dims, got {action_tensor.shape[1]}."
            )
        if valid_action_step is not None:
            action_tensor = action_tensor[:valid_action_step]

        return [
            self._build_step_command(action_tensor[step_idx], layout)
            for step_idx in range(action_tensor.shape[0])
        ]

    @staticmethod
    def _require_action_layout(
        obs: CanonicalPolicyInput,
    ) -> CompiledActionLayout:
        layout = obs.action_layout
        if not isinstance(layout, CompiledActionLayout):
            raise ValueError("Motus observation requires an action layout.")
        return layout

    @staticmethod
    def _expected_action_dim(layout: CompiledActionLayout) -> int:
        return sum(
            layout.manipulators[slot].model_dim
            for slot in layout.manipulator_order
        )

    def _build_state(
        self,
        obs: CanonicalPolicyInput,
        layout: CompiledActionLayout,
    ) -> tuple[np.ndarray, np.ndarray]:
        arm_pieces: list[np.ndarray] = []
        gripper_pieces: list[np.ndarray] = []
        for slot in layout.manipulator_order:
            manipulator = layout.manipulators[slot]
            manipulator_obs = obs.manipulators[slot]
            joint_position = (
                manipulator_obs["joint_position"][0].detach().cpu().numpy()
            )
            if joint_position.shape[0] < manipulator.arm_dim:
                raise ValueError(
                    f"Manipulator {slot!r} joint_position has "
                    f"{joint_position.shape[0]} dims, expected at least "
                    f"{manipulator.arm_dim}."
                )
            arm_pieces.append(joint_position[: manipulator.arm_dim])
            if manipulator.gripper_joint_names:
                gripper_pieces.append(
                    manipulator.extract_gripper_policy(
                        manipulator_obs,
                        joint_position=joint_position,
                    )
                )

        joint_position = np.concatenate(arm_pieces).astype(np.float32)
        if gripper_pieces:
            gripper_position = np.concatenate(gripper_pieces).astype(
                np.float32
            )
        else:
            gripper_position = np.zeros((0,), dtype=np.float32)
        return joint_position, gripper_position

    def _build_step_command(
        self,
        action: torch.Tensor,
        layout: CompiledActionLayout,
    ) -> UnifiedJointCommand:
        cursor = 0
        commands: list[UnifiedJointCommand] = []
        for slot in layout.manipulator_order:
            manipulator = layout.manipulators[slot]
            arm = action[cursor : cursor + manipulator.arm_dim].reshape(1, -1)
            commands.append(
                UnifiedJointCommand(
                    values=arm,
                    joint_names=manipulator.arm_joint_names,
                )
            )
            cursor += manipulator.arm_dim

            if manipulator.gripper_joint_names:
                gripper_policy = action[
                    cursor : cursor + manipulator.gripper_policy_dim
                ]
                gripper = policy_to_gripper_positions_torch(
                    gripper_policy,
                    gripper_policy_representation=(
                        manipulator.gripper_policy_representation
                    ),
                    gripper_decode_coupling=(
                        manipulator.gripper_decode_coupling
                    ),
                    gripper_policy_scale=manipulator.gripper_policy_scale,
                    joint_count=len(manipulator.gripper_joint_names),
                )
                commands.append(
                    UnifiedJointCommand(
                        values=gripper,
                        joint_names=manipulator.gripper_joint_names,
                    )
                )
                cursor += manipulator.gripper_policy_dim
        return UnifiedJointCommand.merge(*commands)

    def _build_camera_request(
        self,
        obs: CanonicalPolicyInput,
    ) -> dict[str, np.ndarray]:
        images_by_position: dict[str, np.ndarray] = {}
        request: dict[str, np.ndarray] = {}
        for spec in _MOTUS_CAMERA_SPECS:
            image = self._resolve_camera_image(
                obs,
                spec.position,
                images_by_position,
            )
            images_by_position[spec.position] = image
            request[spec.request_key] = image
        return request

    def _resolve_camera_image(
        self,
        obs: CanonicalPolicyInput,
        position: str,
        images_by_position: dict[str, np.ndarray],
    ) -> np.ndarray:
        mapping = self._camera_mapping[position]
        if mapping.startswith("duplicate:"):
            source_position = mapping.removeprefix("duplicate:")
            if source_position not in images_by_position:
                raise ValueError(
                    f"Motus camera {position!r} duplicates "
                    f"{source_position!r}, but source is not available yet."
                )
            return images_by_position[source_position].copy()

        if mapping not in obs.cameras:
            raise ValueError(
                f"Motus camera {position!r} requires canonical slot "
                f"{mapping!r}, got {tuple(sorted(obs.cameras))}."
            )
        camera_obs = obs.cameras[mapping]
        if "rgb" not in camera_obs:
            raise ValueError(
                f"Motus camera slot {mapping!r} must contain rgb data."
            )
        return self._as_rgb_uint8(camera_obs["rgb"].sensor_data)

    @staticmethod
    def _as_rgb_uint8(sensor_data: Any) -> np.ndarray:
        image = sensor_data
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(
                    "Motus adapter supports single-environment camera "
                    f"batches only, got shape {image.shape}."
                )
            image = image[0]
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                "Motus camera images must have shape [H, W, 3], got "
                f"{image.shape}."
            )
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        # Motus server currently expects OpenCV-style BGR images, while
        # RoboOrchard camera observations are RGB.
        image = image[..., ::-1]
        return np.ascontiguousarray(image)

    def _validate_camera_mapping(self) -> None:
        missing_positions = {
            spec.position for spec in _MOTUS_CAMERA_SPECS
        } - set(self._camera_mapping)
        if missing_positions:
            raise ValueError(
                "Motus camera_mapping missing positions: "
                f"{tuple(sorted(missing_positions))}."
            )
