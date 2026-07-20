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

"""Adapt canonical policy data to Cosmos DROID policy servers."""

from __future__ import annotations
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

CosmosAction = UnifiedJointCommand

DEFAULT_CAMERA_MAP = {
    "observation/wrist_image_left": "wrist_camera",
    "observation/exterior_image_1_left": "ext1_camera",
    "observation/exterior_image_2_left": "ext2_camera",
}


class CosmosAdapter:
    """Build Cosmos requests and decode absolute-joint action chunks."""

    def __init__(
        self,
        camera_map: dict[str, str] | None = None,
        manipulator_slot: str = "single_arm",
        default_instruction: str | None = None,
    ) -> None:
        self._camera_map = dict(camera_map or DEFAULT_CAMERA_MAP)
        self._manipulator_slot = manipulator_slot
        self._default_instruction = default_instruction

    def build_request(self, obs: CanonicalPolicyInput) -> dict[str, Any]:
        """Build the flat openpi-DROID request payload."""
        layout = self._require_action_layout(obs)
        validate_action_layout_compatibility(
            manipulator_observations=obs.manipulators,
            layout=layout,
            context="Cosmos observation",
        )
        request: dict[str, Any] = {}
        for request_key, slot in self._camera_map.items():
            if slot not in obs.cameras:
                raise ValueError(
                    f"Cosmos requires canonical camera slot {slot!r} for "
                    f"request key {request_key!r}."
                )
            frame = obs.cameras[slot]["rgb"].sensor_data[0].detach().cpu()
            request[request_key] = self._as_rgb_uint8(frame.numpy())
        manip = layout.manipulators[self._manipulator_slot]
        manipulator_obs = obs.manipulators[self._manipulator_slot]
        joint_position = (
            manipulator_obs["joint_position"][0].detach().cpu().numpy()
        )
        if joint_position.shape[0] < manip.arm_dim:
            raise ValueError(
                f"Manipulator {manip.slot!r} joint_position has "
                f"{joint_position.shape[0]} dims, expected at least "
                f"{manip.arm_dim}."
            )
        request["observation/joint_position"] = joint_position[
            : manip.arm_dim
        ].astype(np.float32)
        request["observation/gripper_position"] = manip.extract_gripper_policy(
            manipulator_obs, joint_position=joint_position
        ).astype(np.float32)
        request["prompt"] = self._require_instruction(obs)
        return request

    def build_action_sequence(
        self,
        action: Any,
        obs: CanonicalPolicyInput,
        *,
        device: torch.device | str,
        open_loop_horizon: int | None = None,
    ) -> list[CosmosAction]:
        """Decode a [T, arm+gripper] chunk into per-step joint commands."""
        layout = self._require_action_layout(obs)
        manip = layout.manipulators[self._manipulator_slot]
        values = np.asarray(action)
        if not values.flags.writeable:
            values = values.copy()
        chunk = torch.as_tensor(values, dtype=torch.float32, device=device)
        if chunk.ndim == 3:
            chunk = chunk[0]
        if chunk.ndim != 2:
            raise ValueError(
                "Cosmos action chunk must have shape [horizon, dim] or "
                f"[batch, horizon, dim], got {tuple(chunk.shape)}."
            )
        expected = manip.arm_dim + manip.gripper_policy_dim
        if chunk.shape[1] != expected:
            raise ValueError(
                f"Cosmos action dim {chunk.shape[1]} does not match "
                f"manipulator {manip.slot!r} arm+gripper dim {expected}."
            )
        horizon = chunk.shape[0]
        if open_loop_horizon is not None:
            horizon = min(horizon, open_loop_horizon)

        sequence: list[CosmosAction] = []
        for step in range(horizon):
            commands = [
                UnifiedJointCommand(
                    values=chunk[step, : manip.arm_dim].reshape(1, -1),
                    joint_names=manip.arm_joint_names,
                )
            ]
            if manip.gripper_joint_names:
                commands.append(
                    UnifiedJointCommand(
                        values=policy_to_gripper_positions_torch(
                            chunk[step, manip.arm_dim :],
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

    def _require_instruction(self, obs: CanonicalPolicyInput) -> str:
        instruction = obs.instruction or self._default_instruction
        if not instruction:
            raise ValueError(
                "Cosmos observation requires an instruction (set it on the "
                "task or via the policy config instruction)."
            )
        return instruction

    @staticmethod
    def _require_action_layout(
        obs: CanonicalPolicyInput,
    ) -> CompiledActionLayout:
        layout = obs.action_layout
        if not isinstance(layout, CompiledActionLayout):
            raise ValueError(
                "Cosmos observation requires a compiled action layout"
            )
        return layout

    @staticmethod
    def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image
