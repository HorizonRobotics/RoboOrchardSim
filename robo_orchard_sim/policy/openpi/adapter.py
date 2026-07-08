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
from typing import Any

import cv2
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

OpenPiAction = UnifiedJointCommand
DEFAULT_TARGET_INTRINSIC = [
    [290.0, 0.0, 196.0, 0.0],
    [0.0, 310.0, 126.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


_OPENPI_MODEL_CAMERA_SLOTS_BY_EMBODIMENT = {
    "franka_panda": {
        "left_wrist_0_rgb": "wrist_camera",
        "right_wrist_0_rgb": "ext2_camera",
        "base_0_rgb": "ext1_camera",
    },
    "panda_droid": {
        # base + wrist only; the unset right_wrist_0 slot is zeroed and masked.
        "left_wrist_0_rgb": "wrist_camera",
        "base_0_rgb": "ext1_camera",
    },
    "dualarm_piper": {
        "left_wrist_0_rgb": "left_wrist",
        "right_wrist_0_rgb": "right_wrist",
        "base_0_rgb": "base",
    },
    "dualarm_piperx": {
        "left_wrist_0_rgb": "left_wrist",
        "right_wrist_0_rgb": "right_wrist",
        "base_0_rgb": "base",
    },
}
OPENPI_IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)
DEFAULT_CAMERAS = {
    model_image_key: {
        "target_size": (392, 252),
        "target_intrinsic": [row[:] for row in DEFAULT_TARGET_INTRINSIC],
    }
    for model_image_key in OPENPI_IMAGE_KEYS
}


class OpenPiAdapter:
    """Transforms simulator observations to OpenPI inputs and actions back."""

    @classmethod
    def required_observation_fields(cls) -> dict[str, Any]:
        return {
            "camera_terms": [
                "left_hand_camera_term",
                "right_hand_camera_term",
                "static_camera_term",
            ],
            "include_rgb": True,
            "include_depth": False,
            "include_intrinsic": True,
            "include_pose": False,
            "robot_keys": [
                "left_joint_position",
                "right_joint_position",
            ],
        }

    def __init__(
        self,
        *,
        embodiment_type: str,
        cameras: dict[str, Any] | None = None,
        enable_intrinsic_remap: bool = True,
        gripper_binarize: bool = False,
        client_resize: str | None = None,
    ) -> None:
        self._embodiment_type = embodiment_type
        self._gripper_binarize = gripper_binarize
        self._client_resize = client_resize
        try:
            self._model_camera_slots = (
                _OPENPI_MODEL_CAMERA_SLOTS_BY_EMBODIMENT[embodiment_type]
            )
        except KeyError as exc:
            supported = tuple(_OPENPI_MODEL_CAMERA_SLOTS_BY_EMBODIMENT)
            raise ValueError(
                "Unsupported OpenPi embodiment_type "
                f"{embodiment_type!r}. Expected one of {supported}."
            ) from exc
        self._enable_intrinsic_remap = enable_intrinsic_remap
        self._camera_resize = (
            self._build_camera_resize(cameras)
            if enable_intrinsic_remap
            else {}
        )

    def build_model_input(self, obs: CanonicalPolicyInput) -> dict[str, Any]:
        instruction = self._require_instruction(obs)
        layout = self._require_action_layout(obs)
        validate_action_layout_compatibility(
            manipulator_observations=obs.manipulators,
            layout=layout,
            context="OpenPi observation",
        )
        images, intrinsics = self._extract_camera_inputs(obs)
        joint_state = self._build_joint_state(obs, layout=layout)
        images_for_model = (
            self._resize_images(images, intrinsics)
            if self._enable_intrinsic_remap
            else images
        )
        if self._client_resize is not None:
            images_for_model = {
                key: self._resize_with_pad(
                    image, 224, 224, method=self._client_resize
                )
                for key, image in images_for_model.items()
            }
        hist_joint_state = self._build_hist_joint_state(joint_state)
        return {
            "image": self._build_openpi_images(images_for_model),
            "image_mask": self._build_openpi_image_masks(images),
            "state": hist_joint_state[0, :],
            "prompt": instruction,
        }

    @staticmethod
    def _require_instruction(obs: CanonicalPolicyInput) -> str:
        instruction = obs.instruction
        if not instruction:
            raise ValueError("OpenPi observation requires instruction")
        return instruction

    @staticmethod
    def _require_action_layout(
        obs: CanonicalPolicyInput,
    ) -> CompiledActionLayout:
        layout = obs.action_layout
        if not isinstance(layout, CompiledActionLayout):
            raise ValueError(
                "OpenPi observation requires a compiled action layout"
            )
        return layout

    def _extract_camera_inputs(
        self,
        obs: CanonicalPolicyInput,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        images: dict[str, np.ndarray] = {}
        intrinsics: dict[str, np.ndarray] = {}
        for model_image_key, camera_slot in self._model_camera_slots.items():
            if camera_slot not in obs.cameras:
                raise ValueError(
                    f"OpenPi requires canonical camera slot {camera_slot!r}."
                )
            camera_obs = obs.cameras[camera_slot]
            rgb_sensor = camera_obs["rgb"]
            images[model_image_key] = rgb_sensor.sensor_data[0].cpu().numpy()

            intrinsic = np.eye(4, dtype=np.float64)
            intrinsic[:3, :3] = (
                rgb_sensor.intrinsic_matrices[0].cpu().numpy()[:3, :3]
            )
            intrinsics[model_image_key] = intrinsic
        return images, intrinsics

    def _build_camera_resize(
        self,
        cameras: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        cameras = cameras or DEFAULT_CAMERAS
        resize_params = {}
        for model_image_key in OPENPI_IMAGE_KEYS:
            if model_image_key not in cameras:
                raise ValueError(
                    f"Missing OpenPi camera resize config: {model_image_key}"
                )
            camera_cfg = cameras[model_image_key]
            target_size = tuple(self._cfg_get(camera_cfg, "target_size"))
            target_intrinsic = self._build_target_intrinsic(
                self._cfg_get(camera_cfg, "target_intrinsic")
            )
            resize_params[model_image_key] = {
                "target_size": target_size,
                "target_intrinsic": target_intrinsic,
                "target_points": self._build_target_points(
                    target_size=target_size,
                    target_intrinsic=target_intrinsic,
                ),
            }
        return resize_params

    @staticmethod
    def _cfg_get(cfg: Any, name: str) -> Any:
        if isinstance(cfg, dict):
            return cfg[name]
        return getattr(cfg, name)

    def _build_joint_state(
        self,
        obs: CanonicalPolicyInput,
        *,
        layout: CompiledActionLayout,
    ) -> np.ndarray:
        pieces = [
            self._build_manipulator_joint_state(
                obs.manipulators[slot],
                manipulator=layout.manipulators[slot],
            )
            for slot in layout.manipulator_order
        ]
        return np.concatenate(pieces, axis=0)[None, :]

    @staticmethod
    def _build_manipulator_joint_state(
        manipulator_obs: dict[str, Any],
        *,
        manipulator: ManipulatorActionSpec,
    ) -> np.ndarray:
        joint_position = (
            manipulator_obs["joint_position"][0].detach().cpu().numpy()
        )
        if joint_position.shape[0] < manipulator.arm_dim:
            raise ValueError(
                f"Manipulator {manipulator.slot!r} joint_position has "
                f"{joint_position.shape[0]} dims, expected at least "
                f"{manipulator.arm_dim}."
            )
        state = [joint_position[: manipulator.arm_dim]]
        if manipulator.gripper_joint_names:
            state.append(
                manipulator.extract_gripper_policy(
                    manipulator_obs,
                    joint_position=joint_position,
                )
            )
        return np.concatenate(state, axis=0)

    @staticmethod
    def _build_target_intrinsic(
        target_intrinsic: list[list[float]],
    ) -> np.ndarray:
        target_intrinsic = np.asarray(target_intrinsic, dtype=np.float64)
        if target_intrinsic.shape != (4, 4):
            raise ValueError(
                "OpenPi target_intrinsic must be a 4x4 matrix, got "
                f"{target_intrinsic.shape}"
            )
        output = np.eye(4, dtype=np.float64)
        output[:3, :3] = target_intrinsic[:3, :3]
        return output

    @staticmethod
    def _build_target_points(
        *,
        target_size: tuple[int, int],
        target_intrinsic: np.ndarray,
    ) -> np.ndarray:
        u, v = np.arange(target_size[0]), np.arange(target_size[1])
        u = np.repeat(u[None], target_size[1], 0)
        v = np.repeat(v[:, None], target_size[0], 1)
        uv = np.stack([u, v, np.ones_like(u)], axis=-1)
        return uv @ np.linalg.inv(target_intrinsic[:3, :3]).T

    def _resize_images(
        self,
        images: dict[str, np.ndarray],
        intrinsics: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        resized_images = {}
        for camera_position in images:
            resize_param = self._camera_resize[camera_position]
            src_intrinsic = intrinsics[camera_position][:3, :3]
            src_uv = (resize_param["target_points"] @ src_intrinsic.T).astype(
                np.float32
            )
            resized_images[camera_position] = cv2.remap(
                images[camera_position],
                src_uv[..., 0],
                src_uv[..., 1],
                cv2.INTER_LINEAR,
            )
        return resized_images

    def _build_openpi_images(
        self,
        images: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        openpi_images = {}
        for model_image_key in OPENPI_IMAGE_KEYS:
            image = images.get(model_image_key)
            if image is None:
                target_size = DEFAULT_CAMERAS[model_image_key]["target_size"]
                image = np.zeros(
                    (target_size[1], target_size[0], 3),
                    dtype=np.uint8,
                )
            openpi_images[model_image_key] = self._as_rgb_uint8(image)
        return openpi_images

    def _build_openpi_image_masks(
        self,
        images: dict[str, np.ndarray],
    ) -> dict[str, np.bool_]:
        return {
            model_image_key: np.bool_(model_image_key in images)
            for model_image_key in OPENPI_IMAGE_KEYS
        }

    @staticmethod
    def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            # Scale only unit-range floats; [0, 255] floats are cast as-is.
            if image.size == 0 or image.max() <= 1.0:
                image = image * 255
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    @staticmethod
    def _resize_with_pad(
        image: np.ndarray,
        target_h: int,
        target_w: int,
        *,
        method: str = "cv2",
    ) -> np.ndarray:
        """Aspect-preserving letterbox to target size, via cv2 or pil."""
        image = OpenPiAdapter._as_rgb_uint8(image)
        if method == "pil":
            from openpi_client import image_tools

            return image_tools.resize_with_pad(image, target_h, target_w)
        if method != "cv2":
            raise ValueError(f"Unknown client_resize method: {method!r}")
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
        return canvas

    @staticmethod
    def _build_hist_joint_state(joint_state: np.ndarray) -> np.ndarray:
        return np.clip(joint_state[-1:], -3.14, 3.14)

    def build_action_sequence(
        self,
        actions: np.ndarray | torch.Tensor,
        obs: CanonicalPolicyInput,
        *,
        device: torch.device | str,
        valid_action_step: int | None = None,
    ) -> list[OpenPiAction]:
        layout = self._require_action_layout(obs)
        actions = self._extract_action_tensor(actions, device=device)
        actions = self._truncate_action_dims(actions, layout=layout)
        actions = self._truncate_action_tensor(
            actions,
            valid_action_step=valid_action_step,
        )
        return self._actions_to_sequence(actions, layout=layout)

    @staticmethod
    def _extract_action_tensor(
        actions: np.ndarray | torch.Tensor,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        action_tensor = torch.as_tensor(
            actions,
            dtype=torch.float32,
            device=device,
        )
        if action_tensor.ndim != 2:
            raise ValueError("OpenPi model output actions must be 2D")
        return action_tensor

    @staticmethod
    def _truncate_action_tensor(
        actions: torch.Tensor,
        *,
        valid_action_step: int | None,
    ) -> torch.Tensor:
        if valid_action_step is None:
            return actions
        return actions[:valid_action_step]

    def _truncate_action_dims(
        self,
        actions: torch.Tensor,
        *,
        layout: CompiledActionLayout,
    ) -> torch.Tensor:
        expected_dim = sum(
            layout.manipulators[slot].model_dim
            for slot in layout.manipulator_order
        )
        if actions.shape[1] < expected_dim:
            raise ValueError(
                "OpenPi model output actions must provide at least "
                f"{expected_dim} action dimensions"
            )
        return actions[:, :expected_dim]

    def _actions_to_sequence(
        self,
        actions: torch.Tensor,
        *,
        layout: CompiledActionLayout,
    ) -> list[OpenPiAction]:
        sequence: list[OpenPiAction] = []
        for step_idx in range(actions.shape[0]):
            cursor = 0
            commands: list[UnifiedJointCommand] = []
            for slot in layout.manipulator_order:
                manipulator = layout.manipulators[slot]
                commands.append(
                    UnifiedJointCommand(
                        values=actions[
                            step_idx : step_idx + 1,
                            cursor : cursor + manipulator.arm_dim,
                        ],
                        joint_names=manipulator.arm_joint_names,
                    )
                )
                cursor += manipulator.arm_dim
                if manipulator.gripper_joint_names:
                    gripper = actions[
                        step_idx,
                        cursor : cursor + manipulator.gripper_policy_dim,
                    ]
                    if self._gripper_binarize:
                        # DROID binary gripper: snap at 0.5 (full open/close).
                        gripper = (gripper > 0.5).to(gripper.dtype)
                    commands.append(
                        UnifiedJointCommand(
                            values=policy_to_gripper_positions_torch(
                                gripper,
                                gripper_policy_representation=(
                                    manipulator.gripper_policy_representation
                                ),
                                gripper_decode_coupling=(
                                    manipulator.gripper_decode_coupling
                                ),
                                gripper_policy_scale=(
                                    manipulator.gripper_policy_scale
                                ),
                                joint_count=len(
                                    manipulator.gripper_joint_names
                                ),
                            ),
                            joint_names=manipulator.gripper_joint_names,
                        )
                    )
                    cursor += manipulator.gripper_policy_dim
            sequence.append(UnifiedJointCommand.merge(*commands))
        return sequence
