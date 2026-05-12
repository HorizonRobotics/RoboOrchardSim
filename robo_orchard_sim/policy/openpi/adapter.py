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

from __future__ import annotations
from typing import Any

import cv2
import numpy as np
import torch

OpenPiAction = dict[str, torch.Tensor]
OPENPI_IMAGE_KEYS = (
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
    "base_0_rgb",
)
CAMERA_BINDINGS = {
    "left": {
        "obs_term": "left_hand_camera_term",
        "model_image_key": "left_wrist_0_rgb",
    },
    "right": {
        "obs_term": "right_hand_camera_term",
        "model_image_key": "right_wrist_0_rgb",
    },
    "middle": {
        "obs_term": "static_camera_term",
        "model_image_key": "base_0_rgb",
    },
}
DEFAULT_TARGET_INTRINSIC = [
    [290.0, 0.0, 196.0, 0.0],
    [0.0, 310.0, 126.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
DEFAULT_CAMERAS = {
    camera_position: {
        "target_size": (392, 252),
        "target_intrinsic": [row[:] for row in DEFAULT_TARGET_INTRINSIC],
    }
    for camera_position in CAMERA_BINDINGS
}


class OpenPiAdapter:
    """Transforms simulator observations to OpenPI inputs and actions back."""

    def __init__(
        self,
        joint_num: int = 7,
        cameras: dict[str, Any] | None = None,
        enable_intrinsic_remap: bool = True,
    ) -> None:
        self._joint_num = joint_num
        self._enable_intrinsic_remap = enable_intrinsic_remap
        self._camera_resize = (
            self._build_camera_resize(cameras)
            if enable_intrinsic_remap
            else {}
        )

    def build_model_input(self, obs: dict[str, Any]) -> dict[str, Any]:
        instruction = self._require_instruction(obs)
        images, intrinsics = self._extract_camera_inputs(obs)
        joint_state = self._build_joint_state(obs)
        images_for_model = (
            self._resize_images(images, intrinsics)
            if self._enable_intrinsic_remap
            else images
        )
        hist_joint_state = self._build_hist_joint_state(joint_state)
        return {
            "image": self._build_openpi_images(images_for_model),
            "image_mask": self._build_openpi_image_masks(),
            "state": hist_joint_state[0, :],
            "prompt": instruction,
        }

    @staticmethod
    def _require_instruction(obs: dict[str, Any]) -> str:
        instruction = obs.get("instruction")
        if not instruction:
            raise ValueError("OpenPi observation requires instruction")
        return instruction

    def _extract_camera_inputs(
        self,
        obs: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        images = {}
        intrinsics = {}
        for camera_binding in CAMERA_BINDINGS.values():
            obs_term = camera_binding["obs_term"]
            model_image_key = camera_binding["model_image_key"]
            camera_obs = self._require_camera_obs(obs, obs_term)
            rgb_sensor = camera_obs["rgb"]
            image = rgb_sensor.sensor_data[0].cpu().numpy()
            images[model_image_key] = image

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
        for camera_position in CAMERA_BINDINGS:
            if camera_position not in cameras:
                raise ValueError(
                    f"Missing OpenPi camera resize config: {camera_position}"
                )
            camera_cfg = cameras[camera_position]
            target_size = tuple(self._cfg_get(camera_cfg, "target_size"))
            target_intrinsic = self._build_target_intrinsic(
                self._cfg_get(camera_cfg, "target_intrinsic")
            )
            resize_params[camera_position] = {
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

    @staticmethod
    def _require_camera_obs(obs: dict[str, Any], obs_term: str) -> Any:
        try:
            return obs["/camera"][obs_term]
        except KeyError as exc:
            raise ValueError(
                f"Missing required camera observation term: {obs_term}"
            ) from exc

    def _build_joint_state(self, obs: dict[str, Any]) -> np.ndarray:
        robot_obs = obs["/robot"]
        left_joint_state = (
            robot_obs["left_joint_position"][0, : self._joint_num]
            .cpu()
            .numpy()
        )
        right_joint_state = (
            robot_obs["right_joint_position"][0, : self._joint_num]
            .cpu()
            .numpy()
        )
        joint_state = np.concatenate(
            [left_joint_state, right_joint_state],
            axis=0,
        )[None, :]
        joint_state[:, self._joint_num - 1] *= 2.0
        joint_state[:, 2 * self._joint_num - 1] *= 2.0
        return joint_state

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
        for camera_position, camera_binding in CAMERA_BINDINGS.items():
            model_image_key = camera_binding["model_image_key"]
            resize_param = self._camera_resize[camera_position]
            src_intrinsic = intrinsics[model_image_key][:3, :3]
            src_uv = (resize_param["target_points"] @ src_intrinsic.T).astype(
                np.float32
            )
            resized_images[model_image_key] = cv2.remap(
                images[model_image_key],
                src_uv[..., 0],
                src_uv[..., 1],
                cv2.INTER_LINEAR,
            )
        return resized_images

    @staticmethod
    def _build_openpi_images(
        images: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        return {
            key: OpenPiAdapter._as_rgb_uint8(images[key])
            for key in OPENPI_IMAGE_KEYS
        }

    @staticmethod
    def _build_openpi_image_masks() -> dict[str, np.bool_]:
        return {key: np.True_ for key in OPENPI_IMAGE_KEYS}

    @staticmethod
    def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = image.astype(np.uint8)
        return image

    @staticmethod
    def _build_hist_joint_state(joint_state: np.ndarray) -> np.ndarray:
        return np.clip(joint_state[-1:], -3.14, 3.14)

    def build_action_sequence(
        self,
        actions: np.ndarray | torch.Tensor,
        *,
        device: torch.device | str,
        valid_action_step: int | None = None,
    ) -> list[OpenPiAction]:
        actions = self._extract_action_tensor(actions, device=device)
        actions = self._truncate_action_dims(actions)
        actions = self._truncate_action_tensor(
            actions,
            valid_action_step=valid_action_step,
        )
        return self._actions_to_sequence(actions)

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

    def _truncate_action_dims(self, actions: torch.Tensor) -> torch.Tensor:
        expected_dim = 2 * self._joint_num
        if actions.shape[1] < expected_dim:
            raise ValueError(
                "OpenPi model output actions must provide at least "
                f"{expected_dim} action dimensions"
            )
        return actions[:, :expected_dim]

    def _actions_to_sequence(
        self, actions: torch.Tensor
    ) -> list[OpenPiAction]:
        sequence: list[OpenPiAction] = []
        left_joint_end = self._joint_num - 1
        right_joint_start = self._joint_num
        right_joint_end = 2 * self._joint_num - 1

        for step_idx in range(actions.shape[0]):
            left_gripper = actions[step_idx, left_joint_end]
            right_gripper = actions[step_idx, right_joint_end]
            sequence.append(
                {
                    "left_robot_joint_position": actions[
                        step_idx : step_idx + 1, :left_joint_end
                    ],
                    "left_robot_gripper_control": self._build_gripper_control(
                        left_gripper
                    ),
                    "right_robot_joint_position": actions[
                        step_idx : step_idx + 1,
                        right_joint_start:right_joint_end,
                    ],
                    "right_robot_gripper_control": self._build_gripper_control(
                        right_gripper
                    ),
                }
            )
        return sequence

    @staticmethod
    def _build_gripper_control(gripper: torch.Tensor) -> torch.Tensor:
        half_gripper = gripper / 2
        return torch.stack((half_gripper, -half_gripper)).unsqueeze(0)
