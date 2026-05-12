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
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from robo_orchard_lab.models.holobrain.processor import (
        MultiArmManipulationInput,
        MultiArmManipulationOutput,
    )


class HolobrainAdapter:
    """Transforms sim observations to Holobrain inputs and back."""

    _T_SIM_WORLD_TO_WORLD = np.array(
        [[1, 0, 0, 0.3], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    _CAMERA_TERMS = (
        "left_hand_camera_term",
        "static_camera_term",
        "right_hand_camera_term",
    )

    @classmethod
    def required_observation_fields(cls) -> dict[str, Any]:
        return {
            "camera_terms": list(cls._CAMERA_TERMS),
            "include_rgb": True,
            "include_depth": True,
            "include_intrinsic": True,
            "include_pose": True,
            "robot_keys": [
                "left_joint_position",
                "right_joint_position",
            ],
        }

    def __init__(self, joint_num: int) -> None:
        self._camera_terms = {
            "left": "left_hand_camera_term",
            "middle": "static_camera_term",
            "right": "right_hand_camera_term",
        }
        self._joint_num = joint_num

    def build_model_input(
        self,
        obs: dict,
    ) -> "MultiArmManipulationInput":
        instruction = self._require_instruction(obs)
        images, depths, intrinsics, t_world2cam = self._extract_camera_inputs(
            obs
        )
        joint_state = self._build_joint_state(obs)

        from robo_orchard_lab.models.holobrain.processor import (
            MultiArmManipulationInput,
        )

        return MultiArmManipulationInput(
            image=images,
            depth=depths,
            intrinsic=intrinsics,
            t_world2cam=t_world2cam,
            history_joint_state=joint_state,
            instruction=instruction,
        )

    @staticmethod
    def _require_instruction(obs: dict) -> str:
        instruction = obs.get("instruction")
        if not instruction:
            raise ValueError("Holobrain observation requires instruction")
        return instruction

    def _extract_camera_inputs(
        self,
        obs: dict,
    ) -> tuple[dict, dict, dict, dict]:
        images = {}
        depths = {}
        intrinsics = {}
        t_world2cam = {}

        for logical_name, term in self._camera_terms.items():
            camera_obs = self._require_camera_obs(obs, term)
            rgb, depth, intrinsic, camera_t_world2cam = (
                self._extract_single_camera_input(camera_obs)
            )
            images[logical_name] = [rgb]
            depths[logical_name] = [depth]
            intrinsics[logical_name] = intrinsic
            t_world2cam[logical_name] = camera_t_world2cam

        return images, depths, intrinsics, t_world2cam

    def _extract_single_camera_input(
        self,
        camera_obs: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rgb_sensor = camera_obs["rgb"]
        depth_sensor = camera_obs["depth"]

        rgb = rgb_sensor.sensor_data[0].cpu().numpy().astype(np.uint8)
        # Holobrain expects BGR images as input.
        rgb = rgb[..., ::-1]
        depth = depth_sensor.sensor_data[0].cpu().numpy()
        intrinsic = (
            rgb_sensor.intrinsic_matrices[0]
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        return (
            rgb,
            depth,
            self._to_homogeneous_intrinsic(intrinsic),
            self._compute_world_to_camera(camera_obs),
        )

    def _build_joint_state(self, obs: dict) -> np.ndarray:
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

    def _require_camera_obs(self, obs: dict, term: str) -> Any:
        try:
            return obs["/camera"][term]
        except KeyError as exc:
            raise ValueError(f"Missing required camera term: {term}") from exc

    def _compute_world_to_camera(self, camera_obs: Any) -> np.ndarray:
        pos = camera_obs["rgb"].pose.xyz.cpu().numpy()[0]
        quat = camera_obs["rgb"].pose.quat.cpu().numpy()[0]

        t_cam_to_sim_world = np.eye(4, dtype=np.float64)
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        t_cam_to_sim_world[:3, :3] = rot.as_matrix().astype(
            np.float64, copy=False
        )
        t_cam_to_sim_world[:3, 3] = pos.astype(np.float64, copy=False)
        t_cam_to_world = t_cam_to_sim_world @ self._T_SIM_WORLD_TO_WORLD
        return np.linalg.inv(t_cam_to_world).astype(np.float64, copy=False)

    def build_action_sequence(
        self,
        output: "MultiArmManipulationOutput" | Any,
        *,
        device: torch.device | str,
        valid_action_step: int | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        actions = self._extract_action_tensor(output, device=device)
        actions = self._truncate_action_tensor(
            actions,
            valid_action_step=valid_action_step,
        )
        return self._actions_to_sequence(actions, device=device)

    @staticmethod
    def _extract_action_tensor(
        output: "MultiArmManipulationOutput" | Any,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        actions = torch.as_tensor(
            output.action,
            dtype=torch.float32,
            device=device,
        )
        if actions.ndim != 2:
            raise ValueError("Holobrain model output action must be 2D")
        return actions

    @staticmethod
    def _truncate_action_tensor(
        actions: torch.Tensor,
        *,
        valid_action_step: int | None,
    ) -> torch.Tensor:
        if valid_action_step is None:
            return actions
        return actions[:valid_action_step]

    def _actions_to_sequence(
        self,
        actions: torch.Tensor,
        *,
        device: torch.device | str,
    ) -> list[dict[str, torch.Tensor]]:
        sequence: list[dict[str, torch.Tensor]] = []
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

    @staticmethod
    def _to_homogeneous_intrinsic(intrinsic: np.ndarray) -> np.ndarray:
        intrinsic = np.asarray(intrinsic, dtype=np.float64)
        if intrinsic.shape == (4, 4):
            return intrinsic
        if intrinsic.shape == (3, 3):
            output = np.eye(4, dtype=np.float64)
            output[:3, :3] = intrinsic
            return output
        raise ValueError(
            f"Expected intrinsic shape (3, 3) or (4, 4), got {intrinsic.shape}"
        )
