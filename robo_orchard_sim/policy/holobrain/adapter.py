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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from robo_orchard_sim.orchard_env.joint_command import UnifiedJointCommand
from robo_orchard_sim.policy.action_layout import (
    CompiledActionLayout,
    ManipulatorActionSpec,
    validate_action_layout_compatibility,
)
from robo_orchard_sim.policy.gripper_codec import (
    policy_to_gripper_positions_torch,
)
from robo_orchard_sim.policy.schema import CanonicalPolicyInput

if TYPE_CHECKING:
    from robo_orchard_lab.models.holobrain.processor import (
        MultiArmManipulationInput,
        MultiArmManipulationOutput,
    )


@dataclass(frozen=True)
class _HolobrainCameraSpec:
    model_key: str
    single_arm_slot: str | None
    dual_arm_slot: str | None
    required_for_single_arm: bool
    required_for_dual_arm: bool

    def slot_for_arm_count(self, arm_count: int) -> str | None:
        if arm_count == 1:
            return self.single_arm_slot
        if arm_count == 2:
            return self.dual_arm_slot
        raise ValueError(
            f"Holobrain supports only 1 or 2 arms, got {arm_count}."
        )

    def required_for_arm_count(self, arm_count: int) -> bool:
        if arm_count == 1:
            return self.required_for_single_arm
        if arm_count == 2:
            return self.required_for_dual_arm
        raise ValueError(
            f"Holobrain supports only 1 or 2 arms, got {arm_count}."
        )


# This order determines the input camera order.
_HOLOBRAIN_CAMERA_SPECS = (
    _HolobrainCameraSpec(
        model_key="left",
        single_arm_slot="wrist",
        dual_arm_slot="left_wrist",
        required_for_single_arm=True,
        required_for_dual_arm=True,
    ),
    _HolobrainCameraSpec(
        model_key="right",
        single_arm_slot=None,
        dual_arm_slot="right_wrist",
        required_for_single_arm=False,
        required_for_dual_arm=True,
    ),
    _HolobrainCameraSpec(
        model_key="middle",
        single_arm_slot="base",
        dual_arm_slot="base",
        required_for_single_arm=True,
        required_for_dual_arm=True,
    ),
)


class HolobrainAdapter:
    """Transforms sim observations to Holobrain inputs and back."""

    _T_SIM_WORLD_TO_ROBOT_BASE_BY_EMBODIMENT = {
        "dualarm_piperx": np.array(
            [[1, 0, 0, 0.3], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        ),
        "dualarm_piper": np.array(
            [[1, 0, 0, 0.3], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        ),
        "franka_panda": np.eye(4, dtype=np.float64),
    }

    def __init__(self, *, embodiment_type: str) -> None:
        try:
            self._t_sim_world_to_robot_base = (
                self._T_SIM_WORLD_TO_ROBOT_BASE_BY_EMBODIMENT[embodiment_type]
            )
        except KeyError as exc:
            supported = tuple(self._T_SIM_WORLD_TO_ROBOT_BASE_BY_EMBODIMENT)
            raise ValueError(
                "Unsupported Holobrain embodiment_type "
                f"{embodiment_type!r}. Expected one of {supported}."
            ) from exc

    @classmethod
    def required_observation_fields(cls) -> dict[str, Any]:
        return {
            "camera_terms": ["left", "right", "middle"],
            "include_rgb": True,
            "include_depth": True,
            "include_intrinsic": True,
            "include_pose": True,
            "robot_keys": [
                "left_joint_position",
                "right_joint_position",
            ],
        }

    def build_model_input(
        self,
        obs: CanonicalPolicyInput,
    ) -> "MultiArmManipulationInput":
        instruction = self._require_instruction(obs)
        layout = self._require_action_layout(obs)
        validate_action_layout_compatibility(
            manipulator_observations=obs.manipulators,
            layout=layout,
            context="Holobrain observation",
        )
        images, depths, intrinsics, t_world2cam = self._extract_camera_inputs(
            obs,
            layout=layout,
        )
        joint_state = self._build_joint_state(obs, layout=layout)

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
    def _require_instruction(obs: CanonicalPolicyInput) -> str:
        instruction = obs.instruction
        if not instruction:
            raise ValueError("Holobrain observation requires instruction")
        return instruction

    @staticmethod
    def _require_action_layout(
        obs: CanonicalPolicyInput,
    ) -> CompiledActionLayout:
        layout = obs.action_layout
        if not isinstance(layout, CompiledActionLayout):
            raise ValueError(
                "Holobrain observation requires a compiled action layout"
            )
        return layout

    def _extract_camera_inputs(
        self,
        obs: CanonicalPolicyInput,
        *,
        layout: CompiledActionLayout,
    ) -> tuple[dict, dict, dict, dict]:
        images = {}
        depths = {}
        intrinsics = {}
        t_world2cam = {}
        arm_count = len(layout.manipulator_order)

        for camera_spec in _HOLOBRAIN_CAMERA_SPECS:
            camera_slot = camera_spec.slot_for_arm_count(arm_count)
            if camera_slot is None:
                continue
            if camera_slot not in obs.cameras:
                raise ValueError(
                    f"Holobrain requires camera slot {camera_slot!r}."
                )
            camera_obs = obs.cameras[camera_slot]
            self._validate_camera_obs(camera_obs, camera_slot=camera_slot)
            rgb, depth, intrinsic, camera_t_world2cam = (
                self._extract_single_camera_input(camera_obs)
            )
            camera_key = camera_spec.model_key
            images[camera_key] = [rgb]
            depths[camera_key] = [depth]
            intrinsics[camera_key] = intrinsic
            t_world2cam[camera_key] = camera_t_world2cam

        return images, depths, intrinsics, t_world2cam

    def _extract_single_camera_input(
        self,
        camera_obs: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rgb_sensor = camera_obs["rgb"]
        depth_sensor = camera_obs["depth"]

        rgb = rgb_sensor.sensor_data[0].cpu().numpy().astype(np.uint8)
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
    def _validate_camera_obs(
        camera_obs: dict[str, Any],
        *,
        camera_slot: str,
    ) -> None:
        required_modalities = ("rgb", "depth", "intrinsic", "pose")
        for modality in required_modalities:
            if modality in ("intrinsic", "pose"):
                rgb_sensor = camera_obs.get("rgb")
                if rgb_sensor is None:
                    raise ValueError(
                        "Holobrain requires modality 'rgb' on camera "
                        f"slot {camera_slot!r}."
                    )
                attr_name = (
                    "intrinsic_matrices" if modality == "intrinsic" else "pose"
                )
                if getattr(rgb_sensor, attr_name, None) is None:
                    raise ValueError(
                        "Holobrain requires modality "
                        f"{modality!r} on camera slot {camera_slot!r}."
                    )
                continue
            if modality not in camera_obs:
                raise ValueError(
                    "Holobrain requires modality "
                    f"{modality!r} on camera slot {camera_slot!r}."
                )

    def _compute_world_to_camera(self, camera_obs: Any) -> np.ndarray:
        pos = camera_obs["rgb"].pose.xyz.cpu().numpy()[0]
        quat = camera_obs["rgb"].pose.quat.cpu().numpy()[0]

        t_cam_to_sim_world = np.eye(4, dtype=np.float64)
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        t_cam_to_sim_world[:3, :3] = rot.as_matrix().astype(
            np.float64, copy=False
        )
        t_cam_to_sim_world[:3, 3] = pos.astype(np.float64, copy=False)
        t_cam_to_robot_base = (
            t_cam_to_sim_world @ self._t_sim_world_to_robot_base
        )
        return np.linalg.inv(t_cam_to_robot_base).astype(
            np.float64, copy=False
        )

    def build_action_sequence(
        self,
        output: "MultiArmManipulationOutput" | Any,
        obs: CanonicalPolicyInput,
        *,
        device: torch.device | str,
        valid_action_step: int | None = None,
    ) -> list[UnifiedJointCommand]:
        layout = self._require_action_layout(obs)
        actions = self._extract_action_tensor(output, device=device)
        actions = self._truncate_action_tensor(
            actions,
            valid_action_step=valid_action_step,
        )
        actions = self._truncate_action_dims(actions, layout=layout)
        return self._actions_to_sequence(actions, layout=layout)

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
                "Holobrain model output action must provide at least "
                f"{expected_dim} dimensions"
            )
        return actions[:, :expected_dim]

    def _actions_to_sequence(
        self,
        actions: torch.Tensor,
        *,
        layout: CompiledActionLayout,
    ) -> list[UnifiedJointCommand]:
        sequence: list[UnifiedJointCommand] = []
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
