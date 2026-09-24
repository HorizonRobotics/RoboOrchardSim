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
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from robo_orchard_server.server.types import (
    ActionLayout,
    CameraObservation,
    JointAction,
    ManipulatorLayout,
    ManipulatorObservation,
    Observation,
    Pose,
)

if TYPE_CHECKING:
    from robo_orchard_lab.models.holobrain.processor import (
        MultiArmManipulationInput,
        MultiArmManipulationOutput,
    )


class _HolobrainCamera(TypedDict):
    """Camera fields required and validated by Holobrain."""

    rgb: np.ndarray
    depth: np.ndarray
    intrinsic_matrices: np.ndarray
    pose: Pose


class HolobrainAdapter:
    """Transform server observations to Holobrain inputs and joint actions."""

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

    _MODEL_CAMERA_SLOTS_BY_EMBODIMENT = {
        "franka_panda": {
            "wrist_camera": "wrist_camera",
            "ext1_camera": "ext1_camera",
            "ext2_camera": "ext2_camera",
        },
        "dualarm_piper": {
            "left": "left_wrist",
            "right": "right_wrist",
            "middle": "base",
        },
        "dualarm_piperx": {
            "left": "left_wrist",
            "right": "right_wrist",
            "middle": "base",
        },
    }

    def __init__(self, *, embodiment_type: str) -> None:
        try:
            self._t_sim_world_to_robot_base = (
                self._T_SIM_WORLD_TO_ROBOT_BASE_BY_EMBODIMENT[embodiment_type]
            )
            self._model_camera_slots = self._MODEL_CAMERA_SLOTS_BY_EMBODIMENT[
                embodiment_type
            ]
        except KeyError as exc:
            supported = tuple(self._T_SIM_WORLD_TO_ROBOT_BASE_BY_EMBODIMENT)
            raise ValueError(
                "Unsupported Holobrain embodiment_type "
                f"{embodiment_type!r}. Expected one of {supported}."
            ) from exc

    def build_model_input(
        self, obs: Observation
    ) -> "MultiArmManipulationInput":
        instruction = self._require_instruction(obs)
        layout = self._require_action_layout(obs)
        images, depths, intrinsics, t_world2cam = self._extract_camera_inputs(
            obs
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
            # The processor uses np.stack on this [T, D] array, but its
            # annotation only declares a list. Preserve the original format.
            history_joint_state=cast(Any, joint_state),
            instruction=instruction,
        )

    @staticmethod
    def _require_instruction(obs: Observation) -> str:
        instruction = obs["instruction"]
        if instruction is None:
            raise ValueError("Holobrain observation requires instruction")
        return instruction

    @staticmethod
    def _require_action_layout(obs: Observation) -> ActionLayout:
        layout = obs["action_layout"]
        order = layout["manipulator_order"]
        if not order or len(set(order)) != len(order):
            raise ValueError(
                "Holobrain layout requires unique manipulator slots"
            )
        if set(order) != set(layout["manipulators"]) or set(order) != set(
            obs["manipulators"]
        ):
            raise ValueError("Holobrain observation and layout slots differ")
        return layout

    def _extract_camera_inputs(
        self, obs: Observation
    ) -> tuple[dict, dict, dict, dict]:
        images = {}
        depths = {}
        intrinsics = {}
        t_world2cam = {}
        for model_key, camera_slot in self._model_camera_slots.items():
            if camera_slot not in obs["cameras"]:
                raise ValueError(
                    f"Holobrain requires camera slot {camera_slot!r} "
                    f"for model key {model_key!r}."
                )
            camera_obs = self._validate_camera_obs(
                obs["cameras"][camera_slot], camera_slot=camera_slot
            )
            rgb, depth, intrinsic, camera_t_world2cam = (
                self._extract_single_camera_input(camera_obs)
            )
            images[model_key] = [rgb]
            depths[model_key] = [depth]
            intrinsics[model_key] = intrinsic
            t_world2cam[model_key] = camera_t_world2cam
        return images, depths, intrinsics, t_world2cam

    def _extract_single_camera_input(
        self, camera_obs: _HolobrainCamera
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rgb = camera_obs["rgb"][0].astype(np.uint8)
        rgb = rgb[..., ::-1]
        depth = camera_obs["depth"][0]
        intrinsic = camera_obs["intrinsic_matrices"][0].astype(
            np.float64, copy=False
        )
        return (
            rgb,
            depth,
            self._to_homogeneous_intrinsic(intrinsic),
            self._compute_world_to_camera(camera_obs),
        )

    def _build_joint_state(
        self, obs: Observation, *, layout: ActionLayout
    ) -> np.ndarray:
        pieces = [
            self._build_manipulator_joint_state(
                obs["manipulators"][slot],
                manipulator=layout["manipulators"][slot],
            )
            for slot in layout["manipulator_order"]
        ]
        return np.concatenate(pieces, axis=0)[None, :]

    @classmethod
    def _build_manipulator_joint_state(
        cls,
        manipulator_obs: ManipulatorObservation,
        *,
        manipulator: ManipulatorLayout,
    ) -> np.ndarray:
        positions = manipulator_obs["joint_position"]
        if positions.ndim != 2 or positions.shape[0] != 1:
            raise ValueError("Holobrain joint_position must have shape [1, N]")
        joint_position = positions[0]
        arm_dim = len(manipulator["arm_joint_names"])
        if joint_position.shape[0] < arm_dim:
            raise ValueError(
                f"Manipulator {manipulator['slot']!r} joint_position has "
                f"{joint_position.shape[0]} dims, expected at least {arm_dim}."
            )
        state = [joint_position[:arm_dim]]
        if manipulator["gripper_joint_names"]:
            cls._gripper_policy_dim(manipulator)
            if "gripper_position" in manipulator_obs:
                gripper_positions = manipulator_obs["gripper_position"]
                if (
                    gripper_positions.ndim != 2
                    or gripper_positions.shape[0] != 1
                ):
                    raise ValueError(
                        "Holobrain gripper_position must have shape [1, N]"
                    )
                gripper = gripper_positions[0]
            else:
                gripper = joint_position[arm_dim:]
            if gripper.size != len(manipulator["gripper_joint_names"]):
                raise ValueError(
                    "Holobrain gripper positions do not match layout"
                )
            if manipulator["gripper_policy_representation"] == "first_joint":
                gripper = gripper[:1]
            state.append(
                np.asarray(
                    gripper * manipulator["gripper_policy_scale"],
                    dtype=np.float32,
                )
            )
        return np.concatenate(state, axis=0)

    @staticmethod
    def _validate_camera_obs(
        camera_obs: CameraObservation, *, camera_slot: str
    ) -> _HolobrainCamera:
        for modality in ("rgb", "depth", "intrinsic_matrices", "pose"):
            if camera_obs.get(modality) is None:
                raise ValueError(
                    f"Holobrain requires modality {modality!r} "
                    f"on camera slot {camera_slot!r}."
                )
        validated = cast(_HolobrainCamera, camera_obs)
        rgb = validated["rgb"]
        depth = validated["depth"]
        intrinsic = validated["intrinsic_matrices"]
        pose = validated["pose"]
        if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[-1] != 3:
            raise ValueError("Holobrain rgb must have shape [1, H, W, 3]")
        if depth.ndim not in (3, 4) or depth.shape[:3] != rgb.shape[:3]:
            raise ValueError(
                "Holobrain depth must match the rgb batch and size"
            )
        if intrinsic.ndim != 3 or intrinsic.shape[0] != 1:
            raise ValueError(
                "Holobrain intrinsics require a single camera batch"
            )
        if pose["xyz"].shape != (1, 3) or pose["quat"].shape != (1, 4):
            raise ValueError(
                "Holobrain camera pose requires xyz [1,3], quat [1,4]"
            )
        return validated

    def _compute_world_to_camera(
        self, camera_obs: _HolobrainCamera
    ) -> np.ndarray:
        pos = camera_obs["pose"]["xyz"][0]
        quat = camera_obs["pose"]["quat"][0]
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
        obs: Observation,
        *,
        device: torch.device | str = "cpu",
        valid_action_step: int | None = None,
    ) -> list[JointAction]:
        layout = self._require_action_layout(obs)
        actions = self._extract_action_tensor(output, device=device)
        actions = self._truncate_action_tensor(
            actions, valid_action_step=valid_action_step
        )
        actions = self._truncate_action_dims(actions, layout=layout)
        if not torch.isfinite(actions).all():
            raise ValueError("Holobrain model output action must be finite")
        return self._actions_to_sequence(actions, layout=layout)

    @staticmethod
    def _extract_action_tensor(
        output: "MultiArmManipulationOutput" | Any,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        actions = torch.as_tensor(
            output.action, dtype=torch.float32, device=device
        ).detach()
        if actions.ndim != 2:
            raise ValueError("Holobrain model output action must be 2D")
        if actions.shape[0] == 0:
            raise ValueError("Holobrain model output action must be nonempty")
        return actions

    @staticmethod
    def _truncate_action_tensor(
        actions: torch.Tensor, *, valid_action_step: int | None
    ) -> torch.Tensor:
        if valid_action_step is None:
            return actions
        if valid_action_step <= 0:
            raise ValueError("valid_action_step must be greater than 0")
        return actions[:valid_action_step]

    @classmethod
    def _truncate_action_dims(
        cls, actions: torch.Tensor, *, layout: ActionLayout
    ) -> torch.Tensor:
        expected_dim = sum(
            len(layout["manipulators"][slot]["arm_joint_names"])
            + cls._gripper_policy_dim(layout["manipulators"][slot])
            for slot in layout["manipulator_order"]
        )
        if actions.shape[1] < expected_dim:
            raise ValueError(
                "Holobrain model output action must provide at least "
                f"{expected_dim} dimensions"
            )
        return actions[:, :expected_dim]

    @staticmethod
    def _gripper_policy_dim(manipulator: ManipulatorLayout) -> int:
        joint_count = len(manipulator["gripper_joint_names"])
        if not joint_count:
            return 0
        scale = manipulator["gripper_policy_scale"]
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError("gripper_policy_scale must be finite and nonzero")
        representation = manipulator["gripper_policy_representation"]
        coupling = manipulator["gripper_decode_coupling"]
        if representation == "all_joints":
            if coupling != "identity":
                raise ValueError(
                    "all_joints requires identity decode coupling"
                )
            return joint_count
        if representation == "first_joint":
            if coupling not in ("symmetric", "mirrored", "identity"):
                raise ValueError(
                    f"Unsupported gripper decode coupling {coupling!r}"
                )
            if coupling == "identity" and joint_count != 1:
                raise ValueError(
                    "identity with first_joint requires one joint"
                )
            return 1
        raise ValueError(
            f"Unsupported gripper policy representation {representation!r}"
        )

    @staticmethod
    def _decode_gripper(
        gripper: torch.Tensor, manipulator: ManipulatorLayout
    ) -> torch.Tensor:
        physical = gripper / manipulator["gripper_policy_scale"]
        joint_count = len(manipulator["gripper_joint_names"])
        if manipulator["gripper_policy_representation"] == "all_joints":
            return physical.reshape(1, joint_count)
        value = physical[0]
        coupling = manipulator["gripper_decode_coupling"]
        if coupling == "identity":
            return value.reshape(1, 1)
        if coupling == "symmetric":
            return value.repeat(joint_count).reshape(1, joint_count)
        return torch.cat(
            (value.reshape(1), -value.repeat(joint_count - 1))
        ).reshape(1, joint_count)

    def _actions_to_sequence(
        self, actions: torch.Tensor, *, layout: ActionLayout
    ) -> list[JointAction]:
        sequence: list[JointAction] = []
        for step_idx in range(actions.shape[0]):
            cursor = 0
            pieces = []
            names: list[str] = []
            for slot in layout["manipulator_order"]:
                manipulator = layout["manipulators"][slot]
                arm_dim = len(manipulator["arm_joint_names"])
                pieces.append(
                    actions[step_idx : step_idx + 1, cursor : cursor + arm_dim]
                )
                names.extend(manipulator["arm_joint_names"])
                cursor += arm_dim
                if manipulator["gripper_joint_names"]:
                    gripper_dim = self._gripper_policy_dim(manipulator)
                    gripper = actions[step_idx, cursor : cursor + gripper_dim]
                    pieces.append(self._decode_gripper(gripper, manipulator))
                    names.extend(manipulator["gripper_joint_names"])
                    cursor += gripper_dim
            values = torch.cat(pieces, dim=1).cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError(
                    "Holobrain decoded joint targets must be finite"
                )
            sequence.append({"joint_names": names, "values": values})
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
