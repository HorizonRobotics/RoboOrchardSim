# ruff: noqa: E402
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
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robo_orchard_sim.contracts.policy_binding import (
    CameraBinding,
    ManipulatorBinding,
    PolicyBindingSchema,
    PolicyRequirement,
)
from robo_orchard_sim.policy.canonicalizer import (
    canonicalize_observations,
    validate_policy_compatibility,
)

_FRANKA_SCHEMA_PATH = (
    _REPO_ROOT
    / "robo_orchard_sim"
    / "orchard_env"
    / "embodiments"
    / "franka_panda"
    / "schema.py"
)


def _load_franka_schema_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "franka_panda_schema_for_test",
        _FRANKA_SCHEMA_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Sensor:
    def __init__(
        self,
        sensor_data: torch.Tensor,
        intrinsic_matrices: torch.Tensor | None = None,
        pose: object | None = None,
    ) -> None:
        self.sensor_data = sensor_data
        self.intrinsic_matrices = intrinsic_matrices
        self.pose = pose


def _build_dualarm_schema() -> PolicyBindingSchema:
    return PolicyBindingSchema(
        schema_version="1",
        embodiment_type="dualarm_piper",
        camera_slots={
            "left_wrist": CameraBinding(
                obs_term="left_hand_camera_term",
                rgb=True,
                intrinsic=True,
            ),
            "base": CameraBinding(
                obs_term="static_camera_term",
                rgb=True,
                intrinsic=True,
            ),
        },
        manipulator_slots={
            "left_arm": ManipulatorBinding(
                joint_position_obs_key="left_joint_position",
                gripper_position_obs_key="left_gripper_position",
                arm_joint_name_specs=("left_joint[1-6]",),
                gripper_joint_name_specs=("left_joint[7-8]",),
                gripper_decode_coupling="mirrored",
                gripper_policy_scale=2.0,
            )
        },
    )


def _build_franka_schema() -> PolicyBindingSchema:
    return PolicyBindingSchema(
        schema_version="1",
        embodiment_type="franka_panda",
        camera_slots={
            "wrist": CameraBinding(
                obs_term="hand_camera_term",
                rgb=True,
            ),
        },
        manipulator_slots={
            "single_arm": ManipulatorBinding(
                joint_position_obs_key="joint_position",
                arm_joint_name_specs=("panda_joint[1-7]",),
                gripper_joint_name_specs=(
                    "panda_finger_joint1",
                    "panda_finger_joint2",
                ),
                gripper_policy_representation="first_joint",
                gripper_decode_coupling="symmetric",
                gripper_policy_scale=2.0,
            )
        },
    )


def _build_raw_observation() -> dict[str, object]:
    rgb = torch.ones((1, 2, 2, 3), dtype=torch.uint8)
    intrinsic = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    sensor = {"rgb": _Sensor(rgb, intrinsic)}
    return {
        "/camera": {
            "left_hand_camera_term": sensor,
            "static_camera_term": sensor,
        },
        "/robot": {
            "left_joint_position": torch.ones((1, 7), dtype=torch.float32),
            "left_gripper_position": torch.tensor(
                [[0.1, 0.2]],
                dtype=torch.float32,
            ),
        },
    }


def _build_franka_raw_observation() -> dict[str, object]:
    rgb = torch.ones((1, 2, 2, 3), dtype=torch.uint8)
    sensor = {"rgb": _Sensor(rgb)}
    return {
        "/camera": {
            "hand_camera_term": sensor,
        },
        "/robot": {
            "joint_position": torch.tensor(
                [[1, 2, 3, 4, 5, 6, 7, 0.1, 0.1]],
                dtype=torch.float32,
            ),
        },
    }


def _build_franka_droid_raw_observation() -> dict[str, object]:
    rgb = torch.ones((1, 2, 2, 3), dtype=torch.uint8)
    depth = torch.ones((1, 2, 2), dtype=torch.float32)
    intrinsic = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    sensor = {
        "rgb": _Sensor(rgb, intrinsic_matrices=intrinsic, pose=object()),
        "depth": _Sensor(depth),
    }
    return {
        "/camera": {
            "wrist_camera_term": sensor,
            "ext1_camera_term": sensor,
            "ext2_camera_term": sensor,
        },
        "/robot": {
            "joint_position": torch.tensor(
                [[1, 2, 3, 4, 5, 6, 7, 0.1, 0.1]],
                dtype=torch.float32,
            ),
        },
    }


def test_canonicalize_observations_given_schema_maps_expected_slots() -> None:
    canonical = canonicalize_observations(
        observations=_build_raw_observation(),
        instruction="pick apple",
        schema=_build_dualarm_schema(),
    )

    assert canonical.instruction == "pick apple"
    assert canonical.cameras["left_wrist"]["rgb"].sensor_data.shape == (
        1,
        2,
        2,
        3,
    )
    assert canonical.manipulators["left_arm"]["joint_position"].shape == (
        1,
        7,
    )
    assert canonical.manipulators["left_arm"]["gripper_position"].shape == (
        1,
        2,
    )


def test_canonicalize_observations_single_arm_preserves_physical_joints() -> (
    None
):
    canonical = canonicalize_observations(
        observations=_build_franka_raw_observation(),
        instruction="pick apple",
        schema=_build_franka_schema(),
    )

    assert canonical.manipulators["single_arm"]["joint_position"].shape == (
        1,
        9,
    )
    torch.testing.assert_close(
        canonical.manipulators["single_arm"]["joint_position"],
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0.1, 0.1]], dtype=torch.float32),
    )


def test_franka_schema_droid_camera_terms_maps_expected_slots() -> None:
    franka_schema = _load_franka_schema_module()
    canonical = canonicalize_observations(
        observations=_build_franka_droid_raw_observation(),
        instruction="pick apple",
        schema=franka_schema.build_franka_panda_policy_binding_schema(
            "franka_panda"
        ),
    )

    assert "wrist_camera" in canonical.cameras
    assert "ext1_camera" in canonical.cameras
    assert "ext2_camera" in canonical.cameras


@pytest.mark.parametrize(
    ("metadata_name", "schema_kwargs"),
    [
        ("intrinsic", {"intrinsic": True}),
        ("pose", {"pose": True}),
    ],
)
def test_canonicalize_observations_missing_camera_metadata_raises_value_error(
    metadata_name: str,
    schema_kwargs: dict[str, bool],
) -> None:
    rgb = torch.ones((1, 2, 2, 3), dtype=torch.uint8)
    schema = PolicyBindingSchema(
        schema_version="1",
        embodiment_type="camera_metadata",
        camera_slots={
            "wrist": CameraBinding(
                obs_term="hand_camera_term",
                rgb=True,
                **schema_kwargs,
            )
        },
    )
    observations = {
        "/camera": {"hand_camera_term": {"rgb": _Sensor(rgb)}},
    }

    with pytest.raises(ValueError, match=metadata_name):
        canonicalize_observations(
            observations=observations,
            instruction=None,
            schema=schema,
        )


@pytest.mark.parametrize(
    ("observations", "message"),
    [
        ({}, "/camera"),
        ({"/camera": {}}, "hand_camera_term"),
        ({"/camera": {"hand_camera_term": {}}}, "rgb"),
    ],
)
def test_canonicalize_observations_missing_camera_input_raises_value_error(
    observations: dict[str, object],
    message: str,
) -> None:
    schema = PolicyBindingSchema(
        schema_version="1",
        embodiment_type="camera_input",
        camera_slots={
            "wrist": CameraBinding(
                obs_term="hand_camera_term",
                rgb=True,
            )
        },
    )

    with pytest.raises(ValueError, match=message):
        canonicalize_observations(
            observations=observations,
            instruction=None,
            schema=schema,
        )


def test_policy_compatibility_few_manipulators_raises_value_error() -> None:
    canonical = canonicalize_observations(
        observations=_build_raw_observation(),
        instruction="pick apple",
        schema=_build_dualarm_schema(),
    )
    requirement = PolicyRequirement(
        min_manipulator_count=2,
    )

    with pytest.raises(ValueError, match="at least 2 manipulators"):
        validate_policy_compatibility(
            canonical=canonical,
            requirement=requirement,
        )


def test_policy_compatibility_missing_modality_raises_value_error() -> None:
    canonical = canonicalize_observations(
        observations=_build_raw_observation(),
        instruction="pick apple",
        schema=_build_dualarm_schema(),
    )
    requirement = PolicyRequirement(
        required_camera_modalities=("rgb", "depth"),
    )

    with pytest.raises(ValueError, match="modality requirements"):
        validate_policy_compatibility(
            canonical=canonical,
            requirement=requirement,
        )
