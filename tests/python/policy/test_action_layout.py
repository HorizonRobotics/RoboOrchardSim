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
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robo_orchard_sim.contracts.policy_binding import (
    CameraBinding,
    ManipulatorBinding,
    PolicyBindingSchema,
)
from robo_orchard_sim.policy.action_layout import (
    CompiledActionLayout,
    compile_action_layout,
    validate_action_layout_compatibility,
)


def _build_schema() -> PolicyBindingSchema:
    return PolicyBindingSchema(
        schema_version="1",
        embodiment_type="dualarm_piper",
        camera_slots={
            "left_wrist": CameraBinding(
                obs_term="left_hand_camera_term",
                rgb=True,
                depth=True,
                intrinsic=True,
            ),
            "base": CameraBinding(obs_term="static_camera_term", rgb=True),
        },
        manipulator_slots={
            "left_arm": ManipulatorBinding(
                joint_position_obs_key="left_joint_position",
                gripper_position_obs_key="left_gripper_position",
                arm_joint_name_specs=("left_joint[1-6]",),
                gripper_joint_name_specs=("left_joint[7-8]",),
                gripper_policy_scale=2.0,
            )
        },
    )


def test_compile_layout_schema_binding_excluded_from_payload() -> None:
    layout = compile_action_layout(_build_schema())

    payload = layout.to_payload()

    assert payload == {
        "embodiment_type": "dualarm_piper",
        "schema_version": "1",
        "manipulator_order": ["left_arm"],
        "manipulators": {
            "left_arm": {
                "slot": "left_arm",
                "arm_joint_names": [
                    "left_joint1",
                    "left_joint2",
                    "left_joint3",
                    "left_joint4",
                    "left_joint5",
                    "left_joint6",
                ],
                "gripper_joint_names": ["left_joint7", "left_joint8"],
                "gripper_policy_representation": "first_joint",
                "gripper_decode_coupling": "symmetric",
                "gripper_policy_scale": 2.0,
            }
        },
    }


def test_compiled_layout_payload_roundtrip_preserves_runtime_layout() -> None:
    layout = compile_action_layout(_build_schema())

    rebuilt = CompiledActionLayout.from_payload(layout.to_payload())

    assert rebuilt == layout


def test_manipulator_action_spec_gripper_source_returns_scalar() -> None:
    spec = compile_action_layout(_build_schema()).manipulators["left_arm"]
    from_separate_obs = {
        "gripper_position": torch.tensor([[0.6, -0.2]], dtype=torch.float32),
    }

    separate_scalar = spec.extract_gripper_policy(
        from_separate_obs,
        joint_position=np.zeros(8, dtype=np.float32),
    )
    suffix_scalar = spec.extract_gripper_policy(
        {},
        joint_position=np.array([1, 2, 3, 4, 5, 6, 0.6, -0.2]),
    )

    assert np.allclose(separate_scalar, np.array([1.2], dtype=np.float32))
    assert np.allclose(suffix_scalar, np.array([1.2], dtype=np.float32))


def test_compile_layout_all_joints_policy_representation_updates_model_dim():
    schema = _build_schema()
    schema.manipulator_slots[
        "left_arm"
    ].gripper_policy_representation = "all_joints"
    schema.manipulator_slots["left_arm"].gripper_decode_coupling = "identity"

    spec = compile_action_layout(schema).manipulators["left_arm"]

    assert spec.model_dim == 8


@pytest.mark.parametrize(
    ("representation", "coupling", "message"),
    [
        ("all_joints", "symmetric", "all_joints"),
        ("first_joint", "identity", "identity"),
    ],
)
def test_compile_layout_invalid_gripper_policy_combo_raises_value_error(
    representation: str,
    coupling: str,
    message: str,
) -> None:
    schema = _build_schema()
    binding = schema.manipulator_slots["left_arm"]
    binding.gripper_policy_representation = representation
    binding.gripper_decode_coupling = coupling

    with pytest.raises(ValueError, match=message):
        compile_action_layout(schema)


@pytest.mark.parametrize(
    ("manipulator_observations", "message"),
    [
        ({}, "missing manipulator slots"),
        (
            {
                "left_arm": {},
                "extra_arm": {},
            },
            "unexpected manipulator slots",
        ),
    ],
)
def test_action_layout_compatibility_slot_mismatch_raises_value_error(
    manipulator_observations: dict[str, object],
    message: str,
) -> None:
    layout = compile_action_layout(_build_schema())

    with pytest.raises(ValueError, match=message):
        validate_action_layout_compatibility(
            manipulator_observations=manipulator_observations,
            layout=layout,
            context="test observation",
        )
