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

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robo_orchard_sim.policy.gripper_codec import (
    gripper_positions_to_policy_torch,
    policy_to_gripper_positions_torch,
)


def test_gripper_policy_first_joint_symmetric_roundtrip_restores_values():
    positions = torch.tensor([[0.04, 0.04]], dtype=torch.float32)

    policy = gripper_positions_to_policy_torch(
        positions,
        gripper_policy_representation="first_joint",
        gripper_policy_scale=2.0,
    )
    decoded = policy_to_gripper_positions_torch(
        policy[0],
        gripper_policy_representation="first_joint",
        gripper_decode_coupling="symmetric",
        gripper_policy_scale=2.0,
        joint_count=2,
    )

    assert torch.allclose(policy, torch.tensor([[0.08]], dtype=torch.float32))
    assert torch.allclose(decoded, positions)


def test_gripper_policy_first_joint_mirrored_roundtrip_restores_values():
    positions = torch.tensor([[0.04, -0.04]], dtype=torch.float32)

    policy = gripper_positions_to_policy_torch(
        positions,
        gripper_policy_representation="first_joint",
        gripper_policy_scale=2.0,
    )
    decoded = policy_to_gripper_positions_torch(
        policy[0],
        gripper_policy_representation="first_joint",
        gripper_decode_coupling="mirrored",
        gripper_policy_scale=2.0,
        joint_count=2,
    )

    assert torch.allclose(policy, torch.tensor([[0.08]], dtype=torch.float32))
    assert torch.allclose(decoded, positions)


def test_gripper_policy_all_joints_identity_roundtrip_restores_values():
    positions = torch.tensor([[0.04, -0.02, 0.01]], dtype=torch.float32)

    policy = gripper_positions_to_policy_torch(
        positions,
        gripper_policy_representation="all_joints",
        gripper_policy_scale=10.0,
    )
    decoded = policy_to_gripper_positions_torch(
        policy[0],
        gripper_policy_representation="all_joints",
        gripper_decode_coupling="identity",
        gripper_policy_scale=10.0,
        joint_count=3,
    )

    assert torch.allclose(
        policy,
        torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float32),
    )
    assert torch.allclose(decoded, positions)


def test_gripper_policy_all_joints_symmetric_raises_value_error():
    with pytest.raises(ValueError, match="all_joints"):
        policy_to_gripper_positions_torch(
            torch.tensor([0.1, 0.2], dtype=torch.float32),
            gripper_policy_representation="all_joints",
            gripper_decode_coupling="symmetric",
            gripper_policy_scale=1.0,
            joint_count=2,
        )
