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

# ruff: noqa: E402

from __future__ import annotations
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robo_orchard_sim.contracts.policy_binding import CanonicalPolicyInput
from robo_orchard_sim.policy.dummy import DummyPolicyCfg


def test_dummy_policy_act_given_canonical_input_returns_fixed_action():
    policy = DummyPolicyCfg()()
    observations = CanonicalPolicyInput()

    action = policy.act(observations)

    assert torch.equal(
        action["left_robot_joint_position"],
        torch.zeros((1, 6), dtype=torch.float32),
    )
