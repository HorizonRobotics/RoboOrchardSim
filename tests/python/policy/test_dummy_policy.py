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

from robo_orchard_sim.contracts.policy_binding import (
    CanonicalPolicyInput,
    ManipulatorBinding,
    PolicyBindingSchema,
)
from robo_orchard_sim.policy.action_layout import compile_action_layout
from robo_orchard_sim.policy.dummy import DummyPolicyCfg


def _single_arm_layout():
    return compile_action_layout(
        PolicyBindingSchema(
            schema_version="1",
            embodiment_type="panda_droid",
            manipulator_slots={
                "single_arm": ManipulatorBinding(
                    joint_position_obs_key="joint_position",
                    arm_joint_name_specs=("panda_joint[1-7]",),
                    gripper_joint_name_specs=("finger_joint",),
                )
            },
        )
    )


def test_dummy_policy_act_given_canonical_input_holds_joint_position():
    policy = DummyPolicyCfg()()
    joint_position = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    observations = CanonicalPolicyInput(
        manipulators={"single_arm": {"joint_position": joint_position}},
        action_layout=_single_arm_layout(),
    )

    action = policy.act(observations)

    assert action.joint_names == (
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "finger_joint",
    )
    assert torch.equal(action.values, joint_position)
