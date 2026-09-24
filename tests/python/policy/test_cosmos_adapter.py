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

"""Tests for Cosmos canonical-input adaptation."""

import torch

from robo_orchard_sim.contracts.policy_binding import (
    CanonicalPolicyInput,
    ManipulatorBinding,
    PolicyBindingSchema,
)
from robo_orchard_sim.policy.action_layout import compile_action_layout
from robo_orchard_sim.policy.cosmos.adapter import CosmosAdapter


class _Sensor:
    def __init__(self) -> None:
        self.sensor_data = torch.zeros((1, 2, 2, 3), dtype=torch.uint8)


def _empty_instruction_observation() -> CanonicalPolicyInput:
    schema = PolicyBindingSchema(
        schema_version="1",
        embodiment_type="franka_panda",
        manipulator_slots={
            "single_arm": ManipulatorBinding(
                joint_position_obs_key="joint_position",
                arm_joint_name_specs=("panda_joint[1-7]",),
                gripper_joint_name_specs=(
                    "panda_finger_joint1",
                    "panda_finger_joint2",
                ),
            )
        },
    )
    camera = {"rgb": _Sensor()}
    return CanonicalPolicyInput(
        instruction="",
        cameras={
            "wrist_camera": camera,
            "ext1_camera": camera,
            "ext2_camera": camera,
        },
        manipulators={
            "single_arm": {
                "joint_position": torch.zeros((1, 9), dtype=torch.float32),
            }
        },
        action_layout=compile_action_layout(schema),
    )


def test_cosmos_adapter_empty_instruction_not_replaced_by_default() -> None:
    adapter = CosmosAdapter(default_instruction="lift the cube")

    request = adapter.build_request(_empty_instruction_observation())

    assert request["prompt"] == ""
