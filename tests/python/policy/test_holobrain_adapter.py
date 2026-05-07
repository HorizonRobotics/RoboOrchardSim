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
import types

import numpy as np
import pytest
import torch

from robo_orchard_sim.policy.holobrain.adapter import HolobrainAdapter


class _Sensor:
    def __init__(
        self,
        sensor_data: torch.Tensor,
        intrinsic_matrices: torch.Tensor | None = None,
        pose=None,
    ) -> None:
        self.sensor_data = sensor_data
        self.intrinsic_matrices = intrinsic_matrices
        self.pose = pose


class _Pose:
    def __init__(self, xyz: torch.Tensor, quat: torch.Tensor) -> None:
        self.xyz = xyz
        self.quat = quat


def _build_obs() -> dict:
    rgb = torch.arange(12, dtype=torch.uint8).reshape(1, 2, 2, 3)
    depth = torch.ones((1, 2, 2, 1), dtype=torch.float32)
    intrinsic = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    pose = _Pose(
        xyz=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    camera_output = {
        "rgb": _Sensor(rgb, intrinsic, pose=pose),
        "depth": _Sensor(depth),
    }
    return {
        "/camera": {
            "left_hand_camera_term": camera_output,
            "static_camera_term": camera_output,
            "right_hand_camera_term": camera_output,
        },
        "/robot": {
            "left_joint_position": torch.tensor(
                [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.4]],
                dtype=torch.float32,
            ),
            "right_joint_position": torch.tensor(
                [[7.0, 8.0, 9.0, 10.0, 11.0, 12.0, -0.2]],
                dtype=torch.float32,
            ),
        },
        "instruction": "pick apple",
    }


def _install_fake_holobrain_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    processor_module = types.ModuleType(
        "robo_orchard_lab.models.holobrain.processor"
    )

    class _FakeMultiArmManipulationInput:
        def __init__(
            self,
            *,
            image,
            depth,
            intrinsic,
            t_world2cam,
            history_joint_state,
            instruction,
        ) -> None:
            self.image = image
            self.depth = depth
            self.intrinsic = intrinsic
            self.t_world2cam = t_world2cam
            self.history_joint_state = history_joint_state
            self.instruction = instruction

    processor_module.MultiArmManipulationInput = _FakeMultiArmManipulationInput
    monkeypatch.setitem(
        sys.modules,
        "robo_orchard_lab.models.holobrain.processor",
        processor_module,
    )


def test_build_model_input_valid_observation_returns_expected_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HolobrainAdapter(joint_num=7)
    _install_fake_holobrain_processor(monkeypatch)

    model_input = adapter.build_model_input(_build_obs())

    assert model_input.instruction == "pick apple"
    assert set(model_input.image) == {"left", "middle", "right"}
    assert model_input.history_joint_state.shape == (1, 14)
    assert np.isclose(model_input.history_joint_state[0, 6], 0.8)
    assert np.isclose(model_input.history_joint_state[0, 13], -0.4)


def test_build_model_input_missing_instruction_raises_value_error() -> None:
    adapter = HolobrainAdapter(joint_num=7)
    obs = _build_obs()
    obs.pop("instruction")

    with pytest.raises(ValueError, match="instruction"):
        adapter.build_model_input(obs)


class _FakePipelineOutput:
    def __init__(self, action: torch.Tensor) -> None:
        self.action = action


def test_build_action_sequence_valid_action_step_truncates_result() -> None:
    adapter = HolobrainAdapter(joint_num=7)
    output = _FakePipelineOutput(
        action=torch.tensor(
            [
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    0.4,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    -0.2,
                ],
                [
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    0.6,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    -0.4,
                ],
            ],
            dtype=torch.float32,
        )
    )

    actions = adapter.build_action_sequence(
        output,
        device="cpu",
        valid_action_step=1,
    )

    assert len(actions) == 1
    assert actions[0]["left_robot_joint_position"][0, 0].item() == 1.0


def test_build_action_sequence_gripper_controls_match_expected_result() -> (
    None
):
    adapter = HolobrainAdapter(joint_num=7)
    output = _FakePipelineOutput(
        action=torch.tensor(
            [
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    0.4,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    -0.2,
                ],
            ],
            dtype=torch.float32,
        )
    )

    actions = adapter.build_action_sequence(output, device="cpu")

    assert len(actions) == 1
    assert torch.equal(
        actions[0]["left_robot_gripper_control"],
        torch.tensor([[0.2, -0.2]], dtype=torch.float32),
    )
    assert torch.equal(
        actions[0]["right_robot_gripper_control"],
        torch.tensor([[-0.1, 0.1]], dtype=torch.float32),
    )
