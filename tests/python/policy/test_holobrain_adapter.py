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

from robo_orchard_sim.contracts.policy_binding import (
    CameraBinding,
    CanonicalPolicyInput,
    ManipulatorBinding,
    PolicyBindingSchema,
)
from robo_orchard_sim.policy.action_layout import compile_action_layout
from robo_orchard_sim.policy.holobrain.adapter import HolobrainAdapter


def _build_dualarm_schema() -> PolicyBindingSchema:
    return PolicyBindingSchema(
        schema_version="1",
        embodiment_type="dualarm_piper",
        camera_slots={
            "left_wrist": CameraBinding(obs_term="left_hand_camera_term"),
            "base": CameraBinding(obs_term="static_camera_term"),
            "right_wrist": CameraBinding(obs_term="right_hand_camera_term"),
        },
        manipulator_slots={
            "left_arm": ManipulatorBinding(
                joint_position_obs_key="left_joint_position",
                arm_joint_name_specs=("left_joint[1-6]",),
                gripper_joint_name_specs=("left_joint[7-8]",),
                gripper_decode_coupling="mirrored",
                gripper_policy_scale=2.0,
            ),
            "right_arm": ManipulatorBinding(
                joint_position_obs_key="right_joint_position",
                arm_joint_name_specs=("right_joint[1-6]",),
                gripper_joint_name_specs=("right_joint[7-8]",),
                gripper_decode_coupling="mirrored",
                gripper_policy_scale=2.0,
            ),
        },
    )


def _build_reordered_dualarm_schema() -> PolicyBindingSchema:
    return PolicyBindingSchema(
        schema_version="1",
        embodiment_type="dualarm_piper",
        camera_slots={
            "right_wrist": CameraBinding(obs_term="right_hand_camera_term"),
            "base": CameraBinding(obs_term="static_camera_term"),
            "left_wrist": CameraBinding(obs_term="left_hand_camera_term"),
        },
        manipulator_slots={
            "left_arm": ManipulatorBinding(
                joint_position_obs_key="left_joint_position",
                arm_joint_name_specs=("left_joint[1-6]",),
                gripper_joint_name_specs=("left_joint[7-8]",),
                gripper_decode_coupling="mirrored",
                gripper_policy_scale=2.0,
            ),
            "right_arm": ManipulatorBinding(
                joint_position_obs_key="right_joint_position",
                arm_joint_name_specs=("right_joint[1-6]",),
                gripper_joint_name_specs=("right_joint[7-8]",),
                gripper_decode_coupling="mirrored",
                gripper_policy_scale=2.0,
            ),
        },
    )


def _build_franka_schema() -> PolicyBindingSchema:
    return PolicyBindingSchema(
        schema_version="1",
        embodiment_type="franka_panda",
        camera_slots={
            "wrist_camera": CameraBinding(obs_term="wrist_camera_term"),
            "ext1_camera": CameraBinding(obs_term="ext1_camera_term"),
            "ext2_camera": CameraBinding(obs_term="ext2_camera_term"),
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


def _build_obs() -> CanonicalPolicyInput:
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
    return CanonicalPolicyInput(
        cameras={
            "left_wrist": camera_output,
            "base": camera_output,
            "right_wrist": camera_output,
        },
        manipulators={
            "left_arm": {
                "joint_position": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.4]],
                    dtype=torch.float32,
                ),
            },
            "right_arm": {
                "joint_position": torch.tensor(
                    [[7.0, 8.0, 9.0, 10.0, 11.0, 12.0, -0.2]],
                    dtype=torch.float32,
                ),
            },
        },
        instruction="pick apple",
        action_layout=compile_action_layout(_build_dualarm_schema()),
    )


def _build_reordered_camera_obs() -> CanonicalPolicyInput:
    obs = _build_obs()
    return obs.model_copy(
        update={
            "action_layout": compile_action_layout(
                _build_reordered_dualarm_schema()
            )
        }
    )


def _build_single_arm_obs() -> CanonicalPolicyInput:
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
    return CanonicalPolicyInput(
        cameras={
            "wrist_camera": camera_output,
            "ext1_camera": camera_output,
            "ext2_camera": camera_output,
        },
        manipulators={
            "single_arm": {
                "joint_position": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.1]],
                    dtype=torch.float32,
                ),
            },
        },
        instruction="pick apple",
        action_layout=compile_action_layout(_build_franka_schema()),
    )


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
    adapter = HolobrainAdapter(embodiment_type="dualarm_piper")
    _install_fake_holobrain_processor(monkeypatch)

    model_input = adapter.build_model_input(_build_obs())

    assert model_input.instruction == "pick apple"
    assert set(model_input.image) == {"left", "middle", "right"}
    assert set(model_input.depth) == {"left", "middle", "right"}
    assert set(model_input.intrinsic) == {"left", "middle", "right"}
    assert set(model_input.t_world2cam) == {"left", "middle", "right"}
    assert model_input.history_joint_state.shape == (1, 14)
    assert np.isclose(model_input.history_joint_state[0, 6], 0.8)
    assert np.isclose(model_input.history_joint_state[0, 13], -0.4)


def test_build_model_input_reordered_schema_uses_adapter_camera_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HolobrainAdapter(embodiment_type="dualarm_piper")
    _install_fake_holobrain_processor(monkeypatch)

    model_input = adapter.build_model_input(_build_reordered_camera_obs())

    assert tuple(model_input.image) == ("left", "right", "middle")
    assert tuple(model_input.depth) == ("left", "right", "middle")
    assert tuple(model_input.intrinsic) == ("left", "right", "middle")
    assert tuple(model_input.t_world2cam) == ("left", "right", "middle")


def test_build_model_input_missing_instruction_raises_value_error() -> None:
    adapter = HolobrainAdapter(embodiment_type="dualarm_piper")
    obs = _build_obs()
    obs = obs.model_copy(update={"instruction": None})

    with pytest.raises(ValueError, match="instruction"):
        adapter.build_model_input(obs)


def test_build_model_input_single_arm_observation_returns_expected_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HolobrainAdapter(embodiment_type="franka_panda")
    _install_fake_holobrain_processor(monkeypatch)

    model_input = adapter.build_model_input(_build_single_arm_obs())

    assert model_input.instruction == "pick apple"
    assert set(model_input.image) == {
        "wrist_camera",
        "ext1_camera",
        "ext2_camera",
    }
    assert set(model_input.depth) == {
        "wrist_camera",
        "ext1_camera",
        "ext2_camera",
    }
    assert set(model_input.intrinsic) == {
        "wrist_camera",
        "ext1_camera",
        "ext2_camera",
    }
    assert set(model_input.t_world2cam) == {
        "wrist_camera",
        "ext1_camera",
        "ext2_camera",
    }
    assert model_input.history_joint_state.shape == (1, 8)
    assert np.isclose(model_input.history_joint_state[0, 7], 0.2)


def test_build_model_input_missing_depth_or_pose_raises_value_error() -> None:
    adapter = HolobrainAdapter(embodiment_type="dualarm_piper")
    obs = _build_obs()
    obs.cameras["left_wrist"] = {
        "rgb": obs.cameras["left_wrist"]["rgb"],
    }

    with pytest.raises(ValueError, match="requires modality"):
        adapter.build_model_input(obs)


def test_build_model_input_missing_camera_slot_raises_value_error() -> None:
    adapter = HolobrainAdapter(embodiment_type="dualarm_piper")
    obs = _build_obs()
    del obs.cameras["base"]

    with pytest.raises(ValueError, match="requires camera slot"):
        adapter.build_model_input(obs)


class _FakePipelineOutput:
    def __init__(self, action: torch.Tensor) -> None:
        self.action = action


def test_build_action_sequence_valid_action_step_truncates_result() -> None:
    adapter = HolobrainAdapter(embodiment_type="dualarm_piper")
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
        _build_obs(),
        device="cpu",
        valid_action_step=1,
    )

    assert len(actions) == 1
    assert actions[0].select("left_joint1")[0, 0].item() == 1.0


def test_build_action_sequence_gripper_controls_match_expected_result() -> (
    None
):
    adapter = HolobrainAdapter(embodiment_type="dualarm_piper")
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

    actions = adapter.build_action_sequence(output, _build_obs(), device="cpu")

    assert len(actions) == 1
    assert torch.equal(
        actions[0].select("left_joint7", "left_joint8"),
        torch.tensor([[0.2, -0.2]], dtype=torch.float32),
    )
    assert torch.equal(
        actions[0].select("right_joint7", "right_joint8"),
        torch.tensor([[-0.1, 0.1]], dtype=torch.float32),
    )
