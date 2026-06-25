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
from typing import Any

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robo_orchard_sim.contracts.joint_command import UnifiedJointCommand
from robo_orchard_sim.contracts.policy_binding import (
    CameraBinding,
    CanonicalPolicyInput,
    ManipulatorBinding,
    PolicyBindingSchema,
)
from robo_orchard_sim.policy.action_layout import compile_action_layout
from robo_orchard_sim.policy.canonicalizer import canonicalize_observations
from robo_orchard_sim.policy.factory import create_policy_from_model_cfg
from robo_orchard_sim.policy.motus.adapter import (
    DEFAULT_CAMERA_MAPPING,
    MotusAdapter,
)
from robo_orchard_sim.policy.motus.policy import MotusPolicy, MotusPolicyCfg


class _Sensor:
    def __init__(self, sensor_data: torch.Tensor) -> None:
        self.sensor_data = sensor_data


class _FakeMotusClient:
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self.chunks = chunks
        self.requests: list[dict[str, Any]] = []
        self.reset_calls = 0
        self.closed = False

    def infer(self, obs: dict[str, Any]) -> np.ndarray:
        self.requests.append(obs)
        return self.chunks.pop(0)

    def reset(self, reset_info: dict[str, Any] | None = None) -> None:
        del reset_info
        self.reset_calls += 1

    def close(self) -> None:
        self.closed = True


def _build_dualarm_schema() -> PolicyBindingSchema:
    return PolicyBindingSchema(
        schema_version="1",
        embodiment_type="dualarm_piper",
        camera_slots={
            "left_wrist": CameraBinding(obs_term="left_hand_camera_term"),
            "right_wrist": CameraBinding(obs_term="right_hand_camera_term"),
            "base": CameraBinding(obs_term="static_camera_term"),
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
            "wrist": CameraBinding(obs_term="hand_camera_term"),
            "base": CameraBinding(obs_term="static_camera_term"),
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


def _image(value: int) -> torch.Tensor:
    return torch.full((1, 2, 3, 3), value, dtype=torch.uint8)


def _build_dualarm_obs() -> CanonicalPolicyInput:
    return CanonicalPolicyInput(
        cameras={
            "base": {"rgb": _Sensor(_image(10))},
            "left_wrist": {"rgb": _Sensor(_image(20))},
            "right_wrist": {"rgb": _Sensor(_image(30))},
        },
        manipulators={
            "left_arm": {
                "joint_position": torch.tensor(
                    [[1, 2, 3, 4, 5, 6, 0.2, -0.2]],
                    dtype=torch.float32,
                )
            },
            "right_arm": {
                "joint_position": torch.tensor(
                    [[7, 8, 9, 10, 11, 12, 0.3, -0.3]],
                    dtype=torch.float32,
                )
            },
        },
        instruction="pick apple",
        action_layout=compile_action_layout(_build_dualarm_schema()),
    )


def _build_franka_obs() -> CanonicalPolicyInput:
    return CanonicalPolicyInput(
        cameras={
            "base": {"rgb": _Sensor(_image(40))},
            "wrist": {"rgb": _Sensor(_image(50))},
        },
        manipulators={
            "single_arm": {
                "joint_position": torch.tensor(
                    [[1, 2, 3, 4, 5, 6, 7, 0.1, 0.1]],
                    dtype=torch.float32,
                )
            },
        },
        instruction="pick cube",
        action_layout=compile_action_layout(_build_franka_schema()),
    )


def _build_dualarm_raw_observation() -> dict[str, object]:
    return {
        "/camera": {
            "static_camera_term": {"rgb": _Sensor(_image(10))},
            "left_hand_camera_term": {"rgb": _Sensor(_image(20))},
            "right_hand_camera_term": {"rgb": _Sensor(_image(30))},
        },
        "/robot": {
            "left_joint_position": torch.tensor(
                [[1, 2, 3, 4, 5, 6, 0.2, -0.2]],
                dtype=torch.float32,
            ),
            "right_joint_position": torch.tensor(
                [[7, 8, 9, 10, 11, 12, 0.3, -0.3]],
                dtype=torch.float32,
            ),
        },
    }


def test_motus_adapter_builds_dualarm_request_in_layout_order() -> None:
    adapter = MotusAdapter(camera_mapping=DEFAULT_CAMERA_MAPPING)

    request = adapter.build_request(_build_dualarm_obs(), session_id="ep-1")

    assert request["prompt"] == "pick apple"
    assert request["session_id"] == "ep-1"
    assert np.array_equal(
        request["observation/joint_position"],
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32),
    )
    assert np.allclose(
        request["observation/gripper_position"],
        np.array([0.4, 0.6], dtype=np.float32),
    )
    assert request["observation/exterior_image_0_left"].shape == (2, 3, 3)
    assert request["observation/exterior_image_0_left"][0, 0, 0] == 10
    assert request["observation/exterior_image_1_left"][0, 0, 0] == 20
    assert request["observation/wrist_image_left"][0, 0, 0] == 30


def test_motus_adapter_rgb_image_converts_to_bgr_request() -> None:
    adapter = MotusAdapter(camera_mapping=DEFAULT_CAMERA_MAPPING)
    obs = _build_dualarm_obs()
    rgb_image = torch.tensor(
        [[[[1, 2, 3], [4, 5, 6]]]],
        dtype=torch.uint8,
    )
    obs.cameras["base"]["rgb"] = _Sensor(rgb_image)

    request = adapter.build_request(obs, session_id="ep-1")

    assert np.array_equal(
        request["observation/exterior_image_0_left"],
        np.array([[[3, 2, 1], [6, 5, 4]]], dtype=np.uint8),
    )


def test_motus_adapter_canonicalized_physical_gripper_encodes_once() -> None:
    adapter = MotusAdapter(camera_mapping=DEFAULT_CAMERA_MAPPING)
    obs = canonicalize_observations(
        observations=_build_dualarm_raw_observation(),
        instruction="pick apple",
        schema=_build_dualarm_schema(),
    )

    request = adapter.build_request(obs, session_id="ep-1")

    assert np.allclose(
        request["observation/gripper_position"],
        np.array([0.4, 0.6], dtype=np.float32),
    )


def test_motus_adapter_duplicates_franka_wrist_for_right_hand_camera() -> None:
    adapter = MotusAdapter(
        camera_mapping={
            "static_camera": "base",
            "left_hand_camera": "wrist",
            "right_hand_camera": "duplicate:left_hand_camera",
        }
    )

    request = adapter.build_request(_build_franka_obs(), session_id="ep-1")

    assert np.array_equal(
        request["observation/joint_position"],
        np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32),
    )
    assert np.allclose(
        request["observation/gripper_position"],
        np.array([0.2], dtype=np.float32),
    )
    assert request["observation/exterior_image_0_left"][0, 0, 0] == 40
    assert request["observation/exterior_image_1_left"][0, 0, 0] == 50
    assert request["observation/wrist_image_left"][0, 0, 0] == 50


def test_motus_adapter_decodes_dualarm_action_chunk() -> None:
    adapter = MotusAdapter(camera_mapping=DEFAULT_CAMERA_MAPPING)
    actions = np.array(
        [
            [1, 2, 3, 4, 5, 6, 0.4, 7, 8, 9, 10, 11, 12, 0.6],
            [2, 3, 4, 5, 6, 7, 0.8, 8, 9, 10, 11, 12, 13, 1.0],
        ],
        dtype=np.float32,
    )

    sequence = adapter.build_action_sequence(
        actions,
        _build_dualarm_obs(),
        device="cpu",
    )

    assert len(sequence) == 2
    assert isinstance(sequence[0], UnifiedJointCommand)
    assert sequence[0].joint_names == (
        "left_joint1",
        "left_joint2",
        "left_joint3",
        "left_joint4",
        "left_joint5",
        "left_joint6",
        "left_joint7",
        "left_joint8",
        "right_joint1",
        "right_joint2",
        "right_joint3",
        "right_joint4",
        "right_joint5",
        "right_joint6",
        "right_joint7",
        "right_joint8",
    )
    assert torch.allclose(
        sequence[0].values,
        torch.tensor(
            [[1, 2, 3, 4, 5, 6, 0.2, -0.2, 7, 8, 9, 10, 11, 12, 0.3, -0.3]],
            dtype=torch.float32,
        ),
    )


def test_motus_policy_caches_action_chunk_and_resets_remote_policy() -> None:
    fake_client = _FakeMotusClient(
        [
            np.array(
                [
                    [1, 2, 3, 4, 5, 6, 0.4, 7, 8, 9, 10, 11, 12, 0.6],
                    [2, 3, 4, 5, 6, 7, 0.8, 8, 9, 10, 11, 12, 13, 1.0],
                ],
                dtype=np.float32,
            )
        ]
    )
    policy = MotusPolicy(
        cfg=MotusPolicyCfg(host="127.0.0.1", port=8000),
        client=fake_client,
    )
    obs = _build_dualarm_obs()

    first = policy.act(obs)
    second = policy.act(obs)
    policy.reset()

    assert len(fake_client.requests) == 1
    assert first.select("left_joint[1-6]").shape == (1, 6)
    assert torch.allclose(second.select("left_joint1"), torch.tensor([[2.0]]))
    assert fake_client.reset_calls == 1


def test_motus_adapter_rejects_action_dim_mismatch() -> None:
    adapter = MotusAdapter(camera_mapping=DEFAULT_CAMERA_MAPPING)

    with pytest.raises(ValueError, match="Expected Motus actions"):
        adapter.build_action_sequence(
            np.zeros((2, 13), dtype=np.float32),
            _build_dualarm_obs(),
            device="cpu",
        )


def test_create_policy_from_model_cfg_given_motus_returns_cfg() -> None:
    policy_cfg = create_policy_from_model_cfg(
        {
            "policy": "motus",
            "host": "127.0.0.1",
            "port": 9000,
            "logging_tag": "motus-eval",
            "valid_action_step": 4,
            "camera_mapping": {
                "static_camera": "base",
                "left_hand_camera": "wrist",
                "right_hand_camera": "duplicate:left_hand_camera",
            },
        }
    )

    assert isinstance(policy_cfg, MotusPolicyCfg)
    assert policy_cfg.host == "127.0.0.1"
    assert policy_cfg.port == 9000
    assert policy_cfg.logging_tag == "motus-eval"
    assert policy_cfg.valid_action_step == 4
    assert policy_cfg.camera_mapping["right_hand_camera"] == (
        "duplicate:left_hand_camera"
    )
