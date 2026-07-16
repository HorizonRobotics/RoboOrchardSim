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

# INTERNAL

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

from robo_orchard_sim.contracts.policy_binding import (
    CameraBinding,
    CanonicalPolicyInput,
    ManipulatorBinding,
    PolicyBindingSchema,
)
from robo_orchard_sim.policy.action_layout import compile_action_layout
from robo_orchard_sim.policy.factory import create_policy_from_model_cfg
from robo_orchard_sim.policy.groot.adapter import GrootAdapter, GrootArmSpec
from robo_orchard_sim.policy.groot.policy import GrootPolicy, GrootPolicyCfg


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


class _Sensor:
    def __init__(self, sensor_data: torch.Tensor) -> None:
        self.sensor_data = sensor_data


def _build_obs(batch_size: int = 1) -> CanonicalPolicyInput:
    rgb = torch.tensor(
        [[[[10, 20, 30], [40, 50, 60]], [[1, 2, 3], [4, 5, 6]]]],
        dtype=torch.uint8,
    ).repeat(batch_size, 1, 1, 1)
    camera_obs = {"rgb": _Sensor(rgb)}
    return CanonicalPolicyInput(
        cameras={
            "left_wrist": camera_obs,
            "right_wrist": camera_obs,
            "base": camera_obs,
        },
        manipulators={
            "left_arm": {
                "joint_position": torch.tensor(
                    [[1, 2, 3, 4, 5, 6, 0.2]],
                    dtype=torch.float32,
                ).repeat(batch_size, 1),
            },
            "right_arm": {
                "joint_position": torch.tensor(
                    [[7, 8, 9, 10, 11, 12, -0.4]],
                    dtype=torch.float32,
                ).repeat(batch_size, 1),
            },
        },
        instruction="pick apple",
        action_layout=compile_action_layout(_build_dualarm_schema()),
    )


def _obs_with_fill(fill: int) -> CanonicalPolicyInput:
    obs = _build_obs()
    rgb = torch.full((1, 2, 2, 3), fill, dtype=torch.uint8)
    sensor = {"rgb": _Sensor(rgb)}
    obs.cameras["base"] = sensor
    obs.cameras["left_wrist"] = sensor
    obs.cameras["right_wrist"] = sensor
    return obs


def _action_chunk(offset: float = 0.0, horizon: int = 2) -> dict[str, Any]:
    left_arm = np.zeros((1, horizon, 6), dtype=np.float32)
    right_arm = np.zeros((1, horizon, 6), dtype=np.float32)
    for step in range(horizon):
        left_arm[0, step, 0] = offset + step
    return {
        "left_arm": left_arm,
        "left_gripper": np.full((1, horizon, 1), 0.4, dtype=np.float32),
        "right_arm": right_arm,
        "right_gripper": np.full((1, horizon, 1), -0.4, dtype=np.float32),
    }


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0
        self.reset_called = False
        self.last_observation: dict[str, Any] | None = None

    def get_action(
        self, observation: dict[str, Any], options: Any = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls += 1
        self.last_observation = observation
        return _action_chunk(offset=self.calls * 100.0), {}

    def reset(self, options: Any = None) -> None:
        self.reset_called = True


def _build_policy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    open_loop_horizon: int | None = 2,
) -> tuple[GrootPolicy, _FakeClient]:
    fake = _FakeClient()
    # Mock at the network boundary (the ZMQ client), not the policy itself.
    monkeypatch.setattr(
        "robo_orchard_sim.policy.groot.policy.GrootZmqClient",
        lambda **kwargs: fake,
    )
    policy = GrootPolicy(
        cfg=GrootPolicyCfg(open_loop_horizon=open_loop_horizon)
    )
    return policy, fake


def test_groot_policy_requirement_declares_contract() -> None:
    requirement = GrootPolicy.policy_requirement()

    assert requirement.required_camera_modalities == ("rgb",)
    assert requirement.min_camera_count == 1
    assert requirement.min_manipulator_count == 1
    assert requirement.require_instruction is False


def test_groot_adapter_build_model_input_returns_native_contract() -> None:
    adapter = GrootAdapter()

    model_input = adapter.build_model_input(_build_obs())

    assert set(model_input) == {"video", "state", "language"}
    assert set(model_input["video"]) == {
        "static_camera",
        "left_hand_camera",
        "right_hand_camera",
    }
    assert model_input["video"]["static_camera"].shape == (1, 1, 2, 2, 3)
    assert model_input["video"]["static_camera"].dtype == np.uint8
    assert set(model_input["state"]) == {
        "left_arm",
        "left_gripper",
        "right_arm",
        "right_gripper",
    }
    assert model_input["state"]["left_arm"].shape == (1, 1, 6)
    assert model_input["state"]["left_gripper"].shape == (1, 1, 1)
    assert model_input["state"]["left_arm"].dtype == np.float32
    np.testing.assert_allclose(
        model_input["state"]["left_arm"][0, 0],
        np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
    )
    np.testing.assert_allclose(
        model_input["state"]["left_gripper"][0, 0],
        np.array([0.4], dtype=np.float32),
    )
    np.testing.assert_allclose(
        model_input["state"]["right_gripper"][0, 0],
        np.array([-0.8], dtype=np.float32),
    )
    assert model_input["language"] == {
        "annotation.human.task_description": [["pick apple"]]
    }


def test_groot_adapter_build_model_input_missing_instruction_raises() -> None:
    adapter = GrootAdapter()
    obs = _build_obs().model_copy(update={"instruction": None})

    with pytest.raises(ValueError, match="requires an instruction"):
        adapter.build_model_input(obs)


def test_groot_adapter_default_instruction_used_when_obs_missing() -> None:
    adapter = GrootAdapter(default_instruction="lift the cube")
    obs = _build_obs().model_copy(update={"instruction": None})

    model_input = adapter.build_model_input(obs)

    assert model_input["language"] == {
        "annotation.human.task_description": [["lift the cube"]]
    }


def test_groot_adapter_missing_camera_slot_raises() -> None:
    adapter = GrootAdapter()
    obs = _build_obs()
    del obs.cameras["base"]

    with pytest.raises(ValueError, match="requires canonical camera slot"):
        adapter.build_model_input(obs)


def test_groot_adapter_relative_arm_adds_current_state() -> None:
    adapter = GrootAdapter(
        arm_specs=(
            GrootArmSpec("left_arm", "left_arm", "left_gripper", True),
            GrootArmSpec("right_arm", "right_arm", "right_gripper", True),
        )
    )
    action = _action_chunk(offset=10.0, horizon=2)

    sequence = adapter.build_action_sequence(
        action,
        _build_obs(),
        device="cpu",
        open_loop_horizon=None,
    )

    assert len(sequence) == 2
    # left_arm joint1 = current (1) + delta (offset + step)
    torch.testing.assert_close(
        sequence[0].select("left_joint1"),
        torch.tensor([[11.0]]),
    )
    torch.testing.assert_close(
        sequence[1].select("left_joint1"),
        torch.tensor([[12.0]]),
    )
    # right_arm has zero delta -> equals current joint state
    torch.testing.assert_close(
        sequence[0].select("right_joint[1-6]"),
        torch.tensor([[7, 8, 9, 10, 11, 12]], dtype=torch.float32),
    )
    # left gripper policy 0.4 -> physical 0.2 -> mirrored [0.2, -0.2]
    torch.testing.assert_close(
        sequence[0].select("left_joint7", "left_joint8"),
        torch.tensor([[0.2, -0.2]]),
    )
    # right gripper policy -0.4 -> physical -0.2 -> mirrored [-0.2, 0.2]
    torch.testing.assert_close(
        sequence[0].select("right_joint7", "right_joint8"),
        torch.tensor([[-0.2, 0.2]]),
    )


def test_groot_adapter_absolute_arm_skips_current_state() -> None:
    adapter = GrootAdapter(
        arm_specs=(
            GrootArmSpec("left_arm", "left_arm", "left_gripper", False),
            GrootArmSpec("right_arm", "right_arm", "right_gripper", False),
        )
    )
    action = _action_chunk(offset=10.0, horizon=1)

    sequence = adapter.build_action_sequence(
        action, _build_obs(), device="cpu", open_loop_horizon=None
    )

    torch.testing.assert_close(
        sequence[0].select("left_joint1"),
        torch.tensor([[10.0]]),
    )


def test_groot_adapter_default_arm_is_absolute() -> None:
    # Default arm is absolute: command == action, not current + action.
    adapter = GrootAdapter()
    action = _action_chunk(offset=10.0, horizon=1)

    sequence = adapter.build_action_sequence(
        action, _build_obs(), device="cpu", open_loop_horizon=None
    )

    torch.testing.assert_close(
        sequence[0].select("left_joint1"),
        torch.tensor([[10.0]]),
    )
    torch.testing.assert_close(
        sequence[0].select("right_joint[1-6]"),
        torch.zeros((1, 6)),
    )


def test_groot_policy_act_caches_chunk_then_reinfers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, _ = _build_policy(monkeypatch, open_loop_horizon=2)

    first = policy.act(_build_obs())
    second = policy.act(_build_obs())
    third = policy.act(_build_obs())

    # Observable: step 2 reuses the cached chunk (101 = chunk-1 step 1,
    # absolute target); step 3 re-infers a fresh chunk (200 = chunk-2 step 0).
    assert first.select("left_joint1")[0, 0].item() == 100.0
    assert second.select("left_joint1")[0, 0].item() == 101.0
    assert third.select("left_joint1")[0, 0].item() == 200.0


def test_groot_policy_open_loop_horizon_truncates_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, _ = _build_policy(monkeypatch, open_loop_horizon=1)

    first = policy.act(_build_obs())
    second = policy.act(_build_obs())

    # open_loop_horizon=1 truncates each chunk to one step, so every act
    # re-infers: step 2 is chunk-2 step 0 (200), not chunk-1 step 1 (101).
    assert first.select("left_joint1")[0, 0].item() == 100.0
    assert second.select("left_joint1")[0, 0].item() == 200.0


def test_groot_policy_act_sequence_returns_full_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, _ = _build_policy(monkeypatch, open_loop_horizon=2)

    sequence = policy.act_sequence(_build_obs())

    # A single inference returns the full horizon (chunk-1 steps 0 and 1).
    assert len(sequence) == 2
    assert sequence[0].select("left_joint1")[0, 0].item() == 100.0
    assert sequence[1].select("left_joint1")[0, 0].item() == 101.0


def test_groot_policy_reset_clears_cache_and_calls_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, fake = _build_policy(monkeypatch, open_loop_horizon=2)

    first = policy.act(_build_obs())
    policy.reset()
    after_reset = policy.act(_build_obs())

    # Cache cleared: the post-reset act re-infers (200 = chunk-2 step 0)
    # instead of returning the cached chunk-1 step 1 (101).
    assert first.select("left_joint1")[0, 0].item() == 100.0
    assert after_reset.select("left_joint1")[0, 0].item() == 200.0
    # reset propagated to the network-boundary client.
    assert fake.reset_called


def test_groot_policy_multi_env_obs_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, _ = _build_policy(monkeypatch, open_loop_horizon=2)

    with pytest.raises(ValueError, match="single environment"):
        policy.act(_build_obs(batch_size=2))


def test_groot_adapter_emits_identity_eef_9d_when_eef_key_set() -> None:
    adapter = GrootAdapter(
        arm_specs=(
            GrootArmSpec(
                "left_arm", "left_arm", "left_gripper", eef_key="left_eef"
            ),
            GrootArmSpec(
                "right_arm", "right_arm", "right_gripper", eef_key="right_eef"
            ),
        )
    )

    state = adapter.build_model_input(_build_obs())["state"]

    assert "left_eef" in state and "right_eef" in state
    assert state["left_eef"].shape == (1, 1, 9)
    assert state["left_eef"].dtype == np.float32
    np.testing.assert_allclose(
        state["left_eef"][0, 0],
        np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32),
    )


def _add_ee_frames(
    obs: CanonicalPolicyInput,
    *,
    ee_pose: list[float],
    base_pose: list[float],
    slot: str = "left_arm",
) -> CanonicalPolicyInput:
    obs.manipulators[slot]["ee_pose"] = torch.tensor(
        [ee_pose], dtype=torch.float32
    )
    obs.manipulators[slot]["base_pose"] = torch.tensor(
        [base_pose], dtype=torch.float32
    )
    return obs


def test_groot_adapter_eef_9d_from_ee_and_base_pose() -> None:
    adapter = GrootAdapter(
        arm_specs=(
            GrootArmSpec("left_arm", "left_arm", "left_gripper", eef_key="e"),
        )
    )
    obs = _add_ee_frames(
        _build_obs(),
        ee_pose=[0.4, 0.0, 0.3, 1, 0, 0, 0],
        base_pose=[0.0, 0.0, 0.0, 1, 0, 0, 0],
    )

    eef = adapter.build_model_input(obs)["state"]["e"][0, 0]

    # base at origin, ee at (0.4,0,0.3) identity -> eef == ee, identity rot6d.
    np.testing.assert_allclose(
        eef,
        np.array([0.4, 0, 0.3, 1, 0, 0, 0, 1, 0], dtype=np.float32),
        atol=1e-5,
    )


def test_groot_adapter_eef_9d_applies_tcp_offset() -> None:
    adapter = GrootAdapter(
        arm_specs=(
            GrootArmSpec("left_arm", "left_arm", "left_gripper", eef_key="e"),
        ),
        eef_ee_to_tcp_offset=(0.0, 0.0, 0.1),
    )
    obs = _add_ee_frames(
        _build_obs(),
        ee_pose=[0.4, 0.0, 0.3, 1, 0, 0, 0],
        base_pose=[0.0, 0.0, 0.0, 1, 0, 0, 0],
    )

    eef = adapter.build_model_input(obs)["state"]["e"][0, 0]

    # +0.1 along the ee's local z shifts the TCP to z=0.4.
    np.testing.assert_allclose(
        eef[:3], np.array([0.4, 0, 0.4], dtype=np.float32), atol=1e-5
    )


def test_groot_adapter_eef_9d_rot6d_is_first_two_rows() -> None:
    adapter = GrootAdapter(
        arm_specs=(
            GrootArmSpec("left_arm", "left_arm", "left_gripper", eef_key="e"),
        )
    )
    # 90deg about z (wxyz) -> R first two rows [0,-1,0, 1,0,0].
    obs = _add_ee_frames(
        _build_obs(),
        ee_pose=[0.0, 0.0, 0.0, 0.70710678, 0, 0, 0.70710678],
        base_pose=[0.0, 0.0, 0.0, 1, 0, 0, 0],
    )

    eef = adapter.build_model_input(obs)["state"]["e"][0, 0]

    np.testing.assert_allclose(
        eef[3:], np.array([0, -1, 0, 1, 0, 0], dtype=np.float32), atol=1e-5
    )


def test_groot_adapter_video_single_timestep() -> None:
    adapter = GrootAdapter()

    video = adapter.build_model_input(_obs_with_fill(7))["video"]

    # [B=1, T=1, H, W, 3] — single frame per camera.
    assert video["static_camera"].shape == (1, 1, 2, 2, 3)
    assert video["static_camera"][0, 0, 0, 0, 0] == 7


def test_groot_cfg_wires_eef() -> None:
    policy_cfg = create_policy_from_model_cfg(
        {
            "policy": "groot",
            "eef_ee_to_tcp_offset": [0.0, 0.0, 0.1],
            "arms": [
                {
                    "manipulator_slot": "single_arm",
                    "arm_key": "joint_position",
                    "gripper_key": "gripper_position",
                    "eef_key": "eef_9d",
                    "arm_relative": False,
                }
            ],
        }
    )

    assert isinstance(policy_cfg, GrootPolicyCfg)
    assert policy_cfg.eef_ee_to_tcp_offset == [0.0, 0.0, 0.1]
    assert policy_cfg.arms[0].eef_key == "eef_9d"


def test_create_policy_from_model_cfg_groot_returns_cfg() -> None:
    policy_cfg = create_policy_from_model_cfg(
        {
            "policy": "groot",
            "host": "1.2.3.4",
            "port": 6000,
            "open_loop_horizon": 4,
            "instruction": "lift the cube",
            "video_map": {
                "static_camera": "base",
                "left_hand_camera": "left_wrist",
                "right_hand_camera": "right_wrist",
            },
            "arms": [
                {
                    "manipulator_slot": "left_arm",
                    "arm_key": "left_arm",
                    "gripper_key": "left_gripper",
                    "arm_relative": True,
                },
                {
                    "manipulator_slot": "right_arm",
                    "arm_key": "right_arm",
                    "gripper_key": "right_gripper",
                    "arm_relative": True,
                },
            ],
        }
    )

    assert isinstance(policy_cfg, GrootPolicyCfg)
    assert policy_cfg.host == "1.2.3.4"
    assert policy_cfg.port == 6000
    assert policy_cfg.open_loop_horizon == 4
    assert policy_cfg.instruction == "lift the cube"
    assert len(policy_cfg.arms) == 2
    assert policy_cfg.arms[0].manipulator_slot == "left_arm"
