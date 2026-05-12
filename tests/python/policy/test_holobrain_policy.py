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
import types
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robo_orchard_sim.policy.holobrain.policy import (
    HolobrainPolicy,
    HolobrainPolicyCfg,
)


class _Pose:
    def __init__(self) -> None:
        self.xyz = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        self.quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)


class _Sensor:
    def __init__(
        self,
        sensor_data: torch.Tensor,
        intrinsic_matrices: torch.Tensor | None = None,
        pose: _Pose | None = None,
    ) -> None:
        self.sensor_data = sensor_data
        self.intrinsic_matrices = intrinsic_matrices
        self.pose = pose


class _FakePipelineOutput:
    def __init__(self, action: torch.Tensor) -> None:
        self.action = action


class _FakePipeline:
    def __init__(self) -> None:
        class _Model:
            def eval(self) -> None:
                return None

        self.model = _Model()
        self._inference_idx = 0

    def __call__(self, model_input):
        del model_input
        self._inference_idx += 1
        offset = float(self._inference_idx * 10)
        return _FakePipelineOutput(
            action=torch.tensor(
                [
                    [
                        offset + 1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        0.4,
                        offset + 7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        -0.2,
                    ],
                    [
                        offset + 2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        0.6,
                        offset + 8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        -0.4,
                    ],
                ],
                dtype=torch.float32,
            )
        )


def _build_obs(batch_size: int = 1) -> dict:
    rgb = torch.arange(batch_size * 12, dtype=torch.uint8).reshape(
        batch_size, 2, 2, 3
    )
    depth = torch.ones((batch_size, 2, 2, 1), dtype=torch.float32)
    intrinsic = (
        torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    )
    pose = _Pose()
    camera_obs = {
        "rgb": _Sensor(rgb, intrinsic, pose=pose),
        "depth": _Sensor(depth),
    }
    return {
        "/camera": {
            "left_hand_camera_term": camera_obs,
            "static_camera_term": camera_obs,
            "right_hand_camera_term": camera_obs,
        },
        "/robot": {
            "left_joint_position": torch.ones((batch_size, 7)),
            "right_joint_position": torch.ones((batch_size, 7)),
        },
        "instruction": "pick apple",
    }


def _install_fake_holobrain_modules(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: _FakePipeline,
) -> None:
    processor_module = types.ModuleType(
        "robo_orchard_lab.models.holobrain.processor"
    )

    class _FakeMultiArmManipulationInput:
        def __init__(self, **kwargs) -> None:
            self.payload = kwargs

    processor_module.MultiArmManipulationInput = _FakeMultiArmManipulationInput
    monkeypatch.setitem(
        sys.modules,
        "robo_orchard_lab.models.holobrain.processor",
        processor_module,
    )

    pipeline_module = types.ModuleType(
        "robo_orchard_lab.models.holobrain.pipeline"
    )

    class _FakeInferencePipeline:
        @staticmethod
        def load_pipeline(**kwargs):
            del kwargs
            return pipeline

    pipeline_module.HoloBrainInferencePipeline = _FakeInferencePipeline
    monkeypatch.setitem(
        sys.modules,
        "robo_orchard_lab.models.holobrain.pipeline",
        pipeline_module,
    )


def _build_policy(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: _FakePipeline,
    *,
    model_dir: str | None = "/tmp/holobrain_export/model",
    valid_action_step: int | None = None,
) -> HolobrainPolicy:
    _install_fake_holobrain_modules(monkeypatch, pipeline)
    cfg = HolobrainPolicyCfg(
        model_dir=model_dir,
        inference_prefix="isaac_pick_place",
        valid_action_step=valid_action_step,
    )
    return HolobrainPolicy(cfg=cfg)


def test_holobrain_policy_act_reuses_cached_actions_expected_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _FakePipeline()
    policy = _build_policy(monkeypatch, pipeline)

    first = policy.act(_build_obs())
    second = policy.act(_build_obs())
    third = policy.act(_build_obs())

    assert first["left_robot_joint_position"][0, 0].item() == 11.0
    assert second["left_robot_joint_position"][0, 0].item() == 12.0
    assert third["left_robot_joint_position"][0, 0].item() == 21.0


def test_holobrain_policy_act_valid_step_refreshes_cache_expected_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _FakePipeline()
    policy = _build_policy(
        monkeypatch,
        pipeline,
        valid_action_step=1,
    )

    first = policy.act(_build_obs())
    second = policy.act(_build_obs())

    assert first["left_robot_joint_position"][0, 0].item() == 11.0
    assert second["left_robot_joint_position"][0, 0].item() == 21.0


def test_holobrain_policy_act_sequence_returns_refreshed_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _FakePipeline()
    policy = _build_policy(monkeypatch, pipeline)

    sequence = policy.act_sequence(_build_obs())
    next_action = policy.act(_build_obs())

    assert len(sequence) == 2
    assert sequence[0]["left_robot_joint_position"][0, 0].item() == 11.0
    assert sequence[1]["left_robot_joint_position"][0, 0].item() == 12.0
    assert next_action["left_robot_joint_position"][0, 0].item() == 21.0


def test_holobrain_policy_init_uses_env_model_dir_expected_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _FakePipeline()
    monkeypatch.setenv(
        "ROBO_ORCHARD_HOLOBRAIN_MODEL_DIR",
        "/tmp/from-env-model",
    )
    policy = _build_policy(
        monkeypatch,
        pipeline,
        model_dir=None,
    )

    action = policy.act(_build_obs())

    assert action["left_robot_joint_position"][0, 0].item() == 11.0


def test_holobrain_policy_act_multi_env_input_raises_value_error() -> None:
    pipeline = _FakePipeline()
    monkeypatch = pytest.MonkeyPatch()
    try:
        policy = _build_policy(monkeypatch, pipeline)

        with pytest.raises(ValueError, match="single environment"):
            policy.act(_build_obs(batch_size=2))
    finally:
        monkeypatch.undo()
