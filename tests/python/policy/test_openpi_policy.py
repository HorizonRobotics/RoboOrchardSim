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
import types
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
from robo_orchard_sim.policy.openpi import policy as openpi_policy_module
from robo_orchard_sim.policy.openpi.adapter import OpenPiAdapter
from robo_orchard_sim.policy.openpi.openpi_config import (
    build_openpi_model_config,
    build_openpi_transform_pipeline,
)
from robo_orchard_sim.policy.openpi.policy import (
    OpenPiInferenceConfig,
    OpenPiModelConfig,
    OpenPiPolicy,
    OpenPiPolicyCfg,
)


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
            "wrist": CameraBinding(obs_term="wrist_camera_term"),
            "base": CameraBinding(obs_term="ext1_camera_term"),
            "right_wrist": CameraBinding(obs_term="ext2_camera_term"),
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


def _camera_cfg(size: tuple[int, int]) -> dict[str, Any]:
    center_x = float(size[0] / 2)
    center_y = float(size[1] / 2)
    return {
        "target_size": size,
        "target_intrinsic": [
            [1.0, 0.0, center_x, 0.0],
            [0.0, 1.0, center_y, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def test_openpi_policy_requirement_declares_broad_contract() -> None:
    requirement = OpenPiPolicy.policy_requirement()

    assert requirement.required_camera_modalities == ("rgb", "intrinsic")
    assert requirement.min_camera_count == 1
    assert requirement.min_manipulator_count == 1
    assert requirement.require_instruction is True


class _Sensor:
    def __init__(
        self,
        sensor_data: torch.Tensor,
        intrinsic_matrices: torch.Tensor | None = None,
    ) -> None:
        self.sensor_data = sensor_data
        self.intrinsic_matrices = intrinsic_matrices


def _build_obs(batch_size: int = 1) -> CanonicalPolicyInput:
    rgb = torch.tensor(
        [[[[10, 20, 30], [40, 50, 60]]]],
        dtype=torch.uint8,
    ).repeat(batch_size, 1, 1, 1)
    intrinsic = (
        torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    )
    camera_obs = {"rgb": _Sensor(rgb, intrinsic)}
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


def _build_single_arm_obs(batch_size: int = 1) -> CanonicalPolicyInput:
    rgb = torch.tensor(
        [[[[10, 20, 30], [40, 50, 60]]]],
        dtype=torch.uint8,
    ).repeat(batch_size, 1, 1, 1)
    intrinsic = (
        torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    )
    camera_obs = {"rgb": _Sensor(rgb, intrinsic)}
    return CanonicalPolicyInput(
        cameras={
            "wrist": camera_obs,
            "right_wrist": camera_obs,
            "base": camera_obs,
        },
        manipulators={
            "single_arm": {
                "joint_position": torch.tensor(
                    [[1, 2, 3, 4, 5, 6, 7, 0.1]],
                    dtype=torch.float32,
                ).repeat(batch_size, 1),
            },
        },
        instruction="pick apple",
        action_layout=compile_action_layout(_build_franka_schema()),
    )


class _FakeOpenPiPolicy:
    def __init__(self) -> None:
        self.inference_count = 0

    def infer(self, model_input: dict[str, Any]) -> dict[str, np.ndarray]:
        assert model_input["prompt"] == "pick apple"
        self.inference_count += 1
        offset = float(self.inference_count * 10)
        return {
            "actions": np.array(
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
                dtype=np.float32,
            )
        }


def _install_fake_openpi_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    policy_factory: Any | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class _ModelTypeValue:
        def __init__(self, name: str) -> None:
            self.name = name

    class _ModelType:
        PI0 = _ModelTypeValue("PI0")
        PI05 = _ModelTypeValue("PI05")
        PI0_FAST = _ModelTypeValue("PI0_FAST")

    class _Pi0Config:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.model_type = (
                _ModelType.PI05 if kwargs.get("pi05") else _ModelType.PI0
            )
            self.max_token_len = kwargs["max_token_len"]
            self.discrete_state_input = True
            self.action_dim = 32

        def load(self, params: Any) -> str:
            captured["loaded_params"] = params
            return "loaded-model"

    class _Pi0FASTConfig:
        def __init__(self) -> None:
            self.model_type = _ModelType.PI0_FAST
            self.max_token_len = 128
            self.fast_model_tokenizer = None
            self.fast_model_tokenizer_kwargs = None
            self.action_horizon = 50
            self.action_dim = 32

        def load(self, params: Any) -> str:
            captured["loaded_params"] = params
            return "loaded-model"

    class _PaligemmaTokenizer:
        def __init__(self, max_token_len: int) -> None:
            self.max_token_len = max_token_len

    class _FASTTokenizer:
        def __init__(self, max_token_len: int, **kwargs: Any) -> None:
            self.max_token_len = max_token_len
            self.kwargs = kwargs

    class _Group:
        def __init__(self, inputs=None, outputs=None):
            self.inputs = inputs or []
            self.outputs = outputs or []

        def push(self, inputs=None, outputs=None):
            return _Group(
                inputs=self.inputs + (inputs or []),
                outputs=self.outputs + (outputs or []),
            )

    class _TransformsModule:
        Group = _Group

        @staticmethod
        def make_bool_mask(*args: Any) -> tuple[Any, ...]:
            return args

        @staticmethod
        def DeltaActions(mask: Any) -> tuple[str, Any]:
            return ("delta", mask)

        @staticmethod
        def AbsoluteActions(mask: Any) -> tuple[str, Any]:
            return ("absolute", mask)

        @staticmethod
        def ResizeImages(height: int, width: int) -> tuple[str, int, int]:
            return ("resize", height, width)

        @staticmethod
        def TokenizePrompt(
            tokenizer: _PaligemmaTokenizer,
            discrete_state_input: bool = False,
        ) -> tuple[str, int, bool]:
            return (
                "tokenize_prompt",
                tokenizer.max_token_len,
                discrete_state_input,
            )

        @staticmethod
        def PadStatesAndActions(action_dim: int) -> tuple[str, int]:
            return ("pad", action_dim)

        @staticmethod
        def TokenizeFASTInputs(
            tokenizer: _FASTTokenizer,
        ) -> tuple[str, int, dict[str, Any]]:
            return (
                "tokenize_fast_inputs",
                tokenizer.max_token_len,
                tokenizer.kwargs,
            )

        @staticmethod
        def ExtractFASTActions(
            tokenizer: _FASTTokenizer,
            *,
            action_horizon: int,
            action_dim: int,
        ) -> tuple[str, int, dict[str, Any], int, int]:
            return (
                "extract_fast_actions",
                tokenizer.max_token_len,
                tokenizer.kwargs,
                action_horizon,
                action_dim,
            )

        @staticmethod
        def InjectDefaultPrompt(prompt: str | None) -> tuple[str, str | None]:
            return ("prompt", prompt)

        @staticmethod
        def Normalize(
            norm_stats: Any,
            use_quantiles: bool,
        ) -> tuple[str, Any, bool]:
            return ("normalize", norm_stats, use_quantiles)

        @staticmethod
        def Unnormalize(
            norm_stats: Any,
            use_quantiles: bool,
        ) -> tuple[str, Any, bool]:
            return ("unnormalize", norm_stats, use_quantiles)

    def _policy(
        loaded_model: str,
        *,
        transforms: list[Any],
        output_transforms: list[Any],
        sample_kwargs: Any,
        metadata: Any,
        is_pytorch: bool,
        pytorch_device: Any,
    ) -> Any:
        captured["loaded_model"] = loaded_model
        captured["transforms"] = transforms
        captured["output_transforms"] = output_transforms
        captured["sample_kwargs"] = sample_kwargs
        captured["metadata"] = metadata
        captured["is_pytorch"] = is_pytorch
        captured["pytorch_device"] = pytorch_device
        if policy_factory is not None:
            return policy_factory()
        return "policy-instance"

    monkeypatch.setitem(sys.modules, "jax", types.ModuleType("jax"))
    jax_numpy_module = types.ModuleType("jax.numpy")
    jax_numpy_module.bfloat16 = "bfloat16"
    monkeypatch.setitem(sys.modules, "jax.numpy", jax_numpy_module)

    monkeypatch.setitem(sys.modules, "openpi", types.ModuleType("openpi"))
    monkeypatch.setitem(
        sys.modules, "openpi.models", types.ModuleType("openpi.models")
    )
    model_module = types.ModuleType("openpi.models.model")
    model_module.ModelType = _ModelType
    model_module.restore_params = lambda path, dtype: (
        "params",
        str(path),
        dtype,
    )
    monkeypatch.setitem(sys.modules, "openpi.models.model", model_module)

    pi0_config_module = types.ModuleType("openpi.models.pi0_config")
    pi0_config_module.Pi0Config = _Pi0Config
    monkeypatch.setitem(
        sys.modules,
        "openpi.models.pi0_config",
        pi0_config_module,
    )

    pi0_fast_module = types.ModuleType("openpi.models.pi0_fast")
    pi0_fast_module.Pi0FASTConfig = _Pi0FASTConfig
    monkeypatch.setitem(sys.modules, "openpi.models.pi0_fast", pi0_fast_module)

    tokenizer_module = types.ModuleType("openpi.models.tokenizer")
    tokenizer_module.PaligemmaTokenizer = _PaligemmaTokenizer
    tokenizer_module.FASTTokenizer = _FASTTokenizer
    monkeypatch.setitem(
        sys.modules, "openpi.models.tokenizer", tokenizer_module
    )

    monkeypatch.setitem(
        sys.modules, "openpi.policies", types.ModuleType("openpi.policies")
    )
    policy_module = types.ModuleType("openpi.policies.policy")
    policy_module.Policy = _policy
    monkeypatch.setitem(sys.modules, "openpi.policies.policy", policy_module)

    monkeypatch.setitem(
        sys.modules, "openpi.shared", types.ModuleType("openpi.shared")
    )
    download_module = types.ModuleType("openpi.shared.download")
    download_module.maybe_download = lambda path: path
    monkeypatch.setitem(sys.modules, "openpi.shared.download", download_module)

    normalize_module = types.ModuleType("openpi.shared.normalize")
    normalize_module.load = lambda path: ("norm-stats", str(path))
    monkeypatch.setitem(
        sys.modules, "openpi.shared.normalize", normalize_module
    )

    transforms_module = types.ModuleType("openpi.transforms")
    transforms_module.Group = _TransformsModule.Group
    transforms_module.make_bool_mask = _TransformsModule.make_bool_mask
    transforms_module.DeltaActions = _TransformsModule.DeltaActions
    transforms_module.AbsoluteActions = _TransformsModule.AbsoluteActions
    transforms_module.ResizeImages = _TransformsModule.ResizeImages
    transforms_module.TokenizePrompt = _TransformsModule.TokenizePrompt
    transforms_module.PadStatesAndActions = (
        _TransformsModule.PadStatesAndActions
    )
    transforms_module.TokenizeFASTInputs = _TransformsModule.TokenizeFASTInputs
    transforms_module.ExtractFASTActions = _TransformsModule.ExtractFASTActions
    transforms_module.InjectDefaultPrompt = (
        _TransformsModule.InjectDefaultPrompt
    )
    transforms_module.Normalize = _TransformsModule.Normalize
    transforms_module.Unnormalize = _TransformsModule.Unnormalize
    monkeypatch.setitem(sys.modules, "openpi.transforms", transforms_module)

    return captured


def _build_openpi_policy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    valid_action_step: int | None = None,
) -> OpenPiPolicy:
    _install_fake_openpi_runtime(
        monkeypatch,
        policy_factory=_FakeOpenPiPolicy,
    )
    cfg = OpenPiPolicyCfg(
        model=OpenPiModelConfig(
            model_type="pi05",
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ),
        inference=OpenPiInferenceConfig(norm_stats_name="pi05"),
        model_dir="/tmp/pi-checkpoint",
        valid_action_step=valid_action_step,
    )
    return OpenPiPolicy(cfg=cfg)


def test_openpi_adapter_build_action_sequence_given_valid_actions_splits() -> (
    None
):
    adapter = OpenPiAdapter(joint_num=7)
    actions = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 0.4, 7, 8, 9, 10, 11, 12, -0.2],
            [2, 3, 4, 5, 6, 7, 0.6, 8, 9, 10, 11, 12, 13, -0.4],
        ],
        dtype=torch.float32,
    )

    sequence = adapter.build_action_sequence(
        actions,
        _build_obs(),
        device="cpu",
        valid_action_step=None,
    )

    assert len(sequence) == 2
    torch.testing.assert_close(
        sequence[0].select("left_joint[1-6]"),
        torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        sequence[0].select("right_joint[1-6]"),
        torch.tensor([[7, 8, 9, 10, 11, 12]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        sequence[0].select("left_joint7", "left_joint8"),
        torch.tensor([[0.2, -0.2]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        sequence[0].select("right_joint7", "right_joint8"),
        torch.tensor([[-0.1, 0.1]], dtype=torch.float32),
    )


def test_openpi_adapter_build_model_input_given_valid_obs_returns_input() -> (
    None
):
    adapter = OpenPiAdapter(joint_num=7)

    model_input = adapter.build_model_input(_build_obs())

    assert set(model_input) == {"image", "image_mask", "state", "prompt"}
    assert model_input["prompt"] == "pick apple"
    assert model_input["image"]["left_wrist_0_rgb"].shape == (252, 392, 3)
    assert model_input["image"]["base_0_rgb"].shape == (252, 392, 3)
    assert model_input["image_mask"] == {
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.True_,
        "base_0_rgb": np.True_,
    }
    np.testing.assert_allclose(
        model_input["state"],
        np.array(
            [
                1,
                2,
                3,
                3.14,
                3.14,
                3.14,
                0.4,
                3.14,
                3.14,
                3.14,
                3.14,
                3.14,
                3.14,
                -0.8,
            ]
        ),
    )


def test_openpi_adapter_build_model_input_missing_instruction_raises() -> None:
    adapter = OpenPiAdapter(joint_num=7)
    obs = _build_obs()
    obs = obs.model_copy(update={"instruction": None})

    with pytest.raises(ValueError, match="requires instruction"):
        adapter.build_model_input(obs)


def test_openpi_adapter_single_arm_obs_returns_model_input() -> None:
    adapter = OpenPiAdapter(joint_num=8)

    model_input = adapter.build_model_input(_build_single_arm_obs())

    assert set(model_input) == {"image", "image_mask", "state", "prompt"}
    assert model_input["prompt"] == "pick apple"
    assert model_input["image_mask"] == {
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.True_,
        "base_0_rgb": np.True_,
    }
    np.testing.assert_allclose(
        model_input["state"],
        np.array([1, 2, 3, 3.14, 3.14, 3.14, 3.14, 0.2]),
    )


def test_openpi_adapter_dualarm_missing_right_camera_raises() -> None:
    adapter = OpenPiAdapter(joint_num=7)
    obs = _build_obs()
    del obs.cameras["right_wrist"]

    with pytest.raises(ValueError, match="requires canonical camera slot"):
        adapter.build_model_input(obs)


def test_openpi_adapter_single_arm_missing_base_camera_raises() -> None:
    adapter = OpenPiAdapter(joint_num=8)
    obs = _build_single_arm_obs()
    del obs.cameras["base"]

    with pytest.raises(ValueError, match="requires canonical camera slot"):
        adapter.build_model_input(obs)


def test_openpi_adapter_missing_manipulator_slot_raises_value_error() -> None:
    adapter = OpenPiAdapter(joint_num=7)
    obs = _build_obs()
    del obs.manipulators["right_arm"]

    with pytest.raises(ValueError, match="missing manipulator slots"):
        adapter.build_model_input(obs)


def test_openpi_policy_act_given_cached_actions_reuses_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _build_openpi_policy(monkeypatch)

    first = policy.act(_build_obs())
    second = policy.act(_build_obs())
    third = policy.act(_build_obs())

    assert first.select("left_joint1")[0, 0].item() == 11.0
    assert second.select("left_joint1")[0, 0].item() == 12.0
    assert third.select("left_joint1")[0, 0].item() == 21.0


def test_openpi_policy_act_sequence_returns_refreshed_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _build_openpi_policy(monkeypatch)

    sequence = policy.act_sequence(_build_obs())
    next_action = policy.act(_build_obs())

    assert len(sequence) == 2
    assert sequence[0].select("left_joint1")[0, 0].item() == 11.0
    assert sequence[1].select("left_joint1")[0, 0].item() == 12.0
    assert next_action.select("left_joint1")[0, 0].item() == 21.0


def test_openpi_policy_act_given_multi_env_obs_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _build_openpi_policy(monkeypatch)

    with pytest.raises(ValueError, match="single environment"):
        policy.act(_build_obs(batch_size=2))


def test_openpi_policy_act_given_single_arm_obs_returns_single_arm_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openpi_runtime(
        monkeypatch,
        policy_factory=_FakeOpenPiPolicy,
    )
    cfg = OpenPiPolicyCfg(
        model=OpenPiModelConfig(
            model_type="pi05",
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ),
        inference=OpenPiInferenceConfig(norm_stats_name="pi05"),
        model_dir="/tmp/pi-checkpoint",
        joint_num=8,
    )
    policy = OpenPiPolicy(cfg=cfg)

    action = policy.act(_build_single_arm_obs())

    torch.testing.assert_close(
        action.select("panda_joint[1-7]"),
        torch.tensor([[11, 2, 3, 4, 5, 6, 0.4]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        action.select("panda_finger_joint1", "panda_finger_joint2"),
        torch.tensor([[8.5, 8.5]], dtype=torch.float32),
    )


def test_create_policy_from_model_cfg_openpi_camera_cfg_returns_cfg() -> None:
    policy_cfg = create_policy_from_model_cfg(
        {
            "policy": "openpi",
            "model": {
                "model_type": "pi05",
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
            },
            "inference": {
                "norm_stats_name": "pi05",
            },
            "model_dir": "/tmp/pi-checkpoint",
            "valid_action_step": 3,
            "enable_intrinsic_remap": False,
            "cameras": {
                "left": _camera_cfg((28, 28)),
                "right": _camera_cfg((28, 28)),
                "middle": _camera_cfg((56, 28)),
            },
        }
    )

    assert isinstance(policy_cfg, OpenPiPolicyCfg)
    assert policy_cfg.model.model_type == "pi05"
    assert policy_cfg.valid_action_step == 3
    assert policy_cfg.enable_intrinsic_remap is False
    assert policy_cfg.cameras["middle"].target_size == (56, 28)


def test_create_openpi_policy_from_config_given_default_prompt_injects_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_openpi_runtime(monkeypatch)
    model = build_openpi_model_config(
        OpenPiModelConfig(
            model_type="pi05",
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
    )

    policy = openpi_policy_module.create_openpi_policy_from_config(
        inference_cfg=OpenPiInferenceConfig(
            default_prompt="pick apple",
            norm_stats_name="pi05",
        ),
        model=model,
        checkpoint_dir="/tmp/pi-checkpoint",
    )

    assert policy == "policy-instance"
    assert captured["transforms"] == [
        ("prompt", "pick apple"),
        ("normalize", ("norm-stats", "/tmp/pi-checkpoint/assets/pi05"), True),
        ("resize", 224, 224),
        ("tokenize_prompt", 128, True),
        ("pad", 32),
    ]
    assert captured["transforms"].count(("prompt", "pick apple")) == 1
    assert captured["output_transforms"] == [
        (
            "unnormalize",
            ("norm-stats", "/tmp/pi-checkpoint/assets/pi05"),
            True,
        ),
    ]


def test_build_openpi_model_config_given_pi05_returns_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openpi_runtime(monkeypatch)

    config = build_openpi_model_config(
        OpenPiModelConfig(
            model_type="pi05",
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
    )

    assert config.kwargs == {
        "pi05": True,
        "max_token_len": 128,
        "paligemma_variant": "gemma_2b_lora",
        "action_expert_variant": "gemma_300m_lora",
    }


def test_build_openpi_transform_pipeline_given_delta_actions_wraps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openpi_runtime(monkeypatch)
    model = build_openpi_model_config(
        OpenPiModelConfig(
            model_type="pi05",
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
    )

    runtime = build_openpi_transform_pipeline(
        OpenPiInferenceConfig(
            use_delta_joint_actions=True,
            norm_stats_name="fake_asset",
        ),
        model,
    )

    assert runtime.model_transforms.inputs == [
        ("resize", 224, 224),
        ("tokenize_prompt", 128, True),
        ("pad", 32),
    ]
    assert runtime.use_quantile_norm is True
    assert runtime.data_transforms.inputs == [("delta", (6, -1, 6, -1))]
    assert runtime.data_transforms.outputs == [("absolute", (6, -1, 6, -1))]


def test_build_openpi_transform_pipeline_franka_delta_actions_uses_arm_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openpi_runtime(monkeypatch)
    model = build_openpi_model_config(
        OpenPiModelConfig(
            model_type="pi05",
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
    )

    runtime = build_openpi_transform_pipeline(
        OpenPiInferenceConfig(
            use_delta_joint_actions=True,
            norm_stats_name="fake_asset",
            delta_action_embodiment="franka_panda",
        ),
        model,
    )

    assert runtime.data_transforms.inputs == [("delta", (7, -1))]
    assert runtime.data_transforms.outputs == [("absolute", (7, -1))]
