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
import logging
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
from pydantic import BaseModel, Field, field_validator
from robo_orchard_core.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_core.utils.config import ClassType

from robo_orchard_sim.policy.openpi.adapter import (
    DEFAULT_CAMERAS,
    OpenPiAction,
    OpenPiAdapter,
)
from robo_orchard_sim.policy.openpi.openpi_config import (
    OpenPiInferenceConfig,
    OpenPiModelConfig,
    build_openpi_model_config,
    build_openpi_transform_pipeline,
)

logger = logging.getLogger(__name__)
_MODEL_DIR_ENV_VAR = "ROBO_ORCHARD_OPENPI_MODEL_DIR"


def create_openpi_policy_from_config(
    *,
    inference_cfg: OpenPiInferenceConfig,
    model: Any,
    checkpoint_dir: Path | str,
) -> Any:
    """Create an OpenPI policy from a trained checkpoint."""
    import jax.numpy as jnp
    import openpi.models.model as _model
    import openpi.policies.policy as _policy
    import openpi.shared.download as download
    import openpi.shared.normalize as _normalize
    import openpi.transforms as transforms

    checkpoint_dir = Path(download.maybe_download(str(checkpoint_dir)))

    logger.info("Loading OpenPi model from %s", checkpoint_dir)
    loaded_model = model.load(
        _model.restore_params(
            checkpoint_dir / "params",
            dtype=jnp.bfloat16,
        )
    )
    pipeline = build_openpi_transform_pipeline(inference_cfg, model)
    norm_stats_dir = checkpoint_dir / "assets" / inference_cfg.norm_stats_name
    logger.info("Loading OpenPi norm stats from %s", norm_stats_dir)
    norm_stats = _normalize.load(norm_stats_dir)

    return _policy.Policy(
        loaded_model,
        transforms=[
            transforms.InjectDefaultPrompt(inference_cfg.default_prompt),
            *pipeline.data_transforms.inputs,
            transforms.Normalize(
                norm_stats,
                use_quantiles=pipeline.use_quantile_norm,
            ),
            *pipeline.model_transforms.inputs,
        ],
        output_transforms=[
            *pipeline.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats,
                use_quantiles=pipeline.use_quantile_norm,
            ),
            *pipeline.data_transforms.outputs,
        ],
        sample_kwargs=None,
        metadata=None,
        is_pytorch=False,
        pytorch_device=None,
    )


def create_openpi_policy(cfg: "OpenPiPolicyCfg") -> Any:
    """Create the OpenPI policy configured for simulator inference."""
    model_dir = cfg.model_dir or os.getenv(_MODEL_DIR_ENV_VAR)
    if not model_dir:
        raise ValueError(
            "OpenPi model_dir must be set in the config or via "
            f"{_MODEL_DIR_ENV_VAR}"
        )
    model = build_openpi_model_config(cfg.model)
    return create_openpi_policy_from_config(
        inference_cfg=cfg.inference,
        model=model,
        checkpoint_dir=model_dir,
    )


class OpenPiPolicy(PolicyMixin[dict[str, Any], OpenPiAction]):
    """Local OpenPI policy with simulator observation/action adaptation."""

    cfg: "OpenPiPolicyCfg"

    def __init__(
        self,
        cfg: "OpenPiPolicyCfg",
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            observation_space=observation_space,
            action_space=action_space,
        )
        self._adapter = OpenPiAdapter(
            joint_num=cfg.joint_num,
            cameras=cfg.cameras,
            enable_intrinsic_remap=cfg.enable_intrinsic_remap,
        )
        self._policy = self._load_policy(cfg)
        self._cached_actions: list[OpenPiAction] = []
        self._cached_index = 0

    def reset(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._cached_actions = []
        self._cached_index = 0

    def act(self, obs: dict[str, Any]) -> OpenPiAction:
        """Return one action, reusing the local cached horizon if possible."""
        self._validate_observation_batch(obs)
        if self._cached_index >= len(self._cached_actions):
            self._cached_actions = self._get_fresh_action_sequence(obs)
            self._cached_index = 0
        action = self._cached_actions[self._cached_index]
        self._cached_index += 1
        return action

    def act_sequence(self, obs: dict[str, Any]) -> list[OpenPiAction]:
        """Return a freshly inferred action horizon for remote batch use."""
        sequence = self._get_fresh_action_sequence(obs)
        self._cached_actions = []
        self._cached_index = 0
        return sequence

    def _get_fresh_action_sequence(
        self, obs: dict[str, Any]
    ) -> list[OpenPiAction]:
        self._validate_observation_batch(obs)
        self._refresh_action_cache(obs)
        return list(self._cached_actions)

    def _refresh_action_cache(self, obs: dict[str, Any]) -> None:
        self._cached_actions = self._run_inference(obs)
        self._cached_index = 0

    def _run_inference(self, obs: dict[str, Any]) -> list[OpenPiAction]:
        model_input = self._adapter.build_model_input(obs)
        result = self._policy.infer(model_input)
        device = obs["/robot"]["left_joint_position"].device
        return self._adapter.build_action_sequence(
            result["actions"],
            device=device,
            valid_action_step=self.cfg.valid_action_step,
        )

    @staticmethod
    def _validate_observation_batch(obs: dict[str, Any]) -> None:
        batch_size = int(obs["/robot"]["left_joint_position"].shape[0])
        if batch_size != 1:
            raise ValueError(
                "OpenPiPolicy currently supports single environment only"
            )

    @staticmethod
    def _load_policy(cfg: "OpenPiPolicyCfg") -> Any:
        return create_openpi_policy(cfg)


class OpenPiCameraCfg(BaseModel):
    """Per-camera image resize config for :class:`OpenPiPolicy`."""

    target_size: tuple[int, int]
    target_intrinsic: list[list[float]]

    @field_validator("target_size")
    @classmethod
    def _validate_target_size(
        cls,
        value: tuple[int, int],
    ) -> tuple[int, int]:
        if len(value) != 2 or value[0] <= 0 or value[1] <= 0:
            raise ValueError("target_size must contain two positive integers")
        return value

    @field_validator("target_intrinsic")
    @classmethod
    def _validate_target_intrinsic(
        cls,
        value: list[list[float]],
    ) -> list[list[float]]:
        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("target_intrinsic must be a 4x4 matrix")
        return value


class OpenPiPolicyCfg(PolicyConfig[OpenPiPolicy]):
    """Config for :class:`OpenPiPolicy`."""

    class_type: ClassType[OpenPiPolicy] = OpenPiPolicy

    model: OpenPiModelConfig
    inference: OpenPiInferenceConfig
    model_dir: str | None = None
    logging_tag: str | None = None
    joint_num: int = 7
    valid_action_step: int | None = 50
    enable_intrinsic_remap: bool = True
    cameras: dict[str, OpenPiCameraCfg] = Field(
        default_factory=lambda: {
            camera_name: OpenPiCameraCfg(**camera_cfg)
            for camera_name, camera_cfg in DEFAULT_CAMERAS.items()
        }
    )

    @field_validator("valid_action_step")
    @classmethod
    def _validate_valid_action_step(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("valid_action_step must be greater than 0")
        return value
