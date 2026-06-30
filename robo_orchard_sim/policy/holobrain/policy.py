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
import os
from typing import Any, TypeAlias

import gymnasium as gym
import torch
from pydantic import field_validator
from robo_orchard_core.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_core.utils.config import ClassType
from robo_orchard_core.utils.logging import LoggerManager

from robo_orchard_sim.contracts.joint_command import UnifiedJointCommand
from robo_orchard_sim.contracts.policy_binding import (
    CanonicalPolicyInput,
    PolicyRequirement,
)
from robo_orchard_sim.policy.holobrain.adapter import HolobrainAdapter

logger = LoggerManager().get_child(__name__)
HolobrainAction: TypeAlias = UnifiedJointCommand
_MODEL_DIR_ENV_VAR = "ROBO_ORCHARD_HOLOBRAIN_MODEL_DIR"


class HolobrainPolicy(PolicyMixin[CanonicalPolicyInput, HolobrainAction]):
    """Local Holobrain policy with an embedded inference pipeline."""

    cfg: "HolobrainPolicyCfg"

    def __init__(
        self,
        cfg: "HolobrainPolicyCfg",
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            observation_space=observation_space,
            action_space=action_space,
        )
        self._adapter: HolobrainAdapter | None = None
        self._embodiment_type: str | None = cfg.embodiment_type
        if self._embodiment_type is not None:
            self._adapter = self._build_adapter(
                embodiment_type=self._embodiment_type
            )
        self._pipeline = self._load_pipeline(cfg)
        self._pipeline.model.eval()
        self._cached_actions: list[HolobrainAction] = []
        self._cached_index = 0

    def reset(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._cached_actions = []
        self._cached_index = 0

    def act(self, obs: CanonicalPolicyInput) -> HolobrainAction:
        """Return one action, reusing the local cached horizon if possible."""
        self._validate_observation_batch(obs)
        if self._cached_index >= len(self._cached_actions):
            self._cached_actions = self._get_fresh_action_sequence(obs)
            self._cached_index = 0
        action = self._cached_actions[self._cached_index]
        self._cached_index += 1
        return action

    def act_sequence(
        self,
        obs: CanonicalPolicyInput,
    ) -> list[HolobrainAction]:
        """Return a freshly inferred action horizon for remote batch use."""
        sequence = self._get_fresh_action_sequence(obs)
        self._cached_actions = []
        self._cached_index = 0
        return sequence

    def _get_fresh_action_sequence(
        self, obs: CanonicalPolicyInput
    ) -> list[HolobrainAction]:
        self._validate_observation_batch(obs)
        self._refresh_action_cache(obs)
        return list(self._cached_actions)

    def _refresh_action_cache(self, obs: CanonicalPolicyInput) -> None:
        self._cached_actions = self._run_inference(obs)
        self._cached_index = 0

    def _run_inference(
        self,
        obs: CanonicalPolicyInput,
    ) -> list[HolobrainAction]:
        adapter = self._ensure_adapter(obs)
        model_input = adapter.build_model_input(obs)
        model_output = self._pipeline(model_input)
        _, device = self._observation_batch_info(obs)
        return adapter.build_action_sequence(
            model_output,
            obs,
            device=device,
            valid_action_step=self.cfg.valid_action_step,
        )

    @staticmethod
    def _validate_observation_batch(obs: CanonicalPolicyInput) -> None:
        batch_size, _ = HolobrainPolicy._observation_batch_info(obs)
        if batch_size != 1:
            raise ValueError(
                "HolobrainPolicy currently supports single environment only"
            )

    @staticmethod
    def _observation_batch_info(
        obs: CanonicalPolicyInput,
    ) -> tuple[int, torch.device | str]:
        try:
            manipulator_obs = next(iter(obs.manipulators.values()))
        except StopIteration as exc:
            raise ValueError(
                "Holobrain observation must contain at least one manipulator"
            ) from exc
        joint_position = manipulator_obs["joint_position"]
        return int(joint_position.shape[0]), joint_position.device

    @classmethod
    def policy_requirement(cls) -> PolicyRequirement:
        return PolicyRequirement(
            required_camera_modalities=("rgb", "depth", "intrinsic", "pose"),
            min_camera_count=1,
            min_manipulator_count=1,
            require_instruction=True,
        )

    @staticmethod
    def _build_adapter(*, embodiment_type: str) -> HolobrainAdapter:
        return HolobrainAdapter(embodiment_type=embodiment_type)

    def _ensure_adapter(self, obs: CanonicalPolicyInput) -> HolobrainAdapter:
        layout = obs.action_layout
        runtime_embodiment_type = getattr(layout, "embodiment_type", None)
        if not isinstance(runtime_embodiment_type, str):
            raise ValueError(
                "Holobrain observation requires action_layout with "
                "embodiment_type"
            )
        if self._embodiment_type is None:
            self._embodiment_type = runtime_embodiment_type
            self._adapter = self._build_adapter(
                embodiment_type=runtime_embodiment_type
            )
            return self._adapter
        if runtime_embodiment_type != self._embodiment_type:
            raise ValueError(
                "HolobrainPolicy is already bound to embodiment_type "
                f"{self._embodiment_type!r}, got "
                f"{runtime_embodiment_type!r}."
            )
        if self._adapter is None:
            self._adapter = self._build_adapter(
                embodiment_type=self._embodiment_type
            )
        return self._adapter

    @staticmethod
    def _load_pipeline(cfg: "HolobrainPolicyCfg") -> Any:
        from robo_orchard_lab.models.holobrain.pipeline import (
            HoloBrainInferencePipeline,
        )

        model_dir = cfg.model_dir or os.getenv(_MODEL_DIR_ENV_VAR)
        if not model_dir:
            raise ValueError(
                "Holobrain model_dir must be set in the config or via "
                f"{_MODEL_DIR_ENV_VAR}"
            )
        logger.info("Resolved Holobrain model_dir: %s", model_dir)
        device = cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return HoloBrainInferencePipeline.load_pipeline(
            directory=model_dir,
            device=device,
            load_weights=True,
            load_impl="native",
            inference_prefix=cfg.inference_prefix,
        )


class HolobrainPolicyCfg(PolicyConfig[HolobrainPolicy]):
    """Config for :class:`HolobrainPolicy`."""

    class_type: ClassType[HolobrainPolicy] = HolobrainPolicy

    model_dir: str | None = None
    logging_tag: str | None = None
    inference_prefix: str
    embodiment_type: str | None = None
    device: str | None = None
    valid_action_step: int | None = None

    @field_validator("valid_action_step")
    @classmethod
    def _validate_valid_action_step(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("valid_action_step must be greater than 0")
        return value
