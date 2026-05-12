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
from typing import Any

import gymnasium as gym
import torch
from pydantic import field_validator
from robo_orchard_core.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_core.utils.config import ClassType

from robo_orchard_sim.policy.holobrain.adapter import HolobrainAdapter

HolobrainAction = dict[str, torch.Tensor]
_MODEL_DIR_ENV_VAR = "ROBO_ORCHARD_HOLOBRAIN_MODEL_DIR"


class HolobrainPolicy(PolicyMixin[dict[str, Any], HolobrainAction]):
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
        self._adapter = self._build_adapter(cfg)
        self._pipeline = self._load_pipeline(cfg)
        self._pipeline.model.eval()
        self._cached_actions: list[HolobrainAction] = []
        self._cached_index = 0

    def reset(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._cached_actions = []
        self._cached_index = 0

    def act(self, obs: dict[str, Any]) -> HolobrainAction:
        """Return one action, reusing the local cached horizon if possible."""
        self._validate_observation_batch(obs)
        if self._cached_index >= len(self._cached_actions):
            self._cached_actions = self._get_fresh_action_sequence(obs)
            self._cached_index = 0
        action = self._cached_actions[self._cached_index]
        self._cached_index += 1
        return action

    def act_sequence(self, obs: dict[str, Any]) -> list[HolobrainAction]:
        """Return a freshly inferred action horizon for remote batch use."""
        sequence = self._get_fresh_action_sequence(obs)
        self._cached_actions = []
        self._cached_index = 0
        return sequence

    def _get_fresh_action_sequence(
        self, obs: dict[str, Any]
    ) -> list[HolobrainAction]:
        self._validate_observation_batch(obs)
        self._refresh_action_cache(obs)
        return list(self._cached_actions)

    def _refresh_action_cache(self, obs: dict[str, Any]) -> None:
        self._cached_actions = self._run_inference(obs)
        self._cached_index = 0

    def _run_inference(self, obs: dict[str, Any]) -> list[HolobrainAction]:
        model_input = self._adapter.build_model_input(obs)
        model_output = self._pipeline(model_input)
        device = obs["/robot"]["left_joint_position"].device
        return self._adapter.build_action_sequence(
            model_output,
            device=device,
            valid_action_step=self.cfg.valid_action_step,
        )

    @staticmethod
    def _validate_observation_batch(obs: dict[str, Any]) -> None:
        batch_size = int(obs["/robot"]["left_joint_position"].shape[0])
        if batch_size != 1:
            raise ValueError(
                "HolobrainPolicy currently supports single environment only"
            )

    @staticmethod
    def _build_adapter(cfg: "HolobrainPolicyCfg") -> HolobrainAdapter:
        return HolobrainAdapter(joint_num=cfg.joint_num)

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
    joint_num: int = 7
    device: str | None = None
    valid_action_step: int | None = None

    @field_validator("valid_action_step")
    @classmethod
    def _validate_valid_action_step(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("valid_action_step must be greater than 0")
        return value
