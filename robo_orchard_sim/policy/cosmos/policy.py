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
from typing import Any

import gymnasium as gym
import torch
from pydantic import Field, field_validator
from robo_orchard_core.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_core.utils.config import ClassType

from robo_orchard_sim.contracts.policy_binding import (
    CanonicalPolicyInput,
    PolicyRequirement,
)
from robo_orchard_sim.policy.cosmos.adapter import (
    DEFAULT_CAMERA_MAP,
    CosmosAction,
    CosmosAdapter,
)
from robo_orchard_sim.policy.cosmos.client import CosmosWsClient


class CosmosPolicy(PolicyMixin[CanonicalPolicyInput, CosmosAction]):
    """Remote Cosmos policy: an openpi-protocol websocket client."""

    cfg: "CosmosPolicyCfg"

    def __init__(
        self,
        cfg: "CosmosPolicyCfg",
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            observation_space=observation_space,
            action_space=action_space,
        )
        self._adapter = CosmosAdapter(
            camera_map=cfg.camera_map,
            manipulator_slot=cfg.manipulator_slot,
            default_instruction=cfg.instruction,
        )
        self._client = self._create_client(cfg)
        self._cached_actions: list[CosmosAction] = []
        self._cached_index = 0

    def reset(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._cached_actions = []
        self._cached_index = 0
        self._client.reset()

    def act(self, obs: CanonicalPolicyInput) -> CosmosAction:
        """Return one action, reusing the cached chunk if available."""
        self._validate_observation_batch(obs)
        if self._cached_index >= len(self._cached_actions):
            self._cached_actions = self._run_inference(obs)
            self._cached_index = 0
        action = self._cached_actions[self._cached_index]
        self._cached_index += 1
        return action

    def act_sequence(
        self,
        obs: CanonicalPolicyInput,
    ) -> list[CosmosAction]:
        """Return a freshly inferred action horizon for remote batch use."""
        sequence = self._run_inference(obs)
        self._cached_actions = []
        self._cached_index = 0
        return sequence

    def _run_inference(
        self,
        obs: CanonicalPolicyInput,
    ) -> list[CosmosAction]:
        self._validate_observation_batch(obs)
        request = self._adapter.build_request(obs)
        response = self._client.infer(request)
        if isinstance(response, dict):
            if self.cfg.action_key not in response:
                raise ValueError(
                    f"Cosmos response is missing action key "
                    f"{self.cfg.action_key!r}; got {sorted(response)}."
                )
            action = response[self.cfg.action_key]
        else:
            action = response
        _, device = self._observation_batch_info(obs)
        return self._adapter.build_action_sequence(
            action,
            obs,
            device=device,
            open_loop_horizon=self.cfg.open_loop_horizon,
        )

    @staticmethod
    def _validate_observation_batch(obs: CanonicalPolicyInput) -> None:
        batch_size, _ = CosmosPolicy._observation_batch_info(obs)
        if batch_size != 1:
            raise ValueError(
                "CosmosPolicy currently supports single environment only"
            )

    @staticmethod
    def _observation_batch_info(
        obs: CanonicalPolicyInput,
    ) -> tuple[int, torch.device | str]:
        try:
            manipulator_obs = next(iter(obs.manipulators.values()))
        except StopIteration as exc:
            raise ValueError(
                "Cosmos observation must contain at least one manipulator"
            ) from exc
        joint_position = manipulator_obs["joint_position"]
        return int(joint_position.shape[0]), joint_position.device

    @classmethod
    def policy_requirement(cls) -> PolicyRequirement:
        return PolicyRequirement(
            required_camera_modalities=("rgb",),
            min_camera_count=len(DEFAULT_CAMERA_MAP),
            min_manipulator_count=1,
            require_instruction=False,
        )

    @staticmethod
    def _create_client(cfg: "CosmosPolicyCfg") -> CosmosWsClient:
        return CosmosWsClient(host=cfg.host, port=cfg.port)


class CosmosPolicyCfg(PolicyConfig[CosmosPolicy]):
    """Config for :class:`CosmosPolicy`."""

    class_type: ClassType[CosmosPolicy] = CosmosPolicy

    host: str = "127.0.0.1"
    port: int = 8000
    open_loop_horizon: int | None = None
    instruction: str | None = None
    action_key: str = "action"
    manipulator_slot: str = "single_arm"
    camera_map: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_CAMERA_MAP)
    )

    @field_validator("open_loop_horizon")
    @classmethod
    def _validate_open_loop_horizon(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("open_loop_horizon must be greater than 0")
        return value
