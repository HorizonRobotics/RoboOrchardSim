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
from pydantic import BaseModel, Field, field_validator
from robo_orchard_core.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_core.utils.config import ClassType

from robo_orchard_sim.contracts.policy_binding import (
    CanonicalPolicyInput,
    PolicyRequirement,
)
from robo_orchard_sim.policy.groot.adapter import (
    DEFAULT_ARM_SPECS,
    DEFAULT_LANGUAGE_KEY,
    DEFAULT_VIDEO_MAP,
    GrootAction,
    GrootAdapter,
    GrootArmSpec,
)
from robo_orchard_sim.policy.groot.client import GrootZmqClient


class GrootArmMapCfg(BaseModel):
    """Map one canonical manipulator slot to GR00T arm/gripper keys."""

    manipulator_slot: str
    arm_key: str
    gripper_key: str
    arm_relative: bool = False


def _default_arm_cfgs() -> list[GrootArmMapCfg]:
    return [
        GrootArmMapCfg(
            manipulator_slot=spec.manipulator_slot,
            arm_key=spec.arm_key,
            gripper_key=spec.gripper_key,
            arm_relative=spec.arm_relative,
        )
        for spec in DEFAULT_ARM_SPECS
    ]


class GrootPolicy(PolicyMixin[CanonicalPolicyInput, GrootAction]):
    """Remote GR00T policy: a ZMQ client to ``run_gr00t_server.py``."""

    cfg: "GrootPolicyCfg"

    def __init__(
        self,
        cfg: "GrootPolicyCfg",
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            observation_space=observation_space,
            action_space=action_space,
        )
        self._adapter = GrootAdapter(
            video_map=cfg.video_map,
            arm_specs=tuple(
                GrootArmSpec(
                    manipulator_slot=arm.manipulator_slot,
                    arm_key=arm.arm_key,
                    gripper_key=arm.gripper_key,
                    arm_relative=arm.arm_relative,
                )
                for arm in cfg.arms
            ),
            language_key=cfg.language_key,
            default_instruction=cfg.instruction,
        )
        self._client = self._create_client(cfg)
        self._cached_actions: list[GrootAction] = []
        self._cached_index = 0

    def reset(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._cached_actions = []
        self._cached_index = 0
        self._client.reset()

    def act(self, obs: CanonicalPolicyInput) -> GrootAction:
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
    ) -> list[GrootAction]:
        """Return a freshly inferred action horizon for remote batch use."""
        sequence = self._run_inference(obs)
        self._cached_actions = []
        self._cached_index = 0
        return sequence

    def _run_inference(
        self,
        obs: CanonicalPolicyInput,
    ) -> list[GrootAction]:
        self._validate_observation_batch(obs)
        model_input = self._adapter.build_model_input(obs)
        action, _ = self._client.get_action(model_input)
        _, device = self._observation_batch_info(obs)
        return self._adapter.build_action_sequence(
            action,
            obs,
            device=device,
            open_loop_horizon=self.cfg.open_loop_horizon,
        )

    @staticmethod
    def _validate_observation_batch(obs: CanonicalPolicyInput) -> None:
        batch_size, _ = GrootPolicy._observation_batch_info(obs)
        if batch_size != 1:
            raise ValueError(
                "GrootPolicy currently supports single environment only"
            )

    @staticmethod
    def _observation_batch_info(
        obs: CanonicalPolicyInput,
    ) -> tuple[int, torch.device | str]:
        try:
            manipulator_obs = next(iter(obs.manipulators.values()))
        except StopIteration as exc:
            raise ValueError(
                "GR00T observation must contain at least one manipulator"
            ) from exc
        joint_position = manipulator_obs["joint_position"]
        return int(joint_position.shape[0]), joint_position.device

    @classmethod
    def policy_requirement(cls) -> PolicyRequirement:
        return PolicyRequirement(
            required_camera_modalities=("rgb",),
            min_camera_count=1,
            min_manipulator_count=1,
            require_instruction=False,
        )

    @staticmethod
    def _create_client(cfg: "GrootPolicyCfg") -> GrootZmqClient:
        return GrootZmqClient(
            host=cfg.host,
            port=cfg.port,
            timeout_ms=cfg.timeout_ms,
            api_token=cfg.api_token,
        )


class GrootPolicyCfg(PolicyConfig[GrootPolicy]):
    """Config for :class:`GrootPolicy`."""

    class_type: ClassType[GrootPolicy] = GrootPolicy

    host: str = "127.0.0.1"
    port: int = 5555
    timeout_ms: int = 15000
    api_token: str | None = None
    open_loop_horizon: int | None = None
    instruction: str | None = None
    logging_tag: str | None = None
    language_key: str = DEFAULT_LANGUAGE_KEY
    video_map: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_VIDEO_MAP)
    )
    arms: list[GrootArmMapCfg] = Field(default_factory=_default_arm_cfgs)

    @field_validator("open_loop_horizon")
    @classmethod
    def _validate_open_loop_horizon(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("open_loop_horizon must be greater than 0")
        return value
