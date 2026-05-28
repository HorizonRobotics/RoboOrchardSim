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

"""RoboOrchard policy wrapper for remote Motus inference."""

from __future__ import annotations
import uuid
from typing import Any

import gymnasium as gym
import torch
from pydantic import Field
from robo_orchard_core.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_core.utils.config import ClassType

from robo_orchard_sim.orchard_env.joint_command import UnifiedJointCommand
from robo_orchard_sim.policy.motus.adapter import (
    DEFAULT_CAMERA_MAPPING,
    MotusAdapter,
)
from robo_orchard_sim.policy.motus.client import MotusClient
from robo_orchard_sim.policy.schema import (
    CanonicalPolicyInput,
    PolicyRequirement,
)


class MotusPolicy(PolicyMixin[CanonicalPolicyInput, UnifiedJointCommand]):
    """Remote Motus policy with canonical observation/action adaptation."""

    cfg: "MotusPolicyCfg"

    def __init__(
        self,
        cfg: "MotusPolicyCfg",
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            observation_space=observation_space,
            action_space=action_space,
        )
        self._adapter = MotusAdapter(camera_mapping=cfg.camera_mapping)
        self._client = client or MotusClient(host=cfg.host, port=cfg.port)
        self._cached_actions: list[UnifiedJointCommand] = []
        self._cached_index = 0
        self._session_id = self._new_session_id()

    def reset(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._cached_actions = []
        self._cached_index = 0
        self._session_id = self._new_session_id()
        self._client.reset({"session_id": self._session_id})

    def act(self, obs: CanonicalPolicyInput) -> UnifiedJointCommand:
        if self._cached_index >= len(self._cached_actions):
            self._cached_actions = self._request_action_sequence(obs)
            self._cached_index = 0
        action = self._cached_actions[self._cached_index]
        self._cached_index += 1
        return action

    @property
    def logging_tag(self) -> str | None:
        return self.cfg.logging_tag

    def policy_requirement(self) -> PolicyRequirement:
        return PolicyRequirement(
            required_camera_modalities=("rgb",),
            min_camera_count=1,
            min_manipulator_count=1,
            require_instruction=True,
        )

    def close(self) -> None:
        self._client.close()

    def __del__(self) -> None:
        self.close()

    def _request_action_sequence(
        self,
        obs: CanonicalPolicyInput,
    ) -> list[UnifiedJointCommand]:
        request = self._adapter.build_request(obs, session_id=self._session_id)
        actions = self._client.infer(request)
        device = self._infer_action_device(obs)
        return self._adapter.build_action_sequence(
            actions,
            obs,
            device=device,
            valid_action_step=self.cfg.valid_action_step,
        )

    @staticmethod
    def _new_session_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _infer_action_device(obs: CanonicalPolicyInput) -> torch.device | str:
        for manipulator_obs in obs.manipulators.values():
            joint_position = manipulator_obs.get("joint_position")
            if isinstance(joint_position, torch.Tensor):
                return joint_position.device
        return "cuda" if torch.cuda.is_available() else "cpu"


class MotusPolicyCfg(PolicyConfig[MotusPolicy]):
    """Config for :class:`MotusPolicy`."""

    class_type: ClassType[MotusPolicy] = MotusPolicy

    host: str = "localhost"
    port: int = 8000
    logging_tag: str | None = None
    valid_action_step: int | None = None
    camera_mapping: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_CAMERA_MAPPING)
    )
