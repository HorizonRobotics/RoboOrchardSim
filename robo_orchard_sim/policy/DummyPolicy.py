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

"""Dummy policy implementation for pipeline-level action flow checks."""

from __future__ import annotations
from typing import Any

import gymnasium as gym
import torch
from robo_orchard_core.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_core.utils.config import ClassType

__all__ = ["DummyPolicy", "DummyPolicyCfg"]

DummyAction = dict[str, torch.Tensor] | torch.Tensor


class DummyPolicy(PolicyMixin[dict[str, Any], DummyAction]):
    """Policy that emits fixed actions for end-to-end flow tests."""

    cfg: DummyPolicyCfg

    def __init__(
        self,
        cfg: DummyPolicyCfg,
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            observation_space=observation_space,
            action_space=action_space,
        )
        self._cached_action: DummyAction | None = None
        self._remaining_inference_steps = 0

    def reset(self, *args: Any, **kwargs: Any) -> None:
        """Clear cached actions for a new rollout."""
        del args, kwargs
        self._cached_action = None
        self._remaining_inference_steps = 0

    def act(self, obs: dict[str, Any]) -> DummyAction:
        """Return a fixed action payload.

        Args:
            obs (dict[str, Any]): Input observations used only to infer batch
                size when possible.

        Returns:
            DummyAction: A fixed action tensor or action dict.
        """
        left_joint_position = obs["/robot"]["left_joint_position"]
        device = left_joint_position.device

        actions = {
            "left_robot_joint_position": torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device
            ),
            "left_robot_gripper_control": torch.tensor(
                [[0.0, 0.0]], device=device
            ),
            "right_robot_joint_position": torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device
            ),
            "right_robot_gripper_control": torch.tensor(
                [[0.0, 0.0]], device=device
            ),
        }
        return actions


class DummyPolicyCfg(PolicyConfig[DummyPolicy]):
    """Config for :class:`DummyPolicy`."""

    class_type: ClassType[DummyPolicy] = DummyPolicy

    inference_steps: int = 32
    """How many consecutive ``act`` calls reuse one generated action."""
