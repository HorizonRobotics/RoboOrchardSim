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
from typing import Any, TypeAlias

import gymnasium as gym
from robo_orchard_core.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_core.utils.config import ClassType

from robo_orchard_sim.contracts.joint_command import UnifiedJointCommand
from robo_orchard_sim.contracts.policy_binding import CanonicalPolicyInput
from robo_orchard_sim.policy.action_layout import (
    CompiledActionLayout,
    ManipulatorActionSpec,
    validate_action_layout_compatibility,
)

__all__ = ["DummyPolicy", "DummyPolicyCfg"]

DummyAction: TypeAlias = UnifiedJointCommand


class DummyPolicy(PolicyMixin[CanonicalPolicyInput, DummyAction]):
    """Policy that commands every joint to hold its current position.

    Useful for exercising the observation -> action -> environment path on
    any embodiment without a trained model. Joint names come from the
    canonical action layout, so no embodiment is hard-coded here.
    """

    cfg: "DummyPolicyCfg"

    def __init__(
        self,
        cfg: "DummyPolicyCfg",
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            observation_space=observation_space,
            action_space=action_space,
        )

    def reset(self, *args: Any, **kwargs: Any) -> None:
        """Reset nothing; each action depends only on the latest input."""
        del args, kwargs

    def act(self, obs: CanonicalPolicyInput) -> DummyAction:
        """Return a command holding the observed pose of every joint.

        Args:
            obs (CanonicalPolicyInput): Canonical input carrying the current
                joint positions and the compiled action layout.

        Returns:
            DummyAction: Joint command repeating the observed positions.
        """
        layout = obs.action_layout
        if not isinstance(layout, CompiledActionLayout):
            raise ValueError(
                "DummyPolicy observation requires a compiled action layout"
            )
        validate_action_layout_compatibility(
            manipulator_observations=obs.manipulators,
            layout=layout,
            context="DummyPolicy input",
        )
        return UnifiedJointCommand.merge(
            *(
                self._hold_pose(
                    obs.manipulators[slot],
                    manipulator=layout.manipulators[slot],
                )
                for slot in layout.manipulator_order
            )
        )

    @staticmethod
    def _hold_pose(
        manipulator_obs: dict[str, Any],
        *,
        manipulator: ManipulatorActionSpec,
    ) -> UnifiedJointCommand:
        """Label one manipulator's observed joint positions as a command."""
        joint_names = (
            manipulator.arm_joint_names + manipulator.gripper_joint_names
        )
        joint_position = manipulator_obs["joint_position"]
        return UnifiedJointCommand(
            values=joint_position[:, : len(joint_names)].clone(),
            joint_names=joint_names,
        )


class DummyPolicyCfg(PolicyConfig[DummyPolicy]):
    """Config for :class:`DummyPolicy`."""

    class_type: ClassType[DummyPolicy] = DummyPolicy
