# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
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

"""Minimal base interfaces for atomic action executors."""

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch
from robo_orchard_core.utils.config import ClassConfig

from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ResolvedManipulatorProfile,
)
from robo_orchard_sim.tasks.trajs_gen.manipulator_resolver import (
    ManipulatorBindingContext,
    ManipulatorResolver,
)
from robo_orchard_sim.utils.config import ClassType_co


@dataclass
class ObjectInfo:
    name: str
    mode: Literal["active", "passive"]
    action: str
    part: str


@dataclass
class Trajectories:
    trajectories: list[torch.Tensor]
    success: bool
    resolved_manipulator: ResolvedManipulatorProfile


class BaseExecutor(metaclass=ABCMeta):
    """Base executor interface for one atomic action.

    Concrete executors implement :meth:`plan`.  Callers pass the
    manipulator binding context explicitly so each planning request declares
    the sequence-scoped resolver state it should use.
    """

    cfg: "BaseExecutorCfg"

    def __init__(self, cfg: "BaseExecutorCfg") -> None:
        self.cfg = cfg
        self.last_resolved_manipulator: ResolvedManipulatorProfile | None = (
            None
        )

    def reset(self) -> None:
        """Reset executor state for a new action sequence."""
        self.last_resolved_manipulator = None

    @abstractmethod
    def plan(
        self,
        env: Any,
        context: ManipulatorBindingContext,
    ) -> Trajectories:
        """Plan one atomic action.

        Args:
            env: Runtime environment used by the executor during planning.
            context: Sequence-scoped manipulator binding context.

        Returns:
            tuple[AtomicActionTrajectories, bool]:
                - Planned batched trajectories. Each element is one
                  ``torch.Tensor`` trajectory for one environment.
                - Whether the planning stage succeeded.
        """

        raise NotImplementedError


class BaseExecutorCfg(ClassConfig[BaseExecutor]):
    """Configuration for :class:`BaseExecutor`."""

    class_type: ClassType_co[BaseExecutor] = BaseExecutor
    robot_info: ManipulatorResolver
    priority: int = 0
    action_type: str = ""

    def __call__(self, **kwargs: Any) -> BaseExecutor:
        return self.class_type(self, **kwargs)

    def resolve_manipulator_info(
        self,
        env: Any,
        context: ManipulatorBindingContext | None = None,
    ) -> ResolvedManipulatorProfile:
        """Resolve the configured manipulator into runtime ids."""
        if self.robot_info is None:
            raise ValueError("BaseExecutorCfg.robot_info must not be None.")
        return self.robot_info.resolve(env=env, context=context)
