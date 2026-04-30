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

"""Runtime manipulator selection helpers for trajectory executors."""

from __future__ import annotations
from typing import Any, Callable, Protocol, runtime_checkable

from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ResolvedManipulatorProfile,
)


@runtime_checkable
class ManipulatorResolver(Protocol):
    """Object that can resolve a manipulator against a runtime env."""

    def resolve(
        self,
        env: Any,
        context: "ManipulatorBindingContext | None" = None,
    ) -> ResolvedManipulatorProfile:
        """Resolve this object for the provided runtime env.

        Args:
            env: Runtime environment.
            context: Optional binding context for caching resolved
                manipulators across related actions.
        """
        ...


ManipulatorPredicate = Callable[[Any], bool]


class ManipulatorBindingContext:
    """Runtime cache for manipulator bindings and planner instances."""

    def __init__(self) -> None:
        self._selected: dict[str, ManipulatorResolver] = {}
        self._planner_instances: dict[tuple[str, str], Any] = {}

    def resolve_once(
        self,
        binding_key: str,
        selector: "PredicateManipulatorResolver",
        env: Any,
    ) -> ResolvedManipulatorProfile:
        """Resolve a selector once and reuse it for this binding key."""
        if binding_key not in self._selected:
            self._selected[binding_key] = selector.select(env)
        return self._selected[binding_key].resolve(env=env, context=self)

    def resolve_planner_instance(
        self,
        robot_name: str,
        manipulator_name: str,
        planner_cfg: Any,
        env_nums: int,
    ) -> Any:
        """Create or reuse one planner instance per robot manipulator."""
        key = (robot_name, manipulator_name)
        if key not in self._planner_instances:
            self._planner_instances[key] = planner_cfg.class_type(
                planner_cfg,
                env_nums,
            )
        return self._planner_instances[key]

    def reset(self, *, clear_planner_instances: bool = False) -> None:
        """Clear context state for the next action sequence.

        Args:
            clear_planner_instances: Whether to also discard cached planner
                instances. Defaults to ``False`` so expensive planner setup can
                be reused across independent action sequences.
        """
        self._selected.clear()
        if clear_planner_instances:
            self._planner_instances.clear()


class PredicateManipulatorResolver:
    """Resolve one of two manipulator configs from an env predicate."""

    def __init__(
        self,
        predicate: ManipulatorPredicate,
        true_robot_info: ManipulatorResolver,
        false_robot_info: ManipulatorResolver,
    ) -> None:
        self.predicate = predicate
        self.true_robot_info = self._validate_robot_info(
            true_robot_info,
            "true_robot_info",
        )
        self.false_robot_info = self._validate_robot_info(
            false_robot_info,
            "false_robot_info",
        )

    def select(self, env: Any) -> ManipulatorResolver:
        """Return the manipulator selected by the predicate result."""
        result = self.predicate(env)
        if not isinstance(result, bool):
            raise TypeError("Manipulator predicate must return bool.")
        if result:
            return self.true_robot_info
        return self.false_robot_info

    def _validate_robot_info(
        self,
        robot_info: ManipulatorResolver,
        field_name: str,
    ) -> ManipulatorResolver:
        if not isinstance(robot_info, ManipulatorResolver):
            raise TypeError(
                f"PredicateManipulatorResolver.{field_name} must provide "
                "resolve(env)."
            )
        return robot_info

    def resolve(
        self,
        env: Any,
        context: ManipulatorBindingContext | None = None,
    ) -> ResolvedManipulatorProfile:
        """Resolve the selected manipulator against the runtime env."""
        return self.select(env).resolve(env=env, context=context)


class BoundManipulatorResolver:
    """Resolve a selector once per binding context lifecycle."""

    def __init__(
        self,
        binding_key: str,
        selector: PredicateManipulatorResolver,
    ) -> None:
        if not binding_key:
            raise ValueError("BoundManipulatorResolver.binding_key is empty.")
        self.binding_key = binding_key
        self.selector = selector

    def resolve(
        self,
        env: Any,
        context: ManipulatorBindingContext | None = None,
    ) -> ResolvedManipulatorProfile:
        """Resolve once when context is available, otherwise delegate."""
        if context is None:
            return self.selector.resolve(env=env)
        return context.resolve_once(
            binding_key=self.binding_key,
            selector=self.selector,
            env=env,
        )
