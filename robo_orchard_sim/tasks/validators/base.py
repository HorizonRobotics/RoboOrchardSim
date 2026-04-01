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

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from robo_orchard_sim.envs.manager_based_env import IsaacManagerBasedEnv


@dataclass
class ValidatorOutput:
    """Result from evaluating a validator."""

    success: bool  # Binary task success
    progress: float  # Progress score 0.0 - 1.0
    metrics: dict[str, Any]  # Additional metrics for logging


class Validator:
    """Validators compute success/progress by inspecting simulation state.

    They're called after each step and on reset to populate info dict.

    """

    def __init__(
        self,
        actors: list[str] | str,
        criteria: list[Callable | tuple[Callable, list[int]]],
        criteria_name: list[str],
        **kwargs,
    ):
        """Initialize the validator rubric.

        Args:
            actors (list[str] | str): Object identifiers used for scene
                lookup.
            criteria (list[Callable | tuple[Callable, list[int]]]): Criteria
                callables or dependency tuples used to compute progress.
            criteria_name (list[str]): Human-readable names for each
                criterion.
            **kwargs: Task-specific validator configuration.
        """
        actor_list = actors if isinstance(actors, list) else [actors]
        self.actors = actor_list
        self.config = kwargs
        self.criteria = criteria
        self.criteria_reached = [False] * len(criteria)
        self.criteria_name = criteria_name

        # -------------------------
        # internal cache (private)
        # -------------------------
        self._actor_tags: dict[str, dict[str, Any]] = {}
        self._actor_category: dict[str, str] = {}
        self._actor_type: dict[str, str] = {}
        self._actor_uuid: dict[str, str] = {}
        self._init_state: dict[str, Any] = {}
        self._final_state: dict[str, Any] = {}

    def _get_semantic_tags(self, scene):
        """Cache semantic information once."""
        self._actor_tags.clear()
        self._actor_category.clear()
        self._actor_type.clear()
        self._actor_uuid.clear()

        for actor in self.actors:
            tags = dict(scene[actor].cfg.spawn.semantic_tags)

            self._actor_tags[actor] = tags
            self._actor_category[actor] = tags.get("class", "unknown")
            self._actor_type[actor] = tags.get("actor_type", "unknown")
            self._actor_uuid[actor] = tags.get("uuid", "unknown")

    @property
    def actor_tags(self):
        return self._actor_tags

    @property
    def actor_category(self):
        return self._actor_category

    @property
    def actor_type(self):
        return self._actor_type

    @property
    def actor_uuid(self):
        return self._actor_uuid

    @property
    def init_state(self):
        return self._init_state

    @property
    def final_state(self):
        return self._final_state

    def set_init_state(self, scene):
        """Capture initial state + semantic info."""
        self._get_semantic_tags(scene)

        self._init_state = {
            actor: self._get_world_pose(scene[actor]) for actor in self.actors
        }

    def set_final_state(self, scene):
        """Capture final state."""
        self._final_state = {
            actor: self._get_world_pose(scene[actor]) for actor in self.actors
        }

    def _get_world_pose(self, obj):
        obj_pos = obj.data.root_pos_w.cpu().numpy()
        obj_quat = obj.data.root_quat_w.cpu().numpy()  # w, x, y, z
        return np.concatenate([obj_pos, obj_quat], axis=-1)

    def _call_criterion(self, criterion: Callable, env, env_idx: int) -> bool:
        """Call criteria with env_idx when the callable supports it."""
        parameters = inspect.signature(criterion).parameters.values()
        supports_env_idx = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            or parameter.name == "env_idx"
            for parameter in parameters
        )
        if supports_env_idx:
            return bool(criterion(env, env_idx=env_idx))
        return bool(criterion(env))

    def evaluate(
        self, env: "IsaacManagerBasedEnv", env_idx: int = 0
    ) -> ValidatorOutput:
        """Evaluate the current state and cumulative progress.

        Supports criteria with optional dependencies.
        Criteria can be:
            - callable (no dependency, can be achieved in any order)
            - (callable, [dep_indices]) (only counts if all deps are met)
        This allows for some to require others, but leaves most unconstrained.

        Returns the current-frame criterion status in
        ``metrics["criteria_met_now"]`` and the cumulative latched state in
        ``metrics["criteria_reached"]``.

        """
        metrics = {}
        num_criteria = len(self.criteria)
        metrics["criteria_met_now"] = {}
        metrics["criteria_reached"] = {}

        criteria_met_now = []
        for idx, c in enumerate(self.criteria):
            # Check if c is (callable, [deps]), else treat as callable only
            if isinstance(c, tuple):
                fn, deps = c
                # Only evaluate if all deps ever reached
                deps_met = all(self.criteria_reached[d] for d in deps)
                result = (
                    self._call_criterion(fn, env, env_idx)
                    if deps_met
                    else False
                )
            else:
                fn = c
                result = self._call_criterion(fn, env, env_idx)
            # Update max-ever reached for this criterion
            self.criteria_reached[idx] = self.criteria_reached[idx] or bool(
                result
            )
            criteria_met_now.append(bool(result))
            metrics["criteria_met_now"][self.criteria_name[idx]] = bool(result)

        num_met_now = sum(criteria_met_now)
        num_reached_ever = sum(self.criteria_reached)
        progress = num_reached_ever / num_criteria if num_criteria > 0 else 0.0
        metrics["criteria_met_now_count"] = num_met_now
        metrics["criteria_ever_reached"] = num_reached_ever
        metrics["criteria_total"] = num_criteria
        for idx, c in enumerate(self.criteria_name):
            metrics["criteria_reached"][c] = self.criteria_reached[idx]

        success = num_reached_ever == num_criteria
        return ValidatorOutput(
            success=success, progress=progress, metrics=metrics
        )

    def reset(self):
        """Called when environment resets. Override for stateful rubrics."""
        self.criteria_reached = [False] * len(self.criteria)

    def check_success(self) -> bool:
        """Return whether all criteria have been met."""
        return all(self.criteria_reached)
