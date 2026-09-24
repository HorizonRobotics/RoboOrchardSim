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

from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from robo_orchard_sim.task_components.validators.metrics import MetricStore

if TYPE_CHECKING:
    from robo_orchard_sim.ext.envs.manager_based_env import (
        IsaacManagerBasedEnv,
    )
    from robo_orchard_sim.task_components.validators.checkers import (
        SceneCheckerSuite,
    )
    from robo_orchard_sim.task_components.validators.context import (
        ValidatorContext,
    )


@dataclass
class ValidatorOutput:
    """Result from evaluating a validator."""

    success: bool  # Binary task success
    progress: float  # Progress score 0.0 - 1.0
    metrics: dict[str, Any]  # Additional metrics for logging
    # Actual rubric contributions, before the success-to-full-score override.
    # Validators with only a terminal rubric leave this empty.
    stage_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GripperRange:
    """A gripper joint's open/close positions, for the open/close check."""

    name: str
    open_val: float
    close_val: float


# ---------------------------------------------------------------------
# Validator runtime and rubrics
# ---------------------------------------------------------------------


class Validator:
    """Validators compute success/progress by inspecting simulation state.

    They're called after each step and on reset to populate info dict.

    """

    fixed_horizon: ClassVar[bool] = False

    def __init__(
        self,
        criteria: list[Callable | tuple[Callable, list[int]]],
        criteria_name: list[str],
        success_criteria: list[int] | None = None,
        *,
        progress_criteria: list[int] | None = None,
        independent_criteria: list[Callable] | None = None,
        context: ValidatorContext | None = None,
        checker_suite: SceneCheckerSuite | None = None,
        metric_store: MetricStore | None = None,
        **kwargs,
    ):
        """Initialize the validator rubric.

        Args:
            criteria (list[Callable | tuple[Callable, list[int]]]): Criteria
                callables or dependency tuples used to compute progress.
            criteria_name (list[str]): Human-readable names for each
                criterion.
            success_criteria (list[int] | None): Indices of the criteria
                that decide success. A task may treat some criteria as
                milestones rather than requirements. Defaults to requiring
                all of them.
            progress_criteria (list[int] | None): Indices of the criteria
                that contribute to progress. Diagnostic criteria remain in
                the output but can be excluded from the progress score.
                Defaults to including all criteria.
            independent_criteria (list[Callable] | None): Optional independent
                observations, one per named criterion. Stateful selectors must
                be separate instances from scoring selectors so diagnostic
                evaluation cannot advance scoring dwell counters.
            context (ValidatorContext | None): Runtime context used by
                scene-wide checkers.
            checker_suite (SceneCheckerSuite | None): Optional scene-wide
                checker suite evaluated before criteria.
            metric_store (MetricStore | None): Optional store shared with
                metric selectors.
            **kwargs: Task-specific validator configuration.
        """
        scene_runtime = (context, checker_suite, metric_store)
        if any(value is not None for value in scene_runtime) and not all(
            value is not None for value in scene_runtime
        ):
            raise ValueError(
                "context, checker_suite, and metric_store must be "
                "provided together."
            )
        if independent_criteria is not None and len(
            independent_criteria
        ) != len(criteria_name):
            raise ValueError(
                "independent_criteria must match criteria_name length."
            )
        self.independent_criteria = independent_criteria
        self.independent_criteria_reached = [False] * len(criteria_name)
        self.config = kwargs
        self.criteria = criteria
        self.criteria_reached = [False] * len(criteria)
        self.criteria_name = criteria_name
        self.context = context
        self.checker_suite = checker_suite
        self.metric_store = metric_store
        if success_criteria is None:
            success_criteria = list(range(len(criteria)))
        out_of_range = [
            idx for idx in success_criteria if not 0 <= idx < len(criteria)
        ]
        if out_of_range:
            raise ValueError(
                f"success_criteria has out-of-range indices: {out_of_range}."
            )
        self.success_criteria = tuple(success_criteria)
        if progress_criteria is None:
            progress_criteria = list(range(len(criteria)))
        out_of_range = [
            idx for idx in progress_criteria if not 0 <= idx < len(criteria)
        ]
        if out_of_range:
            raise ValueError(
                f"progress_criteria has out-of-range indices: {out_of_range}."
            )
        self.progress_criteria = tuple(progress_criteria)

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
        ``metrics["criteria_reached"]``. These retain dependency-gated scoring
        semantics. When configured, ``criteria_independent_met_now`` and
        ``criteria_independent_reached`` report independent current and
        cumulative observations without affecting success or progress.

        """
        if self.checker_suite is not None:
            assert self.context is not None
            assert self.metric_store is not None
            observed_metrics = self.checker_suite.evaluate(
                env,
                self.context,
                self.metric_store.scope,
                env_idx=env_idx,
            )
            self.metric_store.update(observed_metrics)

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
        num_progress_criteria = len(self.progress_criteria)
        num_progress_reached = sum(
            self.criteria_reached[idx] for idx in self.progress_criteria
        )
        progress = (
            num_progress_reached / num_progress_criteria
            if num_progress_criteria > 0
            else 0.0
        )
        metrics["criteria_met_now_count"] = num_met_now
        metrics["criteria_ever_reached"] = num_reached_ever
        metrics["criteria_total"] = num_criteria
        metrics["progress_criteria_reached"] = num_progress_reached
        metrics["progress_criteria_total"] = num_progress_criteria
        for idx, c in enumerate(self.criteria_name):
            metrics["criteria_reached"][c] = self.criteria_reached[idx]

        if self.independent_criteria is not None:
            independent_now = {}
            independent_reached = {}
            for idx, criterion in enumerate(self.independent_criteria):
                result = self._call_criterion(criterion, env, env_idx)
                self.independent_criteria_reached[idx] |= result
                name = self.criteria_name[idx]
                independent_now[name] = result
                independent_reached[name] = self.independent_criteria_reached[
                    idx
                ]
            metrics["criteria_independent_met_now"] = independent_now
            metrics["criteria_independent_reached"] = independent_reached

        success = self.check_success()
        if success:
            progress = 1.0
        return ValidatorOutput(
            success=success,
            progress=progress,
            metrics=metrics,
            stage_scores={
                self.criteria_name[idx]: (
                    float(self.criteria_reached[idx]) / num_progress_criteria
                )
                for idx in self.progress_criteria
            },
        )

    def reset(self):
        """Clear latched progress and every stateful criterion."""
        self.criteria_reached = [False] * len(self.criteria)
        self.independent_criteria_reached = [False] * len(self.criteria_name)
        for criterion in [
            *self.criteria,
            *(self.independent_criteria or []),
        ]:
            checker = (
                criterion[0] if isinstance(criterion, tuple) else criterion
            )
            reset_hook = getattr(checker, "reset", None)
            if callable(reset_hook):
                reset_hook()
        if self.checker_suite is not None:
            self.checker_suite.reset()
        if self.metric_store is not None:
            self.metric_store.reset()

    def check_success(self) -> bool:
        """Return whether every success criterion has been met."""
        return all(self.criteria_reached[idx] for idx in self.success_criteria)

    def finalize(self) -> ValidatorOutput:
        """Return the result of fixed-horizon validation."""
        raise RuntimeError(
            f"{type(self).__name__} does not support finalization."
        )
