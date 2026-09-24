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

"""Shared fixed-horizon lifecycle for counterfactual validation."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from robo_orchard_sim.task_components.validators.base import (
    Validator,
    ValidatorOutput,
)
from robo_orchard_sim.task_components.validators.role_scope import EntityId

if TYPE_CHECKING:
    from robo_orchard_sim.task_components.validators.checkers import (
        SceneCheckerSuite,
    )
    from robo_orchard_sim.task_components.validators.context import (
        ValidatorContext,
    )
    from robo_orchard_sim.task_components.validators.metrics import MetricStore

CounterfactualRubric = Callable[
    [Mapping[str, bool]],
    tuple[bool, float],
]
Criterion = Callable | tuple[Callable, list[int]]


@dataclass(frozen=True, slots=True)
class CounterfactualSpec:
    """Task-owned criteria and final rubric for counterfactual evaluation."""

    criteria: tuple[Criterion, ...]
    criteria_name: tuple[str, ...]
    rubric: CounterfactualRubric

    def __post_init__(self) -> None:
        if len(self.criteria) != len(self.criteria_name):
            raise ValueError(
                "counterfactual criteria and names must have equal length."
            )
        if len(set(self.criteria_name)) != len(self.criteria_name):
            raise ValueError("counterfactual criteria names must be unique.")


@dataclass(frozen=True, slots=True)
class CounterfactualStageSpec:
    """Describe one ordered stage backed by current unary metrics."""

    name: str
    metrics: tuple[str, ...]
    dwell_steps: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Counterfactual stage name must be non-empty.")
        if not self.metrics:
            raise ValueError(
                f"Counterfactual stage '{self.name}' needs metrics."
            )
        if self.dwell_steps < 1:
            raise ValueError(
                f"Counterfactual stage '{self.name}' dwell_steps must be "
                f">= 1, got {self.dwell_steps}."
            )


class CounterfactualStageTracker:
    """Track ordered stages independently for each entity in one role."""

    def __init__(
        self,
        *,
        metric_store: "MetricStore",
        role_id: str,
        stages: tuple[CounterfactualStageSpec, ...],
    ) -> None:
        if not metric_store.scope.entities_for_role(role_id):
            raise ValueError(
                f"Role '{role_id}' has no entities in the role scope."
            )
        if not stages:
            raise ValueError("Counterfactual tracker requires stages.")
        stage_names = tuple(stage.name for stage in stages)
        if len(set(stage_names)) != len(stage_names):
            raise ValueError(
                f"Counterfactual stage names must be unique: {stage_names}."
            )
        self.metric_store = metric_store
        self.role_id = role_id
        self.stages = stages
        self._stage_names = frozenset(stage_names)
        self._streaks: dict[tuple[int, EntityId, str], int] = {}
        self._completed: dict[int, dict[EntityId, set[str]]] = {}
        self._observed_metrics: dict[int, object] = {}

    def completed_entities(
        self,
        stage: str,
        *,
        env_idx: int = 0,
    ) -> tuple[EntityId, ...]:
        """Return entities that completed the requested ordered stage."""
        self.validate_stage(stage)
        self._observe(env_idx)
        completed = self._completed.get(env_idx, {})
        return tuple(
            entity_id
            for entity_id in self.metric_store.scope.entities_for_role(
                self.role_id
            )
            if stage in completed.get(entity_id, set())
        )

    def reset(self) -> None:
        """Clear every environment and entity stage state."""
        self._streaks.clear()
        self._completed.clear()
        self._observed_metrics.clear()

    def _observe(self, env_idx: int) -> None:
        current = self.metric_store.current
        if self._observed_metrics.get(env_idx) is current:
            return
        completed_by_entity = self._completed.setdefault(env_idx, {})
        for entity_id in self.metric_store.scope.entities_for_role(
            self.role_id
        ):
            completed = completed_by_entity.setdefault(entity_id, set())
            for stage_idx, stage in enumerate(self.stages):
                prerequisite_met = (
                    stage_idx == 0
                    or self.stages[stage_idx - 1].name in completed
                )
                streak_key = (env_idx, entity_id, stage.name)
                if not prerequisite_met:
                    self._streaks[streak_key] = 0
                    continue
                matched = all(
                    bool(current.unary.get(metric, {}).get(entity_id, False))
                    for metric in stage.metrics
                )
                self._streaks[streak_key] = (
                    self._streaks.get(streak_key, 0) + 1 if matched else 0
                )
                if self._streaks[streak_key] >= stage.dwell_steps:
                    completed.add(stage.name)
        self._observed_metrics[env_idx] = current

    def validate_stage(self, stage: str) -> None:
        """Raise when a requested stage is not tracked."""
        if stage not in self._stage_names:
            raise ValueError(
                f"Unknown counterfactual stage '{stage}'. "
                f"Available: {tuple(stage.name for stage in self.stages)}."
            )


class CounterfactualStageSelector:
    """Select whether any one entity completed an ordered stage."""

    def __init__(
        self,
        tracker: CounterfactualStageTracker,
        stage: str,
    ) -> None:
        tracker.validate_stage(stage)
        self.tracker = tracker
        self.stage = stage

    def __call__(self, env, env_idx: int = 0) -> bool:
        del env
        return bool(
            self.tracker.completed_entities(self.stage, env_idx=env_idx)
        )

    def reset(self) -> None:
        """Clear the shared stage state."""
        self.tracker.reset()


class CounterfactualStageRelationSelector:
    """Match a current relation to an entity completing a tracked stage."""

    def __init__(
        self,
        *,
        tracker: CounterfactualStageTracker,
        completed_stage: str,
        object_role: str,
        relation_metric: str,
    ) -> None:
        tracker.validate_stage(completed_stage)
        relations = tracker.metric_store.scope.relation_pairs(
            tracker.role_id,
            object_role,
        )
        if not relations:
            raise ValueError(
                "Metric selection requires at least one valid "
                f"{tracker.role_id!r} to {object_role!r} relation."
            )
        self.tracker = tracker
        self.completed_stage = completed_stage
        self.relations = relations
        self.relation_metric = relation_metric

    def __call__(self, env, env_idx: int = 0) -> bool:
        del env
        completed = set(
            self.tracker.completed_entities(
                self.completed_stage,
                env_idx=env_idx,
            )
        )
        current_relations = self.tracker.metric_store.current.binary.get(
            self.relation_metric,
            {},
        )
        return any(
            subject in completed
            and current_relations.get((subject, object_), False)
            for subject, object_ in self.relations
        )

    def reset(self) -> None:
        """Clear the shared stage state."""
        self.tracker.reset()


class CounterfactualValidator(Validator):
    """Collect behavior for a full horizon, then apply a final rubric."""

    fixed_horizon = True

    def __init__(
        self,
        *,
        context: "ValidatorContext",
        checker_suite: "SceneCheckerSuite",
        metric_store: "MetricStore",
        spec: CounterfactualSpec,
    ) -> None:
        super().__init__(
            criteria=list(spec.criteria),
            criteria_name=list(spec.criteria_name),
            success_criteria=[],
            context=context,
            checker_suite=checker_suite,
            metric_store=metric_store,
        )
        self.rubric = spec.rubric
        self._last_metrics = self._initial_metrics()

    def _initial_metrics(self) -> dict[str, Any]:
        return {
            "criteria_met_now": {name: False for name in self.criteria_name},
            "criteria_reached": {name: False for name in self.criteria_name},
            "criteria_met_now_count": 0,
            "criteria_ever_reached": 0,
            "criteria_total": len(self.criteria),
            "progress_criteria_reached": 0,
            "progress_criteria_total": len(self.progress_criteria),
        }

    def _with_counterfactual_report(
        self,
        metrics: dict[str, Any],
        *,
        finalized: bool,
    ) -> dict[str, Any]:
        assert self.metric_store is not None
        updated = dict(metrics)
        updated["counterfactual"] = {
            "criteria": dict(metrics["criteria_reached"]),
            "observed": self.metric_store.behavior_report(),
            "finalized": finalized,
        }
        return updated

    def evaluate(self, env: Any, env_idx: int = 0) -> ValidatorOutput:
        """Accumulate behavior without producing an early success."""
        output = super().evaluate(env, env_idx=env_idx)
        self._last_metrics = output.metrics
        return ValidatorOutput(
            success=False,
            progress=0.0,
            metrics=self._with_counterfactual_report(
                output.metrics,
                finalized=False,
            ),
        )

    def finalize(self) -> ValidatorOutput:
        """Apply the counterfactual rubric after the full horizon."""
        reached = {
            name: value
            for name, value in zip(
                self.criteria_name,
                self.criteria_reached,
                strict=True,
            )
        }
        success, progress = self.rubric(reached)
        return ValidatorOutput(
            success=success,
            progress=progress,
            metrics=self._with_counterfactual_report(
                self._last_metrics,
                finalized=True,
            ),
        )

    def reset(self) -> None:
        """Clear normal and counterfactual episode state."""
        super().reset()
        self._last_metrics = self._initial_metrics()
