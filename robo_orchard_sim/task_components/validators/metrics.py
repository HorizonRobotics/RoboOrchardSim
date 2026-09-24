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

"""Scene metrics, episode accumulation, and metric selectors."""

from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from robo_orchard_sim.asset_manager.metadata.joint_operations import (
    OperationStartMode,
    get_joint,
    get_operation_start_position,
)
from robo_orchard_sim.task_components.validators.physical_entity import (
    SceneBodyKey,
    SceneEntityKey,
    SceneJointKey,
    resolve_bound_operation,
    scene_name_of,
)
from robo_orchard_sim.task_components.validators.role_scope import (
    EntityId,
    RoleScope,
)

if TYPE_CHECKING:
    from robo_orchard_sim.task_components.validators.context import (
        ValidatorContext,
    )

RelationKey = tuple[SceneEntityKey, SceneEntityKey]


def _serialize_entity_key(entity: SceneEntityKey) -> str:
    """Return a stable, JSON-compatible physical entity identifier."""
    if isinstance(entity, str):
        return entity
    if isinstance(entity, SceneBodyKey):
        return f"{entity.scene_name}/body:{entity.body_name}"
    if isinstance(entity, SceneJointKey):
        return f"{entity.scene_name}/joint:{entity.joint_name}"
    raise TypeError(f"Unsupported scene entity key: {type(entity).__name__}")


@dataclass(frozen=True, slots=True)
class SceneMetrics:
    """Scene-wide metric values for one step or accumulated episode."""

    unary: Mapping[str, Mapping[EntityId, bool]] = field(default_factory=dict)
    binary: Mapping[str, Mapping[RelationKey, bool]] = field(
        default_factory=dict
    )
    global_values: Mapping[str, bool] = field(default_factory=dict)
    values: Mapping[str, Mapping[SceneEntityKey, float]] = field(
        default_factory=dict
    )


def _copy_scene_metrics(metrics: SceneMetrics) -> SceneMetrics:
    return SceneMetrics(
        unary={
            metric: dict(values) for metric, values in metrics.unary.items()
        },
        binary={
            metric: dict(values) for metric, values in metrics.binary.items()
        },
        global_values=dict(metrics.global_values),
        values={
            metric: dict(values) for metric, values in metrics.values.items()
        },
    )


class MetricStore:
    """Keep current and ever-observed facts for one episode."""

    def __init__(self, scope: RoleScope) -> None:
        self.scope = scope
        self.current = SceneMetrics()
        self._unary_ever: dict[str, dict[EntityId, bool]] = {}
        self._binary_ever: dict[str, dict[RelationKey, bool]] = {}
        self._global_ever: dict[str, bool] = {}

    def reset(self) -> None:
        """Clear current and accumulated episode facts."""
        self.current = SceneMetrics()
        self._unary_ever.clear()
        self._binary_ever.clear()
        self._global_ever.clear()

    def update(self, metrics: SceneMetrics) -> None:
        """Store one step and OR boolean facts into the episode history."""
        self._validate(metrics)
        self.current = _copy_scene_metrics(metrics)
        for metric, values in metrics.unary.items():
            accumulated = self._unary_ever.setdefault(metric, {})
            for entity_id, value in values.items():
                accumulated[entity_id] = accumulated.get(
                    entity_id, False
                ) or bool(value)
        for metric, values in metrics.binary.items():
            accumulated = self._binary_ever.setdefault(metric, {})
            for relation, value in values.items():
                accumulated[relation] = accumulated.get(
                    relation, False
                ) or bool(value)
        for metric, value in metrics.global_values.items():
            self._global_ever[metric] = self._global_ever.get(
                metric, False
            ) or bool(value)

    @property
    def accumulated(self) -> SceneMetrics:
        """Return a copy of all facts observed in the current episode."""
        return SceneMetrics(
            unary={
                metric: dict(values)
                for metric, values in self._unary_ever.items()
            },
            binary={
                metric: dict(values)
                for metric, values in self._binary_ever.items()
            },
            global_values=dict(self._global_ever),
        )

    def true_entities(
        self,
        metric: str,
        *,
        role_id: str | None = None,
        accumulated: bool = True,
    ) -> list[EntityId]:
        """Return true unary entities in stable scope or role order."""
        metrics = self.accumulated if accumulated else self.current
        values = metrics.unary.get(metric, {})
        domain = (
            self.scope.entities
            if role_id is None
            else self.scope.entities_for_role(role_id)
        )
        return [entity_id for entity_id in domain if values.get(entity_id)]

    def checker_summary(self) -> dict[str, object]:
        """Snapshot episode-wide checker facts before criterion dwell gating.

        Each boolean means true on at least one evaluated step. Per-metric
        keys identify entities actually checked, including those always
        false; missing keys mean unobserved. Scope lists declared candidates,
        while checkers may also report their component bodies or joints.
        Numeric samples and per-step history are not retained here.
        """
        metrics = self.accumulated
        return {
            "aggregation": "any_step_before_criteria_dwell",
            "scope": [
                _serialize_entity_key(entity) for entity in self.scope.entities
            ],
            "unary": {
                metric: {
                    _serialize_entity_key(entity): value
                    for entity, value in values.items()
                }
                for metric, values in metrics.unary.items()
            },
            "binary": {
                metric: [
                    {
                        "subject": _serialize_entity_key(subject),
                        "object": _serialize_entity_key(object_),
                        "ever_true": value,
                    }
                    for (subject, object_), value in values.items()
                ]
                for metric, values in metrics.binary.items()
            },
            "global": dict(metrics.global_values),
        }

    def behavior_report(self) -> dict[str, object]:
        """Return accumulated boolean behavior in a serializable form."""
        metrics = self.accumulated
        return {
            "unary": {
                metric: [
                    _serialize_entity_key(entity)
                    for entity in self.true_entities(metric)
                ]
                for metric in metrics.unary
            },
            "binary": {
                metric: [
                    [
                        _serialize_entity_key(subject),
                        _serialize_entity_key(object_),
                    ]
                    for (subject, object_), value in values.items()
                    if value
                ]
                for metric, values in metrics.binary.items()
            },
            "global": dict(metrics.global_values),
        }

    def _validate(self, metrics: SceneMetrics) -> None:
        known = {scene_name_of(key) for key in self.scope.entities}
        for domain, all_metrics in (
            ("Unary", metrics.unary),
            ("Value", metrics.values),
        ):
            for metric, values in all_metrics.items():
                unknown = {
                    scene_name_of(key)
                    for key in values
                    if scene_name_of(key) not in known
                }
                if unknown:
                    raise ValueError(
                        f"{domain} metric '{metric}' contains scene entities "
                        f"outside RoleScope: {sorted(unknown)}."
                    )
        for metric, values in metrics.binary.items():
            unknown = {
                scene_name_of(key)
                for relation in values
                for key in relation
                if scene_name_of(key) not in known
            }
            if unknown:
                raise ValueError(
                    f"Binary metric '{metric}' contains scene entities "
                    f"outside RoleScope: {sorted(unknown)}."
                )


class BoundRoleMetricSelector:
    """Select one metric using the role's current TargetRef binding."""

    def __init__(
        self,
        *,
        metric_store: MetricStore,
        context: ValidatorContext,
        role_id: str,
        metric: str,
    ) -> None:
        scope = metric_store.scope
        candidates = scope.entities_for_role(role_id)
        if not candidates:
            raise ValueError(
                f"Role '{role_id}' has no entities in the interaction scope."
            )
        bound_entity = context.scene_name_of(role_id)
        if bound_entity not in candidates:
            raise ValueError(
                f"Role '{role_id}' is bound to '{bound_entity}', which is "
                "outside its interaction-scope candidates."
            )
        self.metric_store = metric_store
        self.context = context
        self.role_id = role_id
        self.metric = metric

    def __call__(self, env, env_idx: int = 0) -> bool:
        del env
        entity_id = self.context.scene_name_of(self.role_id, env_idx)
        if entity_id not in self.metric_store.scope.entities_for_role(
            self.role_id
        ):
            raise ValueError(
                f"Role '{self.role_id}' is bound to '{entity_id}', which is "
                "outside its interaction-scope candidates."
            )
        return bool(
            self.metric_store.current.unary.get(self.metric, {}).get(
                entity_id,
                False,
            )
        )


class AnyRoleMetricSelector:
    """Select whether any entity in one role has a current metric."""

    def __init__(
        self,
        *,
        metric_store: MetricStore,
        role_id: str,
        metric: str,
    ) -> None:
        if not metric_store.scope.entities_for_role(role_id):
            raise ValueError(
                f"Role '{role_id}' has no entities in the role scope."
            )
        self.metric_store = metric_store
        self.role_id = role_id
        self.metric = metric

    def __call__(self, env, env_idx: int = 0) -> bool:
        del env, env_idx
        return bool(
            self.metric_store.true_entities(
                self.metric,
                role_id=self.role_id,
                accumulated=False,
            )
        )


class _BoundOperationSelector:
    """Shared validation and resolution for operation-bound role selectors."""

    def __init__(
        self,
        *,
        metric_store: MetricStore,
        context: ValidatorContext,
        role_id: str,
    ) -> None:
        if not metric_store.scope.entities_for_role(role_id):
            raise ValueError(
                f"Role '{role_id}' has no entities in the interaction scope."
            )
        self.metric_store = metric_store
        self.context = context
        self.role_id = role_id

    def _resolve(self, env, env_idx: int):
        target = self.context.role_registry.resolve_one(self.role_id, env_idx)
        resolved = resolve_bound_operation(env, target)
        candidates = self.metric_store.scope.entities_for_role(self.role_id)
        if (
            target.scene_name not in candidates
            and not any(
                key in candidates for key in resolved.interaction_body_keys
            )
            and resolved.outcome_body_key not in candidates
            and resolved.joint_key not in candidates
        ):
            raise ValueError(
                f"Role '{self.role_id}' is bound to '{target.scene_name}', "
                "which is outside its interaction-scope candidates."
            )
        return resolved


class BoundRoleInteractionBodyMetricSelector(_BoundOperationSelector):
    """Select an interaction fact satisfied by any configured body."""

    def __init__(
        self,
        *,
        metric_store: MetricStore,
        context: ValidatorContext,
        role_id: str,
        metric: str,
    ) -> None:
        super().__init__(
            metric_store=metric_store,
            context=context,
            role_id=role_id,
        )
        self.metric = metric

    def __call__(self, env, env_idx: int = 0) -> bool:
        resolved = self._resolve(env, env_idx)
        observed = self.metric_store.current.unary.get(self.metric, {})
        return any(
            bool(observed.get(key, False))
            for key in resolved.interaction_body_keys
        )


class BoundRoleOutcomeBodyMetricSelector(_BoundOperationSelector):
    """Select the joint's outcome-body fact."""

    def __init__(
        self,
        *,
        metric_store: MetricStore,
        context: ValidatorContext,
        role_id: str,
        metric: str,
    ) -> None:
        super().__init__(
            metric_store=metric_store,
            context=context,
            role_id=role_id,
        )
        self.metric = metric

    def __call__(self, env, env_idx: int = 0) -> bool:
        resolved = self._resolve(env, env_idx)
        return bool(
            self.metric_store.current.unary.get(self.metric, {}).get(
                resolved.outcome_body_key,
                False,
            )
        )


class _BoundJointOperationSelector(_BoundOperationSelector):
    """Resolve the start shared by joint reset and operation evaluation."""

    def __init__(
        self,
        *,
        metric_store: MetricStore,
        context: ValidatorContext,
        role_id: str,
        operation_start_mode: OperationStartMode = "metadata",
    ) -> None:
        super().__init__(
            metric_store=metric_store,
            context=context,
            role_id=role_id,
        )
        self.operation_start_mode: OperationStartMode = operation_start_mode

    def _initial_position(self, env, env_idx: int) -> float:
        target = self.context.role_registry.resolve_one(self.role_id, env_idx)
        if target.semantic_name is None or target.operation is None:
            raise ValueError(
                f"Target '{target}' must include semantic_name and operation."
            )
        articulation = env.scene[target.scene_name]
        joint = get_joint(
            articulation.cfg.joint_operations, target.semantic_name
        )
        start = get_operation_start_position(
            joint, target.operation, self.operation_start_mode
        )
        if self.operation_start_mode == "metadata":
            return start
        joint_id = articulation.joint_names.index(joint.joint_name)
        limits = articulation.data.soft_joint_pos_limits[env_idx, joint_id]
        # Direction tasks reset without noise and always clamp to soft limits.
        # Match the simulator's position dtype before clipping the midpoint.
        return float(
            articulation.data.joint_pos.new_tensor(start).clamp(
                min=limits[0], max=limits[1]
            )
        )


class BoundRoleJointMovedSelector(_BoundJointOperationSelector):
    """Compare one bound joint against its configured operation start."""

    def __init__(
        self,
        *,
        metric_store: MetricStore,
        context: ValidatorContext,
        role_id: str,
        min_delta: float | None = None,
        min_linear_delta: float = 0.0001,
        min_angular_delta: float = math.pi / 180,
        metric: str = "joint_position",
        operation_start_mode: OperationStartMode = "metadata",
    ) -> None:
        for threshold in (min_delta, min_linear_delta, min_angular_delta):
            if threshold is not None and (
                not math.isfinite(threshold) or threshold <= 0.0
            ):
                raise ValueError(
                    "Movement thresholds must be finite and positive"
                )
        super().__init__(
            metric_store=metric_store,
            context=context,
            role_id=role_id,
            operation_start_mode=operation_start_mode,
        )
        self.min_delta = min_delta
        self.min_linear_delta = min_linear_delta
        self.min_angular_delta = min_angular_delta
        self.metric = metric

    def __call__(self, env, env_idx: int = 0) -> bool:
        resolved = self._resolve(env, env_idx)
        position = self.metric_store.current.values.get(self.metric, {}).get(
            resolved.joint_key
        )
        if position is None:
            return False
        threshold = self.min_delta
        if threshold is None:
            articulation = env.scene[resolved.joint_key.scene_name]
            joint_id = articulation.joint_names.index(
                resolved.joint_key.joint_name
            )
            # PhysX DOF types: 0 is rotation, 1 is translation.
            dof_type = int(
                articulation.root_physx_view.shared_metatype.dof_types[
                    joint_id
                ]
            )
            if dof_type not in (0, 1):
                raise ValueError(f"Unsupported joint DOF type: {dof_type}")
            threshold = (
                self.min_angular_delta
                if dof_type == 0
                else self.min_linear_delta
            )
        return (
            abs(position - self._initial_position(env, env_idx)) >= threshold
        )


class BoundRoleJointFractionSelector(_BoundJointOperationSelector):
    """Accept reaching or passing a target in the declared motion direction."""

    def __init__(
        self,
        *,
        metric_store: MetricStore,
        context: ValidatorContext,
        role_id: str,
        tolerance: float,
        metric: str = "joint_fraction",
        operation_start_mode: OperationStartMode = "metadata",
    ) -> None:
        if tolerance < 0.0:
            raise ValueError(
                f"tolerance must be non-negative, got {tolerance}."
            )
        super().__init__(
            metric_store=metric_store,
            context=context,
            role_id=role_id,
            operation_start_mode=operation_start_mode,
        )
        self.tolerance = tolerance
        self.metric = metric

    def __call__(self, env, env_idx: int = 0) -> bool:
        resolved = self._resolve(env, env_idx)
        target = resolved.operation.target_joint_fraction
        if target is None:
            return False
        fraction = self.metric_store.current.values.get(self.metric, {}).get(
            resolved.joint_key
        )
        if fraction is None:
            return False
        if not math.isfinite(fraction):
            return False
        articulation = env.scene[resolved.joint_key.scene_name]
        joint_id = articulation.joint_names.index(
            resolved.joint_key.joint_name
        )
        limits = articulation.data.joint_pos_limits[env_idx, joint_id]
        lower, upper = float(limits[0]), float(limits[1])
        if not (math.isfinite(lower) and math.isfinite(upper)):
            return False
        if upper <= lower:
            return False
        initial = (self._initial_position(env, env_idx) - lower) / (
            upper - lower
        )
        if math.isclose(target, initial, abs_tol=1e-6):
            # No declared motion direction: preserve the position band.
            return abs(fraction - target) <= self.tolerance
        if target > initial:
            return fraction >= target - self.tolerance
        return fraction <= target + self.tolerance


class BoundRoleJointDeltaSelector(_BoundJointOperationSelector):
    """Compare one bound joint displacement against its operation target."""

    def __init__(
        self,
        *,
        metric_store: MetricStore,
        context: ValidatorContext,
        role_id: str,
        metric: str = "joint_position",
        operation_start_mode: OperationStartMode = "metadata",
    ) -> None:
        super().__init__(
            metric_store=metric_store,
            context=context,
            role_id=role_id,
            operation_start_mode=operation_start_mode,
        )
        self.metric = metric

    def __call__(self, env, env_idx: int = 0) -> bool:
        resolved = self._resolve(env, env_idx)
        target = resolved.operation.target_joint_delta
        if target is None:
            return False
        position = self.metric_store.current.values.get(self.metric, {}).get(
            resolved.joint_key
        )
        if position is None:
            return False
        return abs(position - self._initial_position(env, env_idx)) >= target


class BoundRolePairMetricSelector:
    """Select one binary metric using two current role bindings."""

    def __init__(
        self,
        *,
        metric_store: MetricStore,
        context: ValidatorContext,
        subject_role: str,
        object_role: str,
        metric: str,
    ) -> None:
        for role_id in (subject_role, object_role):
            candidates = metric_store.scope.entities_for_role(role_id)
            if not candidates:
                raise ValueError(
                    f"Role '{role_id}' has no entities in the role scope."
                )
        self.metric_store = metric_store
        self.context = context
        self.subject_role = subject_role
        self.object_role = object_role
        self.metric = metric

    def __call__(self, env, env_idx: int = 0) -> bool:
        subject = self._bound_entity(env, self.subject_role, env_idx)
        object_ = self._bound_entity(env, self.object_role, env_idx)
        for role_id, entity_id in (
            (self.subject_role, subject),
            (self.object_role, object_),
        ):
            if entity_id not in self.metric_store.scope.entities_for_role(
                role_id
            ):
                raise ValueError(
                    f"Role '{role_id}' is bound to '{entity_id}', which is "
                    "outside its role-scope candidates."
                )
        return bool(
            self.metric_store.current.binary.get(self.metric, {}).get(
                (subject, object_),
                False,
            )
        )

    def _bound_entity(
        self,
        env,
        role_id: str,
        env_idx: int,
    ) -> SceneEntityKey:
        target = self.context.role_registry.resolve_one(role_id, env_idx)
        if target.semantic_name is None:
            return target.scene_name
        return resolve_bound_operation(env, target).interaction_body_key


class EntityMetricSelector:
    """Select one fixed scene entity from a current metric."""

    def __init__(
        self,
        *,
        metric_store: MetricStore,
        entity_id: SceneEntityKey,
        metric: str,
    ) -> None:
        if entity_id not in metric_store.scope.entities:
            raise ValueError(
                f"Entity '{entity_id}' is outside the interaction scope."
            )
        self.metric_store = metric_store
        self.entity_id = entity_id
        self.metric = metric

    def __call__(self, env, env_idx: int = 0) -> bool:
        del env, env_idx
        return bool(
            self.metric_store.current.unary.get(self.metric, {}).get(
                self.entity_id,
                False,
            )
        )


class EntityPairMetricSelector:
    """Select one fixed entity pair from a current binary metric."""

    def __init__(
        self,
        *,
        metric_store: MetricStore,
        subject_id: SceneEntityKey,
        object_id: SceneEntityKey,
        metric: str,
    ) -> None:
        unknown = {
            entity_id
            for entity_id in (subject_id, object_id)
            if entity_id not in metric_store.scope.entities
        }
        if unknown:
            raise ValueError(
                "Relation entities are outside the interaction scope: "
                f"{sorted(map(str, unknown))}."
            )
        self.metric_store = metric_store
        self.relation = (subject_id, object_id)
        self.metric = metric

    def __call__(self, env, env_idx: int = 0) -> bool:
        del env, env_idx
        return bool(
            self.metric_store.current.binary.get(self.metric, {}).get(
                self.relation,
                False,
            )
        )


class AccumulatedUnaryRelationMetricSelector:
    """Match a current relation to the same subject's accumulated fact."""

    def __init__(
        self,
        *,
        metric_store: MetricStore,
        subject_role: str,
        object_role: str,
        unary_metric: str,
        relation_metric: str,
    ) -> None:
        self.metric_store = metric_store
        self.subject_role = subject_role
        self.relations = metric_store.scope.relation_pairs(
            subject_role,
            object_role,
        )
        if not self.relations:
            raise ValueError(
                "Metric selection requires at least one valid "
                f"{subject_role!r} to {object_role!r} relation."
            )
        self.unary_metric = unary_metric
        self.relation_metric = relation_metric

    def __call__(self, env, env_idx: int = 0) -> bool:
        del env, env_idx
        accumulated_subjects = set(
            self.metric_store.true_entities(
                self.unary_metric,
                role_id=self.subject_role,
            )
        )
        current_relations = self.metric_store.current.binary.get(
            self.relation_metric,
            {},
        )
        return any(
            subject in accumulated_subjects
            and current_relations.get((subject, object_), False)
            for subject, object_ in self.relations
        )


class GlobalMetricSelector:
    """Select one current scene-global metric."""

    def __init__(
        self,
        *,
        metric_store: MetricStore,
        metric: str,
    ) -> None:
        self.metric_store = metric_store
        self.metric = metric

    def __call__(self, env, env_idx: int = 0) -> bool:
        del env, env_idx
        return bool(
            self.metric_store.current.global_values.get(self.metric, False)
        )


class AnyMetricSelector:
    """Require at least one child metric selector to pass."""

    def __init__(self, selectors: Sequence[Callable]) -> None:
        if not selectors:
            raise ValueError("selectors is required for any-metric selector.")
        self.selectors = tuple(selectors)

    def __call__(self, env, env_idx: int = 0) -> bool:
        return any(
            bool(selector(env, env_idx=env_idx)) for selector in self.selectors
        )


class AllMetricSelector:
    """Require every child metric selector to pass on the current frame."""

    def __init__(self, selectors: Sequence[Callable]) -> None:
        if not selectors:
            raise ValueError("selectors is required for all-metric selector.")
        self.selectors = tuple(selectors)

    def __call__(self, env, env_idx: int = 0) -> bool:
        return all(
            bool(selector(env, env_idx=env_idx)) for selector in self.selectors
        )


class NotMetricSelector:
    """Negate one current-frame metric selector."""

    def __init__(self, selector: Callable) -> None:
        self.selector = selector

    def __call__(self, env, env_idx: int = 0) -> bool:
        return not bool(self.selector(env, env_idx=env_idx))

    def reset(self) -> None:
        """Reset the wrapped selector when it carries episode state."""
        reset_hook = getattr(self.selector, "reset", None)
        if callable(reset_hook):
            reset_hook()


class DwellMetricSelector:
    """Require one current-frame selector to hold for consecutive steps."""

    def __init__(self, selector: Callable, n_steps: int) -> None:
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
        self.selector = selector
        self.n_steps = n_steps
        self._streaks: dict[int, int] = {}

    def __call__(self, env, env_idx: int = 0) -> bool:
        if self.selector(env, env_idx=env_idx):
            self._streaks[env_idx] = self._streaks.get(env_idx, 0) + 1
        else:
            self._streaks[env_idx] = 0
        return self._streaks[env_idx] >= self.n_steps

    def reset(self) -> None:
        """Clear consecutive-step state for every environment."""
        self._streaks.clear()
        reset_hook = getattr(self.selector, "reset", None)
        if callable(reset_hook):
            reset_hook()
