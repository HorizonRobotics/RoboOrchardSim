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

"""Behavioral tests for task selection over shared scene metrics."""

import pytest

from robo_orchard_sim.task_components.role_registry import (
    RoleRegistry,
    TargetRef,
)
from robo_orchard_sim.task_components.validators.base import Validator
from robo_orchard_sim.task_components.validators.checkers import (
    SceneCheckerSuite,
)
from robo_orchard_sim.task_components.validators.context import (
    ValidatorContext,
)
from robo_orchard_sim.task_components.validators.metrics import (
    BoundRoleMetricSelector,
    BoundRolePairMetricSelector,
    DwellMetricSelector,
    MetricStore,
    SceneMetrics,
)
from robo_orchard_sim.task_components.validators.role_scope import RoleScope


class _FlagSceneChecker:
    def __init__(self, metric: str) -> None:
        self.metric = metric

    def evaluate(self, env, context, scope, env_idx: int = 0):
        del context
        return SceneMetrics(
            unary={
                self.metric: {
                    entity_id: bool(
                        env[env_idx].get((self.metric, entity_id), False)
                    )
                    for entity_id in scope.entities
                }
            }
        )

    def reset(self) -> None:
        pass


@pytest.mark.parametrize(
    "runtime_kwargs",
    [
        {
            "context": ValidatorContext(
                robot=None, role_registry=RoleRegistry()
            )
        },
        {"checker_suite": SceneCheckerSuite((_FlagSceneChecker("flag"),))},
        {
            "metric_store": MetricStore(
                RoleScope.from_entities(("objects/pick",))
            )
        },
    ],
)
def test_validator_partial_scene_runtime_raises_value_error(runtime_kwargs):
    with pytest.raises(ValueError, match="must be provided together"):
        Validator(
            criteria=[],
            criteria_name=[],
            **runtime_kwargs,
        )


def _validator(bound_entity: str):
    registry = RoleRegistry()
    registry.bind_one(0, "pick", TargetRef(bound_entity))
    context = ValidatorContext(robot=None, role_registry=registry)
    scope = RoleScope.from_role_members(
        {"pick": ("objects/pick_0", "objects/pick_1")}
    )
    metric_store = MetricStore(scope)
    return Validator(
        context=context,
        checker_suite=SceneCheckerSuite(
            (
                _FlagSceneChecker("reached"),
                _FlagSceneChecker("lifted"),
            )
        ),
        metric_store=metric_store,
        criteria=[
            BoundRoleMetricSelector(
                metric_store=metric_store,
                context=context,
                role_id="pick",
                metric="reached",
            ),
            (
                BoundRoleMetricSelector(
                    metric_store=metric_store,
                    context=context,
                    role_id="pick",
                    metric="lifted",
                ),
                [0],
            ),
        ],
        criteria_name=["reach_pick", "lift_pick"],
    )


def test_bound_role_pair_selector_rebound_roles_selects_current_pair():
    registry = RoleRegistry()
    registry.bind_one(0, "pick", TargetRef("apple"))
    registry.bind_one(0, "place", TargetRef("basket"))
    context = ValidatorContext(robot=None, role_registry=registry)
    metric_store = MetricStore(
        RoleScope.from_role_members(
            {
                "pick": ("apple", "mug"),
                "place": ("basket", "tray"),
            }
        )
    )
    selector = BoundRolePairMetricSelector(
        metric_store=metric_store,
        context=context,
        subject_role="pick",
        object_role="place",
        metric="within_xy",
    )
    metric_store.update(
        SceneMetrics(binary={"within_xy": {("apple", "basket"): True}})
    )
    first_pair = selector(None)

    registry.bind_one(0, "pick", TargetRef("mug"))
    registry.bind_one(0, "place", TargetRef("tray"))
    metric_store.update(
        SceneMetrics(binary={"within_xy": {("mug", "tray"): True}})
    )

    assert (first_pair, selector(None)) == (True, True)


@pytest.mark.parametrize(
    ("bound_entity", "expected_success"),
    [
        ("objects/pick_0", False),
        ("objects/pick_1", True),
    ],
)
def test_bound_role_validator_interaction_selects_bound_entity(
    bound_entity,
    expected_success,
):
    validator = _validator(bound_entity)
    env = [
        {
            ("reached", "objects/pick_1"): True,
            ("lifted", "objects/pick_1"): True,
        }
    ]

    output = validator.evaluate(env)

    assert output.success is expected_success


def test_validator_success_without_optional_stage_returns_full_progress():
    validator = Validator(
        criteria=[lambda env: False, lambda env: True],
        criteria_name=["contact", "settled"],
        success_criteria=[1],
    )
    output = validator.evaluate(None)
    assert (
        output.success,
        output.progress,
        output.metrics["criteria_reached"],
        output.stage_scores,
    ) == (
        True,
        1.0,
        {"contact": False, "settled": True},
        {"contact": 0.0, "settled": 0.5},
    )


def test_validator_diagnostic_criterion_excluded_returns_full_progress() -> (
    None
):
    validator = Validator(
        criteria=[
            lambda env: bool(env["required"]),
            lambda env: bool(env["diagnostic"]),
        ],
        criteria_name=["required", "diagnostic"],
        success_criteria=[0],
        progress_criteria=[0],
    )

    output = validator.evaluate({"required": True, "diagnostic": False})

    assert (
        output.success,
        output.progress,
        output.metrics["criteria_reached"],
        output.stage_scores,
    ) == (
        True,
        1.0,
        {"required": True, "diagnostic": False},
        {"required": 1.0},
    )


@pytest.mark.parametrize("target", [False, True])
def test_validator_blocked_dependency_reports_independent_state(target):
    def reached(env):
        return env["reach"]

    def moved(env):
        return env["moved"]

    validator = Validator(
        criteria=[reached, (moved, [0])],
        criteria_name=["reach", "moved"],
        success_criteria=[1],
        independent_criteria=[reached, moved],
    )

    output = validator.evaluate({"reach": False, "moved": target})

    assert (
        output.metrics["criteria_independent_met_now"],
        output.metrics["criteria_reached"],
        output.success,
        output.progress,
    ) == (
        {"reach": False, "moved": target},
        {"reach": False, "moved": False},
        False,
        0.0,
    )


def test_validator_independent_dwell_preserves_scoring_timing():
    def reached(env):
        return env["reach"]

    def target(env, env_idx=0):
        del env_idx
        return env["target"]

    validator = Validator(
        criteria=[reached, (DwellMetricSelector(target, 3), [0])],
        criteria_name=["reach", "target"],
        success_criteria=[1],
        independent_criteria=[reached, DwellMetricSelector(target, 3)],
    )
    outputs = [
        validator.evaluate({"reach": reach, "target": True})
        for reach in [False, False, True, True, True]
    ]

    assert [
        (
            output.metrics["criteria_independent_met_now"]["target"],
            output.success,
            output.progress,
        )
        for output in outputs
    ] == [
        (False, False, 0.0),
        (False, False, 0.0),
        (True, False, 0.5),
        (True, False, 0.5),
        (True, True, 1.0),
    ]


def test_validator_independent_state_reset_clears_history_and_dwell():
    def target(env, env_idx=0):
        del env_idx
        return env["target"]

    validator = Validator(
        criteria=[DwellMetricSelector(target, 2)],
        criteria_name=["target"],
        independent_criteria=[DwellMetricSelector(target, 2)],
    )
    validator.evaluate({"target": True})
    validator.evaluate({"target": True})
    before_reset = validator.evaluate({"target": False})
    validator.reset()
    after_reset = validator.evaluate({"target": True})

    assert [
        (
            output.metrics["criteria_independent_met_now"]["target"],
            output.metrics["criteria_independent_reached"]["target"],
            output.success,
        )
        for output in [before_reset, after_reset]
    ] == [(False, True, True), (False, False, False)]
