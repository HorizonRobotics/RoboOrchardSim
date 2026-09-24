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

"""Behavioral tests for rigid counterfactual validation."""

import pytest

from robo_orchard_sim.task_components.role_registry import RoleRegistry
from robo_orchard_sim.task_components.validators.base import Validator
from robo_orchard_sim.task_components.validators.checkers import (
    SceneCheckerSuite,
)
from robo_orchard_sim.task_components.validators.context import (
    ValidatorContext,
)
from robo_orchard_sim.task_components.validators.counterfactual import (
    CounterfactualSpec,
    CounterfactualValidator,
)
from robo_orchard_sim.task_components.validators.metrics import (
    AnyRoleMetricSelector,
    MetricStore,
    SceneMetrics,
)
from robo_orchard_sim.task_components.validators.role_scope import RoleScope


class _FlagChecker:
    def evaluate(self, env, context, scope, env_idx: int = 0) -> SceneMetrics:
        del context
        return SceneMetrics(
            unary={
                "reached": {
                    entity: bool(env[env_idx].get(("reached", entity)))
                    for entity in scope.entities
                },
                "lifted": {
                    entity: bool(env[env_idx].get(("lifted", entity)))
                    for entity in scope.entities
                },
            }
        )

    def reset(self) -> None:
        """The test checker has no state."""


def _counterfactual_validator() -> CounterfactualValidator:
    scope = RoleScope.from_role_members(
        {
            "pick": ("apple", "mug"),
            "place": ("basket",),
        }
    )
    store = MetricStore(scope)
    context = ValidatorContext(robot=None, role_registry=RoleRegistry())
    checker_suite = SceneCheckerSuite((_FlagChecker(),))
    return CounterfactualValidator(
        context=context,
        checker_suite=checker_suite,
        metric_store=store,
        spec=CounterfactualSpec(
            criteria=(
                AnyRoleMetricSelector(
                    metric_store=store,
                    role_id="pick",
                    metric="reached",
                ),
                AnyRoleMetricSelector(
                    metric_store=store,
                    role_id="pick",
                    metric="lifted",
                ),
            ),
            criteria_name=("any_reached", "any_lifted"),
            rubric=lambda reached: (
                reached["any_lifted"],
                float(reached["any_lifted"]),
            ),
        ),
    )


def test_counterfactual_validator_evaluate_defers_success_until_finalize():
    validator = _counterfactual_validator()

    step_output = validator.evaluate([{("lifted", "apple"): True}])
    final_output = validator.finalize()

    assert (
        validator.fixed_horizon,
        step_output.success,
        final_output.success,
        final_output.metrics["counterfactual"]["finalized"],
        final_output.stage_scores,
    ) == (True, False, True, True, {})


def test_counterfactual_validator_reset_clears_accumulated_behavior():
    validator = _counterfactual_validator()
    validator.evaluate([{("lifted", "apple"): True}])

    validator.reset()
    output = validator.finalize()

    assert (
        output.success,
        output.metrics["counterfactual"]["observed"]["unary"],
    ) == (False, {})


def test_any_role_metric_selector_other_role_true_returns_false():
    scope = RoleScope.from_role_members(
        {"pick": ("apple",), "place": ("basket",)}
    )
    store = MetricStore(scope)
    store.update(SceneMetrics(unary={"lifted": {"basket": True}}))
    selector = AnyRoleMetricSelector(
        metric_store=store,
        role_id="pick",
        metric="lifted",
    )

    assert selector(None) is False


def test_validator_default_lifecycle_is_not_fixed_horizon():
    validator = Validator(criteria=[], criteria_name=[])

    with pytest.raises(RuntimeError, match="does not support finalization"):
        validator.finalize()

    assert validator.fixed_horizon is False
