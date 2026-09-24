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

"""Behavioral tests for scene metric accumulation."""

import pytest

from robo_orchard_sim.task_components.validators.metrics import (
    MetricStore,
    SceneMetrics,
)
from robo_orchard_sim.task_components.validators.physical_entity import (
    SceneBodyKey,
    SceneJointKey,
)
from robo_orchard_sim.task_components.validators.role_scope import (
    RoleScope,
)


def _scope() -> RoleScope:
    return RoleScope.from_role_members(
        {
            "pick": ("apple", "mug"),
            "place": ("basket",),
        }
    )


def test_metric_store_unary_values_latch_in_role_order():
    store = MetricStore(_scope())
    store.update(
        SceneMetrics(unary={"reached": {"apple": True, "mug": False}})
    )
    store.update(
        SceneMetrics(unary={"reached": {"apple": False, "mug": True}})
    )

    assert store.true_entities("reached", role_id="pick") == ["apple", "mug"]
    assert store.true_entities(
        "reached", role_id="pick", accumulated=False
    ) == ["mug"]


def test_metric_store_binary_and_global_values_latch():
    store = MetricStore(_scope())
    store.update(
        SceneMetrics(
            binary={"within_xy": {("apple", "basket"): True}},
            global_values={"gripper_open": True},
        )
    )
    store.update(
        SceneMetrics(
            binary={"within_xy": {("apple", "basket"): False}},
            global_values={"gripper_open": False},
        )
    )

    assert store.accumulated.binary["within_xy"][("apple", "basket")] is True
    assert store.accumulated.global_values["gripper_open"] is True


def test_metric_store_reset_clears_episode_values():
    store = MetricStore(_scope())
    store.update(SceneMetrics(unary={"lifted": {"apple": True}}))

    store.reset()

    assert store.true_entities("lifted") == []
    assert store.current == SceneMetrics()


def test_metric_store_unknown_entity_raises_value_error():
    store = MetricStore(_scope())

    with pytest.raises(ValueError, match="outside RoleScope"):
        store.update(SceneMetrics(unary={"reached": {"unknown": True}}))


@pytest.mark.parametrize(
    ("metric", "key", "first", "second"),
    [
        ("moved", SceneJointKey("apple", "hinge"), True, False),
        ("stationary", SceneBodyKey("apple", "lid"), False, True),
    ],
)
def test_metric_store_component_boolean_multiple_steps_latches_true(
    metric,
    key,
    first,
    second,
):
    store = MetricStore(_scope())
    store.update(SceneMetrics(unary={metric: {key: first}}))
    store.update(SceneMetrics(unary={metric: {key: second}}))

    assert store.accumulated.unary[metric][key] is True


def test_metric_store_component_scalar_multiple_steps_keeps_current_value():
    store = MetricStore(_scope())
    joint = SceneJointKey("apple", "hinge")
    store.update(SceneMetrics(values={"fraction": {joint: 0.25}}))
    store.update(SceneMetrics(values={"fraction": {joint: 0.75}}))

    assert store.current.values["fraction"][joint] == 0.75


def test_metric_store_behavior_report_component_keys_returns_strings():
    body = SceneBodyKey("objects/cabinet", "drawer")
    joint = SceneJointKey("objects/cabinet", "drawer_joint")
    store = MetricStore(RoleScope.from_entities((body, joint)))
    store.update(
        SceneMetrics(
            unary={"moved": {body: True}},
            binary={"related": {(body, joint): True}},
            global_values={"gripper_open": True},
        )
    )

    assert store.behavior_report() == {
        "unary": {"moved": ["objects/cabinet/body:drawer"]},
        "binary": {
            "related": [
                [
                    "objects/cabinet/body:drawer",
                    "objects/cabinet/joint:drawer_joint",
                ]
            ]
        },
        "global": {"gripper_open": True},
    }


@pytest.mark.parametrize(
    "metrics",
    [
        SceneMetrics(
            unary={
                "moved": {
                    SceneJointKey("unknown", "hinge"): True,
                }
            }
        ),
        SceneMetrics(
            unary={
                "stationary": {
                    SceneBodyKey("unknown", "lid"): True,
                }
            }
        ),
        SceneMetrics(
            values={
                "fraction": {
                    SceneJointKey("unknown", "hinge"): 0.5,
                }
            }
        ),
    ],
)
def test_metric_store_component_outside_scope_raises_value_error(
    metrics: SceneMetrics,
) -> None:
    store = MetricStore(_scope())

    with pytest.raises(ValueError, match="outside RoleScope"):
        store.update(metrics)


def test_scene_component_keys_same_fields_compare_equal() -> None:
    assert {
        SceneJointKey("objects/laptop", "hinge"),
        SceneJointKey("objects/laptop", "hinge"),
    } == {SceneJointKey("objects/laptop", "hinge")}


def test_checker_summary_component_facts_preserve_observed_false_and_history():
    import json

    upper = SceneBodyKey("cabinet", "upper_handle")
    lower = SceneBodyKey("cabinet", "lower_handle")
    joint = SceneJointKey("cabinet", "lower_joint")
    store = MetricStore(RoleScope.from_entities(("cabinet", "unobserved")))
    store.update(
        SceneMetrics(
            unary={"contacted": {upper: False, lower: True}},
            binary={"related": {(lower, joint): False}},
            global_values={"gripper_open": True},
        )
    )
    store.update(SceneMetrics(unary={"contacted": {lower: False}}))

    assert json.loads(json.dumps(store.checker_summary())) == {
        "aggregation": "any_step_before_criteria_dwell",
        "scope": ["cabinet", "unobserved"],
        "unary": {
            "contacted": {
                "cabinet/body:upper_handle": False,
                "cabinet/body:lower_handle": True,
            }
        },
        "binary": {
            "related": [
                {
                    "subject": "cabinet/body:lower_handle",
                    "object": "cabinet/joint:lower_joint",
                    "ever_true": False,
                }
            ]
        },
        "global": {"gripper_open": True},
    }


def test_checker_summary_reset_preserves_previous_snapshot():
    store = MetricStore(RoleScope.from_entities(("apple",)))
    store.update(SceneMetrics(unary={"lifted": {"apple": True}}))
    snapshot = store.checker_summary()
    store.reset()
    assert (snapshot["unary"], store.checker_summary()["unary"]) == (
        {"lifted": {"apple": True}},
        {},
    )
