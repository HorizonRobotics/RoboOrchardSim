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

"""Action-plan coverage for the stack-cubes benchmark."""

from types import SimpleNamespace

import pytest

from robo_orchard_sim.benchmark.manipulation.stack_cubes import (
    stack_cubes_env,
)
from robo_orchard_sim.benchmark.manipulation.stack_cubes.action_plan import (
    build_task_atomic_action_plan,
)
from robo_orchard_sim.benchmark.registry import (
    build_task_atomic_action_plan as build_registered_action_plan,
)
from robo_orchard_sim.task_components.role_registry import (
    RoleRegistry,
    TargetRef,
)

_ORDER = ("blue_cube", "red_cube", "green_cube", "yellow_cube")


class _FakeRobotInfo:
    """Minimal object satisfying the ManipulatorResolver protocol.

    ``PickExecutorCfg.robot_info`` is validated against the
    ``@runtime_checkable`` ManipulatorResolver protocol, so the fake must
    expose ``resolve`` to pass pydantic's isinstance check.
    """

    def __init__(self, manipulator_name: str) -> None:
        self.manipulator_name = manipulator_name

    def resolve(self, env, context=None):
        """Refuse to resolve; plans are inspected, never executed here."""
        raise NotImplementedError(
            "Action plans are inspected as config, never resolved here."
        )


def _role_registry(*, bound: bool = True) -> RoleRegistry:
    registry = RoleRegistry(num_envs=1)
    if bound:
        registry.bind_many(0, "members", [TargetRef(name) for name in _ORDER])
    return registry


def _orchard_env(robot_name: str = "panda_droid"):
    embodiment = SimpleNamespace(
        name=robot_name,
        get_robot_info_cfg=lambda key: _FakeRobotInfo(key),
        get_robot_info_cfgs=lambda: {"main_arm": object()},
    )
    return SimpleNamespace(task=SimpleNamespace(), embodiment=embodiment)


def _action_plan(robot_name: str = "panda_droid"):
    return build_task_atomic_action_plan(
        _orchard_env(robot_name),
        role_registry=_role_registry(),
    )


def test_build_task_atomic_action_plan_panda_droid_returns_13_executors():
    plan = _action_plan()
    assert len(plan) == 13


def test_build_task_atomic_action_plan_orders_actions_bottom_to_top():
    plan = _action_plan()
    picked = [
        step.pick_object_info.name
        for step in plan
        if step.action_type == "pick"
    ]
    placed_onto = [
        step.place_object_info.name
        for step in plan
        if step.action_type == "place"
    ]
    assert picked == list(_ORDER[1:])
    assert placed_onto == list(_ORDER[:-1])


def test_build_task_atomic_action_plan_place_reads_place_annotations():
    plan = _action_plan()
    place_steps = [step for step in plan if step.action_type == "place"]
    for step in place_steps:
        assert step.pick_object_info.mode == "active"
        assert step.pick_object_info.action == "place"
        assert step.place_object_info.mode == "passive"
        assert step.place_object_info.action == "place"


def test_build_task_atomic_action_plan_ends_with_back_to_default():
    plan = _action_plan()
    assert plan[-1].action_type == "back_to_default"


def test_build_task_atomic_action_plan_grasp_stays_near_vertical():
    """The demonstration should approach the way a person would.

    A wide cone lets the wrist reach in sideways. That still stacks, but
    the trajectory is what a policy imitates, so its shape matters on its
    own.
    """
    plan = _action_plan()
    pick_steps = [step for step in plan if step.action_type == "pick"]
    assert pick_steps
    for step in pick_steps:
        assert step.top_down_max_tilt_deg <= 15.0


def test_build_task_atomic_action_plan_place_keeps_the_cube_level():
    plan = _action_plan()
    place_steps = [step for step in plan if step.action_type == "place"]
    assert place_steps
    for step in place_steps:
        assert step.constrain == "align_dir_axis"


def test_build_task_atomic_action_plan_place_minimizes_joint_motion():
    plan = _action_plan()
    place_steps = [step for step in plan if step.action_type == "place"]
    assert place_steps
    for step in place_steps:
        assert step.candidate_selection == "joint_distance"
        assert step.axis_rotation_candidates == 4


def test_build_task_atomic_action_plan_unknown_robot_raises_value_error():
    with pytest.raises(ValueError, match="some_other_arm"):
        _action_plan(robot_name="some_other_arm")


def test_build_task_atomic_action_plan_unbound_roles_raises_value_error():
    with pytest.raises(ValueError, match="role_registry"):
        build_task_atomic_action_plan(_orchard_env())


def test_stack_cubes_definition_role_registry_builds_action_plan():
    plan = stack_cubes_env.StackCubesTaskDefinition.build_atomic_action_plan(
        _orchard_env(),
        role_registry=_role_registry(),
    )

    assert len(plan) == 13


def test_stack_cubes_registry_role_registry_builds_action_plan():
    plan = build_registered_action_plan(
        "stack_cubes",
        _orchard_env(),
        role_registry=_role_registry(),
    )

    assert len(plan) == 13
