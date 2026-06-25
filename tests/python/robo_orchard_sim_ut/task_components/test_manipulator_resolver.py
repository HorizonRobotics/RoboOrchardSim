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

from __future__ import annotations
from typing import Any, cast

import pytest

from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ResolvedManipulatorProfile,
)
from robo_orchard_sim.task_components.trajs_gen.manipulator_resolver import (
    BoundManipulatorResolver,
    ManipulatorBindingContext,
    ManipulatorPredicate,
    PredicateManipulatorResolver,
)


class _FakePlanner:
    pass


class _FakePlannerInstance:
    def __init__(self, cfg: "_FakePlannerCfg", env_nums: int) -> None:
        self.env_nums = env_nums
        cfg.instances.append(self)

    def close(self) -> None:
        pass


class _FakePlannerCfg:
    class_type = _FakePlannerInstance

    def __init__(self) -> None:
        self.instances: list[_FakePlannerInstance] = []


class _RobotInfo:
    def __init__(self, name: str) -> None:
        self.name = name

    def resolve(
        self, env: Any, context: Any = None
    ) -> ResolvedManipulatorProfile:
        return ResolvedManipulatorProfile(
            robot_name="robots/dualarm_piper",
            manipulator_name=self.name,
            joint_ids=(0,),
            joint_names=("joint1",),
            gripper_joint_ids=(),
            gripper_joint_names=(),
            body_ids=(),
            body_names=(),
            ee_body_id=0,
            ee_body_name="link1",
            planner=cast(Any, _FakePlanner()),
        )


class _Env:
    def __init__(self, use_left: bool) -> None:
        self.use_left = use_left


@pytest.mark.parametrize(
    ("predicate_result", "expected_name"),
    [
        (True, "left_arm"),
        (False, "right_arm"),
    ],
)
def test_predicate_resolver_bool_result_selects_expected_robot_info(
    predicate_result: bool,
    expected_name: str,
) -> None:
    robot_infos = {
        "left_arm": _RobotInfo("left_arm"),
        "right_arm": _RobotInfo("right_arm"),
    }
    resolver = PredicateManipulatorResolver(
        predicate=lambda env: predicate_result,
        true_robot_info=robot_infos["left_arm"],
        false_robot_info=robot_infos["right_arm"],
    )

    selected = resolver.select(_Env(use_left=True))

    assert selected is robot_infos[expected_name]


def test_predicate_resolver_non_bool_result_raises_type_error():
    resolver = PredicateManipulatorResolver(
        predicate=cast(ManipulatorPredicate, lambda env: "left"),
        true_robot_info=_RobotInfo("left_arm"),
        false_robot_info=_RobotInfo("right_arm"),
    )

    with pytest.raises(TypeError, match="predicate must return bool"):
        resolver.select(_Env(use_left=True))


def test_predicate_resolver_env_change_reselects_robot_info():
    env = _Env(use_left=True)
    resolver = PredicateManipulatorResolver(
        predicate=lambda current_env: current_env.use_left,
        true_robot_info=_RobotInfo("left_arm"),
        false_robot_info=_RobotInfo("right_arm"),
    )

    first = resolver.resolve(env)
    env.use_left = False
    second = resolver.resolve(env)

    assert [first.manipulator_name, second.manipulator_name] == [
        "left_arm",
        "right_arm",
    ]


def test_predicate_resolver_distinct_instances_select_independently():
    env = _Env(use_left=True)
    resolver = PredicateManipulatorResolver(
        predicate=lambda current_env: current_env.use_left,
        true_robot_info=_RobotInfo("left_arm"),
        false_robot_info=_RobotInfo("right_arm"),
    )

    first = resolver.resolve(env)
    env.use_left = False
    second = PredicateManipulatorResolver(
        predicate=lambda current_env: current_env.use_left,
        true_robot_info=_RobotInfo("left_arm"),
        false_robot_info=_RobotInfo("right_arm"),
    ).resolve(env)

    assert [first.manipulator_name, second.manipulator_name] == [
        "left_arm",
        "right_arm",
    ]


def test_bound_resolver_context_reset_reselects_robot_info():
    env = _Env(use_left=True)
    context = ManipulatorBindingContext()
    resolver = BoundManipulatorResolver(
        binding_key="test.pick_move_place_arm",
        selector=PredicateManipulatorResolver(
            predicate=lambda current_env: current_env.use_left,
            true_robot_info=_RobotInfo("left_arm"),
            false_robot_info=_RobotInfo("right_arm"),
        ),
    )

    first = resolver.resolve(env=env, context=context)
    env.use_left = False
    second = resolver.resolve(env=env, context=context)
    context.reset()
    third = resolver.resolve(env=env, context=context)

    assert [
        first.manipulator_name,
        second.manipulator_name,
        third.manipulator_name,
    ] == ["left_arm", "left_arm", "right_arm"]


def test_manipulator_context_reset_default_keeps_planner_instance():
    context = ManipulatorBindingContext()
    planner_cfg = _FakePlannerCfg()

    first = context.resolve_planner_instance(
        robot_name="robots/fake",
        manipulator_name="left_arm",
        planner_cfg=planner_cfg,
        env_nums=1,
    )
    context.reset()
    second = context.resolve_planner_instance(
        robot_name="robots/fake",
        manipulator_name="left_arm",
        planner_cfg=planner_cfg,
        env_nums=1,
    )

    assert (first is second, len(planner_cfg.instances)) == (True, 1)


def test_manipulator_context_reset_clear_planner_instances_creates_new_one():
    context = ManipulatorBindingContext()
    planner_cfg = _FakePlannerCfg()

    first = context.resolve_planner_instance(
        robot_name="robots/fake",
        manipulator_name="left_arm",
        planner_cfg=planner_cfg,
        env_nums=1,
    )
    context.reset(clear_planner_instances=True)
    second = context.resolve_planner_instance(
        robot_name="robots/fake",
        manipulator_name="left_arm",
        planner_cfg=planner_cfg,
        env_nums=1,
    )

    assert (first is second, len(planner_cfg.instances)) == (False, 2)
