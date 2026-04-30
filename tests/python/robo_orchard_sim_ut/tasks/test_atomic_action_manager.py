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
from typing import Any

import pytest
import torch

from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ResolvedManipulatorProfile,
)
from robo_orchard_sim.tasks.trajs_gen.atomic_action_manager import (
    AtomicActionManagerCfg,
)
from robo_orchard_sim.tasks.trajs_gen.base_executor import (
    BaseExecutor,
    BaseExecutorCfg,
    DebugTargetPose,
    Trajectories,
)
from robo_orchard_sim.tasks.trajs_gen.debug_vis import (
    AtomicActionDebugVisualizer,
)
from robo_orchard_sim.tasks.trajs_gen.manipulator_resolver import (
    BoundManipulatorResolver,
    ManipulatorBindingContext,
    PredicateManipulatorResolver,
)
from robo_orchard_sim.utils.config import ClassType_co

_LEFT_ARM_KEY = "robots/fake/left_arm"
_RIGHT_ARM_KEY = "robots/fake/right_arm"


class _FakePlanner:
    pass


class _FakePlannerInstance:
    def __init__(self, cfg: "_FakePlannerCfg", env_nums: int) -> None:
        self.env_nums = env_nums
        cfg.instances.append(self)


class _FakePlannerCfg:
    class_type: ClassType_co[_FakePlannerInstance] = _FakePlannerInstance

    def __init__(self) -> None:
        self.instances: list[_FakePlannerInstance] = []


class _FakeEnv:
    def __init__(self, num_envs: int = 1) -> None:
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.use_left = True


class _StaticManipulatorResolver:
    def __init__(
        self,
        *,
        robot_name: str = "robots/fake",
        manipulator_name: str = "left_arm",
    ) -> None:
        self.robot_name = robot_name
        self.manipulator_name = manipulator_name

    def resolve(
        self,
        env: Any,
        context: ManipulatorBindingContext | None = None,
    ) -> ResolvedManipulatorProfile:
        del env, context
        return ResolvedManipulatorProfile(
            robot_name=self.robot_name,
            manipulator_name=self.manipulator_name,
            joint_ids=(0,),
            joint_names=("joint",),
            gripper_joint_ids=(),
            gripper_joint_names=(),
            body_ids=(),
            body_names=(),
            ee_body_id=0,
            ee_body_name="ee",
            planner=_FakePlanner(),
        )


class _PlannerResolvingManipulatorResolver:
    def __init__(
        self,
        *,
        planner_cfg: _FakePlannerCfg,
        robot_name: str = "robots/fake",
        manipulator_name: str = "left_arm",
    ) -> None:
        self.planner_cfg = planner_cfg
        self.robot_name = robot_name
        self.manipulator_name = manipulator_name

    def resolve(
        self,
        env: Any,
        context: ManipulatorBindingContext | None = None,
    ) -> ResolvedManipulatorProfile:
        if context is None:
            raise ValueError("planner resolution requires context")
        planner = context.resolve_planner_instance(
            robot_name=self.robot_name,
            manipulator_name=self.manipulator_name,
            planner_cfg=self.planner_cfg,
            env_nums=env.num_envs,
        )
        return ResolvedManipulatorProfile(
            robot_name=self.robot_name,
            manipulator_name=self.manipulator_name,
            joint_ids=(0,),
            joint_names=("joint",),
            gripper_joint_ids=(),
            gripper_joint_names=(),
            body_ids=(),
            body_names=(),
            ee_body_id=0,
            ee_body_name="ee",
            planner=planner,
        )


class _FakeTrajectoryExecutor(BaseExecutor):
    cfg: "_FakeTrajectoryExecutorCfg"

    def plan(
        self,
        env: Any,
        context: ManipulatorBindingContext,
    ) -> Trajectories:
        resolved = self.cfg.resolve_manipulator_info(
            env,
            context=context,
        )
        trajectories = [
            torch.tensor(item, dtype=torch.float32, device=env.device)
            for item in self.cfg.trajectories
        ]
        target_poses = tuple(
            DebugTargetPose(
                name=name,
                pose_w=torch.tensor(
                    [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                    device=env.device,
                ),
            )
            for name in self.cfg.debug_target_names
        )
        return Trajectories(
            trajectories=trajectories,
            success=self.cfg.success,
            resolved_manipulator=resolved,
            debug_target_poses=target_poses,
        )


class _FakeTrajectoryExecutorCfg(BaseExecutorCfg):
    class_type: ClassType_co[_FakeTrajectoryExecutor] = _FakeTrajectoryExecutor
    trajectories: list[Any]
    success: bool = True
    debug_target_names: tuple[str, ...] = ()


def _make_cfg(
    *,
    manipulator_name: str = "left_arm",
    robot_name: str = "robots/fake",
    trajectories: list[Any] | None = None,
    priority: int = 0,
    action_type: str = "fake",
    success: bool = True,
    debug_target_names: tuple[str, ...] = (),
) -> _FakeTrajectoryExecutorCfg:
    return _FakeTrajectoryExecutorCfg(
        robot_info=_StaticManipulatorResolver(
            robot_name=robot_name,
            manipulator_name=manipulator_name,
        ),
        priority=priority,
        action_type=action_type,
        trajectories=[[[1.0]]] if trajectories is None else trajectories,
        success=success,
        debug_target_names=debug_target_names,
    )


def _make_manager(
    *executor_cfgs: BaseExecutorCfg,
    debug_vis: bool = False,
) -> Any:
    manager = AtomicActionManagerCfg(debug_vis=debug_vis)()
    manager.register(list(executor_cfgs))
    return manager


def _action_values(actions: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {key: value.tolist() for key, value in actions.items()}


def test_get_action_single_step_trajectory_emits_action_command():
    manager = _make_manager(_make_cfg(trajectories=[[[1.0, 2.0]]]))

    actions, _ = manager.get_action(_FakeEnv())

    assert _action_values(actions) == {_LEFT_ARM_KEY: [[1.0, 2.0]]}


def test_get_action_completed_single_step_reports_completed_state():
    manager = _make_manager(_make_cfg(trajectories=[[[1.0]]]))

    _, state = manager.get_action(_FakeEnv())

    assert (
        state.running_actions[_LEFT_ARM_KEY].status,
        state.pending_count,
        state.env_busy.tolist(),
    ) == ("COMPLETED", 0, [True])


def test_get_action_multi_step_trajectory_emits_stepwise_commands():
    manager = _make_manager(_make_cfg(trajectories=[[[1.0], [2.0]]]))
    env = _FakeEnv()

    outputs = [_action_values(manager.get_action(env)[0]) for _ in range(3)]

    assert outputs == [
        {_LEFT_ARM_KEY: [[1.0]]},
        {_LEFT_ARM_KEY: [[2.0]]},
        {},
    ]


def test_get_action_multi_env_shorter_trajectory_reuses_last_command():
    manager = _make_manager(_make_cfg(trajectories=[[[1.0], [2.0]], [[10.0]]]))
    env = _FakeEnv(num_envs=2)

    first_actions, first_state = manager.get_action(env)
    second_actions, second_state = manager.get_action(env)

    assert (
        _action_values(first_actions),
        first_state.env_busy.tolist(),
        _action_values(second_actions),
        second_state.env_busy.tolist(),
    ) == (
        {_LEFT_ARM_KEY: [[1.0], [10.0]]},
        [True, True],
        {_LEFT_ARM_KEY: [[2.0], [10.0]]},
        [True, False],
    )


def test_get_action_parallel_manipulators_emit_actions_same_tick():
    manager = _make_manager(
        _make_cfg(manipulator_name="left_arm", trajectories=[[[1.0]]]),
        _make_cfg(manipulator_name="right_arm", trajectories=[[[2.0]]]),
    )

    actions, _ = manager.get_action(_FakeEnv())

    assert _action_values(actions) == {
        _LEFT_ARM_KEY: [[1.0]],
        _RIGHT_ARM_KEY: [[2.0]],
    }


def test_get_action_same_manipulator_sequence_waits_until_idle():
    manager = _make_manager(
        _make_cfg(manipulator_name="left_arm", trajectories=[[[1.0], [2.0]]]),
        _make_cfg(manipulator_name="left_arm", trajectories=[[[3.0]]]),
    )
    env = _FakeEnv()

    outputs = [_action_values(manager.get_action(env)[0]) for _ in range(4)]

    assert outputs == [
        {_LEFT_ARM_KEY: [[1.0]]},
        {_LEFT_ARM_KEY: [[2.0]]},
        {_LEFT_ARM_KEY: [[3.0]]},
        {},
    ]


def test_get_action_lower_priority_action_waits_until_current_priority_done():
    manager = _make_manager(
        _make_cfg(
            manipulator_name="left_arm",
            priority=1,
            trajectories=[[[1.0]]],
        ),
        _make_cfg(
            manipulator_name="right_arm",
            priority=2,
            trajectories=[[[2.0]]],
        ),
    )
    env = _FakeEnv()

    snapshots = []
    for _ in range(3):
        actions, state = manager.get_action(env)
        snapshots.append((_action_values(actions), state.current_priority))

    assert snapshots == [
        ({_LEFT_ARM_KEY: [[1.0]]}, 1),
        ({_RIGHT_ARM_KEY: [[2.0]]}, 2),
        ({}, None),
    ]


def test_get_action_failed_executor_reports_failed_state():
    manager = _make_manager(_make_cfg(trajectories=[[[1.0]]], success=False))

    actions, state = manager.get_action(_FakeEnv())

    assert (
        actions,
        state.running_actions[_LEFT_ARM_KEY].status,
        state.running_actions[_LEFT_ARM_KEY].success,
        state.pending_count,
    ) == ({}, "FAILED", False, 0)


def test_get_action_mismatched_env_trajectory_count_raises_value_error():
    manager = _make_manager(_make_cfg(trajectories=[[[1.0]]]))

    with pytest.raises(ValueError, match="one trajectory per env"):
        manager.get_action(_FakeEnv(num_envs=2))


def test_get_action_resolved_robot_name_appears_in_action_key():
    manager = _make_manager(
        _make_cfg(
            robot_name="robots/custom",
            manipulator_name="left_arm",
            trajectories=[[[1.0]]],
        )
    )

    actions, _ = manager.get_action(_FakeEnv())

    assert _action_values(actions) == {"robots/custom/left_arm": [[1.0]]}


def test_get_action_resolved_robot_name_appears_in_state_key():
    manager = _make_manager(
        _make_cfg(
            robot_name="robots/custom",
            manipulator_name="left_arm",
            trajectories=[[[1.0]]],
        )
    )

    _, state = manager.get_action(_FakeEnv())

    assert set(state.running_actions) == {"robots/custom/left_arm"}


def test_get_action_status_uses_configured_action_type():
    manager = _make_manager(
        _make_cfg(action_type="place", trajectories=[[[1.0], [2.0]]])
    )

    _, state = manager.get_action(_FakeEnv())

    assert state.running_actions[_LEFT_ARM_KEY].action_type == "place"


def test_get_action_missing_action_type_infers_executor_name():
    manager = _make_manager(
        _make_cfg(action_type="", trajectories=[[[1.0], [2.0]]])
    )

    _, state = manager.get_action(_FakeEnv())

    assert state.running_actions[_LEFT_ARM_KEY].action_type == (
        "_faketrajectory"
    )


def test_get_action_predicate_resolver_false_branch_uses_selected_key():
    resolver = PredicateManipulatorResolver(
        predicate=lambda env: False,
        true_robot_info=_StaticManipulatorResolver(
            manipulator_name="left_arm"
        ),
        false_robot_info=_StaticManipulatorResolver(
            manipulator_name="right_arm"
        ),
    )
    manager = _make_manager(
        _FakeTrajectoryExecutorCfg(
            robot_info=resolver,
            action_type="fake",
            trajectories=[[[1.0]]],
        )
    )

    actions, _ = manager.get_action(_FakeEnv())

    assert _action_values(actions) == {_RIGHT_ARM_KEY: [[1.0]]}


def test_get_action_same_manipulator_reuses_planner_instance():
    planner_cfg = _FakePlannerCfg()
    trajectories = [[[1.0]], [[1.0]], [[1.0]]]
    manager = _make_manager(
        _FakeTrajectoryExecutorCfg(
            robot_info=_PlannerResolvingManipulatorResolver(
                planner_cfg=planner_cfg,
                manipulator_name="left_arm",
            ),
            priority=1,
            action_type="fake",
            trajectories=trajectories,
        ),
        _FakeTrajectoryExecutorCfg(
            robot_info=_PlannerResolvingManipulatorResolver(
                planner_cfg=planner_cfg,
                manipulator_name="right_arm",
            ),
            priority=1,
            action_type="fake",
            trajectories=trajectories,
        ),
        _FakeTrajectoryExecutorCfg(
            robot_info=_PlannerResolvingManipulatorResolver(
                planner_cfg=planner_cfg,
                manipulator_name="left_arm",
            ),
            priority=2,
            action_type="fake",
            trajectories=trajectories,
        ),
    )
    env = _FakeEnv(num_envs=3)

    manager.get_action(env)
    manager.get_action(env)

    assert [instance.env_nums for instance in planner_cfg.instances] == [3, 3]


def test_reset_sequence_bound_resolver_reselects_manipulator():
    resolver = BoundManipulatorResolver(
        binding_key="test.sequence_arm",
        selector=PredicateManipulatorResolver(
            predicate=lambda env: env.use_left,
            true_robot_info=_StaticManipulatorResolver(
                manipulator_name="left_arm"
            ),
            false_robot_info=_StaticManipulatorResolver(
                manipulator_name="right_arm"
            ),
        ),
    )
    manager = _make_manager(
        _FakeTrajectoryExecutorCfg(
            robot_info=resolver,
            action_type="fake",
            trajectories=[[[1.0]]],
        ),
        _FakeTrajectoryExecutorCfg(
            robot_info=resolver,
            action_type="fake",
            trajectories=[[[2.0]]],
        ),
    )
    env = _FakeEnv()

    first_actions, _ = manager.get_action(env)
    env.use_left = False
    second_actions, _ = manager.get_action(env)
    manager.reset_sequence()
    third_actions, _ = manager.get_action(env)

    assert [
        set(first_actions),
        set(second_actions),
        set(third_actions),
    ] == [{_LEFT_ARM_KEY}, {_LEFT_ARM_KEY}, {_RIGHT_ARM_KEY}]


def test_clear_completed_segment_allows_new_registered_segment():
    manager = _make_manager(_make_cfg(trajectories=[[[1.0]]]))
    env = _FakeEnv()
    manager.get_action(env)
    manager.get_action(env)

    manager.clear()
    manager.register(
        [_make_cfg(manipulator_name="right_arm", trajectories=[[[2.0]]])]
    )
    actions, _ = manager.get_action(env)

    assert _action_values(actions) == {_RIGHT_ARM_KEY: [[2.0]]}


def test_clear_default_keeps_cached_planner_instance():
    planner_cfg = _FakePlannerCfg()
    manager = _make_manager(
        _FakeTrajectoryExecutorCfg(
            robot_info=_PlannerResolvingManipulatorResolver(
                planner_cfg=planner_cfg
            ),
            action_type="fake",
            trajectories=[[[1.0]]],
        )
    )
    env = _FakeEnv()

    manager.get_action(env)
    manager.clear()
    manager.register(
        [
            _FakeTrajectoryExecutorCfg(
                robot_info=_PlannerResolvingManipulatorResolver(
                    planner_cfg=planner_cfg
                ),
                action_type="fake",
                trajectories=[[[2.0]]],
            )
        ]
    )
    manager.get_action(env)

    assert len(planner_cfg.instances) == 1


def test_clear_clear_planner_instances_discards_cached_planner_instance():
    planner_cfg = _FakePlannerCfg()
    manager = _make_manager(
        _FakeTrajectoryExecutorCfg(
            robot_info=_PlannerResolvingManipulatorResolver(
                planner_cfg=planner_cfg
            ),
            action_type="fake",
            trajectories=[[[1.0]]],
        )
    )
    env = _FakeEnv()

    manager.get_action(env)
    manager.clear(clear_planner_instances=True)
    manager.register(
        [
            _FakeTrajectoryExecutorCfg(
                robot_info=_PlannerResolvingManipulatorResolver(
                    planner_cfg=planner_cfg
                ),
                action_type="fake",
                trajectories=[[[2.0]]],
            )
        ]
    )
    manager.get_action(env)

    assert len(planner_cfg.instances) == 2


def test_manager_cfg_debug_vis_disabled_skips_target_marker(monkeypatch):
    records: list[str] = []

    def _record_pose(
        self: AtomicActionDebugVisualizer,
        *,
        marker_name: str,
        pose_w: torch.Tensor,
    ) -> None:
        del self, pose_w
        records.append(marker_name)

    monkeypatch.setattr(
        AtomicActionDebugVisualizer,
        "visualize_pose",
        _record_pose,
    )
    manager = _make_manager(
        _make_cfg(debug_target_names=("target",)),
        debug_vis=False,
    )

    manager.get_action(_FakeEnv())

    assert records == []


def test_manager_cfg_debug_vis_enabled_publishes_target_marker(monkeypatch):
    records: list[str] = []

    def _record_pose(
        self: AtomicActionDebugVisualizer,
        *,
        marker_name: str,
        pose_w: torch.Tensor,
    ) -> None:
        del self, pose_w
        records.append(marker_name)

    monkeypatch.setattr(
        AtomicActionDebugVisualizer,
        "visualize_pose",
        _record_pose,
    )
    manager = _make_manager(
        _make_cfg(debug_target_names=("target",)),
        debug_vis=True,
    )

    manager.get_action(_FakeEnv())

    assert records == ["robots_fake_left_arm_target"]


def test_clear_debug_vis_enabled_clears_markers(monkeypatch):
    records: list[str] = []

    def _record_pose(
        self: AtomicActionDebugVisualizer,
        *,
        marker_name: str,
        pose_w: torch.Tensor,
    ) -> None:
        del self, marker_name, pose_w
        records.append("visualize")

    def _record_clear(self: AtomicActionDebugVisualizer) -> None:
        del self
        records.append("clear")

    monkeypatch.setattr(
        AtomicActionDebugVisualizer,
        "visualize_pose",
        _record_pose,
    )
    monkeypatch.setattr(
        AtomicActionDebugVisualizer,
        "clear",
        _record_clear,
        raising=False,
    )
    manager = _make_manager(
        _make_cfg(debug_target_names=("target",)),
        debug_vis=True,
    )

    manager.get_action(_FakeEnv())
    manager.clear()

    assert records == ["visualize", "clear"]
