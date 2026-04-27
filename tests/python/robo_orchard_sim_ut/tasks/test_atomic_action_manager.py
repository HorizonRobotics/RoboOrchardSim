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

import torch

from robo_orchard_sim.orchard_env.embodiments.dualarm_piper.profile import (
    DUALARM_PIPER_ROBOT_INFO_CFGS,
)
from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ResolvedManipulatorProfile,
)
from robo_orchard_sim.tasks.trajs_gen.atomic_action_manager import (
    AtomicActionManagerCfg,
)
from robo_orchard_sim.tasks.trajs_gen.base_executor import (
    BaseExecutor,
    BaseExecutorCfg,
    ObjectInfo,
    Trajectories,
)
from robo_orchard_sim.tasks.trajs_gen.executors.pick import PickExecutorCfg
from robo_orchard_sim.tasks.trajs_gen.manipulator_resolver import (
    BoundManipulatorResolver,
    ManipulatorBindingContext,
    PredicateManipulatorResolver,
)
from robo_orchard_sim.utils.config import ClassType_co


class _FakeTrajectoryExecutor(BaseExecutor):
    cfg: "_FakeTrajectoryExecutorCfg"

    def __init__(self, cfg: "_FakeTrajectoryExecutorCfg") -> None:
        super().__init__(cfg)
        self.plan_count = 0

    def plan(
        self,
        env: Any,
        context: ManipulatorBindingContext,
    ) -> Trajectories:
        self.plan_count += 1
        resolved = ResolvedManipulatorProfile(
            robot_name="robots/fake",
            manipulator_name=self.cfg.manipulator_name,
            joint_ids=(0,),
            joint_names=("joint",),
            gripper_joint_ids=(),
            gripper_joint_names=(),
            body_ids=(),
            body_names=(),
            ee_body_id=0,
            ee_body_name="ee",
        )
        self.last_resolved_manipulator = resolved
        trajectories = [
            torch.tensor(item, dtype=torch.float32, device=env.device)
            for item in self.cfg.trajectory
        ]
        return Trajectories(
            trajectories=trajectories,
            success=self.cfg.success,
            resolved_manipulator=resolved,
        )


class _FakeTrajectoryExecutorCfg(BaseExecutorCfg):
    class_type: ClassType_co[_FakeTrajectoryExecutor] = _FakeTrajectoryExecutor
    manipulator_name: str
    trajectory: list[list[list[float]]]
    success: bool = True


class _FakeArticulation:
    def __init__(self) -> None:
        self.data = type(
            "_Data",
            (),
            {
                "root_pos_w": torch.tensor(
                    [[0.0, 0.0, 0.0]], dtype=torch.float32
                )
            },
        )()

    def find_joints(self, names):
        mapping = {
            "left_joint[1-6]": (
                [0, 1, 2, 3, 4, 5],
                [f"left_joint{i}" for i in range(1, 7)],
            ),
            "left_joint7": ([6], ["left_joint7"]),
            "left_joint8": ([7], ["left_joint8"]),
            "right_joint[1-6]": (
                [8, 9, 10, 11, 12, 13],
                [f"right_joint{i}" for i in range(1, 7)],
            ),
            "right_joint7": ([14], ["right_joint7"]),
            "right_joint8": ([15], ["right_joint8"]),
        }
        ids = []
        resolved_names = []
        for name in names:
            if name not in mapping:
                continue
            found_ids, found_names = mapping[name]
            ids.extend(found_ids)
            resolved_names.extend(found_names)
        return ids, resolved_names

    def find_bodies(self, names):
        mapping = {
            "left_base_link": ([10], ["left_base_link"]),
            "left_link6": ([16], ["left_link6"]),
            "right_base_link": ([20], ["right_base_link"]),
            "right_link6": ([26], ["right_link6"]),
        }
        if isinstance(names, str):
            return mapping.get(names, ([], []))

        ids = []
        resolved_names = []
        for name in names:
            if name not in mapping:
                continue
            found_ids, found_names = mapping[name]
            ids.extend(found_ids)
            resolved_names.extend(found_names)
        return ids, resolved_names


class _FakeEnv:
    def __init__(self) -> None:
        self.scene: dict[str, Any] = {
            "robots/dualarm_piper": _FakeArticulation()
        }
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.use_left = True


_FAKE_LEFT_ARM_KEY = "robots/fake/left_arm"
_FAKE_RIGHT_ARM_KEY = "robots/fake/right_arm"
_PIPER_LEFT_ARM_KEY = "robots/dualarm_piper/left_arm"
_PIPER_RIGHT_ARM_KEY = "robots/dualarm_piper/right_arm"


def test_atomic_action_manager_pick_get_action_returns_correct_trajectory():
    manager = AtomicActionManagerCfg()()
    executor_cfg = PickExecutorCfg(
        robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["left_arm"],
        pick_object_info=ObjectInfo(
            name="objects/pick_object",
            mode="active",
            action="pick",
            part="gripper",
        ),
    )

    manager.register([executor_cfg])
    actions, _ = manager.get_action(_FakeEnv())

    assert set(actions) == {_PIPER_LEFT_ARM_KEY}
    assert torch.allclose(
        actions[_PIPER_LEFT_ARM_KEY],
        torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.04, 0.04]],
            dtype=torch.float32,
        ),
    )


def test_atomic_action_manager_pick_get_action_state_shows_running_status():
    manager = AtomicActionManagerCfg()()
    executor_cfg = PickExecutorCfg(
        robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["left_arm"],
        pick_object_info=ObjectInfo(
            name="objects/pick_object",
            mode="active",
            action="pick",
            part="gripper",
        ),
    )

    manager.register([executor_cfg])
    _, state = manager.get_action(_FakeEnv())

    assert state.running_actions[_PIPER_LEFT_ARM_KEY].action_type == "pick"
    assert state.running_actions[_PIPER_LEFT_ARM_KEY].status == "RUNNING"
    assert state.running_actions[_PIPER_LEFT_ARM_KEY].success is True


def test_atomic_action_manager_output_key_includes_resolved_robot_name():
    manager = AtomicActionManagerCfg()()
    manager.register(
        [
            _FakeTrajectoryExecutorCfg(
                robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["left_arm"],
                manipulator_name="left_arm",
                trajectory=[[[1.0]]],
            )
        ]
    )

    actions, _ = manager.get_action(_FakeEnv())

    assert set(actions) == {_FAKE_LEFT_ARM_KEY}


def test_atomic_action_manager_state_key_includes_resolved_robot_name():
    manager = AtomicActionManagerCfg()()
    manager.register(
        [
            _FakeTrajectoryExecutorCfg(
                robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["left_arm"],
                manipulator_name="left_arm",
                trajectory=[[[1.0]]],
            )
        ]
    )

    _, state = manager.get_action(_FakeEnv())

    assert set(state.running_actions) == {_FAKE_LEFT_ARM_KEY}


def test_atomic_action_manager_dynamic_resolver_selects_right_action_key():
    manager = AtomicActionManagerCfg()()
    robot_info = PredicateManipulatorResolver(
        predicate=lambda env: False,
        true_robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["left_arm"],
        false_robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["right_arm"],
    )
    executor_cfg = PickExecutorCfg(
        robot_info=robot_info,
        pick_object_info=ObjectInfo(
            name="objects/pick_object",
            mode="active",
            action="pick",
            part="gripper",
        ),
    )

    manager.register([executor_cfg])
    actions, _ = manager.get_action(_FakeEnv())
    _, state = manager.get_action(_FakeEnv())

    assert set(actions) == {_PIPER_RIGHT_ARM_KEY}
    assert state.running_actions[_PIPER_RIGHT_ARM_KEY].status == "COMPLETED"


def test_atomic_action_manager_reset_sequence_reselects_bound_arm():
    manager = AtomicActionManagerCfg()()
    robot_info = BoundManipulatorResolver(
        binding_key="test.pick_move_place_arm",
        selector=PredicateManipulatorResolver(
            predicate=lambda env: env.use_left,
            true_robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["left_arm"],
            false_robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["right_arm"],
        ),
    )
    executor_cfg = PickExecutorCfg(
        robot_info=robot_info,
        pick_object_info=ObjectInfo(
            name="objects/pick_object",
            mode="active",
            action="pick",
            part="gripper",
        ),
    )
    env = _FakeEnv()

    manager.register([executor_cfg, executor_cfg])
    first_actions, _ = manager.get_action(env)
    env.use_left = False
    second_actions, _ = manager.get_action(env)
    manager.reset_sequence()
    third_actions, _ = manager.get_action(env)

    assert [
        set(first_actions),
        set(second_actions),
        set(third_actions),
    ] == [{_PIPER_LEFT_ARM_KEY}, {_PIPER_LEFT_ARM_KEY}, {_PIPER_RIGHT_ARM_KEY}]


def test_atomic_action_manager_parallel_arms_emit_stepwise_actions():
    manager = AtomicActionManagerCfg()()
    manager.register(
        [
            _FakeTrajectoryExecutorCfg(
                robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["left_arm"],
                manipulator_name="left_arm",
                trajectory=[[[1.0], [2.0]]],
            ),
            _FakeTrajectoryExecutorCfg(
                robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["right_arm"],
                manipulator_name="right_arm",
                trajectory=[[[10.0], [20.0]]],
            ),
        ]
    )

    first_actions, first_state = manager.get_action(_FakeEnv())
    second_actions, second_state = manager.get_action(_FakeEnv())
    third_actions, third_state = manager.get_action(_FakeEnv())

    assert {
        "first": {
            name: action.tolist() for name, action in first_actions.items()
        },
        "second": {
            name: action.tolist() for name, action in second_actions.items()
        },
        "third": third_actions,
        "busy": [
            first_state.env_busy.tolist(),
            second_state.env_busy.tolist(),
            third_state.env_busy.tolist(),
        ],
    } == {
        "first": {
            _FAKE_LEFT_ARM_KEY: [[1.0]],
            _FAKE_RIGHT_ARM_KEY: [[10.0]],
        },
        "second": {
            _FAKE_LEFT_ARM_KEY: [[2.0]],
            _FAKE_RIGHT_ARM_KEY: [[20.0]],
        },
        "third": {},
        "busy": [[True], [True], [False]],
    }


def test_atomic_action_manager_same_arm_waits_for_previous_completion():
    manager = AtomicActionManagerCfg()()
    manager.register(
        [
            _FakeTrajectoryExecutorCfg(
                robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["left_arm"],
                manipulator_name="left_arm",
                trajectory=[[[1.0], [2.0]]],
            ),
            _FakeTrajectoryExecutorCfg(
                robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["left_arm"],
                manipulator_name="left_arm",
                trajectory=[[[3.0]]],
            ),
        ]
    )

    outputs = [manager.get_action(_FakeEnv())[0] for _ in range(4)]

    left_arm_outputs = [
        output.get(_FAKE_LEFT_ARM_KEY, torch.empty(0)).tolist()
        for output in outputs
    ]

    assert left_arm_outputs == [
        [[1.0]],
        [[2.0]],
        [[3.0]],
        [],
    ]


def test_atomic_action_manager_lower_priority_blocks_higher_priority():
    manager = AtomicActionManagerCfg()()
    manager.register(
        [
            _FakeTrajectoryExecutorCfg(
                robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["left_arm"],
                manipulator_name="left_arm",
                priority=1,
                trajectory=[[[1.0]]],
            ),
            _FakeTrajectoryExecutorCfg(
                robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["right_arm"],
                manipulator_name="right_arm",
                priority=2,
                trajectory=[[[2.0]]],
            ),
        ]
    )

    states = []
    outputs = []
    for _ in range(3):
        actions, state = manager.get_action(_FakeEnv())
        outputs.append(set(actions))
        states.append(state.current_priority)

    assert outputs == [{_FAKE_LEFT_ARM_KEY}, {_FAKE_RIGHT_ARM_KEY}, set()]
    assert states == [1, 2, None]


def test_atomic_action_manager_failed_executor_emits_no_actions():
    manager = AtomicActionManagerCfg()()
    manager.register(
        [
            _FakeTrajectoryExecutorCfg(
                robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["left_arm"],
                manipulator_name="left_arm",
                trajectory=[[[1.0]]],
                success=False,
            )
        ]
    )

    actions, _ = manager.get_action(_FakeEnv())

    assert actions == {}


def test_atomic_action_manager_failed_executor_state_shows_failed_status():
    manager = AtomicActionManagerCfg()()
    manager.register(
        [
            _FakeTrajectoryExecutorCfg(
                robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["left_arm"],
                manipulator_name="left_arm",
                trajectory=[[[1.0]]],
                success=False,
            )
        ]
    )

    _, state = manager.get_action(_FakeEnv())

    assert state.running_actions[_FAKE_LEFT_ARM_KEY].status == "FAILED"
    assert state.running_actions[_FAKE_LEFT_ARM_KEY].success is False


def test_atomic_action_manager_mismatched_env_trajs_raises_value_error():
    manager = AtomicActionManagerCfg()()
    manager.register(
        [
            _FakeTrajectoryExecutorCfg(
                robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["left_arm"],
                manipulator_name="left_arm",
                trajectory=[[[1.0]], [[2.0]]],
            )
        ]
    )

    try:
        manager.get_action(_FakeEnv())
    except ValueError as exc:
        assert "one trajectory per env" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for mismatched trajectories."
        )


def test_atomic_action_manager_segment_runs_until_all_actions_complete():
    manager = AtomicActionManagerCfg()()
    manager.register(
        [
            _FakeTrajectoryExecutorCfg(
                robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["left_arm"],
                manipulator_name="left_arm",
                trajectory=[[[1.0]]],
            )
        ]
    )

    actions, _ = manager.get_action(_FakeEnv())
    _, state = manager.get_action(_FakeEnv())

    assert set(actions) == {_FAKE_LEFT_ARM_KEY}
    assert state.pending_count == 0


def test_atomic_action_manager_clear_allows_registering_new_segment():
    manager = AtomicActionManagerCfg()()
    manager.register(
        [
            _FakeTrajectoryExecutorCfg(
                robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["left_arm"],
                manipulator_name="left_arm",
                trajectory=[[[1.0]]],
            )
        ]
    )
    # Run first segment to completion
    manager.get_action(_FakeEnv())
    manager.get_action(_FakeEnv())

    manager.clear()
    manager.register(
        [
            _FakeTrajectoryExecutorCfg(
                robot_info=DUALARM_PIPER_ROBOT_INFO_CFGS["right_arm"],
                manipulator_name="right_arm",
                trajectory=[[[2.0]]],
            )
        ]
    )

    actions, state = manager.get_action(_FakeEnv())

    assert set(actions) == {_FAKE_RIGHT_ARM_KEY}
    assert _FAKE_LEFT_ARM_KEY not in actions
    assert state.running_actions[_FAKE_RIGHT_ARM_KEY].status == "COMPLETED"
