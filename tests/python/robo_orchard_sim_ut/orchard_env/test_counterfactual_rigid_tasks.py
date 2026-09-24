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

"""Integration tests for rigid counterfactual validator dispatch."""

from types import SimpleNamespace

import pytest
import torch

from robo_orchard_sim.orchard_env.assets import RigidObjectSpec, TaskAssets
from robo_orchard_sim.orchard_env.task_templates.pick_task import PickTask
from robo_orchard_sim.orchard_env.task_templates.place_a2b_task import (
    PlaceA2BTask,
)
from robo_orchard_sim.task_components.instructions.base import (
    InstructionWrapper,
)
from robo_orchard_sim.task_components.role_registry import (
    RoleRegistry,
    TargetRef,
)
from robo_orchard_sim.task_components.validators.base import GripperRange
from robo_orchard_sim.task_components.validators.context import (
    ValidatorContext,
    ValidatorRobotContext,
)


class _RigidAsset:
    def __init__(self, name: str, x: float) -> None:
        root_state = torch.zeros((1, 13), dtype=torch.float32)
        root_state[0, 0] = x
        root_state[0, 3] = 1.0
        self.cfg = SimpleNamespace(prim_path=f"/World/{name}")
        self.data = SimpleNamespace(
            root_state_w=root_state,
            root_pos_w=root_state[:, :3],
            root_quat_w=root_state[:, 3:7],
            root_lin_vel_w=root_state[:, 7:10],
            root_ang_vel_w=root_state[:, 10:13],
        )

    def set_position(self, x: float, z: float) -> None:
        self.data.root_pos_w[0] = torch.tensor([x, 0.0, z])


class _RobotAsset:
    def __init__(self) -> None:
        self.body_names = ["tool", "left_finger", "right_finger"]
        self.data = SimpleNamespace(
            body_com_pos_w=torch.tensor(
                [[[100.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
            ),
            body_pos_w=torch.tensor(
                [[[100.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
            ),
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 3]),
            joint_pos=torch.tensor([[0.0]]),
        )

    def find_bodies(self, name: str):
        return [self.body_names.index(name)], [name]

    def find_joints(self, name: str):
        return [0], [name]


class _Scene(dict):
    stage = object()


class _ContactSensor:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(filter_prim_paths_expr=[])
        self.contact_physx_view = SimpleNamespace(filter_paths=[[]])
        self.data = SimpleNamespace(force_matrix_w=None, contact_pos_w=None)

    def _initialize_impl(self) -> None:
        paths = list(self.cfg.filter_prim_paths_expr)
        self.contact_physx_view.filter_paths = [paths]
        self.data.force_matrix_w = torch.zeros(
            (1, 1, len(paths), 3),
            dtype=torch.float32,
        )

        self.data.contact_pos_w = torch.full_like(
            self.data.force_matrix_w, torch.nan
        )

    def set_force(
        self,
        prim_path: str,
        force: tuple[float, float, float],
        position: tuple[float, float, float],
    ) -> None:
        column = self.contact_physx_view.filter_paths[0].index(prim_path)
        self.data.force_matrix_w[0, 0, column] = torch.tensor(force)
        self.data.contact_pos_w[0, 0, column] = torch.tensor(position)

    def clear_forces(self) -> None:
        if self.data.force_matrix_w is not None:
            self.data.force_matrix_w.zero_()
            self.data.contact_pos_w.fill_(torch.nan)


class _Env:
    def __init__(self, scene_names: list[str]) -> None:
        self.num_envs = 1
        self.scene = _Scene(
            {
                name: _RigidAsset(name, float(index * 2))
                for index, name in enumerate(scene_names)
            }
        )
        self.scene["robots/test"] = _RobotAsset()
        self.scene["sensors/gripper_contact_0_0"] = _ContactSensor()
        self.scene["sensors/gripper_contact_0_1"] = _ContactSensor()


def _set_end_effector(env: _Env, scene_name: str) -> None:
    env.scene["robots/test"].data.body_pos_w[0, 0] = env.scene[
        scene_name
    ].data.root_pos_w[0]


def _set_grasp(env: _Env, scene_name: str) -> None:
    target_path = env.scene[scene_name].cfg.prim_path
    first = env.scene["sensors/gripper_contact_0_0"]
    second = env.scene["sensors/gripper_contact_0_1"]
    first.clear_forces()
    second.clear_forces()
    first.set_force(target_path, (-1.0, 0.0, 0.0), (-0.02, 0.0, 0.0))
    second.set_force(target_path, (1.0, 0.0, 0.0), (0.02, 0.0, 0.0))


def _clear_contacts(env: _Env) -> None:
    env.scene["sensors/gripper_contact_0_0"].clear_forces()
    env.scene["sensors/gripper_contact_0_1"].clear_forces()


def _set_gripper_open(env: _Env) -> None:
    env.scene["robots/test"].data.joint_pos[0, 0] = 1.0


def _object(name: str) -> RigidObjectSpec:
    return RigidObjectSpec(
        name=name,
        usd_path=f"/tmp/{name}.usd",
    )


def _context(**bindings: str) -> ValidatorContext:
    registry = RoleRegistry()
    for role_id, scene_name in bindings.items():
        registry.bind_one(0, role_id, TargetRef(scene_name))
    return ValidatorContext(
        robot=ValidatorRobotContext(
            robot_name="robots/test",
            ee_links=("tool",),
            tcp_offsets=(("tool", (0.0, 0.0, 0.0)),),
            gripper_joints=(
                GripperRange(
                    name="gripper_joint",
                    open_val=1.0,
                    close_val=0.0,
                ),
            ),
            gripper_body_groups=(("left_finger", "right_finger"),),
        ),
        role_registry=registry,
    )


def _build_runtime(task, context: ValidatorContext):
    env = _Env(task.get_operable_scene_names())
    context.capture_init_states(env, task.get_operable_scene_names())
    validator = task.build_validator(context=context)
    validator.reset()
    return env, validator


def test_pick_generic_instruction_builds_counterfactual_validator() -> None:
    role_candidates = {
        "pick": [_object("apple")],
        "distractors": [_object("mug")],
    }
    task = PickTask(
        TaskAssets(role_candidates=role_candidates),
        instruction=InstructionWrapper("generic_object"),
    )

    context = _context(pick="objects/apple")
    env, validator = _build_runtime(task, context)
    env.scene["objects/mug"].set_position(2.0, 0.1)
    _set_end_effector(env, "objects/mug")

    for _ in range(15):
        step_output = validator.evaluate(env)
    _set_grasp(env, "objects/mug")
    step_output = validator.evaluate(env)
    final_output = validator.finalize()

    assert (
        validator.fixed_horizon,
        step_output.success,
        final_output.success,
    ) == (True, False, True)


def test_pick_task_empty_instruction_no_behavior_returns_success() -> None:
    task = PickTask(
        TaskAssets(role_candidates={"pick": [_object("apple")]}),
        instruction=InstructionWrapper("empty"),
    )

    validator = task.build_validator(
        context=_context(pick="objects/apple"),
    )
    output = validator.finalize()

    assert (output.success, output.progress) == (True, 1.0)


def test_counterfactual_cross_entity_stages_returns_failure() -> None:
    task = PickTask(
        TaskAssets(
            role_candidates={
                "pick": [_object("apple")],
                "distractors": [_object("mug")],
            }
        ),
        instruction=InstructionWrapper("generic_object"),
    )
    context = _context(pick="objects/apple")
    env, validator = _build_runtime(task, context)
    _set_end_effector(env, "objects/apple")
    for _ in range(15):
        validator.evaluate(env)

    _set_grasp(env, "objects/mug")
    env.scene["objects/mug"].set_position(2.0, 0.1)
    validator.evaluate(env)

    assert validator.finalize().success is False


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("initial_overlap", False),
        ("different_subject", False),
        ("same_subject", True),
    ],
)
def test_place_task_counterfactual_behavior_returns_expected_result(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    expected: bool,
) -> None:
    task = PlaceA2BTask(
        TaskAssets(
            role_candidates={
                "pick": [_object("apple")],
                "place": [_object("basket")],
                "distractors_pick": [_object("mug")],
            }
        ),
        instruction=InstructionWrapper("place_a2b_generic_object"),
    )
    context = _context(pick="objects/apple", place="objects/basket")
    env, validator = _build_runtime(task, context)
    monkeypatch.setattr(
        "robo_orchard_sim.task_components.validators.utils."
        "is_object_center_in_obb",
        lambda stage, prim_path, object_pose_array, subject_pos, idx_env: (
            float(subject_pos[0]) == 9.0
        ),
    )

    if scenario != "initial_overlap":
        _set_end_effector(env, "objects/apple")
        for _ in range(15):
            validator.evaluate(env)
        _set_grasp(env, "objects/apple")
        env.scene["objects/apple"].set_position(0.0, 0.1)
        validator.evaluate(env)
        env.scene["objects/apple"].set_position(0.0, 0.0)

    placed_subject = (
        "objects/mug" if scenario == "different_subject" else "objects/apple"
    )
    env.scene[placed_subject].set_position(9.0, 0.0)
    _set_gripper_open(env)
    validator.evaluate(env)

    assert validator.finalize().success is expected


def test_pick_task_normal_instruction_preserves_normal_success_path() -> None:
    task = PickTask(
        TaskAssets(role_candidates={"pick": [_object("apple")]}),
        instruction=InstructionWrapper("pick_default"),
    )
    context = _context(pick="objects/apple")
    env, validator = _build_runtime(task, context)
    _set_end_effector(env, "objects/apple")
    for _ in range(15):
        validator.evaluate(env)
    _set_grasp(env, "objects/apple")
    env.scene["objects/apple"].set_position(0.0, 0.1)
    _set_end_effector(env, "objects/apple")
    output = validator.evaluate(env)

    assert (validator.fixed_horizon, output.success) == (False, True)


def test_pick_task_transient_reach_does_not_advance_progress() -> None:
    task = PickTask(
        TaskAssets(role_candidates={"pick": [_object("apple")]}),
        instruction=InstructionWrapper("pick_default"),
    )
    context = _context(pick="objects/apple")
    env, validator = _build_runtime(task, context)
    _set_end_effector(env, "objects/apple")
    for _ in range(14):
        validator.evaluate(env)
    env.scene["robots/test"].data.body_pos_w[0, 0, 0] = 100.0

    output = validator.evaluate(env)

    assert output.progress == 0.0


def test_pick_task_lift_without_current_grasp_returns_incomplete() -> None:
    task = PickTask(
        TaskAssets(role_candidates={"pick": [_object("apple")]}),
        instruction=InstructionWrapper("pick_default"),
    )
    context = _context(pick="objects/apple")
    env, validator = _build_runtime(task, context)
    _set_end_effector(env, "objects/apple")
    for _ in range(15):
        validator.evaluate(env)
    _set_grasp(env, "objects/apple")
    validator.evaluate(env)
    _clear_contacts(env)
    env.scene["objects/apple"].set_position(0.0, 0.1)

    output = validator.evaluate(env)

    assert (output.progress, output.success) == (0.75, False)
