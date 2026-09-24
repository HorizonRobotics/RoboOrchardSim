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

"""Behavioral tests for instruction-ordered cube stacking."""

from __future__ import annotations
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from robo_orchard_sim.orchard_env.assets.object_spec import (
    ObjectSpec,
    RigidObjectSpec,
)
from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets
from robo_orchard_sim.orchard_env.task_templates.stack_cubes_task import (
    MEMBERS_ROLE,
    StackCubesTask,
    StackCubesTaskParams,
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


def _task(
    *,
    dwell_steps: int = 2,
    colors: tuple[str, ...] = ("red", "yellow", "blue", "green"),
) -> StackCubesTask:
    members: list[ObjectSpec] = [
        RigidObjectSpec(
            name=f"{color}_cube",
            usd_path=f"/tmp/{color}.usd",
            attributes={"color": (color,)},
        )
        for color in colors
    ]
    return StackCubesTask(
        assets=TaskAssets(role_candidates={MEMBERS_ROLE: members}),
        params=StackCubesTaskParams(success_dwell_steps=dwell_steps),
        instruction=InstructionWrapper("stack_cubes_color_order"),
    )


def _context(order: tuple[str, ...]) -> ValidatorContext:
    registry = RoleRegistry()
    registry.bind_many(
        0,
        MEMBERS_ROLE,
        [TargetRef(f"objects/{color}_cube") for color in order],
    )
    return ValidatorContext(
        robot=ValidatorRobotContext(
            robot_name="robots/panda",
            gripper_joints=(
                GripperRange(
                    name="finger_joint",
                    open_val=0.04,
                    close_val=0.0,
                ),
            ),
            gripper_body_groups=(("left_finger", "right_finger"),),
        ),
        role_registry=registry,
    )


def _rigid_object(position: tuple[float, float, float]) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([position], dtype=torch.float32),
            root_quat_w=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
            root_lin_vel_w=torch.zeros((1, 3), dtype=torch.float32),
            root_ang_vel_w=torch.zeros((1, 3), dtype=torch.float32),
        ),
        cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Cube"),
    )


class _Robot:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            joint_pos=torch.tensor([[0.04]], dtype=torch.float32),
            body_pos_w=torch.zeros((1, 2, 3), dtype=torch.float32),
        )

    def find_joints(self, name: str) -> tuple[list[int], list[str]]:
        return [0], [name]

    def find_bodies(self, name: str) -> tuple[list[int], list[str]]:
        body_id = {"left_finger": 0, "right_finger": 1}[name]
        return [body_id], [name]


class _ContactSensor:
    def __init__(self, force: float) -> None:
        self.force = force
        self.cfg = SimpleNamespace(filter_prim_paths_expr=[])
        self.contact_physx_view = SimpleNamespace(filter_paths=[[]])
        self.data = SimpleNamespace(force_matrix_w=None)

    def _initialize_impl(self) -> None:
        paths = list(self.cfg.filter_prim_paths_expr)
        self.contact_physx_view.filter_paths = [paths]
        forces = torch.zeros((1, 1, len(paths), 3), dtype=torch.float32)
        forces[..., 0] = self.force
        self.data.force_matrix_w = forces


def _env(
    physical_order: tuple[str, ...],
    *,
    contact_force: float = 0.0,
) -> SimpleNamespace:
    positions = {
        color: (0.0, 0.0, 0.025 + index * 0.05)
        for index, color in enumerate(physical_order)
    }
    return _env_from_positions(positions, contact_force=contact_force)


def _env_from_positions(
    positions: dict[str, tuple[float, float, float]],
    *,
    contact_force: float = 0.0,
) -> SimpleNamespace:
    scene: dict[str, Any] = {
        f"objects/{color}_cube": _rigid_object(position)
        for color, position in positions.items()
    }
    scene["robots/panda"] = _Robot()
    scene["sensors/gripper_contact_0_0"] = _ContactSensor(contact_force)
    scene["sensors/gripper_contact_0_1"] = _ContactSensor(contact_force)
    return SimpleNamespace(scene=scene, num_envs=1)


def test_stack_cubes_instruction_generated_binding_renders_matching_order():
    colors = ("orange", "blue", "magenta", "green")
    task = _task(colors=colors)
    context = _context(colors)
    assert task.instruction is not None

    instruction = task.instruction.render(
        task.build_instruction_context(
            None,
            actor_description_seed=0,
            context=context,
        )
    )

    assert (
        instruction
        == "Stack the cubes bottom-to-top: orange, blue, magenta, green."
    )


def test_stack_cubes_pose_reset_all_members_uses_random_non_overlap():
    task = _task()

    reset = task.get_event_cfg().terms["random_pose_event"]
    asset_cfgs = cast(Any, reset).asset_cfgs

    assert [asset.name for asset in asset_cfgs] == [
        "objects/red_cube",
        "objects/yellow_cube",
        "objects/blue_cube",
        "objects/green_cube",
    ]
    assert cast(Any, reset).mode == "random_non_overlap"


def test_stack_cubes_enabled_domain_randomization_adds_reset_terms():
    task = _task()
    task.params = StackCubesTaskParams(
        light_reset={
            "enabled": True,
            "preset": "default_distant_light",
        },
        texture_reset={
            "enabled": True,
            "preset": "default_table_texture",
        },
    )

    terms = task.get_event_cfg().terms

    assert "light_reset_event" in terms
    assert "texture_reset_event" in terms


def test_stack_validator_complete_stack_reaches_success_after_dwell() -> None:
    task = _task(dwell_steps=2)
    order = ("red", "blue", "green", "yellow")
    validator = task.build_validator(_context(order))
    env = _env(order)

    first = validator.evaluate(cast(Any, env))
    second = validator.evaluate(cast(Any, env))

    assert (first.progress, first.success) == (0.75, False)
    assert (second.progress, second.success) == (1.0, True)


def test_stack_cubes_validator_wrong_physical_order_does_not_progress():
    task = _task(dwell_steps=1)
    target_order = ("red", "blue", "green", "yellow")
    validator = task.build_validator(_context(target_order))

    result = validator.evaluate(
        cast(Any, _env(("red", "green", "blue", "yellow")))
    )

    assert (result.progress, result.success) == (0.0, False)


def test_stack_validator_gripper_contact_prevents_final_success() -> None:
    task = _task(dwell_steps=1)
    order = ("red", "blue", "green", "yellow")
    validator = task.build_validator(_context(order))

    result = validator.evaluate(cast(Any, _env(order, contact_force=1.0)))

    assert (result.progress, result.success) == (0.75, False)


@pytest.mark.parametrize(
    ("positions", "expected_progress"),
    [
        (
            {
                "red": (0.0, 0.0, 0.025),
                "blue": (0.2, 0.0, 0.025),
                "green": (0.4, 0.0, 0.025),
                "yellow": (0.6, 0.0, 0.025),
            },
            0.0,
        ),
        (
            {
                "red": (0.0, 0.0, 0.025),
                "blue": (0.0, 0.0, 0.075),
                "green": (0.3, 0.0, 0.025),
                "yellow": (0.5, 0.0, 0.025),
            },
            0.25,
        ),
        (
            {
                "red": (0.0, 0.0, 0.025),
                "blue": (0.0, 0.0, 0.075),
                "green": (0.0, 0.0, 0.125),
                "yellow": (0.3, 0.0, 0.025),
            },
            0.5,
        ),
    ],
)
def test_stack_validator_correct_prefix_returns_incremental_progress(
    positions: dict[str, tuple[float, float, float]],
    expected_progress: float,
) -> None:
    order = ("red", "blue", "green", "yellow")
    validator = _task(dwell_steps=2).build_validator(_context(order))

    result = validator.evaluate(cast(Any, _env_from_positions(positions)))

    assert result.progress == expected_progress
