## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

"""Unit tests for the mutable ValidatorContext role portal."""

import pytest
import torch

from robo_orchard_sim.task_components.role_registry import (
    RoleRegistry,
    TargetRef,
)
from robo_orchard_sim.task_components.validators.context import (
    ValidatorContext,
    ValidatorRobotContext,
)


class _EmbodimentWithoutRobots:
    scene_name = "robots/none"

    def get_robot_info_cfgs(self):
        return {}


class _FakeObjectData:
    def __init__(self, x: float) -> None:
        self.root_pos_w = torch.tensor([[x, 0.0, 0.0]])
        self.root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.root_lin_vel_w = torch.zeros((1, 3))
        self.root_ang_vel_w = torch.zeros((1, 3))


class _FakeEnv:
    def __init__(self, **positions: float) -> None:
        self.scene = {
            name: type("_Obj", (), {"data": _FakeObjectData(x)})()
            for name, x in positions.items()
        }


def _make_context() -> tuple[ValidatorContext, RoleRegistry]:
    registry = RoleRegistry()
    context = ValidatorContext(
        robot=ValidatorRobotContext(robot_name="robots/dualarm_piper"),
        role_registry=registry,
    )
    return context, registry


def test_context_exposes_robot_and_role_registry():
    context, registry = _make_context()

    assert context.robot is not None
    assert context.robot.robot_name == "robots/dualarm_piper"
    assert context.role_registry is registry


def test_scene_name_of_resolves_through_the_registry():
    context, registry = _make_context()
    registry.bind_one(0, "pick", TargetRef("pick_1"))

    assert context.scene_name_of("pick") == "pick_1"


def test_a_semantic_name_resolves_to_its_owning_entity():
    context, registry = _make_context()
    registry.bind_one(0, "place", TargetRef("cabinet", "UpperDrawerJoint"))

    assert context.scene_name_of("place") == "cabinet"


def test_rebinding_the_registry_changes_what_the_context_resolves():
    context, registry = _make_context()

    registry.bind_one(0, "pick", TargetRef("pick_0"))
    assert context.scene_name_of("pick") == "pick_0"

    registry.bind_one(0, "pick", TargetRef("pick_1"))
    assert context.scene_name_of("pick") == "pick_1"


def test_captured_states_keep_pose_and_velocity():
    context, _ = _make_context()

    context.capture_init_states(_FakeEnv(cube=0.5), ["cube"])

    state = context.init_state_of("cube")
    assert state.shape == (13,)
    assert state[0].item() == pytest.approx(0.5)


def test_init_and_final_states_are_kept_apart():
    context, _ = _make_context()

    context.capture_init_states(_FakeEnv(cube=0.1), ["cube"])
    context.capture_final_states(_FakeEnv(cube=0.9), ["cube"])

    assert context.init_state_of("cube")[0].item() == pytest.approx(0.1)
    assert context.final_state_of("cube")[0].item() == pytest.approx(0.9)


def test_all_envs_are_available_without_an_index():
    context, _ = _make_context()

    context.capture_init_states(_FakeEnv(cube=0.5), ["cube"])

    assert context.init_state_of("cube", env_idx=None).shape == (1, 13)


def test_reading_a_state_that_was_never_captured_says_so():
    context, _ = _make_context()

    with pytest.raises(RuntimeError, match="capture_init_states"):
        context.init_state_of("cube")

    with pytest.raises(RuntimeError, match="capture_final_states"):
        context.final_state_of("cube")


def test_from_embodiment_without_manipulator_still_carries_registry():
    registry = RoleRegistry()

    context = ValidatorContext.from_embodiment(
        _EmbodimentWithoutRobots(), registry
    )

    assert context.robot is None
    assert context.role_registry is registry
