## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

"""Tests for direction-agnostic, embodiment-derived gripper-open judging."""

import importlib

import pytest
import torch

from robo_orchard_sim.task_components.validators.base import GripperRange


class _DummyRobot:
    def __init__(self, joint_positions, joint_names):
        self.data = type(
            "D", (), {"joint_pos": torch.tensor(joint_positions)}
        )()
        self._joint_names = list(joint_names)

    def find_joints(self, name):
        return [self._joint_names.index(name)], [name]


class _DummyEnv:
    def __init__(self, scene):
        self.scene = scene


def _checkers():
    return importlib.import_module(
        "robo_orchard_sim.task_components.validators.checkers"
    )


@pytest.mark.parametrize(
    "open_val, close_val, value, expected",
    [
        (0.05, 0.0, 0.041, True),
        (0.05, 0.0, 0.039, False),
        (0.05, 0.0, 0.04, True),
        (0.0, 0.725, 0.02, True),
        (0.0, 0.725, 0.5, False),
        (-0.05, 0.0, -0.045, True),
        (-0.05, 0.0, -0.02, False),
        (0.1, 0.1, 0.3, True),
    ],
    ids=[
        "piper-above",
        "piper-below",
        "piper-exact-boundary",
        "inverted-open",
        "inverted-closed",
        "negative-open",
        "negative-not-open",
        "degenerate",
    ],
)
def test_both_gripper_open_matches_expected_for_joint_range(
    open_val, close_val, value, expected
):
    env = _DummyEnv({"robots/robot": _DummyRobot([[value]], ["j"])})
    spec = GripperRange(name="j", open_val=open_val, close_val=close_val)
    checker = _checkers().is_both_gripper_open(
        gripper_joints=(spec,), robot_name="robots/robot"
    )

    assert checker(env, env_idx=0) is expected


def test_within_xy_gate_gripper_closed_blocks_containment(monkeypatch):
    checkers = _checkers()
    utils_module_name = "robo_orchard_sim.task_components.validators.utils"
    utils = importlib.import_module(utils_module_name)
    monkeypatch.setattr(utils, "is_object_center_in_obb", lambda *a, **k: True)

    class _Obj:
        data = type(
            "D",
            (),
            {
                "root_pos_w": torch.zeros((2, 3)),
                "root_state_w": torch.zeros((2, 13)),
            },
        )()
        cfg = type("C", (), {"prim_path": "/World/envs/env_.*/Obj"})()

    scene = type("S", (dict,), {"stage": object()})(
        {
            "objects/cube": _Obj(),
            "objects/goal": _Obj(),
            "robots/robot": _DummyRobot([[0.01], [0.05]], ["left_joint7"]),
        }
    )
    spec = GripperRange(name="left_joint7", open_val=0.05, close_val=0.0)
    checker = checkers.is_within_xy(
        "objects/cube",
        "objects/goal",
        require_gripper_open=True,
        robot_name="robots/robot",
        gripper_joints=(spec,),
    )

    assert checker(_DummyEnv(scene), env_idx=0) is False  # gripper closed
    assert checker(_DummyEnv(scene), env_idx=1) is True  # gripper open
