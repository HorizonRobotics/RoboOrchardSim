## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

import importlib
import inspect

import torch

from robo_orchard_sim.tasks.validators.base import Validator, ValidatorActor


class _DummyObjectData:
    def __init__(self, positions, default_heights):
        self.root_pos_w = torch.tensor(positions, dtype=torch.float32)
        self.root_quat_w = torch.zeros(
            (len(positions), 4), dtype=torch.float32
        )
        self.root_quat_w[:, 0] = 1.0
        self.root_state_w = torch.zeros(
            (len(positions), 13), dtype=torch.float32
        )
        self.root_state_w[:, :3] = self.root_pos_w
        self.root_state_w[:, 3] = 1.0
        self.default_root_state = torch.zeros(
            (len(default_heights), 13), dtype=torch.float32
        )
        self.default_root_state[:, 2] = torch.tensor(
            default_heights, dtype=torch.float32
        )


class _DummyObject:
    def __init__(
        self,
        positions,
        default_heights,
        prim_path: str = "/World/envs/env_.*/Cube",
    ):
        self.data = _DummyObjectData(positions, default_heights)
        self.cfg = type("Cfg", (), {"prim_path": prim_path})()


class _DummyRobotData:
    def __init__(self, body_positions, joint_positions):
        self.body_com_pos_w = torch.tensor(body_positions, dtype=torch.float32)
        self.joint_pos = torch.tensor(joint_positions, dtype=torch.float32)


class _DummyRobot:
    def __init__(
        self, body_positions, joint_positions, body_names, joint_names
    ):
        self.data = _DummyRobotData(body_positions, joint_positions)
        self._body_names = list(body_names)
        self._joint_names = list(joint_names)

    def find_bodies(self, name):
        return [self._body_names.index(name)], [name]

    def find_joints(self, name):
        return [self._joint_names.index(name)], [name]


class _DummyEnv:
    def __init__(self, scene):
        self.scene = scene


class _DummyScene(dict):
    def __init__(self, *args, stage=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = stage


def test_checkers_can_be_imported_without_pxr():
    module = importlib.import_module(
        "robo_orchard_sim.tasks.validators.checkers"
    )
    assert hasattr(module, "lift")


def test_validator_forwards_env_idx_to_criteria():
    seen_env_indices = []

    def criterion(_env, env_idx=0):
        seen_env_indices.append(env_idx)
        return env_idx == 1

    validator = Validator(
        actors=[
            ValidatorActor(
                name="objects/cube", uuid="", category="", actor_type=""
            )
        ],
        criteria=[criterion],
        criteria_name=["criterion"],
    )

    result = validator.evaluate(_DummyEnv(scene={}), env_idx=1)

    assert result.success is True
    assert result.progress == 1.0
    assert seen_env_indices == [1]


def test_validator_does_not_treat_plain_second_arg_as_env_idx():
    class _Criterion:
        __signature__ = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    "env", inspect.Parameter.POSITIONAL_OR_KEYWORD
                ),
                inspect.Parameter(
                    "threshold", inspect.Parameter.POSITIONAL_OR_KEYWORD
                ),
            ]
        )

        def __call__(self, _env):
            return True

    validator = Validator(
        actors=[
            ValidatorActor(
                name="objects/cube", uuid="", category="", actor_type=""
            )
        ],
        criteria=[_Criterion()],
        criteria_name=["criterion"],
    )

    result = validator.evaluate(_DummyEnv(scene={}), env_idx=1)

    assert result.success is True
    assert result.progress == 1.0


def test_lift_checker_reads_requested_env_index():
    checkers = importlib.import_module(
        "robo_orchard_sim.tasks.validators.checkers"
    )
    env = _DummyEnv(
        scene={
            "objects/cube": _DummyObject(
                positions=[(0.0, 0.0, 0.50), (0.0, 0.0, 0.65)],
                default_heights=[0.50, 0.50],
            )
        }
    )

    checker = checkers.lift("objects/cube", threshold=0.05)

    assert not checker(env, env_idx=0)
    assert checker(env, env_idx=1)


def test_lift_checker_uses_per_env_default_height():
    checkers = importlib.import_module(
        "robo_orchard_sim.tasks.validators.checkers"
    )
    env = _DummyEnv(
        scene={
            "objects/cube": _DummyObject(
                positions=[(0.0, 0.0, 0.56), (0.0, 0.0, 0.65)],
                default_heights=[0.50, 0.60],
            )
        }
    )

    checker = checkers.lift("objects/cube", threshold=0.05)

    assert checker(env, env_idx=0) is True
    assert checker(env, env_idx=1) is False


def test_reach_checker_reads_requested_env_index():
    checkers = importlib.import_module(
        "robo_orchard_sim.tasks.validators.checkers"
    )
    env = _DummyEnv(
        scene={
            "objects/cube": _DummyObject(
                positions=[(0.5, 0.0, 0.5), (0.02, 0.0, 0.0)],
                default_heights=[0.0, 0.0],
            ),
            "robots/robot": _DummyRobot(
                body_positions=[
                    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                ],
                joint_positions=[[0.0, 0.0], [0.0, 0.0]],
                body_names=["left_link6", "right_link6"],
                joint_names=["left_joint7", "right_joint7"],
            ),
        }
    )

    checker = checkers.reach(
        "objects/cube",
        threshold=0.05,
        robot_name="robots/robot",
        ee_links=("left_link6",),
    )

    assert not checker(env, env_idx=0)
    assert checker(env, env_idx=1)


def test_alignment_xy_checker_reads_requested_env_index():
    checkers = importlib.import_module(
        "robo_orchard_sim.tasks.validators.checkers"
    )
    env = _DummyEnv(
        scene={
            "objects/cube": _DummyObject(
                positions=[(0.0, 0.0, 0.5), (0.1, 0.1, 0.5)],
                default_heights=[0.0, 0.0],
            ),
            "objects/goal": _DummyObject(
                positions=[(0.3, 0.3, 0.5), (0.105, 0.095, 0.5)],
                default_heights=[0.0, 0.0],
            ),
        }
    )

    checker = checkers.is_alignment_xy(
        "objects/cube",
        "objects/goal",
        eps=(0.02, 0.02),
    )

    assert not checker(env, env_idx=0)
    assert checker(env, env_idx=1)


def test_alignment_xyz_checker_reads_requested_env_index():
    checkers = importlib.import_module(
        "robo_orchard_sim.tasks.validators.checkers"
    )
    env = _DummyEnv(
        scene={
            "objects/base": _DummyObject(
                positions=[(0.0, 0.0, 0.5), (0.2, 0.2, 0.6)],
                default_heights=[0.0, 0.0],
            ),
            "objects/stack": _DummyObject(
                positions=[(0.2, 0.2, 0.2), (0.205, 0.195, 0.64)],
                default_heights=[0.0, 0.0],
            ),
        }
    )

    checker = checkers.is_alignment_xyz(
        "objects/base",
        "objects/stack",
        eps=(0.02, 0.02, 0.02),
        target_height_offset=0.04,
    )

    assert not checker(env, env_idx=0)
    assert checker(env, env_idx=1)


def test_gripper_checkers_read_requested_env_index():
    checkers = importlib.import_module(
        "robo_orchard_sim.tasks.validators.checkers"
    )
    env = _DummyEnv(
        scene={
            "robots/robot": _DummyRobot(
                body_positions=[
                    [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
                    [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
                ],
                joint_positions=[[0.01, 0.01], [0.05, 0.06]],
                body_names=["left_link6", "right_link6"],
                joint_names=["left_joint7", "right_joint7"],
            )
        }
    )

    left_checker = checkers.is_gripper_open(
        "left",
        open_gripper_threshold=0.04,
        robot_name="robots/robot",
    )
    both_checker = checkers.is_both_gripper_open(
        open_gripper_threshold=0.04,
        robot_name="robots/robot",
    )

    assert left_checker(env, env_idx=0) is False
    assert left_checker(env, env_idx=1) is True
    assert both_checker(env, env_idx=0) is False
    assert both_checker(env, env_idx=1) is True


def test_within_xy_checker_uses_asset_prim_path(monkeypatch):
    checkers = importlib.import_module(
        "robo_orchard_sim.tasks.validators.checkers"
    )
    captured = {}

    def fake_is_object_center_in_obb(
        stage, prim_path, obb_pose, point, idx_env=0, axes="xy"
    ):
        captured["stage"] = stage
        captured["prim_path"] = prim_path
        captured["idx_env"] = idx_env
        captured["axes"] = axes
        return True

    utils = importlib.import_module("robo_orchard_sim.tasks.validators.utils")
    monkeypatch.setattr(
        utils, "is_object_center_in_obb", fake_is_object_center_in_obb
    )
    stage = object()
    env = _DummyEnv(
        scene=_DummyScene(
            {
                "objects/cube": _DummyObject(
                    positions=[(0.0, 0.0, 0.5), (0.1, 0.1, 0.5)],
                    default_heights=[0.0, 0.0],
                ),
                "objects/goal": _DummyObject(
                    positions=[(0.0, 0.0, 0.5), (0.1, 0.1, 0.5)],
                    default_heights=[0.0, 0.0],
                    prim_path="/World/envs/env_.*/PlaceObject",
                ),
            },
            stage=stage,
        )
    )

    checker = checkers.is_within_xy("objects/cube", "objects/goal")

    assert checker(env, env_idx=1) is True
    assert captured["stage"] is stage
    assert captured["prim_path"] == "/World/envs/env_1/PlaceObject"
    assert captured["idx_env"] == 1
    assert captured["axes"] == "xy"


def test_validator_reports_current_and_cumulative_criteria():
    state = {"met": False}

    def criterion(_env, env_idx=0):
        return state["met"]

    validator = Validator(
        actors=[
            ValidatorActor(
                name="objects/cube", uuid="", category="", actor_type=""
            )
        ],
        criteria=[criterion],
        criteria_name=["criterion"],
    )

    state["met"] = True
    first = validator.evaluate(_DummyEnv(scene={}))
    state["met"] = False
    second = validator.evaluate(_DummyEnv(scene={}))

    assert first.metrics["criteria_met_now"]["criterion"] is True
    assert first.metrics["criteria_reached"]["criterion"] is True
    assert second.metrics["criteria_met_now"]["criterion"] is False
    assert second.metrics["criteria_reached"]["criterion"] is True


def test_validator_actor_from_rigid_object_captures_cfg_and_pose():
    cube = _DummyObject(
        positions=[(0.0, 0.0, 0.5)],
        default_heights=[0.0],
    )
    cube.cfg.spawn = type(
        "SpawnCfg",
        (),
        {"semantic_tags": {}},
    )()
    cube.cfg.category = "cube"
    cube.cfg.actor_type = "rigid_object"
    cube.cfg.uuid = "cube-uuid"
    actor = ValidatorActor.from_rigid_object("objects/cube", cube)
    actor.capture_init_state(cube)
    actor.capture_final_state(cube)

    assert actor.name == "objects/cube"
    assert actor.category == "cube"
    assert actor.actor_type == "rigid_object"
    assert actor.uuid == "cube-uuid"
    assert actor.init_state is not None
    assert actor.final_state is not None


def test_lift_checker_accepts_plain_identifier():
    checkers = importlib.import_module(
        "robo_orchard_sim.tasks.validators.checkers"
    )
    checker = checkers.lift("cube", threshold=0.05)

    assert checker.actor_name == "cube"


def test_gripper_checker_accepts_plain_robot_identifier():
    checkers = importlib.import_module(
        "robo_orchard_sim.tasks.validators.checkers"
    )
    checker = checkers.is_gripper_open(
        "left",
        open_gripper_threshold=0.04,
        robot_name="robot",
    )

    assert checker.robot_name == "robot"


def test_validator_accepts_plain_actor_identifier():
    validator = Validator(
        actors=[
            ValidatorActor(name="cube", uuid="", category="", actor_type="")
        ],
        criteria=[lambda _env, env_idx=0: True],
        criteria_name=["criterion"],
    )

    assert validator.actor_names == ["cube"]
    assert [actor.name for actor in validator.actors] == ["cube"]
