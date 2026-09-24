## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

import importlib
import inspect
import types

import pytest
import torch

from robo_orchard_sim.task_components.role_registry import RoleRegistry
from robo_orchard_sim.task_components.validators.base import (
    GripperRange,
    Validator,
)
from robo_orchard_sim.task_components.validators.context import (
    ValidatorContext,
    ValidatorRobotContext,
)
from robo_orchard_sim.task_components.validators.physical_entity import (
    resolve_rigid_body_prim_path,
)
from robo_orchard_sim.task_components.validators.role_scope import RoleScope


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
        self.root_lin_vel_w = torch.zeros(
            (len(positions), 3), dtype=torch.float32
        )
        self.root_ang_vel_w = torch.zeros(
            (len(positions), 3), dtype=torch.float32
        )
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


class _DummyManipulatorProfile:
    def __init__(
        self,
        ee_body_name,
        gripper_joint_names=(),
        gripper_body_names=(),
    ):
        self.ee_body_name = ee_body_name
        self.gripper_joint_names = tuple(gripper_joint_names)
        self.gripper_body_names = tuple(gripper_body_names)


class _DummyRobotInfo:
    def __init__(
        self, manipulator_profile, gripper_open_val, gripper_close_val
    ):
        self.t_standard_tcp_to_robot_ee = torch.eye(4).tolist()
        self.manipulator_profile = manipulator_profile
        self.gripper_open_val = list(gripper_open_val)
        self.gripper_close_val = list(gripper_close_val)


class _DummyEmbodiment:
    scene_name = "robots/dualarm_piperx"

    def get_robot_info_cfgs(self):
        return {
            "left_arm": _DummyRobotInfo(
                _DummyManipulatorProfile(
                    ee_body_name="left_link6",
                    gripper_joint_names=("left_joint7", "left_joint8"),
                ),
                gripper_open_val=[0.05, -0.05],
                gripper_close_val=[0.0, 0.0],
            ),
            "right_arm": _DummyRobotInfo(
                _DummyManipulatorProfile(
                    ee_body_name="right_link6",
                    gripper_joint_names=("right_joint7", "right_joint8"),
                ),
                gripper_open_val=[0.05, -0.05],
                gripper_close_val=[0.0, 0.0],
            ),
        }


def test_checkers_can_be_imported_without_pxr():
    module = importlib.import_module(
        "robo_orchard_sim.task_components.validators.checkers"
    )
    assert hasattr(module, "LiftChecker")


class _FakeRigidAsset:
    def __init__(self, configured, prim_paths=None):
        self.cfg = types.SimpleNamespace(prim_path=configured)
        if prim_paths is not None:
            self.root_physx_view = types.SimpleNamespace(prim_paths=prim_paths)


def test_rigid_body_prim_path_prefers_physx_owning_prim():
    asset = _FakeRigidAsset(
        "/World/envs/env_.*/pick_object",
        [
            "/World/envs/env_0/pick_object/geometry/mesh",
            "/World/envs/env_1/pick_object/geometry/mesh",
        ],
    )

    assert (
        resolve_rigid_body_prim_path(asset, 1)
        == "/World/envs/env_1/pick_object/geometry/mesh"
    )


def test_rigid_body_prim_path_falls_back_to_configured_path():
    asset = _FakeRigidAsset("/World/envs/env_.*/pick_object")

    assert (
        resolve_rigid_body_prim_path(asset, 0)
        == "/World/envs/env_0/pick_object"
    )


def test_from_embodiment_uses_runtime_embodiment_robot_metadata():
    context = ValidatorContext.from_embodiment(
        _DummyEmbodiment(), RoleRegistry()
    )

    assert context.robot is not None
    assert context.robot.robot_name == "robots/dualarm_piperx"
    assert context.robot.ee_links == ("left_link6", "right_link6")
    assert [spec.name for spec in context.robot.gripper_joints] == [
        "left_joint7",
        "left_joint8",
        "right_joint7",
        "right_joint8",
    ]


def test_validator_forwards_env_idx_to_criteria():
    seen_env_indices = []

    def criterion(_env, env_idx=0):
        seen_env_indices.append(env_idx)
        return env_idx == 1

    validator = Validator(
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
        criteria=[_Criterion()],
        criteria_name=["criterion"],
    )

    result = validator.evaluate(_DummyEnv(scene={}), env_idx=1)

    assert result.success is True
    assert result.progress == 1.0


def test_lift_checker_reads_requested_env_index():
    checkers = importlib.import_module(
        "robo_orchard_sim.task_components.validators.checkers"
    )
    initial_env = _DummyEnv(
        scene={
            "objects/cube": _DummyObject(
                positions=[(0.0, 0.0, 0.50), (0.0, 0.0, 0.50)],
                default_heights=[0.50, 0.50],
            )
        }
    )
    context = ValidatorContext(robot=None, role_registry=RoleRegistry())
    context.capture_init_states(initial_env, ["objects/cube"])
    env = _DummyEnv(
        scene={
            "objects/cube": _DummyObject(
                positions=[(0.0, 0.0, 0.50), (0.0, 0.0, 0.65)],
                default_heights=[0.50, 0.50],
            )
        }
    )

    checker = checkers.LiftChecker(threshold=0.05)
    scope = RoleScope.from_entities(("objects/cube",))

    assert not checker.evaluate(env, context, scope, env_idx=0).unary[
        "lifted"
    ]["objects/cube"]
    assert checker.evaluate(env, context, scope, env_idx=1).unary["lifted"][
        "objects/cube"
    ]


def test_lift_checker_uses_per_env_init_height():
    checkers = importlib.import_module(
        "robo_orchard_sim.task_components.validators.checkers"
    )
    initial_env = _DummyEnv(
        scene={
            "objects/cube": _DummyObject(
                positions=[(0.0, 0.0, 0.50), (0.0, 0.0, 0.60)],
                default_heights=[0.10, 0.10],
            )
        }
    )
    context = ValidatorContext(robot=None, role_registry=RoleRegistry())
    context.capture_init_states(initial_env, ["objects/cube"])
    env = _DummyEnv(
        scene={
            "objects/cube": _DummyObject(
                positions=[(0.0, 0.0, 0.56), (0.0, 0.0, 0.65)],
                default_heights=[0.10, 0.10],
            )
        }
    )

    checker = checkers.LiftChecker(threshold=0.05)
    scope = RoleScope.from_entities(("objects/cube",))

    assert (
        checker.evaluate(env, context, scope, env_idx=0).unary["lifted"][
            "objects/cube"
        ]
        is True
    )
    assert (
        checker.evaluate(env, context, scope, env_idx=1).unary["lifted"][
            "objects/cube"
        ]
        is False
    )


def test_lift_checker_missing_init_state_raises_runtime_error():
    checkers = importlib.import_module(
        "robo_orchard_sim.task_components.validators.checkers"
    )
    env = _DummyEnv(
        scene={
            "objects/cube": _DummyObject(
                positions=[(0.0, 0.0, 0.56)],
                default_heights=[0.50],
            )
        }
    )

    context = ValidatorContext(robot=None, role_registry=RoleRegistry())
    checker = checkers.LiftChecker(threshold=0.05)

    with pytest.raises(RuntimeError, match="not captured"):
        checker.evaluate(
            env,
            context,
            RoleScope.from_entities(("objects/cube",)),
            env_idx=0,
        )


def test_reach_checker_multiple_envs_uses_requested_index():
    checkers = importlib.import_module(
        "robo_orchard_sim.task_components.validators.checkers"
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

    robot_data = env.scene["robots/robot"].data
    robot_data.body_pos_w = robot_data.body_com_pos_w.clone()
    robot_data.body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2] * 2)
    checker = checkers.ReachChecker(threshold=0.05)
    context = ValidatorContext(
        robot=ValidatorRobotContext(
            robot_name="robots/robot",
            ee_links=("left_link6",),
            tcp_offsets=(("left_link6", (0, 0, 0)),),
        ),
        role_registry=RoleRegistry(),
    )
    scope = RoleScope.from_entities(("objects/cube",))

    assert not checker.evaluate(env, context, scope, env_idx=0).unary[
        "reached"
    ]["objects/cube"]
    assert checker.evaluate(env, context, scope, env_idx=1).unary["reached"][
        "objects/cube"
    ]


def test_alignment_xy_checker_reads_requested_env_index():
    checkers = importlib.import_module(
        "robo_orchard_sim.task_components.validators.checkers"
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

    checker = checkers.AlignmentXYChecker(
        "pick",
        "place",
        eps=(0.02, 0.02),
    )
    context = ValidatorContext(robot=None, role_registry=RoleRegistry())
    scope = RoleScope.from_role_members(
        {
            "pick": ("objects/cube",),
            "place": ("objects/goal",),
        }
    )

    assert not checker.evaluate(env, context, scope, env_idx=0).binary[
        "alignment_xy"
    ][("objects/cube", "objects/goal")]
    assert checker.evaluate(env, context, scope, env_idx=1).binary[
        "alignment_xy"
    ][("objects/cube", "objects/goal")]


def test_alignment_xyz_checker_reads_requested_env_index():
    checkers = importlib.import_module(
        "robo_orchard_sim.task_components.validators.checkers"
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

    checker = checkers.AlignmentXYZChecker(
        "pick",
        "place",
        eps=(0.02, 0.02, 0.02),
        target_height_offset=0.04,
    )
    context = ValidatorContext(robot=None, role_registry=RoleRegistry())
    scope = RoleScope.from_role_members(
        {
            "pick": ("objects/base",),
            "place": ("objects/stack",),
        }
    )

    assert not checker.evaluate(env, context, scope, env_idx=0).binary[
        "alignment_xyz"
    ][("objects/base", "objects/stack")]
    assert checker.evaluate(env, context, scope, env_idx=1).binary[
        "alignment_xyz"
    ][("objects/base", "objects/stack")]


def test_axis_align_checker_multiple_entities_returns_complete_mapping():
    checkers = importlib.import_module(
        "robo_orchard_sim.task_components.validators.checkers"
    )
    upright = _DummyObject([(0.0, 0.0, 0.0)], [0.0])
    tipped = _DummyObject([(0.0, 0.0, 0.0)], [0.0])
    tipped.data.root_quat_w[0] = torch.tensor((0.7071068, 0.7071068, 0.0, 0.0))
    env = _DummyEnv(
        scene={
            "objects/upright": upright,
            "objects/tipped": tipped,
        }
    )
    scope = RoleScope.from_entities(("objects/upright", "objects/tipped"))

    metrics = checkers.AxisAlignChecker(
        local_axis=(0.0, 0.0, 1.0),
        world_target=(0.0, 0.0, 1.0),
        max_deg=15.0,
    ).evaluate(
        env,
        ValidatorContext(robot=None, role_registry=RoleRegistry()),
        scope,
    )

    assert metrics.unary["axis_aligned"] == {
        "objects/upright": True,
        "objects/tipped": False,
    }


def test_stationary_checker_multiple_entities_returns_complete_mapping():
    checkers = importlib.import_module(
        "robo_orchard_sim.task_components.validators.checkers"
    )
    stationary = _DummyObject([(0.0, 0.0, 0.0)], [0.0])
    moving = _DummyObject([(0.0, 0.0, 0.0)], [0.0])
    moving.data.root_lin_vel_w[0, 0] = 0.5
    env = _DummyEnv(
        scene={
            "objects/stationary": stationary,
            "objects/moving": moving,
        }
    )
    scope = RoleScope.from_entities(("objects/stationary", "objects/moving"))

    metrics = checkers.StationaryChecker(
        linear_threshold=0.01,
    ).evaluate(
        env,
        ValidatorContext(robot=None, role_registry=RoleRegistry()),
        scope,
    )

    assert metrics.unary["stationary"] == {
        "objects/stationary": True,
        "objects/moving": False,
    }


def test_gripper_checkers_read_requested_env_index():
    checkers = importlib.import_module(
        "robo_orchard_sim.task_components.validators.checkers"
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

    left_spec = GripperRange(name="left_joint7", open_val=0.05, close_val=0.0)
    right_spec = GripperRange(
        name="right_joint7", open_val=0.05, close_val=0.0
    )
    context = ValidatorContext(
        robot=ValidatorRobotContext(
            robot_name="robots/robot",
            gripper_joints=(left_spec, right_spec),
        ),
        role_registry=RoleRegistry(),
    )
    scope = RoleScope.from_entities(())
    left_checker = checkers.GripperOpenChecker("left_joint7")
    both_checker = checkers.BothGripperOpenChecker()

    assert (
        left_checker.evaluate(env, context, scope, env_idx=0).global_values[
            "gripper_open:left_joint7"
        ]
        is False
    )
    assert (
        left_checker.evaluate(env, context, scope, env_idx=1).global_values[
            "gripper_open:left_joint7"
        ]
        is True
    )
    assert (
        both_checker.evaluate(env, context, scope, env_idx=0).global_values[
            "gripper_open"
        ]
        is False
    )
    assert (
        both_checker.evaluate(env, context, scope, env_idx=1).global_values[
            "gripper_open"
        ]
        is True
    )


def test_within_xy_checker_uses_asset_prim_path(monkeypatch):
    checkers = importlib.import_module(
        "robo_orchard_sim.task_components.validators.checkers"
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

    utils = importlib.import_module(
        "robo_orchard_sim.task_components.validators.utils"
    )
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

    checker = checkers.WithinXYChecker(
        "pick",
        "place",
    )
    context = ValidatorContext(robot=None, role_registry=RoleRegistry())
    scope = RoleScope.from_role_members(
        {
            "pick": ("objects/cube",),
            "place": ("objects/goal",),
        }
    )

    assert (
        checker.evaluate(env, context, scope, env_idx=1).binary["within_xy"][
            ("objects/cube", "objects/goal")
        ]
        is True
    )
    assert captured["stage"] is stage
    assert captured["prim_path"] == "/World/envs/env_1/PlaceObject"
    assert captured["idx_env"] == 1
    assert captured["axes"] == "xy"


def test_validator_reports_current_and_cumulative_criteria():
    state = {"met": False}

    def criterion(_env, env_idx=0):
        return state["met"]

    validator = Validator(
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
