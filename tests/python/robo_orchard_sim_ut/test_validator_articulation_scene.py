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

"""Behavioral tests for scene-wide articulation facts and role selection."""

from types import SimpleNamespace

import pytest
import torch

from robo_orchard_sim.asset_manager.metadata import (
    ArticulationOperationMeta,
    JointOperationMeta,
)
from robo_orchard_sim.task_components.role_registry import (
    RoleRegistry,
    TargetRef,
)
from robo_orchard_sim.task_components.validators.checkers import (
    AlignmentXYChecker,
    ContactChecker,
    JointStateChecker,
    ReachChecker,
    StationaryChecker,
)
from robo_orchard_sim.task_components.validators.context import (
    ValidatorContext,
    ValidatorRobotContext,
)
from robo_orchard_sim.task_components.validators.metrics import (
    AllMetricSelector,
    AnyMetricSelector,
    BoundRoleInteractionBodyMetricSelector,
    BoundRoleJointDeltaSelector,
    BoundRoleJointFractionSelector,
    BoundRoleJointMovedSelector,
    BoundRoleOutcomeBodyMetricSelector,
    BoundRolePairMetricSelector,
    DwellMetricSelector,
    MetricStore,
    SceneMetrics,
)
from robo_orchard_sim.task_components.validators.physical_entity import (
    SceneBodyKey,
    SceneJointKey,
)
from robo_orchard_sim.task_components.validators.role_scope import RoleScope


class _FakeArticulation:
    def __init__(self, *, target_joint_delta: float | None = None) -> None:
        open_target = (
            {"target_joint_fraction": 0.0}
            if target_joint_delta is None
            else {"target_joint_delta": target_joint_delta}
        )
        joint_limits = (
            [-1.75, -0.576]
            if target_joint_delta is None
            else [-float("inf"), float("inf")]
        )
        self.joint_names = ["hinge_joint"]
        self.body_names = ["base", "button_link", "lid_link"]
        self.cfg = SimpleNamespace(
            prim_path="/World/envs/env_.*/Laptop",
            joint_operations=(
                JointOperationMeta(
                    joint_name="hinge_joint",
                    semantic_name="lid",
                    outcome_link="lid_link",
                    operations={
                        "open": ArticulationOperationMeta(
                            interaction_link="button_link",
                            initial_joint_position=-0.576,
                            **open_target,
                        ),
                        "close": ArticulationOperationMeta(
                            interaction_link="lid_link",
                            initial_joint_position=-1.75,
                            target_joint_fraction=1.0,
                        ),
                    },
                ),
            ),
        )
        self.data = SimpleNamespace(
            joint_pos=torch.tensor([[-0.576]], dtype=torch.float32),
            joint_pos_limits=torch.tensor(
                [[joint_limits]],
                dtype=torch.float32,
            ),
            body_com_pos_w=torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.10, 0.0, 0.0],
                        [0.50, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            ),
            body_pos_w=torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.10, 0.0, 0.0],
                        [0.50, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            ),
            body_quat_w=torch.tensor(
                [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            ),
            body_lin_vel_w=torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            ),
            body_ang_vel_w=torch.zeros((1, 3, 3), dtype=torch.float32),
            root_pos_w=torch.zeros((1, 3), dtype=torch.float32),
            root_quat_w=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
            root_lin_vel_w=torch.zeros((1, 3), dtype=torch.float32),
            root_ang_vel_w=torch.zeros((1, 3), dtype=torch.float32),
            root_state_w=torch.tensor(
                [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
        )
        self.root_physx_view = SimpleNamespace(
            shared_metatype=SimpleNamespace(dof_types=[0]),
            link_paths=[
                [
                    "/World/envs/env_0/Laptop/base",
                    "/World/envs/env_0/Laptop/button_link",
                    "/World/envs/env_0/Laptop/lid_link",
                ]
            ],
        )

    def find_joints(self, name: str) -> tuple[list[int], list[str]]:
        matches = [
            (index, joint_name)
            for index, joint_name in enumerate(self.joint_names)
            if joint_name == name
        ]
        return (
            [index for index, _ in matches],
            [joint_name for _, joint_name in matches],
        )

    def find_bodies(self, name: str) -> tuple[list[int], list[str]]:
        matches = [
            (index, body_name)
            for index, body_name in enumerate(self.body_names)
            if body_name == name
        ]
        return (
            [index for index, _ in matches],
            [body_name for _, body_name in matches],
        )


class _FakeRobot:
    def __init__(self) -> None:
        self.body_names = ["ee", "left_finger", "right_finger"]
        self.data = SimpleNamespace(
            body_com_pos_w=torch.tensor(
                [[[0.12, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
            body_pos_w=torch.tensor(
                [[[0.12, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
        )

        self.data.body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 3])

    def find_bodies(self, name: str) -> tuple[list[int], list[str]]:
        matches = [
            (index, body_name)
            for index, body_name in enumerate(self.body_names)
            if body_name == name
        ]
        return (
            [index for index, _ in matches],
            [body_name for _, body_name in matches],
        )


class _FakeContactSensor:
    def __init__(
        self,
        force: tuple[float, float, float],
        position: tuple[float, float, float] = (-0.02, 0.0, 0.0),
    ) -> None:
        self.force = force
        self.position = position
        self.cfg = SimpleNamespace(filter_prim_paths_expr=[])
        self.contact_physx_view = SimpleNamespace(filter_paths=[[]])
        self.data = SimpleNamespace(
            force_matrix_w=torch.tensor(
                [[[[*force]]]],
                dtype=torch.float32,
            )
        )

    def _initialize_impl(self) -> None:
        self.contact_physx_view.filter_paths = [
            list(self.cfg.filter_prim_paths_expr)
        ]
        self.data.contact_pos_w = torch.tensor(
            [[[list(self.position) for _ in self.cfg.filter_prim_paths_expr]]],
            dtype=torch.float32,
        )
        self.data.force_matrix_w = torch.tensor(
            [[[list(self.force) for _ in self.cfg.filter_prim_paths_expr]]],
            dtype=torch.float32,
        )


def _runtime(
    *,
    target_joint_delta: float | None = None,
) -> tuple[
    SimpleNamespace,
    ValidatorContext,
    RoleScope,
    MetricStore,
]:
    registry = RoleRegistry()
    registry.bind_one(
        0,
        "primary",
        TargetRef("objects/laptop", "lid", "open"),
    )
    context = ValidatorContext(
        robot=ValidatorRobotContext(
            robot_name="robots/panda",
            ee_links=("ee",),
            tcp_offsets=(("ee", (0, 0, 0)),),
            gripper_body_groups=(("left_finger", "right_finger"),),
        ),
        role_registry=registry,
    )
    scope = RoleScope.from_role_members({"primary": ("objects/laptop",)})
    env = SimpleNamespace(
        num_envs=1,
        scene={
            "objects/laptop": _FakeArticulation(
                target_joint_delta=target_joint_delta
            ),
            "robots/panda": _FakeRobot(),
            "sensors/gripper_contact_0_0": _FakeContactSensor(
                (-1.0, 0.0, 0.0)
            ),
            "sensors/gripper_contact_0_1": _FakeContactSensor(
                (1.0, 0.0, 0.0), (0.02, 0.0, 0.0)
            ),
        },
    )
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    for name in ["base", "button_link", "lid_link"]:
        path = "/World/envs/env_0/Laptop/" + name
        body = UsdGeom.Xform.Define(stage, path)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        cube = UsdGeom.Cube.Define(stage, path + "/collision")
        cube.CreateSizeAttr(0.002)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    env.sim = SimpleNamespace(stage=stage)
    return env, context, scope, MetricStore(scope)


def test_reach_checker_articulation_links_returns_operation_body_facts() -> (
    None
):
    env, context, scope, _ = _runtime()

    metrics = ReachChecker(threshold=0.05).evaluate(
        env,
        context,
        scope,
    )

    assert metrics.unary["reached"] == {
        "objects/laptop": False,
        SceneBodyKey("objects/laptop", "lid_link"): False,
        SceneBodyKey("objects/laptop", "button_link"): True,
    }


def test_joint_state_checker_runtime_returns_expected_measurements() -> None:
    env, context, scope, _ = _runtime()

    metrics = JointStateChecker().evaluate(env, context, scope)

    key = SceneJointKey("objects/laptop", "hinge_joint")
    assert metrics.values["joint_position"][key] == pytest.approx(-0.576)
    assert metrics.values["joint_fraction"][key] == 1.0


def test_joint_state_checker_unbounded_joint_omits_fraction() -> None:
    env, context, scope, _ = _runtime(target_joint_delta=0.5)

    metrics = JointStateChecker().evaluate(env, context, scope)

    key = SceneJointKey("objects/laptop", "hinge_joint")
    assert metrics.values["joint_position"][key] == pytest.approx(-0.576)
    assert key not in metrics.values["joint_fraction"]


def test_stationary_checker_articulation_links_returns_body_facts() -> None:
    env, context, scope, _ = _runtime()

    metrics = StationaryChecker().evaluate(
        env,
        context,
        scope,
    )

    assert metrics.unary["stationary"] == {
        "objects/laptop": True,
        SceneBodyKey("objects/laptop", "lid_link"): True,
        SceneBodyKey("objects/laptop", "button_link"): False,
    }


@pytest.mark.parametrize(
    ("first_force", "second_force", "expected"),
    [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (False, False)),
        ((-0.005, 0.0, 0.0), (0.0, 0.0, 0.0), (False, False)),
        ((-0.047, 0.0, 0.0), (0.0, 0.0, 0.0), (True, False)),
        ((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (True, False)),
        ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (True, True)),
        ((-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (True, False)),
    ],
)
def test_contact_checker_articulation_link_forces_return_expected_facts(
    first_force: tuple[float, float, float],
    second_force: tuple[float, float, float],
    expected: tuple[bool, bool],
) -> None:
    env, context, scope, _ = _runtime()
    env.scene["sensors/gripper_contact_0_0"] = _FakeContactSensor(first_force)
    env.scene["sensors/gripper_contact_0_1"] = _FakeContactSensor(
        second_force, (0.02, 0.0, 0.0)
    )

    metrics = ContactChecker().evaluate(env, context, scope)

    target = SceneBodyKey("objects/laptop", "lid_link")
    assert (
        metrics.unary["contacted"][target],
        metrics.unary["grasped"][target],
    ) == expected


@pytest.mark.parametrize("origin_x", [(1.0, -1.0), (0.0, 0.0)])
def test_contact_checker_displaced_link_origins_preserves_grasp(
    origin_x: tuple[float, float],
) -> None:
    env, context, scope, _ = _runtime()
    env.scene["robots/panda"].data.body_pos_w[0, 1:, 0] = torch.tensor(
        origin_x
    )

    metrics = ContactChecker().evaluate(env, context, scope)

    assert metrics.unary["grasped"][SceneBodyKey("objects/laptop", "lid_link")]


@pytest.mark.parametrize(
    "position",
    [
        (-0.02, 0.0, 0.0),
        (-0.04, 0.0, 0.0),
        (float("nan"), 0.0, 0.0),
        (float("inf"), 0.0, 0.0),
        (-0.02, 0.04, 0.0),
    ],
    ids=["coincident", "outward", "missing", "infinite", "tangential"],
)
def test_contact_checker_invalid_contact_axis_returns_not_grasped(
    position: tuple[float, float, float],
) -> None:
    env, context, scope, _ = _runtime()
    env.scene["sensors/gripper_contact_0_1"] = _FakeContactSensor(
        (1.0, 0.0, 0.0), position
    )

    metrics = ContactChecker().evaluate(env, context, scope)

    target = SceneBodyKey("objects/laptop", "lid_link")
    assert (
        metrics.unary["contacted"][target],
        metrics.unary["grasped"][target],
    ) == (True, False)


def test_contact_checker_distinct_contacts_returns_per_body_grasps() -> None:
    env, context, scope, _ = _runtime()
    checker = ContactChecker()
    checker.evaluate(env, context, scope)
    sensor = env.scene["sensors/gripper_contact_0_1"]
    column = sensor.contact_physx_view.filter_paths[0].index(
        "/World/envs/env_0/Laptop/lid_link"
    )
    sensor.data.contact_pos_w[0, 0, column] = torch.tensor([-0.04, 0.0, 0.0])

    metrics = checker.evaluate(env, context, scope)

    assert metrics.unary["grasped"] == {
        SceneBodyKey("objects/laptop", "button_link"): True,
        SceneBodyKey("objects/laptop", "lid_link"): False,
    }


def test_bound_fraction_selector_rebound_operation_selects_new_goal() -> None:
    env, context, scope, store = _runtime()
    store.update(JointStateChecker().evaluate(env, context, scope))
    selector = BoundRoleJointFractionSelector(
        metric_store=store,
        context=context,
        role_id="primary",
        tolerance=0.01,
    )

    open_result = selector(env)
    context.role_registry.bind_one(
        0,
        "primary",
        TargetRef("objects/laptop", "lid", "close"),
    )

    assert (open_result, selector(env)) == (False, True)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (-0.2, False),
        (0.0, True),
        (-1.2, True),
    ],
)
def test_bound_delta_selector_joint_displacement_returns_expected_result(
    position: float,
    expected: bool,
) -> None:
    env, context, scope, store = _runtime(target_joint_delta=0.5)
    env.scene["objects/laptop"].data.joint_pos[0, 0] = position
    store.update(JointStateChecker().evaluate(env, context, scope))
    selector = BoundRoleJointDeltaSelector(
        metric_store=store,
        context=context,
        role_id="primary",
    )

    assert selector(env) is expected


@pytest.mark.parametrize(
    ("target_joint_delta", "position", "expected"),
    [
        (None, -1.75, True),
        (0.5, 0.0, True),
        (0.5, -0.2, False),
    ],
)
def test_any_joint_target_selector_target_mode_returns_expected_result(
    target_joint_delta: float | None,
    position: float,
    expected: bool,
) -> None:
    env, context, scope, store = _runtime(
        target_joint_delta=target_joint_delta
    )
    env.scene["objects/laptop"].data.joint_pos[0, 0] = position
    store.update(JointStateChecker().evaluate(env, context, scope))
    fraction_selector = BoundRoleJointFractionSelector(
        metric_store=store,
        context=context,
        role_id="primary",
        tolerance=0.01,
    )
    delta_selector = BoundRoleJointDeltaSelector(
        metric_store=store,
        context=context,
        role_id="primary",
    )
    selector = AnyMetricSelector((fraction_selector, delta_selector))

    assert selector(env) is expected


def test_bound_moved_selector_rebound_operation_uses_new_start() -> None:
    env, context, scope, store = _runtime()
    articulation = env.scene["objects/laptop"]
    articulation.data.joint_pos[0, 0] = -0.60
    store.update(JointStateChecker().evaluate(env, context, scope))
    selector = BoundRoleJointMovedSelector(
        metric_store=store,
        context=context,
        role_id="primary",
        min_delta=0.1,
    )

    open_result = selector(env)
    context.role_registry.bind_one(
        0,
        "primary",
        TargetRef("objects/laptop", "lid", "close"),
    )

    assert (open_result, selector(env)) == (False, True)


def test_bound_interaction_body_selector_rebound_operation_selects_link() -> (
    None
):
    env, context, scope, store = _runtime()
    store.update(ReachChecker(threshold=0.05).evaluate(env, context, scope))
    selector = BoundRoleInteractionBodyMetricSelector(
        metric_store=store,
        context=context,
        role_id="primary",
        metric="reached",
    )

    open_result = selector(env)
    context.role_registry.bind_one(
        0,
        "primary",
        TargetRef("objects/laptop", "lid", "close"),
    )

    assert (open_result, selector(env)) == (True, False)


def test_bound_outcome_body_selector_operations_select_same_link() -> None:
    env, context, scope, store = _runtime()
    store.update(StationaryChecker().evaluate(env, context, scope))
    selector = BoundRoleOutcomeBodyMetricSelector(
        metric_store=store,
        context=context,
        role_id="primary",
        metric="stationary",
    )

    open_result = selector(env)
    context.role_registry.bind_one(
        0,
        "primary",
        TargetRef("objects/laptop", "lid", "close"),
    )

    assert (open_result, selector(env)) == (True, True)


def test_bound_grasp_selector_rebound_operation_selects_interaction_link() -> (
    None
):
    env, context, scope, store = _runtime()
    store.update(
        SceneMetrics(
            unary={
                "grasped": {
                    SceneBodyKey("objects/laptop", "button_link"): True,
                    SceneBodyKey("objects/laptop", "lid_link"): False,
                }
            }
        )
    )
    selector = BoundRoleInteractionBodyMetricSelector(
        metric_store=store,
        context=context,
        role_id="primary",
        metric="grasped",
    )

    open_result = selector(env)
    context.role_registry.bind_one(
        0,
        "primary",
        TargetRef("objects/laptop", "lid", "close"),
    )

    assert (open_result, selector(env)) == (True, False)


def test_alignment_xy_checker_articulation_link_relation_returns_true() -> (
    None
):
    env, context, _, _ = _runtime()
    env.scene["objects/goal"] = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.11, 0.0, 0.0]]),
        )
    )
    link = SceneBodyKey("objects/laptop", "button_link")
    scope = RoleScope.from_role_members(
        {
            "subject": (link,),
            "object": ("objects/goal",),
        }
    )

    metrics = AlignmentXYChecker(
        "subject",
        "object",
        eps=(0.02, 0.02),
    ).evaluate(env, context, scope)

    assert metrics.binary["alignment_xy"][(link, "objects/goal")] is True


def test_bound_pair_selector_articulation_binding_selects_link_relation() -> (
    None
):
    env, context, _, _ = _runtime()
    env.scene["objects/goal"] = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.11, 0.0, 0.0]]),
        )
    )
    context.role_registry.bind_one(0, "goal", TargetRef("objects/goal"))
    link = SceneBodyKey("objects/laptop", "button_link")
    scope = RoleScope.from_role_members(
        {
            "primary": (link,),
            "goal": ("objects/goal",),
        }
    )
    store = MetricStore(scope)
    store.update(
        AlignmentXYChecker(
            "primary",
            "goal",
            eps=(0.02, 0.02),
        ).evaluate(env, context, scope)
    )
    selector = BoundRolePairMetricSelector(
        metric_store=store,
        context=context,
        subject_role="primary",
        object_role="goal",
        metric="alignment_xy",
    )

    assert selector(env) is True


def test_dwell_selector_interrupted_sequence_resets_streak() -> None:
    results = iter((True, True, False, True, True, True))

    def current_value(env, env_idx: int = 0) -> bool:
        del env, env_idx
        return next(results)

    selector = DwellMetricSelector(current_value, n_steps=3)

    observed = [selector(None) for _ in range(6)]

    assert observed == [False, False, False, False, False, True]


@pytest.mark.parametrize(
    ("initial_fraction", "fraction", "expected"),
    [
        (0.0, 0.25, False),
        (0.0, 0.375, True),
        (0.0, 0.875, True),
        (1.0, 0.75, False),
        (1.0, 0.625, True),
        (1.0, 0.125, True),
        (0.0, float("nan"), False),
        (0.0, float("inf"), False),
        (0.5, 0.5, True),
        (0.5, 0.875, False),
    ],
)
def test_bound_fraction_selector_directional_target_returns_expected_result(
    initial_fraction: float,
    fraction: float,
    expected: bool,
) -> None:
    env, context, scope, store = _runtime()
    articulation = env.scene["objects/laptop"]
    articulation.data.joint_pos_limits[0, 0] = torch.tensor([-3.0, 1.0])
    articulation.data.joint_pos[0, 0] = -3.0 + 4.0 * fraction
    operation = articulation.cfg.joint_operations[0].operations["open"]
    operation.initial_joint_position = -3.0 + 4.0 * initial_fraction
    operation.target_joint_fraction = 0.5
    store.update(JointStateChecker().evaluate(env, context, scope))
    selector = BoundRoleJointFractionSelector(
        metric_store=store,
        context=context,
        role_id="primary",
        tolerance=0.125,
    )

    assert selector(env) is expected


def test_fraction_selector_overshoot_preserves_settling_dwell() -> None:
    env, context, scope, store = _runtime()
    articulation = env.scene["objects/laptop"]
    articulation.data.joint_pos[0, 0] = -1.6
    articulation.cfg.joint_operations[0].operations[
        "open"
    ].target_joint_fraction = 0.5
    target = BoundRoleJointFractionSelector(
        metric_store=store,
        context=context,
        role_id="primary",
        tolerance=0.05,
    )
    stationary = BoundRoleOutcomeBodyMetricSelector(
        metric_store=store,
        context=context,
        role_id="primary",
        metric="stationary",
    )
    criterion = DwellMetricSelector(AllMetricSelector((target, stationary)), 5)
    results = []
    for is_stationary in [True] * 4 + [False] + [True] * 5:
        articulation.data.body_lin_vel_w[0, 2, 0] = (
            0.0 if is_stationary else 1.0
        )
        facts = JointStateChecker().evaluate(env, context, scope)
        settled = StationaryChecker().evaluate(env, context, scope)
        store.update(SceneMetrics(values=facts.values, unary=settled.unary))
        results.append(criterion(env))

    assert results == [False] * 9 + [True]


@pytest.mark.parametrize("metric", ["reached", "contacted", "grasped"])
@pytest.mark.parametrize(
    ("active_body", "expected"),
    [
        ("lid_link", True),
        ("button_link", True),
        ("base", False),
        (None, False),
    ],
)
def test_bound_interaction_selector_alternative_links_selects_allowed_body(
    metric, active_body, expected
):
    env, context, _, store = _runtime()
    operation = env.scene["objects/laptop"].cfg.joint_operations[0]
    operation.operations["close"] = ArticulationOperationMeta(
        interaction_links=("lid_link", "button_link"),
        initial_joint_position=-1.75,
        target_joint_fraction=1.0,
    )
    context.role_registry.bind_one(
        0, "primary", TargetRef("objects/laptop", "lid", "close")
    )
    store.update(
        SceneMetrics(
            unary={
                metric: {
                    SceneBodyKey("objects/laptop", body): body == active_body
                    for body in ("base", "button_link", "lid_link")
                }
            }
        )
    )
    selector = BoundRoleInteractionBodyMetricSelector(
        metric_store=store, context=context, role_id="primary", metric=metric
    )
    assert selector(env) is expected


@pytest.mark.parametrize("active_body", ["button_link", "lid_link"])
def test_bound_contact_selector_legacy_metadata_uses_interaction_link(
    active_body,
):
    env, context, _, store = _runtime()
    store.update(
        SceneMetrics(
            unary={
                "contacted": {
                    SceneBodyKey("objects/laptop", active_body): True,
                }
            }
        )
    )
    selector = BoundRoleInteractionBodyMetricSelector(
        metric_store=store,
        context=context,
        role_id="primary",
        metric="contacted",
    )
    assert selector(env) is (active_body == "button_link")


def test_reach_checker_additional_interaction_body_included_in_observations():
    env, context, scope, _ = _runtime()
    operation = env.scene["objects/laptop"].cfg.joint_operations[0]
    operation.operations["close"] = ArticulationOperationMeta(
        interaction_links=("base",),
        initial_joint_position=-1.75,
        target_joint_fraction=1.0,
    )
    metrics = ReachChecker(threshold=0.2).evaluate(env, context, scope)
    assert SceneBodyKey("objects/laptop", "base") in metrics.unary["reached"]


@pytest.mark.parametrize("links", [[], [""], [" "], ["lid", "lid"]])
def test_operation_metadata_invalid_interaction_links_raises_value_error(
    links,
):
    with pytest.raises(ValueError, match="interaction_links"):
        ArticulationOperationMeta(
            interaction_links=links,
            initial_joint_position=0.0,
            target_joint_fraction=0.0,
        )


@pytest.mark.parametrize(
    "fields",
    [{}, {"interaction_link": "lid", "interaction_links": ["handle"]}],
)
def test_operation_metadata_ambiguous_link_configuration_raises_value_error(
    fields,
):
    with pytest.raises(ValueError, match="exactly one of interaction_link"):
        ArticulationOperationMeta(
            initial_joint_position=0, target_joint_fraction=0, **fields
        )


def test_reach_dwell_alternating_allowed_bodies_counts_consecutive_steps():
    env, context, _, store = _runtime()
    env.scene["objects/laptop"].cfg.joint_operations[0].operations["open"] = (
        ArticulationOperationMeta(
            interaction_links=("lid_link", "button_link"),
            initial_joint_position=0,
            target_joint_fraction=0,
        )
    )
    selector = DwellMetricSelector(
        BoundRoleInteractionBodyMetricSelector(
            metric_store=store,
            context=context,
            role_id="primary",
            metric="reached",
        ),
        3,
    )
    results = []
    for body in ["lid_link", "button_link", "lid_link"]:
        store.update(
            SceneMetrics(
                unary={"reached": {SceneBodyKey("objects/laptop", body): True}}
            )
        )
        results.append(selector(env))
    assert results == [False, False, True]


def test_operation_metadata_multiple_links_round_trip_preserves_candidates():
    operation = ArticulationOperationMeta(
        interaction_links=["lid", "handle"],
        initial_joint_position=0,
        target_joint_fraction=0,
    )
    restored = ArticulationOperationMeta.model_validate_json(
        operation.model_dump_json()
    )
    assert restored.effective_interaction_links == ("lid", "handle")


@pytest.mark.parametrize(
    ("origin", "com", "expected"),
    [([0.1, 0, -0.12], [4, 4, 4], True), ([4, 4, 4], [0.1, 0, 0], False)],
)
def test_reach_checker_tcp_offset_uses_link_pose_not_com(
    origin, com, expected
):
    from dataclasses import replace

    env, context, scope, _ = _runtime()
    context.robot = replace(context.robot, tcp_offsets=(("ee", (0, 0, 0.12)),))
    data = env.scene["robots/panda"].data
    data.body_pos_w[0, 0] = torch.tensor(origin)
    data.body_com_pos_w[0, 0] = torch.tensor(com)
    metrics = ReachChecker(threshold=0.01).evaluate(env, context, scope)
    assert (
        metrics.unary["reached"][SceneBodyKey("objects/laptop", "button_link")]
        is expected
    )


def test_tcp_positions_rotated_link_transforms_offset_in_world():
    env, _, _, _ = _runtime()
    robot = ValidatorRobotContext(
        robot_name="robots/panda",
        tcp_offsets=(("ee", (0.12, 0, 0)),),
    )
    data = env.scene["robots/panda"].data
    data.body_pos_w[0, 0] = torch.tensor([1.0, 2.0, 3.0])
    data.body_quat_w[0, 0] = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
    assert robot.tcp_positions_w(env).tolist()[0] == pytest.approx(
        [1, 2.12, 3]
    )


@pytest.mark.parametrize("near_arm", ["left_finger", "right_finger"])
def test_reach_checker_two_tcp_offsets_either_arm_can_reach(near_arm):
    from dataclasses import replace

    env, context, scope, _ = _runtime()
    context.robot = replace(
        context.robot,
        tcp_offsets=tuple(
            (name, (0, 0, 0.12)) for name in ["left_finger", "right_finger"]
        ),
    )
    robot = env.scene["robots/panda"]
    index = robot.body_names.index(near_arm)
    robot.data.body_pos_w[0, index] = torch.tensor([0.1, 0, -0.12])
    metrics = ReachChecker(threshold=0.01).evaluate(env, context, scope)
    assert metrics.unary["reached"][
        SceneBodyKey("objects/laptop", "button_link")
    ]


@pytest.mark.parametrize("frames", [(), (("missing", (0, 0, 0)),)])
def test_tcp_positions_missing_configuration_or_link_raises_value_error(
    frames,
):
    env, _, _, _ = _runtime()
    robot = ValidatorRobotContext(
        robot_name="robots/panda", tcp_offsets=frames
    )
    with pytest.raises(ValueError, match="TCP"):
        robot.tcp_positions_w(env)


def test_validator_context_rotated_tcp_transform_uses_inverse_convention():
    profile = SimpleNamespace(
        ee_body_name="ee", gripper_body_names=(), gripper_joint_names=()
    )
    info = SimpleNamespace(
        manipulator_profile=profile,
        gripper_open_val=[],
        gripper_close_val=[],
        t_standard_tcp_to_robot_ee=[
            [0, -1, 0, 0],
            [1, 0, 0, -0.12],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )
    embodiment = SimpleNamespace(
        scene_name="robots/panda", get_robot_info_cfgs=lambda: {"arm": info}
    )
    context = ValidatorContext.from_embodiment(embodiment, RoleRegistry())
    env, _, _, _ = _runtime()
    env.scene["robots/panda"].data.body_pos_w[0, 0] = torch.zeros(3)
    assert context.robot.tcp_positions_w(env).tolist()[0] == pytest.approx(
        [0.12, 0, 0]
    )


@pytest.mark.parametrize("transform", [None, [[1, 0], [0, 1]], [[0] * 4] * 4])
def test_validator_context_invalid_tcp_transform_raises_value_error(transform):
    profile = SimpleNamespace(
        ee_body_name="ee", gripper_body_names=(), gripper_joint_names=()
    )
    info = SimpleNamespace(
        manipulator_profile=profile, t_standard_tcp_to_robot_ee=transform
    )
    embodiment = SimpleNamespace(
        scene_name="robot", get_robot_info_cfgs=lambda: {"arm": info}
    )
    with pytest.raises(ValueError, match="Invalid TCP transform"):
        ValidatorContext.from_embodiment(embodiment, RoleRegistry())


@pytest.mark.parametrize(
    "dof_type,displacement,expected",
    [
        (1, 0.00005, False),
        (1, 0.0002, True),
        (0, 0.005, False),
        (0, 0.02, True),
    ],
)
def test_moved_selector_joint_units_uses_small_absolute_threshold(
    dof_type,
    displacement,
    expected,
):
    env, context, scope, store = _runtime()
    asset = env.scene["objects/laptop"]
    asset.root_physx_view.shared_metatype.dof_types = [dof_type]
    asset.data.joint_pos[0, 0] = -0.576 - displacement
    store.update(JointStateChecker().evaluate(env, context, scope))
    selector = BoundRoleJointMovedSelector(
        metric_store=store, context=context, role_id="primary"
    )
    assert selector(env) is expected
