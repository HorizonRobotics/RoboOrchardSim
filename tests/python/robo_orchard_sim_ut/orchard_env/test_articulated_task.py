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

import json
from types import SimpleNamespace

import pytest
import torch

from robo_orchard_sim.asset_manager.metadata import (
    ArticulationOperationMeta,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners import UsdFileCfg
from robo_orchard_sim.ext.models.assets.articulation import ArticulationCfg
from robo_orchard_sim.orchard_env.assets import (
    ArticulatedObjectSpec,
    JointOperationMeta,
)
from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets
from robo_orchard_sim.orchard_env.task_templates.articulated.affordance_task import (  # noqa: E501
    AffordanceTask,
)
from robo_orchard_sim.orchard_env.task_templates.articulated.base import (
    PRIMARY_ROLE,
    ArticulatedTaskBase,
    ArticulatedTaskParams,
)
from robo_orchard_sim.orchard_env.task_templates.articulated.close_task import (  # noqa: E501
    JointCloseTask,
)
from robo_orchard_sim.orchard_env.task_templates.articulated.joint_direction_task import (  # noqa: E501
    JointDirectionTask,
)
from robo_orchard_sim.orchard_env.task_templates.articulated.open_task import (
    JointOpenTask,
)
from robo_orchard_sim.task_components.instructions.base import (
    InstructionWrapper,
)
from robo_orchard_sim.task_components.role_registry import (
    RoleRegistry,
    TargetRef,
)
from robo_orchard_sim.task_components.validators.context import (
    ValidatorContext,
    ValidatorRobotContext,
)
from robo_orchard_sim.task_components.validators.physical_entity import (
    SceneBodyKey,
)
from robo_orchard_sim.task_components.validators.role_scope import RoleScope


def _primary_assets(
    spec: "ArticulatedObjectSpec | None" = None,
) -> TaskAssets:
    return TaskAssets(
        role_candidates={PRIMARY_ROLE: [spec or _articulated_object_spec()]}
    )


def _joint(
    joint_name: str,
    semantic_name: str,
    operations: frozenset[str] = frozenset({"open", "close"}),
) -> JointOperationMeta:
    operation_metadata = {
        operation: ArticulationOperationMeta(
            interaction_link=(
                f"{joint_name}_button"
                if operation in {"open", "pull"}
                else f"{joint_name}_link"
            ),
            initial_joint_position=(
                0.0 if operation in {"open", "pull"} else 1.0
            ),
            target_joint_fraction=(
                0.0 if operation in {"open", "pull"} else 1.0
            ),
        )
        for operation in operations
    }
    return JointOperationMeta(
        joint_name=joint_name,
        semantic_name=semantic_name,
        outcome_link=f"{joint_name}_link",
        operations=operation_metadata,
    )


def _articulated_object_spec(
    *joints: JointOperationMeta,
    caption_path: str | None = None,
    joint_pos: dict[str, float] | None = None,
) -> ArticulatedObjectSpec:
    return ArticulatedObjectSpec(
        name="cabinet",
        template_cfg=ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/cabinet",
            spawn=UsdFileCfg(usd_path="/tmp/cabinet.usd"),
            actuators={},
        ),
        caption_path=caption_path,
        uuid="cabinet-uuid",
        category="cabinet",
        description="wooden cabinet",
        joint_pos=joint_pos,
        joint_operations=joints or (_joint("drawer_joint", "drawer"),),
    )


def _context_bound_to(*targets: tuple[str, str]) -> ValidatorContext:
    """A context whose primary role names the given (joint, op) target."""
    registry = RoleRegistry(num_envs=1)
    semantic_name, operation = targets[0]
    registry.bind_one(
        0,
        PRIMARY_ROLE,
        TargetRef("objects/cabinet", semantic_name, operation),
    )
    return ValidatorContext(robot=None, role_registry=registry)


def _runtime_env(spec: ArticulatedObjectSpec) -> SimpleNamespace:
    runtime_spec = spec.with_default_namespace("objects")
    return SimpleNamespace(
        scene={
            runtime_spec.scene_name: SimpleNamespace(
                cfg=runtime_spec.to_isaac_cfg(),
            ),
        }
    )


def test_affordance_candidates_swap_modes_infer_default_operation() -> None:
    spec = _articulated_object_spec(
        _joint(
            "drawer_joint",
            "drawer",
            frozenset({"open", "close"}),
        ),
    )
    task = AffordanceTask(assets=_primary_assets(spec))

    without_swap = task.get_role_candidates(PRIMARY_ROLE, swap=False)
    with_swap = task.get_role_candidates(PRIMARY_ROLE, swap=True)

    expected = [TargetRef("objects/cabinet", "drawer", "open")]
    assert without_swap == expected
    assert with_swap == expected


def test_affordance_candidates_multiple_joints_returns_each_default() -> None:
    spec = _articulated_object_spec(
        _joint(
            "drawer_joint",
            "drawer",
            frozenset({"open", "close"}),
        ),
        _joint(
            "door_joint",
            "door",
            frozenset({"pull", "push"}),
        ),
    )
    task = AffordanceTask(assets=_primary_assets(spec))

    candidates = task.get_role_candidates(PRIMARY_ROLE, swap=True)

    assert candidates == [
        TargetRef("objects/cabinet", "drawer", "open"),
        TargetRef("objects/cabinet", "door", "pull"),
    ]


def test_affordance_candidates_nonzero_default_infers_matching_operation() -> (
    None
):
    spec = _articulated_object_spec(
        _joint("drawer_joint", "drawer"),
        joint_pos={"drawer_joint": 1.0},
    )
    task = AffordanceTask(assets=_primary_assets(spec))

    candidates = task.get_role_candidates(PRIMARY_ROLE, swap=True)

    assert candidates == [
        TargetRef("objects/cabinet", "drawer", "close"),
    ]


@pytest.mark.parametrize("matching_default", [0.0, 1.0, 0.5])
def test_affordance_candidates_unmatched_joint_returns_only_matches(
    matching_default: float,
) -> None:
    spec = _articulated_object_spec(
        _joint("door_joint", "door"),
        _joint("drawer_joint", "drawer"),
        joint_pos={"door_joint": 0.5, "drawer_joint": matching_default},
    )
    task = AffordanceTask(assets=_primary_assets(spec))

    expected = (
        []
        if matching_default == 0.5
        else [
            TargetRef(
                "objects/cabinet",
                "drawer",
                "open" if matching_default == 0.0 else "close",
            )
        ]
    )
    assert task.get_role_candidates(PRIMARY_ROLE) == expected


def test_affordance_candidates_ambiguous_operations_raises_value_error() -> (
    None
):
    joint = _joint("drawer_joint", "drawer")
    joint.operations["close"].initial_joint_position = 0.0
    task = AffordanceTask(
        assets=_primary_assets(_articulated_object_spec(joint))
    )

    with pytest.raises(ValueError, match="must match exactly one operation"):
        task.get_role_candidates(PRIMARY_ROLE, swap=True)


@pytest.mark.parametrize("task_type", [AffordanceTask, JointDirectionTask])
def test_articulated_task_counterfactual_instruction_raises_not_implemented(
    task_type,
) -> None:
    task = task_type(
        assets=_primary_assets(),
        instruction=InstructionWrapper("empty"),
    )

    with pytest.raises(NotImplementedError, match="counterfactual"):
        task.build_validator(
            context=_context_bound_to(("drawer", "open")),
        )


def test_articulated_role_candidates_disallowed_operations_excludes_them() -> (
    None
):
    spec = _articulated_object_spec(
        _joint(
            "drawer_joint",
            "drawer",
            frozenset({"open", "close", "pull", "push"}),
        ),
    )
    task = JointDirectionTask(assets=_primary_assets(spec))
    task.allowed_operations = frozenset({"open", "close"})

    candidates = task.get_role_candidates(PRIMARY_ROLE, swap=True)

    assert candidates == [
        TargetRef("objects/cabinet", "drawer", "close"),
        TargetRef("objects/cabinet", "drawer", "open"),
    ]


def test_joint_direction_candidates_incomplete_pair_returns_empty() -> None:
    spec = _articulated_object_spec(
        _joint("drawer_joint", "drawer", frozenset({"open"})),
        _joint("door_joint", "door", frozenset({"close"})),
    )
    task = JointDirectionTask(assets=_primary_assets(spec))

    assert task.get_role_candidates(PRIMARY_ROLE) == []


def test_open_task_accepts_joint_annotated_only_with_open() -> None:
    spec = _articulated_object_spec(
        _joint("drawer_joint", "drawer", frozenset({"open"})),
    )

    assert JointOpenTask(assets=_primary_assets(spec)).get_role_candidates(
        PRIMARY_ROLE
    ) == [TargetRef("objects/cabinet", "drawer", "open")]
    assert (
        JointCloseTask(assets=_primary_assets(spec)).get_role_candidates(
            PRIMARY_ROLE
        )
        == []
    )


def test_open_task_on_bidirectional_joint_omits_the_close_direction() -> None:
    spec = _articulated_object_spec(
        _joint("drawer_joint", "drawer", frozenset({"open", "close"})),
    )

    assert JointOpenTask(assets=_primary_assets(spec)).get_role_candidates(
        PRIMARY_ROLE
    ) == [TargetRef("objects/cabinet", "drawer", "open")]
    assert JointCloseTask(assets=_primary_assets(spec)).get_role_candidates(
        PRIMARY_ROLE
    ) == [TargetRef("objects/cabinet", "drawer", "close")]


def test_articulated_task_role_scope_operation_candidates_returns_links() -> (
    None
):
    spec = _articulated_object_spec()
    task = AffordanceTask(assets=_primary_assets(spec))

    scope = RoleScope.from_task(task)

    assert scope.entities_for_role(PRIMARY_ROLE) == (
        SceneBodyKey("objects/cabinet", "drawer_joint_button"),
        SceneBodyKey("objects/cabinet", "drawer_joint_link"),
    )


def test_articulated_role_candidates_unknown_role_returns_whole_entity() -> (
    None
):
    task = AffordanceTask(assets=_primary_assets())

    assert task.get_role_candidates("nonexistent") == [
        TargetRef("objects/cabinet")
    ]


def test_articulated_task_event_terms_pose_reset_precedes_joint_reset() -> (
    None
):
    task = AffordanceTask(assets=_primary_assets())

    event_cfg = task.get_event_cfg()

    assert tuple(event_cfg.terms) == (
        "random_pose_event",
        "joint_state_reset_event",
    )


def test_articulated_task_event_joint_reset_targets_primary_asset() -> None:
    task = JointDirectionTask(assets=_primary_assets())

    reset = task.get_event_cfg().terms["joint_state_reset_event"]

    assert reset.asset_cfgs[0].name == "objects/cabinet"


def test_affordance_task_event_joint_reset_uses_asset_default_state() -> None:
    task = AffordanceTask(assets=_primary_assets())

    reset = task.get_event_cfg().terms["joint_state_reset_event"]

    assert reset.operation_role_id is None


@pytest.mark.parametrize(
    "task_type,start_mode",
    [
        (JointDirectionTask, "opposing_midpoint"),
        (JointOpenTask, "metadata"),
        (JointCloseTask, "metadata"),
    ],
)
def test_joint_operation_event_task_type_uses_expected_reset_start(
    task_type, start_mode
) -> None:
    task = task_type(assets=_primary_assets())

    reset = task.get_event_cfg().terms["joint_state_reset_event"]

    assert (reset.operation_role_id, reset.operation_start_mode) == (
        PRIMARY_ROLE,
        start_mode,
    )


def test_articulated_task_enabled_light_reset_adds_event_term() -> None:
    task = AffordanceTask(
        assets=_primary_assets(),
        params=ArticulatedTaskParams(
            light_reset={
                "enabled": True,
                "preset": "default_distant_light",
            }
        ),
    )

    light_reset = task.get_event_cfg().terms["light_reset_event"]

    assert light_reset.asset_cfgs[0].name == "background/dis_light"


def test_articulated_task_enabled_texture_reset_adds_event_term() -> None:
    task = AffordanceTask(
        assets=_primary_assets(),
        params=ArticulatedTaskParams(
            texture_reset={
                "enabled": True,
                "asset_names": ["background/table"],
            }
        ),
    )

    texture_reset = task.get_event_cfg().terms["texture_reset_event"]

    assert texture_reset.asset_cfgs[0].name == "background/table"


@pytest.mark.parametrize("task_type", [JointDirectionTask, AffordanceTask])
@pytest.mark.parametrize(
    ("description_mode", "expected_description"),
    [
        ("raw", "cabinet"),
        ("seen", "wooden storage cabinet"),
        ("unseen", "four-drawer organizer"),
    ],
)
def test_instruction_context_description_mode_uses_articulation_caption(
    tmp_path,
    task_type: type[ArticulatedTaskBase],
    description_mode: str,
    expected_description: str,
) -> None:
    caption_path = tmp_path / "caption_candidates.json"
    caption_path.write_text(
        json.dumps(
            {
                "uuid": "cabinet-uuid",
                "raw": "cabinet",
                "seen": ["wooden storage cabinet"],
                "unseen": ["four-drawer organizer"],
            }
        )
    )
    spec = _articulated_object_spec(caption_path=str(caption_path))
    task = task_type(
        assets=_primary_assets(spec),
        instruction=InstructionWrapper(
            "articulated_operation",
            actor_description_mode=description_mode,
        ),
    )
    context = _context_bound_to(("drawer", "open"))

    instruction = InstructionWrapper("articulated_operation").render(
        actors=task.build_instruction_context(
            _runtime_env(spec),
            actor_description_seed=7,
            context=context,
        )
    )

    assert instruction == f"Open the drawer of the {expected_description}."


def test_instruction_context_rebound_target_returns_updated_description(
    tmp_path,
) -> None:
    # Rebinding the role — the way an evaluator would between episodes —
    # changes what the instruction describes without rebuilding the task.
    spec = _articulated_object_spec(
        _joint("drawer_joint", "drawer"),
        _joint("door_joint", "door"),
    )
    caption_path = tmp_path / "caption_candidates.json"
    caption_path.write_text(
        json.dumps(
            {
                "uuid": "cabinet-uuid",
                "raw": "cabinet",
                "seen": [],
                "unseen": [],
            }
        )
    )
    spec = spec.model_copy(update={"caption_path": str(caption_path)})
    task = AffordanceTask(assets=_primary_assets(spec))
    env = _runtime_env(spec)
    registry = RoleRegistry(num_envs=1)
    context = ValidatorContext(robot=None, role_registry=registry)

    registry.bind_one(
        0, PRIMARY_ROLE, TargetRef("objects/cabinet", "drawer", "open")
    )
    first = task.build_instruction_context(
        env, actor_description_seed=0, context=context
    )["actor1"]

    registry.bind_one(
        0, PRIMARY_ROLE, TargetRef("objects/cabinet", "door", "pull")
    )
    second = task.build_instruction_context(
        env, actor_description_seed=0, context=context
    )["actor1"]

    assert first["joint_description"] == "drawer"
    assert second["joint_description"] == "door"
    assert second["operation"] == "pull"


def test_instruction_context_missing_context_raises_value_error() -> None:
    task = AffordanceTask(assets=_primary_assets())

    with pytest.raises(ValueError, match="ValidatorContext"):
        task.build_instruction_context(None, actor_description_seed=0)


def test_build_validator_missing_context_raises_value_error() -> None:
    task = AffordanceTask(assets=_primary_assets())

    with pytest.raises(ValueError, match="ValidatorContext"):
        task.build_validator()


@pytest.mark.parametrize("task_type", [JointDirectionTask, AffordanceTask])
def test_articulated_task_validator_valid_context_returns_expected_rubric(
    task_type: type[ArticulatedTaskBase],
) -> None:
    spec = _articulated_object_spec()
    task = task_type(assets=_primary_assets(spec))
    registry = RoleRegistry()
    registry.bind_one(
        0,
        PRIMARY_ROLE,
        TargetRef("objects/cabinet", "drawer", "open"),
    )
    context = ValidatorContext(
        robot=ValidatorRobotContext(
            robot_name="robots/panda",
            ee_links=("ee",),
            gripper_body_groups=(("left_finger", "right_finger"),),
        ),
        role_registry=registry,
    )

    validator = task.build_validator(context)

    assert validator.criteria_name == [
        "reach",
        "contact",
        "grasp",
        "moved",
        "target_joint_state",
        "settled_at_target",
    ]
    assert (
        validator.success_criteria,
        validator.progress_criteria,
    ) == ((5,), (0, 1, 3, 4, 5))


@pytest.mark.parametrize(
    "operation,contact,displacement,expected",
    [
        ("press", True, 0.0002, True),
        ("press", False, 0.0002, False),
        ("press", True, 0.00005, False),
        ("open", True, 0.0002, False),
    ],
)
def test_affordance_success_contact_and_motion_applies_press_only(
    operation,
    contact,
    displacement,
    expected,
):
    # Reuse the simulated sensor/asset boundary used by checker tests.
    from robo_orchard_sim_ut.test_validator_articulation_scene import (
        _runtime,
    )

    env, context, _, _ = _runtime()
    asset = env.scene["objects/laptop"]
    joint = asset.cfg.joint_operations[0]
    joint.operations["press"] = joint.operations["open"].model_copy()
    joint.operations[operation].initial_joint_position = 0.0
    asset.root_physx_view.shared_metatype.dof_types = [1]
    asset.data.joint_pos[0, 0] = -displacement
    for sensor_name in (
        "sensors/gripper_contact_0_0",
        "sensors/gripper_contact_0_1",
    ):
        if not contact:
            env.scene[sensor_name].force = (0.0, 0.0, 0.0)
    context.role_registry.bind_one(
        0, PRIMARY_ROLE, TargetRef("objects/laptop", "lid", operation)
    )
    spec = _articulated_object_spec(joint).model_copy(
        update={"name": "laptop"}
    )
    task = AffordanceTask(assets=_primary_assets(spec))
    validator = task.build_validator(context)

    assert validator.evaluate(env).success is expected
    if expected:
        # Returning to the initial state after release must not undo a press.
        asset.data.joint_pos[0, 0] = 0.0
        for sensor_name in (
            "sensors/gripper_contact_0_0",
            "sensors/gripper_contact_0_1",
        ):
            env.scene[sensor_name].data.force_matrix_w.zero_()
        assert validator.evaluate(env).success is True


@pytest.mark.parametrize(
    "task_type,operation,partial,target",
    [
        (JointOpenTask, "open", -0.9, -1.75),
        (JointCloseTask, "close", -1.4, -0.576),
        (AffordanceTask, "open", -0.9, -1.75),
    ],
)
def test_articulated_progress_missing_contact_counts_motion_and_target(
    task_type, operation, partial, target
):
    from robo_orchard_sim_ut.test_validator_articulation_scene import _runtime

    env, context, _, _ = _runtime()
    asset = env.scene["objects/laptop"]
    joint = asset.cfg.joint_operations[0]
    for name in ("sensors/gripper_contact_0_0", "sensors/gripper_contact_0_1"):
        env.scene[name].force = (0.0, 0.0, 0.0)
    env.scene["robots/panda"].data.body_pos_w[0, 0, 0] = 100.0
    context.role_registry.bind_one(
        0, PRIMARY_ROLE, TargetRef("objects/laptop", "lid", operation)
    )
    spec = _articulated_object_spec(
        joint,
        joint_pos={
            joint.joint_name: joint.operations[
                operation
            ].initial_joint_position
        },
    ).model_copy(update={"name": "laptop"})
    validator = task_type(assets=_primary_assets(spec)).build_validator(
        context
    )
    asset.data.joint_pos[0, 0] = partial
    moved = validator.evaluate(env)
    assert (moved.success, moved.progress) == (False, 0.2)

    asset.data.joint_pos[0, 0] = target
    asset.data.body_lin_vel_w[0, 2, 0] = 1.0
    for _ in range(5):
        at_target = validator.evaluate(env)
    assert (at_target.success, at_target.progress) == (False, 0.4)

    asset.data.body_lin_vel_w[0, 2, 0] = 0.0
    for _ in range(4):
        settling = validator.evaluate(env)
    assert (settling.success, settling.progress) == (False, 0.4)
    complete = validator.evaluate(env)
    assert (
        complete.success,
        complete.progress,
        complete.metrics["criteria_reached"]["contact"],
        complete.metrics["criteria_reached"]["reach"],
    ) == (True, 1.0, False, False)


@pytest.mark.parametrize("operation", ["open", "close", "pull", "push"])
@pytest.mark.parametrize("soft_limits", [(0.0, 1.0), (0.7, 0.9), (0.0, 0.3)])
@pytest.mark.parametrize(
    "offset,expected", [(0.0, False), (0.01, False), (0.03, True)]
)
def test_joint_direction_moved_from_clipped_midpoint_returns_expected_result(
    operation, soft_limits, offset, expected
):
    from robo_orchard_sim_ut.test_validator_articulation_scene import _runtime

    env, context, _, _ = _runtime()
    asset = env.scene["objects/laptop"]
    joint = asset.cfg.joint_operations[0].model_copy(deep=True)
    joint.operations["open"].initial_joint_position = 0.0
    joint.operations["close"].initial_joint_position = 1.0
    joint.operations["pull"] = joint.operations["open"].model_copy()
    joint.operations["push"] = joint.operations["close"].model_copy()
    asset.cfg.joint_operations = (joint,)
    asset.data.soft_joint_pos_limits = torch.tensor([[soft_limits]])
    asset.data.joint_pos_limits = torch.tensor([[[0.0, 1.0]]])
    start = min(max(0.5, soft_limits[0]), soft_limits[1])
    asset.data.joint_pos[0, 0] = start + offset
    context.role_registry.bind_one(
        0, PRIMARY_ROLE, TargetRef("objects/laptop", "lid", operation)
    )
    spec = _articulated_object_spec(joint).model_copy(
        update={"name": "laptop"}
    )
    validator = JointDirectionTask(
        assets=_primary_assets(spec)
    ).build_validator(context)

    result = validator.evaluate(env)

    assert result.metrics["criteria_reached"]["moved"] is expected


@pytest.mark.parametrize("operation", ["open", "close"])
@pytest.mark.parametrize(
    "offset,expected", [(0.0, False), (0.1, False), (0.3, True)]
)
def test_joint_direction_delta_target_from_midpoint_returns_expected_result(
    operation, offset, expected
):
    from robo_orchard_sim_ut.test_validator_articulation_scene import _runtime

    env, context, _, _ = _runtime()
    asset = env.scene["objects/laptop"]
    joint = asset.cfg.joint_operations[0]
    joint.operations[operation] = ArticulationOperationMeta(
        interaction_link="lid_link",
        initial_joint_position=(0.0 if operation == "open" else 1.0),
        target_joint_delta=0.2,
    )
    opposite = "close" if operation == "open" else "open"
    joint.operations[opposite].initial_joint_position = (
        1.0 if operation == "open" else 0.0
    )
    asset.data.soft_joint_pos_limits = torch.tensor([[[0.7, 1.0]]])
    asset.data.joint_pos[0, 0] = 0.7 + offset
    context.role_registry.bind_one(
        0, PRIMARY_ROLE, TargetRef("objects/laptop", "lid", operation)
    )
    spec = _articulated_object_spec(joint).model_copy(
        update={"name": "laptop"}
    )
    validator = JointDirectionTask(
        assets=_primary_assets(spec)
    ).build_validator(context)

    for _ in range(5):
        result = validator.evaluate(env)

    assert result.metrics["criteria_reached"]["target_joint_state"] is expected


@pytest.mark.parametrize(
    "operation,target,position,soft_limits,expected",
    [
        ("open", 0.25, 0.1, (0.0, 1.0), True),
        ("open", 0.25, 0.75, (0.0, 1.0), False),
        ("close", 0.75, 0.9, (0.0, 1.0), True),
        ("close", 0.75, 0.1, (0.0, 1.0), False),
        ("open", 0.6, 0.7, (0.6, 1.0), False),
        ("open", 0.6, 0.6, (0.6, 1.0), True),
        ("close", 0.4, 0.3, (0.0, 0.4), False),
        ("close", 0.4, 0.4, (0.0, 0.4), True),
    ],
)
def test_joint_direction_fraction_target_midpoint_direction_returns_expected(
    operation, target, position, soft_limits, expected
):
    from robo_orchard_sim_ut.test_validator_articulation_scene import _runtime

    env, context, _, _ = _runtime()
    asset = env.scene["objects/laptop"]
    joint = asset.cfg.joint_operations[0]
    joint.operations["open"].initial_joint_position = 0.0
    joint.operations["close"].initial_joint_position = 1.0
    joint.operations[operation].target_joint_fraction = target
    asset.data.joint_pos_limits = torch.tensor([[[0.0, 1.0]]])
    asset.data.soft_joint_pos_limits = torch.tensor([[soft_limits]])
    asset.data.joint_pos[0, 0] = position
    context.role_registry.bind_one(
        0, PRIMARY_ROLE, TargetRef("objects/laptop", "lid", operation)
    )
    spec = _articulated_object_spec(joint).model_copy(
        update={"name": "laptop"}
    )
    validator = JointDirectionTask(
        assets=_primary_assets(spec)
    ).build_validator(context)

    for _ in range(5):
        result = validator.evaluate(env)

    assert result.success is expected
