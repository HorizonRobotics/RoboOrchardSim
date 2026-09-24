## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

"""Unit tests for TargetRef and RoleRegistry."""

import dataclasses

import pytest

from robo_orchard_sim.task_components.role_registry import (
    RoleRegistry,
    TargetRef,
)

PICK_0 = TargetRef("pick_0")
PICK_2 = TargetRef("pick_2")
DPICK_0 = TargetRef("dpick_0")
DPICK_1 = TargetRef("dpick_1")
UPPER_DRAWER = TargetRef("cabinet", "UpperDrawerJoint")
LOWER_DRAWER = TargetRef("cabinet", "LowerDrawerJoint")
OPEN_UPPER_DRAWER = TargetRef("cabinet", "UpperDrawerJoint", "open")


def test_rigid_target_defaults_to_whole_entity():
    ref = TargetRef("pick_0")
    assert ref.scene_name == "pick_0"
    assert ref.semantic_name is None
    assert ref.operation is None


def test_articulation_semantic_name_carries_joint_name():
    ref = TargetRef("cabinet", "UpperDrawerJoint")
    assert ref.scene_name == "cabinet"
    assert ref.semantic_name == "UpperDrawerJoint"


def test_semantic_name_carries_the_operation_to_perform():
    ref = TargetRef("cabinet", "UpperDrawerJoint", "open")
    assert ref.operation == "open"


def test_opposing_operations_on_one_joint_are_distinct_targets():
    opening = TargetRef("cabinet", "UpperDrawerJoint", "open")
    closing = TargetRef("cabinet", "UpperDrawerJoint", "close")

    assert opening != closing
    assert len({opening, closing}) == 2


def test_operation_requires_a_semantic_name_to_act_on():
    with pytest.raises(ValueError, match="needs a semantic_name"):
        TargetRef("cabinet", operation="open")


def test_unknown_operation_is_rejected_with_the_registered_vocabulary():
    with pytest.raises(ValueError) as exc_info:
        TargetRef("cabinet", "UpperDrawerJoint", "yank")  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert "yank" in message
    assert "open" in message


def test_target_ref_is_frozen():
    ref = TargetRef("pick_0")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.scene_name = "pick_1"  # type: ignore[misc]


def test_target_ref_equality_and_hashability():
    assert TargetRef("pick_0") == TargetRef("pick_0")
    assert TargetRef("cabinet", "UpperDrawerJoint") != TargetRef("cabinet")
    assert len({TargetRef("pick_0"), TargetRef("pick_0")}) == 1
    assert {TargetRef("cabinet", "UpperDrawerJoint"): 1}


def test_target_ref_rejects_empty_scene_name():
    with pytest.raises(ValueError, match="scene_name"):
        TargetRef("")


def test_target_ref_rejects_empty_semantic_name():
    with pytest.raises(ValueError, match="semantic_name"):
        TargetRef("cabinet", "")


def test_target_ref_str_renders_whole_entity_and_semantic_name():
    assert str(TargetRef("pick_0")) == "pick_0"
    assert str(TargetRef("cabinet", "UpperDrawerJoint")) == (
        "cabinet/UpperDrawerJoint"
    )
    assert str(OPEN_UPPER_DRAWER) == "cabinet/UpperDrawerJoint:open"


def test_registry_round_trips_an_operation_target():
    registry = RoleRegistry()
    registry.bind_one(0, "articulate", OPEN_UPPER_DRAWER)

    resolved = registry.resolve_one("articulate")
    assert resolved.semantic_name == "UpperDrawerJoint"
    assert resolved.operation == "open"


def test_bind_and_resolve_round_trip():
    registry = RoleRegistry()
    registry.bind_one(0, "pick", PICK_0)
    registry.bind_many(0, "distractors_pick", [DPICK_0, DPICK_1])

    assert registry.resolve("pick") == PICK_0
    assert registry.resolve("distractors_pick") == [DPICK_0, DPICK_1]


def test_resolve_one_returns_target_and_rejects_many_binding():
    registry = RoleRegistry()
    registry.bind_one(0, "pick", PICK_0)
    assert registry.resolve_one("pick") == PICK_0

    registry.bind_many(0, "distractors_pick", [DPICK_0, DPICK_1])
    with pytest.raises(TypeError, match="resolve_many"):
        registry.resolve_one("distractors_pick")


def test_resolve_many_returns_list_and_rejects_one_binding():
    registry = RoleRegistry()
    registry.bind_many(0, "distractors_pick", [DPICK_0])
    assert registry.resolve_many("distractors_pick") == [DPICK_0]

    registry.bind_one(0, "pick", PICK_0)
    with pytest.raises(TypeError, match="resolve_one"):
        registry.resolve_many("pick")


def test_articulation_semantic_name_binding_survives_cardinality_checks():
    """A single non-str target must not be mistaken for a 'many' binding."""
    registry = RoleRegistry()
    registry.bind_one(0, "place", UPPER_DRAWER)

    assert registry.resolve_one("place") == UPPER_DRAWER
    with pytest.raises(TypeError, match="resolve_one"):
        registry.resolve_many("place")


def test_many_role_holds_articulation_semantic_names():
    registry = RoleRegistry()
    registry.bind_many(0, "drawers", [UPPER_DRAWER, LOWER_DRAWER])

    assert registry.resolve_many("drawers") == [UPPER_DRAWER, LOWER_DRAWER]
    with pytest.raises(TypeError, match="resolve_many"):
        registry.resolve_one("drawers")


def test_rigid_and_semantic_name_roles_coexist():
    registry = RoleRegistry()
    registry.bind_one(0, "pick", PICK_0)
    registry.bind_one(0, "place", UPPER_DRAWER)

    assert registry.resolve_one("pick").semantic_name is None
    assert registry.resolve_one("place").semantic_name == "UpperDrawerJoint"


def test_resolve_unbound_role_lists_bound_roles():
    registry = RoleRegistry()
    registry.bind_one(0, "pick", PICK_0)
    registry.bind_one(0, "place", UPPER_DRAWER)

    with pytest.raises(KeyError) as exc_info:
        registry.resolve("missing")

    message = str(exc_info.value)
    assert "missing" in message
    assert "'pick'" in message
    assert "'place'" in message


def test_rebinding_a_role_overwrites_previous_value():
    registry = RoleRegistry()
    registry.bind_one(0, "pick", PICK_0)
    registry.bind_one(0, "pick", PICK_2)

    assert registry.resolve_one("pick") == PICK_2


def test_rebinding_can_switch_cardinality():
    registry = RoleRegistry()
    registry.bind_one(0, "pick", PICK_0)
    registry.bind_many(0, "pick", [PICK_0, PICK_2])

    assert registry.resolve_many("pick") == [PICK_0, PICK_2]
    with pytest.raises(TypeError, match="resolve_many"):
        registry.resolve_one("pick")


def test_bindings_are_independent_per_env():
    registry = RoleRegistry(num_envs=2)
    registry.bind_one(0, "pick", PICK_0)
    registry.bind_one(1, "pick", PICK_2)

    assert registry.num_envs == 2
    assert registry.resolve_one("pick", env_idx=0) == PICK_0
    assert registry.resolve_one("pick", env_idx=1) == PICK_2


def test_out_of_range_env_idx_raises_index_error():
    registry = RoleRegistry(num_envs=1)
    with pytest.raises(IndexError):
        registry.bind_one(5, "pick", PICK_0)


def test_clear_targets_single_env_or_all_envs():
    registry = RoleRegistry(num_envs=2)
    registry.bind_one(0, "pick", PICK_0)
    registry.bind_one(1, "pick", PICK_2)

    registry.clear(env_idx=0)
    assert registry.bindings(0) == {}
    assert registry.bindings(1) == {"pick": PICK_2}

    registry.clear()
    assert registry.bindings(1) == {}


def test_bindings_exposes_targets_without_cardinality():
    registry = RoleRegistry()
    registry.bind_one(0, "pick", PICK_0)
    registry.bind_many(0, "distractors_pick", [DPICK_0])

    assert registry.bindings() == {
        "pick": PICK_0,
        "distractors_pick": [DPICK_0],
    }


def test_bindings_returns_a_copy():
    registry = RoleRegistry()
    registry.bind_one(0, "pick", PICK_0)

    snapshot = registry.bindings()
    snapshot["pick"] = PICK_2

    assert registry.resolve_one("pick") == PICK_0


def test_bind_many_copies_the_input_list():
    registry = RoleRegistry()
    targets = [DPICK_0]
    registry.bind_many(0, "distractors_pick", targets)
    targets.append(DPICK_1)

    assert registry.resolve_many("distractors_pick") == [DPICK_0]


def test_num_envs_must_be_positive():
    with pytest.raises(ValueError):
        RoleRegistry(num_envs=0)
