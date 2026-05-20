# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""LayoutBuilder.build named-role + rest->distractor convention."""

from __future__ import annotations
from unittest.mock import MagicMock

import pytest

from robo_orchard_sim.orchard_env.assets.object_spec import RigidObjectSpec
from robo_orchard_sim.orchard_env.assets.pool_spec import PoolSpec
from robo_orchard_sim.orchard_env.layout.builder import LayoutBuilder
from robo_orchard_sim.orchard_env.layout.loader import (
    Layout,
    LayoutObject,
    LayoutSequence,
    LayoutValidationError,
)


def _obj(category: str) -> LayoutObject:
    return LayoutObject(
        category=category,
        position=(0.0, 0.0, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
    )


def _seq(*role_to_category: dict[str, str]) -> LayoutSequence:
    entries = [
        Layout(objects={r: _obj(c) for r, c in mapping.items()}, raw={})
        for mapping in role_to_category
    ]
    return LayoutSequence(entries=entries, raw=[])


def _resolver():
    """Resolver mock: returns one RigidObjectSpec per asset_configs key."""
    resolver = MagicMock()

    def resolve(asset_configs):
        return {
            key: RigidObjectSpec(name=key, usd_path=f"/d/{key}.usd")
            for key in asset_configs
        }

    resolver.resolve.side_effect = resolve
    return resolver


def test_named_roles_map_to_slots_rest_become_distractors_in_order():
    seq = _seq(
        {"src": "bread", "ref": "gum", "dest": "mug", "distractor_0": "bread"}
    )
    assets, builder = LayoutBuilder.build(
        seq, _resolver(), {"src": "pick", "dest": "place"}
    )
    # named roles -> declared slots; other roles (ref, distractor_0) ->
    # distractor_0, distractor_1 in insertion order.
    assert set(assets) == {"pick", "place", "distractor_0", "distractor_1"}
    # role_member_by_category stays keyed by UPSTREAM json role.
    assert set(builder.role_member_by_category) == {
        "src",
        "ref",
        "dest",
        "distractor_0",
    }


def test_single_category_role_yields_object_spec():
    seq = _seq({"src": "bread"})
    assets, _ = LayoutBuilder.build(seq, _resolver(), {"src": "pick"})
    assert isinstance(assets["pick"], RigidObjectSpec)


def test_multi_category_role_yields_pool_spec():
    seq = _seq({"src": "bread"}, {"src": "toast"})
    assets, _ = LayoutBuilder.build(seq, _resolver(), {"src": "pick"})
    assert isinstance(assets["pick"], PoolSpec)


def test_named_role_absent_from_entry_raises():
    seq = _seq({"ref": "gum"})
    with pytest.raises(LayoutValidationError, match="src"):
        LayoutBuilder.build(seq, _resolver(), {"src": "pick"})


def test_role_key_set_differs_across_entries_raises():
    seq = _seq(
        {"src": "bread", "ref": "gum"},
        {"src": "bread", "distractor_0": "gum"},
    )
    with pytest.raises(LayoutValidationError, match="role keys differ"):
        LayoutBuilder.build(seq, _resolver(), {"src": "pick"})


def test_empty_layout_sequence_raises():
    seq = LayoutSequence(entries=[], raw=[])
    with pytest.raises(LayoutValidationError, match="empty layout sequence"):
        LayoutBuilder.build(seq, _resolver(), {"src": "pick"})


def test_validate_slot_filters_none_returns_empty():
    out = LayoutBuilder._validate_slot_filters(None, {"pick"})
    assert out == {}


def test_validate_slot_filters_empty_returns_empty():
    out = LayoutBuilder._validate_slot_filters({}, {"pick"})
    assert out == {}


def test_validate_slot_filters_basic_overlay_extracts_filter():
    overlay = {"pick": {"filter": {"tags": ["is_graspable"]}}}
    out = LayoutBuilder._validate_slot_filters(overlay, {"pick"})
    assert out == {"pick": {"tags": ["is_graspable"]}}


def test_validate_slot_filters_unknown_slot_raises():
    overlay = {"not_a_slot": {"filter": {"tags": ["is_graspable"]}}}
    with pytest.raises(LayoutValidationError, match="unknown slot"):
        LayoutBuilder._validate_slot_filters(overlay, {"pick"})


def test_validate_slot_filters_entry_has_non_filter_key_raises():
    overlay = {"pick": {"filter": {}, "prim_name": "x"}}
    with pytest.raises(LayoutValidationError, match="prim_name"):
        LayoutBuilder._validate_slot_filters(overlay, {"pick"})


def test_validate_slot_filters_entry_missing_filter_key_raises():
    overlay = {"pick": {}}
    with pytest.raises(LayoutValidationError, match="must contain 'filter'"):
        LayoutBuilder._validate_slot_filters(overlay, {"pick"})


def test_validate_slot_filters_category_inside_filter_raises():
    overlay = {"pick": {"filter": {"category": "peach"}}}
    with pytest.raises(LayoutValidationError, match="category"):
        LayoutBuilder._validate_slot_filters(overlay, {"pick"})


def test_validate_slot_filters_distractor_slot_name_allowed():
    overlay = {"distractor_0": {"filter": {"tags": ["is_graspable"]}}}
    out = LayoutBuilder._validate_slot_filters(
        overlay, {"pick", "distractor_0"}
    )
    assert out == {"distractor_0": {"tags": ["is_graspable"]}}


def test_validate_slot_filters_filter_not_mapping_raises():
    overlay = {"pick": {"filter": ["is_graspable"]}}
    with pytest.raises(
        LayoutValidationError, match="filter must be a mapping"
    ):
        LayoutBuilder._validate_slot_filters(overlay, {"pick"})


def test_validate_slot_filters_entry_not_mapping_raises():
    overlay = {"pick": "is_graspable"}
    with pytest.raises(LayoutValidationError, match="must be a mapping"):
        LayoutBuilder._validate_slot_filters(overlay, {"pick"})  # type: ignore[arg-type]


def test_validate_slot_filters_multi_slot_overlay():
    overlay = {
        "pick": {"filter": {"tags": ["is_graspable"]}},
        "place": {"filter": {"color": "red"}},
    }
    out = LayoutBuilder._validate_slot_filters(overlay, {"pick", "place"})
    assert out == {
        "pick": {"tags": ["is_graspable"]},
        "place": {"color": "red"},
    }


def test_slot_filters_none_passes_only_category_filter():
    """Backward-compat: no overlay → filter just has category."""
    seq = _seq({"src": "bread"})
    resolver = _resolver()
    LayoutBuilder.build(seq, resolver, {"src": "pick"})
    asset_configs = resolver.resolve.call_args[0][0]
    assert asset_configs["pick"]["filter"] == {"category": "bread"}


def test_slot_filters_merges_tags_into_filter():
    seq = _seq({"src": "bread"})
    resolver = _resolver()
    LayoutBuilder.build(
        seq,
        resolver,
        {"src": "pick"},
        slot_filters={"pick": {"filter": {"tags": ["is_graspable"]}}},
    )
    asset_configs = resolver.resolve.call_args[0][0]
    assert asset_configs["pick"]["filter"] == {
        "category": "bread",
        "tags": ["is_graspable"],
    }


def test_slot_filters_merges_multiple_dimensions():
    seq = _seq({"src": "bread"})
    resolver = _resolver()
    LayoutBuilder.build(
        seq,
        resolver,
        {"src": "pick"},
        slot_filters={
            "pick": {"filter": {"tags": ["is_graspable"], "color": "red"}},
        },
    )
    asset_configs = resolver.resolve.call_args[0][0]
    assert asset_configs["pick"]["filter"] == {
        "category": "bread",
        "tags": ["is_graspable"],
        "color": "red",
    }


def test_slot_filters_applies_to_each_pool_member():
    """Multi-category role → each pool member gets the overlay."""
    seq = _seq({"src": "bread"}, {"src": "toast"})
    resolver = _resolver()
    LayoutBuilder.build(
        seq,
        resolver,
        {"src": "pick"},
        slot_filters={"pick": {"filter": {"tags": ["is_graspable"]}}},
    )
    asset_configs = resolver.resolve.call_args[0][0]
    keys = sorted(asset_configs)
    assert keys == ["pick_pool_0", "pick_pool_1"]
    for k in keys:
        assert asset_configs[k]["filter"]["tags"] == ["is_graspable"]
    assert {asset_configs[k]["filter"]["category"] for k in keys} == {
        "bread",
        "toast",
    }


def test_slot_filters_unknown_slot_raises_via_build():
    seq = _seq({"src": "bread"})
    with pytest.raises(LayoutValidationError, match="unknown slot"):
        LayoutBuilder.build(
            seq,
            _resolver(),
            {"src": "pick"},
            slot_filters={"nonexistent": {"filter": {"tags": ["x"]}}},
        )
