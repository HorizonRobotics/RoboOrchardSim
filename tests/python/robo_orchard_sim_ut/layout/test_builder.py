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


def test_validate_slot_overlays_none_returns_empty():
    out = LayoutBuilder._validate_slot_overlays(None, {"pick"})
    assert out == {}


def test_validate_slot_overlays_empty_returns_empty():
    out = LayoutBuilder._validate_slot_overlays({}, {"pick"})
    assert out == {}


def test_validate_slot_overlays_filter_only():
    overlay = {"pick": {"filter": {"tags": ["is_graspable"]}}}
    out = LayoutBuilder._validate_slot_overlays(overlay, {"pick"})
    assert out == {
        "pick": {"filter": {"tags": ["is_graspable"]}, "split": None}
    }


def test_validate_slot_overlays_split_only():
    overlay = {"pick": {"split": "seen"}}
    out = LayoutBuilder._validate_slot_overlays(overlay, {"pick"})
    assert out == {"pick": {"filter": {}, "split": "seen"}}


def test_validate_slot_overlays_filter_and_split():
    overlay = {"pick": {"filter": {"tags": ["is_graspable"]}, "split": "seen"}}
    out = LayoutBuilder._validate_slot_overlays(overlay, {"pick"})
    assert out == {
        "pick": {"filter": {"tags": ["is_graspable"]}, "split": "seen"}
    }


def test_validate_slot_overlays_distractors_key_allowed():
    overlay = {"distractors": {"split": "unseen_category"}}
    out = LayoutBuilder._validate_slot_overlays(overlay, {"pick"})
    assert out == {"distractors": {"filter": {}, "split": "unseen_category"}}


def test_validate_slot_overlays_unknown_key_raises():
    overlay = {"not_a_slot": {"filter": {"tags": ["is_graspable"]}}}
    with pytest.raises(LayoutValidationError, match="unknown role class"):
        LayoutBuilder._validate_slot_overlays(overlay, {"pick"})


def test_validate_slot_overlays_distractor_n_rejected_with_hint():
    overlay = {"distractor_0": {"filter": {"tags": ["is_graspable"]}}}
    with pytest.raises(LayoutValidationError, match="'distractors'"):
        LayoutBuilder._validate_slot_overlays(overlay, {"pick"})


def test_validate_slot_overlays_extra_entry_key_raises():
    overlay = {"pick": {"filter": {}, "prim_name": "x"}}
    with pytest.raises(LayoutValidationError, match="prim_name"):
        LayoutBuilder._validate_slot_overlays(overlay, {"pick"})


def test_validate_slot_overlays_empty_entry_raises():
    overlay = {"pick": {}}
    with pytest.raises(LayoutValidationError, match="'filter' and/or 'split'"):
        LayoutBuilder._validate_slot_overlays(overlay, {"pick"})


def test_validate_slot_overlays_category_inside_filter_raises():
    overlay = {"pick": {"filter": {"category": "peach"}}}
    with pytest.raises(LayoutValidationError, match="category"):
        LayoutBuilder._validate_slot_overlays(overlay, {"pick"})


def test_validate_slot_overlays_filter_not_mapping_raises():
    overlay = {"pick": {"filter": ["is_graspable"]}}
    with pytest.raises(
        LayoutValidationError, match="filter must be a mapping"
    ):
        LayoutBuilder._validate_slot_overlays(overlay, {"pick"})


def test_validate_slot_overlays_entry_not_mapping_raises():
    overlay = {"pick": "is_graspable"}
    with pytest.raises(LayoutValidationError, match="must be a mapping"):
        LayoutBuilder._validate_slot_overlays(overlay, {"pick"})  # type: ignore[arg-type]


def test_validate_slot_overlays_split_not_string_raises():
    overlay = {"pick": {"split": ["seen"]}}
    with pytest.raises(
        LayoutValidationError, match="split must be a non-empty string"
    ):
        LayoutBuilder._validate_slot_overlays(overlay, {"pick"})


def test_validate_slot_overlays_multi_key():
    overlay = {
        "pick": {"filter": {"tags": ["is_graspable"]}},
        "place": {"filter": {"color": "red"}, "split": "seen"},
    }
    out = LayoutBuilder._validate_slot_overlays(overlay, {"pick", "place"})
    assert out == {
        "pick": {"filter": {"tags": ["is_graspable"]}, "split": None},
        "place": {"filter": {"color": "red"}, "split": "seen"},
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
    with pytest.raises(LayoutValidationError, match="unknown role class"):
        LayoutBuilder.build(
            seq,
            _resolver(),
            {"src": "pick"},
            slot_filters={"nonexistent": {"filter": {"tags": ["x"]}}},
        )


def test_build_split_forwarded_to_resolver_entry():
    seq = _seq({"src": "bread"})
    resolver = _resolver()
    LayoutBuilder.build(
        seq,
        resolver,
        {"src": "pick"},
        slot_filters={
            "pick": {"filter": {"tags": ["is_graspable"]}, "split": "seen"}
        },
    )
    asset_configs = resolver.resolve.call_args[0][0]
    assert asset_configs["pick"]["split"] == "seen"
    assert asset_configs["pick"]["filter"] == {
        "category": "bread",
        "tags": ["is_graspable"],
    }


def test_build_no_split_key_when_overlay_has_none():
    seq = _seq({"src": "bread"})
    resolver = _resolver()
    LayoutBuilder.build(
        seq,
        resolver,
        {"src": "pick"},
        slot_filters={"pick": {"filter": {"tags": ["is_graspable"]}}},
    )
    asset_configs = resolver.resolve.call_args[0][0]
    assert "split" not in asset_configs["pick"]


def test_build_distractors_overlay_broadcasts_to_all_auto_slots():
    seq = _seq({"src": "bread", "ref": "gum", "extra": "mug"})
    resolver = _resolver()
    LayoutBuilder.build(
        seq,
        resolver,
        {"src": "pick"},
        slot_filters={
            "distractors": {
                "filter": {"tags": ["is_graspable"]},
                "split": "seen",
            }
        },
    )
    asset_configs = resolver.resolve.call_args[0][0]
    for slot in ("distractor_0", "distractor_1"):
        assert asset_configs[slot]["filter"]["tags"] == ["is_graspable"]
        assert asset_configs[slot]["split"] == "seen"
    assert "split" not in asset_configs["pick"]


def test_build_distractors_overlay_unused_when_no_auto_slots():
    seq = _seq({"src": "bread"})
    resolver = _resolver()
    LayoutBuilder.build(
        seq,
        resolver,
        {"src": "pick"},
        slot_filters={"distractors": {"split": "seen"}},
    )
    asset_configs = resolver.resolve.call_args[0][0]
    assert set(asset_configs) == {"pick"}
    assert "split" not in asset_configs["pick"]


def test_build_distractors_overlay_does_not_touch_named_slots():
    seq = _seq({"src": "bread", "ref": "gum"})
    resolver = _resolver()
    LayoutBuilder.build(
        seq,
        resolver,
        {"src": "pick", "ref": "anchor"},
        slot_filters={"distractors": {"split": "seen"}},
    )
    asset_configs = resolver.resolve.call_args[0][0]
    assert set(asset_configs) == {"pick", "anchor"}
    assert "split" not in asset_configs["anchor"]


def test_build_distractor_n_overlay_key_rejected():
    seq = _seq({"src": "bread", "ref": "gum"})
    with pytest.raises(LayoutValidationError, match="'distractors'"):
        LayoutBuilder.build(
            seq,
            _resolver(),
            {"src": "pick"},
            slot_filters={"distractor_0": {"filter": {"tags": ["x"]}}},
        )
