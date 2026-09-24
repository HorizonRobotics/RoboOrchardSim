# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""SpatialTask build path: task YAML -> OrchardEnv, no layout involved.

Both shipped variants come through the same task; what separates them
is only what their YAML says. The tests below cover the shared path
once and then each variant's own claim about its targets.
"""

from __future__ import annotations
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from robo_orchard_sim.benchmark.manipulation.spatial_relation.spatial_relation_env import (  # noqa: E501
    SpatialDirectionsTaskDefinition,
    SpatialNearFarTaskDefinition,
    SpatialRelationTaskDefinitionBase,
)
from robo_orchard_sim.orchard_env.assets.object_spec import RigidObjectSpec
from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets
from robo_orchard_sim.orchard_env.task_templates.spatial_task import (
    SpatialTask,
    SpatialTaskParams,
)
from robo_orchard_sim.task_components.role_registry import (
    RoleRegistry,
    TargetRef,
)
from robo_orchard_sim.task_components.validators.context import (
    ValidatorContext,
)

_APPLE_UUID = "05c8e2d73c5b5e319c1951f4f860bcf7"
_TOMATO_UUID = "3216a55b93cf5faba0e8238b96ce9c61"

_FOUR_WAY_KEYS = (
    "ref",
    "pick_front",
    "pick_left",
    "pick_behind",
    "pick_right",
)
_NEAR_FAR_KEYS = ("ref", "pick_near", "pick_far", "clutter_0", "clutter_1")
_REFERENCE_NAMES = {
    SpatialDirectionsTaskDefinition: "objects/ref_0",
    SpatialNearFarTaskDefinition: "objects/ref",
}

_FOUR_WAY_CANDIDATES = {
    "pick_front": {
        "relation": "front_of",
        "axis": 0.0,
        "radius": (0.15, 0.30),
    },
    "pick_left": {
        "relation": "left_of",
        "axis": 90.0,
        "radius": (0.15, 0.30),
    },
    "pick_behind": {
        "relation": "behind",
        "axis": 180.0,
        "radius": (0.15, 0.30),
    },
    "pick_right": {
        "relation": "right_of",
        "axis": 270.0,
        "radius": (0.15, 0.30),
    },
}
_NEAR_FAR_CANDIDATES = {
    "pick_near": {"relation": "near", "radius": (0.11, 0.16)},
    "pick_far": {"relation": "far", "radius": (0.30, 0.38)},
    "clutter_0": {"radius": (0.20, 0.26)},
    "clutter_1": {"radius": (0.20, 0.26)},
}


def _spec(tmp_path: Path, *, name: str, category: str, uuid: str):
    """A spec whose caption is on disk, so instructions can render."""
    caption = tmp_path / f"{uuid}_{name}.json"
    caption.write_text(
        json.dumps({"uuid": uuid, "raw": category, "seen": [category]}),
        encoding="utf-8",
    )
    return RigidObjectSpec(
        name=name,
        usd_path=f"/d/{category}.usd",
        caption_path=str(caption),
        uuid=uuid,
        category=category,
    )


def _make_resolver(tmp_path: Path):
    """Stand in for the resolver's target, clone and group outputs.

    Mirrors the real one closely enough for the build path: a slot
    carrying a filter draws an asset of its own, and a slot naming
    ``same_as`` receives whatever its source drew. The real resolver
    picks by tag and keeps drawing slots off each other's assets, so the
    reference and the candidate that draws come out different here too.
    """
    uuid_by_category = {"apple": _APPLE_UUID, "tomato": _TOMATO_UUID}

    def resolve(asset_configs):
        drawn: dict[str, str] = {}
        for key, entry in asset_configs.items():
            if "same_as" not in entry:
                assert "filter" in entry, f"{key} draws without a filter"
                drawn[key] = "apple" if key == "ref" else "tomato"

        resolved = {}
        for key, entry in asset_configs.items():
            source = entry.get("same_as", key)
            category = drawn[source]
            if "anchor" in entry:
                prefix = entry.get("prim_name_prefix", key)
                resolved[key] = [
                    _spec(
                        tmp_path,
                        name=f"{prefix}_{index}",
                        category=category,
                        uuid=uuid_by_category[category],
                    )
                    for index in range(entry["min_count"])
                ]
                continue
            resolved[key] = _spec(
                tmp_path,
                name=entry["prim_name"],
                category=category,
                uuid=uuid_by_category[category],
            )
        return resolved

    resolver = MagicMock()
    resolver.resolve.side_effect = resolve
    return resolver


@contextmanager
def _patched_build(cls):
    with (
        patch.object(cls, "resolve_scene", return_value=MagicMock()),
        patch.object(cls, "resolve_embodiment", return_value=MagicMock()),
        patch("robo_orchard_sim.orchard_env.orchard_env.OrchardEnv") as orch,
    ):
        yield orch


def _build_task(cls, tmp_path: Path):
    return cls.build(resolver=_make_resolver(tmp_path)).task


def _fake_env(task):
    return SimpleNamespace(
        scene={
            spec.scene_name: SimpleNamespace(cfg=spec.to_isaac_cfg())
            for specs in task.assets.role_candidates.values()
            for spec in specs
        }
    )


def _render_each_candidate(task) -> dict[str, str]:
    """Instruction produced for every candidate, keyed by scene name."""
    registry = RoleRegistry(num_envs=1)
    context = ValidatorContext(robot=None, role_registry=registry)
    env = _fake_env(task)

    rendered = {}
    for target in task.get_role_candidates("pick"):
        registry.bind_one(0, "pick", target)
        rendered[target.scene_name] = task.instruction.render(
            actors=task.build_instruction_context(
                env, actor_description_seed=0, context=context
            )
        )
    return rendered


def _assets(*keys: str) -> TaskAssets:
    return TaskAssets(
        role_candidates={
            key: [RigidObjectSpec(name=key, usd_path=f"/d/{key}.usd")]
            for key in keys
        }
    )


def _params(candidates=None, **overrides) -> SpatialTaskParams:
    pose_reset = {
        "anchor_range": {"x": (0.48, 0.62), "y": (-0.08, 0.08)},
        "workspace": {"x": (0.25, 0.80), "y": (-0.40, 0.40)},
        "candidates": {
            key: dict(value)
            for key, value in (
                candidates if candidates is not None else _NEAR_FAR_CANDIDATES
            ).items()
        },
    }
    pose_reset.update(overrides)
    return SpatialTaskParams(pose_reset=pose_reset)


_VARIANTS = [
    pytest.param(
        SpatialDirectionsTaskDefinition, _FOUR_WAY_KEYS, id="four_way"
    ),
    pytest.param(SpatialNearFarTaskDefinition, _NEAR_FAR_KEYS, id="near_far"),
]


@pytest.mark.parametrize("cls,keys", _VARIANTS)
def test_spatial_task_build_resolves_singleton_and_group_scene_names(
    cls, keys, tmp_path
):
    """Singletons use YAML keys; grouped references use indexed names."""
    task = _build_task(cls, tmp_path)
    expected = {key: [f"objects/{key}"] for key in keys}
    expected["ref"] = [_REFERENCE_NAMES[cls]]
    assert {
        key: [spec.scene_name for spec in task.assets.by_role(key)]
        for key in keys
    } == expected


@pytest.mark.parametrize("cls,keys", _VARIANTS)
def test_build_takes_the_non_layout_path(cls, keys, tmp_path):
    """A spatial-relation env carries no layout builder."""
    del keys
    with _patched_build(cls) as orch:
        cls.build(resolver=_make_resolver(tmp_path))
    assert "layout_builder" not in orch.call_args.kwargs


@pytest.mark.parametrize("cls,keys", _VARIANTS)
def test_spatial_pose_reset_dr_disabled_uses_one_placement_term(
    cls, keys, tmp_path
):
    """A single term places everything, so nothing has to be shared.

    Splitting the objects across terms would mean coordinating their
    placements through a shared cache, where getting the order wrong
    silently stops them avoiding each other.
    """
    task = _build_task(cls, tmp_path)
    task.params.light_reset.enabled = False
    task.params.texture_reset.enabled = False
    terms = task.get_event_cfg().terms

    assert list(terms) == ["pose_reset_event"]
    term = terms["pose_reset_event"]
    assert term.anchor_name == _REFERENCE_NAMES[cls]
    assert not hasattr(term, "group_key")
    assert set(term.radius) == {
        f"objects/{key}" for key in keys if key != "ref"
    }


@pytest.mark.parametrize("cls,keys", _VARIANTS)
def test_spatial_config_candidate_sampling_uses_one_split_aware_draw(
    cls, keys
):
    """Candidates share a draw rather than each pinning a uuid.

    Pinning would also make them identical, but a pinned uuid overrides
    the split, so an evaluation could no longer be pointed at unseen
    assets. Sharing keeps the draw -- and its split -- intact.
    """
    del keys
    asset_configs = cls.resolve_asset_configs()
    drawing = {
        key: entry
        for key, entry in asset_configs.items()
        if key != "ref" and "same_as" not in entry
    }
    cloning = {
        key: entry["same_as"]
        for key, entry in asset_configs.items()
        if "same_as" in entry
    }

    assert not any("uuid" in entry for entry in asset_configs.values()), (
        "a pinned uuid would silently disable the split"
    )
    assert len(drawing) == 1 and set(drawing) == set(cloning.values()), (
        "exactly one candidate should draw; the others copy it"
    )
    assert all("split" in entry for entry in drawing.values())


@pytest.mark.parametrize("cls,keys", _VARIANTS)
def test_candidates_share_one_asset_instance(cls, keys, tmp_path):
    """Candidates look alike, so only placement can tell them apart."""
    del keys
    task = _build_task(cls, tmp_path)
    uuids = {
        spec.uuid
        for key in SpatialTask.candidate_keys(task.assets)
        for spec in task.assets.by_role(key)
    }
    assert uuids == {_TOMATO_UUID}
    assert task.assets.by_role("ref")[0].uuid == _APPLE_UUID


@pytest.mark.parametrize("cls,keys", _VARIANTS)
def test_spatial_task_reference_role_is_excluded_from_pick_candidates(
    cls, keys, tmp_path
):
    """What the target is described against cannot be the target."""
    del keys
    task = _build_task(cls, tmp_path)
    candidates = [t.scene_name for t in task.get_role_candidates("pick")]
    assert _REFERENCE_NAMES[cls] not in candidates
    assert [t.scene_name for t in task.get_role_candidates("ref")] == [
        _REFERENCE_NAMES[cls]
    ]


@pytest.mark.parametrize("cls,keys", _VARIANTS)
def test_pick_candidates_ignore_the_swap_flag(cls, keys, tmp_path):
    """Every candidate describes itself, so swap widens nothing."""
    del keys
    task = _build_task(cls, tmp_path)
    assert task.get_role_candidates(
        "pick", swap=True
    ) == task.get_role_candidates("pick", swap=False)


def test_four_way_describes_each_candidate_by_its_own_direction(tmp_path):
    """The wording follows whichever candidate the episode binds."""
    task = _build_task(SpatialDirectionsTaskDefinition, tmp_path)
    assert _render_each_candidate(task) == {
        "objects/pick_left": "Pick up the tomato to the left of the apple.",
        "objects/pick_right": "Pick up the tomato to the right of the apple.",
        "objects/pick_front": "Pick up the tomato in front of the apple.",
        "objects/pick_behind": "Pick up the tomato behind the apple.",
    }


def test_four_way_pins_each_direction_to_its_own_axis(tmp_path):
    """The axes the config states are the ones the placement uses.

    Directions read outward from the arm, so "in front of" is the axis
    pointing away from it and "behind" the one pointing back.
    """
    task = _build_task(SpatialDirectionsTaskDefinition, tmp_path)
    term = task.get_event_cfg().terms["pose_reset_event"]
    assert term.axis == {
        "objects/pick_front": 0.0,
        "objects/pick_left": 90.0,
        "objects/pick_behind": 180.0,
        "objects/pick_right": 270.0,
    }


def test_the_axis_comes_from_the_config_not_from_the_wording():
    """A stated axis is honoured even where no word implies one.

    Clutter has no relation to infer a direction from, so pinning it is
    only possible if the config is what decides -- which is what lets a
    scene be arranged without every object having to be describable.
    """
    candidates = {
        key: dict(value) for key, value in _NEAR_FAR_CANDIDATES.items()
    }
    candidates["clutter_0"]["axis"] = 180.0

    task = SpatialTask(
        assets=_assets(*_NEAR_FAR_KEYS),
        params=_params(candidates=candidates),
    )
    term = task.get_event_cfg().terms["pose_reset_event"]
    assert term.axis == {"objects/clutter_0": 180.0}


def test_a_direction_word_without_an_axis_fails_the_build():
    """A direction is only true of an object that is held there.

    Left to roam, the object would be dealt a fresh direction every
    episode while the instruction went on calling it the same thing.
    """
    candidates = {
        key: dict(value) for key, value in _FOUR_WAY_CANDIDATES.items()
    }
    del candidates["pick_left"]["axis"]

    with pytest.raises(ValueError, match="states no axis"):
        SpatialTask(
            assets=_assets(*_FOUR_WAY_KEYS),
            params=_params(candidates=candidates),
        )


def test_near_far_describes_each_candidate_by_its_own_distance(tmp_path):
    """The wording follows whichever candidate the episode binds."""
    task = _build_task(SpatialNearFarTaskDefinition, tmp_path)
    assert _render_each_candidate(task) == {
        "objects/pick_near": "Pick up the tomato near the apple.",
        "objects/pick_far": "Pick up the tomato far from the apple.",
    }


def test_near_far_leaves_every_direction_free(tmp_path):
    """Distance words must not pin a direction.

    Holding the direction still would let it identify the target, which
    is exactly what the wording is meant to make a policy read past.
    """
    task = _build_task(SpatialNearFarTaskDefinition, tmp_path)
    term = task.get_event_cfg().terms["pose_reset_event"]
    assert term.axis == {}


def test_near_far_clutter_is_placed_but_never_named(tmp_path):
    """Clutter sits in the scene to be mistaken for the target.

    Nothing in the instruction could pick it out, so it must not be
    bindable -- but it does have to be there.
    """
    task = _build_task(SpatialNearFarTaskDefinition, tmp_path)

    candidates = [t.scene_name for t in task.get_role_candidates("pick")]
    assert candidates == ["objects/pick_far", "objects/pick_near"]
    assert "objects/clutter_0" not in candidates
    assert "objects/clutter_0" in task.assets.all_scene_names()


def test_instruction_context_rejects_an_undescribable_target(tmp_path):
    """Binding clutter would leave the instruction with nothing to say."""
    task = _build_task(SpatialNearFarTaskDefinition, tmp_path)
    registry = RoleRegistry(num_envs=1)
    registry.bind_one(0, "pick", TargetRef("objects/clutter_0"))

    with pytest.raises(KeyError, match="carries no relation"):
        task.build_instruction_context(
            _fake_env(task),
            actor_description_seed=0,
            context=ValidatorContext(robot=None, role_registry=registry),
        )


def test_a_direction_word_contradicting_its_axis_fails_the_build():
    """A pinned axis has to agree with the word describing it."""
    candidates = {
        key: dict(value) for key, value in _FOUR_WAY_CANDIDATES.items()
    }
    candidates["pick_left"]["axis"] = 270.0  # left is 90

    with pytest.raises(ValueError, match="claims 'left_of'"):
        SpatialTask(
            assets=_assets(*_FOUR_WAY_KEYS),
            params=_params(candidates=candidates),
        )


def test_a_distance_word_pinning_a_direction_fails_the_build():
    """Distance words leave the direction free, by definition."""
    candidates = {
        key: dict(value) for key, value in _NEAR_FAR_CANDIDATES.items()
    }
    candidates["pick_near"]["axis"] = 90.0

    with pytest.raises(ValueError, match="described by distance"):
        SpatialTask(
            assets=_assets(*_NEAR_FAR_KEYS),
            params=_params(candidates=candidates),
        )


def test_overlapping_near_and_far_bands_fail_the_build():
    """Bands that meet would let the two candidates cross."""
    candidates = {
        key: dict(value) for key, value in _NEAR_FAR_CANDIDATES.items()
    }
    candidates["pick_far"]["radius"] = (0.17, 0.38)  # near reaches 0.16

    with pytest.raises(ValueError, match="margin"):
        SpatialTask(
            assets=_assets(*_NEAR_FAR_KEYS),
            params=_params(candidates=candidates),
        )


def test_clutter_reaching_past_a_candidate_fails_the_build():
    """Clutter outside the gap could out-rank the object being named."""
    candidates = {
        key: dict(value) for key, value in _NEAR_FAR_CANDIDATES.items()
    }
    candidates["clutter_0"]["radius"] = (0.05, 0.26)  # nearer than near

    with pytest.raises(ValueError, match="outside the"):
        SpatialTask(
            assets=_assets(*_NEAR_FAR_KEYS),
            params=_params(candidates=candidates),
        )


def test_object_without_a_placement_fails_the_build():
    """Everything placed around the reference needs a placement."""
    with pytest.raises(ValueError, match="pose_reset.candidates describes"):
        SpatialTask(
            assets=_assets(*_NEAR_FAR_KEYS, "pick_extra"),
            params=_params(),
        )


def test_placement_without_an_object_fails_the_build():
    """A placement naming an object the YAML never spawned is a typo."""
    with pytest.raises(ValueError, match="pose_reset.candidates describes"):
        SpatialTask(
            assets=_assets("ref", "pick_near", "pick_far", "clutter_0"),
            params=_params(),
        )


def test_no_relation_at_all_fails_the_build():
    """Without a relation no episode could ask for anything."""
    candidates = {
        key: {"radius": value["radius"]}
        for key, value in _NEAR_FAR_CANDIDATES.items()
    }
    with pytest.raises(ValueError, match="no candidate carries a relation"):
        SpatialTask(
            assets=_assets(*_NEAR_FAR_KEYS),
            params=_params(candidates=candidates),
        )


def test_missing_reference_fails_the_build():
    """Without a reference there is nothing to describe against."""
    with pytest.raises(ValueError, match="'ref'"):
        SpatialTask(
            assets=_assets("pick_near", "pick_far", "clutter_0", "clutter_1"),
            params=_params(),
        )


def test_no_candidate_at_all_fails_the_build():
    """The pick role is filled by prefix, not by a role named 'pick'."""
    with pytest.raises(ValueError, match="starts with 'pick'"):
        SpatialTask(
            assets=_assets("ref"),
            params=_params(candidates={}),
        )


def test_both_variants_are_registered():
    """The evaluator finds each task by the name eval configs use."""
    assert SpatialDirectionsTaskDefinition.namespace == "spatial_directions"
    assert SpatialNearFarTaskDefinition.namespace == "spatial_near_far"
    for cls in (
        SpatialDirectionsTaskDefinition,
        SpatialNearFarTaskDefinition,
    ):
        assert issubclass(cls, SpatialRelationTaskDefinitionBase)


@pytest.mark.parametrize("cls,keys", _VARIANTS)
def test_each_config_is_named_after_its_task(cls, keys):
    """An eval config naming only a task_type has to find its yaml.

    The runner looks one up by globbing ``configs/<task_type>.yaml``, so
    a config file named anything else leaves the task unrunnable from an
    eval config -- and the failure surfaces only once the evaluator
    starts, far from the rename that caused it.
    """
    del keys
    config_path = Path(cls.config_path)
    assert config_path.name == f"{cls.namespace}.yaml"
    assert config_path.is_file()


def test_both_variants_share_one_task_class(tmp_path):
    """Changing task is a change of YAML, not of code."""
    four_way = _build_task(SpatialDirectionsTaskDefinition, tmp_path)
    near_far = _build_task(SpatialNearFarTaskDefinition, tmp_path)
    assert type(four_way) is type(near_far) is SpatialTask
