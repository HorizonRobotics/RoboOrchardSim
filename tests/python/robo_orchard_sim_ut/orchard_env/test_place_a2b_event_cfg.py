# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""Tests for PlaceA2BTask.get_event_cfg pool/classic emission."""

from __future__ import annotations

from robo_orchard_sim.orchard_env.assets.object_spec import RigidObjectSpec
from robo_orchard_sim.orchard_env.assets.pool_spec import PoolSpec
from robo_orchard_sim.orchard_env.task_templates.place_a2b_task import (
    PlaceA2BTask,
    PlaceA2BTaskAssets,
    PlaceA2BTaskParams,
)
from robo_orchard_sim.orchard_env.task_templates.task_params import (
    PoseRangeConfig,
    TaskLightResetConfig,
    TaskPoseResetConfig,
    TaskTextureResetConfig,
)


def _obj(name: str) -> RigidObjectSpec:
    return RigidObjectSpec(name=name, usd_path=f"/tmp/{name}.usd")


def _pool(role: str, n: int, active: int = 1) -> PoolSpec:
    return PoolSpec(
        role_id=role,
        members=[_obj(f"{role}_pool_{i}") for i in range(n)],
        active_count=active,
    )


def test_get_event_cfg_classic_emits_only_random_pose():
    cfg = PlaceA2BTask(
        assets=PlaceA2BTaskAssets(
            pick=_obj("pick_object"), place=_obj("place_object")
        )
    ).get_event_cfg()
    assert "pool_reset_event" not in cfg.terms
    assert "random_pose_event" in cfg.terms


def test_get_event_cfg_distractor_pool_emits_active_count_slots():
    """Distractor PoolSpec emits active_count slots sharing one member list."""
    cfg = PlaceA2BTask(
        assets=PlaceA2BTaskAssets(
            pick=_pool("pick_object", 3),
            place=_obj("place_object"),
            distractors=_pool("distractor", 6, active=3),
        )
    ).get_event_cfg()
    pool_term = cfg.terms["pool_reset_event"]
    role_ids = [s.role_id for s in pool_term.slots]
    # pick + 3 distractor slots; pose term routes the classic place_object.
    assert "pick_object" in role_ids
    assert {"distractor_0", "distractor_1", "distractor_2"}.issubset(role_ids)
    assert "distractor_3" not in role_ids
    dist = [s for s in pool_term.slots if s.role_id.startswith("distractor_")]
    for s in dist[1:]:
        assert s.members == dist[0].members
    # Mixed routing: classic place_object goes to random_pose_event with
    # namespace prefix; pool roles do not appear in pose_actor names.
    pose_actor_names = {
        ac.name for ac in cfg.terms["random_pose_event"].asset_cfgs
    }
    assert "objects/place_object" in pose_actor_names
    assert "pick_object" not in pose_actor_names


def test_get_event_cfg_enabled_light_and_texture_adds_terms():
    cfg = PlaceA2BTask(
        assets=PlaceA2BTaskAssets(
            pick=_obj("pick_object"),
            place=_obj("place_object"),
        ),
        params=PlaceA2BTaskParams(
            light_reset=TaskLightResetConfig(
                enabled=True,
                asset_names=["background/dis_light"],
                distant_light={"asset_name": "dis_light"},
                randomize_intensity=True,
                intensity_range={"range": (1000.0, 5000.0)},
            ),
            texture_reset=TaskTextureResetConfig(
                enabled=True,
                asset_names=["background/table"],
            ),
        ),
    ).get_event_cfg()

    assert "light_reset_event" in cfg.terms
    assert "texture_reset_event" in cfg.terms


def test_place_a2b_task_params_legacy_pose_fields_are_ignored():
    params = PlaceA2BTaskParams(
        mode="drop",
        min_separation=0.07,
        pose_range=PoseRangeConfig(
            x=(0.1, 0.2),
            y=(-0.2, 0.4),
        ),
    )

    assert params.pose_reset == TaskPoseResetConfig()
    assert not hasattr(params, "mode")
    assert not hasattr(params, "min_separation")
    assert not hasattr(params, "pose_range")
