# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""PoolResetTermCfg behavioural tests (mocked scene + alias state)."""

from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from robo_orchard_sim.ext.envs.managers.events.pool_reset import (
    PoolResetTermCfg,
    PoolSlot,
    pool_reset,
    sample_pose_with_aabb_separation,
)
from robo_orchard_sim.ext.envs.managers.events.pose_reset import (
    _CROSS_GROUP_CACHE,
)
from robo_orchard_sim.ext.models.scenes.pool_alias_state import PoolAliasState


def _make_env(slot_role_ids, teleport_recorder=None):
    env = MagicMock()
    env.cfg = MagicMock(seed=0)
    env._sim_step_counter = 0
    env.device = "cpu"
    env.scene = MagicMock()
    env.scene.env_origins = torch.zeros((1, 3))

    def make_entity(name):
        ent = MagicMock()
        ent.cfg = SimpleNamespace(aabb_z_min=None, prim_path=None)
        if teleport_recorder is not None:
            ent.write_root_pose_to_sim.side_effect = (
                lambda pose, env_ids: teleport_recorder.append(
                    (name, pose[0, :3].clone())
                )
            )
        return ent

    env.scene.__getitem__.side_effect = make_entity

    state = PoolAliasState()
    for rid in slot_role_ids:
        state.register_pool(rid)
    env.pool_alias_state = state
    return env


def _cfg(slots, **kwargs):
    base = dict(
        slots=slots,
        pose_range={"x": (0.1, 0.4), "y": (0.1, 0.4), "z": (0.5, 0.6)},
        min_separation=0.0,
    )
    base.update(kwargs)
    return PoolResetTermCfg(**base)


def test_pool_reset_assigns_distinct_actives_per_pool():
    """Slots sharing a pool get distinct actives; each slot binds to one."""
    env = _make_env(["pick_object", "distractor_0", "distractor_1"])
    members = [f"d_{i}" for i in range(5)]
    pool_reset(
        env,
        env_ids=[0],
        cfg=_cfg(
            [
                PoolSlot("pick_object", [f"pick_{i}" for i in range(3)]),
                PoolSlot("distractor_0", members),
                PoolSlot("distractor_1", members),
            ]
        ),
    )
    pick = env.pool_alias_state.resolve("pick_object")
    d0 = env.pool_alias_state.resolve("distractor_0")
    d1 = env.pool_alias_state.resolve("distractor_1")
    assert pick.startswith("pick_")
    assert d0 in members and d1 in members and d0 != d1


def test_pool_reset_clear_write_and_seed_from_shared_cache():
    """Default clear=True wipes prior entries; pool publishes new active."""
    _CROSS_GROUP_CACHE.clear()
    _CROSS_GROUP_CACHE.setdefault("g", {}).setdefault(0, []).append(
        ("stale", (0.1, 0.1), (0.05, 0.05), None, None)
    )
    env = _make_env(["pick_object"])
    pool_reset(
        env,
        env_ids=[0],
        cfg=_cfg(
            [PoolSlot("pick_object", [f"p_{i}" for i in range(3)])],
            group_key="g",
        ),
    )
    cache = _CROSS_GROUP_CACHE["g"][0]
    assert all(e[0] != "stale" for e in cache)
    assert len(cache) == 1
    # Layout: (name, (cx, cy), (hx, hy), hz, top_z)
    assert cache[0][0].startswith("p_")
    assert 0.1 <= cache[0][1][0] <= 0.4 and 0.1 <= cache[0][1][1] <= 0.4


def test_pool_reset_avoids_aabb_overlap_with_cached_entries():
    """When clear=False, sampler respects prior cache entries."""
    _CROSS_GROUP_CACHE.clear()
    _CROSS_GROUP_CACHE.setdefault("g", {}).setdefault(0, []).append(
        ("table", (0.15, 0.15), (0.05, 0.05), 0.5, 0.05)
    )
    env = _make_env(["pick_object"])
    pool_reset(
        env,
        env_ids=[0],
        cfg=PoolResetTermCfg(
            slots=[PoolSlot("pick_object", [f"p_{i}" for i in range(3)])],
            pose_range={"x": (0.0, 0.5), "y": (0.0, 0.5), "z": (0.5, 0.6)},
            min_separation=0.10,
            group_key="g",
            clear_cross_group_cache=False,
        ),
    )
    active = env.pool_alias_state.resolve("pick_object")
    entry = next(e for e in _CROSS_GROUP_CACHE["g"][0] if e[0] == active)
    threshold = 0.05 + 0.01 + 0.10  # ohx + hx_fallback + min_sep
    assert (
        abs(entry[1][0] - 0.15) >= threshold - 1e-6
        or abs(entry[1][1] - 0.15) >= threshold - 1e-6
    )


def test_pool_reset_stows_non_actives_and_preserves_slot_order():
    """Non-actives go to z=-50; slots placed in declaration order."""
    teleports: list[tuple[str, torch.Tensor]] = []
    env = _make_env(
        ["place_object", "pick_object"], teleport_recorder=teleports
    )
    pool_reset(
        env,
        env_ids=[0],
        cfg=_cfg(
            [
                PoolSlot("place_object", [f"pl_{i}" for i in range(3)]),
                PoolSlot("pick_object", [f"pi_{i}" for i in range(3)]),
            ]
        ),
    )

    # Stowed members are at z=-50.
    stowed = [name for name, pose in teleports if pose[2].item() == -50.0]
    actives = [name for name, pose in teleports if pose[2].item() != -50.0]
    assert len(stowed) == 4  # 2+2 non-actives across two pools
    place_active = env.pool_alias_state.resolve("place_object")
    pick_active = env.pool_alias_state.resolve("pick_object")
    # Active teleports happen in slot declaration order: place before pick.
    assert actives.index(place_active) < actives.index(pick_active)


def test_sample_pose_clamps_z_up_for_deep_asset():
    """Sampled z lifts so a deep asset's AABB bottom clears the plane."""
    rng = torch.Generator().manual_seed(0)
    pose_range = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.02, 0.02)}
    pose = sample_pose_with_aabb_separation(
        pose_range=pose_range,
        min_separation=-1.0,
        candidate_extents=(0.05, 0.05, -0.10),
        already_placed=[],
        max_retries=1,
        rng=rng,
    )
    # sampled z=0.02; floor = 0.005 - (-0.10) = 0.105; clamp raises to 0.105
    assert pose[2].item() == pytest.approx(0.105, abs=1e-6)


def test_sample_pose_z_already_above_floor_keeps_sampled_z():
    """Small flat asset with z_min near 0 should keep its sampled z."""
    rng = torch.Generator().manual_seed(0)
    pose_range = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.02, 0.02)}
    pose = sample_pose_with_aabb_separation(
        pose_range=pose_range,
        min_separation=-1.0,
        candidate_extents=(0.05, 0.05, -0.01),
        already_placed=[],
        max_retries=1,
        rng=rng,
    )
    assert pose[2].item() == pytest.approx(0.02, abs=1e-6)


def test_sample_pose_skips_clamp_when_z_min_unknown():
    """z_min None (no registry AABB) leaves the sampled z untouched."""
    rng = torch.Generator().manual_seed(0)
    pose_range = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.02, 0.02)}
    pose = sample_pose_with_aabb_separation(
        pose_range=pose_range,
        min_separation=-1.0,
        candidate_extents=(0.05, 0.05, None),
        already_placed=[],
        max_retries=1,
        rng=rng,
    )
    assert pose[2].item() == pytest.approx(0.02, abs=1e-6)
