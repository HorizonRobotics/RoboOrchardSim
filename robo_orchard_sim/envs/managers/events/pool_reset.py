# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""Per-episode pool reset event term."""

from __future__ import annotations
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from robo_orchard_core.envs.manager_based_env import ResetEvent
from robo_orchard_core.envs.managers.events.event_term import (
    EventTermBase,
    EventTermBaseCfg,
)

from robo_orchard_sim.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg as LabSceneEntityCfg,
)
from robo_orchard_sim.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.envs.managers.events.pose_reset import _CROSS_GROUP_CACHE
from robo_orchard_sim.utils.config import ClassType_co
from robo_orchard_sim.utils.usd import get_prim_aabb

if TYPE_CHECKING:
    pass

__all__ = [
    "PoolSlot",
    "PoolResetTerm",
    "PoolResetTermCfg",
    "pool_reset",
]

logger = logging.getLogger(__name__)

_POOL_FALLBACK_EXTENT = (0.01, 0.01)
"""Used when a member's USD AABB cannot be resolved (mocked test envs)."""

_PlacementEntry = tuple[tuple[float, float, float], tuple[float, float]]
"""(center_xyz, half_extent_xy) describing one placed AABB."""


@dataclass
class PoolSlot:
    """One alias slot bound to a candidate-member list."""

    role_id: str
    members: list[str]


def _sample_pose_with_aabb_separation(
    pose_range: dict[str, tuple[float, float]],
    min_separation: float,
    candidate_extents: tuple[float, float],
    already_placed: list[_PlacementEntry],
    max_retries: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """Sample one (x, y, z) pose retrying until the AABB gap holds.

    A candidate is accepted iff for every placed AABB the gap on at least
    one axis is at least ``min_separation``::

        |dx| >= hx + ohx + min_separation  OR
        |dy| >= hy + ohy + min_separation

    On retry exhaustion logs WARNING and returns the last attempt without
    separation guarantee.
    """
    x_lo, x_hi = pose_range.get("x", (0.0, 0.0))
    y_lo, y_hi = pose_range.get("y", (0.0, 0.0))
    z_lo, z_hi = pose_range.get("z", (0.0, 0.0))

    hx, hy = candidate_extents
    last: torch.Tensor | None = None
    for _ in range(max_retries):
        x = torch.empty(1).uniform_(x_lo, x_hi, generator=rng).item()
        y = torch.empty(1).uniform_(y_lo, y_hi, generator=rng).item()
        z = torch.empty(1).uniform_(z_lo, z_hi, generator=rng).item()
        candidate = torch.tensor([x, y, z])
        last = candidate
        if min_separation < 0.0 or not already_placed:
            return candidate
        ok = True
        for (cx, cy, _cz), (ohx, ohy) in already_placed:
            dx = abs(x - cx)
            dy = abs(y - cy)
            if (
                dx < hx + ohx + min_separation
                and dy < hy + ohy + min_separation
            ):
                ok = False
                break
        if ok:
            return candidate
    logger.warning(
        "pool_reset: pose retry exhausted after %d attempts; "
        "returning last candidate without separation guarantee",
        max_retries,
    )
    assert last is not None
    return last


def _compute_member_extents(
    env, members: Sequence[str]
) -> dict[str, tuple[float, float]]:
    """Per-member XY half-extents from USD AABB; fallback if unavailable."""
    extents: dict[str, tuple[float, float]] = {}
    stage = getattr(env.scene, "stage", None)
    for m in members:
        hx_hy = _POOL_FALLBACK_EXTENT
        if stage is not None:
            try:
                asset = env.scene[m]
                cfg = getattr(asset, "cfg", None)
                prim_path = getattr(cfg, "prim_path", None)
                if isinstance(prim_path, str):
                    prim_path = prim_path.replace("env_.*", "env_0")
                    aabb = get_prim_aabb(stage, prim_path)
                    if aabb is not None:
                        (x_max, x_min), (y_max, y_min), _z = aabb
                        hx_hy = (
                            abs(x_max - x_min) * 0.5,
                            abs(y_max - y_min) * 0.5,
                        )
            except Exception:
                pass
        extents[m] = hx_hy
    return extents


def _seed_already_placed_from_cache(
    group_key: str | None,
    env_id: int,
) -> list[_PlacementEntry]:
    """Return [(center, extents)] from the shared cache for env_id.

    Cache entries follow ``pose_reset._CacheEntry`` layout:
    ``(name, (cx, cy), (hx, hy), hz_or_None, top_z_or_None)``.
    """
    if not group_key:
        return []
    cache = _CROSS_GROUP_CACHE.get(group_key, {}).get(int(env_id), [])
    if not cache:
        return []
    return [((entry[1][0], entry[1][1], 0.0), entry[2]) for entry in cache]


def _publish_to_cache(
    group_key: str | None,
    env_id: int,
    name: str,
    pose: torch.Tensor,
    extents: tuple[float, float],
) -> None:
    """Append a pool active member's placement using pose_reset layout."""
    if not group_key:
        return
    entry = (
        name,
        (float(pose[0]), float(pose[1])),
        extents,
        None,
        None,
    )
    _CROSS_GROUP_CACHE.setdefault(group_key, {}).setdefault(
        int(env_id), []
    ).append(entry)


def _derive_rng(seed: int, episode: int, env_id: int) -> np.random.Generator:
    ss = np.random.SeedSequence([seed, episode, env_id])
    return np.random.default_rng(ss)


def _group_slots_by_pool(
    slots: Sequence[PoolSlot],
) -> list[list[PoolSlot]]:
    """Group slots that share the same members list (same underlying pool)."""
    groups: list[list[PoolSlot]] = []
    seen: dict[tuple, list[PoolSlot]] = {}
    for s in slots:
        key = tuple(s.members)
        if key in seen:
            seen[key].append(s)
        else:
            new_group = [s]
            seen[key] = new_group
            groups.append(new_group)
    return groups


def _teleport_member(
    env,
    name: str,
    pose_xyz: torch.Tensor,
    env_id: int,
    zero_velocity: bool = True,
    add_env_origin: bool = True,
    quat_wxyz: torch.Tensor | None = None,
) -> None:
    """Teleport ``name`` in ``env_id`` to ``pose_xyz`` with optional rotation.

    When ``add_env_origin`` is True (default; active members), the pose is
    treated as env-local and offset by ``env.scene.env_origins[env_id]``.
    When False (members on the inactive-pool storage), absolute world coords.
    ``quat_wxyz`` defaults to identity ``(1, 0, 0, 0)`` to preserve the
    behavior of every existing caller.
    """
    entity = env.scene[name]
    device = getattr(env, "device", "cpu")
    pose_xyz = pose_xyz.to(device)
    if add_env_origin:
        env_origins = env.scene.env_origins
        pose_xyz = pose_xyz + env_origins[env_id]
    if quat_wxyz is None:
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    else:
        quat = quat_wxyz.to(device)
    full_pose = torch.cat([pose_xyz, quat]).unsqueeze(0)
    env_ids_t = torch.tensor([env_id], device=device)
    entity.write_root_pose_to_sim(full_pose, env_ids=env_ids_t)
    if zero_velocity:
        zeros = torch.zeros((1, 6), device=device)
        entity.write_root_velocity_to_sim(zeros, env_ids=env_ids_t)


def _stow_member(env, name, slot, env_id, cfg) -> None:
    """Move an inactive member to the inactive-pool storage (z=-50)."""
    member_idx = slot.members.index(name)
    px = cfg.storage_origin[0] + member_idx * cfg.storage_spread[0]
    py = cfg.storage_origin[1] + member_idx * cfg.storage_spread[1]
    pz = cfg.storage_origin[2] + member_idx * cfg.storage_spread[2]
    pose = torch.tensor([px, py, pz])
    _teleport_member(
        env,
        name,
        pose,
        env_id,
        zero_velocity=True,
        add_env_origin=False,
    )


def _run_pool_reset_for_env_ids(env, env_ids, cfg: "PoolResetTermCfg") -> None:
    """Core per-env pool reset loop shared by Term and standalone fn."""
    if cfg.clear_cross_group_cache and cfg.group_key is not None:
        _CROSS_GROUP_CACHE.setdefault(cfg.group_key, {}).clear()
    for env_id in env_ids:
        rng = _derive_rng(
            int(env.cfg.seed or 0),
            getattr(env, "_sim_step_counter", 0),
            int(env_id),
        )
        torch_seed = int(rng.integers(0, 2**31 - 1))
        torch_rng = torch.Generator().manual_seed(torch_seed)
        # Seed already_placed from the shared cross-event cache so pool
        # actives do not land on top of pose_reset-managed assets.
        already_placed = _seed_already_placed_from_cache(
            cfg.group_key,
            int(env_id),
        )

        for group in _group_slots_by_pool(cfg.slots):
            n_active = len(group)
            members = group[0].members
            if n_active > len(members):
                raise RuntimeError(
                    f"pool group needs {n_active} active but pool has "
                    f"only {len(members)} members"
                )
            active_idxs = rng.choice(
                len(members), size=n_active, replace=False
            )
            active_names = [members[int(i)] for i in active_idxs]

            # Stow every non-active member in the inactive-pool storage.
            # Covers first-episode init (no aliases yet, USD spawn poses
            # still on the table) and later episodes where active rotates.
            for member in members:
                if member not in active_names:
                    _stow_member(env, member, group[0], env_id, cfg)

            # Pre-compute AABB extents for active members.
            extents = _compute_member_extents(env, active_names)

            # Sample valid poses for the new active set.
            poses = []
            for name in active_names:
                hx_hy = extents[name]
                pose = _sample_pose_with_aabb_separation(
                    pose_range=cfg.pose_range,
                    min_separation=cfg.min_separation,
                    candidate_extents=hx_hy,
                    already_placed=already_placed,
                    max_retries=cfg.max_retries,
                    rng=torch_rng,
                )
                poses.append(pose)
                already_placed.append(
                    (
                        (float(pose[0]), float(pose[1]), float(pose[2])),
                        hx_hy,
                    )
                )

            # Teleport actives and publish to shared placement cache.
            for name, pose in zip(active_names, poses, strict=False):
                _teleport_member(env, name, pose, env_id)
                _publish_to_cache(
                    cfg.group_key,
                    int(env_id),
                    name,
                    pose,
                    extents[name],
                )

            # Bind each slot's alias to its active.
            for slot, active_name in zip(group, active_names, strict=False):
                env.pool_alias_state.set_active(slot.role_id, active_name)


class PoolResetTerm(
    EventTermBase[ResetEvent, IsaacEnvType_co, "PoolResetTermCfg"],
):
    """Event term that activates one pool member per slot each episode."""

    def __init__(self, cfg: "PoolResetTermCfg", env: IsaacEnvType_co):
        """Initialise the term, capturing cfg and env."""
        super().__init__(cfg, env)
        self._cfg = cfg
        self._env = env
        state = env.pool_alias_state
        if state is not None:
            for slot in cfg.slots:
                if not state.has_pool(slot.role_id):
                    state.register_pool(slot.role_id)

    def __call__(self, event_msg: ResetEvent) -> None:
        """Run pool reset for the env_ids carried by event_msg."""
        if not hasattr(event_msg, "env_ids"):
            env_ids = torch.arange(self._env.num_envs).to(self._env.device)
        elif event_msg.env_ids is None:
            env_ids = torch.arange(self._env.num_envs).to(self._env.device)
        else:
            env_ids = torch.tensor(event_msg.env_ids).to(self._env.device)

        _run_pool_reset_for_env_ids(self._env, env_ids, self._cfg)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """No per-reset state to clear."""
        pass


class PoolResetTermCfg(EventTermBaseCfg[PoolResetTerm, LabSceneEntityCfg]):
    """Configuration for the pool reset event term."""

    class_type: ClassType_co[PoolResetTerm] = PoolResetTerm

    trigger_topic: str = "reset"
    """Event topic that triggers reset; defaults to ``reset``."""

    slots: list[PoolSlot]
    """Alias slots, each binding a role_id to a list of candidate members."""

    pose_range: dict[str, tuple[float, float]]
    """Pose sampling range keyed by axis (``x``, ``y``, ``z``)."""

    min_separation: float = 0.0
    """Minimum XY separation between placed members."""

    max_retries: int = 256
    """Maximum rejection-sampling attempts per member."""

    storage_origin: tuple[float, float, float] = (0.0, 0.0, -50.0)
    """World-space origin for stowing inactive pool members."""

    storage_spread: tuple[float, float, float] = (0.5, 0.5, 0.0)
    """Per-member offset applied to storage_origin to avoid stacking."""

    group_key: str | None = None
    """Shared placement-cache key with pose_reset events for non-overlap."""

    clear_cross_group_cache: bool = True
    """Clear the shared placement cache before sampling.

    The first reset event each episode should clear; later events in the
    same episode read+write to accumulate placements for collision checks.
    """

    model_config = {"arbitrary_types_allowed": True}


def pool_reset(env, env_ids, cfg: PoolResetTermCfg) -> None:
    """Per-env per-reset pool reset entry point (backward compat)."""
    _run_pool_reset_for_env_ids(env, env_ids, cfg)
