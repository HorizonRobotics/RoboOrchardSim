# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""Deterministic per-episode layout cycling event term."""

from __future__ import annotations
from collections.abc import Sequence

import torch
from robo_orchard_core.envs.manager_based_env import ResetEvent
from robo_orchard_core.envs.managers.events.event_term import (
    EventTermBase,
    EventTermBaseCfg,
)

from robo_orchard_sim.ext.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg as LabSceneEntityCfg,
)
from robo_orchard_sim.ext.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.ext.envs.managers.events.pool_reset import (
    PoolSlot,
    _stow_member,
    _teleport_member,
)
from robo_orchard_sim.orchard_env.layout.loader import LayoutSequence
from robo_orchard_sim.utils.config import ClassType_co

__all__ = [
    "LayoutResetTerm",
    "LayoutResetTermCfg",
]


class LayoutResetTerm(
    EventTermBase[ResetEvent, IsaacEnvType_co, "LayoutResetTermCfg"],
):
    """Cycle through `cfg.layouts.entries` one per reset, deterministically."""

    def __init__(
        self, cfg: "LayoutResetTermCfg", env: IsaacEnvType_co
    ) -> None:
        super().__init__(cfg, env)
        assert env.num_envs == 1, (
            f"LayoutResetTerm requires num_envs == 1; got {env.num_envs}"
        )
        self._cfg = cfg
        self._env = env
        self._count = 0
        needs_pool = any(
            len(members) >= 2
            for members in cfg.role_member_by_category.values()
        )
        if needs_pool and env.pool_alias_state is None:
            raise RuntimeError(
                "LayoutResetTerm requires env.pool_alias_state for "
                "multi-category roles; got None"
            )
        if env.pool_alias_state is not None:
            for role, members in cfg.role_member_by_category.items():
                if len(members) >= 2 and not env.pool_alias_state.has_pool(
                    role
                ):
                    env.pool_alias_state.register_pool(role)

    def __call__(self, event_msg: ResetEvent) -> None:
        idx = self._count
        n = len(self._cfg.layouts.entries)
        if idx >= n:
            raise RuntimeError(
                f"LayoutResetTerm exhausted: requested episode {idx} "
                f"but only {n} layout entries exist; outer caller should "
                f"stop at task.num_episodes"
            )
        layout = self._cfg.layouts.entries[idx]
        env_id = 0
        for role, obj in layout.objects.items():
            members = self._cfg.role_member_by_category[role]
            active_name = members[obj.category]
            if len(members) >= 2:
                slot = PoolSlot(role_id=role, members=list(members.values()))
                for name in members.values():
                    if name != active_name:
                        _stow_member(self._env, name, slot, env_id, self._cfg)
                self._env.pool_alias_state.set_active(role, active_name)
            _teleport_member(
                self._env,
                active_name,
                pose_xyz=torch.tensor(obj.position),
                env_id=env_id,
                quat_wxyz=torch.tensor(obj.rotation),
            )
        self._count += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """No per-reset state to clear."""
        pass


class LayoutResetTermCfg(EventTermBaseCfg[LayoutResetTerm, LabSceneEntityCfg]):
    """Configuration for `LayoutResetTerm`."""

    class_type: ClassType_co[LayoutResetTerm] = LayoutResetTerm

    trigger_topic: str = "reset"
    """Event topic that triggers cycling; defaults to ``reset``."""

    layouts: LayoutSequence
    """Parsed layout sequence; ``entries[k]`` drives episode ``k``."""

    role_member_by_category: dict[str, dict[str, str]]
    """Per-role mapping ``{role: {category: scene_name}}``.

    Roles with exactly one entry are treated as single-asset (no stow, no
    alias). Roles with two or more get the pool stow/activate path.
    """

    storage_origin: tuple[float, float, float] = (0.0, 0.0, -50.0)
    """World-space origin used by ``_stow_member`` for inactive members."""

    storage_spread: tuple[float, float, float] = (0.5, 0.5, 0.0)
    """Per-member offset applied to ``storage_origin`` to avoid stacking."""

    model_config = {"arbitrary_types_allowed": True}
