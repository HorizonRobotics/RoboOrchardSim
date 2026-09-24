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
from robo_orchard_sim.orchard_env.layout.loader import LayoutSequence
from robo_orchard_sim.utils.config import ClassType_co

__all__ = [
    "LayoutResetTerm",
    "LayoutResetTermCfg",
]


def _teleport_layout_object(
    env,
    name: str,
    pose_xyz: torch.Tensor,
    quat_wxyz: torch.Tensor,
) -> None:
    """Apply an env-local layout pose and clear the object's velocity."""
    env_id = 0
    device = getattr(env, "device", "cpu")
    position = pose_xyz.to(device) + env.scene.env_origins[env_id]
    rotation = quat_wxyz.to(device)
    full_pose = torch.cat([position, rotation]).unsqueeze(0)
    env_ids = torch.tensor([env_id], device=device)
    entity = env.scene[name]
    entity.write_root_pose_to_sim(full_pose, env_ids=env_ids)
    entity.write_root_velocity_to_sim(
        torch.zeros((1, 6), device=device),
        env_ids=env_ids,
    )


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
        for role, obj in layout.objects.items():
            _teleport_layout_object(
                self._env,
                self._cfg.scene_name_by_role[role],
                pose_xyz=torch.tensor(obj.position),
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

    scene_name_by_role: dict[str, str]
    """Per-layout-role mapping to the fixed scene object for that role."""

    model_config = {"arbitrary_types_allowed": True}
