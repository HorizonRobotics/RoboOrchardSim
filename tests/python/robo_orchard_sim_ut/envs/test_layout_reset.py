# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""LayoutResetTerm core behavioural tests (mocked scene + alias state)."""

from __future__ import annotations
from unittest.mock import MagicMock

import torch

from robo_orchard_sim.ext.envs.managers.events.layout_reset import (
    LayoutResetTerm,
    LayoutResetTermCfg,
)
from robo_orchard_sim.ext.models.scenes.pool_alias_state import PoolAliasState
from robo_orchard_sim.orchard_env.layout.loader import (
    Layout,
    LayoutObject,
    LayoutSequence,
)


def _env_with_recorder():
    env = MagicMock()
    env.num_envs = 1
    env.cfg = MagicMock(seed=0)
    env.device = "cpu"
    env.scene = MagicMock()
    env.scene.env_origins = torch.zeros((1, 3))
    env.pool_alias_state = PoolAliasState()
    poses: list[tuple[str, torch.Tensor]] = []

    def make_entity(name):
        ent = MagicMock()
        ent.write_root_pose_to_sim.side_effect = (
            lambda pose, env_ids: poses.append((name, pose[0].clone()))
        )
        return ent

    env.scene.__getitem__.side_effect = make_entity
    return env, poses


def _layout(role_categories: dict[str, tuple[str, tuple, tuple]]) -> Layout:
    return Layout(
        objects={
            r: LayoutObject(category=c, position=p, rotation=q)
            for r, (c, p, q) in role_categories.items()
        },
        raw={},
    )


def _cfg(entries: list[Layout], role_member_by_category: dict):
    return LayoutResetTermCfg(
        layouts=LayoutSequence(entries=entries, raw=[]),
        role_member_by_category=role_member_by_category,
    )


def test_cycle_applies_layout_pose_and_rotation_per_episode():
    """k-th call teleports active to entries[k]'s pose+rotation."""
    env, poses = _env_with_recorder()
    cfg = _cfg(
        entries=[
            _layout({"src": ("apple", (0.1, 0.2, 0.8), (1.0, 0.0, 0.0, 0.0))}),
            _layout(
                {"src": ("orange", (0.3, 0.4, 0.9), (0.0, 1.0, 0.0, 0.0))}
            ),
        ],
        role_member_by_category={
            "src": {"apple": "src__apple", "orange": "src__orange"}
        },
    )
    term = LayoutResetTerm(cfg, env)

    for name, xyz, quat in [
        ("src__apple", (0.1, 0.2, 0.8), (1.0, 0.0, 0.0, 0.0)),
        ("src__orange", (0.3, 0.4, 0.9), (0.0, 1.0, 0.0, 0.0)),
    ]:
        before = len(poses)
        term(MagicMock(env_ids=None))
        active = [p for p in poses[before:] if p[0] == name]
        assert active, f"no teleport for {name}"
        pose = active[0][1]
        assert torch.allclose(pose[:3], torch.tensor(xyz))
        assert torch.allclose(pose[3:], torch.tensor(quat))


def test_pool_role_stows_inactive_members_and_binds_alias():
    env, poses = _env_with_recorder()
    cfg = _cfg(
        entries=[
            _layout({"src": ("apple", (0.0,) * 3, (1.0, 0.0, 0.0, 0.0))})
        ],
        role_member_by_category={
            "src": {"apple": "src__apple", "orange": "src__orange"},
        },
    )
    LayoutResetTerm(cfg, env)(MagicMock(env_ids=None))
    assert any(name == "src__orange" for name, _ in poses), (
        "inactive not stowed"
    )
    assert env.pool_alias_state.resolve("src") == "src__apple"


def test_single_category_role_works_without_pool_alias_state():
    """Single-entry layout (all ObjectSpec) must not require pool_alias_state.

    Regression: LayoutResetTerm.__init__ used to raise RuntimeError whenever
    env.pool_alias_state was None, only attached for pooled layouts.
    """
    env, poses = _env_with_recorder()
    env.pool_alias_state = None
    cfg = _cfg(
        entries=[
            _layout({"src": ("apple", (0.5, 0.6, 0.7), (1.0, 0.0, 0.0, 0.0))})
        ],
        role_member_by_category={"src": {"apple": "src__apple"}},
    )

    term = LayoutResetTerm(cfg, env)
    term(MagicMock(env_ids=None))

    teleported = [p for p in poses if p[0] == "src__apple"]
    assert teleported, "single-category role was not teleported"
    assert torch.allclose(teleported[0][1][:3], torch.tensor((0.5, 0.6, 0.7)))
