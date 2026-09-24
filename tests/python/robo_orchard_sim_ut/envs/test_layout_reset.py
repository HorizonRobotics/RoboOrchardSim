# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""LayoutResetTerm core behavioural tests with a mocked scene."""

from __future__ import annotations
from unittest.mock import MagicMock

import torch

from robo_orchard_sim.ext.envs.managers.events.layout_reset import (
    LayoutResetTerm,
    LayoutResetTermCfg,
)
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
    poses: list[tuple[str, torch.Tensor]] = []

    def make_entity(name):
        ent = MagicMock()
        ent.write_root_pose_to_sim.side_effect = lambda pose, env_ids: (
            poses.append((name, pose[0].clone()))
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


def _cfg(entries: list[Layout], scene_name_by_role: dict[str, str]):
    return LayoutResetTermCfg(
        layouts=LayoutSequence(entries=entries, raw=[]),
        scene_name_by_role=scene_name_by_role,
    )


def test_layout_reset_consecutive_episodes_applies_pose_and_rotation():
    """Each call applies the next pose to the role's fixed scene object."""
    env, poses = _env_with_recorder()
    cfg = _cfg(
        entries=[
            _layout({"src": ("apple", (0.1, 0.2, 0.8), (1.0, 0.0, 0.0, 0.0))}),
            _layout({"src": ("apple", (0.3, 0.4, 0.9), (0.0, 1.0, 0.0, 0.0))}),
        ],
        scene_name_by_role={"src": "src__apple"},
    )
    term = LayoutResetTerm(cfg, env)

    for xyz, quat in [
        ((0.1, 0.2, 0.8), (1.0, 0.0, 0.0, 0.0)),
        ((0.3, 0.4, 0.9), (0.0, 1.0, 0.0, 0.0)),
    ]:
        before = len(poses)
        term(MagicMock(env_ids=None))
        pose = poses[before][1]
        assert torch.allclose(pose[:3], torch.tensor(xyz))
        assert torch.allclose(pose[3:], torch.tensor(quat))


def test_layout_reset_nonzero_env_origin_offsets_position():
    env, poses = _env_with_recorder()
    env.scene.env_origins = torch.tensor([[1.0, 2.0, 3.0]])
    cfg = _cfg(
        entries=[
            _layout({"src": ("apple", (0.5, 0.6, 0.7), (1.0, 0.0, 0.0, 0.0))})
        ],
        scene_name_by_role={"src": "src__apple"},
    )

    LayoutResetTerm(cfg, env)(MagicMock(env_ids=None))

    assert torch.allclose(
        poses[0][1][:3],
        torch.tensor((1.5, 2.6, 3.7)),
    )
