# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""InteractiveScene.__getitem__ alias resolution via PoolAliasState."""

from __future__ import annotations
from unittest.mock import MagicMock

from robo_orchard_sim.ext.models.scenes.pool_alias_state import PoolAliasState


def _scene_with_state(monkeypatch, state):
    from robo_orchard_sim.ext.models.scenes.interactive_scene import (
        InteractiveScene,
    )

    parent_get = MagicMock(side_effect=lambda k: f"entity({k})")
    monkeypatch.setattr(
        InteractiveScene.__bases__[0],
        "__getitem__",
        lambda self, k: parent_get(k),
    )
    scene = InteractiveScene.__new__(InteractiveScene)
    scene._pool_alias_state = state
    return scene, parent_get


def test_getitem_without_alias_state_passes_through(monkeypatch):
    scene, parent_get = _scene_with_state(monkeypatch, state=None)
    assert scene["pick_object"] == "entity(pick_object)"
    parent_get.assert_called_once_with("pick_object")


def test_getitem_with_bound_alias_resolves_to_active(monkeypatch):
    state = PoolAliasState()
    state.register_pool("pick_object")
    state.set_active("pick_object", "pick_object_pool_3")
    scene, parent_get = _scene_with_state(monkeypatch, state=state)
    assert scene["pick_object"] == "entity(pick_object_pool_3)"
    parent_get.assert_called_once_with("pick_object_pool_3")
