# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""Tests for PoolAliasState."""

from __future__ import annotations

import pytest

from robo_orchard_sim.ext.models.scenes.pool_alias_state import PoolAliasState


def test_register_set_active_resolve_roundtrip():
    pm = PoolAliasState()
    pm.register_pool("pick_object")
    assert pm.has_pool("pick_object") and not pm.has_pool("place_object")
    pm.set_active("pick_object", "pick_object_pool_5")
    assert pm.resolve("pick_object") == "pick_object_pool_5"
    pm.set_active("pick_object", None)
    assert pm.resolve("pick_object") == "pick_object"


def test_resolve_unknown_returns_input_unchanged():
    assert PoolAliasState().resolve("not_a_role") == "not_a_role"


def test_register_pool_rejects_duplicate():
    pm = PoolAliasState()
    pm.register_pool("pick_object")
    with pytest.raises(ValueError):
        pm.register_pool("pick_object")


def test_set_active_unregistered_raises():
    with pytest.raises(KeyError):
        PoolAliasState().set_active("not_registered", "some_name")
