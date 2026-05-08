# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Tests for PoolSpec wrapper type."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from robo_orchard_sim.orchard_env.assets.object_spec import RigidObjectSpec
from robo_orchard_sim.orchard_env.assets.pool_spec import PoolSpec


def _members(n: int) -> list[RigidObjectSpec]:
    return [
        RigidObjectSpec(name=f"pick_pool_{i}", usd_path=f"/tmp/p{i}.usd")
        for i in range(n)
    ]


def test_pool_spec_exposes_role_id_and_member_names():
    pool = PoolSpec(role_id="pick_object", members=_members(3))
    assert pool.scene_name == "pick_object"
    assert pool.member_scene_names == [
        "pick_pool_0",
        "pick_pool_1",
        "pick_pool_2",
    ]
    assert pool.active_count == 1  # default


@pytest.mark.parametrize("n_members", [0, 1])
def test_pool_spec_rejects_too_few_members(n_members):
    with pytest.raises(ValidationError):
        PoolSpec(role_id="pick_object", members=_members(n_members))


@pytest.mark.parametrize("active_count", [0, -1, 5])
def test_pool_spec_rejects_invalid_active_count(active_count):
    """active_count must be >= 1 and <= len(members)."""
    with pytest.raises(ValidationError):
        PoolSpec(
            role_id="pick_object",
            members=_members(3),
            active_count=active_count,
        )
