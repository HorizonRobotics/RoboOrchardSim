# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""TaskAssetsBase.from_resolved distractor grouping."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from robo_orchard_sim.orchard_env.assets.object_spec import RigidObjectSpec
from robo_orchard_sim.orchard_env.assets.pool_spec import PoolSpec
from robo_orchard_sim.orchard_env.task_templates.place_a2b_task import (
    PlaceA2BTaskAssets,
)


def _obj(name: str) -> RigidObjectSpec:
    return RigidObjectSpec(name=name, usd_path=f"/tmp/{name}.usd")


def _pool(role: str, n: int, active: int = 1) -> PoolSpec:
    return PoolSpec(
        role_id=role,
        members=[_obj(f"{role}_pool_{i}") for i in range(n)],
        active_count=active,
    )


def test_from_resolved_merges_arbitrary_distractor_roles():
    assets = PlaceA2BTaskAssets.from_resolved(
        {
            "pick": _obj("pick_object"),
            "place": _obj("place_object"),
            "distractors_graspable": [_obj("g_0"), _obj("g_1"), _obj("g_2")],
            "distractors_container": [_obj("c_0")],
        }
    )
    flat = assets.flatten_distractors()
    assert set(flat) == {f"distractor_{i}" for i in range(4)}


def test_from_resolved_single_group_unchanged():
    assets = PlaceA2BTaskAssets.from_resolved(
        {
            "pick": _obj("pick_object"),
            "place": _obj("place_object"),
            "distractors": [_obj("d_0"), _obj("d_1")],
        }
    )
    flat = assets.flatten_distractors()
    assert set(flat) == {"distractor_0", "distractor_1"}


def test_from_resolved_single_pool_preserved():
    assets = PlaceA2BTaskAssets.from_resolved(
        {
            "pick": _obj("pick_object"),
            "place": _obj("place_object"),
            "distractors": _pool("distractor", 4, active=3),
        }
    )
    assert "distractors_pool" in assets.flatten_distractors()


def test_from_resolved_no_distractors():
    assets = PlaceA2BTaskAssets.from_resolved(
        {"pick": _obj("pick_object"), "place": _obj("place_object")}
    )
    assert assets.flatten_distractors() == {}


def test_from_resolved_missing_required_target_raises():
    with pytest.raises(ValidationError):
        PlaceA2BTaskAssets.from_resolved({"pick": _obj("pick_object")})
