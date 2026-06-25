# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""PlaceA2BTaskAssets / PlaceA2BTask interaction with PoolSpec."""

from __future__ import annotations

from robo_orchard_sim.orchard_env.assets.object_spec import (
    ObjectSpec,
    RigidObjectSpec,
)
from robo_orchard_sim.orchard_env.assets.pool_spec import PoolSpec
from robo_orchard_sim.orchard_env.task_templates.place_a2b_task import (
    PlaceA2BTask,
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


def test_assets_accept_either_object_spec_or_pool_spec():
    """pick/place fields accept ObjectSpec or PoolSpec interchangeably."""
    a = PlaceA2BTaskAssets(pick=_pool("pick_object", 3), place=_obj("p"))
    assert isinstance(a.pick, PoolSpec) and isinstance(a.place, ObjectSpec)


def test_distractor_pool_routes_to_distractors_pool_field():
    """A PoolSpec distractor populates task.distractors_pool, not list."""
    task = PlaceA2BTask(
        assets=PlaceA2BTaskAssets(
            pick=_obj("pick_object"),
            place=_obj("place_object"),
            distractors=_pool("distractor", 4, active=3),
        )
    )
    assert task.distractors_pool is not None
    assert task.distractors_pool.active_count == 3
    assert task.distractors == []


def test_classic_distractor_list_routes_to_distractors_field():
    task = PlaceA2BTask(
        assets=PlaceA2BTaskAssets(
            pick=_obj("pick_object"),
            place=_obj("place_object"),
            distractors=[_obj("d_0"), _obj("d_1")],
        )
    )
    assert task.distractors_pool is None
    assert len(task.distractors) == 2
