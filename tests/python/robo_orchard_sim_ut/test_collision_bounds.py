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

"""Behavioral tests of cached collision bounds using in-memory USD."""

from types import SimpleNamespace

import pytest
import torch
from pxr import Usd, UsdGeom, UsdPhysics

from robo_orchard_sim.task_components.validators.physical_entity import (
    CollisionBodyBounds,
    SceneBodyKey,
)


def runtime():
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/World/Body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/World/Body/collision")
    cube.CreateSizeAttr(2)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    data = SimpleNamespace(
        body_pos_w=torch.zeros(1, 1, 3),
        body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
    )
    asset = SimpleNamespace(
        data=data,
        find_bodies=lambda name: ([0], ["body"]),
        root_physx_view=SimpleNamespace(link_paths=[["/World/Body"]]),
    )
    return SimpleNamespace(
        sim=SimpleNamespace(stage=stage), scene={"asset": asset}
    )


@pytest.mark.parametrize(
    ("point", "expected"),
    [([0, 0, 0], 0), ([1, 0, 0], 0), ([1.3, 0, 0], 0.3), ([1.3, 1.4, 0], 0.5)],
)
def test_collision_bounds_point_location_returns_box_distance(point, expected):
    env = runtime()
    result = CollisionBodyBounds().distances(
        env,
        SceneBodyKey("asset", "body"),
        torch.tensor([point], dtype=torch.float32),
    )
    assert result.item() == pytest.approx(expected, abs=1e-6)


def test_collision_bounds_cached_box_follows_live_translation_and_rotation():
    env = runtime()
    stage = env.sim.stage
    UsdGeom.Xformable(stage.GetPrimAtPath("/World/Body")).AddScaleOp().Set(
        (2, 1, 1)
    )
    UsdGeom.Xformable(
        stage.GetPrimAtPath("/World/Body/collision")
    ).AddTranslateOp().Set((1, 0, 0))
    bounds = CollisionBodyBounds()
    key = SceneBodyKey("asset", "body")
    first = bounds.distances(env, key, torch.tensor([[4.3, 0, 0]])).item()
    data = env.scene["asset"].data
    data.body_pos_w[0, 0] = torch.tensor([10.0, 0.0, 0.0])
    data.body_quat_w[0, 0] = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
    moved = bounds.distances(env, key, torch.tensor([[10.0, 4.3, 0.0]])).item()
    assert (first, moved) == pytest.approx((0.3, 0.3), abs=1e-5)


@pytest.mark.parametrize("kind", ["visual", "disabled", "nested_body"])
def test_collision_bounds_non_owned_geometry_does_not_expand_box(kind):
    env = runtime()
    stage = env.sim.stage
    parent = UsdGeom.Xform.Define(stage, "/World/Body/other")
    cube = UsdGeom.Cube.Define(stage, "/World/Body/other/mesh")
    cube.CreateSizeAttr(100)
    if kind != "visual":
        api = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        api.CreateCollisionEnabledAttr(kind != "disabled")
    if kind == "nested_body":
        UsdPhysics.RigidBodyAPI.Apply(parent.GetPrim())
    result = CollisionBodyBounds().distances(
        env, SceneBodyKey("asset", "body"), torch.tensor([[1.3, 0, 0]])
    )
    assert result.item() == pytest.approx(0.3, abs=1e-6)


def test_collision_bounds_reset_reloads_changed_geometry():
    env = runtime()
    bounds = CollisionBodyBounds()
    key, point = SceneBodyKey("asset", "body"), torch.tensor([[2.0, 0, 0]])
    first = bounds.distances(env, key, point).item()
    UsdGeom.Cube(
        env.sim.stage.GetPrimAtPath("/World/Body/collision")
    ).GetSizeAttr().Set(4)
    bounds.reset()
    second = bounds.distances(env, key, point).item()
    assert (first, second) == pytest.approx((1, 0))


def test_collision_bounds_missing_colliders_raises_value_error():
    env = runtime()
    env.sim.stage.RemovePrim("/World/Body/collision")
    with pytest.raises(ValueError, match="No enabled collision bounds"):
        CollisionBodyBounds().distances(
            env, SceneBodyKey("asset", "body"), torch.zeros(1, 3)
        )


def test_collision_bounds_instance_proxy_includes_collision_geometry():
    env = runtime()
    stage = env.sim.stage
    stage.RemovePrim("/World/Body/collision")
    UsdGeom.Xform.Define(stage, "/Prototype")
    cube = UsdGeom.Cube.Define(stage, "/Prototype/collision")
    cube.CreateSizeAttr(2)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    child = UsdGeom.Xform.Define(stage, "/World/Body/instance").GetPrim()
    child.GetReferences().AddInternalReference("/Prototype")
    child.SetInstanceable(True)
    result = CollisionBodyBounds().distances(
        env, SceneBodyKey("asset", "body"), torch.tensor([[1.3, 0, 0]])
    )
    assert result.item() == pytest.approx(0.3, abs=1e-6)
