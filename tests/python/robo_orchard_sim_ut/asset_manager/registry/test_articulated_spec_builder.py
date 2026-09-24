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

"""Behavioral coverage for registry articulated-object construction."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from robo_orchard_sim.asset_manager.registry.spec_builders import (
    build_articulated_object_spec,
)
from robo_orchard_sim.asset_manager.registry.types import (
    ARTICULATION_SPEC_TYPE,
    AssetMeta,
)
from robo_orchard_sim.orchard_env.assets import ArticulatedObjectSpec


def _asset_meta(tmp_path: Path) -> AssetMeta:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "articulation": {
                    "root": {"fix_root_link": True},
                    "joints": [
                        {
                            "joint_name": "hinge",
                            "semantic_name": "lid",
                            "outcome_link": "lid_link",
                            "operations": {
                                "open": {
                                    "interaction_link": "button_link",
                                    "initial_joint_position": 0.0,
                                    "target_joint_fraction": 1.0,
                                }
                            },
                        }
                    ],
                }
            }
        )
    )
    return AssetMeta(
        uuid="direct-articulation",
        asset_id="object",
        relative_path=str(tmp_path),
        domain="showcase",
        super_category="showcase",
        category="laptop",
        name="laptop",
        description="silver laptop",
        color=None,
        shape=None,
        material=None,
        real_height=0.0,
        real_mass=1.0,
        min_height=0.0,
        max_height=0.0,
        min_mass=1.0,
        max_mass=1.0,
        usd_path=str(tmp_path / "object.usd"),
        urdf_path="",
        interaction_path="",
        caption_path=str(tmp_path / "caption_candidates.json"),
        metadata_path=str(metadata_path),
        spec_type=ARTICULATION_SPEC_TYPE,
    )


def test_build_articulated_object_spec_valid_metadata_returns_semantic_spec(
    tmp_path: Path,
) -> None:
    spec = build_articulated_object_spec(
        _asset_meta(tmp_path),
        "laptop",
        "primary",
        {},
    )

    assert isinstance(spec, ArticulatedObjectSpec)
    assert spec.uuid == "direct-articulation"
    assert spec.joint_operations[0].outcome_link == "lid_link"
    assert (
        spec.joint_operations[0].operations["open"].interaction_link
        == "button_link"
    )
    assert spec.template_cfg.spawn.articulation_props.fix_root_link is True


def test_build_articulated_object_spec_carries_registry_metadata(
    tmp_path: Path,
) -> None:
    """Registry metadata must survive into the spawned scene cfg."""
    spec = build_articulated_object_spec(
        _asset_meta(tmp_path),
        "laptop",
        "primary",
        {},
    )

    assert spec.category == "laptop"
    assert spec.actor_type == "primary"
    assert spec.caption_path.endswith("caption_candidates.json")

    cfg = spec.to_isaac_cfg()

    assert cfg.caption_path == spec.caption_path
    assert cfg.uuid == "direct-articulation"
    assert cfg.category == "laptop"
    assert cfg.actor_type == "primary"
    assert cfg.joint_operations[0].outcome_link == "lid_link"
    assert (
        cfg.joint_operations[0].operations["open"].interaction_link
        == "button_link"
    )


def test_build_articulated_object_spec_offset_root_uses_root_relative_aabb(
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "object.usd"
    stage = Usd.Stage.CreateNew(str(usd_path))
    default_prim = UsdGeom.Xform.Define(stage, "/RootNode").GetPrim()
    stage.SetDefaultPrim(default_prim)
    articulation_root = UsdGeom.Xform.Define(
        stage,
        "/RootNode/body",
    )
    articulation_root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.2))
    UsdPhysics.ArticulationRootAPI.Apply(articulation_root.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/RootNode/body/geometry")
    cube.CreateSizeAttr(0.2)
    stage.GetRootLayer().Save()
    meta = replace(
        _asset_meta(tmp_path),
        usd_path=str(usd_path),
        aabb_x_min=-0.1,
        aabb_y_min=-0.1,
        aabb_z_min=0.1,
        aabb_x_max=0.1,
        aabb_y_max=0.1,
        aabb_z_max=0.3,
    )

    spec = build_articulated_object_spec(
        meta,
        "laptop",
        "primary",
        {},
    )

    assert spec.aabb_min == pytest.approx((-0.1, -0.1, -0.1))
