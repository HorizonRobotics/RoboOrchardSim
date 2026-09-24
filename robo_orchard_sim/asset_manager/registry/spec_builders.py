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

"""Built-in AssetMeta-to-ObjectSpec builders."""

from __future__ import annotations
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from robo_orchard_sim.asset_manager.registry.types import (
    ARTICULATION_SPEC_TYPE,
    RIGID_OBJECT_SPEC_TYPE,
)

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.registry.types import AssetMeta
    from robo_orchard_sim.orchard_env.assets import ObjectSpec

AssetSpecBuilder = Callable[
    ["AssetMeta", str, str, Mapping[str, Any]],
    "ObjectSpec",
]


def _aabb_corners(
    meta: "AssetMeta",
) -> tuple[
    tuple[float, float, float] | None, tuple[float, float, float] | None
]:
    """Reassemble the index's flat columns into corner tuples.

    The index stores six scalars, but they are only meaningful as the
    two corners the URDF declares, and are written as a set. Treat a
    partially populated row as no box at all rather than passing a
    half-formed corner downstream.
    """
    lower = (meta.aabb_x_min, meta.aabb_y_min, meta.aabb_z_min)
    upper = (meta.aabb_x_max, meta.aabb_y_max, meta.aabb_z_max)
    if any(v is None for v in lower) or any(v is None for v in upper):
        return None, None
    return lower, upper  # type: ignore[return-value]


def _root_relative_aabb_corners(
    usd_path: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Compute geometry bounds in the PhysX articulation-root frame."""
    if not Path(usd_path).is_file():
        return None

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        return None
    default_prim = stage.GetDefaultPrim()
    articulation_roots = [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if not default_prim or len(articulation_roots) != 1:
        return None

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=True,
    )
    aligned_range = cache.ComputeRelativeBound(
        default_prim,
        articulation_roots[0],
    ).ComputeAlignedRange()
    lower = tuple(float(value) for value in aligned_range.GetMin())
    upper = tuple(float(value) for value in aligned_range.GetMax())
    if not all(math.isfinite(value) for value in (*lower, *upper)):
        return None
    return lower, upper


def build_rigid_object_spec(
    meta: "AssetMeta",
    name: str,
    role: str,
    config: Mapping[str, Any],
) -> "ObjectSpec":
    """Build a configured rigid-object spec."""
    from robo_orchard_sim.orchard_env.assets import RigidObjectSpec

    properties = dict(config)
    mass = properties.pop("mass", meta.real_mass)
    aabb_min, aabb_max = _aabb_corners(meta)
    return RigidObjectSpec(
        name=name,
        usd_path=meta.usd_path,
        mass=mass,
        caption_path=meta.caption_path,
        interaction_path=meta.interaction_path,
        uuid=meta.uuid,
        category=meta.category,
        actor_type=role,
        attributes={
            "color": tuple(sorted(meta.color or ())),
            "shape": tuple(sorted(meta.shape or ())),
            "material": tuple(sorted(meta.material or ())),
        },
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        **properties,
    )


def build_articulated_object_spec(
    meta: "AssetMeta",
    name: str,
    role: str,
    config: Mapping[str, Any],
) -> "ObjectSpec":
    """Build an articulation with typed asset operation metadata."""
    from robo_orchard_sim.asset_manager.metadata import (
        load_articulation_metadata,
    )
    from robo_orchard_sim.ext.cfg_wrappers.sim.schemas import (
        ArticulationRootPropertiesCfg,
    )
    from robo_orchard_sim.ext.cfg_wrappers.sim.spawners import UsdFileCfg
    from robo_orchard_sim.ext.models.assets.articulation import ArticulationCfg
    from robo_orchard_sim.orchard_env.assets import ArticulatedObjectSpec

    expected_uuid = meta.uuid if meta.urdf_path else None
    metadata = load_articulation_metadata(
        meta.metadata_path,
        expected_uuid=expected_uuid,
    )
    properties = dict(config)
    template_cfg = properties.pop("template_cfg", None)
    aabb_min, aabb_max = _aabb_corners(meta)
    root_relative_aabb = _root_relative_aabb_corners(meta.usd_path)
    if root_relative_aabb is not None:
        aabb_min, aabb_max = root_relative_aabb
    if template_cfg is None:
        root_cfg = None
        if metadata.root.fix_root_link is not None:
            root_cfg = ArticulationRootPropertiesCfg(
                fix_root_link=metadata.root.fix_root_link,
            )
        template_cfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            spawn=UsdFileCfg(
                usd_path=meta.usd_path,
                articulation_props=root_cfg,
            ),
            actuators={},
        )

    return ArticulatedObjectSpec(
        name=name,
        caption_path=meta.caption_path,
        uuid=metadata.uuid or meta.uuid,
        description=meta.description or meta.name or meta.category,
        usd_path=meta.usd_path,
        category=meta.category,
        actor_type=role,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        tags=meta.tags,
        joint_operations=metadata.joints,
        template_cfg=template_cfg,
        **properties,
    )


def default_spec_builders() -> dict[str, AssetSpecBuilder]:
    """Return fresh built-in spec builders."""
    return {
        RIGID_OBJECT_SPEC_TYPE: build_rigid_object_spec,
        ARTICULATION_SPEC_TYPE: build_articulated_object_spec,
    }


__all__ = [
    "AssetSpecBuilder",
    "build_articulated_object_spec",
    "build_rigid_object_spec",
    "default_spec_builders",
]
