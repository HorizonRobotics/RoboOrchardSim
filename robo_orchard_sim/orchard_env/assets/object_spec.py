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

"""Common object asset specs for orchard env modules."""

from __future__ import annotations
import math
from abc import ABC

from pydantic import Field, field_validator, model_validator

from robo_orchard_sim.asset_manager.metadata.joint_operations import (
    JointOperationMeta,
)
from robo_orchard_sim.ext.cfg_wrappers.assets_cfg import ArticulationCfg
from robo_orchard_sim.ext.cfg_wrappers.materials import PreviewSurfaceCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.schemas import (
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners import UsdFileCfg
from robo_orchard_sim.ext.models.assets.articulation import (
    ArticulationCfg as AssetArticulationCfg,
)
from robo_orchard_sim.ext.models.assets.rigid_object import RigidObjectCfg
from robo_orchard_sim.orchard_env.assets.asset_spec import AssetSpec


def _scaled(
    corner: tuple[float, float, float] | None,
    scale: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    if corner is None:
        return None
    return (corner[0] * scale[0], corner[1] * scale[1], corner[2] * scale[2])


class ObjectSpec(AssetSpec, ABC):
    """Abstract base class for task-usable object assets."""

    usd_path: str | None = None
    scale: tuple[float, float, float] | None = None
    mass: float | None = None
    initial_pos: tuple[float, float, float] | None = None
    initial_rot: tuple[float, float, float, float] | None = None

    @field_validator("scale")
    @classmethod
    def validate_positive_scale(
        cls, value: tuple[float, float, float] | None
    ) -> tuple[float, float, float] | None:
        """Reject non-positive scales.

        A negative factor mirrors the asset, which would flip the
        bounding box corners so that ``min`` no longer sits below
        ``max``. Nothing needs mirrored assets today, so rather than
        carry that case through every consumer of the box, refuse it
        here.
        """
        if value is not None and any(factor <= 0.0 for factor in value):
            raise ValueError(
                f"scale must be positive in every axis, got {value}"
            )
        return value


class RigidObjectSpec(ObjectSpec):
    """User-facing description for rigid object assets."""

    interaction_path: str | None = None
    caption_path: str | None = None
    uuid: str | None = None
    category: str | None = None
    actor_type: str = "object"
    attributes: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    visual_color: tuple[float, float, float] | None = None
    aabb_min: tuple[float, float, float] | None = None
    """Bounding-box corner in the asset's local frame, in meters, as
    authored in the URDF. ``None`` when the URDF declares no box."""
    aabb_max: tuple[float, float, float] | None = None

    @field_validator("visual_color")
    @classmethod
    def validate_visual_color(
        cls,
        value: tuple[float, float, float] | None,
    ) -> tuple[float, float, float] | None:
        """Require finite normalized RGB channels."""
        if value is not None and any(
            not math.isfinite(channel) or not 0.0 <= channel <= 1.0
            for channel in value
        ):
            raise ValueError("visual_color channels must be within [0, 1]")
        return value

    def to_isaac_cfg(self) -> RigidObjectCfg:
        """Convert this spec into a rigid object cfg."""
        if self.usd_path is None:
            raise ValueError(
                "RigidObjectSpec.usd_path must be set before conversion"
            )
        scale = self.scale or (1.0, 1.0, 1.0)
        cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + self.name,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=list(self.initial_pos or (0.0, 0.0, 0.0)),
                rot=list(self.initial_rot or (1.0, 0.0, 0.0, 0.0)),
            ),
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=scale,
                semantic_tags=[],
                visual_material=(
                    PreviewSurfaceCfg(diffuse_color=self.visual_color)
                    if self.visual_color is not None
                    else None
                ),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=100.0,
                    max_linear_velocity=1.0,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(
                    mass=0.05 if self.mass is None else self.mass
                ),
            ),
            interaction_path=self.interaction_path,
            caption_path=self.caption_path,
            uuid=self.uuid,
            category=self.category,
            actor_type=self.actor_type,
            attributes=self.attributes,
            # The box describes the prim that gets spawned, so it takes
            # the same scale applied to the spawner above.
            aabb_min=_scaled(self.aabb_min, scale),
            aabb_max=_scaled(self.aabb_max, scale),
        )
        return cfg


class ArticulationSpec(ObjectSpec):
    """User-facing description for articulation assets."""

    template_cfg: ArticulationCfg
    joint_pos: dict[str, float] | None = None

    def to_isaac_cfg(self) -> ArticulationCfg:
        """Copy the template cfg and patch orchard-owned identity fields."""
        cfg = self.template_cfg.copy()
        cfg.prim_path = "{ENV_REGEX_NS}/" + self.name

        init_state = cfg.init_state.copy()
        if self.initial_pos is not None:
            init_state.pos = self.initial_pos
        if self.initial_rot is not None:
            init_state.rot = self.initial_rot
        if self.joint_pos is not None:
            init_state.joint_pos = dict(self.joint_pos)
        cfg.init_state = init_state
        return cfg


class ArticulatedObjectSpec(ArticulationSpec):
    """Articulation asset with task-facing joint-operation metadata.

    The registry constructs this type by combining the current physical
    ``ArticulationSpec`` template with validated adjacent ``metadata.json``
    semantics.
    """

    caption_path: str | None = None
    uuid: str
    description: str
    category: str | None = None
    actor_type: str = "object"
    aabb_min: tuple[float, float, float] | None = None
    """Bounding-box corner, measured at a single joint configuration;
    see :attr:`RigidObjectSpec.aabb_min`."""
    aabb_max: tuple[float, float, float] | None = None
    tags: frozenset[str] = Field(default_factory=frozenset)
    joint_operations: tuple[JointOperationMeta, ...]

    def to_isaac_cfg(self) -> AssetArticulationCfg:
        """Patch registry-owned metadata onto the physical cfg."""
        cfg = super().to_isaac_cfg()
        if not isinstance(cfg, AssetArticulationCfg):
            cfg = AssetArticulationCfg(
                **{name: getattr(cfg, name) for name in type(cfg).model_fields}
            )
        cfg.caption_path = self.caption_path
        cfg.uuid = self.uuid
        cfg.category = self.category
        cfg.actor_type = self.actor_type
        # Unscaled: this path spawns from ``template_cfg`` and never
        # applies ``self.scale``.
        cfg.aabb_min = self.aabb_min
        cfg.aabb_max = self.aabb_max
        cfg.joint_operations = self.joint_operations
        return cfg

    @field_validator("uuid", "description")
    @classmethod
    def validate_non_empty_metadata(cls, value: str) -> str:
        """Reject blank task-facing identity fields."""
        if not value.strip():
            raise ValueError("must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_joint_operations(self) -> "ArticulatedObjectSpec":
        """Reject ambiguous mechanical or semantic joint descriptions."""
        joint_names = [item.joint_name for item in self.joint_operations]
        semantic_names = [item.semantic_name for item in self.joint_operations]
        if len(joint_names) != len(set(joint_names)):
            raise ValueError(
                "Asset metadata contains a duplicate mechanical joint name."
            )
        if len(semantic_names) != len(set(semantic_names)):
            raise ValueError(
                "Asset metadata contains a duplicate semantic joint "
                "description."
            )
        return self
