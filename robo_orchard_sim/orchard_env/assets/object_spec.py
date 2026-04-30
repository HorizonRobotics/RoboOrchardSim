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
from abc import ABC

from robo_orchard_sim.cfg_wrappers.assets_cfg import ArticulationCfg
from robo_orchard_sim.cfg_wrappers.sim.schemas import (
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.cfg_wrappers.sim.spawners import UsdFileCfg
from robo_orchard_sim.models.assets.rigid_object import RigidObjectCfg
from robo_orchard_sim.orchard_env.assets.asset_spec import AssetSpec


class ObjectSpec(AssetSpec, ABC):
    """Abstract base class for task-usable object assets."""

    usd_path: str | None = None
    scale: tuple[float, float, float] | None = None
    mass: float | None = None
    initial_pos: tuple[float, float, float] | None = None
    initial_rot: tuple[float, float, float, float] | None = None


class RigidObjectSpec(ObjectSpec):
    """User-facing description for rigid object assets."""

    interaction_path: str | None = None
    caption_path: str | None = None
    uuid: str | None = None
    category: str | None = None
    actor_type: str = "object"
    attributes: tuple[str, ...] = ()

    def to_isaac_cfg(self) -> RigidObjectCfg:
        """Convert this spec into a rigid object cfg."""
        if self.usd_path is None:
            raise ValueError(
                "RigidObjectSpec.usd_path must be set before conversion"
            )
        cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + self.name,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=list(self.initial_pos or (0.0, 0.0, 0.0)),
                rot=list(self.initial_rot or (1.0, 0.0, 0.0, 0.0)),
            ),
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale or (1.0, 1.0, 1.0),
                semantic_tags=[],
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
