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
"""Inactive-pool-storage cfg factory for pool support.

Provides a kinematic shelf far below the table where inactive pool members
are stowed off-screen. Each pool member rotates between the table (active)
and this storage (inactive) per episode reset.
"""

from __future__ import annotations

from robo_orchard_sim.ext.cfg_wrappers.materials.visual_materials_cfg import (
    PreviewSurfaceCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.schemas.schemas_cfg import (
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.shapes_cfg import CuboidCfg
from robo_orchard_sim.ext.models.assets.rigid_object import RigidObjectCfg

INACTIVE_POOL_STORAGE_NAME = "inactive_pool_storage"
INACTIVE_POOL_STORAGE_PRIM_PATH = "/World/inactive_pool_storage"
INACTIVE_POOL_STORAGE_Z = -50.0
INACTIVE_POOL_STORAGE_SIZE = (100.0, 100.0, 0.5)


def make_inactive_pool_storage_cfg() -> RigidObjectCfg:
    """Build the kinematic storage shelf used to stow inactive pool members."""
    return RigidObjectCfg(
        prim_path=INACTIVE_POOL_STORAGE_PRIM_PATH,
        spawn=CuboidCfg(
            size=INACTIVE_POOL_STORAGE_SIZE,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.2, 0.2),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, INACTIVE_POOL_STORAGE_Z),
        ),
    )
