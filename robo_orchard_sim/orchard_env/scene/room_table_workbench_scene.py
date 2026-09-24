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

"""Room-table scene with a raised tabletop workbench."""

from robo_orchard_sim.ext.cfg_wrappers.sim.schemas import (
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.from_files import (
    UsdFileCfg,
)
from robo_orchard_sim.ext.models.assets.asset_cfg import (
    ORCHARD_ASSET,
    GroupAssetCfg,
)
from robo_orchard_sim.ext.models.assets.rigid_object import RigidObjectCfg
from robo_orchard_sim.orchard_env.scene.room_table_scene import RoomTableScene


class RoomTableWorkbenchScene(RoomTableScene):
    """Add a 0.25 m high support for articulated tabletop objects."""

    def _set_default_scene(self) -> dict[str, GroupAssetCfg]:
        """Place the workbench on the original table's z=0 surface."""
        grouped = super()._set_default_scene()
        table_cfg = grouped["background"]["table"].copy()
        assert table_cfg.spawn is not None
        table_cfg.spawn = table_cfg.spawn.copy()
        table_cfg.spawn.scale = (40.0, 40.0, 16.0)
        grouped["background"]["table"] = table_cfg
        grouped["background"]["workbench"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/workbench",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.84, 0.0, 0.0),
                rot=(0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
            ),
            spawn=UsdFileCfg(
                usd_path=(
                    f"{ORCHARD_ASSET}/PUBLIC_OBJECTS/evaluation_assets"
                    "/tables/tabletop_workbench/tabletop_workbench.usd"
                ),
                # Raise the original 0.18 m support without moving its base.
                scale=(1.0, 1.0, 0.25 / 0.18),
                semantic_tags=[("class", "workbench")],
                rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
        )
        return grouped


__all__ = ["RoomTableWorkbenchScene"]
