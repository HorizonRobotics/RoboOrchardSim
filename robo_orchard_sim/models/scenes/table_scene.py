# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
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


from robo_orchard_sim.cfg_wrappers.assets_cfg import (  # noqa: F401
    ArticulationCfg,
    RigidObjectCfg,
)
from robo_orchard_sim.cfg_wrappers.scenes_cfg import InteractiveSceneCfg
from robo_orchard_sim.cfg_wrappers.sim.spawners.from_files import (
    GroundPlaneCfg,
    UsdFileCfg,
)
from robo_orchard_sim.cfg_wrappers.sim.spawners.lights_cfg import (
    DomeLightCfg,
)
from robo_orchard_sim.models.assets.asset_cfg import (  # noqa: F401
    NV_ISAAC_DIR,
    ASSET_CFG_co,
    AssetBaseCfg,
    GroupAssetCfg,
    GroupAssetCfgType,
    asset_replace_usd_path,
)
from robo_orchard_sim.models.assets.xform_asset import XFormPrimAsset
from robo_orchard_sim.models.sensors.camera import (
    CameraCfg,
    SemanticCameraCfg,
)


class TableSceneCfg(InteractiveSceneCfg):
    # model_config = ConfigDict(arbitrary_types_allowed=True)

    robots: GroupAssetCfgType | ArticulationCfg | None = None

    objects: GroupAssetCfgType | None = None

    cameras: GroupAssetCfgType | CameraCfg | SemanticCameraCfg | None = None

    # Table
    table: AssetBaseCfg | GroupAssetCfgType | None = asset_replace_usd_path(
        AssetBaseCfg(
            class_type=XFormPrimAsset,
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.5, 0, 0), rot=(0.707, 0, 0, 0.707)
            ),
            spawn=UsdFileCfg(
                usd_path=f"{NV_ISAAC_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"  # noqa
            ),
        )
    )

    """`ENV_REGEX_NS` is a special variable that is replaced with the
    environment name during scene creation. Any entity that has the
    `ENV_REGEX_NS` variable in its prim path will be cloned for each
    environment.
    This path is replaced by the scene object with /World/envs/env_{i}
    where i is the environment index.
    """

    # plane
    plane: AssetBaseCfg | None = asset_replace_usd_path(
        AssetBaseCfg(
            class_type=XFormPrimAsset,
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)),
            spawn=GroundPlaneCfg(),
        )
    )

    # lights
    light: AssetBaseCfg | GroupAssetCfgType | None = asset_replace_usd_path(
        AssetBaseCfg(
            class_type=XFormPrimAsset,
            prim_path="/World/light",
            spawn=DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
    )
