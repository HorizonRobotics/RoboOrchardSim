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

"""Plane-and-table scene provider for phase0 minimal closure."""

import os

from robo_orchard_core.utils.logging import LoggerManager

from robo_orchard_sim.cfg_wrappers.envs.env_cfg import ViewerCfg
from robo_orchard_sim.cfg_wrappers.sim.schemas import (
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.cfg_wrappers.sim.spawners.from_files import (
    GroundPlaneCfg,
    UsdFileCfg,
)
from robo_orchard_sim.cfg_wrappers.sim.spawners.lights_cfg import (
    DomeLightCfg,
)
from robo_orchard_sim.models.assets.asset_cfg import (  # noqa: F401
    NV_ISAAC_DIR,
    ORCHARD_ASSET,
    AssetBaseCfg,
    GroupAssetCfg,
)
from robo_orchard_sim.models.assets.rigid_object import RigidObjectCfg
from robo_orchard_sim.models.assets.xform_asset import (  # noqa: E402
    XFormPrimAsset,
)
from robo_orchard_sim.orchard_env.assets import AssetSpec
from robo_orchard_sim.orchard_env.scene.scene_base import SceneBase

logger = LoggerManager().get_child(__name__)


class PlaneTableScene(SceneBase):
    """Scene provider with a default plane, table, and dome light."""

    def __init__(
        self,
        num_envs: int = 1,
        env_spacing: float = 2.5,
        physics_fps: int = 600,
        render_fps: int = 30,
        step_fps: int = 30,
        assets: list[AssetSpec] | None = None,
    ):
        super().__init__(
            num_envs=num_envs,
            env_spacing=env_spacing,
            physics_fps=physics_fps,
            render_fps=render_fps,
            step_fps=step_fps,
        )
        self.assets = list(assets or [])

    def get_assets_cfg(self) -> dict[str, GroupAssetCfg]:
        """Return scene-owned assets grouped by namespace."""
        grouped: dict[str, dict[str, object]] = {
            namespace: dict(group_cfg)
            for namespace, group_cfg in self._set_default_scene().items()
        }
        for spec in self.assets:
            spec = spec.with_default_namespace("background")
            grouped.setdefault(spec.namespace, {})
            if spec.name in grouped[spec.namespace]:
                logger.warning(
                    "Overriding default scene asset '%s' with external asset "
                    "spec.",
                    spec.scene_name,
                )
            grouped[spec.namespace][spec.name] = spec.to_isaac_cfg()
        return {
            namespace: GroupAssetCfg(**group_assets)
            for namespace, group_assets in grouped.items()
        }

    def get_viewer_cfg(self) -> ViewerCfg:
        """Return default viewer cfg used by the phase0 builder."""
        return ViewerCfg(
            eye=(1.5, 1.5, 3.0),
            lookat=(0.0, 0.0, 0.0),
        )

    def _set_default_scene(self) -> dict[str, GroupAssetCfg]:
        """Return the default plane-table scene assets."""
        return {
            "background": GroupAssetCfg(
                plane=AssetBaseCfg(
                    class_type=XFormPrimAsset,
                    prim_path="/World/GroundPlane",
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.64)),
                    spawn=GroundPlaneCfg(
                        color=(0, 0, 0),
                        usd_path=os.path.join(
                            NV_ISAAC_DIR,
                            "Environments/Grid/default_environment.usd",
                        ),
                    ),
                ),
                table=RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/table",
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=[0.4, 0.0, -0.32],
                        rot=[1, 0, 0, 0],
                    ),
                    spawn=UsdFileCfg(
                        usd_path=f"{ORCHARD_ASSET}/PUBLIC_OBJECTS/evaluation_assets/tables/table_001/table_001.usd",  # noqa: E501
                        scale=(25.0, 40.0, 16.0),
                        semantic_tags=[("class", "desk")],
                        # keep static
                        rigid_props=RigidBodyPropertiesCfg(
                            kinematic_enabled=True,
                        ),
                        # rigid_props=RigidBodyPropertiesCfg(
                        #     solver_position_iteration_count=4,
                        #     solver_velocity_iteration_count=1,
                        #     max_angular_velocity=0.01,
                        #     max_linear_velocity=0.01,
                        #     max_depenetration_velocity=1.0,
                        #     disable_gravity=False,
                        # ),
                        # mass_props=MassPropertiesCfg(mass=10000),
                    ),
                    object_elements_path=f"{ORCHARD_ASSET}/PUBLIC_OBJECTS/evaluation_assets/tables/table_001/table_interaction.json",  # noqa: E501
                ),
                light=AssetBaseCfg(
                    class_type=XFormPrimAsset,
                    prim_path="/World/light",
                    spawn=DomeLightCfg(
                        color=(0.75, 0.75, 0.75), intensity=3000.0
                    ),
                ),
            )
        }
