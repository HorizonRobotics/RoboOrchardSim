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

"""Room-and-table scene provider."""

from robo_orchard_core.utils.logging import LoggerManager

from robo_orchard_sim.ext.cfg_wrappers.envs.env_cfg import ViewerCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.schemas import (
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.simulation_cfg import (
    PhysxCfg,
    RenderCfg,
    SimulationCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.from_files import (
    UsdFileCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.lights_cfg import (
    DomeLightCfg,
)
from robo_orchard_sim.ext.models.assets.asset_cfg import (
    ORCHARD_ASSET,
    AssetBaseCfg,
    GroupAssetCfg,
)
from robo_orchard_sim.ext.models.assets.rigid_object import RigidObjectCfg
from robo_orchard_sim.ext.models.assets.xform_asset import XFormPrimAsset
from robo_orchard_sim.orchard_env.assets import AssetSpec
from robo_orchard_sim.orchard_env.scene.scene_base import SceneBase

logger = LoggerManager().get_child(__name__)


class RoomTableScene(SceneBase):
    """Scene provider with a DeQing room, table, and dome light."""

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

    # override
    def get_sim_cfg(self) -> SimulationCfg:
        """Return scene-level simulation cfg."""
        return SimulationCfg(
            render_interval=self.get_render_interval(),
            dt=1.0 / self.physics_fps,
            physx=PhysxCfg(enable_ccd=True),
            # 5090
            render=RenderCfg(
                # enable_translucency=True,
                # enable_reflections=True,
                # enable_global_illumination=True,
                # enable_ambient_occlusion=True,
                enable_dlssg=False,
                # enable_dl_denoiser=True,
                antialiasing_mode="FXAA",
                dlss_mode=3,
                samples_per_pixel=64,
            ),
            # 4090
            # render=RenderCfg(
            #     # enable_translucency=True,
            #     # enable_reflections=True,
            #     # enable_global_illumination=True,
            #     # enable_ambient_occlusion=True,
            #     enable_dlssg=True,
            #     enable_dl_denoiser=True,
            #     # antialiasing_mode="FXAA",
            #     dlss_mode=3,
            #     # samples_per_pixel=4
            # ),
        )

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
        """Return default viewer cfg for the room-table scene."""
        return ViewerCfg(
            eye=(1.2, 2.0, 1.4),
            lookat=(0.0, 0.0, 0.0),
        )

    def _set_default_scene(self) -> dict[str, GroupAssetCfg]:
        """Return the default room-table scene assets."""
        return {
            "background": GroupAssetCfg(
                room=AssetBaseCfg(
                    class_type=XFormPrimAsset,
                    prim_path="{ENV_REGEX_NS}/room",
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=(-2.55, 0.0, -0.64),
                    ),
                    spawn=UsdFileCfg(
                        usd_path=f"{ORCHARD_ASSET}/BACKGROUND/Collected_DeQing_HomeScene_1F/DeQing_HomeScene_1F.usd",
                        semantic_tags=[("class", "room")],
                        rigid_props=RigidBodyPropertiesCfg(
                            kinematic_enabled=True,
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
                        rigid_props=RigidBodyPropertiesCfg(
                            kinematic_enabled=True,
                        ),
                    ),
                    interaction_path=f"{ORCHARD_ASSET}/PUBLIC_OBJECTS/evaluation_assets/tables/table_001/table_interaction.json",  # noqa: E501
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
