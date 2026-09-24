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

"""Resolver-backed definition for the affordance benchmark."""

from pathlib import Path
from typing import TYPE_CHECKING

from robo_orchard_sim.benchmark.base import TaskDefinition
from robo_orchard_sim.benchmark.registration import register_task

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
        AssetResolver,
    )
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv

_CONFIG_PATH = Path(__file__).resolve().parent / "configs/affordance.yaml"


@register_task
class AffordanceTaskDefinition(TaskDefinition):
    """Build the single-target semantic-affordance benchmark."""

    namespace = "affordance"
    config_path = str(_CONFIG_PATH)

    @staticmethod
    def _apply_light_reset_scene_overrides(scene, task_params) -> None:
        """Inject scene light overrides required by light randomization."""
        light_reset = task_params.light_reset
        if light_reset is None or not light_reset.enabled:
            return
        if not hasattr(scene, "assets"):
            return

        distant_light = light_reset.distant_light
        if distant_light is None:
            return

        from robo_orchard_sim.ext.cfg_wrappers.assets_cfg import AssetBaseCfg
        from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.lights_cfg import (
            DistantLightCfg,
            DomeLightCfg,
        )
        from robo_orchard_sim.ext.models.assets.xform_asset import (
            XFormPrimAsset,
        )
        from robo_orchard_sim.orchard_env.assets import CustomAssetSpec

        scene.assets.append(
            CustomAssetSpec(
                name="light",
                namespace="background",
                cfg=AssetBaseCfg(
                    class_type=XFormPrimAsset,
                    prim_path="/World/light",
                    spawn=DomeLightCfg(
                        color=(0.75, 0.75, 0.75),
                        intensity=3000.0,
                        visible=False,
                    ),
                ),
            )
        )
        scene.assets.append(
            CustomAssetSpec(
                name=distant_light.asset_name,
                namespace="background",
                cfg=AssetBaseCfg(
                    class_type=XFormPrimAsset,
                    prim_path=distant_light.prim_path,
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=distant_light.init_pos,
                        rot=distant_light.init_rot,
                    ),
                    spawn=DistantLightCfg(
                        color=distant_light.color,
                        intensity=distant_light.intensity,
                    ),
                ),
            )
        )

    @classmethod
    def build(
        cls,
        resolver: "AssetResolver | None" = None,
        config_path: str | None = None,
    ) -> "OrchardEnv":
        """Build an affordance environment from resolved assets."""
        from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets
        from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
        from robo_orchard_sim.orchard_env.task_templates.articulated.affordance_task import (  # noqa: E501
            AffordanceTask,
            AffordanceTaskParams,
        )

        if resolver is None:
            raise ValueError(
                "AffordanceTaskDefinition.build() requires an AssetResolver."
            )
        asset_configs = cls.resolve_asset_configs(config_path=config_path)
        if asset_configs is None:
            raise ValueError(
                "AffordanceTaskDefinition needs asset_configs in "
                f"{config_path or cls.config_path}."
            )

        resolved = resolver.resolve(asset_configs)
        task_params = AffordanceTaskParams(
            **cls.resolve_task_params(config_path=config_path)
        )
        scene = cls.resolve_scene(config_path=config_path)
        cls._apply_light_reset_scene_overrides(scene, task_params)

        return OrchardEnv(
            scene=scene,
            embodiment=cls.resolve_embodiment(config_path=config_path),
            task=AffordanceTask(
                assets=TaskAssets.from_resolved(resolved),
                params=task_params,
                instruction=cls.resolve_instruction(config_path=config_path),
            ),
        )


__all__ = ["AffordanceTaskDefinition"]
