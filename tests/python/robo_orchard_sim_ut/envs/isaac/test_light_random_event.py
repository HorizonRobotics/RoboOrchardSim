# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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

import pytest
import torch
from pxr import Gf, UsdGeom
from robo_orchard_core.envs.managers.events.event_manager import (
    EventManagerCfg,
)

from robo_orchard_sim.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.cfg_wrappers.sim.spawners.lights_cfg import (
    DistantLightCfg,
)
from robo_orchard_sim.envs import (
    IsaacEnvContextManager,
    IsaacManagerBasedEnv,
    IsaacManagerBasedEnvCfg,
)
from robo_orchard_sim.envs.managers.events.light_reset import (
    LightPoseCfg,
    LightResetTermCfg,
    RangeCfg,
)
from robo_orchard_sim.models.assets.asset_cfg import (
    AssetBaseCfg,
    asset_replace_usd_path,
)
from robo_orchard_sim.models.assets.xform_asset import XFormPrimAsset
from robo_orchard_sim.models.scenes.table_scene import TableSceneCfg


class TestLightResetEvent:
    def test_light_reset(
        self,
        simple_table_scene_cfg_with_camera: TableSceneCfg,
    ):
        """Test color/intensity/position randomization in one simulation."""
        scene_cfg = simple_table_scene_cfg_with_camera.copy()
        scene_cfg.light = asset_replace_usd_path(
            AssetBaseCfg(
                class_type=XFormPrimAsset,
                prim_path="/World/light",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
                spawn=DistantLightCfg(
                    color=(0.75, 0.75, 0.75), intensity=3000.0
                ),
            )
        )
        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=scene_cfg,
            events=EventManagerCfg(
                terms={
                    "light_reset": LightResetTermCfg(
                        trigger_topic="reset",
                        asset_cfgs=[SceneEntityCfg(name="light")],
                        randomize_color=True,
                        color_temperature_range=RangeCfg(
                            range=(2000.0, 8000.0)
                        ),
                        randomize_intensity=True,
                        intensity_range=RangeCfg(range=(1000.0, 5000.0)),
                        randomize_position=True,
                        position_cfg=LightPoseCfg(
                            center_pose=(0.0, 0.0, 0.0),
                            radius=0.5,
                            elevation=RangeCfg(range=(0.0, 1.57)),
                        ),
                    )
                },
            ),
        )
        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            _ = env.step()

            # Get the light prim
            light_prim = env.sim.stage.GetPrimAtPath("/World/light")
            assert light_prim.IsValid(), "Light prim not found at /World/light"

            # Get initial color
            color_attr = light_prim.GetAttribute("inputs:color")
            assert color_attr, "Light prim has no color attribute"
            initial_color = color_attr.Get()
            assert isinstance(initial_color, Gf.Vec3f)

            # Get initial intensity
            intensity_attr = light_prim.GetAttribute("inputs:intensity")
            assert intensity_attr, "Light prim has no intensity attribute"
            initial_intensity = intensity_attr.Get()
            assert isinstance(initial_intensity, (int, float))

            # Get initial position from USD
            xformable = UsdGeom.Xformable(light_prim)
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            initial_position = torch.tensor(
                list(world_transform.ExtractTranslation()),
                dtype=torch.float32,
                device=env.device,
            ).unsqueeze(0)

            # Reset and verify all three dimensions changed
            env.reset(seed=123)
            for _ in range(5):
                _ = env.step()

            new_color = color_attr.Get()
            assert isinstance(new_color, Gf.Vec3f)

            new_intensity = intensity_attr.Get()
            assert isinstance(new_intensity, (int, float))

            # Get new position from USD (root_state_w may be stale for lights)
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            new_position = torch.tensor(
                list(world_transform.ExtractTranslation()),
                dtype=torch.float32,
                device=env.device,
            ).unsqueeze(0)

            # Verify color has changed
            assert not torch.allclose(
                torch.tensor(
                    [initial_color[0], initial_color[1], initial_color[2]]
                ),
                torch.tensor([new_color[0], new_color[1], new_color[2]]),
                rtol=1e-05,
                atol=1e-08,
            ), "Light color should have changed after reset"

            # Verify intensity has changed
            assert not torch.allclose(
                torch.tensor([float(initial_intensity)]),
                torch.tensor([float(new_intensity)]),
                rtol=1e-05,
                atol=1e-08,
            ), "Light intensity should have changed after reset"

            # Verify position has changed
            assert not torch.allclose(
                initial_position,
                new_position,
                rtol=1e-05,
                atol=1e-08,
            ), "Light position should have changed after reset"


if __name__ == "__main__":
    pytest.main(["-s", "test_light_random_event.py"])
