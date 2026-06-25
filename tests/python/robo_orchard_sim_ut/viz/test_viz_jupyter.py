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
import time

import pytest
from omni.isaac.kit import SimulationApp

from robo_orchard_sim.ext.cfg_wrappers.assets_cfg import RigidObjectCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.schemas.schemas_cfg import (
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.simulation_cfg import SimulationCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.from_files import (
    UsdFileCfg,
)
from robo_orchard_sim.ext.envs.env_base import (
    IsaacEnv,
    IsaacEnvCfg,
    IsaacEnvContextManager,
)
from robo_orchard_sim.ext.models.assets.asset_cfg import (
    NV_ISAAC_DIR,
    GroupAssetCfg,
)
from robo_orchard_sim.ext.models.scenes.table_scene import TableSceneCfg
from robo_orchard_sim.ext.models.sensors.camera import (
    Camera,
    CameraCfg,
    CameraOffset,
    PinholeCameraCfg,
)
from robo_orchard_sim.viz.jupyter.viewports import IsaacIpyViewportViz


@pytest.fixture()
def simple_table_with_cam_fixture() -> TableSceneCfg:
    usd_cfg = UsdFileCfg(
        usd_path=f"{NV_ISAAC_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",  # noqa
        scale=(0.8, 0.8, 0.8),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    )
    scene_cfg = TableSceneCfg(
        num_envs=1,
        env_spacing=2,
        objects=GroupAssetCfg(
            cube1=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.5, 0, 0.555), rot=(1, 0, 0, 0)
                ),
                spawn=usd_cfg,
            )
        ),
        cameras=CameraCfg(
            prim_path="{ENV_REGEX_NS}/camera",
            class_type=Camera,
            offset=CameraOffset(
                xyz=(0.5, -0.45, 1),
                toward_target=(0.5, 0, 0),
            ),
            height=480,
            width=640,
            data_types=[
                "rgb",
            ],
            spawn=PinholeCameraCfg(
                focal_length=15.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
        ),
    )

    return scene_cfg


class TestIsaacIpyViewportViz:
    def test_init(
        self, simple_table_with_cam_fixture: TableSceneCfg, app: SimulationApp
    ):
        scene_cfg = simple_table_with_cam_fixture

        env_cfg: IsaacEnvCfg[IsaacEnv, TableSceneCfg] = IsaacEnvCfg(
            class_type=IsaacEnv,
            sim=SimulationCfg(dt=0.01),
            decimation=1,
            scene=scene_cfg,
        )

        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            env.reset()
            viz = IsaacIpyViewportViz(
                height=480,
                width=640,
                sim_ctx=env.sim,
            )
            env.step()
            pose = viz.get_pose_view_world()
            print(pose)
            time.sleep(0.01)
            img = viz.get_rendered_image()
            assert img.shape[0] != 0
