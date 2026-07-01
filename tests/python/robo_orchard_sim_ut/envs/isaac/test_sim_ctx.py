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


import time

import pytest
from omni.isaac.kit import SimulationApp

from robo_orchard_sim.ext.cfg_wrappers.sim.schemas.schemas_cfg import (
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.simulation_cfg import SimulationCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.from_files import (
    UsdFileCfg,
)
from robo_orchard_sim.ext.models.assets.asset_cfg import NV_ISAAC_DIR
from robo_orchard_sim.ext.models.scenes.interactive_scene import (
    InteractiveScene,
    InteractiveSceneCfg,
)
from robo_orchard_sim.ext.models.scenes.table_scene import (
    GroupAssetCfg,
    RigidObjectCfg,
    TableSceneCfg,
)
from robo_orchard_sim.sim_ctx import SimulationContextManager

FPS_THRESHOLD = 0.3


@pytest.fixture(scope="module")
def simple_scene_cfg() -> InteractiveSceneCfg:
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
    )
    return scene_cfg


class TestSimulationContext:
    def test_fps_no_renderer(
        self, app: SimulationApp, simple_scene_cfg: InteractiveSceneCfg
    ):
        with SimulationContextManager(
            cfg=SimulationCfg(dt=0.01),
            with_new_stage=True,
            disable_exit_on_stop=True,
        ) as sim:
            InteractiveScene(simple_scene_cfg)
            sim.reset()
            # sim_dt = sim.get_physics_dt()
            frame_num = 200
            # run the simulation for a few frames to get stable fps
            for _ in range(50):
                sim.step(render=False)
            start_time = time.time()
            for _ in range(frame_num):
                sim.step(render=False)

            end_time = time.time()
            real_fps = frame_num / (end_time - start_time)
            print(f"no render real_fps: {real_fps}")
            print(f"no render sim_fps: {sim.fps}")

            # assert min(real_fps / sim.fps, sim.fps / real_fps) > FPS_THRESHOLD # noqa: E501

    def test_fps_no_scene_update(
        self, app: SimulationApp, simple_scene_cfg: InteractiveSceneCfg
    ):
        with SimulationContextManager(
            cfg=SimulationCfg(dt=0.01),
            with_new_stage=True,
            disable_exit_on_stop=True,
        ) as sim:
            InteractiveScene(simple_scene_cfg)
            sim.reset()
            # sim_dt = sim.get_physics_dt()
            frame_num = 200
            # run the simulation for a few frames to get stable fps
            for _ in range(50):
                sim.step()
            start_time = time.time()
            for _ in range(frame_num):
                sim.step()

            end_time = time.time()
            real_fps = frame_num / (end_time - start_time)
            print(f"no scene update real_fps: {real_fps}")
            print(f"no scene update sim_fps: {sim.fps}")

            # assert min(real_fps / sim.fps, sim.fps / real_fps) > FPS_THRESHOLD # noqa: E501

    def test_fps(
        self, app: SimulationApp, simple_scene_cfg: InteractiveSceneCfg
    ):
        with SimulationContextManager(
            cfg=SimulationCfg(dt=0.01),
            with_new_stage=True,
            disable_exit_on_stop=True,
        ) as sim:
            scene = InteractiveScene(simple_scene_cfg)
            sim.reset()
            sim_dt = sim.get_physics_dt()
            frame_num = 200
            # run the simulation for a few frames to get stable fps
            for _ in range(50):
                sim.step()
                scene.update(sim_dt)
            start_time = time.time()
            for _ in range(frame_num):
                sim.step()
                # update the scene to reflect the physics changes
                scene.update(sim_dt)

            end_time = time.time()
            real_fps = frame_num / (end_time - start_time)
            print(f"real_fps: {real_fps}")
            print(f"sim_fps: {sim.fps}")

            # assert min(real_fps / sim.fps, sim.fps / real_fps) > FPS_THRESHOLD # noqa: E501


if __name__ == "__main__":
    pytest.main(["-s", __file__])
