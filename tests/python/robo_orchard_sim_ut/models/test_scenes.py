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

import pytest
from isaaclab.sensors.contact_sensor import ContactSensorCfg

from robo_orchard_sim.ext.cfg_wrappers.envs.env_cfg import SimulationCfg
from robo_orchard_sim.ext.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.schemas.schemas_cfg import (
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.from_files import (
    UsdFileCfg,
)
from robo_orchard_sim.ext.models.assets.asset_cfg import NV_ISAAC_DIR
from robo_orchard_sim.ext.models.scenes.interactive_scene import (
    InteractiveScene,
)
from robo_orchard_sim.ext.models.scenes.table_scene import (
    GroupAssetCfg,
    RigidObjectCfg,
    TableSceneCfg,
)
from robo_orchard_sim.orchard_env.embodiments.franka_panda.cfg import (
    FRANKA_PANDA_CFG,
)
from robo_orchard_sim.sim_ctx import SimulationContextManager


@pytest.fixture(scope="module")
def simple_table_scene_cfg() -> TableSceneCfg:
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


@pytest.fixture(scope="module")
def simple_scene_entity_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        name="cube1",
    )


class TestInteractiveScene:
    def _test_asset_by_cfg(self, scene_cfg: TableSceneCfg, app):
        sim_cfg = SimulationCfg(dt=0.01)

        with SimulationContextManager(
            cfg=sim_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as sim:
            self.sim = sim

            scene = InteractiveScene(scene_cfg)
            print("[INFO]: Scene assets: ")
            for asset in scene.keys():
                print(f"\t{asset}")

            self.sim.reset()
            sim_dt = self.sim.get_physics_dt()
            obj_name = "objects/cube1"
            obj = scene.rigid_objects[obj_name]
            init_pos = obj.data.root_pos_w
            for _ in range(20):
                assert app.is_running()
                self.sim.step()
                # update the scene to reflect the physics changes
                scene.update(sim_dt)

            end_pos = obj.data.root_pos_w
            end_vel = obj.data.root_lin_vel_w
            print(
                f"Object {obj_name} initial position: {init_pos}, final position: {end_pos}, velocity: {end_vel}"  # noqa
            )
            # object should fall down
            # self.assertGreater(end_pos[0, 2], init_pos[0, 2])
            assert end_pos[0, 2] < init_pos[0, 2]
            assert end_vel[0, 2] < 0.0

            # self.sim.stop()
            # self.sim._app.update()
            # scene.clear_stage()

    def test_group_asset_cfg(self, app):
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
        self._test_asset_by_cfg(scene_cfg, app)

    def test_with_robot(self, app):
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
            # replace prim_path with your own robot prim_path
            robots=FRANKA_PANDA_CFG.replace(  # type: ignore
                prim_path="{ENV_REGEX_NS}/robot_franka"
            ),
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
        self._test_asset_by_cfg(scene_cfg, app)

    def test_scene_with_ContactSensor(self, app):
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
            activate_contact_sensors=True,
        )
        scene_cfg = TableSceneCfg(
            num_envs=1,
            env_spacing=2,
            # robots=FRANKA_PANDA_CFG.replace(  # type: ignore
            #     prim_path="{ENV_REGEX_NS}/robot_franka",
            #     activate_contact_sensors=True,
            # ),
            objects=GroupAssetCfg(
                cube1=RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(0.5, 0, 0.555), rot=(1, 0, 0, 0)
                    ),
                    spawn=usd_cfg,
                ),
                contact_sensor=ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                ),
            ),
        )
        self._test_asset_by_cfg(scene_cfg, app)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
