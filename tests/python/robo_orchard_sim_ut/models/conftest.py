# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
#
#
#       http://www.apache.org/licenses/LICENSE-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# You may obtain a copy of the License at
# distributed under the License is distributed on an "AS IS" BASIS,
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# you may not use this file except in compliance with the License.

import pytest
from robo_orchard_core.utils.config import Config

from robo_orchard_sim.cfg_wrappers.assets_cfg import RigidObjectCfg
from robo_orchard_sim.cfg_wrappers.envs.env_cfg import ViewerCfg
from robo_orchard_sim.cfg_wrappers.managers.manager_term_cfg import (
    ManagerTermBaseCfg,
)
from robo_orchard_sim.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.cfg_wrappers.sim.schemas.schemas_cfg import (
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.cfg_wrappers.sim.simulation_cfg import (
    PhysxCfg,
    SimulationCfg,
)
from robo_orchard_sim.cfg_wrappers.sim.spawners import (
    FisheyeCameraCfg,
    PinholeCameraCfg,
    UsdFileCfg,
)
from robo_orchard_sim.models.assets.asset_cfg import NV_ISAAC_DIR
from robo_orchard_sim.models.scenes.table_scene import (
    GroupAssetCfg,
    TableSceneCfg,
)


@pytest.fixture()
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


@pytest.fixture()
def simple_scene_entity_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        name="cube1",
    )


@pytest.fixture()
def simple_physx_cfg() -> PhysxCfg:
    return PhysxCfg()


@pytest.fixture()
def simple_simulation_cfg() -> SimulationCfg:
    return SimulationCfg()


@pytest.fixture()
def simple_viewer_cfg() -> ViewerCfg:
    return ViewerCfg()


def func():
    pass


@pytest.fixture()
def simple_manager_term_cfg() -> ManagerTermBaseCfg:
    return ManagerTermBaseCfg(func=func)


@pytest.fixture()
def simple_pinhole_camera_cfg() -> PinholeCameraCfg:
    return PinholeCameraCfg()


@pytest.fixture()
def simple_fisheye_camera_cfg() -> FisheyeCameraCfg:
    return FisheyeCameraCfg()


@pytest.fixture(
    params=[
        "simple_table_scene_cfg",
        "simple_scene_entity_cfg",
        "simple_physx_cfg",
        "simple_simulation_cfg",
        "simple_viewer_cfg",
        "simple_manager_term_cfg",
        "simple_pinhole_camera_cfg",
        "simple_fisheye_camera_cfg",
    ],
)
def simple_isaac_wrapped_cfg(request) -> Config:
    """Fixture for Isaac wrapped cfgs.

    This fixure will be used for config test.
    """
    return request.getfixturevalue(request.param)
