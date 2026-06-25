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
from robo_orchard_core.envs.managers.actions.action_manager import (
    ActionManagerCfg,
)

from robo_orchard_sim.ext.cfg_wrappers.assets_cfg import RigidObjectCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.schemas.schemas_cfg import (
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.from_files import (
    UsdFileCfg,
)
from robo_orchard_sim.ext.envs.managers.observations import (
    ObservationGroupCfg,
    ObservationManagerCfg,
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
from robo_orchard_sim.orchard_env.embodiments.franka_panda.cfg import (
    FRANKA_PANDA_CFG,
)


@pytest.fixture()
def simple_obs_mgr_cfg() -> ObservationManagerCfg:
    a = ObservationManagerCfg(
        groups={"g0": ObservationGroupCfg(terms={})},
    )

    return a


@pytest.fixture()
def simple_act_mgr_cfg() -> ActionManagerCfg:
    return ActionManagerCfg(
        terms={},
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
def simple_table_scene_cfg_with_camera(
    simple_table_scene_cfg: TableSceneCfg,
) -> TableSceneCfg:
    ret = simple_table_scene_cfg.copy()
    ret.cameras = GroupAssetCfg(
        rgb_camera=CameraCfg(
            prim_path="{ENV_REGEX_NS}/camera",
            class_type=Camera,
            offset=CameraOffset(
                trans=(0.5, -0.45, 1),
                toward_target=(0.5, 0, 0),
            ),
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=PinholeCameraCfg(
                focal_length=15.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
        )
    )
    return ret


@pytest.fixture()
def simple_two_object_scene_cfg(
    simple_table_scene_cfg: TableSceneCfg,
) -> TableSceneCfg:
    ret = simple_table_scene_cfg.copy()
    ret.objects["cube2"] = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, 0.2, 0.1), rot=(1, 0, 0, 0)
        ),
        spawn=ret.objects["cube1"].spawn,
    )
    return ret


@pytest.fixture()
def franka_table() -> TableSceneCfg:
    """Fixture for a table scene with a Franka Panda robot.

    We use FRANKA_PANDA_HIGH_PD_CFG for better control of the robot.
    """
    scene_cfg = TableSceneCfg(
        num_envs=1,
        env_spacing=2,
        # replace prim_path with your own robot prim_path
        robots={
            "robot_franka": FRANKA_PANDA_CFG.replace(
                prim_path="{ENV_REGEX_NS}/robot_franka"
            )
        },
    )
    return scene_cfg
