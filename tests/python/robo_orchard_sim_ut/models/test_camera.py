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


import allure
import cv2
import pytest
from omni.isaac.kit import SimulationApp

from robo_orchard_sim.ext.cfg_wrappers.envs.env_cfg import SimulationCfg
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
from robo_orchard_sim.ext.models.sensors.camera import (
    Camera,
    CameraCfg,
    CameraOffset,
    FisheyeCameraCfg,
    PinholeCameraCfg,
)
from robo_orchard_sim.sim_ctx import SimulationContextManager


class TestCameraOffsetConfig:
    def test_camera_offset_with_toward_target(self):
        camera_offset = CameraOffset(
            trans=(1.0, 2.0, 3.0),
            toward_target=(0.0, 0.0, 1.0),
        )
        print(
            "camera_offset.rot: ",
            camera_offset.rot,
            type(camera_offset.rot),
            type((1.0, 0.0, 0.0, 0.0)),
        )
        assert camera_offset.rot != (1.0, 0.0, 0.0, 0.0)

    def test_camera_offset_rot_conflict(self):
        with pytest.raises(ValueError) as excinfo:
            camera_offset = CameraOffset(
                trans=(1.0, 2.0, 3.0),
                rot=(0, 1, 0, 0),
                toward_target=(0.0, 0.0, 1.0),
            )
            print(camera_offset)
        print("Get expected error:", excinfo)

    def test_camera_offset_serialization(self):
        camera_offset = CameraOffset(
            trans=(1.0, 2.0, 3.0),
            rot=(1.0, 0, 0, 0),
        )

        json_str = camera_offset.to_str(format="json")
        print(json_str)
        cam_offset_from_json = CameraOffset.from_str(json_str, format="json")
        print(cam_offset_from_json)
        assert camera_offset.content_equal(cam_offset_from_json)


class TestCameraConfig:
    @pytest.mark.parametrize(
        "spawn",
        [
            pytest.param(
                PinholeCameraCfg(
                    focal_length=15.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1.0e5),
                ),
                id="PinholeCameraCfg",
            ),
            pytest.param(
                FisheyeCameraCfg(
                    focal_length=15.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1.0e5),
                ),
                id="FisheyeCameraCfg",
            ),
        ],
    )
    def test_camera_cfg_serialization(
        self, spawn: PinholeCameraCfg | FisheyeCameraCfg
    ):
        camera_cfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/camera",
            class_type=Camera,
            offset=CameraOffset(
                trans=(0.5, -0.45, 1),
                toward_target=(0.5, 0, 0),
            ),
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=spawn,
        )
        camera_cfg_json = camera_cfg.to_str(format="json")
        print(camera_cfg_json)
        camera_cfg_from_json = CameraCfg.from_str(
            camera_cfg_json, format="json"
        )
        print(camera_cfg_from_json.to_str(format="json"))
        assert camera_cfg.content_equal(camera_cfg_from_json)


class TestSceneWithCamera:
    @pytest.mark.parametrize(
        "camera_cfg",
        [
            CameraCfg(
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
            ),
            CameraCfg(
                prim_path="{ENV_REGEX_NS}/camera",
                class_type=Camera,
                offset=CameraOffset(
                    trans=(0.5, -0.45, 1),
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
            ),
        ],
    )
    def test_camera_toward_target(
        self, app: SimulationApp, camera_cfg: CameraCfg
    ):
        sim_cfg = SimulationCfg(dt=0.01)
        with SimulationContextManager(
            sim_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as sim:
            obj_usd = UsdFileCfg(
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
                env_spacing=2.0,
                objects=GroupAssetCfg(
                    object=RigidObjectCfg(
                        prim_path="{ENV_REGEX_NS}/Object",
                        init_state=RigidObjectCfg.InitialStateCfg(
                            # pos=[0.5, 0, 0.555],
                            pos=(0.5, 0, 0.03),  # on the table
                            rot=(1, 0, 0, 0),
                        ),
                        spawn=obj_usd,
                    )
                ),
                cameras=GroupAssetCfg(rgb_camera=camera_cfg),
            )
            scene = InteractiveScene(scene_cfg)
            sim.reset()
            assert app.is_running()
            sim_dt = sim.get_physics_dt()
            # step the simulation for 20 frames to get stable camera image
            for _ in range(60):
                sim.step()
                scene.update(sim_dt)

            sensor: Camera = scene.sensors["cameras/rgb_camera"]  # type: ignore
            sensor_data = sensor.data
            print("fps: ", sim.fps)
            print("current time: ", sim.current_time)
            print("sensor: ", scene.sensors)
            print("sensor intrinsics: ", sensor_data.intrinsic_matrices)
            print(f"sensor position: {sensor_data.pos_w}")
            print(f"sensor rotation: {sensor_data.quat_w_world}")
            sensor_img_np = (
                sensor_data.output["rgb"].cpu().numpy()[0, :, :, 0:3]
            )
            # convert to BGR
            sensor_img_np = sensor_img_np[:, :, ::-1]
            allure.attach(
                cv2.imencode(".png", sensor_img_np)[1].tobytes(),
                name="sensor_img",
                attachment_type=allure.attachment_type.PNG,
            )
            scene.delete_all_assets()

            # sensor.__del__()

    @pytest.mark.parametrize(
        "camera_cfg",
        [
            CameraCfg(
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
            ),
        ],
    )
    def test_camera_orchard_mixin(
        self, app: SimulationApp, camera_cfg: CameraCfg
    ):
        sim_cfg = SimulationCfg(dt=0.01)
        with SimulationContextManager(
            sim_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as sim:
            obj_usd = UsdFileCfg(
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
                env_spacing=2.0,
                objects=GroupAssetCfg(
                    object=RigidObjectCfg(
                        prim_path="{ENV_REGEX_NS}/Object",
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=(0.5, 0, 0.03),  # on the table
                            rot=(1, 0, 0, 0),
                        ),
                        spawn=obj_usd,
                    )
                ),
                cameras=GroupAssetCfg(rgb_camera=camera_cfg),
            )
            scene = InteractiveScene(scene_cfg)
            sim.reset()
            assert app.is_running()
            sim_dt = sim.get_physics_dt()
            # step the simulation for 20 frames to get stable camera image
            for _ in range(60):
                sim.step()
                scene.update(sim_dt)

            sensor: Camera = scene.sensors["cameras/rgb_camera"]  # type: ignore
            camera_data = sensor.get_camera_data()
            print("camera_data: ", camera_data)

            # check topic
            for topic in camera_cfg.data_types:
                assert topic in camera_data.keys()
                assert camera_data[topic].topic == topic

            scene.delete_all_assets()


if __name__ == "__main__":
    pytest.main(["-s", __file__])
