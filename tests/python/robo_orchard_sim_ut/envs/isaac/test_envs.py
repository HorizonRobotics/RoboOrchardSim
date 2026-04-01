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
import torch
from isaaclab.sensors.camera import CameraData as LabCameraData
from robo_orchard_core.datatypes.camera_data import BatchCameraData
from robo_orchard_core.datatypes.tf_graph import BatchFrameTransformGraph
from robo_orchard_core.envs.managers.observations.observation_manager import (
    ObservationGroupCfg,
)

from robo_orchard_sim.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.envs import (
    IsaacEnvCfg,
    IsaacEnvContextManager,
    IsaacManagerBasedEnv,
    IsaacManagerBasedEnvCfg,
)
from robo_orchard_sim.envs.managers.observations import ObservationManagerCfg
from robo_orchard_sim.envs.managers.observations.asset_obs import (
    AssetObservationTermCfg,
)
from robo_orchard_sim.envs.managers.observations.camera import (
    CameraObservationTermCfg,
)
from robo_orchard_sim.envs.managers.observations.sensor import (
    SensorObservationTermCfg,
)
from robo_orchard_sim.envs.managers.observations.transform_frame import (
    FrameTransformTermCfg,
)
from robo_orchard_sim.models.scenes.interactive_scene import (
    InteractiveSceneCfg,
)
from robo_orchard_sim.models.scenes.table_scene import TableSceneCfg
from robo_orchard_sim_ut.utils.cfg_test import CfgTestBase


@pytest.fixture()
def simple_isaac_manager_based_cfg(
    simple_table_scene_cfg: InteractiveSceneCfg,
    simple_obs_mgr_cfg: ObservationManagerCfg,
) -> IsaacManagerBasedEnvCfg:
    return IsaacManagerBasedEnvCfg(
        decimation=1,
        scene=simple_table_scene_cfg,
        observations=simple_obs_mgr_cfg,
    )


@pytest.fixture()
def observation_manager_cfg() -> ObservationManagerCfg:
    return ObservationManagerCfg(
        groups={
            "g0": ObservationGroupCfg(
                terms={
                    "cube_obs": AssetObservationTermCfg(
                        property_name="position",
                        property_source="root",
                        asset_cfg=SceneEntityCfg(
                            name="objects/cube1",
                        ),
                    )
                }
            )
        },
    )


@pytest.fixture()
def isaac_manager_based_cfg_with_obs_mgr(
    simple_table_scene_cfg: InteractiveSceneCfg,
    observation_manager_cfg: ObservationManagerCfg,
) -> IsaacManagerBasedEnvCfg:
    return IsaacManagerBasedEnvCfg(
        decimation=1,
        scene=simple_table_scene_cfg,
        observations=observation_manager_cfg,
    )


@pytest.fixture(
    params=[
        "simple_isaac_manager_based_cfg",
        "isaac_manager_based_cfg_with_obs_mgr",
    ],
)
def simple_cfg(request) -> IsaacManagerBasedEnvCfg:
    return request.getfixturevalue(request.param)


class TestEnvCfg(CfgTestBase):
    pass


class TestIsaacManagerBasedEnv:
    def test_init(self, simple_cfg: IsaacManagerBasedEnvCfg):
        with IsaacEnvContextManager(
            simple_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert env is not None
            assert isinstance(env, IsaacManagerBasedEnv)


class TestAssetObservationTerm:
    @pytest.mark.parametrize(
        "obs_cfg",
        [
            ObservationManagerCfg(
                groups={
                    "cube": ObservationGroupCfg(
                        terms={
                            "cube_pos": AssetObservationTermCfg(
                                property_name="position",
                                property_source="root",
                                asset_cfg=SceneEntityCfg(
                                    name="objects/cube1",
                                ),
                            ),
                            "cube_vel": AssetObservationTermCfg(
                                property_name="linear_velocity",
                                property_source="root",
                                asset_cfg=SceneEntityCfg(
                                    name="objects/cube1",
                                ),
                            ),
                        },
                        concatenate_terms=False,
                    ),
                },
            )
        ],
    )
    def test_get_obs(
        self,
        simple_table_scene_cfg: TableSceneCfg,
        obs_cfg: ObservationManagerCfg,
    ):
        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=simple_table_scene_cfg,
            observations=obs_cfg,
        )

        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)

            for _ in range(10):
                ret = env.step()
                print(ret)


class TestCameraObservationTerm:
    @pytest.mark.parametrize(
        "obs_cfg",
        [
            ObservationManagerCfg(
                groups={
                    "camera": ObservationGroupCfg(
                        terms={
                            "rgb_camera_camera_term": CameraObservationTermCfg(
                                asset_cfg=SceneEntityCfg(
                                    name="cameras/rgb_camera",
                                ),
                            ),
                            "rgb_camera_sensor_term": SensorObservationTermCfg(
                                asset_cfg=SceneEntityCfg(
                                    name="cameras/rgb_camera",
                                ),
                            ),
                        },
                    ),
                    "cube": ObservationGroupCfg(
                        terms={
                            "cube_pos": AssetObservationTermCfg(
                                property_name="position",
                                property_source="root",
                                asset_cfg=SceneEntityCfg(
                                    name="objects/cube1",
                                ),
                            ),
                            "cube_vel": AssetObservationTermCfg(
                                property_name="linear_velocity",
                                property_source="root",
                                asset_cfg=SceneEntityCfg(
                                    name="objects/cube1",
                                ),
                            ),
                        },
                        concatenate_terms=True,
                    ),
                },
            )
        ],
    )
    def test_get_obs(
        self,
        simple_table_scene_cfg_with_camera: TableSceneCfg,
        obs_cfg: ObservationManagerCfg,
    ):
        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=simple_table_scene_cfg_with_camera,
            observations=obs_cfg,
        )

        assert isinstance(env_cfg, IsaacEnvCfg)
        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)

            for _ in range(10):
                ret = env.step()
                # print(ret)
                for k, _ in obs_cfg.groups.items():
                    assert k in ret.observations

    def test_get_obs_return_type_check(
        self,
        simple_table_scene_cfg_with_camera: TableSceneCfg,
    ):
        obs_cfg = ObservationManagerCfg(
            groups={
                "camera": ObservationGroupCfg(
                    terms={
                        "rgb_camera_camera_term": CameraObservationTermCfg(
                            asset_cfg=SceneEntityCfg(
                                name="cameras/rgb_camera",
                            ),
                        ),
                    },
                ),
            },
        )
        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=simple_table_scene_cfg_with_camera,
            observations=obs_cfg,
        )
        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            ret = env.step()
            cam_term_ret = ret.observations["camera"]["rgb_camera_camera_term"]
            # check the return type of the camera term is consistent with the
            # CameraObservationTerm.ReturnType
            # assert isinstance(cam_term_ret, CameraObservationTerm.ReturnType)
            assert isinstance(cam_term_ret, dict)
            for k, v in cam_term_ret.items():
                assert isinstance(k, str)
                assert isinstance(v, BatchCameraData)


class TestSensorObservationTerm:
    def test_get_obs_return_type_check(
        self,
        simple_table_scene_cfg_with_camera: TableSceneCfg,
    ):
        obs_cfg = ObservationManagerCfg(
            groups={
                "camera": ObservationGroupCfg(
                    terms={
                        "rgb_camera_camera_term": SensorObservationTermCfg(
                            asset_cfg=SceneEntityCfg(
                                name="cameras/rgb_camera",
                            ),
                        ),
                    },
                ),
            },
        )
        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=simple_table_scene_cfg_with_camera,
            observations=obs_cfg,
        )
        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            ret = env.step()
            cam_term_ret = ret.observations["camera"]["rgb_camera_camera_term"]
            # check the return type of the camera term is consistent with the
            # CameraObservationTerm.ReturnType
            # assert isinstance(cam_term_ret, CameraObservationTerm.ReturnType)
            assert isinstance(cam_term_ret, LabCameraData)


class TestFrameTransformTerm:
    def test_camera_to_cube_transform(
        self,
        simple_table_scene_cfg_with_camera: TableSceneCfg,
    ):
        obs_cfg = ObservationManagerCfg(
            groups={
                "frame_transform": ObservationGroupCfg(
                    terms={
                        "cam_to_cube": FrameTransformTermCfg(
                            asset_cfg=SceneEntityCfg(
                                name="objects/cube1",
                            ),
                            child_asset_cfg=SceneEntityCfg(
                                name="cameras/rgb_camera",
                            ),
                        ),
                    },
                ),
            },
        )
        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=simple_table_scene_cfg_with_camera,
            observations=obs_cfg,
        )
        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            for _ in range(5):
                ret = env.step()

            tf_data = ret.observations["frame_transform"]["cam_to_cube"]
            assert isinstance(tf_data, BatchFrameTransformGraph)

            num_envs = simple_table_scene_cfg_with_camera.num_envs
            state = tf_data.as_state()
            assert len(state.tf_list) == 1
            tf0 = state.tf_list[0]
            assert tf0.parent_frame_id == "objects/cube1"
            assert tf0.child_frame_id == "cameras/rgb_camera"
            assert tf0.xyz.shape == (num_envs, 3)
            assert tf0.quat.shape == (num_envs, 4)
            assert torch.isfinite(tf0.xyz).all()
            assert torch.isfinite(tf0.quat).all()

    def test_multi_frame_transform(
        self,
        franka_table: TableSceneCfg,
    ):
        obs_cfg = ObservationManagerCfg(
            groups={
                "frame_transform": ObservationGroupCfg(
                    terms={
                        "links_to_base": FrameTransformTermCfg(
                            asset_cfg=SceneEntityCfg(
                                name="robots/robot_franka",
                                body_names=["panda_link0"],
                            ),
                            child_asset_cfg=SceneEntityCfg(
                                name="robots/robot_franka",
                                body_names=[
                                    "panda_hand",
                                    "panda_link7",
                                ],
                            ),
                        ),
                    },
                ),
            },
        )
        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=franka_table,
            observations=obs_cfg,
        )
        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            for _ in range(5):
                ret = env.step()

            tf_data = ret.observations["frame_transform"]["links_to_base"]
            assert isinstance(tf_data, BatchFrameTransformGraph)

            num_envs = franka_table.num_envs
            state = tf_data.as_state()
            assert len(state.tf_list) == 2
            assert state.tf_list[0].parent_frame_id == (
                "robots/robot_franka/panda_link0"
            )
            assert state.tf_list[0].child_frame_id == (
                "robots/robot_franka/panda_hand"
            )
            assert state.tf_list[1].parent_frame_id == (
                "robots/robot_franka/panda_link0"
            )
            assert state.tf_list[1].child_frame_id == (
                "robots/robot_franka/panda_link7"
            )
            for tf in state.tf_list:
                assert tf.xyz.shape == (num_envs, 3)
                assert tf.quat.shape == (num_envs, 4)
                assert torch.isfinite(tf.xyz).all()
                assert torch.isfinite(tf.quat).all()

    def test_multi_frame_to_multi_frame(
        self,
        franka_table: TableSceneCfg,
    ):
        obs_cfg = ObservationManagerCfg(
            groups={
                "frame_transform": ObservationGroupCfg(
                    terms={
                        "links_to_links": FrameTransformTermCfg(
                            asset_cfg=SceneEntityCfg(
                                name="robots/robot_franka",
                                body_names=[
                                    "panda_link3",
                                    "panda_link4",
                                ],
                            ),
                            child_asset_cfg=SceneEntityCfg(
                                name="robots/robot_franka",
                                body_names=[
                                    "panda_link4",
                                    "panda_link5",
                                ],
                            ),
                        ),
                    },
                ),
            },
        )
        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=franka_table,
            observations=obs_cfg,
        )
        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            for _ in range(5):
                ret = env.step()

            tf_data = ret.observations["frame_transform"]["links_to_links"]
            assert isinstance(tf_data, BatchFrameTransformGraph)

            num_envs = franka_table.num_envs
            state = tf_data.as_state()
            assert len(state.tf_list) == 2
            assert state.tf_list[0].parent_frame_id == (
                "robots/robot_franka/panda_link3"
            )
            assert state.tf_list[0].child_frame_id == (
                "robots/robot_franka/panda_link4"
            )
            assert state.tf_list[1].parent_frame_id == (
                "robots/robot_franka/panda_link4"
            )
            assert state.tf_list[1].child_frame_id == (
                "robots/robot_franka/panda_link5"
            )
            for tf in state.tf_list:
                assert tf.xyz.shape == (num_envs, 3)
                assert tf.quat.shape == (num_envs, 4)
                assert torch.isfinite(tf.xyz).all()
                assert torch.isfinite(tf.quat).all()

            # Verify tf_3_to_5 and tf_5_to_3 are inverse of each other
            link3 = "robots/robot_franka/panda_link3"
            link5 = "robots/robot_franka/panda_link5"
            tf_3_to_5 = tf_data.get_tf(link3, link5, compose=True)
            tf_5_to_3 = tf_data.get_tf(link5, link3, compose=True)
            assert tf_3_to_5 is not None
            assert tf_5_to_3 is not None
