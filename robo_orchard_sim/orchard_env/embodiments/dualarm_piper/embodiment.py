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

"""Minimal embodiment provider for the dual-arm Piper robot."""

from __future__ import annotations

from robo_orchard_core.envs.managers.actions.action_manager import (
    ActionManagerCfg,
)
from robo_orchard_core.envs.managers.events import EventManagerCfg
from robo_orchard_core.envs.managers.observations.observation_manager import (
    ObservationGroupCfg,
    ObservationManagerCfg,
)

from robo_orchard_sim.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.envs.managers.actions.articulation.joint_position import (  # noqa: E501
    ArticulationJointPositionActionTermCfg,
)
from robo_orchard_sim.envs.managers.events.default_reset import (
    DefaultResetTermCfg,
)
from robo_orchard_sim.envs.managers.observations.asset_obs import (
    AssetObservationTermCfg,
)
from robo_orchard_sim.envs.managers.observations.camera import (
    CameraObservationTermCfg,
)
from robo_orchard_sim.envs.managers.observations.transform_frame import (
    FrameTransformTermCfg,
)
from robo_orchard_sim.models.assets.asset_cfg import GroupAssetCfg
from robo_orchard_sim.orchard_env.assets import ArticulationSpec
from robo_orchard_sim.orchard_env.embodiments.dualarm_piper.cfg import (
    DUALARM_PIPER_CFG,
)
from robo_orchard_sim.orchard_env.embodiments.embodiment_base import (
    EmbodimentBase,
)


class DualArmPiperEmbodiment(EmbodimentBase):
    """Dual-arm Piper embodiment for phase0 minimal build path."""

    @staticmethod
    def _get_camera_assets() -> GroupAssetCfg:
        """Build fresh camera cfgs for each scene assembly."""
        from robo_orchard_sim.orchard_env.embodiments.dualarm_piper.camera_cfgs import (  # noqa: E501
            DUALARM_PIPER_LEFT_HAND_CAMERA_CFG,
            DUALARM_PIPER_RIGHT_HAND_CAMERA_CFG,
            DUALARM_PIPER_STATIC_CAMERA_CFG,
            DUALARM_PIPER_VIS_CAMERA_CFG,
        )

        return GroupAssetCfg(
            static_camera=DUALARM_PIPER_STATIC_CAMERA_CFG,
            left_hand_camera=DUALARM_PIPER_LEFT_HAND_CAMERA_CFG,
            right_hand_camera=DUALARM_PIPER_RIGHT_HAND_CAMERA_CFG,
            vis_camera=DUALARM_PIPER_VIS_CAMERA_CFG,
        )

    @staticmethod
    def _generate_arm_tf_terms(
        arm_prefix: str,
        robot_scene_name: str,
    ) -> dict[str, FrameTransformTermCfg]:
        link_key = f"{arm_prefix}_robot_tf"
        base_link_key = f"{arm_prefix}_baselink_tf"
        return {
            link_key: FrameTransformTermCfg(
                asset_cfg=SceneEntityCfg(
                    name=robot_scene_name,
                    body_names=[
                        f"{arm_prefix}_base_link",
                        f"{arm_prefix}_link1",
                        f"{arm_prefix}_link2",
                        f"{arm_prefix}_link3",
                        f"{arm_prefix}_link4",
                        f"{arm_prefix}_link5",
                    ],
                ),
                child_asset_cfg=SceneEntityCfg(
                    name=robot_scene_name,
                    body_names=[
                        f"{arm_prefix}_link1",
                        f"{arm_prefix}_link2",
                        f"{arm_prefix}_link3",
                        f"{arm_prefix}_link4",
                        f"{arm_prefix}_link5",
                        f"{arm_prefix}_link6",
                    ],
                ),
            ),
            base_link_key: FrameTransformTermCfg(
                asset_cfg=SceneEntityCfg(
                    name=robot_scene_name,
                    body_names=["base_link"],
                ),
                child_asset_cfg=SceneEntityCfg(
                    name=robot_scene_name,
                    body_names=[f"{arm_prefix}_base_link"],
                ),
            ),
        }

    def __init__(
        self,
        namespace: str = "robots",
        name: str = "dualarm_piper",
        initial_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        initial_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        enable_cameras: bool = True,
    ):
        super().__init__(
            robot=ArticulationSpec(
                name=name,
                namespace=namespace,
                template_cfg=DUALARM_PIPER_CFG,
                initial_pos=initial_pos,
                initial_rot=initial_rot,
                joint_pos={".*": 0.0},
            )
        )
        self.enable_cameras = enable_cameras

    def get_assets_cfg(self) -> dict[str, GroupAssetCfg]:
        """Return robot and optionally camera assets."""
        assets = super().get_assets_cfg()
        if not self.enable_cameras:
            return assets
        assets["cameras"] = self._get_camera_assets()
        return assets

    def get_observation_cfg(self) -> ObservationManagerCfg:
        """Return robot state and frame-transform observation groups."""
        robot_scene_name = self.scene_name
        groups = {
            "/robot": ObservationGroupCfg(
                terms={
                    "base_link": AssetObservationTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name=robot_scene_name,
                            body_names=["base_link"],
                        ),
                        property_source="root",
                        property_name="pose",
                    ),
                    "left_joint_position": AssetObservationTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name=robot_scene_name,
                            joint_names=["left_joint[1-8]"],
                        ),
                        property_source="joint",
                        property_name="position",
                    ),
                    "left_joint_velocity": AssetObservationTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name=robot_scene_name,
                            joint_names=["left_joint[1-8]"],
                        ),
                        property_source="joint",
                        property_name="linear_velocity",
                    ),
                    "left_joint_effort": AssetObservationTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name=robot_scene_name,
                            joint_names=["left_joint[1-8]"],
                        ),
                        property_source="joint",
                        property_name="effort",
                    ),
                    "right_joint_position": AssetObservationTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name=robot_scene_name,
                            joint_names=["right_joint[1-8]"],
                        ),
                        property_source="joint",
                        property_name="position",
                    ),
                    "right_joint_velocity": AssetObservationTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name=robot_scene_name,
                            joint_names=["right_joint[1-8]"],
                        ),
                        property_source="joint",
                        property_name="linear_velocity",
                    ),
                    "right_joint_effort": AssetObservationTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name=robot_scene_name,
                            joint_names=["right_joint[1-8]"],
                        ),
                        property_source="joint",
                        property_name="effort",
                    ),
                }
            ),
            "/tf": ObservationGroupCfg(
                terms={
                    **self._generate_arm_tf_terms(
                        arm_prefix="left",
                        robot_scene_name=robot_scene_name,
                    ),
                    **self._generate_arm_tf_terms(
                        arm_prefix="right",
                        robot_scene_name=robot_scene_name,
                    ),
                }
            ),
        }
        if self.enable_cameras:
            groups["/camera"] = ObservationGroupCfg(
                terms=dict(
                    rgb_camera_term=CameraObservationTermCfg(
                        asset_cfg=SceneEntityCfg(name="cameras/static_camera")
                    ),
                    left_rgb_hand_camera_term=CameraObservationTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name="cameras/left_hand_camera"
                        )
                    ),
                    right_rgb_hand_camera_term=CameraObservationTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name="cameras/right_hand_camera"
                        )
                    ),
                    vis_camera_term=CameraObservationTermCfg(
                        asset_cfg=SceneEntityCfg(name="cameras/vis_camera")
                    ),
                )
            )
        return ObservationManagerCfg(groups=groups)

    def get_action_cfg(self) -> ActionManagerCfg:
        """Return robot joint-position actions for both arms and grippers."""
        robot_scene_name = self.scene_name
        return ActionManagerCfg(
            terms={
                "left_robot_joint_position": (
                    ArticulationJointPositionActionTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name=robot_scene_name,
                            joint_names=["left_joint[1-6]"],
                        ),
                        scale=1.0,
                        use_default_offset=False,
                    )
                ),
                "left_robot_gripper_control": (
                    ArticulationJointPositionActionTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name=robot_scene_name,
                            joint_names=["left_joint[7-8]"],
                        ),
                        scale=1.0,
                        use_default_offset=False,
                    )
                ),
                "right_robot_joint_position": (
                    ArticulationJointPositionActionTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name=robot_scene_name,
                            joint_names=["right_joint[1-6]"],
                        ),
                        scale=1.0,
                        use_default_offset=False,
                    )
                ),
                "right_robot_gripper_control": (
                    ArticulationJointPositionActionTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name=robot_scene_name,
                            joint_names=["right_joint[7-8]"],
                        ),
                        scale=1.0,
                        use_default_offset=False,
                    )
                ),
            }
        )

    def get_event_cfg(self) -> EventManagerCfg:
        """Return robot-specific embodiment events."""
        return EventManagerCfg(
            terms={
                "reset_robot_default": DefaultResetTermCfg(
                    asset_cfgs=[SceneEntityCfg(name=self.scene_name)],
                    trigger_topic="reset",
                    reset_joint_targets=True,
                ),
            }
        )
