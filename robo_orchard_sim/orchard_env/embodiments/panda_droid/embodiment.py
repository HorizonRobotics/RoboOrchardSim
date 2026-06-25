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

"""Embodiment provider for the Panda Droid robot."""

from __future__ import annotations

from robo_orchard_core.envs.managers.actions.action_manager import (
    ActionManagerCfg,
)
from robo_orchard_core.envs.managers.events import EventManagerCfg
from robo_orchard_core.envs.managers.observations.observation_manager import (
    ObservationGroupCfg,
    ObservationManagerCfg,
)

from robo_orchard_sim.contracts.policy_binding import PolicyBindingSchema
from robo_orchard_sim.ext.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.ext.envs.managers.actions.articulation.joint_position import (  # noqa: E501
    ArticulationJointPositionActionTermCfg,
)
from robo_orchard_sim.ext.envs.managers.events.default_reset import (
    DefaultResetTermCfg,
)
from robo_orchard_sim.ext.envs.managers.events.joint_state_reset import (
    JointStateResetTermCfg,
)
from robo_orchard_sim.ext.envs.managers.observations.asset_obs import (
    AssetObservationTermCfg,
)
from robo_orchard_sim.ext.envs.managers.observations.camera import (
    CameraObservationTermCfg,
)
from robo_orchard_sim.ext.envs.managers.observations.last_action import (
    LastActionObservationTermCfg,
)
from robo_orchard_sim.ext.envs.managers.observations.transform_frame import (
    FrameTransformTermCfg,
)
from robo_orchard_sim.ext.envs.managers.record import RecordTermBaseCfg
from robo_orchard_sim.ext.envs.managers.record.mcap import (
    McapImageTermCfg,
    McapJointsTermCfg,
    McapTFTermCfg,
)
from robo_orchard_sim.ext.models.assets.asset_cfg import GroupAssetCfg
from robo_orchard_sim.orchard_env.assets import ArticulationSpec
from robo_orchard_sim.orchard_env.embodiments.embodiment_base import (
    EmbodimentBase,
)
from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    RobotInfoCfg,
)
from robo_orchard_sim.orchard_env.embodiments.panda_droid.cfg import (
    PANDA_DROID_HIGH_PD_CFG,
)
from robo_orchard_sim.orchard_env.embodiments.panda_droid.profile import (
    PANDA_DROID_ROBOT_INFO_CFGS,
)
from robo_orchard_sim.orchard_env.embodiments.panda_droid.schema import (
    build_panda_droid_policy_binding_schema,
)

RENDER_FPS = 30
ACTION_FPS = 30


class PandaDroidEmbodiment(EmbodimentBase):
    """Panda Droid embodiment for single-arm manipulation tasks."""

    GRIPPER_JOINT_NAMES: tuple[str, ...] = ("finger_joint",)
    """Joint names that are excluded from init-pose noise."""

    @staticmethod
    def _get_camera_asset_map() -> GroupAssetCfg:
        """Build the camera asset mapping used across the embodiment."""
        from robo_orchard_sim.orchard_env.embodiments.panda_droid.camera_cfgs import (  # noqa: E501
            PANDA_DROID_EXT1_CAMERA_CFG,
            PANDA_DROID_EXT2_CAMERA_CFG,
            PANDA_DROID_WRIST_CAMERA_CFG,
        )

        return {
            "ext1_camera": PANDA_DROID_EXT1_CAMERA_CFG,
            "ext2_camera": PANDA_DROID_EXT2_CAMERA_CFG,
            "wrist_camera": PANDA_DROID_WRIST_CAMERA_CFG,
        }

    @staticmethod
    def _get_camera_assets() -> GroupAssetCfg:
        """Build fresh camera cfgs for each scene assembly."""
        return GroupAssetCfg(**PandaDroidEmbodiment._get_camera_asset_map())

    @staticmethod
    def _generate_robot_tf_terms(
        robot_scene_name: str,
    ) -> dict[str, FrameTransformTermCfg]:
        return {
            "robot_tf": FrameTransformTermCfg(
                asset_cfg=SceneEntityCfg(
                    name=robot_scene_name,
                    body_names=[
                        "panda_link0",
                        "panda_link1",
                        "panda_link2",
                        "panda_link3",
                        "panda_link4",
                        "panda_link5",
                        "panda_link6",
                        "panda_link7",
                    ],
                ),
                child_asset_cfg=SceneEntityCfg(
                    name=robot_scene_name,
                    body_names=[
                        "panda_link1",
                        "panda_link2",
                        "panda_link3",
                        "panda_link4",
                        "panda_link5",
                        "panda_link6",
                        "panda_link7",
                        "base_link",
                    ],
                ),
            ),
        }

    @staticmethod
    def _generate_robot_tf_record_terms() -> dict[str, McapTFTermCfg]:
        return {
            "robot_tf_term": McapTFTermCfg(
                topic="/observation/robot_state/link/panda_link{id}/tf",
                fps=ACTION_FPS,
                key="/tf/robot_tf",
            ),
        }

    @classmethod
    def _generate_camera_tf_terms(
        cls, robot_scene_name: str
    ) -> dict[str, FrameTransformTermCfg]:
        camera_parent_body_names = {
            "ext1_camera": "panda_link0",
            "ext2_camera": "panda_link0",
            "wrist_camera": "panda_link0",
        }
        return {
            f"{camera_name}_tf": FrameTransformTermCfg(
                asset_cfg=SceneEntityCfg(
                    name=robot_scene_name,
                    body_names=[camera_parent_body_names[camera_name]],
                ),
                child_asset_cfg=SceneEntityCfg(name=f"cameras/{camera_name}"),
            )
            for camera_name in cls._get_camera_asset_map()
        }

    def _generate_camera_tf_record_terms(self) -> dict[str, McapTFTermCfg]:
        return {
            f"{camera_name}_tf": McapTFTermCfg(
                topic=f"/observation/cameras/{camera_name}/color_image/tf",
                fps=RENDER_FPS,
                key=f"/tf/{camera_name}_tf",
            )
            for camera_name in self._get_camera_asset_map()
        }

    def _generate_camera_image_record_terms(
        self,
    ) -> dict[str, McapImageTermCfg]:
        terms: dict[str, McapImageTermCfg] = {}
        for camera_name in self._get_camera_asset_map():
            frame_id = f"cameras/{camera_name}"
            key = f"/camera/{camera_name}_term"
            topic_base = f"/observation/cameras/{camera_name}"

            terms[f"{camera_name}_rgb"] = McapImageTermCfg(
                topic=f"{topic_base}/color_image/image_raw",
                fps=RENDER_FPS,
                key=key,
                frame_id=frame_id,
                mode="rgb",
            )
            terms[f"{camera_name}_depth"] = McapImageTermCfg(
                topic=f"{topic_base}/depth_image/image_raw",
                fps=RENDER_FPS,
                key=key,
                frame_id=frame_id,
                mode="depth",
            )
            terms[f"{camera_name}_color_calib"] = McapImageTermCfg(
                topic=f"{topic_base}/color_image/camera_info",
                fps=RENDER_FPS,
                key=key,
                frame_id=frame_id,
                mode="calibration",
            )
            terms[f"{camera_name}_depth_calib"] = McapImageTermCfg(
                topic=f"{topic_base}/depth_image/camera_info",
                fps=RENDER_FPS,
                key=key,
                frame_id=frame_id,
                mode="calibration",
            )

        return terms

    def __init__(
        self,
        namespace: str = "robots",
        name: str = "panda_droid",
        initial_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        initial_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        enable_cameras: bool = True,
        init_joint_noise_std: float = 0.0,
        init_joint_pos: dict[str, float] | None = None,
    ):
        if init_joint_noise_std < 0.0:
            raise ValueError(
                "init_joint_noise_std must be non-negative, "
                f"got {init_joint_noise_std}."
            )
        super().__init__(
            robot=ArticulationSpec(
                name=name,
                namespace=namespace,
                template_cfg=PANDA_DROID_HIGH_PD_CFG,
                initial_pos=initial_pos,
                initial_rot=initial_rot,
            )
        )
        self.enable_cameras = enable_cameras
        self.init_joint_noise_std = float(init_joint_noise_std)
        self.init_joint_pos: dict[str, float] | None
        if init_joint_pos is not None:
            self.init_joint_pos = dict(init_joint_pos)
        else:
            self.init_joint_pos = None

    def get_assets_cfg(self) -> dict[str, GroupAssetCfg]:
        """Return robot and optionally camera assets."""
        assets = super().get_assets_cfg()
        if not self.enable_cameras:
            return assets
        assets["cameras"] = self._get_camera_assets()
        return assets

    def get_robot_info_cfgs(self) -> dict[str, RobotInfoCfg]:
        """Return robot profile metadata for traj planning."""
        return {
            name: robot_info.with_robot_name(self.scene_name)
            for name, robot_info in PANDA_DROID_ROBOT_INFO_CFGS.items()
        }

    def get_policy_binding_schema(self) -> PolicyBindingSchema:
        """Return the canonical policy binding schema for this embodiment."""
        return build_panda_droid_policy_binding_schema(self.name)

    def get_observation_cfg(self) -> ObservationManagerCfg:
        """Return robot state and frame-transform observation groups."""
        robot_scene_name = self.scene_name
        robot_joint_names = ["panda_joint[1-7]", "finger_joint"]
        groups = {
            "/robot": ObservationGroupCfg(
                terms={
                    "base_link": AssetObservationTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name=robot_scene_name,
                            body_names=["panda_link0"],
                        ),
                        property_source="root",
                        property_name="pose",
                    ),
                    "joint_position": AssetObservationTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name=robot_scene_name,
                            joint_names=robot_joint_names,
                        ),
                        property_source="joint",
                        property_name="position",
                    ),
                    "joint_velocity": AssetObservationTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name=robot_scene_name,
                            joint_names=robot_joint_names,
                        ),
                        property_source="joint",
                        property_name="linear_velocity",
                    ),
                    "joint_effort": AssetObservationTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name=robot_scene_name,
                            joint_names=robot_joint_names,
                        ),
                        property_source="joint",
                        property_name="effort",
                    ),
                }
            ),
            "/last_action": ObservationGroupCfg(
                terms={
                    "robot_joint_position": LastActionObservationTermCfg(
                        action_name="robot_joint_position",
                    ),
                    "robot_gripper_control": LastActionObservationTermCfg(
                        action_name="robot_gripper_control",
                    ),
                }
            ),
            "/tf": ObservationGroupCfg(
                terms=self._generate_robot_tf_terms(
                    robot_scene_name=robot_scene_name,
                )
            ),
        }
        if self.enable_cameras:
            groups["/tf"].terms.update(
                self._generate_camera_tf_terms(
                    robot_scene_name=robot_scene_name
                )
            )
            groups["/camera"] = ObservationGroupCfg(
                terms={
                    f"{camera_name}_term": CameraObservationTermCfg(
                        asset_cfg=SceneEntityCfg(name=f"cameras/{camera_name}")
                    )
                    for camera_name in self._get_camera_asset_map()
                }
            )
        return ObservationManagerCfg(groups=groups)

    def get_action_cfg(self) -> ActionManagerCfg:
        """Return robot joint-position actions for the arm and gripper."""
        robot_scene_name = self.scene_name
        return ActionManagerCfg(
            terms={
                "robot_joint_position": ArticulationJointPositionActionTermCfg(
                    asset_cfg=SceneEntityCfg(
                        name=robot_scene_name,
                        joint_names=["panda_joint[1-7]"],
                    ),
                    scale=1.0,
                    use_default_offset=False,
                ),
                "robot_gripper_control": (
                    ArticulationJointPositionActionTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name=robot_scene_name,
                            joint_names=["finger_joint"],
                        ),
                        scale=1.0,
                        use_default_offset=False,
                    )
                ),
            }
        )

    def get_event_cfg(self) -> EventManagerCfg:
        """Return robot-specific embodiment events."""
        needs_init_joint_state_term = (
            self.init_joint_noise_std > 0.0 or self.init_joint_pos is not None
        )
        if not needs_init_joint_state_term:
            return EventManagerCfg(
                terms={
                    "reset_robot_default": DefaultResetTermCfg(
                        asset_cfgs=[SceneEntityCfg(name=self.scene_name)],
                        trigger_topic="reset",
                        reset_joint_targets=True,
                    ),
                }
            )

        return EventManagerCfg(
            terms={
                "reset_robot_default": DefaultResetTermCfg(
                    asset_cfgs=[SceneEntityCfg(name=self.scene_name)],
                    trigger_topic="reset",
                    reset_joint_targets=False,
                ),
                "reset_robot_init_joint_state": JointStateResetTermCfg(
                    asset_cfgs=[SceneEntityCfg(name=self.scene_name)],
                    trigger_topic="reset",
                    noise_std=self.init_joint_noise_std,
                    noise_excluded_joint_names=list(self.GRIPPER_JOINT_NAMES),
                    init_joint_pos=self.init_joint_pos,
                    clamp_to_joint_limits=True,
                    write_joint_state=True,
                    write_joint_position_target=True,
                ),
            }
        )

    def get_record_terms(self) -> dict[str, RecordTermBaseCfg]:
        terms: dict[str, RecordTermBaseCfg] = {
            "joint_term": McapJointsTermCfg(
                topic="/robot/joint_states",
                fps=ACTION_FPS,
                position_key="/robot/joint_position",
                velocity_key="/robot/joint_velocity",
                effort_key="/robot/joint_effort",
                joint_name_prefix="panda_joint",
            ),
            "arm_action_joint": McapJointsTermCfg(
                topic="/action/robot_state/joint/joint_states",
                fps=ACTION_FPS,
                position_key="/last_action/robot_joint_position",
            ),
            "gripper_action_joint": McapJointsTermCfg(
                topic="/action/robot_state/gripper/joint_states",
                fps=ACTION_FPS,
                position_key="/last_action/robot_gripper_control",
            ),
            **self._generate_robot_tf_record_terms(),
        }

        if not self.enable_cameras:
            return terms

        terms.update(
            {
                **self._generate_camera_tf_record_terms(),
                **self._generate_camera_image_record_terms(),
            }
        )
        return terms
