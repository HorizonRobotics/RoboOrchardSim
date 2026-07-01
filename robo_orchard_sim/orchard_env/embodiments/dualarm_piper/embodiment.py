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
from robo_orchard_sim.orchard_env.embodiments.dualarm_piper.cfg import (
    DUALARM_PIPER_CFG,
)
from robo_orchard_sim.orchard_env.embodiments.dualarm_piper.profile import (
    DUALARM_PIPER_ROBOT_INFO_CFGS,
)
from robo_orchard_sim.orchard_env.embodiments.dualarm_piper.schema import (
    build_dualarm_piper_policy_binding_schema,
)
from robo_orchard_sim.orchard_env.embodiments.embodiment_base import (
    EmbodimentBase,
)
from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    RobotInfoCfg,
)

RENDER_FPS = 30
ACTION_FPS = 30


class DualArmPiperEmbodiment(EmbodimentBase):
    """Dual-arm Piper embodiment for phase0 minimal build path."""

    GRIPPER_JOINT_NAMES: tuple[str, ...] = (
        "left_joint7",
        "left_joint8",
        "right_joint7",
        "right_joint8",
    )
    """Joint names that are excluded from init-pose noise."""

    @staticmethod
    def _get_camera_asset_map() -> GroupAssetCfg:
        """Build the camera asset mapping used across the embodiment."""
        from robo_orchard_sim.orchard_env.embodiments.dualarm_piper.camera_cfgs import (  # noqa: E501
            DUALARM_PIPER_LEFT_HAND_CAMERA_CFG,
            DUALARM_PIPER_RIGHT_HAND_CAMERA_CFG,
            DUALARM_PIPER_STATIC_CAMERA_CFG,
            DUALARM_PIPER_VIS_CAMERA_CFG,
        )

        return {
            "static_camera": DUALARM_PIPER_STATIC_CAMERA_CFG,
            "left_hand_camera": DUALARM_PIPER_LEFT_HAND_CAMERA_CFG,
            "right_hand_camera": DUALARM_PIPER_RIGHT_HAND_CAMERA_CFG,
            "vis_camera": DUALARM_PIPER_VIS_CAMERA_CFG,
        }

    @staticmethod
    def _get_camera_assets() -> GroupAssetCfg:
        """Build fresh camera cfgs for each scene assembly."""
        return GroupAssetCfg(**DualArmPiperEmbodiment._get_camera_asset_map())

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

    @staticmethod
    def _generate_arm_tf_record_terms(
        arm_prefix: str,
    ) -> dict[str, McapTFTermCfg]:
        link_key = f"{arm_prefix}_robot_tf"
        base_link_key = f"{arm_prefix}_baselink_tf"
        return {
            f"{link_key}_term": McapTFTermCfg(
                topic=(
                    f"/observation/robot_state/link/{arm_prefix}_link{{id}}/tf"
                ),
                fps=ACTION_FPS,
                key=f"/tf/{link_key}",
            ),
            f"{base_link_key}_term": McapTFTermCfg(
                topic=(
                    f"/observation/robot_state/link/{arm_prefix}_base_link/tf"
                ),
                fps=ACTION_FPS,
                key=f"/tf/{base_link_key}",
            ),
        }

    @classmethod
    def _generate_camera_tf_terms(
        cls, robot_scene_name: str
    ) -> dict[str, FrameTransformTermCfg]:
        return {
            f"{camera_name}_tf": FrameTransformTermCfg(
                asset_cfg=SceneEntityCfg(
                    name=robot_scene_name,
                    body_names=["left_base_link"],
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
            rgb_term_name = f"{camera_name}_rgb"
            depth_term_name = f"{camera_name}_depth"
            color_calibration_term_name = f"{camera_name}_color_calib"
            depth_calibration_term_name = f"{camera_name}_depth_calib"
            key = f"/camera/{camera_name}_term"
            topic_base = f"/observation/cameras/{camera_name}"

            terms[rgb_term_name] = McapImageTermCfg(
                topic=f"{topic_base}/color_image/image_raw",
                fps=RENDER_FPS,
                key=key,
                frame_id=frame_id,
                mode="rgb",
            )
            terms[depth_term_name] = McapImageTermCfg(
                topic=f"{topic_base}/depth_image/image_raw",
                fps=RENDER_FPS,
                key=key,
                frame_id=frame_id,
                mode="depth",
            )
            terms[color_calibration_term_name] = McapImageTermCfg(
                topic=f"{topic_base}/color_image/camera_info",
                fps=RENDER_FPS,
                key=key,
                frame_id=frame_id,
                mode="calibration",
            )
            terms[depth_calibration_term_name] = McapImageTermCfg(
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
        name: str = "dualarm_piper",
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
                template_cfg=DUALARM_PIPER_CFG,
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
            for name, robot_info in DUALARM_PIPER_ROBOT_INFO_CFGS.items()
        }

    def get_policy_binding_schema(self) -> PolicyBindingSchema:
        """Return the canonical policy binding schema for this embodiment."""
        return build_dualarm_piper_policy_binding_schema(self.name)

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
            "/last_action": ObservationGroupCfg(
                terms={
                    "left_robot_joint_position": LastActionObservationTermCfg(
                        action_name="left_robot_joint_position",
                    ),
                    "left_robot_gripper_control": LastActionObservationTermCfg(
                        action_name="left_robot_gripper_control",
                    ),
                    "right_robot_joint_position": LastActionObservationTermCfg(
                        action_name="right_robot_joint_position",
                    ),
                    "right_robot_gripper_control": LastActionObservationTermCfg(  # noqa: E501
                        action_name="right_robot_gripper_control",
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
            groups["/tf"].terms.update(
                self._generate_camera_tf_terms(
                    robot_scene_name=robot_scene_name
                )
            )
            groups["/camera"] = ObservationGroupCfg(
                terms=dict(
                    static_camera_term=CameraObservationTermCfg(
                        asset_cfg=SceneEntityCfg(name="cameras/static_camera")
                    ),
                    left_hand_camera_term=CameraObservationTermCfg(
                        asset_cfg=SceneEntityCfg(
                            name="cameras/left_hand_camera"
                        )
                    ),
                    right_hand_camera_term=CameraObservationTermCfg(
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
        """Return robot-specific embodiment events.

        When ``init_joint_noise_std`` is positive or ``init_joint_pos`` is
        set, the default reset only restores joint positions/velocities
        and a follow-up ``JointStateResetTerm`` writes the (optionally
        Gaussian-perturbed) target joint positions to both the simulator
        state and the controller targets.
        """
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
            "left_joint_term": McapJointsTermCfg(
                topic="/robot/left/joint_states",
                fps=ACTION_FPS,
                position_key="/robot/left_joint_position",
                velocity_key="/robot/left_joint_velocity",
                effort_key="/robot/left_joint_effort",
                joint_name_prefix="left_joint",
            ),
            "right_joint_term": McapJointsTermCfg(
                topic="/robot/right/joint_states",
                fps=ACTION_FPS,
                position_key="/robot/right_joint_position",
                velocity_key="/robot/right_joint_velocity",
                effort_key="/robot/right_joint_effort",
                joint_name_prefix="right_joint",
            ),
            "left_arm_action_joint": McapJointsTermCfg(
                topic="/action/robot_state/left_joint/joint_states",
                fps=ACTION_FPS,
                position_key="/last_action/left_robot_joint_position",
            ),
            "left_arm_action_gripper_joint": McapJointsTermCfg(
                topic="/action/robot_state/left_gripper/joint_states",
                fps=ACTION_FPS,
                position_key="/last_action/left_robot_gripper_control",
            ),
            "right_arm_action_joint": McapJointsTermCfg(
                topic="/action/robot_state/right_joint/joint_states",
                fps=ACTION_FPS,
                position_key="/last_action/right_robot_joint_position",
            ),
            "right_arm_action_gripper_joint": McapJointsTermCfg(
                topic="/action/robot_state/right_gripper/joint_states",
                fps=ACTION_FPS,
                position_key="/last_action/right_robot_gripper_control",
            ),
            **self._generate_arm_tf_record_terms(arm_prefix="left"),
            **self._generate_arm_tf_record_terms(arm_prefix="right"),
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
