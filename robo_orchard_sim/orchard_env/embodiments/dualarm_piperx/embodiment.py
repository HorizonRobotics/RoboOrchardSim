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

"""Minimal embodiment provider for the dual-arm PiperX robot."""

from __future__ import annotations

from robo_orchard_sim.contracts.policy_binding import PolicyBindingSchema
from robo_orchard_sim.ext.models.assets.asset_cfg import GroupAssetCfg
from robo_orchard_sim.orchard_env.assets import ArticulationSpec
from robo_orchard_sim.orchard_env.embodiments.dualarm_piper.embodiment import (
    DualArmPiperEmbodiment,
)
from robo_orchard_sim.orchard_env.embodiments.dualarm_piperx.cfg import (
    DUALARM_PIPERX_CFG,
)
from robo_orchard_sim.orchard_env.embodiments.dualarm_piperx.profile import (
    DUALARM_PIPERX_ROBOT_INFO_CFGS,
)
from robo_orchard_sim.orchard_env.embodiments.dualarm_piperx.schema import (
    build_dualarm_piperx_policy_binding_schema,
)
from robo_orchard_sim.orchard_env.embodiments.embodiment_base import (
    EmbodimentBase,
)
from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    RobotInfoCfg,
)


class DualArmPiperXEmbodiment(DualArmPiperEmbodiment):
    """Dual-arm PiperX embodiment using Piper-compatible manager configs."""

    @staticmethod
    def _get_camera_asset_map() -> GroupAssetCfg:
        """Build the camera asset mapping used across the embodiment."""
        from robo_orchard_sim.orchard_env.embodiments.dualarm_piperx.camera_cfgs import (  # noqa: E501
            DUALARM_PIPERX_LEFT_HAND_CAMERA_CFG,
            DUALARM_PIPERX_RIGHT_HAND_CAMERA_CFG,
            DUALARM_PIPERX_STATIC_CAMERA_CFG,
            DUALARM_PIPERX_VIS_CAMERA_CFG,
        )

        return {
            "static_camera": DUALARM_PIPERX_STATIC_CAMERA_CFG,
            "left_hand_camera": DUALARM_PIPERX_LEFT_HAND_CAMERA_CFG,
            "right_hand_camera": DUALARM_PIPERX_RIGHT_HAND_CAMERA_CFG,
            "vis_camera": DUALARM_PIPERX_VIS_CAMERA_CFG,
        }

    @staticmethod
    def _get_camera_assets() -> GroupAssetCfg:
        """Build fresh camera cfgs for each scene assembly."""
        return GroupAssetCfg(**DualArmPiperXEmbodiment._get_camera_asset_map())

    def __init__(
        self,
        namespace: str = "robots",
        name: str = "dualarm_piperx",
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
        EmbodimentBase.__init__(
            self,
            robot=ArticulationSpec(
                name=name,
                namespace=namespace,
                template_cfg=DUALARM_PIPERX_CFG,
                initial_pos=initial_pos,
                initial_rot=initial_rot,
            ),
        )
        self.enable_cameras = enable_cameras
        self.init_joint_noise_std = float(init_joint_noise_std)
        self.init_joint_pos: dict[str, float] | None
        if init_joint_pos is not None:
            self.init_joint_pos = dict(init_joint_pos)
        else:
            self.init_joint_pos = None

    def get_robot_info_cfgs(self) -> dict[str, RobotInfoCfg]:
        """Return robot profile metadata for traj planning."""
        return {
            name: robot_info.with_robot_name(self.scene_name)
            for name, robot_info in DUALARM_PIPERX_ROBOT_INFO_CFGS.items()
        }

    def get_policy_binding_schema(self) -> PolicyBindingSchema:
        """Return the canonical policy binding schema for this embodiment."""
        return build_dualarm_piperx_policy_binding_schema(self.name)
