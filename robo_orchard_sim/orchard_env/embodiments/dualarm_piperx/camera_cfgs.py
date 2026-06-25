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

"""Dual-arm PiperX camera configs shared by embodiment and example env cfgs."""

from __future__ import annotations

import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
import torch

from robo_orchard_sim.ext.models.sensors.realsense import (
    D405_CFG,
    D435I_CFG,
    D455_CFG,
    CameraOffset,
)

__all__ = [
    "DUALARM_PIPERX_LEFT_HAND_CAMERA_CFG",
    "DUALARM_PIPERX_RIGHT_HAND_CAMERA_CFG",
    "DUALARM_PIPERX_STATIC_CAMERA_CFG",
    "DUALARM_PIPERX_VIS_CAMERA_CFG",
]

DUALARM_PIPERX_STATIC_CAMERA_CFG = D455_CFG.copy()
DUALARM_PIPERX_STATIC_CAMERA_CFG.prim_path = (
    "{ENV_REGEX_NS}/dualarm_piperx/left_base_link/static_camera"
)
DUALARM_PIPERX_STATIC_CAMERA_CFG.offset = CameraOffset(
    xyz=(0.0, -0.3, 0.6036358425132415),
    quat=torch.asarray(
        rot_utils.euler_angles_to_quats(
            np.array([-145, 0, -90]),
            degrees=True,
        )
    ),
)

DUALARM_PIPERX_VIS_CAMERA_CFG = D435I_CFG.copy()
DUALARM_PIPERX_VIS_CAMERA_CFG.prim_path = "{ENV_REGEX_NS}/vis_camera"
DUALARM_PIPERX_VIS_CAMERA_CFG.offset = CameraOffset(
    xyz=(0.9, 0.0, 1.2),
    quat=torch.asarray(
        rot_utils.euler_angles_to_quats(
            np.array([-150, 0, 90]),
            degrees=True,
        )
    ),
)

DUALARM_PIPERX_LEFT_HAND_CAMERA_CFG = D405_CFG.copy()
DUALARM_PIPERX_LEFT_HAND_CAMERA_CFG.offset = CameraOffset(
    xyz=(
        -0.0096489170911759,
        -0.08009372951791657,
        0.04279548930003773,
    ),
    quat=(
        0.98480775,
        -0.17364818,
        0.0,
        0.0,
    ),
)
DUALARM_PIPERX_LEFT_HAND_CAMERA_CFG.prim_path = (
    "{ENV_REGEX_NS}/dualarm_piperx/left_link6/hand_camera"
)

DUALARM_PIPERX_RIGHT_HAND_CAMERA_CFG = (
    DUALARM_PIPERX_LEFT_HAND_CAMERA_CFG.copy()
)
DUALARM_PIPERX_RIGHT_HAND_CAMERA_CFG.prim_path = (
    "{ENV_REGEX_NS}/dualarm_piperx/right_link6/hand_camera"
)
