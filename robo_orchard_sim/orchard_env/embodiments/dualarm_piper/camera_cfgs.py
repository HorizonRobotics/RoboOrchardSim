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

"""Dual-arm Piper camera configs shared by embodiment and example env cfgs."""

from __future__ import annotations

import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
import torch

from robo_orchard_sim.models.sensors.realsense import (
    D435I_CFG,
    CameraOffset,
)

__all__ = [
    "DUALARM_PIPER_LEFT_HAND_CAMERA_CFG",
    "DUALARM_PIPER_RIGHT_HAND_CAMERA_CFG",
    "DUALARM_PIPER_STATIC_CAMERA_CFG",
    "DUALARM_PIPER_VIS_CAMERA_CFG",
]

DUALARM_PIPER_STATIC_CAMERA_CFG = D435I_CFG.copy()
DUALARM_PIPER_STATIC_CAMERA_CFG.prim_path = (
    "{ENV_REGEX_NS}/dualarm_piper/left_base_link/static_camera"
)
DUALARM_PIPER_STATIC_CAMERA_CFG.offset = CameraOffset(
    xyz=(0.0, -0.3, 0.5123743521179518),
    quat=torch.asarray(
        rot_utils.euler_angles_to_quats(
            np.array([-135, 0, -90]),
            degrees=True,
        )
    ),
)

DUALARM_PIPER_VIS_CAMERA_CFG = D435I_CFG.copy()
DUALARM_PIPER_VIS_CAMERA_CFG.prim_path = "{ENV_REGEX_NS}/vis_camera"
DUALARM_PIPER_VIS_CAMERA_CFG.offset = CameraOffset(
    xyz=(0.9, 0.0, 1.2),
    quat=torch.asarray(
        rot_utils.euler_angles_to_quats(
            np.array([-150, 0, 90]),
            degrees=True,
        )
    ),
)

DUALARM_PIPER_LEFT_HAND_CAMERA_CFG = D435I_CFG.copy()
DUALARM_PIPER_LEFT_HAND_CAMERA_CFG.offset = CameraOffset(
    xyz=(
        -0.07354390146283836,
        0.007804615886680326,
        0.038433103882865485,
    ),
    quat=(
        -0.69636424,
        0.1227878,
        -0.1227878,
        0.69636424,
    ),
)
DUALARM_PIPER_LEFT_HAND_CAMERA_CFG.prim_path = (
    "{ENV_REGEX_NS}/dualarm_piper/left_link6/hand_camera"
)

DUALARM_PIPER_RIGHT_HAND_CAMERA_CFG = DUALARM_PIPER_LEFT_HAND_CAMERA_CFG.copy()
DUALARM_PIPER_RIGHT_HAND_CAMERA_CFG.prim_path = (
    "{ENV_REGEX_NS}/dualarm_piper/right_link6/hand_camera"
)
