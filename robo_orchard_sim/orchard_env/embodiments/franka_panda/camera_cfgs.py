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

"""Franka Panda camera configs shared by embodiment and example env cfgs."""

from __future__ import annotations

import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
import torch

from robo_orchard_sim.models.sensors.realsense import (
    D435I_CFG,
    CameraOffset,
)

__all__ = [
    "FRANKA_PANDA_HAND_CAMERA_CFG",
    "FRANKA_PANDA_STATIC_CAMERA_CFG",
    "FRANKA_PANDA_VIS_CAMERA_CFG",
]

FRANKA_PANDA_STATIC_CAMERA_CFG = D435I_CFG.copy()
FRANKA_PANDA_STATIC_CAMERA_CFG.prim_path = (
    "{ENV_REGEX_NS}/franka_panda/panda_link0/static_camera"
)
FRANKA_PANDA_STATIC_CAMERA_CFG.offset = CameraOffset(
    xyz=(1, 0, 0.8),
    quat=torch.asarray(
        rot_utils.euler_angles_to_quats(np.array([30, 180, 270]), degrees=True)
    ),
)

FRANKA_PANDA_VIS_CAMERA_CFG = D435I_CFG.copy()
FRANKA_PANDA_VIS_CAMERA_CFG.prim_path = "{ENV_REGEX_NS}/vis_camera"
FRANKA_PANDA_VIS_CAMERA_CFG.offset = CameraOffset(
    xyz=(0.9, 0.0, 1.2),
    quat=torch.asarray(
        rot_utils.euler_angles_to_quats(
            np.array([-150, 0, 90]),
            degrees=True,
        )
    ),
)

FRANKA_PANDA_HAND_CAMERA_CFG = D435I_CFG.copy()
FRANKA_PANDA_HAND_CAMERA_CFG.prim_path = (
    "{ENV_REGEX_NS}/franka_panda/panda_hand/hand_camera"
)
FRANKA_PANDA_HAND_CAMERA_CFG.offset = CameraOffset(
    xyz=(
        0.12399476432311997,
        -0.001844651806535513,
        -0.049206690374752254,
    ),
    quat=(0.6796847, -0.10977599, -0.15535776, 0.708408),
)
