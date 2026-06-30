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

"""Panda Droid camera configs shared by embodiment and env cfgs."""

from __future__ import annotations

from robo_orchard_sim.ext.models.sensors.realsense import CameraOffset
from robo_orchard_sim.ext.models.sensors.zed import (
    ZED_DROID_EXT1_CFG,
    ZED_DROID_EXT2_CFG,
    ZED_DROID_WRIST_CFG,
)

__all__ = [
    "PANDA_DROID_EXT1_CAMERA_CFG",
    "PANDA_DROID_EXT2_CAMERA_CFG",
    "PANDA_DROID_WRIST_CAMERA_CFG",
]

PANDA_DROID_EXT1_CAMERA_CFG = ZED_DROID_EXT1_CFG.copy()
PANDA_DROID_EXT1_CAMERA_CFG.prim_path = (
    "{ENV_REGEX_NS}/panda_droid/panda_link0/droid_ext1_camera"
)
PANDA_DROID_EXT1_CAMERA_CFG.offset = CameraOffset(
    xyz=(
        0.05,
        0.57,
        0.66,
    ),
    quat=(
        0.195029,
        -0.393059,
        0.805121,
        -0.399060,
    ),
)

PANDA_DROID_EXT2_CAMERA_CFG = ZED_DROID_EXT2_CFG.copy()
PANDA_DROID_EXT2_CAMERA_CFG.prim_path = (
    "{ENV_REGEX_NS}/panda_droid/panda_link0/droid_ext2_camera"
)
PANDA_DROID_EXT2_CAMERA_CFG.offset = CameraOffset(
    xyz=(
        0.2596757315060087,
        -0.36626259649963777,
        0.24849304837972613,
    ),
    quat=(
        0.6031385253978158,
        -0.7167807299976015,
        0.2673321033144963,
        -0.2257938237031771,
    ),
)

PANDA_DROID_WRIST_CAMERA_CFG = ZED_DROID_WRIST_CFG.copy()
PANDA_DROID_WRIST_CAMERA_CFG.prim_path = (
    "{ENV_REGEX_NS}/panda_droid/base_link/droid_wrist_camera"
)
PANDA_DROID_WRIST_CAMERA_CFG.offset = CameraOffset(
    xyz=(-0.073999, 0.030860, 0.009252),
    quat=(
        0.114662,
        0.704006,
        -0.692276,
        -0.109465,
    ),
    convention="opengl",
)
