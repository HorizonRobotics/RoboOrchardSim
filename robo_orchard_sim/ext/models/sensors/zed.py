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

from robo_orchard_sim.ext.models.sensors.camera import (
    CameraOffset,
    PinholeCameraCfg,
    SemanticCameraCfg,
)

__all__ = [
    "ZED_DROID_EXT1_CFG",
    "ZED_DROID_EXT2_CFG",
    "ZED_DROID_WRIST_CFG",
]

DROID_IMAGE_HEIGHT = 720
DROID_IMAGE_WIDTH = 1280

# DROID raw MP4s are 1280x720 for each mono stream. IsaacSim's camera wrapper
# does not support the DROID principal-point offsets because they produce
# aperture offsets above the local validation limit, so these configs keep the
# calibrated focal lengths and use the image center as the principal point.
ZED_DROID_WRIST_INTRINSIC_MATRIX = [
    666.67,
    0.0,
    640.0,
    0.0,
    666.67,
    360.0,
    0.0,
    0.0,
    1.0,
]

ZED_DROID_EXT1_INTRINSIC_MATRIX = [
    500.0,
    0.0,
    640.0,
    0.0,
    500.0,
    360.0,
    0.0,
    0.0,
    1.0,
]

ZED_DROID_EXT2_INTRINSIC_MATRIX = [
    531.8577880859375,
    0.0,
    640.0,
    0.0,
    531.8577880859375,
    360.0,
    0.0,
    0.0,
    1.0,
]


def _make_zed_droid_cfg(
    prim_path: str, intrinsic_matrix: list[float]
) -> SemanticCameraCfg:
    return SemanticCameraCfg(
        prim_path=prim_path,
        offset=CameraOffset(
            xyz=(0.5, 0.0, 1.0),
            quat=(
                4.329780281177467e-17,
                -0.7071067811865475,
                0.7071067811865476,
                4.329780281177466e-17,
            ),
        ),
        height=DROID_IMAGE_HEIGHT,
        width=DROID_IMAGE_WIDTH,
        data_types=[
            "rgb",
            "depth",
        ],
        colorize_semantic_segmentation=False,
        colorize_instance_id_segmentation=False,
        colorize_instance_segmentation=False,
        spawn=PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic_matrix,
            width=DROID_IMAGE_WIDTH,
            height=DROID_IMAGE_HEIGHT,
            clipping_range=(0.01, 1.0e5),
            focal_length=2.1,
            focus_distance=1.0,
        ),
    )


ZED_DROID_WRIST_CFG = _make_zed_droid_cfg(
    "{ENV_REGEX_NS}/zed_droid_wrist_camera",
    ZED_DROID_WRIST_INTRINSIC_MATRIX,
)

ZED_DROID_EXT1_CFG = _make_zed_droid_cfg(
    "{ENV_REGEX_NS}/zed_droid_ext1_camera",
    ZED_DROID_EXT1_INTRINSIC_MATRIX,
)

ZED_DROID_EXT2_CFG = _make_zed_droid_cfg(
    "{ENV_REGEX_NS}/zed_droid_ext2_camera",
    ZED_DROID_EXT2_INTRINSIC_MATRIX,
)
