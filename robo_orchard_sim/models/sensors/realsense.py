## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
import torch

from robo_orchard_sim.models.sensors.camera import (
    CameraOffset,
    PinholeCameraCfg,
    SemanticCameraCfg,
)

__all__ = ["D435I_CFG", "L515_CFG"]

image_height = 720
image_width = 1280
fx, fy = 908.531982421875, 908.5385131835938
cx, cy = image_width / 2.0, image_height / 2.0

d435i_intrinsic_matrix = [fx, 0, cx, 0, fy, cy, 0, 0, 1]

# realsense D435I
D435I_CFG = SemanticCameraCfg(
    prim_path="{ENV_REGEX_NS}/realsense_d435i_camera",
    offset=CameraOffset(
        xyz=(0.5, 0, 1.0),
        quat=torch.asarray(
            rot_utils.euler_angles_to_quats(
                np.array([0, 180, 90]), degrees=True
            )
        ),
    ),
    height=image_height,
    width=image_width,
    data_types=[
        "rgb",
        "depth",
    ],
    colorize_semantic_segmentation=False,
    colorize_instance_id_segmentation=False,
    colorize_instance_segmentation=False,
    spawn=PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=d435i_intrinsic_matrix,
        width=image_width,
        height=image_height,
        clipping_range=(0.01, 1.0e5),
        # Although the documentation states that the unit is in
        # centimeters(cm), in practice, the unit used is based on
        # millimeters (mm). Refer to this link for more details:
        # https://forums.developer.nvidia.com/t/camera-parameter-unit-conversion-from-cm-to-mm/275291/2
        focal_length=1.88,
        focus_distance=1.0,
    ),
)

image_height = 720
image_width = 1280
fx, fy = 896.3739624023438, 896.664794921875
cx, cy = image_width / 2.0, image_height / 2.0

l515_intrinsic_matrix = [fx, 0, cx, 0, fy, cy, 0, 0, 1]

# realsense L515
L515_CFG = SemanticCameraCfg(
    prim_path="{ENV_REGEX_NS}/realsense_l515_camera",
    offset=CameraOffset(
        xyz=(0.5, 0, 2.0),
        quat=torch.asarray(
            rot_utils.euler_angles_to_quats(
                np.array([0, 180, 90]), degrees=True
            )
        ),
    ),
    height=image_height,
    width=image_width,
    data_types=[
        "rgb",
        "depth",
    ],
    colorize_semantic_segmentation=False,
    colorize_instance_id_segmentation=False,
    colorize_instance_segmentation=False,
    spawn=PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=l515_intrinsic_matrix,
        width=image_width,
        height=image_height,
        clipping_range=(0.01, 1.0e5),
        # Although the documentation states that the unit is in
        # centimeters(cm), in practice, the unit used is based on
        # millimeters (mm). Refer to this link for more details:
        # https://forums.developer.nvidia.com/t/camera-parameter-unit-conversion-from-cm-to-mm/275291/2
        focal_length=1.88,
        focus_distance=10.0,
    ),
)
