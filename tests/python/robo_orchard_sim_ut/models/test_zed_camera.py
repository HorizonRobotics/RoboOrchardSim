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

import pytest

from robo_orchard_sim.ext.models.sensors.zed import (
    ZED_DROID_EXT1_CFG,
    ZED_DROID_EXT2_CFG,
    ZED_DROID_WRIST_CFG,
)


@pytest.mark.parametrize(
    ("camera_cfg", "expected_intrinsic_matrix"),
    [
        (
            ZED_DROID_WRIST_CFG,
            [
                732.2020874023438,
                0.0,
                640.0,
                0.0,
                732.2020874023438,
                360.0,
                0.0,
                0.0,
                1.0,
            ],
        ),
        (
            ZED_DROID_EXT1_CFG,
            [
                524.419677734375,
                0.0,
                640.0,
                0.0,
                524.419677734375,
                360.0,
                0.0,
                0.0,
                1.0,
            ],
        ),
        (
            ZED_DROID_EXT2_CFG,
            [
                531.8577880859375,
                0.0,
                640.0,
                0.0,
                531.8577880859375,
                360.0,
                0.0,
                0.0,
                1.0,
            ],
        ),
    ],
)
def test_zed_droid_camera_cfg_original_resolution_uses_centered_intrinsics(
    camera_cfg, expected_intrinsic_matrix
):
    assert camera_cfg.width == 1280
    assert camera_cfg.height == 720
    assert camera_cfg.spawn is not None

    f_x = expected_intrinsic_matrix[0]
    f_y = expected_intrinsic_matrix[4]
    c_x = expected_intrinsic_matrix[2]
    c_y = expected_intrinsic_matrix[5]
    focal_length = camera_cfg.spawn.focal_length

    assert camera_cfg.spawn.horizontal_aperture == pytest.approx(
        camera_cfg.width * focal_length / f_x
    )
    assert camera_cfg.spawn.vertical_aperture == pytest.approx(
        camera_cfg.height * focal_length / f_y
    )
    assert camera_cfg.spawn.horizontal_aperture_offset == pytest.approx(
        (c_x - camera_cfg.width / 2) / f_x
    )
    assert camera_cfg.spawn.vertical_aperture_offset == pytest.approx(
        (c_y - camera_cfg.height / 2) / f_y
    )
