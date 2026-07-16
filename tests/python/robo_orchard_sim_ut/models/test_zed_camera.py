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
    "camera_cfg",
    [
        ZED_DROID_WRIST_CFG,
        ZED_DROID_EXT1_CFG,
        ZED_DROID_EXT2_CFG,
    ],
)
def test_zed_droid_camera_cfg_original_resolution_returns_expected_dimensions(
    camera_cfg,
):
    assert (camera_cfg.width, camera_cfg.height) == (1280, 720)


@pytest.mark.parametrize(
    "camera_cfg",
    [
        ZED_DROID_WRIST_CFG,
        ZED_DROID_EXT1_CFG,
        ZED_DROID_EXT2_CFG,
    ],
)
def test_zed_droid_camera_cfg_centered_principal_point_returns_zero_offsets(
    camera_cfg,
):
    assert camera_cfg.spawn is not None
    assert (
        camera_cfg.spawn.horizontal_aperture_offset,
        camera_cfg.spawn.vertical_aperture_offset,
    ) == pytest.approx((0.0, 0.0))
