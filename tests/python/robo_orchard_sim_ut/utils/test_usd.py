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

import os
import shutil
import tempfile

import pytest

from robo_orchard_sim.models.assets.asset_cfg import NV_ISAACLAB_DIR
from robo_orchard_sim.utils.usd import usd_to_urdf


@pytest.fixture
def franka_usd_path():
    path = f"{NV_ISAACLAB_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    assert os.path.exists(path)
    return path


class TestUSD2URDF:
    def test_convert(self, franka_usd_path: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_output_path = f"{tmpdir}/franka_from_usd_collision"
            print("urdf_output_path: ", urdf_output_path)
            urdf_path = usd_to_urdf(
                usd_path=franka_usd_path,
                urdf_output_path=urdf_output_path,
            )
            assert os.path.exists(urdf_path)
            # check file extension
            assert urdf_path.endswith(".urdf")
            if os.path.exists(urdf_output_path):
                shutil.rmtree(urdf_output_path)
