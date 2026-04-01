# # Project RoboOrchard
# #
# # Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
# #
# #
# #       http://www.apache.org/licenses/LICENSE-2.0
# # Licensed under the Apache License, Version 2.0 (the "License");
# # Unless required by applicable law or agreed to in writing, software
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# # You may obtain a copy of the License at
# # distributed under the License is distributed on an "AS IS" BASIS,
# # implied. See the License for the specific language governing
# # permissions and limitations under the License.
# # you may not use this file except in compliance with the License.

import pytest

from robo_orchard_sim_ut.utils.cfg_test import CfgTestBase


@pytest.fixture()
def simple_cfg(simple_isaac_wrapped_cfg):
    return simple_isaac_wrapped_cfg


class TestConfigWrapper(CfgTestBase):
    pass
