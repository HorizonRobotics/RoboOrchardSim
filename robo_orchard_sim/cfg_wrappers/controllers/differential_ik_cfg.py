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

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import (
    DifferentialIKControllerCfg as _DifferentialIKControllerCfg,
)

from robo_orchard_sim.utils.config import (
    ClassType_co,
    isaac_configclass2pydantic,
)


class DifferentialIKControllerCfg(
    isaac_configclass2pydantic(_DifferentialIKControllerCfg)
):
    """The pydantic version of DifferentialIKControllerCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.controllers.differential_ik_cfg.DifferentialIKControllerCfg`

    """

    __doc__ = _DifferentialIKControllerCfg.__doc__

    class_type: ClassType_co[DifferentialIKController] = (
        DifferentialIKController
    )
