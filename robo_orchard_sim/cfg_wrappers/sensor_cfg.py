# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
#
#
#       http://www.apache.org/licenses/LICENSE-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# You may obtain a copy of the License at
# distributed under the License is distributed on an "AS IS" BASIS,
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# you may not use this file except in compliance with the License.

from isaaclab.sensors import (
    SensorBase,
    SensorBaseCfg as _SensorBaseCfg,
)
from typing_extensions import TypeVar

from robo_orchard_sim.models.prim import PrimClassCfg
from robo_orchard_sim.utils.config import (
    isaac_configclass2pydantic,
)

SensorType = TypeVar("SensorType", bound=SensorBase)
SensorType_co = TypeVar("SensorType_co", bound=SensorBase, covariant=True)


SensorCfgType = TypeVar("SensorCfgType", bound="SensorBaseCfg")
SensorCfgType_co = TypeVar(
    "SensorCfgType_co", bound="SensorBaseCfg", covariant=True
)


class SensorBaseCfg(
    PrimClassCfg[SensorType_co], isaac_configclass2pydantic(_SensorBaseCfg)
):
    """The pydantic version of isaac lab SensorBaseCfg class.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sensors.SensorBaseCfg
    """

    __doc__ = _SensorBaseCfg.__doc__
