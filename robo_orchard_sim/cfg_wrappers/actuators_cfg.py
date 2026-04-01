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

from typing import List

from isaaclab.actuators import (
    ActuatorBase,
    ActuatorBaseCfg as _ActuatorBaseCfg,
    ImplicitActuatorCfg as _ImplicitActuatorCfg,
)
from isaaclab.actuators.actuator_pd import ImplicitActuator
from typing_extensions import TypeVar

from robo_orchard_sim.utils.config import (
    ClassConfig,
    ClassType_co,
    isaac_configclass2pydantic,
)

__all__ = [
    "ActuatorBaseCfg",
    "ImplicitActuatorCfg",
]


ActuatorType_co = TypeVar(
    "ActuatorType_co", bound=ActuatorBase, covariant=True
)


class ActuatorBaseCfg(
    ClassConfig[ActuatorType_co],
    isaac_configclass2pydantic(_ActuatorBaseCfg),
):
    """The pydantic version of isaaclab.actuator.ActuatorBaseCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.actuator.ActuatorBaseCfg`
    """

    class_type: ClassType_co[ActuatorType_co]

    joint_names_expr: List[str]
    """Articulation's joint names that are part of the group.

    Note:
        This can be a list of joint names or a list of
        regex expressions (e.g. ".*").
    """

    stiffness: dict[str, float] | float | None
    """Force/Torque limit of the joints in the group. Defaults to None.

    If None, the limit is set to the value specified in the USD joint prim.
    """

    damping: dict[str, float] | float | None
    """Stiffness gains (also known as p-gain) of the joints in the group.

    If None, the stiffness is set to the value from the USD joint prim.
    """


ActuatorBaseCfgType_co = TypeVar(
    "ActuatorBaseCfgType_co", bound=ActuatorBaseCfg, covariant=True
)


class ImplicitActuatorCfg(
    ActuatorBaseCfg[ImplicitActuator],
    isaac_configclass2pydantic(_ImplicitActuatorCfg),
):
    """The pydantic version of isaaclab.actuator.ImplicitActuatorCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.actuator.ImplicitActuatorCfg`
    """

    class_type: ClassType_co[ImplicitActuator] = ImplicitActuator
