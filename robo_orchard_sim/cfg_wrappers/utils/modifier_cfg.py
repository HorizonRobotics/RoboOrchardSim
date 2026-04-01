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

from __future__ import annotations
from typing import TypeVar

import torch
from isaaclab.utils.modifiers import modifier
from isaaclab.utils.modifiers.modifier_cfg import (
    DigitalFilterCfg as _DigitalFilterCfg,
    IntegratorCfg as _IntegratorCfg,
    ModifierCfg as _ModifierCfg,
)

from robo_orchard_sim.utils.config import (
    CallableType,
    ClassType,
    isaac_configclass2pydantic,
)


class ModifierCfg(isaac_configclass2pydantic(_ModifierCfg)):
    """The pydantic version of ModifierCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.utils.modifiers.ModifierCfg`

    """

    __doc__ = _ModifierCfg.__doc__

    func: CallableType[[..., ModifierCfg], torch.Tensor]


ModifierCfgType = TypeVar("ModifierCfgType", bound=ModifierCfg)


class DigitalFilterCfg(
    ModifierCfg, isaac_configclass2pydantic(_DigitalFilterCfg)
):
    """The pydantic version of DigitalFilterCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.utils.modifiers.DigitalFilterCfg`

    """

    __doc__ = _DigitalFilterCfg.__doc__

    func: ClassType[modifier.DigitalFilter] = modifier.DigitalFilter


class IntegratorCfg(ModifierCfg, isaac_configclass2pydantic(_IntegratorCfg)):
    """The pydantic version of IntegratorCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.utils.modifiers.IntegratorCfg`

    """

    __doc__ = _IntegratorCfg.__doc__

    func: ClassType[modifier.Integrator] = modifier.Integrator
