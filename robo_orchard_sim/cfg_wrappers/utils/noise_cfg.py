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

from __future__ import annotations
from typing import Generic, TypeVar

import torch
from isaaclab.utils.noise import noise_model
from isaaclab.utils.noise.noise_cfg import (
    ConstantNoiseCfg as _ConstantNoiseCfg,
    NoiseCfg as _NoiseCfg,
)

from robo_orchard_sim.utils.config import (
    CallableType,
    TorchTensor,
    isaac_configclass2pydantic,
)

NoiseCfgType = TypeVar("NoiseCfgType", bound="NoiseCfg")


class NoiseCfg(isaac_configclass2pydantic(_NoiseCfg), Generic[NoiseCfgType]):
    """The pydantic version of NoiseCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.utils.noise.NoiseCfg`

    """

    __doc__ = _NoiseCfg.__doc__

    func: CallableType[[torch.Tensor, NoiseCfgType], torch.Tensor]


# NoiseCfgType = TypeVar("NoiseCfgType", bound=NoiseCfg)


class ConstantNoiseCfg(
    NoiseCfg["ConstantNoiseCfg"], isaac_configclass2pydantic(_ConstantNoiseCfg)
):
    """The pydantic version of _ConstantNoiseCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.utils.noise.ConstantBiasNoiseCfg`

    """

    __doc__ = _ConstantNoiseCfg.__doc__

    func: CallableType[[torch.Tensor, ConstantNoiseCfg], torch.Tensor] = (
        noise_model.constant_noise
    )

    bias: TorchTensor | float = 0.0


ConstantBiasNoiseCfg = ConstantNoiseCfg
