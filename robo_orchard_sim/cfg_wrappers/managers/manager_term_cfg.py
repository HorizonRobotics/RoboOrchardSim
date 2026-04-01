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
from typing import Any, Dict, Generic, List, TypeVar

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.manager_term_cfg import (
    EventTermCfg as _EventTermCfg,
    ManagerTermBaseCfg as _ManagerTermBaseCfg,
    ObservationGroupCfg as _ObservationGroupCfg,
    ObservationTermCfg as _ObservationTermCfg,
)
from pydantic import Field

from robo_orchard_sim.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.cfg_wrappers.utils.modifier_cfg import ModifierCfgType
from robo_orchard_sim.cfg_wrappers.utils.noise_cfg import NoiseCfgType
from robo_orchard_sim.utils.config import (
    CallableType,
    TorchTensor,
    isaac_configclass2pydantic,
)

ManagerBasedEnvType = TypeVar("ManagerBasedEnvType", bound=ManagerBasedEnv)


class ManagerTermBaseCfg(isaac_configclass2pydantic(_ManagerTermBaseCfg)):
    """The pydantic version of ManagerTermBaseCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.managers.manager_term_cfg.ManagerTermBaseCfg`

    """

    __doc__ = _ManagerTermBaseCfg.__doc__

    func: CallableType[..., Any]
    """The function or class to be called by the manager term.

    This is the overridden version of the original attribute for pydantic.
    """

    params: Dict[str, Any | SceneEntityCfg] = Field(default_factory=dict)
    """ The parameters to be passed to the function or class.

    Override the original attribute to fit the pydantic schema."""


class ObservationTermCfg(
    ManagerTermBaseCfg,
    isaac_configclass2pydantic(_ObservationTermCfg),
    Generic[NoiseCfgType, ModifierCfgType, ManagerBasedEnvType],
):
    """The pydantic version of ObservationTermCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.managers.manager_term_cfg.ObservationTermCfg`

    """

    __doc__ = _ObservationTermCfg.__doc__

    func: CallableType[[ManagerBasedEnvType, ...], TorchTensor]
    """The name of the function to be called to return the observation.

    Wrapper for the original attribute to fit the pydantic schema.
    """

    noise: NoiseCfgType | None = None
    """Wrapper for the original attribute to fit the pydantic schema."""

    modifiers: List[ModifierCfgType] | None = None


class ObservationGroupCfg(isaac_configclass2pydantic(_ObservationGroupCfg)):
    """The pydantic version of ObservationGroupCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.managers.manager_term_cfg.ObservationGroupCfg`

    """

    __doc__ = _ObservationGroupCfg.__doc__


class EventTermCfg(
    ManagerTermBaseCfg, isaac_configclass2pydantic(_EventTermCfg)
):
    """The pydantic version of EventTermCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.managers.manager_term_cfg.EventTermCfg`

    """

    __doc__ = _EventTermCfg.__doc__

    func: CallableType[..., None]
