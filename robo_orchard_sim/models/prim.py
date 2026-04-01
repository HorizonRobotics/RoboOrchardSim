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


from typing import Generic, TypeVar

from isaaclab.sim.spawners import SpawnerCfg as _SpawnerCfg
from pxr import Usd

from robo_orchard_sim.utils.config import (
    CallableConfig,
    CallableType,
    ClassConfig,
    ClassType,
    isaac_configclass2pydantic,
)

PrimType = TypeVar("PrimType")
PrimType_co = TypeVar("PrimType_co", covariant=True)

USDPrimCreatorType = CallableType[..., Usd.Prim]


class PrimCreatorCfg(CallableConfig[Usd.Prim]):
    """The base configuration for all primitive creators."""

    func: USDPrimCreatorType
    """The function to create the primitive."""


class SpawnerCfg(PrimCreatorCfg, isaac_configclass2pydantic(_SpawnerCfg)):
    """The pydantic version of isaaclab.sim.spawners.SpawnerCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.SpawnerCfg`


    """

    __doc__ = _SpawnerCfg.__doc__


class PrimClassCfg(ClassConfig[PrimType_co], Generic[PrimType_co]):
    """The base configuration for all primitive classes."""

    class_type: ClassType[PrimType_co]
    """The class type of the primitive."""

    prim_path: str
    """The path of the primitive."""

    spawn: SpawnerCfg | None = None
    """The spawner configuration of the primitive.

    If not provided, the primitive will be assumed to be already
    present in the scene.
    """

    def prim_path_format(self, **kwargs):
        """Format the primitive path with the given keyword arguments.

        The prim_path will be formatted with the given keyword arguments.
        """
        self.prim_path = self.prim_path.format(**kwargs)
