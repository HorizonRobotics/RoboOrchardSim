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
from typing import TYPE_CHECKING, Generic, Mapping

from isaaclab.assets import (
    Articulation,
    ArticulationCfg as _ArticulationCfg,
    AssetBase,
    AssetBaseCfg as _AssetBaseCfg,
    RigidObject,
    RigidObjectCfg as _RigidObjectCfg,
)
from typing_extensions import TypeVar

from robo_orchard_sim.ext.cfg_wrappers.actuators_cfg import (
    ActuatorBaseCfgType_co,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.spawner_cfg import (
    SpawnerCfg,
)
from robo_orchard_sim.ext.models.prim import PrimClassCfg
from robo_orchard_sim.utils.config import (
    ClassType_co,
    isaac_configclass2pydantic,
)

if not TYPE_CHECKING:
    _AssetBaseCfg = isaac_configclass2pydantic(_AssetBaseCfg)
    _RigidObjectCfg = isaac_configclass2pydantic(_RigidObjectCfg)
    _ArticulationCfg = isaac_configclass2pydantic(_ArticulationCfg)


AssetType = TypeVar("AssetType", bound=AssetBase)
AssetType_co = TypeVar("AssetType_co", bound=AssetBase, covariant=True)
RigitObjectType = TypeVar(
    "RigitObjectType",
    bound=RigidObject,
    default=RigidObject,
)
RigidObjectType_co = TypeVar(
    "RigidObjectType_co",
    bound=RigidObject,
    covariant=True,
    default=RigidObject,
)
ArticulationType = TypeVar(
    "ArticulationType",
    bound=Articulation,
    default=Articulation,
)
ArticulationType_co = TypeVar(
    "ArticulationType_co",
    bound=Articulation,
    covariant=True,
    default=Articulation,
)
SpawnerCfgType = TypeVar("SpawnerCfgType", bound=SpawnerCfg)
SpawnerCfgType_co = TypeVar(
    "SpawnerCfgType_co", bound=SpawnerCfg, covariant=True
)

if not TYPE_CHECKING:
    _AssetBaseCfg = isaac_configclass2pydantic(_AssetBaseCfg)
    _RigidObjectCfg = isaac_configclass2pydantic(_RigidObjectCfg)
    _ArticulationCfg = isaac_configclass2pydantic(_ArticulationCfg)

ASSET_CFG = TypeVar("ASSET_CFG", bound="AssetBaseCfg")
ASSET_CFG_co = TypeVar("ASSET_CFG_co", bound="AssetBaseCfg", covariant=True)


class AssetBaseCfg(
    PrimClassCfg[AssetType_co],
    _AssetBaseCfg,
    Generic[SpawnerCfgType_co, AssetType_co],
):
    """The pydantic version of isaaclab.assets.AssetBaseCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.assets.AssetBaseCfg`

    """

    class InitialStateCfg(_AssetBaseCfg.InitialStateCfg):
        """Pydantic version of InitialStateCfg."""

        __doc__ = _AssetBaseCfg.InitialStateCfg.__doc__

    class_type: ClassType_co[AssetType_co]

    spawn: SpawnerCfgType_co | None = None

    init_state: AssetBaseCfg.InitialStateCfg = InitialStateCfg()


class RigidObjectCfg(
    AssetBaseCfg[SpawnerCfgType_co, RigidObjectType_co],
    _RigidObjectCfg,
):
    """The class template for all rigid object configurations."""

    class InitialStateCfg(_RigidObjectCfg.InitialStateCfg):
        """Pydantic version of InitialStateCfg."""

        __doc__ = _RigidObjectCfg.InitialStateCfg.__doc__

    class_type: ClassType_co[RigidObjectType_co] = (
        RigidObjectType_co.__default__
    )

    init_state: RigidObjectCfg.InitialStateCfg = InitialStateCfg()


class ArticulationCfg(
    AssetBaseCfg[SpawnerCfgType_co, ArticulationType_co],
    _ArticulationCfg,
):
    """The pydantic version of isaaclab.assets.ArticulationCfg."""

    class InitialStateCfg(_ArticulationCfg.InitialStateCfg):
        """Pydantic version of InitialStateCfg."""

        __doc__ = _ArticulationCfg.InitialStateCfg.__doc__

    actuators: Mapping[str, ActuatorBaseCfgType_co]

    class_type: ClassType_co[ArticulationType_co] = (
        ArticulationType_co.__default__
    )

    init_state: ArticulationCfg.InitialStateCfg = InitialStateCfg()
    """Initial state of the articulated object. Defaults to identity
    pose with zero velocity and zero joint state."""
