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

from typing import TypeVar

from isaaclab.sim.spawners.wrappers import (
    MultiAssetSpawnerCfg as _MultiAssetSpawnerCfg,
    wrappers as _wrappers,
)

from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.spawner_cfg import (
    DeformableObjectSpawnerCfg,
    RigidObjectSpawnerCfg,
)
from robo_orchard_sim.ext.models.prim import SpawnerCfg, USDPrimCreatorType
from robo_orchard_sim.utils.config import (
    isaac_configclass2pydantic,
)

__all__ = [
    "SpawnerCfg",
    "MultiAssetSpawnerCfg",
]

SpanwerCfgType = TypeVar("SpanwerCfgType")
SpanwerCfgType_co = TypeVar(
    "SpanwerCfgType_co", bound=SpawnerCfg, covariant=True
)


class MultiAssetSpawnerCfg(
    RigidObjectSpawnerCfg,
    DeformableObjectSpawnerCfg,
    isaac_configclass2pydantic(_MultiAssetSpawnerCfg),
):
    """The pydantic version of DeformableObjectSpawnerCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.DeformableObjectSpawnerCfg`

    """

    __doc__ = _MultiAssetSpawnerCfg.__doc__

    assets_cfg: list[SpanwerCfgType_co]

    func: USDPrimCreatorType = _wrappers.spawn_multi_asset
