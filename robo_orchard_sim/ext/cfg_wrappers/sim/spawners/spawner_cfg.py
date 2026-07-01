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


from isaaclab.sim.spawners import (
    DeformableObjectSpawnerCfg as _DeformableObjectSpawnerCfg,
    RigidObjectSpawnerCfg as _RigidObjectSpawnerCfg,
)

from robo_orchard_sim.ext.models.prim import SpawnerCfg
from robo_orchard_sim.utils.config import (
    isaac_configclass2pydantic,
)

__all__ = [
    "SpawnerCfg",
    "RigidObjectSpawnerCfg",
    "DeformableObjectSpawnerCfg",
]


class RigidObjectSpawnerCfg(
    SpawnerCfg, isaac_configclass2pydantic(_RigidObjectSpawnerCfg)
):
    """The pydantic version of RigidObjectSpawnerCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.RigidObjectSpawnerCfg`

    """

    __doc__ = _RigidObjectSpawnerCfg.__doc__


class DeformableObjectSpawnerCfg(
    SpawnerCfg, isaac_configclass2pydantic(_DeformableObjectSpawnerCfg)
):
    """The pydantic version of DeformableObjectSpawnerCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.DeformableObjectSpawnerCfg`

    """

    __doc__ = _DeformableObjectSpawnerCfg.__doc__
