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


from isaaclab.sim.spawners.lights import lights
from isaaclab.sim.spawners.lights.lights_cfg import (
    CylinderLightCfg as _CylinderLightCfg,
    DiskLightCfg as _DiskLightCfg,
    DistantLightCfg as _DistantLightCfg,
    DomeLightCfg as _DomeLightCfg,
    LightCfg as _LightCfg,
    SphereLightCfg as _SphereLightCfg,
)

from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.spawner_cfg import (
    SpawnerCfg,
)
from robo_orchard_sim.ext.models.prim import USDPrimCreatorType
from robo_orchard_sim.utils.config import isaac_configclass2pydantic

__all__ = [
    "LightCfg",
    "DiskLightCfg",
    "DistantLightCfg",
    "DomeLightCfg",
    "CylinderLightCfg",
    "SphereLightCfg",
]


class LightCfg(
    SpawnerCfg,
    isaac_configclass2pydantic(_LightCfg),
):
    """The pydantic version of LightCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.lights.LightCfg`

    """

    __doc__ = _LightCfg.__doc__

    prim_type: str  # type: ignore

    func: USDPrimCreatorType = lights.spawn_light


class DiskLightCfg(
    LightCfg,
    isaac_configclass2pydantic(_DiskLightCfg),
):
    """The pydantic version of  DiskLightCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.lights.DiskLightCfg`

    """

    __doc__ = _DiskLightCfg.__doc__

    prim_type: str = "DiskLight"


class DistantLightCfg(
    LightCfg,
    isaac_configclass2pydantic(_DistantLightCfg),
):
    """The pydantic version of DistantLightCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.lights.DistantLightCfg`

    """

    __doc__ = _DistantLightCfg.__doc__

    prim_type: str = "DistantLight"


class DomeLightCfg(
    LightCfg,
    isaac_configclass2pydantic(_DomeLightCfg),
):
    """The pydantic version of DomeLightCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.lights.DomeLightCfg`

    """

    __doc__ = _DomeLightCfg.__doc__

    prim_type: str = "DomeLight"


class CylinderLightCfg(
    LightCfg,
    isaac_configclass2pydantic(_CylinderLightCfg),
):
    """The pydantic version of CylinderLightCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.lights.CylinderLightCfg`

    """

    __doc__ = _CylinderLightCfg.__doc__

    prim_type: str = "CylinderLight"


class SphereLightCfg(
    LightCfg,
    isaac_configclass2pydantic(_SphereLightCfg),
):
    """The pydantic version of SphereLightCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.lights.SphereLightCfg`

    """

    __doc__ = _SphereLightCfg.__doc__

    prim_type: str = "SphereLight"
