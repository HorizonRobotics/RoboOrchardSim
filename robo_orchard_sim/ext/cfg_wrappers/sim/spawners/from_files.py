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


from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.spawners.from_files.from_files_cfg import (
    FileCfg as _FileCfg,
    GroundPlaneCfg as _GroundPlaneCfg,
    UrdfFileCfg as _UrdfFileCfg,
    UsdFileCfg as _UsdFileCfg,
)

from robo_orchard_sim.ext.cfg_wrappers.materials.physics_materials_cfg import (
    RigidBodyMaterialCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.converters import UrdfConverterCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.spawner_cfg import (
    DeformableObjectSpawnerCfg,
    RigidObjectSpawnerCfg,
    SpawnerCfg,
)
from robo_orchard_sim.ext.models.prim import USDPrimCreatorType
from robo_orchard_sim.utils.config import isaac_configclass2pydantic

__all__ = [
    "FileCfg",
    "GroundPlaneCfg",
    "UrdfFileCfg",
    "UsdFileCfg",
]


class FileCfg(
    RigidObjectSpawnerCfg,
    DeformableObjectSpawnerCfg,
    isaac_configclass2pydantic(_FileCfg),
):
    """The pydantic version of isaaclab.sim.spawners.from_files.FileCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.from_files.FileCfg`

    """

    __doc__ = _FileCfg.__doc__


class UsdFileCfg(
    FileCfg,
    isaac_configclass2pydantic(_UsdFileCfg),
):
    """The pydantic version of UsdFileCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.from_files.UsdFileCfg`

    """

    # __doc__ = _UsdFileCfg.__doc__

    func: USDPrimCreatorType = from_files.spawn_from_usd
    # override the default value of func field to be from_files.spawn_from_usd
    # otherwise, the default value is Missing(defines in FileCfg)

    usd_path: str
    # override the default value of the usd_path field to be required


class UrdfFileCfg(
    FileCfg,
    UrdfConverterCfg,
    isaac_configclass2pydantic(_UrdfFileCfg),
):
    """The pydantic version of UrdfFileCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.from_files.UrdfFileCfg`

    """

    __doc__ = _UrdfFileCfg.__doc__

    func: USDPrimCreatorType = from_files.spawn_from_urdf
    # override the default value of func field to be from_files.spawn_from_usd
    # otherwise, the default value is Missing(defines in FileCfg)


class GroundPlaneCfg(
    SpawnerCfg,
    isaac_configclass2pydantic(_GroundPlaneCfg),
):
    """The pydantic version of GroundPlaneCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.from_files.GroundPlaneCfg`

    """

    __doc__ = _GroundPlaneCfg.__doc__

    func: USDPrimCreatorType = from_files.spawn_ground_plane
    # override the default value of func field to be from_files.spawn_from_usd
    # otherwise, the default value is Missing(defines in SpawnerCfg)

    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg()
    # override physics_material to pydantic version of RigidBodyMaterialCfg
