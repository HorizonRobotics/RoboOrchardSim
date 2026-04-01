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
#

from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    DeformableBodyMaterialCfg as _DeformableBodyMaterialCfg,
    PhysicsMaterialCfg as _PhysicsMaterialCfg,
    RigidBodyMaterialCfg as _RigidBodyMaterialCfg,
    physics_materials,
)

from robo_orchard_sim.models.prim import PrimCreatorCfg, USDPrimCreatorType
from robo_orchard_sim.utils.config import isaac_configclass2pydantic


class PhysicsMaterialCfg(
    PrimCreatorCfg, isaac_configclass2pydantic(_PhysicsMaterialCfg)
):
    """The pydantic version of PhysicsMaterialCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.materials.PhysicsMaterialCfg`

    """

    __doc__ = _PhysicsMaterialCfg.__doc__


class RigidBodyMaterialCfg(
    PhysicsMaterialCfg, isaac_configclass2pydantic(_RigidBodyMaterialCfg)
):
    """The pydantic version of RigidBodyMaterialCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.materials.RigidBodyMaterialCfg`

    """

    __doc__ = _RigidBodyMaterialCfg.__doc__

    func: USDPrimCreatorType = physics_materials.spawn_rigid_body_material


class DeformableBodyMaterialCfg(
    PhysicsMaterialCfg, isaac_configclass2pydantic(_DeformableBodyMaterialCfg)
):
    """The pydantic version of DeformableBodyMaterialCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.materials.DeformableBodyMaterialCfg`

    """

    __doc__ = _DeformableBodyMaterialCfg.__doc__

    func: USDPrimCreatorType = physics_materials.spawn_deformable_body_material
