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

from isaaclab.sim.schemas.schemas_cfg import (
    ArticulationRootPropertiesCfg as _ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg as _CollisionPropertiesCfg,
    DeformableBodyPropertiesCfg as _DeformableBodyPropertiesCfg,
    FixedTendonPropertiesCfg as _FixedTendonPropertiesCfg,
    JointDrivePropertiesCfg as _JointDrivePropertiesCfg,
    MassPropertiesCfg as _MassPropertiesCfg,
    RigidBodyPropertiesCfg as _RigidBodyPropertiesCfg,
)

from robo_orchard_sim.utils.config import isaac_configclass2pydantic

__all__ = [
    "ArticulationRootPropertiesCfg",
    "RigidBodyPropertiesCfg",
    "CollisionPropertiesCfg",
    "MassPropertiesCfg",
    "JointDrivePropertiesCfg",
    "FixedTendonPropertiesCfg",
    "DeformableBodyPropertiesCfg",
]


class ArticulationRootPropertiesCfg(
    isaac_configclass2pydantic(_ArticulationRootPropertiesCfg)
):
    __doc__ = _ArticulationRootPropertiesCfg.__doc__


class RigidBodyPropertiesCfg(
    isaac_configclass2pydantic(_RigidBodyPropertiesCfg)
):
    __doc__ = _RigidBodyPropertiesCfg.__doc__


class CollisionPropertiesCfg(
    isaac_configclass2pydantic(_CollisionPropertiesCfg)
):
    __doc__ = _CollisionPropertiesCfg.__doc__


class MassPropertiesCfg(isaac_configclass2pydantic(_MassPropertiesCfg)):
    __doc__ = _MassPropertiesCfg.__doc__


class JointDrivePropertiesCfg(
    isaac_configclass2pydantic(_JointDrivePropertiesCfg)
):
    __doc__ = _JointDrivePropertiesCfg.__doc__


class FixedTendonPropertiesCfg(
    isaac_configclass2pydantic(_FixedTendonPropertiesCfg)
):
    __doc__ = _FixedTendonPropertiesCfg.__doc__


class DeformableBodyPropertiesCfg(
    isaac_configclass2pydantic(_DeformableBodyPropertiesCfg)
):
    __doc__ = _DeformableBodyPropertiesCfg.__doc__
