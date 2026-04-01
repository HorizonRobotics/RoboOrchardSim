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

from isaaclab.sim.spawners.shapes import shapes as _shapes
from isaaclab.sim.spawners.shapes.shapes_cfg import (
    CapsuleCfg as _CapsuleCfg,
    ConeCfg as _ConeCfg,
    CuboidCfg as _CuboidCfg,
    CylinderCfg as _CylinderCfg,
    ShapeCfg as _ShapeCfg,
    SphereCfg as _SphereCfg,
)

from robo_orchard_sim.cfg_wrappers.materials import PhysicsMaterialCfg
from robo_orchard_sim.cfg_wrappers.sim.spawners.spawner_cfg import (
    RigidObjectSpawnerCfg,
)
from robo_orchard_sim.models.prim import USDPrimCreatorType
from robo_orchard_sim.utils.config import (
    isaac_configclass2pydantic,
)

__all__ = [
    "ShapeCfg",
    "SphereCfg",
    "CuboidCfg",
    "CylinderCfg",
    "CapsuleCfg",
    "ConeCfg",
]


class ShapeCfg(RigidObjectSpawnerCfg, isaac_configclass2pydantic(_ShapeCfg)):
    """The pydantic version of ShapeCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.shapes.ShapeCfg`
    """

    __doc__ = _ShapeCfg.__doc__

    physics_material: PhysicsMaterialCfg | None = None


class SphereCfg(ShapeCfg, isaac_configclass2pydantic(_SphereCfg)):
    """The pydantic version of SphereCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.shapes.SphereCfg`
    """

    __doc__ = _SphereCfg.__doc__

    radius: float  # type: ignore

    func: USDPrimCreatorType = _shapes.spawn_sphere


class CuboidCfg(ShapeCfg, isaac_configclass2pydantic(_CuboidCfg)):
    """The pydantic version of CuboidCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.shapes.CuboidCfg`
    """

    __doc__ = _CuboidCfg.__doc__

    size: tuple[float, float, float]  # type: ignore

    func: USDPrimCreatorType = _shapes.spawn_cuboid


class CylinderCfg(ShapeCfg, isaac_configclass2pydantic(_CylinderCfg)):
    """The pydantic version of CylinderCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.shapes.CylinderCfg`
    """

    __doc__ = _CylinderCfg.__doc__

    radius: float  # type: ignore
    """Radius of the cylinder (in m)."""

    height: float  # type: ignore
    """Height of the cylinder (in m)."""

    func: USDPrimCreatorType = _shapes.spawn_cylinder


class CapsuleCfg(ShapeCfg, isaac_configclass2pydantic(_CapsuleCfg)):
    """The pydantic version of CapsuleCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.shapes.CapsuleCfg`
    """

    __doc__ = _CapsuleCfg.__doc__

    radius: float
    """Radius of the capsule (in m)."""

    height: float
    """Height of the capsule (in m)."""

    func: USDPrimCreatorType = _shapes.spawn_capsule


class ConeCfg(ShapeCfg, isaac_configclass2pydantic(_ConeCfg)):
    """The pydantic version of ConeCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.shapes.ConeCfg`
    """

    __doc__ = _ConeCfg.__doc__

    radius: float
    """Radius of the cone (in m)."""

    height: float
    """Height of the v (in m)."""

    func: USDPrimCreatorType = _shapes.spawn_cone
