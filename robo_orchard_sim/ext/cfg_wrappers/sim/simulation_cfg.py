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


from isaaclab.sim.simulation_cfg import (
    PhysxCfg as _PhysxCfg,
    RenderCfg as _RenderCfg,
    SimulationCfg as _SimulationCfg,
)

from robo_orchard_sim.ext.cfg_wrappers.materials.physics_materials_cfg import (
    RigidBodyMaterialCfg,
)
from robo_orchard_sim.utils.config import isaac_configclass2pydantic

__all__ = ["SimulationCfg", "PhysxCfg", "RenderCfg"]


class PhysxCfg(isaac_configclass2pydantic(_PhysxCfg)):
    __doc__ = _PhysxCfg.__doc__


class RenderCfg(isaac_configclass2pydantic(_RenderCfg)):
    __doc__ = _RenderCfg.__doc__


class SimulationCfg(isaac_configclass2pydantic(_SimulationCfg)):
    """The pydantic version of SimulationCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.simulation_cfg.SimulationCfg`

    """

    __doc__ = _SimulationCfg.__doc__

    physx: PhysxCfg = PhysxCfg()
    """PhysX solver settings. Default is PhysxCfg().

    Overrides type `PhysxCfg` from isaaclab.sim.simulation_cfg.
    """

    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg()
    """Overrides type `RigidBodyMaterialCfg` to pydantic version. """

    render: RenderCfg = RenderCfg()
    """Render settings. Default is RenderCfg()."""
