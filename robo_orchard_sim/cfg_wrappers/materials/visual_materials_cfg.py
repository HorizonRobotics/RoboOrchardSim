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


from isaaclab.sim.spawners.materials.visual_materials_cfg import (
    GlassMdlCfg as _GlassMdlCfg,
    MdlFileCfg as _MdlFileCfg,
    PreviewSurfaceCfg as _PreviewSurfaceCfg,
    VisualMaterialCfg as _VisualMaterialCfg,
    visual_materials,
)

from robo_orchard_sim.models.prim import PrimCreatorCfg, USDPrimCreatorType
from robo_orchard_sim.utils.config import isaac_configclass2pydantic

__all__ = [
    "VisualMaterialCfg",
    "PreviewSurfaceCfg",
    "MdlFileCfg",
    "GlassMdlCfg",
]


class VisualMaterialCfg(
    PrimCreatorCfg, isaac_configclass2pydantic(_VisualMaterialCfg)
):
    """The pydantic version of VisualMaterialCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.materials.VisualMaterialCfg`

    """

    __doc__ = _VisualMaterialCfg.__doc__


class PreviewSurfaceCfg(
    VisualMaterialCfg, isaac_configclass2pydantic(_PreviewSurfaceCfg)
):
    """The pydantic version of PreviewSurfaceCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.materials.PreviewSurfaceCfg`

    """

    __doc__ = _PreviewSurfaceCfg.__doc__

    func: USDPrimCreatorType = visual_materials.spawn_preview_surface


class MdlFileCfg(VisualMaterialCfg, isaac_configclass2pydantic(_MdlFileCfg)):
    """The pydantic version of MdlFileCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.materials.MdlFileCfg`

    """

    __doc__ = _MdlFileCfg.__doc__

    func: USDPrimCreatorType = visual_materials.spawn_from_mdl_file


class GlassMdlCfg(VisualMaterialCfg, isaac_configclass2pydantic(_GlassMdlCfg)):
    """The pydantic version of GlassMdlCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.materials.GlassMdlCfg`

    """

    __doc__ = _GlassMdlCfg.__doc__

    func: USDPrimCreatorType = visual_materials.spawn_from_mdl_file
