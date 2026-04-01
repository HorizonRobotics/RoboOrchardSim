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

from typing import Literal

from isaaclab.sim.spawners.sensors import sensors
from isaaclab.sim.spawners.sensors.sensors_cfg import (
    FisheyeCameraCfg as _FisheyeCameraCfg,
    PinholeCameraCfg as _PinholeCameraCfg,
)
from pydantic import Field
from typing_extensions import Self

from robo_orchard_sim.cfg_wrappers.sim.spawners.spawner_cfg import (
    SpawnerCfg,
)
from robo_orchard_sim.models.prim import USDPrimCreatorType
from robo_orchard_sim.utils.config import isaac_configclass2pydantic

__all__ = [
    "PinholeCameraCfg",
    "FisheyeCameraCfg",
]


class PinholeCameraCfg(
    SpawnerCfg,
    isaac_configclass2pydantic(_PinholeCameraCfg),
):
    """The pydantic version of PinholeCameraCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.sensors.PinholeCameraCfg`

    """

    __doc__ = _PinholeCameraCfg.__doc__

    func: USDPrimCreatorType = sensors.spawn_camera
    """The function to spawn the object."""

    horizontal_aperture_offset: float = Field(
        default=0.0,
        le=1e-4,
        description="Offsets Resolution/Film gate horizontally. "
        "Note that due to the implementation limitation of isaacsim, "
        "this value cannot > 1e-4",
    )
    """Offsets Resolution/Film gate horizontally. Defaults to 0.0."""

    vertical_aperture_offset: float = Field(
        default=0.0,
        le=1e-4,
        description="Offsets Resolution/Film gate vertically. "
        "Note that due to the implementation limitation of isaacsim, "
        "this value cannot > 1e-4",
    )
    """Offsets Resolution/Film gate vertically. Defaults to 0.0."""

    @classmethod
    def from_intrinsic_matrix(cls, *args, **kwargs) -> Self:
        return cls(**super().from_intrinsic_matrix(*args, **kwargs).__dict__)


class FisheyeCameraCfg(
    PinholeCameraCfg,
    isaac_configclass2pydantic(_FisheyeCameraCfg),
):
    """The pydantic version of FishEyeCameraCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.sensors.FishEyeCameraCfg`

    """

    __doc__ = _FisheyeCameraCfg.__doc__

    projection_type: Literal[
        "fisheye_orthographic",
        "fisheye_equidistant",
        "fisheye_equisolid",
        "fisheye_polynomial",
        "fisheye_spherical",
    ] = "fisheye_polynomial"
    """Type of projection to use for the camera.

    Ovverides the default projection type defined in PinholeCameraCfg.
    Defaults to "fisheye_polynomial".
    """
