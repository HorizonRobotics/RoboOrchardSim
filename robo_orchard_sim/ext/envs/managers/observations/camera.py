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

"""Observation term for getting camera data from the environment."""

from __future__ import annotations

from robo_orchard_core.datatypes.camera_data import BatchCameraData
from typing_extensions import Generic

from robo_orchard_sim.ext.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.ext.envs.managers.observations.sensor import (
    ClassType_co,
    SensorObservationTerm,
    SensorObservationTermCfg,
)
from robo_orchard_sim.ext.models.sensors.camera import Camera

ReturnType = dict[str, BatchCameraData]


class CameraObservationTerm(
    SensorObservationTerm[
        IsaacEnvType_co, "CameraObservationTermCfg", ReturnType
    ],
    Generic[IsaacEnvType_co],
):
    """Observation term for observing cameras in the scene.

    This class is a specialization of `SensorObservationTerm` for cameras.
    The returned observations is in the form of `dict[str, BatchCameraData]`,
    where the key is the topic of the camera and the value is the camera data.

    Template Args:
        IsaacEnvType_co: The environment type.

    Args:
        cfg (SensorObservationTermCfg[CameraObservationTerm]): The cfg for
            the observation term.
        env (IsaacEnvType): The environment to observe.

    """

    def __init__(
        self,
        cfg: CameraObservationTermCfg,
        env: IsaacEnvType_co,
    ):
        super().__init__(cfg, env)

        if not isinstance(self._asset, Camera):
            raise ValueError(
                f"Asset {self.cfg.asset_cfg.name} is not a camera."
                f"Got {type(self._asset)} instead."
            )
        self._asset: Camera = self._asset

    def __call__(self) -> ReturnType:
        data = self._asset.get_camera_data()
        return data


class CameraObservationTermCfg(SensorObservationTermCfg):
    """Configuration class for the camera observation term.

    This class is a specialization of `SensorObservationTermCfg` for cameras.

    """

    class_type: ClassType_co[CameraObservationTerm] = CameraObservationTerm
    """The class type of the observation term.

    The class type should be subclass of `SensorObservationTerm`.
    """
