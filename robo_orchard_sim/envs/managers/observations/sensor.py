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

"""Observation term for observing properties of sensors in the scene."""

from __future__ import annotations
from typing import Any

from isaaclab.sensors import SensorBase
from robo_orchard_core.envs.managers.observations import (
    ObservationTermBase,
    ObservationTermCfg,
)
from typing_extensions import Generic, Sequence, TypeVar

from robo_orchard_sim.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg as LabSceneEntityCfg,
)
from robo_orchard_sim.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.utils.config import ClassType_co

ReturnType = TypeVar("ReturnType", default=Any)

SensorObservationTermCfgType_co = TypeVar(
    "SensorObservationTermCfgType_co",
    bound="SensorObservationTermCfg",
    covariant=True,
)


class SensorObservationTerm(
    ObservationTermBase[
        IsaacEnvType_co, SensorObservationTermCfgType_co, ReturnType
    ],
    Generic[IsaacEnvType_co, SensorObservationTermCfgType_co, ReturnType],
):
    """Observation term for observing properties of sensors in the scene.

    The `data` property of the sensor is returned as the observation. The data
    type is determined by the sensor type.

    Template Args:
        IsaacEnvType_co: The environment type.
        SensorObservationTermCfgType_co: The configuration type.
        ReturnType: The return type of the observation term.

    Args:
        cfg (SensorObservationTermCfg): The configuration for the observation
            term.
        env (IsaacEnvType): The environment to observe.
    """

    def __init__(
        self, cfg: SensorObservationTermCfgType_co, env: IsaacEnvType_co
    ):
        super().__init__(cfg, env)

        self.cfg.asset_cfg.resolve(env.scene)
        self._asset: SensorBase = env.scene[self.cfg.asset_cfg.name]
        if not isinstance(self._asset, SensorBase):
            raise ValueError(
                f"Asset {self.cfg.asset_cfg.name} is not a sensor."
            )

    def __call__(self) -> ReturnType:
        return self._asset.data

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the observation term."""
        # Do nothing for now
        pass


class SensorObservationTermCfg(
    ObservationTermCfg[
        SensorObservationTerm,
        LabSceneEntityCfg,
    ],
):
    """The configuration for the sensor observation term."""

    class_type: ClassType_co[SensorObservationTerm] = SensorObservationTerm
    """The class type of the observation term.

    The class type should be subclass of `SensorObservationTerm`.
    """
