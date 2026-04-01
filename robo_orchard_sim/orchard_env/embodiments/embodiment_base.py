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

"""Base class for robot embodiment providers."""

from __future__ import annotations

from robo_orchard_core.envs.managers.actions.action_manager import (
    ActionManagerCfg,
)
from robo_orchard_core.envs.managers.events import EventManagerCfg
from robo_orchard_core.envs.managers.observations.observation_manager import (
    ObservationManagerCfg,
)

from robo_orchard_sim.models.assets.asset_cfg import GroupAssetCfg
from robo_orchard_sim.orchard_env.assets import ArticulationSpec


class EmbodimentBase:
    """Abstract base for robot embodiment configuration providers."""

    def __init__(self, robot: ArticulationSpec):
        self.robot = robot.with_default_namespace("robots")

    @property
    def name(self) -> str:
        """Return the embodiment robot name."""
        return self.robot.name

    @property
    def namespace(self) -> str | None:
        """Return the embodiment robot namespace."""
        return self.robot.namespace

    @property
    def scene_name(self) -> str:
        """Return the scene-unique robot reference."""
        return self.robot.scene_name

    def get_assets_cfg(self) -> dict[str, GroupAssetCfg]:
        """Return embodiment-owned assets grouped by namespace."""
        return {
            self.robot.namespace: GroupAssetCfg(
                **{
                    self.robot.name: self.robot.to_isaac_cfg(),
                }
            )
        }

    def get_observation_cfg(self) -> ObservationManagerCfg:
        """Return embodiment observation cfg fragment."""
        return ObservationManagerCfg(groups={})

    def get_action_cfg(self) -> ActionManagerCfg:
        """Return embodiment action cfg fragment."""
        return ActionManagerCfg(terms={})

    def get_event_cfg(self) -> EventManagerCfg:
        """Return embodiment event cfg fragment."""
        return EventManagerCfg(terms={})
