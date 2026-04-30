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

"""Base class for scene providers used by ``EnvBuilder``."""

from abc import ABC, abstractmethod
from collections.abc import Mapping

from robo_orchard_core.envs.managers.actions.action_manager import (
    ActionManagerCfg,
)
from robo_orchard_core.envs.managers.events import EventManagerCfg
from robo_orchard_core.envs.managers.observations.observation_manager import (
    ObservationManagerCfg,
)

from robo_orchard_sim.cfg_wrappers.envs.env_cfg import ViewerCfg
from robo_orchard_sim.cfg_wrappers.sim.simulation_cfg import (
    PhysxCfg,
    SimulationCfg,
)
from robo_orchard_sim.envs.managers.record import RecordTermBaseCfg
from robo_orchard_sim.models.assets.asset_cfg import GroupAssetCfg


class SceneBase(ABC):
    """Base interface for composable scene providers."""

    def __init__(
        self,
        num_envs: int = 1,
        env_spacing: float = 2.5,
        physics_fps: int = 600,
        render_fps: int = 30,
        step_fps: int = 30,
    ):
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self.physics_fps = physics_fps
        self.render_fps = render_fps
        self.step_fps = step_fps

    @abstractmethod
    def get_assets_cfg(self) -> dict[str, GroupAssetCfg]:
        """Return scene-owned assets grouped by namespace."""
        raise NotImplementedError

    def get_sim_cfg(self) -> SimulationCfg:
        """Return scene-level simulation cfg."""
        return SimulationCfg(
            render_interval=self.get_render_interval(),
            dt=1.0 / self.physics_fps,
            physx=PhysxCfg(enable_ccd=True),
        )

    def get_viewer_cfg(self) -> ViewerCfg:
        """Return scene-level viewer cfg."""
        return ViewerCfg(
            eye=(1.5, 1.5, 3.0),
            lookat=(0.0, 0.0, 0.0),
        )

    def get_render_interval(self) -> int:
        """Return render interval from fps settings."""
        return int(self.physics_fps / self.render_fps)

    def get_decimation(self) -> int:
        """Return decimation from fps settings."""
        return int(self.physics_fps / self.step_fps)

    def get_num_envs(self) -> int:
        """Return number of environments."""
        return self.num_envs

    def get_env_spacing(self) -> float:
        """Return spacing between environments."""
        return self.env_spacing

    def get_observation_cfg(self) -> ObservationManagerCfg:
        """Return scene-level observation cfg fragment."""
        return ObservationManagerCfg(groups={})

    def get_action_cfg(self) -> ActionManagerCfg:
        """Return scene-level action cfg fragment (usually empty)."""
        return ActionManagerCfg(terms={})

    def get_event_cfg(self) -> EventManagerCfg:
        """Return scene-level event cfg fragment."""
        return EventManagerCfg(terms={})

    def get_record_terms(self) -> Mapping[str, RecordTermBaseCfg]:
        """Return scene-level record term fragments."""
        return {}
