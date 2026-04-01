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

"""Base abstractions for composable task configuration."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from robo_orchard_core.envs.managers.actions.action_manager import (
    ActionManagerCfg,
)
from robo_orchard_core.envs.managers.events import EventManagerCfg
from robo_orchard_core.envs.managers.observations.observation_manager import (
    ObservationManagerCfg,
)

from robo_orchard_sim.models.assets.asset_cfg import GroupAssetCfg
from robo_orchard_sim.orchard_env.assets import AssetSpec

if TYPE_CHECKING:
    from robo_orchard_sim.tasks.validators.base import Validator


class TaskBase(ABC):
    """Abstract base for composable task configuration.

    A task knows which objects it needs, how to observe them, and
    how to reset them.  Concrete subclasses return ready-to-merge
    cfg fragments via the ``get_*`` methods.
    """

    def __init__(self, assets: dict[str, AssetSpec]):
        self._assets = {
            role: spec.with_default_namespace("objects")
            for role, spec in assets.items()
        }

    # ---------------------------------------------------------
    # Scene assets
    # ---------------------------------------------------------
    def get_assets_cfg(self) -> dict[str, GroupAssetCfg]:
        """Return task-owned assets grouped by namespace."""
        grouped: dict[str, dict[str, object]] = {}
        for spec in self._assets.values():
            grouped.setdefault(spec.namespace, {})
            if spec.name in grouped[spec.namespace]:
                raise ValueError(
                    "Duplicate task asset "
                    f"'{spec.scene_name}' in PlaceA2BTask."
                )
            grouped[spec.namespace][spec.name] = spec.to_isaac_cfg()
        return {
            namespace: GroupAssetCfg(**group_assets)
            for namespace, group_assets in grouped.items()
        }

    # ---------------------------------------------------------
    # Observations
    # ---------------------------------------------------------

    def get_observation_cfg(self) -> ObservationManagerCfg:
        """Return task-specific observation groups.

        Default: an ``/object`` group with a pose term per object asset.
        """
        return ObservationManagerCfg(groups={})

    def get_action_cfg(self) -> ActionManagerCfg:
        """Return task-specific action cfg fragment (default empty)."""
        return ActionManagerCfg(terms={})

    # ---------------------------------------------------------
    # Events
    # ---------------------------------------------------------

    @abstractmethod
    def get_event_cfg(self) -> EventManagerCfg:
        """Return task-specific event terms.

        Typically pose-reset events for each task object.
        """

    @abstractmethod
    def build_validator(self) -> "Validator":
        """Build the task validator used for evaluation.

        Returns:
            Validator: Task-specific success/progress validator.
        """
