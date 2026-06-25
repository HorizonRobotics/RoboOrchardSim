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

"""Observation term for observing the last action."""

from __future__ import annotations
from typing import Any

import torch
from robo_orchard_core.envs.managers.observations import (
    ObservationTermBase,
    ObservationTermCfg,
)
from typing_extensions import Sequence

from robo_orchard_sim.ext.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.utils.config import ClassType_co


class LastActionObservationTerm(
    ObservationTermBase[
        IsaacEnvType_co,
        "LastActionObservationTermCfg",
        dict[str, torch.Tensor] | torch.Tensor,
    ]
):
    """Observation term for observing the last action.

    If ``action_name`` is ``None``, all action terms are returned as a
    dictionary keyed by term name. Otherwise, only the specified action term
    is returned.

    """

    def __call__(self) -> dict[str, torch.Tensor] | torch.Tensor:
        """Get the last action observation."""
        if self.cfg.action_name is None:
            return self._env.action_manager.action

        try:
            return self._env.action_manager.action[self.cfg.action_name]
        except KeyError as err:
            raise ValueError(
                f"Action term {self.cfg.action_name} is not found in "
                f"action manager. Available action terms: "
                f"{self._env.action_manager.active_terms}."
            ) from err

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the observation term."""
        del env_ids


class LastActionObservationTermCfg(
    ObservationTermCfg[LastActionObservationTerm, Any],
):
    """Configuration for the last action observation term."""

    class_type: ClassType_co[LastActionObservationTerm] = (
        LastActionObservationTerm
    )

    action_name: str | None = None
    """Action term name to observe. If ``None``, return all actions."""

    asset_cfg: Any = None
    """Unused placeholder to match the observation term config base class."""
