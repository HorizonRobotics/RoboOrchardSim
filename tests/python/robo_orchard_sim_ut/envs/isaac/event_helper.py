# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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
from __future__ import annotations
from collections.abc import Sequence

from robo_orchard_core.envs.manager_based_env import ResetEvent
from robo_orchard_core.envs.managers.events import (
    EventTermBase,
    EventTermBaseCfg,
)
from typing_extensions import Generic

from robo_orchard_sim.ext.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg as LabSceneEntityCfg,
)
from robo_orchard_sim.ext.envs.env_base import IsaacEnvType_co


class ResetEventTermBase(
    EventTermBase[ResetEvent, IsaacEnvType_co, "ResetEventTermCfg"],
    Generic[IsaacEnvType_co],
):
    """The base class for reset event.

    The reset event is used to trigger the reset event.

    """

    def __init__(self, cfg: ResetEventTermCfg, env: IsaacEnvType_co):
        super().__init__(cfg, env)
        self._id = cfg.id

    def __call__(self, event_msg: ResetEvent) -> None:
        """Trigger the reset event."""
        print(
            f"ResetEventTermBase[{self._id}].__call__ with event_msg:",
            event_msg,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the environment."""
        print("ResetEventTermBase.reset with env_ids:", env_ids)


class ResetEventTermCfg(
    EventTermBaseCfg[ResetEventTermBase[IsaacEnvType_co], LabSceneEntityCfg]
):
    """You should define any additional configuration parameters here."""

    id: int | None = None

    pass
