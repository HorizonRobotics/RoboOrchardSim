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

import deprecated
from isaaclab.managers.observation_manager import ManagerBase as LabManagerBase
from robo_orchard_core.envs.managers.actions.action_manager import (
    ActionManager,
    ActionManagerCfg,
)
from typing_extensions import TypeVar

from robo_orchard_sim.envs.manager_based_env import IsaacManagerBasedEnv

IsaacManagerBasedEnvType_co = TypeVar(
    "IsaacManagerBasedEnvType_co",
    bound=IsaacManagerBasedEnv,
    covariant=True,
)

IsaacActionManagerType_co = TypeVar(
    "IsaacActionManagerType_co",
    bound="IsaacActionManager",
    covariant=True,
)

IsaacActionManagerCfg = ActionManagerCfg

IsaacActionManagerCfgType_co = TypeVar(
    "IsaacActionManagerCfgType_co",
    bound="IsaacActionManagerCfg",
    covariant=True,
)


@deprecated.deprecated(reason="Unnecessary class. Use ActionManager directly.")
class IsaacActionManager(
    ActionManager[IsaacManagerBasedEnvType_co, IsaacActionManagerCfgType_co],
    LabManagerBase,
):
    """The Action Manager for Isaac Lab.

    This class extends `ActionManager` to use the `LabManagerBase` from
    Isaac Lab.

    Args:
        cfg (IsaacActionManagerCfgType_co): The configuration for the action
            manager.
        env (IsaacManagerBasedEnvType_co): The environment.
    """

    def __init__(
        self,
        cfg: IsaacActionManagerCfgType_co,
        env: IsaacManagerBasedEnvType_co,
    ):
        ActionManager.__init__(self, cfg=cfg, env=env)

    def _prepare_terms(self):
        """Prepare the observation terms."""

        # do nothing
        ...
