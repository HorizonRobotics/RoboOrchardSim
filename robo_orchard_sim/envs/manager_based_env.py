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


"""Manager-based Env for Isaac."""

from __future__ import annotations
from typing import Generic

import torch
from isaaclab.envs import ManagerBasedEnv
from robo_orchard_core.envs.env_base import EnvStepReturn
from robo_orchard_core.envs.manager_based_env import (
    TermManagerBasedEnv,
    TermManagerBasedEnvCfg,
)
from robo_orchard_core.envs.managers.actions.action_manager import (
    ActionManagerCfg,
    ActionTermCfg,
)
from robo_orchard_core.envs.managers.events import (
    EventManagerCfg,
    EventTermBaseCfg,
)
from robo_orchard_core.envs.managers.observations.observation_manager import (
    ObservationManagerCfg,
    ObsReturnType,
)
from typing_extensions import Dict, Sequence, TypeAlias, TypeVar

from robo_orchard_sim.cfg_wrappers.scenes_cfg import InteractiveSceneCfg
from robo_orchard_sim.envs.env_base import IsaacEnv, IsaacEnvCfg
from robo_orchard_sim.utils.config import ClassType_co

# EnvReturnType: TypeAlias = Tuple[ObsReturnType, dict]
RewardsType: TypeAlias = (
    None  # Placeholder for rewards type, can be extended later
)
EnvReturnType: TypeAlias = EnvStepReturn[ObsReturnType, RewardsType]


IsaacManagerBasedEnvType_co = TypeVar(
    "IsaacManagerBasedEnvType_co",
    bound="IsaacManagerBasedEnv",
    covariant=True,
)


class IsaacManagerBasedEnv(
    IsaacEnv["IsaacManagerBasedEnvCfg"],
    TermManagerBasedEnv[
        "IsaacManagerBasedEnvCfg", RewardsType
    ],  # TODO: Add reward
    ManagerBasedEnv,
):
    """The Env wrapper for `ManagerBasedEnv` from Isaac Lab.

    This class extends `EnvBase` to use the `ManagerBasedEnv` from Isaac Lab
    implementation.

    Difference from the `ManagerBasedEnv` class:
    - The configuration class is `IsaacManagerBasedEnvCfg`.
    - The scene manager is wrapped with `InteractiveScene`.
    - The observation manager is wrapped with `ObservationManager`.
    - The action manager is wrapped with `ActionManager`.


    """

    def __init__(self, cfg: IsaacManagerBasedEnvCfg):
        IsaacEnv.__init__(self, cfg=cfg)
        self._load_managers()
        # allocate dictionary to store metrics
        self.extras = {}

        # notify the start up event
        self.event_manager.notify(self.START_UP[0], self.START_UP[1]())

    def step(
        self, action: Dict[str, torch.Tensor] | None = None
    ) -> EnvReturnType:
        """Execute one step of the environment.

        Wrapper for the `ManagerBasedEnv.step` method to support empty actions
        and wrapped managers.
        """

        if action is not None:
            self.action_manager.process(action)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            if action is not None or self.cfg.apply_action_when_no_action:
                # set actions into buffers
                self.action_manager.apply()
                # set actions into simulator
                self.scene.write_data_to_sim()

            # inner loop for sim stepping
            self._sim_step_counter += 1
            # simulate
            self.sim.step(render=False)
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

            # notify the step event
            self.event_manager.notify(
                self.STEP[0],
                self.STEP[1](
                    step_id=self._sim_step_counter,
                    step_time=self.physics_dt,
                ),
            )

        return EnvStepReturn(
            observations=self.observation_manager.get_observations(),
            rewards=None,  # TODO: Add rewards
            terminated=None,  # TODO: support termination
            truncated=None,  # TODO:  support truncation
            info=self.extras,
        )

    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None
    ) -> EnvReturnType:
        # reset the environment.
        # _reset_idx is called in IsaacEnv.reset
        IsaacEnv.reset(self, seed=seed, env_ids=env_ids)
        # notify the reset event
        self.event_manager.notify(
            self.RESET[0], self.RESET[1](env_ids=env_ids, seed=seed)
        )
        return EnvStepReturn(
            observations=self.observation_manager.get_observations(),
            rewards=None,  # TODO: Add rewards
            terminated=None,  # TODO: support termination
            truncated=None,  # TODO:  support truncation
            info=self.extras,
        )

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        IsaacEnv._reset_idx(self, env_ids)

        # iterate over all managers and reset them
        self._reset_managers(env_ids)
        # self.extras["log"] = dict()

        # -- observation manager
        # self.extras["log"].update(info)
        # -- action manager
        # self.extras["log"].update(info)

    def close(self):
        """Cleanup for the environment."""
        IsaacEnv.close(self)
        # We don't need to call `del` on the managers since they are weak
        # references and will be garbage collected automatically.


InteractiveSceneCfgType_co = TypeVar(
    "InteractiveSceneCfgType_co",
    bound=InteractiveSceneCfg,
    covariant=True,
    default=InteractiveSceneCfg,
)


class IsaacManagerBasedEnvCfg(
    IsaacEnvCfg[IsaacManagerBasedEnv, InteractiveSceneCfgType_co],
    TermManagerBasedEnvCfg[IsaacManagerBasedEnv],
    Generic[InteractiveSceneCfgType_co],
):
    """The configuration for IsaacManagerBasedEnv.

    This config extends the `IsaacEnvCfg` to include the observation
    and action manager configurations.

    """

    class_type: ClassType_co[IsaacManagerBasedEnv] = IsaacManagerBasedEnv

    observations: ObservationManagerCfg = ObservationManagerCfg(groups={})
    actions: ActionManagerCfg[ActionTermCfg] = ActionManagerCfg(terms={})

    events: EventManagerCfg[EventTermBaseCfg] = EventManagerCfg(terms={})

    apply_action_when_no_action: bool = False
    """Whether to apply the action when no action is provided in step."""
