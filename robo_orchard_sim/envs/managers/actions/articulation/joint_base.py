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

from typing import Sequence

import isaaclab.utils.string as string_utils
import torch
from isaaclab.assets import Articulation
from robo_orchard_core.envs.managers.actions.action_term import (
    ActionTermBase,
    ActionTermCfg,
)
from torch import Tensor
from typing_extensions import TypeVar

from robo_orchard_sim.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg as LabSceneEntityCfg,
)
from robo_orchard_sim.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.utils.config import ClassType_co

__all__ = ["ArticulationJointActionTerm", "ArticulationJointActionTermCfg"]


ArticulationJointActionTermCfgType_co = TypeVar(
    "ArticulationJointActionTermCfgType_co",
    bound="ArticulationJointActionTermCfg",
    covariant=True,
)


class ArticulationJointActionTerm(
    ActionTermBase[IsaacEnvType_co, ArticulationJointActionTermCfgType_co],
):
    def __init__(
        self, cfg: ArticulationJointActionTermCfgType_co, env: IsaacEnvType_co
    ):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(
                self.num_envs, self.action_dim, device=self.device
            )
            # resolve the dictionary config
            (
                index_list,
                _,
                value_list,
            ) = string_utils.resolve_matching_names_values(
                self.cfg.scale, self.joint_names
            )
            self._scale[:, index_list] = torch.tensor(
                value_list, device=self.device
            )
        else:
            raise ValueError(
                f"Unsupported scale type: {type(cfg.scale)}. "
                "Supported types are float and dict."
            )
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            (
                index_list,
                _,
                value_list,
            ) = string_utils.resolve_matching_names_values(
                self.cfg.offset, self.joint_names
            )
            self._offset[:, index_list] = torch.tensor(
                value_list, device=self.device
            )
        else:
            raise ValueError(
                f"Unsupported offset type: {type(cfg.offset)}. "
                "Supported types are float and dict."
            )

    @property
    def action_dim(self) -> int:
        return len(self.cfg.asset_cfg.joint_ids)

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> torch.device:
        return self._env.device

    @property
    def joint_ids(self) -> Sequence[int]:
        return self.cfg.asset_cfg.joint_ids

    @property
    def joint_names(self) -> Sequence[str]:
        return self.cfg.asset_cfg.joint_names

    def _prepare_asset(self) -> None:
        self.cfg.asset_cfg.resolve(self._env.scene)
        self._asset: Articulation = self._env.scene[self.cfg.asset_cfg.name]
        if not isinstance(self._asset, Articulation):
            raise TypeError(
                f"Asset {self.cfg.asset_cfg.name} is not an Articulation"
            )

    def _process_actions_impl(self, raw_actions: Tensor) -> Tensor:
        # apply the affine transformations
        return raw_actions * self._scale + self._offset

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class ArticulationJointActionTermCfg(
    ActionTermCfg[ArticulationJointActionTerm, LabSceneEntityCfg]
):
    """The configuration for the articulation joint action term."""

    class_type: ClassType_co[ArticulationJointActionTerm] = (
        ArticulationJointActionTerm
    )

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions).
    Defaults to 1.0.
    """

    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions).
    Defaults to 0.0."""
