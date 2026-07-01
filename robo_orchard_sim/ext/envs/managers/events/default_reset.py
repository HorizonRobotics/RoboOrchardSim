# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
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

from collections.abc import Sequence

import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.assets.deformable_object import DeformableObject
from isaaclab.assets.rigid_object import RigidObject
from robo_orchard_core.envs.manager_based_env import ResetEvent
from robo_orchard_core.envs.managers.events.event_term import (
    EventTermBase,
    EventTermBaseCfg,
)

from robo_orchard_sim.ext.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg as LabSceneEntityCfg,
)
from robo_orchard_sim.ext.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.utils.config import ClassType_co

__all__ = ["DefaultResetTerm", "DefaultResetTermCfg"]


class DefaultResetTerm(
    EventTermBase[ResetEvent, IsaacEnvType_co, "DefaultResetTermCfg"],
):
    """Reset selected scene assets to their configured default state.

    Reference:
        https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/events.py
    """

    def __init__(self, cfg: "DefaultResetTermCfg", env: IsaacEnvType_co):
        super().__init__(cfg, env)
        self._cfg = cfg
        self._env = env
        self._assets = self._init_assets(cfg.asset_cfgs)

    def __call__(self, event_msg: ResetEvent) -> None:
        """Reset rigid, articulated, and deformable assets for the event."""
        env_ids = self._resolve_env_ids(event_msg)

        for asset in self._assets:
            if isinstance(asset, RigidObject):
                default_root_state = asset.data.default_root_state[
                    env_ids
                ].clone()
                default_root_state[:, 0:3] += self._env.scene.env_origins[
                    env_ids
                ]
                asset.write_root_pose_to_sim(
                    default_root_state[:, :7], env_ids=env_ids
                )
                asset.write_root_velocity_to_sim(
                    default_root_state[:, 7:], env_ids=env_ids
                )
            elif isinstance(asset, Articulation):
                default_root_state = asset.data.default_root_state[
                    env_ids
                ].clone()
                default_root_state[:, 0:3] += self._env.scene.env_origins[
                    env_ids
                ]
                asset.write_root_pose_to_sim(
                    default_root_state[:, :7], env_ids=env_ids
                )
                asset.write_root_velocity_to_sim(
                    default_root_state[:, 7:], env_ids=env_ids
                )

                default_joint_pos = asset.data.default_joint_pos[
                    env_ids
                ].clone()
                default_joint_vel = asset.data.default_joint_vel[
                    env_ids
                ].clone()
                asset.write_joint_state_to_sim(
                    default_joint_pos, default_joint_vel, env_ids=env_ids
                )

                if self._cfg.reset_joint_targets:
                    asset.set_joint_position_target(
                        default_joint_pos, env_ids=env_ids
                    )
                    asset.set_joint_velocity_target(
                        default_joint_vel, env_ids=env_ids
                    )
                    asset.write_data_to_sim()
            elif isinstance(asset, DeformableObject):
                nodal_state = asset.data.default_nodal_state_w[env_ids].clone()
                asset.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)
            else:
                raise TypeError(
                    f"Asset '{asset.name}' is not a RigidObject, Articulation or DeformableObject."  # noqa: E501
                )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset internal state for the term."""
        pass

    def _resolve_env_ids(self, event_msg: ResetEvent) -> torch.Tensor:
        env_ids = getattr(event_msg, "env_ids", None)
        if env_ids is None:
            return torch.arange(self._env.num_envs, device=self._env.device)
        return torch.as_tensor(
            env_ids, device=self._env.device, dtype=torch.long
        )

    def _init_assets(
        self, asset_cfgs: list[LabSceneEntityCfg] | None
    ) -> list[RigidObject | Articulation | DeformableObject]:
        asset_list: list[RigidObject | Articulation | DeformableObject] = []

        if asset_cfgs is None:
            asset_list.extend(self._env.scene.rigid_objects.values())
            asset_list.extend(self._env.scene.articulations.values())
            asset_list.extend(self._env.scene.deformable_objects.values())
        else:
            for asset_cfg in asset_cfgs:
                asset = self._env.scene[asset_cfg.name]
                if not isinstance(
                    asset, (RigidObject, Articulation, DeformableObject)
                ):
                    raise TypeError(
                        f"Asset '{asset_cfg.name}' is not a RigidObject, Articulation or DeformableObject."  # noqa: E501
                    )
                asset_list.append(asset)

        if not asset_list:
            raise ValueError("No asset is found in the scene.")

        return asset_list


class DefaultResetTermCfg(
    EventTermBaseCfg[DefaultResetTerm, LabSceneEntityCfg]
):
    """Configuration for resetting assets to their scene defaults."""

    class_type: ClassType_co[DefaultResetTerm] = DefaultResetTerm

    reset_joint_targets: bool = False
    """Whether articulation joint targets are restored to default values."""
