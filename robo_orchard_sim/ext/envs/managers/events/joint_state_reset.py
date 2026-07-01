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

__all__ = ["JointStateResetTerm", "JointStateResetTermCfg"]


class JointStateResetTerm(
    EventTermBase[ResetEvent, IsaacEnvType_co, "JointStateResetTermCfg"],
):
    """Reset articulation joints to a target pose plus Gaussian noise.

    The term draws ``joint_pos = init_joint_pos + noise`` per environment,
    where ``init_joint_pos`` defaults to
    ``articulation.data.default_joint_pos`` and may be overridden via
    ``cfg.init_joint_pos`` (per-joint dict override). The result is
    optionally clamped to
    ``soft_joint_pos_limits`` and then written into the simulator joint
    state and/or the joint position controller targets.

    Joints listed in ``noise_excluded_joint_names`` always receive zero
    noise; ``per_joint_noise_std`` overrides the default ``noise_std`` on
    a per-joint basis. Joint velocities default to the articulation's
    default values when ``write_joint_state`` is enabled.
    """

    def __init__(
        self,
        cfg: "JointStateResetTermCfg",
        env: IsaacEnvType_co,
    ):
        super().__init__(cfg, env)
        self._cfg = cfg
        self._env = env
        self._articulations: list[Articulation] = []
        self._asset_display_names: list[str] = []
        self._init_articulations(cfg.asset_cfgs)
        self._std_vectors: list[torch.Tensor] = [
            self._build_std_vector(articulation, display_name)
            for articulation, display_name in zip(
                self._articulations,
                self._asset_display_names,
                strict=True,
            )
        ]
        self._center_overrides: list[tuple[torch.Tensor, torch.Tensor]] = [
            self._build_init_pos_override(articulation, display_name)
            for articulation, display_name in zip(
                self._articulations,
                self._asset_display_names,
                strict=True,
            )
        ]

    def __call__(self, event_msg: ResetEvent) -> None:
        """Sample noisy joint positions and apply them to selected assets."""
        env_ids = self._resolve_env_ids(event_msg)

        for articulation, std_vec, override, display_name in zip(
            self._articulations,
            self._std_vectors,
            self._center_overrides,
            self._asset_display_names,
            strict=True,
        ):
            joint_pos_before = articulation.data.joint_pos[env_ids].clone()
            print(
                f"[JointStateResetTerm] asset={display_name} "
                f"env_ids={env_ids.tolist()} "
                f"joint_pos before reset: {joint_pos_before}"
            )
            default_pos = articulation.data.default_joint_pos[env_ids].clone()
            default_vel = articulation.data.default_joint_vel[env_ids].clone()

            override_values, override_mask = override
            if override_mask.any():
                center = default_pos.clone()
                center[:, override_mask] = override_values[override_mask]
            else:
                center = default_pos

            noise = torch.randn_like(default_pos) * std_vec
            joint_pos = center + noise

            if self._cfg.clamp_to_joint_limits:
                limits = articulation.data.soft_joint_pos_limits
                lower = limits[env_ids, :, 0]
                upper = limits[env_ids, :, 1]
                joint_pos = joint_pos.clamp(min=lower, max=upper)

            if self._cfg.write_joint_state:
                if self._cfg.reset_joint_velocity_to_default:
                    joint_vel = default_vel
                else:
                    joint_vel = articulation.data.joint_vel[env_ids]
                articulation.write_joint_state_to_sim(
                    joint_pos, joint_vel, env_ids=env_ids
                )

            if self._cfg.write_joint_position_target:
                articulation.set_joint_position_target(
                    joint_pos, env_ids=env_ids
                )
                articulation.write_data_to_sim()

            joint_pos_after = articulation.data.joint_pos[env_ids].clone()
            print(
                f"[JointStateResetTerm] asset={display_name} "
                f"env_ids={env_ids.tolist()} "
                f"joint_pos after reset: {joint_pos_after}"
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset internal state for the term."""
        del env_ids

    def _resolve_env_ids(self, event_msg: ResetEvent) -> torch.Tensor:
        env_ids = event_msg.env_ids
        if env_ids is None:
            return torch.arange(self._env.num_envs, device=self._env.device)
        return torch.as_tensor(
            env_ids, device=self._env.device, dtype=torch.long
        )

    def _init_articulations(
        self, asset_cfgs: list[LabSceneEntityCfg] | None
    ) -> None:
        if asset_cfgs is None:
            for scene_name, asset in self._env.scene.articulations.items():
                self._articulations.append(asset)
                self._asset_display_names.append(scene_name)
        else:
            for asset_cfg in asset_cfgs:
                asset = self._env.scene[asset_cfg.name]
                if not isinstance(asset, Articulation):
                    raise TypeError(
                        f"JointStateResetTerm only supports Articulation, "
                        f"got {type(asset).__name__} for "
                        f"'{asset_cfg.name}'."
                    )
                self._articulations.append(asset)
                self._asset_display_names.append(asset_cfg.name)

        if not self._articulations:
            raise ValueError(
                "JointStateResetTerm requires at least one articulation "
                "in the scene."
            )

    def _build_std_vector(
        self,
        articulation: Articulation,
        display_name: str,
    ) -> torch.Tensor:
        joint_names = list(articulation.joint_names)
        num_joints = len(joint_names)
        device = self._env.device

        if self._cfg.noise_std < 0.0:
            raise ValueError(
                f"noise_std must be non-negative, got {self._cfg.noise_std}."
            )

        std = torch.full(
            (num_joints,),
            float(self._cfg.noise_std),
            dtype=torch.float32,
            device=device,
        )

        per_joint = self._cfg.per_joint_noise_std or {}
        for name, value in per_joint.items():
            if name not in joint_names:
                raise ValueError(
                    f"per_joint_noise_std references unknown joint "
                    f"'{name}' for asset '{display_name}'. "
                    f"Available joints: {joint_names}"
                )
            if value < 0.0:
                raise ValueError(
                    f"per_joint_noise_std['{name}'] must be non-negative, "
                    f"got {value}."
                )
            std[joint_names.index(name)] = float(value)

        excluded = self._cfg.noise_excluded_joint_names or []
        for name in excluded:
            if name not in joint_names:
                raise ValueError(
                    f"noise_excluded_joint_names references unknown "
                    f"joint '{name}' for asset '{display_name}'. "
                    f"Available joints: {joint_names}"
                )
            std[joint_names.index(name)] = 0.0

        return std

    def _build_init_pos_override(
        self,
        articulation: Articulation,
        display_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve ``cfg.init_joint_pos`` into per-joint values and mask.

        Returns ``(values, mask)`` where ``values`` is a ``(num_joints,)``
        float tensor holding the override value at masked positions
        (zeros elsewhere) and ``mask`` is a ``(num_joints,)`` bool tensor
        marking which joints have an override.
        """
        joint_names = list(articulation.joint_names)
        num_joints = len(joint_names)
        device = self._env.device

        values = torch.zeros(num_joints, dtype=torch.float32, device=device)
        mask = torch.zeros(num_joints, dtype=torch.bool, device=device)

        cfg_val = self._cfg.init_joint_pos
        if cfg_val is None:
            return values, mask

        for name, value in cfg_val.items():
            if name not in joint_names:
                raise ValueError(
                    f"init_joint_pos references unknown joint "
                    f"'{name}' for asset '{display_name}'. "
                    f"Available joints: {joint_names}"
                )
            idx = joint_names.index(name)
            values[idx] = float(value)
            mask[idx] = True

        return values, mask


class JointStateResetTermCfg(
    EventTermBaseCfg[JointStateResetTerm, LabSceneEntityCfg]
):
    """Configuration for resetting articulation joints with Gaussian noise."""

    class_type: ClassType_co[JointStateResetTerm] = JointStateResetTerm

    noise_std: float = 0.0
    """Default Gaussian noise standard deviation applied to all joints."""

    per_joint_noise_std: dict[str, float] | None = None
    """Override ``noise_std`` for specific joints, keyed by joint name."""

    noise_excluded_joint_names: list[str] | None = None
    """Joints that always receive zero noise regardless of std settings."""

    init_joint_pos: dict[str, float] | None = None
    """Override the noise center per joint.

    ``None`` (default) keeps the articulation's
    ``default_joint_pos`` as the noise center.

    A ``dict[str, float]`` provides a partial override keyed by joint
    name; listed joints use the given value as the noise center, while
    unlisted joints fall back to the default. Unknown joint names raise
    ``ValueError``.
    """

    clamp_to_joint_limits: bool = True
    """Whether to clamp sampled positions to ``soft_joint_pos_limits``."""

    write_joint_state: bool = True
    """Whether to write the noisy positions into the simulator joint state."""

    write_joint_position_target: bool = True
    """Whether to push the noisy positions to the joint controller targets."""

    reset_joint_velocity_to_default: bool = True
    """Whether to restore joint velocities to default values on reset."""
