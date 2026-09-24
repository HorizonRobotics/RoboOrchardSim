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

"""Reset that arranges objects at stated distances from a reference."""

from __future__ import annotations
import logging
import math
import random
from collections.abc import Sequence

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets.rigid_object import RigidObject
from pxr import Usd
from robo_orchard_core.envs.managers.events.event_term import (
    EventTermBase,
    EventTermBaseCfg,
)
from robo_orchard_core.utils.config import Config

from robo_orchard_sim.ext.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg as LabSceneEntityCfg,
)
from robo_orchard_sim.ext.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.ext.envs.manager_based_env import ResetEvent
from robo_orchard_sim.utils.config import ClassType_co
from robo_orchard_sim.utils.polar_layout import (
    PolarLayout,
    PolarPlacement,
)
from robo_orchard_sim.utils.usd import get_prim_aabb

__all__ = [
    "PolarResetTerm",
    "PolarResetTermCfg",
    "RegionCfg",
]

logger = logging.getLogger(__name__)

_Z_CLEARANCE = 0.005
"""Gap left under an object so it rests on the surface, not in it."""


class RegionCfg(Config):
    """An axis-aligned span objects may occupy."""

    x: tuple[float, float]
    y: tuple[float, float]


class PolarResetTerm(
    EventTermBase[ResetEvent, IsaacEnvType_co, "PolarResetTermCfg"],
):
    """Place a reference, then ring it with objects at stated distances.

    One term arranges the whole scene, so the objects are compared
    against each other as they are placed and no state has to be shared
    with other terms. Distance is the only thing an object keeps from
    one episode to the next -- direction is redealt every time, which is
    what stops a policy from finding the target by where it sits rather
    than by how far away it is.
    """

    def __init__(self, cfg: "PolarResetTermCfg", env: IsaacEnvType_co) -> None:
        super().__init__(cfg, env)
        self._cfg = cfg
        self._env = env
        self._assets: dict[str, RigidObject] = {}
        self._z_offsets: dict[str, float] = {}
        self._layout = PolarLayout(
            workspace_x=cfg.workspace.x,
            workspace_y=cfg.workspace.y,
            axis_jitter_deg=cfg.axis_jitter,
            min_separation=cfg.min_separation,
        )

        scene = self._env.scene
        stage = getattr(scene, "stage", None)
        if stage is None and Usd is not None:
            stage = Usd.Stage.Open(scene._usd_path)

        names = [cfg.anchor_name, *sorted(cfg.radius)]
        for name in names:
            asset = scene[name]
            if not isinstance(asset, RigidObject):
                raise TypeError(
                    f"PolarResetTerm can only place rigid objects; "
                    f"'{name}' is a {type(asset).__name__}."
                )
            self._assets[name] = asset
            self._z_offsets[name] = self._resolve_z_offset(asset, stage, name)

    @staticmethod
    def _resolve_z_offset(
        asset: RigidObject,
        stage,
        name: str,
    ) -> float:
        """Height that sets an object down on the surface.

        An object's origin is wherever its author put it, which is not
        always its base, so placing every object at the same z would
        bury some and float others. The registry records how far the
        lowest point sits below the origin; undo that and add a sliver
        of clearance.
        """
        aabb_min = getattr(asset.cfg, "aabb_min", None)
        z_min = None if aabb_min is None else aabb_min[2]
        if z_min is None and stage is not None:
            prim_path = getattr(asset.cfg, "prim_path", None)
            if prim_path is not None:
                aabb = get_prim_aabb(
                    stage, prim_path.replace("env_.*", "env_0")
                )
                if aabb is not None:
                    z_min = aabb[2][1]
        if z_min is None:
            logger.warning(
                "PolarResetTerm: no aabb for '%s'; it may rest "
                "slightly above or below the surface.",
                name,
            )
            return 0.0
        return _Z_CLEARANCE - float(z_min)

    def __call__(self, event_msg: ResetEvent) -> None:
        env_ids = self._resolve_env_ids(event_msg)
        placements = {
            name: PolarPlacement(
                radius=tuple(span),
                axis=self._cfg.axis.get(name),
            )
            for name, span in self._cfg.radius.items()
        }
        for env_id in env_ids.tolist():
            anchor = (
                random.uniform(*self._cfg.anchor_region.x),
                random.uniform(*self._cfg.anchor_region.y),
            )
            # Raises when nothing fits rather than leaving the objects
            # where the last episode left them: the instruction is
            # written from the arrangement this call was meant to
            # produce, so a silent skip would describe a scene that is
            # not there.
            layout = self._layout.solve(anchor, placements)
            self._place(self._cfg.anchor_name, env_id, anchor)
            for name, spot in layout.items():
                self._place(name, env_id, spot)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "PolarResetTerm env=%d anchor=(%.3f, %.3f) %s",
                    env_id,
                    anchor[0],
                    anchor[1],
                    " ".join(
                        f"{name}={math.dist(spot, anchor):.3f}m"
                        for name, spot in sorted(layout.items())
                    ),
                )

    def _resolve_env_ids(self, event_msg: ResetEvent) -> torch.Tensor:
        ids = getattr(event_msg, "env_ids", None)
        if ids is None:
            return torch.arange(self._env.num_envs).to(self._env.device)
        return torch.tensor(ids).to(self._env.device)

    def _place(
        self,
        name: str,
        env_id: int,
        spot: tuple[float, float],
    ) -> None:
        """Set one object down, facing anywhere."""
        asset = self._assets[name]
        device = self._env.device
        position = torch.tensor(
            [spot[0], spot[1], self._z_offsets[name]],
            device=device,
        )
        position = position + self._env.scene.env_origins[env_id]
        yaw = torch.tensor(random.uniform(-math.pi, math.pi))
        zero = torch.tensor(0.0)
        orientation = math_utils.quat_from_euler_xyz(zero, zero, yaw).to(
            device
        )
        env_tensor = torch.tensor([env_id], device=device)
        asset.write_root_pose_to_sim(
            torch.cat([position, orientation.reshape(4)]).unsqueeze(0),
            env_ids=env_tensor,
        )
        # Carrying over the last episode's motion would send the object
        # skidding away from where it was just put.
        asset.write_root_velocity_to_sim(
            torch.zeros((1, 6), device=device),
            env_ids=env_tensor,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """No state survives between episodes."""
        del env_ids


class PolarResetTermCfg(EventTermBaseCfg[PolarResetTerm, LabSceneEntityCfg]):
    """Configuration for :class:`PolarResetTerm`."""

    class_type: ClassType_co[PolarResetTerm] = PolarResetTerm

    trigger_topic: str = "reset"

    anchor_name: str
    """Scene name of the object the others are arranged around."""

    anchor_region: RegionCfg
    """Where the reference itself may land."""

    radius: dict[str, tuple[float, float]]
    """Distance band from the reference, keyed by scene name."""

    axis: dict[str, float] = {}
    """Direction an object must hold, keyed by scene name.

    Only for objects whose description is about direction; anything
    left out is dealt whichever axis is free, redrawn each reset.
    """

    workspace: RegionCfg
    """Bounds every placement must fall inside."""

    axis_jitter: float = 40.0
    """Degrees an object may stray from its axis, either way."""

    min_separation: float = 0.03
    """Least distance between any two placed objects."""
