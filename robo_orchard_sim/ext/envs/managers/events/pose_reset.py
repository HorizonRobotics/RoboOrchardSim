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

import logging
import random
from collections.abc import Sequence

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.assets.deformable_object import DeformableObject
from isaaclab.assets.rigid_object import RigidObject
from pxr import Usd
from robo_orchard_core.envs.manager_based_env import ResetEvent
from robo_orchard_core.envs.managers.events.event_term import (
    EventTermBase,
    EventTermBaseCfg,
)
from typing_extensions import Literal

from robo_orchard_sim.ext.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg as LabSceneEntityCfg,
)
from robo_orchard_sim.ext.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.utils.config import ClassType_co
from robo_orchard_sim.utils.env_utils import sample_poses
from robo_orchard_sim.utils.usd import get_prim_aabb

__all__ = ["PoseResetTerm", "PoseResetTermCfg"]

logger = logging.getLogger(__name__)

_Z_CLEARANCE = 0.005

_CacheEntry = tuple[
    str,
    tuple[float, float],
    tuple[float, float],
    float | None,
    float | None,
]
_CROSS_GROUP_CACHE: dict[str, dict[int, list[_CacheEntry]]] = {}


class PoseResetTerm(
    EventTermBase[ResetEvent, IsaacEnvType_co, "PoseResetTermCfg"],
):
    def _get_asset_tag(
        self,
        asset: RigidObject | Articulation | DeformableObject,
        fallback_index: int | None = None,
    ) -> str:
        """Return a stable cache key for an asset."""
        spawn_cfg = getattr(asset.cfg, "spawn", None)
        semantic_tags = getattr(spawn_cfg, "semantic_tags", None)
        if semantic_tags:
            return str(semantic_tags[0][1])

        prim_path = getattr(asset.cfg, "prim_path", None)
        if prim_path:
            return prim_path.rstrip("/").split("/")[-1]

        asset_name = getattr(asset, "name", None)
        if asset_name:
            return str(asset_name)

        if fallback_index is not None:
            return f"asset_{fallback_index}"
        return "asset"

    def __init__(self, cfg: "PoseResetTermCfg", env: IsaacEnvType_co):
        super().__init__(cfg, env)
        self._cfg = cfg
        self._env = env
        self._mode = cfg.mode
        self._count = 0
        self._asset_xy_extents = {}
        # Only used in drop mode.
        self._prefer_stack = getattr(cfg, "prefer_stack", False)

        # Validate mode-specific configuration.
        if self._mode in {"random", "random_non_overlap", "drop"}:
            if not self._cfg.pose_range:
                raise ValueError(
                    "pose_range must be provided in PoseResetTermCfg when mode is 'random'"  # noqa: E501
                )
        elif self._mode == "orderly":
            if not self._cfg.pose_list:
                raise ValueError(
                    "pose_list must be provided in PoseResetTermCfg when mode is 'orderly'"  # noqa: E501
                )
        elif self._mode == "default":
            pass
        else:
            raise ValueError(f"Invalid mode: {self._mode}")

        self._assets = self._init_asset(cfg.asset_cfgs)

        # Shared placement cache for random_non_overlap and drop.
        if self._mode in {"random_non_overlap", "drop"}:
            self._group_key = cfg.group_key
            self._clear_cross_group_cache = bool(cfg.clear_cross_group_cache)
        else:
            self._group_key = None
            self._clear_cross_group_cache = False

        if self._mode in {"random", "random_non_overlap", "drop"}:
            # XY / Z half-extents come from the live USD AABB (frame-agnostic
            # sizes). Spawn-clearance z_min comes from the registry's
            # asset-local AABB carried on the asset cfg.
            self._asset_xy_extents: dict[str, tuple[float, float]] = {}
            self._asset_z_half_extents: dict[str, float] = {}
            self._asset_z_min: dict[str, float | None] = {}
            stage = getattr(self._env.scene, "stage", None)
            if stage is None and Usd is not None:
                stage = Usd.Stage.Open(self._env.scene._usd_path)
            for asset_idx, asset in enumerate(self._assets):
                prim_path = getattr(asset.cfg, "prim_path", None)
                if prim_path is not None:
                    prim_path = prim_path.replace("env_.*", "env_0")
                tag = self._get_asset_tag(asset, asset_idx)
                hx = hy = hz = 0.01  # fallback small extents
                if stage is not None and prim_path is not None:
                    aabb = get_prim_aabb(stage, prim_path)
                    if aabb is not None:
                        (x_max, x_min), (y_max, y_min), (z_max, z_bot) = aabb
                        hx = abs(x_max - x_min) * 0.5
                        hy = abs(y_max - y_min) * 0.5
                        hz = abs(z_max - z_bot) * 0.5
                self._asset_xy_extents[tag] = (hx, hy)
                self._asset_z_half_extents[tag] = hz
                z_min = getattr(asset.cfg, "aabb_z_min", None)
                if z_min is None:
                    logger.warning(
                        "PoseResetTerm: no registry aabb_z_min for '%s'; "
                        "spawn-clearance clamp skipped for it.",
                        tag,
                    )
                self._asset_z_min[tag] = z_min

    def __call__(self, event_msg: ResetEvent):
        """Do the event term operation."""
        self._count += 1
        if not hasattr(event_msg, "env_ids"):
            env_ids = torch.arange(self._env.num_envs).to(self._env.device)
        elif event_msg.env_ids is None:
            env_ids = torch.arange(self._env.num_envs).to(self._env.device)
        else:
            env_ids = torch.tensor(event_msg.env_ids).to(self._env.device)

        if self._mode == "default":
            self._set_default_root_state(env_ids)
        elif self._mode == "random_non_overlap":
            self._apply_random_non_overlap(env_ids)
        elif self._mode == "drop":
            self._apply_drop_reset(env_ids)
        else:
            for asset_idx, asset in enumerate(self._assets):
                rand_samples = self._sample_pose(env_ids)
                if self._mode == "random":
                    tag = self._get_asset_tag(asset, asset_idx)
                    z_min = self._asset_z_min.get(tag)
                    if z_min is not None:
                        default_z = asset.data.default_root_state[env_ids, 2]
                        floor_offset = _Z_CLEARANCE - z_min - default_z
                        rand_samples[:, 2] = torch.maximum(
                            rand_samples[:, 2], floor_offset
                        )
                root_states = asset.data.default_root_state[env_ids].clone()

                if self._mode == "orderly":
                    # pose_list entries are absolute world-frame poses;
                    # env_origins still applies for multi-env layouts.
                    positions = (
                        self._env.scene.env_origins[env_ids]
                        + rand_samples[:, 0:3]
                    )
                    orientations = rand_samples[:, 3:7]
                else:
                    positions = (
                        root_states[:, 0:3]
                        + self._env.scene.env_origins[env_ids]
                        + rand_samples[:, 0:3]
                    )
                    orientations = math_utils.quat_mul(
                        root_states[:, 3:7], rand_samples[:, 3:7]
                    )
                asset.write_root_pose_to_sim(
                    torch.cat([positions, orientations], dim=-1),
                    env_ids=env_ids,
                )
                asset.write_root_velocity_to_sim(
                    root_states[:, 7:], env_ids=env_ids
                )

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset internal state for the term."""
        pass

    def _sample_pose(self, env_ids: Sequence[int]) -> torch.Tensor:
        if self._mode == "random":
            random_poses = torch.tensor(
                sample_poses(
                    self._cfg.pose_range, mode="scattered", len=len(env_ids)
                )
            ).to(self._env.device, dtype=torch.float32)
        elif self._mode == "orderly":
            id = self._count % len(self._cfg.pose_list)
            random_poses = (
                torch.tensor(self._cfg.pose_list[id])
                .to(self._env.device, dtype=torch.float32)
                .repeat(len(env_ids), 1)
            )
        return random_poses

    def _get_group_cache(
        self, env_ids: torch.Tensor
    ) -> tuple[list[int], dict[int, list[_CacheEntry]]]:
        """Return cross-term cache entries keyed by actual environment id."""
        env_id_list = [int(env_id) for env_id in env_ids.tolist()]
        global_cache = _CROSS_GROUP_CACHE.setdefault(self._group_key, {})
        if self._clear_cross_group_cache:
            global_cache.clear()
        for env_id in env_id_list:
            global_cache.setdefault(env_id, [])
        return env_id_list, global_cache

    def _apply_random_non_overlap(self, env_ids: torch.Tensor):
        """Sample poses ensuring XY non-overlap for all configured assets.

        Supports optional absolute sampling, ignoring default root state.
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids).to(self._env.device)

        min_sep = float(self._cfg.min_separation)
        max_retries = int(self._cfg.max_retries)
        absolute_resample = bool(self._cfg.absolute_sampling)
        group_attempts = 256

        # Retry the whole asset group to avoid partial placements.
        attempt = 0
        success = False
        final_env_cache: list[list[_CacheEntry]] = []
        placements: list[
            tuple[RigidObject | Articulation, torch.Tensor, torch.Tensor]
        ] = []
        while attempt < group_attempts and not success:
            # prepare / reset cache this attempt
            if self._group_key:
                env_id_list, global_cache = self._get_group_cache(env_ids)
                env_cache = [
                    list(global_cache[env_id]) for env_id in env_id_list
                ]
            else:
                env_id_list = []
                env_cache = [[] for _ in range(len(env_ids))]
            placements.clear()
            overall_failed = False
            for asset_idx, asset in enumerate(self._assets):
                if overall_failed:
                    break
                tag = self._get_asset_tag(asset, asset_idx)
                hx, hy = self._asset_xy_extents.get(tag, (0.01, 0.01))
                asset_offsets: list[list[float]] = []
                default_root_states = asset.data.default_root_state[
                    env_ids
                ].clone()
                env_origins = self._env.scene.env_origins[env_ids]
                for i_env in range(len(env_ids)):
                    retries = 0
                    accepted: list[float] | None = None
                    while retries < max_retries:
                        cand = self._uniform_sample_pose()
                        if (
                            self._cfg.pose_range
                            and "z" in self._cfg.pose_range
                        ):
                            z_min_cfg, z_max_cfg = self._cfg.pose_range["z"]
                            cand[2] = (z_min_cfg + z_max_cfg) * 0.5
                        else:
                            cand[2] = 0.0
                        z_min = self._asset_z_min.get(tag)
                        if z_min is not None:
                            floor_offset = _Z_CLEARANCE - z_min
                            if not absolute_resample:
                                floor_offset -= float(
                                    default_root_states[i_env, 2]
                                )
                            if cand[2] < floor_offset:
                                cand[2] = floor_offset
                        if absolute_resample:
                            base_x = env_origins[i_env, 0]
                            base_y = env_origins[i_env, 1]
                        else:
                            base_x = (
                                default_root_states[i_env, 0]
                                + env_origins[i_env, 0]
                            )
                            base_y = (
                                default_root_states[i_env, 1]
                                + env_origins[i_env, 1]
                            )
                        center = (
                            float(base_x + cand[0]),
                            float(base_y + cand[1]),
                        )
                        overlap = False
                        for (
                            _,
                            other_center,
                            (ohx, ohy),
                            _cached_hz,
                            _cached_top_z,
                        ) in env_cache[i_env]:
                            dx = abs(center[0] - other_center[0])
                            dy = abs(center[1] - other_center[1])
                            if dx < (hx + ohx + min_sep) and dy < (
                                hy + ohy + min_sep
                            ):
                                overlap = True
                                break
                        if not overlap:
                            accepted = cand
                            env_cache[i_env].append(
                                (
                                    tag,
                                    center,
                                    (hx, hy),
                                    None,
                                    None,
                                )
                            )
                            break
                        retries += 1
                    if accepted is None:
                        overall_failed = True
                        break
                    asset_offsets.append(accepted)
                if overall_failed:
                    break
                offsets_tensor = torch.tensor(asset_offsets).to(
                    self._env.device, dtype=torch.float32
                )
                root_states = asset.data.default_root_state[env_ids].clone()
                placements.append((asset, root_states, offsets_tensor))
            if not overall_failed:
                success = True
                final_env_cache = env_cache
            else:
                attempt += 1
        if not success:
            print(
                f"[PoseResetTerm] Group sampling failed after {group_attempts} attempts; placements skipped."  # noqa: E501
            )
            return
        for asset, root_states, offsets_tensor in placements:
            if absolute_resample:
                positions = (
                    self._env.scene.env_origins[env_ids]
                    + offsets_tensor[:, 0:3]
                )
            else:
                positions = (
                    root_states[:, 0:3]
                    + self._env.scene.env_origins[env_ids]
                    + offsets_tensor[:, 0:3]
                )
            orientations = math_utils.quat_mul(
                root_states[:, 3:7], offsets_tensor[:, 3:7]
            )
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1),
                env_ids=env_ids,
            )
            asset.write_root_velocity_to_sim(
                root_states[:, 7:], env_ids=env_ids
            )
        if self._group_key:
            for env_id, entries in zip(
                env_id_list, final_env_cache, strict=True
            ):
                global_cache[env_id] = list(entries)
            _CROSS_GROUP_CACHE[self._group_key] = global_cache
        return

    def _uniform_sample_pose(self) -> list[float]:
        """Sample one [x,y,z,qw,qx,qy,qz] within ``pose_range``.

        Only yaw rotation is applied; roll/pitch are taken if provided else 0.
        """
        pr = self._cfg.pose_range or {}

        def rng(axis: str):
            if axis in pr:
                lo, hi = pr[axis]
                return random.uniform(float(lo), float(hi))
            return 0.0

        x = rng("x")
        y = rng("y")
        z = rng("z")
        roll = rng("roll")
        pitch = rng("pitch")
        yaw = rng("yaw")

        quat = math_utils.quat_from_euler_xyz(
            torch.tensor(roll, dtype=torch.float32),
            torch.tensor(pitch, dtype=torch.float32),
            torch.tensor(yaw, dtype=torch.float32),
        )
        qw, qx, qy, qz = map(float, quat.tolist())
        return [x, y, z, qw, qx, qy, qz]

    def _apply_drop_reset(self, env_ids: torch.Tensor):
        """Stack objects vertically within z-range using AABB extents.

        Steps per environment:
        1. For each asset, sample (x,y) (and yaw) uniformly from pose_range.
        2. Determine vertical placement: if overlaps in XY with prior placed
           objects, put it on top of the highest overlapping top surface.
           Otherwise place at z_min.
        3. Reject and resample if resulting top would exceed z_max.
        4. After success, set root_z = bottom_z + hz (centered), store top_z.
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids).to(self._env.device)
        pr = self._cfg.pose_range or {}
        if "z" not in pr:
            raise ValueError("drop mode requires pose_range['z']")
        z_min, z_max = pr["z"]
        max_retries = int(self._cfg.max_retries)
        min_sep = float(self._cfg.min_separation)
        absolute_resample = bool(self._cfg.absolute_sampling)

        env_cache: list[
            list[tuple[float, float, float, float, float, float]]
        ] = [[] for _ in range(len(env_ids))]
        if self._group_key:
            env_id_list, global_cache = self._get_group_cache(env_ids)
            for i_env, env_id in enumerate(env_id_list):
                for (
                    tag,
                    center,
                    (ghx, ghy),
                    cached_hz,
                    cached_top_z,
                ) in global_cache[env_id]:
                    ghz = cached_hz
                    if ghz is None:
                        ghz = self._asset_z_half_extents.get(tag, 0.01)
                    env_cache[i_env].append(
                        (
                            center[0],
                            center[1],
                            ghx,
                            ghy,
                            ghz,
                            (
                                cached_top_z
                                if cached_top_z is not None
                                else z_min + 2 * ghz
                            ),
                        )
                    )
        else:
            env_id_list = []

        new_global_entries: list[list[_CacheEntry]] = [
            [] for _ in range(len(env_ids))
        ]
        placements: list[tuple[RigidObject | Articulation, torch.Tensor]] = []
        # Randomize stacking order for each reset.
        asset_iter = random.sample(self._assets, len(self._assets))
        for asset_idx, asset in enumerate(asset_iter):
            tag = self._get_asset_tag(asset, asset_idx)
            hx, hy = self._asset_xy_extents.get(tag, (0.01, 0.01))
            hz = self._asset_z_half_extents.get(tag, 0.01)
            default_root_states = asset.data.default_root_state[
                env_ids
            ].clone()
            env_origins = self._env.scene.env_origins[env_ids]
            asset_offsets: list[list[float]] = []
            for i_env in range(len(env_ids)):
                retries = 0
                accepted: list[float] | None = None
                while retries < max_retries:
                    cand = self._uniform_sample_pose()
                    x_off, y_off = cand[0], cand[1]
                    # Use the sampled quaternion directly.
                    quat = torch.tensor(cand[3:7], dtype=torch.float32)
                    if absolute_resample:
                        base_x = env_origins[i_env, 0]
                        base_y = env_origins[i_env, 1]
                    else:
                        base_x = (
                            default_root_states[i_env, 0]
                            + env_origins[i_env, 0]
                        )
                        base_y = (
                            default_root_states[i_env, 1]
                            + env_origins[i_env, 1]
                        )
                    cx = float(base_x + x_off)
                    cy = float(base_y + y_off)
                    # Compute the placement height from overlapping assets.
                    if self._prefer_stack and env_cache[i_env]:
                        # Always stack on the current highest top surface.
                        highest_top = max(
                            entry[5] for entry in env_cache[i_env]
                        )
                        bottom_z = highest_top
                    else:
                        highest_top = z_min
                        overlap_any = False
                        for (
                            ocx,
                            ocy,
                            ohx,
                            ohy,
                            _ohz,
                            otop,
                        ) in env_cache[i_env]:
                            dx = abs(cx - ocx)
                            dy = abs(cy - ocy)
                            if dx < (hx + ohx + min_sep) and dy < (
                                hy + ohy + min_sep
                            ):
                                overlap_any = True
                                if otop > highest_top:
                                    highest_top = otop
                        bottom_z = highest_top if overlap_any else z_min
                    root_z = bottom_z + hz
                    top_z = root_z + hz
                    # Ignore the z_max limit when forced stacking is enabled.
                    if (not self._prefer_stack) and top_z > z_max:
                        retries += 1
                        continue
                    qw, qx, qy, qz = map(float, quat.tolist())
                    accepted = [x_off, y_off, root_z, qw, qx, qy, qz]
                    env_cache[i_env].append((cx, cy, hx, hy, hz, top_z))
                    break
                if accepted is None:
                    if absolute_resample:
                        pos0 = default_root_states[i_env, 0:3]
                        accepted = [
                            float(pos0[0]),
                            float(pos0[1]),
                            float(pos0[2]),
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                        cx = float(pos0[0] + env_origins[i_env, 0])
                        cy = float(pos0[1] + env_origins[i_env, 1])
                        top_z_cache = float(pos0[2] + hz)
                    else:
                        accepted = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                        cx = (
                            default_root_states[i_env, 0]
                            + env_origins[i_env, 0]
                        )
                        cy = (
                            default_root_states[i_env, 1]
                            + env_origins[i_env, 1]
                        )
                        top_z_cache = z_min + 2 * hz
                    env_cache[i_env].append((cx, cy, hx, hy, hz, top_z_cache))
                else:
                    top_z_cache = top_z

                new_global_entries[i_env].append(
                    (
                        tag,
                        (cx, cy),
                        (hx, hy),
                        hz,
                        top_z_cache,
                    )
                )
                asset_offsets.append(accepted)
            offsets_tensor = torch.tensor(asset_offsets).to(
                self._env.device, dtype=torch.float32
            )
            placements.append((asset, offsets_tensor))
        for asset, offsets_tensor in placements:
            root_states = asset.data.default_root_state[env_ids].clone()
            if absolute_resample:
                positions = (
                    self._env.scene.env_origins[env_ids]
                    + offsets_tensor[:, 0:3]
                )
            else:
                positions = (
                    root_states[:, 0:3]
                    + self._env.scene.env_origins[env_ids]
                    + offsets_tensor[:, 0:3]
                )
            orientations = math_utils.quat_mul(
                root_states[:, 3:7], offsets_tensor[:, 3:7]
            )
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=env_ids
            )
            asset.write_root_velocity_to_sim(
                root_states[:, 7:], env_ids=env_ids
            )

        if self._group_key:
            for env_id, entries in zip(
                env_id_list, new_global_entries, strict=True
            ):
                global_cache[env_id].extend(entries)
            _CROSS_GROUP_CACHE[self._group_key] = global_cache
        return

    def _set_default_root_state(self, env_ids: Sequence[int]):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids).to(self._env.device)

        for asset in self._assets:
            if isinstance(asset, RigidObject):
                # Offset the default root pose by the environment origin.
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
                # Offset the default root pose by the environment origin.
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

            elif isinstance(asset, DeformableObject):
                # Restore the default nodal state directly.
                nodal_state = asset.data.default_nodal_state_w[env_ids].clone()
                asset.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)

            else:
                raise TypeError(
                    f"Asset '{asset.name}' is not a RigidObject, Articulation or DeformableObject."  # noqa: E501
                )

    def _init_asset(
        self, asset_cfgs: list[LabSceneEntityCfg]
    ) -> list[RigidObject | Articulation]:
        asset_list: list[RigidObject | Articulation] = []

        if asset_cfgs is None:
            for rigid_object in self._env.scene.rigid_objects.values():
                asset_list.append(rigid_object)
            for articulation_asset in self._env.scene.articulations.values():
                asset_list.append(articulation_asset)
        else:
            for it in asset_cfgs:
                asset = self._env.scene[it.name]
                if not isinstance(
                    asset, (RigidObject, Articulation, DeformableObject)
                ):
                    raise TypeError(
                        f"Asset '{it.name}' is not a RigidObject or Articulation or DeformableObject."  # noqa: E501
                    )
                asset_list.append(asset)

        if asset_list is None:
            raise ValueError("No asset is found in the scene.")

        return asset_list


class PoseResetTermCfg(EventTermBaseCfg[PoseResetTerm, LabSceneEntityCfg]):
    """Configuration for the pose reset event term."""

    class_type: ClassType_co[PoseResetTerm] = PoseResetTerm

    mode: Literal["random", "random_non_overlap", "orderly", "default", "drop"]
    """Reset asset root state using one of the supported modes.

    * ``random``: Sample offsets from ``pose_range`` and apply them relative
      to the default root pose.
    * ``orderly``: Apply poses from ``pose_list`` in sequence.
    * ``default``: Restore the default root state.
    * ``random_non_overlap``: Sample offsets from ``pose_range`` while
      preventing XY overlap across the selected assets in each environment.
    * ``drop``: Sample offsets from ``pose_range`` and stack assets vertically
      when XY overlap occurs.
    """

    pose_range: dict[str, tuple[float, float]] | None = None
    """Pose sampling range.

    The keys of the dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``,
    and ``yaw``. The values are tuples of the form ``(min, max)``.
    Missing keys default to zero for the corresponding axis.
    """

    pose_list: list[list[float]] | None = None
    """List of poses used by ``orderly`` mode.

    Each pose is ``[x, y, z, qw, qx, qy, qz]``.
    """

    absolute_sampling: bool = False
    """Treat sampled offsets as positions relative to the environment origin.

    When enabled, the default root translation is ignored.
    """

    min_separation: float = 0.0
    """Extra padding applied when checking overlap."""

    max_retries: int = 128
    """Maximum rejection-sampling attempts per asset and environment."""

    group_key: str | None = None
    """Share non-overlap placement cache across instances.

    When set, all ``PoseResetTerm`` instances with the same key share cached
    placements so assets from separate terms can avoid overlap. ``None``
    disables grouping.
    """

    clear_cross_group_cache: bool = False
    """Clear the shared placement cache at the start of each reset.

    When enabled, each invocation starts from a fresh cache. Otherwise cached
    placement information is reused until it is cleared externally.
    """

    prefer_stack: bool = False
    """Stacking policy for drop mode.

    When enabled, every asset is placed above the current highest top surface.
    Otherwise assets are stacked only when XY overlap occurs.
    """
