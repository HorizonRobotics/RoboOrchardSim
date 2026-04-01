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

from collections.abc import Sequence

import robo_orchard_core.utils.math as math_utils
import torch
from pxr import Gf
from robo_orchard_core.envs.manager_based_env import ResetEvent
from robo_orchard_core.envs.managers.events.event_term import (
    EventTermBase,
    EventTermBaseCfg,
)
from robo_orchard_core.utils.config import Config
from typing_extensions import Literal

from robo_orchard_sim.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg as LabSceneEntityCfg,
)
from robo_orchard_sim.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.utils.config import ClassType_co

__all__ = ["LightResetTerm", "LightResetTermCfg", "LightPoseCfg"]


class LightResetTerm(
    EventTermBase[ResetEvent, IsaacEnvType_co, "LightResetTerm"],
):
    def __init__(self, cfg: "LightResetTermCfg", env: IsaacEnvType_co):
        super().__init__(cfg, env)
        self._cfg = cfg
        self._env = env
        self.generator = torch.Generator(device="cpu")

    def __call__(self, event_msg: ResetEvent):
        if hasattr(event_msg, "seed") and event_msg.seed is not None:
            self.generator.manual_seed(event_msg.seed)

        apply_crazy = (
            torch.rand(1, generator=self.generator).item()
            < self._cfg.crazy_randomization_rate
        )

        """Do the event term operation."""
        for asset_cfg in self._cfg.asset_cfgs:
            prim = self._get_prim(asset_cfg)

            if (
                self._cfg.randomize_color
                and self._cfg.color_temperature_range is not None
            ):
                self.__color_randomization_impl(prim, apply_crazy)

            if (
                self._cfg.randomize_intensity
                and self._cfg.intensity_range is not None
            ):
                self.__intensity_randomization_impl(prim, apply_crazy)

            if (
                self._cfg.randomize_position
                and self._cfg.position_cfg is not None
            ):
                self.__position_randomization_impl(asset_cfg)

    def reset(self, env_ids: Sequence[int] | None = None):
        # print("Reset the pose event term")
        # Do nothing for now
        pass

    def __color_randomization_impl(self, prim, apply_crazy: bool):
        color_temp = self._smart_random(
            self._cfg.color_temperature_range, 0.0, 10000.0
        )

        base_rgb = self.kelvin_to_rgb_torch(torch.tensor(color_temp))
        # Add Tint noise
        noise = (
            torch.rand_like(base_rgb) * self._cfg.rgb_noise * 2
        ) - self._cfg.rgb_noise

        final_rgb = torch.clamp(base_rgb + noise, 0.0, 1.0)

        random_color = Gf.Vec3f(
            final_rgb[0].item(), final_rgb[1].item(), final_rgb[2].item()
        )

        if apply_crazy:
            random_color = self._get_crazy_random_value(
                "color", self.generator
            )

        color_attr = prim.GetAttribute("inputs:color")
        if color_attr:
            color_attr.Set(random_color)
        else:
            print(
                f"Warning: Light prim at {prim.GetPath()} has no color "
                f"attribute."
            )

    def __intensity_randomization_impl(self, prim, apply_crazy: bool):
        random_intensity = self._smart_random(
            self._cfg.intensity_range, 0.0, 100000.0
        )

        if apply_crazy:
            random_intensity = self._get_crazy_random_value(
                "intensity", self.generator
            )

        intensity_attr = prim.GetAttribute("inputs:intensity")
        if intensity_attr:
            intensity_attr.Set(random_intensity)
        else:
            print(
                f"Warning: Light prim at {prim.GetPath()} has no intensity "
                f"attribute."
            )

    def __position_randomization_impl(self, asset_cfg):
        random_elevation = self._smart_random(
            self._cfg.position_cfg.elevation, 0.0, torch.pi / 2.0
        )
        random_azimuth = self._smart_random(
            self._cfg.position_cfg.azimuth, 0.0, 2.0 * torch.pi
        )

        pose = self._sample_spherical_poses(
            center_pose=torch.tensor(
                self._cfg.position_cfg.center_pose, device=self._env.device
            ),
            radius=self._cfg.position_cfg.radius,
            angle=random_elevation,
            azimuth=random_azimuth,
        )

        asset = self._env.scene[asset_cfg.name]

        positions = torch.tensor(pose[0:3], device=self._env.device).repeat(
            self._env.num_envs, 1
        )
        orientations = torch.tensor(pose[3:7], device=self._env.device).repeat(
            self._env.num_envs, 1
        )
        asset.write_root_state_to_sim(positions, orientations)

    def _smart_random(
        self, cfg: "RangeCfg", outer_min_limit: float, outer_max_limit: float
    ) -> float:
        min = cfg.range[0]
        max = cfg.range[1]

        if not cfg.inverse:
            val = min + torch.rand(1, generator=self.generator).item() * (
                max - min
            )
        else:
            pick_lower = torch.rand(1, generator=self.generator).item() < 0.5
            if pick_lower:
                # [outer_min_limit, min]
                val = outer_min_limit + torch.rand(
                    1, generator=self.generator
                ).item() * (min - outer_min_limit)
            else:
                # [max, outer_max_limit]
                val = max + torch.rand(1, generator=self.generator).item() * (
                    outer_max_limit - max
                )
        return val

    def _get_prim(self, asset_cfg):
        stage = self._env.sim.stage
        prim_path = self._env.scene[asset_cfg.name].cfg.prim_path
        prim = stage.GetPrimAtPath(prim_path)
        return prim

    def kelvin_to_rgb_torch(self, kelvin_tensor: torch.Tensor) -> torch.Tensor:
        temp = kelvin_tensor.float() / 100.0

        r = torch.zeros_like(temp)
        g = torch.zeros_like(temp)
        b = torch.zeros_like(temp)

        # =================Red =================
        mask_r_low = temp <= 66.0
        r[mask_r_low] = 255.0

        mask_r_high = ~mask_r_low
        r[mask_r_high] = 329.698727446 * torch.pow(
            torch.clamp(temp[mask_r_high] - 60.0, min=0.001), -0.1332047592
        )

        # ================= Green =================
        mask_g_low = temp <= 66.0
        g[mask_g_low] = (
            99.4708025861 * torch.log(torch.clamp(temp[mask_g_low], min=0.001))
            - 161.1195681661
        )

        mask_g_high = ~mask_g_low
        g[mask_g_high] = 288.1221695283 * torch.pow(
            torch.clamp(temp[mask_g_high] - 60.0, min=0.001), -0.0755148492
        )

        # ================= Blue =================
        # Case 1: temp >= 66, Blue = 255
        mask_b_high = temp >= 66.0
        b[mask_b_high] = 255.0

        # Case 2: temp <= 19, Blue = 0
        mask_b_low = temp <= 19.0
        b[mask_b_low] = 0.0

        # Case 3: 19 < temp < 66
        mask_b_mid = ~(mask_b_high | mask_b_low)
        b[mask_b_mid] = (
            138.5177312231
            * torch.log(torch.clamp(temp[mask_b_mid] - 10.0, min=0.001))
            - 305.0447927307
        )

        rgb = torch.stack([r, g, b], dim=-1)
        rgb = torch.clamp(rgb, 0.0, 255.0) / 255.0

        return rgb

    def _sample_spherical_poses(
        self,
        center_pose: torch.Tensor,
        radius: float,
        angle: float,
        azimuth: float,
    ):
        """Generate pose on a spherical ring.

        Args:
            center_pose: [3] (x, y, z) Center position of the sphere.
            radius: Radius of the sphere.
            angle: Elevation angle (radians), 0 indicates +Z direction
            (North Pole), pi/2 indicates Equator.
            azimuth: (0~2pi):

        Returns:
            pose: [1, 7] (x, y, z, w, x, y, z)
        """
        device = self._env.device

        _azimuth = torch.tensor([azimuth], device=device)

        z_offset = radius * torch.cos(torch.tensor(angle, device=device))
        xy_radius = radius * torch.sin(torch.tensor(angle, device=device))

        x = center_pose[0] + xy_radius * torch.cos(_azimuth)
        y = center_pose[1] + xy_radius * torch.sin(_azimuth)
        z = center_pose[2] + z_offset

        positions = torch.stack([x, y, torch.full_like(x, z)], dim=1)  # [N, 3]

        forward = center_pose - positions
        forward = forward / torch.norm(forward, dim=1, keepdim=True)

        # Assume World Up is +Z [0, 0, 1]
        world_up = torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(
            forward
        )

        # Right (local +X) = normalize(cross(forward, world_up))
        right = torch.cross(forward, world_up, dim=1)

        mask = torch.norm(right, dim=1) < 1e-6
        if mask.any():
            right[mask] = torch.tensor([1.0, 0.0, 0.0], device=device)

        right = right / torch.norm(right, dim=1, keepdim=True)
        up = torch.cross(right, forward, dim=1)

        rot_mat = torch.stack([right, up, -forward], dim=2)
        orientations = math_utils.matrix_to_quaternion(rot_mat)
        pose = torch.cat([positions, orientations], dim=1)[0]

        return pose

    def _get_crazy_random_value(
        self, mode: Literal["color", "intensity"], generator: torch.Generator
    ):
        crazy_colors = [
            Gf.Vec3f(1.0, 0.0, 0.0),  # Red
            Gf.Vec3f(0.0, 1.0, 0.0),  # Green
            Gf.Vec3f(0.0, 0.0, 1.0),  # Blue
            Gf.Vec3f(1.0, 1.0, 1.0),  # White
            Gf.Vec3f(0.0, 0.0, 0.0),  # White
        ]
        color_index = torch.randint(
            0, len(crazy_colors), (1,), generator=generator
        ).item()

        crazy_intensities = [0.0, 50000.0, 100000.0]
        intensity_index = torch.randint(
            0, len(crazy_intensities), (1,), generator=generator
        ).item()

        if mode == "color":
            return crazy_colors[color_index]
        elif mode == "intensity":
            return crazy_intensities[intensity_index]


class RangeCfg(Config):
    """Configuration for a range with optional inverse sampling."""

    range: tuple[float, float]
    """The [Min, Max] range for sampling."""

    inverse: bool = False
    """Whether to perform inverse sampling outside the range.
        Example:[~,Min] U [Max, ~]
    """


class LightPoseCfg(Config):
    """Configuration for light pose randomization."""

    center_pose: tuple[float, float, float]
    """Center position around which to sample poses."""

    radius: float
    """Radius of the spherical ring."""

    elevation: RangeCfg
    """Elevation angle (radians), 0 indicates +Z direction (North Pole),
        pi/2 indicates Equator.
    """

    azimuth: RangeCfg = RangeCfg(range=(0.0, 2.0 * torch.pi))
    """Azimuth angle range."""


class LightResetTermCfg(EventTermBaseCfg[LightResetTerm, LabSceneEntityCfg]):
    """The configuration of the pose event term."""

    class_type: ClassType_co[LightResetTerm] = LightResetTerm

    randomize_color: bool = False
    """Whether to randomize the color."""

    color_temperature_range: RangeCfg | None = None
    """"[Min, Max] color temperature in Kelvin."""

    rgb_noise: float = 0.1
    """The maximum RGB noise to add to the base color."""

    randomize_intensity: bool = False
    """Whether to randomize the intensity."""

    intensity_range: RangeCfg | None = None
    """[Min, Max] intensity value."""

    randomize_position: bool = False
    """Whether to randomize the position."""

    position_cfg: LightPoseCfg | None = None

    crazy_randomization_rate: float = 0.0
    """The probability of applying crazy randomization, [0, 1]"""
