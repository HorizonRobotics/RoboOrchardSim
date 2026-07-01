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

"""Reusable task parameter models."""

from __future__ import annotations
import copy
import math

from pydantic import Field, model_validator
from robo_orchard_core.utils.config import Config
from typing_extensions import Literal

from robo_orchard_sim.ext.envs.managers.events.light_reset import (
    LightPoseCfg,
    RangeCfg,
)
from robo_orchard_sim.ext.envs.managers.events.texture_reset import (
    TextureResetTermCfg,
)


def _deep_merge_dicts(
    defaults: dict[str, object], overrides: dict[str, object]
) -> dict[str, object]:
    """Recursively merge override values into defaults."""
    merged = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


class PoseRangeConfig(Config):
    """Pose sampling ranges shared by manipulation tasks."""

    x: tuple[float, float] = (0.25, 0.55)
    y: tuple[float, float] = (-0.35, 0.35)
    z: tuple[float, float] = (0.0, 0.0)
    roll: tuple[float, float] = (0.0, 0.0)
    pitch: tuple[float, float] = (0.0, 0.0)
    yaw: tuple[float, float] = (-math.pi, math.pi)


class TaskPoseResetConfig(Config):
    """Reusable task-level configuration for pose reset events."""

    mode: Literal[
        "random",
        "random_non_overlap",
        "orderly",
        "default",
        "drop",
    ] = "random_non_overlap"
    pose_range: PoseRangeConfig = PoseRangeConfig()
    min_separation: float = 0.03


class TaskDistantLightConfig(Config):
    """Task-level configuration for a generated distant light asset."""

    asset_name: str = "dis_light"
    prim_path: str = "/World/dis_light"
    init_pos: tuple[float, float, float] = (0.4, 0.0, 0.8)
    init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    color: tuple[float, float, float] = (0.3, 0.3, 0.3)
    intensity: float = 3000.0


_LIGHT_RESET_PRESETS: dict[str, dict[str, object]] = {
    "default_distant_light": {
        "asset_names": ["background/dis_light"],
        "distant_light": {
            "asset_name": "dis_light",
            "prim_path": "/World/dis_light",
            "init_pos": (0.4, 0.0, 0.8),
            "init_rot": (1.0, 0.0, 0.0, 0.0),
            "color": (0.3, 0.3, 0.3),
            "intensity": 3000.0,
        },
        "randomize_color": True,
        "color_temperature_range": {
            "range": (4200.0, 5800.0),
            "inverse": False,
        },
        "rgb_noise": 0.1,
        "randomize_intensity": True,
        "intensity_range": {
            "range": (1800.0, 3200.0),
            "inverse": False,
        },
        "randomize_position": True,
        "position_cfg": {
            "center_pose": (0.4, 0.0, 0.0),
            "radius": 0.5,
            "elevation": {
                "range": (0.1, 0.4),
                "inverse": False,
            },
        },
        "crazy_randomization_rate": 0.0,
    }
}

_TEXTURE_RESET_PRESETS: dict[str, dict[str, object]] = {
    "default_table_texture": {
        "asset_names": ["background/table"],
    }
}


class TaskLightResetConfig(Config):
    """Reusable task-level configuration for light reset events."""

    enabled: bool = False
    preset: str | None = None
    asset_names: list[str] = Field(default_factory=list)
    distant_light: TaskDistantLightConfig | None = None
    randomize_color: bool = False
    color_temperature_range: RangeCfg | None = None
    rgb_noise: float = 0.1
    randomize_intensity: bool = False
    intensity_range: RangeCfg | None = None
    randomize_position: bool = False
    position_cfg: LightPoseCfg | None = None
    crazy_randomization_rate: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def apply_preset(cls, value: object) -> object:
        """Expand a light reset preset before field parsing."""
        if not isinstance(value, dict):
            return value
        preset = value.get("preset")
        if preset is None:
            return value
        if preset not in _LIGHT_RESET_PRESETS:
            raise ValueError(f"Unknown light_reset preset: {preset}")
        return _deep_merge_dicts(_LIGHT_RESET_PRESETS[preset], value)

    @model_validator(mode="after")
    def validate_enabled_configuration(self) -> "TaskLightResetConfig":
        """Validate the cross-field contract for task-level light reset."""
        if not 0.0 <= self.crazy_randomization_rate <= 1.0:
            raise ValueError(
                "crazy_randomization_rate must be within [0.0, 1.0]."
            )
        if not self.enabled:
            return self
        if not self.asset_names:
            raise ValueError(
                "asset_names must not be empty when light_reset is enabled."
            )
        if self.distant_light is not None:
            expected_name = f"background/{self.distant_light.asset_name}"
            if expected_name not in self.asset_names:
                raise ValueError(
                    "asset_names must include the generated distant light "
                    f"scene name '{expected_name}'."
                )
        if self.randomize_color and self.color_temperature_range is None:
            raise ValueError(
                "color_temperature_range is required when "
                "randomize_color=True."
            )
        if self.randomize_intensity and self.intensity_range is None:
            raise ValueError(
                "intensity_range is required when randomize_intensity=True."
            )
        if self.randomize_position and self.position_cfg is None:
            raise ValueError(
                "position_cfg is required when randomize_position=True."
            )
        return self


class TaskTextureResetConfig(Config):
    """Reusable task-level configuration for texture reset events."""

    enabled: bool = False
    preset: str | None = None
    asset_names: list[str] = Field(default_factory=list)
    variant_set_name: str = TextureResetTermCfg.model_fields[
        "variant_set_name"
    ].default
    variant_sort: bool = TextureResetTermCfg.model_fields[
        "variant_sort"
    ].default
    variant_index_range: list[int] = Field(
        default_factory=lambda: list(
            TextureResetTermCfg.model_fields["variant_index_range"].default
        )
    )

    @model_validator(mode="before")
    @classmethod
    def apply_preset(cls, value: object) -> object:
        """Expand a texture reset preset before field parsing."""
        if not isinstance(value, dict):
            return value
        preset = value.get("preset")
        if preset is None:
            return value
        if preset not in _TEXTURE_RESET_PRESETS:
            raise ValueError(f"Unknown texture_reset preset: {preset}")
        return _deep_merge_dicts(_TEXTURE_RESET_PRESETS[preset], value)

    @model_validator(mode="after")
    def validate_enabled_configuration(self) -> "TaskTextureResetConfig":
        """Validate the cross-field contract for task-level texture reset."""
        if not self.enabled:
            return self
        if not self.asset_names:
            raise ValueError(
                "asset_names must not be empty when texture_reset is enabled."
            )
        if len(self.variant_index_range) != 2:
            raise ValueError(
                "variant_index_range must contain exactly two integers."
            )
        return self
