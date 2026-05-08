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

"""Pool wrapper for per-episode asset swap."""

from __future__ import annotations

from pydantic import ValidationInfo, field_validator
from robo_orchard_core.utils.config import Config

from robo_orchard_sim.orchard_env.assets.object_spec import ObjectSpec


class PoolSpec(Config):
    """Wraps N candidate ObjectSpecs with a stable role_id as scene_name.

    Attributes:
        role_id (str): Logical role name used as the scene_name alias.
        members (list[ObjectSpec]): All candidate specs in the pool.
        active_count (int, optional): Number of members activated per
            episode. Must be >= 1 and <= len(members). Default is 1.
    """

    role_id: str
    members: list[ObjectSpec]
    active_count: int = 1

    @field_validator("members")
    @classmethod
    def _at_least_two(cls, value: list[ObjectSpec]) -> list[ObjectSpec]:
        if len(value) < 2:
            raise ValueError(
                "PoolSpec requires at least 2 members; a single member "
                "should be an ObjectSpec, not a PoolSpec."
            )
        return value

    @field_validator("active_count")
    @classmethod
    def _active_count_in_range(cls, value: int, info: ValidationInfo) -> int:
        if value < 1:
            raise ValueError(f"active_count must be >= 1, got {value}")
        members = info.data.get("members", [])
        if members and value > len(members):
            raise ValueError(
                f"active_count ({value}) cannot exceed pool size "
                f"({len(members)})"
            )
        return value

    @property
    def scene_name(self) -> str:
        return self.role_id

    @property
    def member_scene_names(self) -> list[str]:
        return [m.scene_name for m in self.members]
