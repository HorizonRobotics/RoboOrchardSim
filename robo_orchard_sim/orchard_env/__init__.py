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

"""Composable environment building blocks for orchard env generation."""

from typing import Any

__all__ = [
    "ArticulationSpec",
    "AssetSpec",
    "CustomAssetSpec",
    "ObjectSpec",
    "OrchardEnv",
    "RigidObjectSpec",
]


def __getattr__(name: str) -> Any:
    """Lazily expose top-level orchard env symbols."""
    if name == "OrchardEnv":
        from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv

        return OrchardEnv
    if name in {
        "ArticulationSpec",
        "AssetSpec",
        "CustomAssetSpec",
        "ObjectSpec",
        "RigidObjectSpec",
    }:
        from robo_orchard_sim.orchard_env.assets import (
            ArticulationSpec,
            AssetSpec,
            CustomAssetSpec,
            ObjectSpec,
            RigidObjectSpec,
        )

        return {
            "ArticulationSpec": ArticulationSpec,
            "AssetSpec": AssetSpec,
            "CustomAssetSpec": CustomAssetSpec,
            "ObjectSpec": ObjectSpec,
            "RigidObjectSpec": RigidObjectSpec,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
