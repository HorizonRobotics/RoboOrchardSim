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

"""PlaceA2B task definition.

Assembles ``PlaneTableScene + DualArmPiperEmbodiment + PlaceA2BTask``
into an ``OrchardEnv`` for the place-A-to-B evaluation task.

Scene and embodiment are configured via ``place_a2b.yaml`` next to
this file. Edit that YAML to switch robot or scene without touching
Python code.

Task assets are always sampled from an ``AssetRegistry`` via an
``AssetResolver`` — the caller must supply a resolver. The default
``asset_configs`` block in ``place_a2b.yaml`` is used when none is
passed explicitly to ``build()``.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robo_orchard_sim.task_suite.base import TaskDefinition
from robo_orchard_sim.task_suite.registration import register_task

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
        AssetResolver,
    )
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv

_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = str(_DIR / "place_a2b.yaml")


@register_task
class PlaceA2BTaskDefinition(TaskDefinition):
    namespace: str = "place_a2b"
    config_path: str = _CONFIG_PATH

    @classmethod
    def build(
        cls,
        resolver: AssetResolver | None = None,
        asset_configs: dict[str, Any] | None = None,
    ) -> OrchardEnv:
        from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
        from robo_orchard_sim.orchard_env.tasks.place_a2b_task import (
            PlaceA2BTask,
            PlaceA2BTaskAssets,
        )

        if resolver is None:
            raise ValueError(
                f"{cls.__name__}.build() requires an AssetResolver. "
                "Construct one from an AssetRegistry and pass it as "
                "resolver=..."
            )
        if asset_configs is None:
            asset_configs = cls.resolve_asset_configs()
        if asset_configs is None:
            raise ValueError(
                f"{cls.__name__} needs asset_configs either from the "
                f"YAML at {cls.config_path} or passed explicitly via "
                "asset_configs=..."
            )

        resolved = resolver.resolve(asset_configs)
        task_assets = PlaceA2BTaskAssets(**resolved)

        return OrchardEnv(
            scene=cls.resolve_scene(),
            embodiment=cls.resolve_embodiment(),
            task=PlaceA2BTask(assets=task_assets),
        )


PlaceA2BEnv = PlaceA2BTaskDefinition
