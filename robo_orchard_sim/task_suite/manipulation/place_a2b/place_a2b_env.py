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

"""PlaceA2B task definition.

Assembles ``PlaneTableScene + DualArmPiperEmbodiment + PlaceA2BTask``
into an ``OrchardEnv`` for the place-A-to-B evaluation task.

Scene and embodiment are configured via ``place_a2b.yaml`` next to
this file.  Edit that YAML to switch robot or scene without touching
Python code.
"""

from __future__ import annotations
from pathlib import Path

from robo_orchard_sim.models.assets.asset_cfg import ORCHARD_ASSET
from robo_orchard_sim.task_suite.base import TaskDefinition
from robo_orchard_sim.task_suite.registration import register_task

_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = str(_DIR / "place_a2b.yaml")


@register_task
class PlaceA2BTaskDefinition(TaskDefinition):
    namespace: str = "place_a2b"
    config_path: str = _CONFIG_PATH

    @classmethod
    def build(cls):
        from robo_orchard_sim.orchard_env.assets import RigidObjectSpec
        from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
        from robo_orchard_sim.orchard_env.tasks.place_a2b_task import (
            PlaceA2BTask,
            PlaceA2BTaskAssets,
        )

        # TODO: use asset spec instead of hardcoded paths.
        PICK_USD_PATH = f"{ORCHARD_ASSET}/PUBLIC_OBJECTS/evaluation_assets/fruits/lemon_001/lemon_001.usd"  # noqa: E501
        PICK_INTERACTION_PATH = f"{ORCHARD_ASSET}/PUBLIC_OBJECTS/evaluation_assets/fruits/lemon_001/interaction.json"  # noqa: E501

        PLACE_USD_PATH = f"{ORCHARD_ASSET}/PUBLIC_OBJECTS/evaluation_assets/containers/plate_001/plate_001.usd"  # noqa: E501
        PLACE_INTERACTION_PATH = f"{ORCHARD_ASSET}/PUBLIC_OBJECTS/evaluation_assets/containers/plate_001/interaction.json"  # noqa: E501

        DISTRACTOR_USD_PATH = f"{ORCHARD_ASSET}/PUBLIC_OBJECTS/evaluation_assets/toys/squirrel_001/squirrel_001.usd"  # noqa: E501
        DISTRACTOR_INTERACTION_PATH = f"{ORCHARD_ASSET}/PUBLIC_OBJECTS/evaluation_assets/toys/squirrel_001/interaction.json"  # noqa: E501

        pick_asset = RigidObjectSpec(
            name="pick_object",
            usd_path=PICK_USD_PATH,
            interaction_path=PICK_INTERACTION_PATH,
        )
        place_asset = RigidObjectSpec(
            name="place_object",
            usd_path=PLACE_USD_PATH,
            interaction_path=PLACE_INTERACTION_PATH,
        )
        distractor_asset = RigidObjectSpec(
            name="distractor_object",
            usd_path=DISTRACTOR_USD_PATH,
            interaction_path=DISTRACTOR_INTERACTION_PATH,
        )

        return OrchardEnv(
            scene=cls.resolve_scene(),
            embodiment=cls.resolve_embodiment(),
            task=PlaceA2BTask(
                assets=PlaceA2BTaskAssets(
                    pick=pick_asset,
                    place=place_asset,
                    distractors=distractor_asset,
                )
            ),
        )


PlaceA2BEnv = PlaceA2BTaskDefinition
