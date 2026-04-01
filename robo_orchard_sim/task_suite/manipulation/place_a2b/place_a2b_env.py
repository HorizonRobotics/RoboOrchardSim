# ruff: noqa: E402
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

"""Example showing how to assemble a placeA2B ``OrchardEnv``.

This script demonstrates the user-facing `orchard_env` API:
1) define pick/place/table assets
2) assemble ``PlaneTableScene + DualArmPiperEmbodiment + PlaceA2BTask``
3) return an ``OrchardEnv``
4) optionally convert it to ``IsaacManagerBasedEnvCfg`` and run a reset
"""

from __future__ import annotations

from robo_orchard_sim.models.assets.asset_cfg import ORCHARD_ASSET
from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
from robo_orchard_sim.task_suite.base import TaskDefinition
from robo_orchard_sim.task_suite.registration import register_task

DEFAULT_PLACE_A2B_ENV_KWARGS = {
    "num_envs": 1,
    "env_spacing": 2.5,
    "physics_fps": 600,
    "render_fps": 30,
    "step_fps": 30,
}


@register_task
class PlaceA2BTaskDefinition(TaskDefinition):
    namespace: str = "place_a2b"
    default_env_kwargs = DEFAULT_PLACE_A2B_ENV_KWARGS

    @staticmethod
    def get_env(
        num_envs: int,
        env_spacing: float,
        physics_fps: int,
        render_fps: int,
        step_fps: int,
    ) -> OrchardEnv:
        from robo_orchard_sim.orchard_env.assets import RigidObjectSpec
        from robo_orchard_sim.orchard_env.embodiments.dualarm_piper import (
            DualArmPiperEmbodiment,
        )
        from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
        from robo_orchard_sim.orchard_env.scene.plane_table_scene import (
            PlaneTableScene,
        )
        from robo_orchard_sim.orchard_env.tasks.place_a2b_task import (
            PlaceA2BRole,
            PlaceA2BTask,
        )

        PICK_USD_PATH = f"{ORCHARD_ASSET}/PUBLIC_OBJECTS/evaluation_assets/fruits/lemon_001/lemon_001.usd"  # noqa: E501
        PICK_INTERACTION_PATH = f"{ORCHARD_ASSET}/PUBLIC_OBJECTS/evaluation_assets/fruits/lemon_001/interaction.json"  # noqa: E501

        PLACE_USD_PATH = f"{ORCHARD_ASSET}/PUBLIC_OBJECTS/evaluation_assets/containers/plate_001/plate_001.usd"  # noqa: E501
        PLACE_INTERACTION_PATH = f"{ORCHARD_ASSET}/PUBLIC_OBJECTS/evaluation_assets/containers/plate_001/interaction.json"  # noqa: E501

        pick_asset = RigidObjectSpec(
            name="pick_object",
            usd_path=PICK_USD_PATH,
            interaction_path=PICK_INTERACTION_PATH,
            mass=0.05,
            initial_pos=(0.35, 0.30, 0.02),
        )
        place_asset = RigidObjectSpec(
            name="place_object",
            usd_path=PLACE_USD_PATH,
            interaction_path=PLACE_INTERACTION_PATH,
            mass=100.0,
            initial_pos=(0.35, 0.0, 0.000036),
        )

        scene = PlaneTableScene(
            num_envs=num_envs,
            env_spacing=env_spacing,
            physics_fps=physics_fps,
            render_fps=render_fps,
            step_fps=step_fps,
            assets=None,
        )
        embodiment = DualArmPiperEmbodiment(
            namespace="robots",
            name="dualarm_piper",
            initial_pos=(0.0, 0.3, 0.0),
        )
        task = PlaceA2BTask(
            assets={
                PlaceA2BRole.PICK: pick_asset,
                PlaceA2BRole.PLACE: place_asset,
            }
        )
        return OrchardEnv(
            scene=scene,
            embodiment=embodiment,
            task=task,
        )

    @classmethod
    def build(cls) -> OrchardEnv:
        """Build the default place_a2b orchard env used by evaluator."""
        return cls.get_env(**cls.default_env_kwargs)


PlaceA2BEnv = PlaceA2BTaskDefinition
