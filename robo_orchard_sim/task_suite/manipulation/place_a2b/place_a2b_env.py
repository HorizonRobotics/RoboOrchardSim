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

"""Shared env/task definitions for place-a2b task variants."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

from robo_orchard_sim.task_suite.base import TaskDefinition
from robo_orchard_sim.task_suite.manipulation.place_a2b.action_plan import (
    build_task_atomic_action_plan,
)
from robo_orchard_sim.task_suite.registration import register_task
from robo_orchard_sim.tasks.trajs_gen.executors.pick import PickExecutorCfg

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
        AssetResolver,
    )
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv

_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _DIR / "configs"


class PlaceA2BTaskDefinitionBase(TaskDefinition):
    """Shared definition logic for resolver-backed place-a2b tasks."""

    @classmethod
    def build(
        cls,
        resolver: "AssetResolver | None" = None,
        config_path: str | None = None,
    ) -> "OrchardEnv":
        from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
        from robo_orchard_sim.orchard_env.tasks.place_a2b_task import (
            PlaceA2BTask,
            PlaceA2BTaskAssets,
            PlaceA2BTaskParams,
        )

        if resolver is None:
            raise ValueError(
                f"{cls.__name__}.build() requires an AssetResolver. "
                "Construct one from an AssetRegistry and pass it as "
                "resolver=..."
            )
        asset_configs = cls.resolve_asset_configs(config_path=config_path)
        if asset_configs is None:
            raise ValueError(
                f"{cls.__name__} needs asset_configs in the task YAML "
                f"at {config_path or cls.config_path}."
            )

        resolved = resolver.resolve(asset_configs)
        task_assets = PlaceA2BTaskAssets(**resolved)
        task_params = PlaceA2BTaskParams(
            **cls.resolve_task_params(config_path=config_path)
        )

        return OrchardEnv(
            scene=cls.resolve_scene(config_path=config_path),
            embodiment=cls.resolve_embodiment(config_path=config_path),
            task=PlaceA2BTask(
                assets=task_assets,
                params=task_params,
                instruction=cls.resolve_instruction(config_path=config_path),
            ),
        )

    @classmethod
    def build_atomic_action_plan(
        cls,
        orchard_env: "OrchardEnv",
    ) -> list[PickExecutorCfg]:
        """Build the default atomic action plan for place-a2b."""
        del cls
        return build_task_atomic_action_plan(orchard_env)


def _make_place_a2b_task_definition_class(
    *,
    class_name: str,
    namespace: str,
    yaml_name: str,
) -> type[PlaceA2BTaskDefinitionBase]:
    task_cls = type(
        class_name,
        (PlaceA2BTaskDefinitionBase,),
        {
            "__doc__": (
                f"Task definition for the '{namespace}' place-a2b variant."
            ),
            "__module__": __name__,
            "namespace": namespace,
            "config_path": str(_CONFIG_DIR / yaml_name),
        },
    )
    return register_task(task_cls)


PlaceA2BEasyTaskDefinition = _make_place_a2b_task_definition_class(
    class_name="PlaceA2BEasyTaskDefinition",
    namespace="place_a2b_easy",
    yaml_name="place_a2b_easy.yaml",
)
PlaceA2BHardTaskDefinition = _make_place_a2b_task_definition_class(
    class_name="PlaceA2BHardTaskDefinition",
    namespace="place_a2b_hard",
    yaml_name="place_a2b_hard.yaml",
)

__all__ = [
    "PlaceA2BTaskDefinitionBase",
    "PlaceA2BEasyTaskDefinition",
    "PlaceA2BHardTaskDefinition",
]
