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

"""Shared env/task definitions for pick task variants."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, cast

from robo_orchard_sim.task_suite.base import TaskDefinition
from robo_orchard_sim.task_suite.manipulation.semantic_pick import (
    action_plan,
)
from robo_orchard_sim.task_suite.registration import register_task

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
        AssetResolver,
    )
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
    from robo_orchard_sim.tasks.trajs_gen.base_executor import BaseExecutorCfg

_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _DIR / "configs"


class PickTaskDefinitionBase(TaskDefinition):
    """Shared definition logic for resolver-backed pick tasks."""

    @classmethod
    def build(
        cls,
        resolver: "AssetResolver | None" = None,
        config_path: str | None = None,
    ) -> "OrchardEnv":
        from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
        from robo_orchard_sim.orchard_env.tasks.pick_task import (
            PickAssets,
            PickTask,
            PickTaskParams,
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
        task_assets = PickAssets(**resolved)
        task_params = PickTaskParams(
            **cls.resolve_task_params(config_path=config_path)
        )

        return OrchardEnv(
            scene=cls.resolve_scene(config_path=config_path),
            embodiment=cls.resolve_embodiment(config_path=config_path),
            task=PickTask(
                assets=task_assets,
                params=task_params,
                instruction=cls.resolve_instruction(config_path=config_path),
            ),
        )

    @classmethod
    def build_atomic_action_plan(
        cls,
        orchard_env: "OrchardEnv",
    ) -> list["BaseExecutorCfg"]:
        """Build the default atomic action plan for semantic pick."""
        del cls
        return action_plan.build_task_atomic_action_plan(orchard_env)


def _make_pick_task_definition_class(
    *,
    class_name: str,
    namespace: str,
    yaml_name: str,
) -> type[PickTaskDefinitionBase]:
    task_cls = type(
        class_name,
        (PickTaskDefinitionBase,),
        {
            "__doc__": f"Task definition for the '{namespace}' pick variant.",
            "__module__": __name__,
            "namespace": namespace,
            "config_path": str(_CONFIG_DIR / yaml_name),
        },
    )
    return cast(type[PickTaskDefinitionBase], register_task(task_cls))


PickCategoryTaskDefinition = _make_pick_task_definition_class(
    class_name="PickCategoryTaskDefinition",
    namespace="pick_category",
    yaml_name="pick_category.yaml",
)
PickAttributeTaskDefinition = _make_pick_task_definition_class(
    class_name="PickAttributeTaskDefinition",
    namespace="pick_attribute",
    yaml_name="pick_attribute.yaml",
)
PickDisambiguationTaskDefinition = _make_pick_task_definition_class(
    class_name="PickDisambiguationTaskDefinition",
    namespace="pick_disambiguation",
    yaml_name="pick_disambiguation.yaml",
)

__all__ = [
    "PickTaskDefinitionBase",
    "PickCategoryTaskDefinition",
    "PickAttributeTaskDefinition",
    "PickDisambiguationTaskDefinition",
]
