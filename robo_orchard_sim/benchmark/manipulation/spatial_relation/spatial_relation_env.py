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

"""Env/task definitions for spatial-relation pick variants.

The variants differ only in what their YAML says: one names its targets
by direction, the other by distance. Both are built the same way here.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, cast

from robo_orchard_sim.benchmark.base import TaskDefinition
from robo_orchard_sim.benchmark.registration import register_task

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
        AssetResolver,
    )
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
    from robo_orchard_sim.task_components.role_registry import RoleRegistry
    from robo_orchard_sim.task_components.trajs_gen.base_executor import (
        BaseExecutorCfg,
    )

_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _DIR / "configs"


class SpatialRelationTaskDefinitionBase(TaskDefinition):
    """Shared definition logic for resolver-backed spatial-relation tasks.

    Assets come from the task YAML rather than a layout file: candidates
    are told apart by where each is placed around the reference, and
    those placements are redrawn every episode, so fixed coordinates
    would have nothing to contribute.
    """

    @classmethod
    def build(
        cls,
        resolver: "AssetResolver | None" = None,
        config_path: str | None = None,
    ) -> "OrchardEnv":
        from robo_orchard_sim.benchmark.manipulation.semantic_pick.pick_env import (  # noqa: E501
            PickTaskDefinitionBase,
        )
        from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets
        from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
        from robo_orchard_sim.orchard_env.task_templates.spatial_task import (
            SpatialTask,
            SpatialTaskParams,
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
        # Name each object after the key that declared it, so the scene
        # names, the placements and the YAML all read the same. Only
        # filled in where the YAML stayed quiet, leaving an explicit
        # prim_name to win.
        for key, entry in asset_configs.items():
            if "anchor" not in entry:
                entry.setdefault("prim_name", key)

        resolved = resolver.resolve(asset_configs)
        task_assets = TaskAssets.from_resolved(resolved)
        task_params = SpatialTaskParams(
            **cls.resolve_task_params(config_path=config_path)
        )
        scene = cls.resolve_scene(config_path=config_path)
        PickTaskDefinitionBase._apply_light_reset_scene_overrides(
            scene, task_params
        )

        return OrchardEnv(
            scene=scene,
            embodiment=cls.resolve_embodiment(config_path=config_path),
            task=SpatialTask(
                assets=task_assets,
                params=task_params,
                instruction=cls.resolve_instruction(config_path=config_path),
            ),
        )

    @classmethod
    def build_atomic_action_plan(
        cls,
        orchard_env: "OrchardEnv",
        *,
        role_registry: RoleRegistry | None = None,
    ) -> list[BaseExecutorCfg]:
        """Build actions for the episode's bound spatial target."""
        from robo_orchard_sim.benchmark.manipulation.spatial_relation import (
            action_plan,
        )

        return action_plan.build_task_atomic_action_plan(
            orchard_env, role_registry=role_registry
        )


def _make_spatial_relation_task_definition_class(
    *,
    class_name: str,
    namespace: str,
    yaml_name: str,
) -> type[SpatialRelationTaskDefinitionBase]:
    task_cls = type(
        class_name,
        (SpatialRelationTaskDefinitionBase,),
        {
            "__doc__": (
                f"Task definition for the '{namespace}' spatial-relation "
                "variant."
            ),
            "__module__": __name__,
            "namespace": namespace,
            "config_path": str(_CONFIG_DIR / yaml_name),
        },
    )
    return cast(
        type[SpatialRelationTaskDefinitionBase], register_task(task_cls)
    )


SpatialDirectionsTaskDefinition = _make_spatial_relation_task_definition_class(
    class_name="SpatialDirectionsTaskDefinition",
    namespace="spatial_directions",
    yaml_name="spatial_directions.yaml",
)

SpatialNearFarTaskDefinition = _make_spatial_relation_task_definition_class(
    class_name="SpatialNearFarTaskDefinition",
    namespace="spatial_near_far",
    yaml_name="spatial_near_far.yaml",
)


__all__ = [
    "SpatialRelationTaskDefinitionBase",
    "SpatialDirectionsTaskDefinition",
    "SpatialNearFarTaskDefinition",
]
