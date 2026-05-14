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

"""Spatial-pick task definitions (D3) driven by upstream layout JSON."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from robo_orchard_sim.task_suite.base import TaskDefinition
from robo_orchard_sim.task_suite.manipulation.spatial_pick import action_plan
from robo_orchard_sim.task_suite.registration import register_task

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
        AssetResolver,
    )
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
    from robo_orchard_sim.tasks.trajs_gen.base_executor import BaseExecutorCfg

logger = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _DIR / "configs"


class SpatialPickTaskDefinitionBase(TaskDefinition):
    """Layout-driven spatial-pick base."""

    LAYOUT_ROLE_MAP: ClassVar[dict[str, str]] = {
        "pick": "pick",
        "ref": "distractor_0",
    }

    @classmethod
    def build(
        cls,
        resolver: "AssetResolver | None" = None,
        config_path: str | None = None,
    ) -> "OrchardEnv":
        from robo_orchard_sim.orchard_env.layout.builder import LayoutBuilder
        from robo_orchard_sim.orchard_env.layout.loader import parse_layout
        from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
        from robo_orchard_sim.orchard_env.tasks.pick_task import (
            PickAssets,
            PickTask,
            PickTaskParams,
        )
        from robo_orchard_sim.task_suite.manipulation.semantic_pick.pick_env import (  # noqa: E501
            PickTaskDefinitionBase,
        )

        if resolver is None:
            raise ValueError(
                f"{cls.__name__}.build() requires an AssetResolver. "
                "Construct one from an AssetRegistry and pass it as "
                "resolver=..."
            )
        path = config_path if config_path is not None else cls.config_path
        if path is None:
            raise ValueError(
                f"{cls.__name__}.build() requires config_path (YAML)."
            )

        cfg = cls._load_config(config_path=path)
        if cfg.layout is None:
            raise ValueError(
                f"{cls.__name__} requires 'layout: <path>' in YAML at "
                f"{path!r}."
            )
        if cfg.asset_configs is not None:
            raise ValueError(
                f"{cls.__name__} rejects 'asset_configs' in YAML at "
                f"{path!r}; asset configs derive from the layout JSON."
            )
        if cfg.scene is not None and cfg.scene.num_envs != 1:
            raise ValueError(
                f"{cls.__name__} requires num_envs == 1; got "
                f"{cfg.scene.num_envs}"
            )
        if cfg.task is not None and "pose_reset" in cfg.task.params:
            logger.warning(
                "%s: pose_reset in YAML is ignored under layout mode "
                "(pose comes from the layout JSON).",
                cls.__name__,
            )
        task_params = PickTaskParams(
            **cls.resolve_task_params(config_path=path)
        )

        yaml_path = cls._resolve_config_path(path)
        seq = parse_layout((yaml_path.parent / cfg.layout).resolve())

        assets, layout_builder = LayoutBuilder.build(
            seq, resolver, cls.LAYOUT_ROLE_MAP
        )
        pick_assets = PickAssets(
            pick=assets["pick"],
            distractors=(
                [assets[slot] for slot in sorted(assets) if slot != "pick"]
                or None
            ),
        )
        scene = cls.resolve_scene(config_path=path)
        PickTaskDefinitionBase._apply_light_reset_scene_overrides(
            scene, task_params
        )

        return OrchardEnv(
            scene=scene,
            embodiment=cls.resolve_embodiment(config_path=path),
            task=PickTask(
                assets=pick_assets,
                params=task_params,
                instruction=cls.resolve_instruction(config_path=path),
            ),
            layout_builder=layout_builder,
        )

    @classmethod
    def build_atomic_action_plan(
        cls,
        orchard_env: "OrchardEnv",
    ) -> list["BaseExecutorCfg"]:
        """Default atomic plan: pick + lift."""
        del cls
        return action_plan.build_task_atomic_action_plan(orchard_env)


def _make_spatial_pick_task_definition_class(
    *,
    class_name: str,
    namespace: str,
    yaml_name: str,
) -> type[SpatialPickTaskDefinitionBase]:
    task_cls = type(
        class_name,
        (SpatialPickTaskDefinitionBase,),
        {
            "__doc__": (
                f"Task definition for the '{namespace}' spatial-pick variant."
            ),
            "__module__": __name__,
            "namespace": namespace,
            "config_path": str(_CONFIG_DIR / yaml_name),
        },
    )
    return cast(type[SpatialPickTaskDefinitionBase], register_task(task_cls))


SpatialPickEasyTaskDefinition = _make_spatial_pick_task_definition_class(
    class_name="SpatialPickEasyTaskDefinition",
    namespace="spatial_pick_easy",
    yaml_name="spatial_pick_easy.yaml",
)


__all__ = [
    "SpatialPickTaskDefinitionBase",
    "SpatialPickEasyTaskDefinition",
]
