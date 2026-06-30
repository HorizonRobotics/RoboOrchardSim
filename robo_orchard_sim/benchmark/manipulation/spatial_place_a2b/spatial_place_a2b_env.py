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

"""Layout-driven spatial place-a2b task definitions (upstream layout JSON)."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from robo_orchard_core.utils.logging import LoggerManager

from robo_orchard_sim.benchmark.base import TaskDefinition
from robo_orchard_sim.benchmark.manipulation.spatial_place_a2b.action_plan import (  # noqa: E501
    build_task_atomic_action_plan,
)
from robo_orchard_sim.benchmark.registration import register_task

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
        AssetResolver,
    )
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
    from robo_orchard_sim.task_components.trajs_gen.base_executor import (
        BaseExecutorCfg,
    )

logger = LoggerManager().get_child(__name__)

_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _DIR / "configs"


class SpatialPlaceA2BTaskDefinitionBase(TaskDefinition):
    """Layout-driven spatial place-a2b base."""

    NAMED_ROLES: ClassVar[dict[str, str]] = {"src": "pick", "dest": "place"}
    """Upstream JSON role -> task slot. Other roles auto-fill distractor_*."""

    @classmethod
    def build(
        cls,
        resolver: "AssetResolver | None" = None,
        config_path: str | None = None,
    ) -> "OrchardEnv":
        from robo_orchard_sim.orchard_env.layout.builder import LayoutBuilder
        from robo_orchard_sim.orchard_env.layout.loader import parse_layout
        from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
        from robo_orchard_sim.orchard_env.task_templates import (
            place_a2b_task as _place_a2b_task,
        )

        PlaceA2BTask = _place_a2b_task.PlaceA2BTask
        PlaceA2BTaskAssets = _place_a2b_task.PlaceA2BTaskAssets

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
        if cfg.scene is not None and cfg.scene.num_envs != 1:
            raise ValueError(
                f"{cls.__name__} requires num_envs == 1; got "
                f"{cfg.scene.num_envs}"
            )
        if cfg.task is not None and "pose_range" in cfg.task.params:
            logger.warning(
                "%s: pose_range in YAML is ignored under layout mode "
                "(pose comes from the layout JSON).",
                cls.__name__,
            )

        yaml_path = cls._resolve_config_path(path)
        seq = parse_layout((yaml_path.parent / cfg.layout).resolve())

        assets, layout_builder = LayoutBuilder.build(
            seq,
            resolver,
            cls.NAMED_ROLES,
            slot_filters=cfg.asset_configs,
        )
        place_assets = PlaceA2BTaskAssets(
            pick=assets["pick"],
            place=assets["place"],
            distractors=(
                [
                    assets[slot]
                    for slot in sorted(assets)
                    if slot not in ("pick", "place")
                ]
                or None
            ),
        )

        return OrchardEnv(
            scene=cls.resolve_scene(config_path=path),
            embodiment=cls.resolve_embodiment(config_path=path),
            task=PlaceA2BTask(
                assets=place_assets,
                params=None,
                instruction=cls.resolve_instruction(config_path=path),
            ),
            layout_builder=layout_builder,
        )

    @classmethod
    def build_atomic_action_plan(
        cls,
        orchard_env: "OrchardEnv",
    ) -> list["BaseExecutorCfg"]:
        """Default atomic plan: pick, lift, place."""
        del cls
        return build_task_atomic_action_plan(orchard_env)


def _make_spatial_place_a2b_task_definition_class(
    *,
    class_name: str,
    namespace: str,
    yaml_name: str,
) -> type[SpatialPlaceA2BTaskDefinitionBase]:
    task_cls = type(
        class_name,
        (SpatialPlaceA2BTaskDefinitionBase,),
        {
            "__doc__": (
                f"Task definition for the '{namespace}' spatial place-a2b "
                "variant."
            ),
            "__module__": __name__,
            "namespace": namespace,
            "config_path": str(_CONFIG_DIR / yaml_name),
        },
    )
    return cast(
        type[SpatialPlaceA2BTaskDefinitionBase], register_task(task_cls)
    )


SpatialPlaceA2BEasyTaskDefinition = (
    _make_spatial_place_a2b_task_definition_class(
        class_name="SpatialPlaceA2BEasyTaskDefinition",
        namespace="spatial_place_a2b_easy",
        yaml_name="spatial_place_a2b_easy.yaml",
    )
)


__all__ = [
    "SpatialPlaceA2BTaskDefinitionBase",
    "SpatialPlaceA2BEasyTaskDefinition",
]
