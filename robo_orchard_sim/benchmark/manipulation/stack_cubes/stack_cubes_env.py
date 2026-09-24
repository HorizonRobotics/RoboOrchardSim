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

"""Resolver-backed definition for instruction-ordered cube stacking."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

from robo_orchard_sim.benchmark.base import TaskDefinition
from robo_orchard_sim.benchmark.manipulation.stack_cubes.colors import (
    CUBE_COLOR_NAMES,
    CUBE_COLOR_PALETTE,
)
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

_CONFIG_PATH = Path(__file__).resolve().parent / "configs/stack_cubes.yaml"
_MEMBERS_ROLE = "members"


@register_task
class StackCubesTaskDefinition(TaskDefinition):
    """Build a seeded four-color instruction-ordered stacking benchmark."""

    namespace = "stack_cubes"
    config_path = str(_CONFIG_PATH)

    @classmethod
    def build(
        cls,
        resolver: "AssetResolver | None" = None,
        config_path: str | None = None,
    ) -> "OrchardEnv":
        """Build a stack-cubes environment from one shared cube asset."""
        from robo_orchard_sim.benchmark.manipulation.semantic_pick import (
            pick_env,
        )
        from robo_orchard_sim.orchard_env.assets import (
            ObjectSpec,
            RigidObjectSpec,
        )
        from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets
        from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
        from robo_orchard_sim.orchard_env.task_templates import (
            stack_cubes_task,
        )

        if resolver is None:
            raise ValueError(
                "StackCubesTaskDefinition.build() requires an AssetResolver."
            )
        asset_configs = cls.resolve_asset_configs(config_path=config_path)
        if asset_configs is None:
            raise ValueError(
                "StackCubesTaskDefinition needs asset_configs in "
                f"{config_path or cls.config_path}."
            )
        config_dir = cls._config_dir(config_path=config_path)
        if config_dir is None:
            raise ValueError("Stack-cubes config directory is unavailable.")
        for entry in asset_configs.values():
            interaction_path = Path(entry["interaction_path"])
            if not interaction_path.is_absolute():
                entry["interaction_path"] = str(
                    (config_dir / interaction_path).resolve()
                )
        resolved = resolver.resolve(asset_configs)
        if set(resolved) != {_MEMBERS_ROLE}:
            raise ValueError(
                "Stack-cubes asset config must contain exactly one "
                f"{_MEMBERS_ROLE!r} prototype."
            )
        prototype = resolved[_MEMBERS_ROLE]
        if isinstance(prototype, list):
            raise ValueError(
                "The stack-cubes prototype must resolve to exactly one asset."
            )
        if not isinstance(prototype, RigidObjectSpec):
            raise TypeError(
                "Stack-cubes prototype must resolve to a rigid object, got "
                f"{type(prototype).__name__}."
            )
        params_payload = cls.resolve_task_params(config_path=config_path)
        task_params = stack_cubes_task.StackCubesTaskParams(**params_payload)
        cube_size = task_params.cube_size
        colors = resolver.sample_without_replacement(
            CUBE_COLOR_NAMES,
            count=stack_cubes_task.STACK_SIZE,
        )
        members: list[RigidObjectSpec] = []
        for color in colors:
            attributes = dict(prototype.attributes)
            attributes["color"] = (color,)
            members.append(
                prototype.model_copy(
                    update={
                        "name": f"{color}_cube",
                        "actor_type": color,
                        "attributes": attributes,
                        "visual_color": CUBE_COLOR_PALETTE[color],
                        "aabb_z_min": -cube_size / 2.0,
                    }
                )
            )
        task_members: list[ObjectSpec] = [*members]
        scene = cls.resolve_scene(config_path=config_path)
        pick_env.PickTaskDefinitionBase._apply_light_reset_scene_overrides(
            scene,
            task_params,
        )

        return OrchardEnv(
            scene=scene,
            embodiment=cls.resolve_embodiment(config_path=config_path),
            task=stack_cubes_task.StackCubesTask(
                assets=TaskAssets(
                    role_candidates={
                        _MEMBERS_ROLE: task_members,
                    }
                ),
                params=task_params,
                instruction=cls.resolve_instruction(config_path=config_path),
            ),
        )

    @classmethod
    def build_atomic_action_plan(
        cls,
        orchard_env: "OrchardEnv",
        *,
        role_registry: "RoleRegistry | None" = None,
    ) -> list["BaseExecutorCfg"]:
        """Build the default atomic action plan for stack-cubes."""
        from robo_orchard_sim.benchmark.manipulation.stack_cubes import (
            action_plan,
        )

        return action_plan.build_task_atomic_action_plan(
            orchard_env,
            role_registry=role_registry,
        )


__all__ = ["StackCubesTaskDefinition"]
