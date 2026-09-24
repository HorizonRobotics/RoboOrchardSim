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

"""Resolver-backed definition for the close benchmark."""

from pathlib import Path
from typing import TYPE_CHECKING

from robo_orchard_sim.benchmark.base import TaskDefinition
from robo_orchard_sim.benchmark.registration import register_task

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
        AssetResolver,
    )
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv

_CONFIG_PATH = Path(__file__).resolve().parent / "configs/close.yaml"


@register_task
class CloseTaskDefinition(TaskDefinition):
    """Build the single-target close benchmark."""

    namespace = "close"
    config_path = str(_CONFIG_PATH)

    @classmethod
    def build(
        cls,
        resolver: "AssetResolver | None" = None,
        config_path: str | None = None,
    ) -> "OrchardEnv":
        """Build a close environment from resolved assets."""
        from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets
        from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
        from robo_orchard_sim.orchard_env.task_templates.articulated.close_task import (  # noqa: E501
            JointCloseTask,
        )
        from robo_orchard_sim.orchard_env.task_templates.articulated.joint_direction_task import (  # noqa: E501
            JointOperationTaskParams,
        )

        if resolver is None:
            raise ValueError(
                "CloseTaskDefinition.build() requires an AssetResolver."
            )
        asset_configs = cls.resolve_asset_configs(config_path=config_path)
        if asset_configs is None:
            raise ValueError(
                "CloseTaskDefinition needs asset_configs in "
                f"{config_path or cls.config_path}."
            )

        resolved = resolver.resolve(asset_configs)
        return OrchardEnv(
            scene=cls.resolve_scene(config_path=config_path),
            embodiment=cls.resolve_embodiment(config_path=config_path),
            task=JointCloseTask(
                assets=TaskAssets.from_resolved(resolved),
                params=JointOperationTaskParams(
                    **cls.resolve_task_params(config_path=config_path)
                ),
                instruction=cls.resolve_instruction(config_path=config_path),
            ),
        )


__all__ = ["CloseTaskDefinition"]
