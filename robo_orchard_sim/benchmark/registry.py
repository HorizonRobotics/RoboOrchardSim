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

"""Task-suite runtime registry helpers."""

from __future__ import annotations
from typing import TYPE_CHECKING

from robo_orchard_sim.benchmark.registration import (
    build_task as _build_task,
    build_task_atomic_action_plan as _build_task_atomic_action_plan,
)
from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
        AssetResolver,
    )
    from robo_orchard_sim.task_components.role_registry import RoleRegistry
    from robo_orchard_sim.task_components.trajs_gen.base_executor import (
        BaseExecutorCfg,
    )


def _bootstrap_task_definitions() -> None:
    """User should register task definitions in this function."""
    from robo_orchard_sim.benchmark.manipulation import (
        affordance as _affordance,
        close as _close,
        joint_direction as _joint_direction,
        open as _open,
        place_a2b as _place_a2b,
        semantic_pick as _pick,
        spatial_pick as _spatial_pick,
        spatial_relation as _spatial_relation,
        stack_cubes as _stack_cubes,
    )

    del (
        _affordance,
        _close,
        _joint_direction,
        _open,
        _pick,
        _place_a2b,
        _spatial_pick,
        _spatial_relation,
        _stack_cubes,
    )


def build_task(
    task_name: str,
    resolver: "AssetResolver | None" = None,
    config_path: str | None = None,
) -> OrchardEnv:
    """Build a fresh orchard task lazily from its registered name."""
    _bootstrap_task_definitions()
    return _build_task(task_name, resolver=resolver, config_path=config_path)


def build_task_atomic_action_plan(
    task_name: str,
    orchard_env: OrchardEnv,
    *,
    role_registry: RoleRegistry | None = None,
) -> list["BaseExecutorCfg"]:
    """Build a default atomic action plan lazily from task name."""
    _bootstrap_task_definitions()
    return _build_task_atomic_action_plan(
        task_name, orchard_env, role_registry=role_registry
    )
