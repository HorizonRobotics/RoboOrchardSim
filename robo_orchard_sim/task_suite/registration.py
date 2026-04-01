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

"""Shared task-definition registration helpers."""

from __future__ import annotations

from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
from robo_orchard_sim.task_suite.base import TaskDefinition

_TASK_REGISTRY: dict[str, type[TaskDefinition]] = {}


def register_task(
    task_definition: type[TaskDefinition],
) -> type[TaskDefinition]:
    """Register one task definition class under its namespace."""
    namespace = task_definition.namespace
    registered = _TASK_REGISTRY.get(namespace)
    if registered is task_definition:
        return task_definition
    if registered is not None:
        raise ValueError(f"Duplicate task name registered: {namespace!r}.")
    _TASK_REGISTRY[namespace] = task_definition
    return task_definition


def build_task(task_name: str) -> OrchardEnv:
    """Build a fresh orchard task lazily from its registered name."""
    try:
        task_definition = _TASK_REGISTRY[task_name]
    except KeyError as exc:
        known_tasks = ", ".join(sorted(_TASK_REGISTRY))
        raise KeyError(
            f"Unknown task name {task_name!r}. Known tasks: {known_tasks}."
        ) from exc
    return task_definition.build()
