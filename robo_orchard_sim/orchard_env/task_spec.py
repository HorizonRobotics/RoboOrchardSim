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

"""Static declaration of the roles a task expects to be filled."""

from __future__ import annotations
from dataclasses import dataclass

from robo_orchard_sim.task_components.role_registry import Cardinality


@dataclass(frozen=True)
class RoleSpec:
    """What a task expects of one role, declared on the task class.

    Read at build time and at episode boundaries, never mutated: the
    concrete targets a role points at live in the RoleRegistry, not here.

    Declaring roles this way is what lets the evaluator bind targets by
    iterating ``task.roles`` instead of hard-coding role names, so one
    loop serves every task type.
    """

    description: str
    cardinality: Cardinality = "one"
    required_traits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("description must be non-empty.")
        if self.cardinality not in ("one", "many"):
            raise ValueError(
                f"Unknown cardinality '{self.cardinality}'; "
                "expected 'one' or 'many'."
            )
