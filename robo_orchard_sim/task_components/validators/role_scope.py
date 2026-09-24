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

"""Stable validator domains derived from declarative task roles."""

from __future__ import annotations
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from robo_orchard_sim.task_components.validators.physical_entity import (
    SceneEntityKey,
)

if TYPE_CHECKING:
    from robo_orchard_sim.orchard_env.task_templates.task_base import TaskBase
    from robo_orchard_sim.task_components.role_registry import TargetRef

EntityId = SceneEntityKey


def _stable_unique(values: Iterable[EntityId]) -> tuple[EntityId, ...]:
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True, slots=True)
class RoleScope:
    """Scene entities grouped by declarative task role."""

    entities: tuple[EntityId, ...]
    by_role: Mapping[str, tuple[EntityId, ...]]

    @classmethod
    def from_task(cls, task: "TaskBase") -> "RoleScope":
        """Build the widest valid role domain declared by one task."""
        return cls.from_role_members(
            {
                role_id: task.get_role_scene_entities(role_id, swap=True)
                for role_id in task.roles
            }
        )

    @classmethod
    def from_role_targets(
        cls,
        by_role: Mapping[str, Iterable["TargetRef"]],
    ) -> "RoleScope":
        """Build a scope from role-grouped target references."""
        return cls.from_role_members(
            {
                role_id: (target.scene_name for target in targets)
                for role_id, targets in by_role.items()
            }
        )

    @classmethod
    def from_role_members(
        cls,
        by_role: Mapping[str, Iterable[EntityId]],
    ) -> "RoleScope":
        """Build a stable scope from role-ordered entity memberships."""
        normalized = {
            role_id: _stable_unique(members)
            for role_id, members in by_role.items()
        }
        empty_roles = sorted(
            role_id for role_id, members in normalized.items() if not members
        )
        if empty_roles:
            raise ValueError(f"Task roles have no candidates: {empty_roles}.")
        return cls(
            entities=_stable_unique(
                entity_id
                for members in normalized.values()
                for entity_id in members
            ),
            by_role=MappingProxyType(normalized),
        )

    @classmethod
    def from_entities(cls, entities: Iterable[EntityId]) -> "RoleScope":
        """Build a role-free scope for low-level checker composition."""
        return cls(
            entities=_stable_unique(entities),
            by_role=MappingProxyType({}),
        )

    def entities_for_role(self, role_id: str) -> tuple[EntityId, ...]:
        """Return the stable entity domain for one role."""
        return self.by_role.get(role_id, ())

    def relation_pairs(
        self,
        subject_role: str,
        object_role: str,
        *,
        exclude_self: bool = True,
    ) -> tuple[tuple[EntityId, EntityId], ...]:
        """Return the role-filtered binary relation domain."""
        return tuple(
            (subject, object_)
            for subject in self.entities_for_role(subject_role)
            for object_ in self.entities_for_role(object_role)
            if not exclude_self or subject != object_
        )
