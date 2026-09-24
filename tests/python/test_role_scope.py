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

"""Behavioral tests for role-based validator domains."""

from robo_orchard_sim.task_components.role_registry import TargetRef
from robo_orchard_sim.task_components.validators.role_scope import (
    RoleScope,
)


class _Task:
    roles = {"pick": object(), "place": object()}

    def get_role_candidates(
        self,
        role_id: str,
        *,
        swap: bool = False,
    ) -> list[TargetRef]:
        if role_id == "pick":
            names = ("apple", "banana") if swap else ("apple",)
        else:
            names = ("basket", "tray") if swap else ("basket",)
        return [TargetRef(name) for name in names]

    def get_role_scene_entities(
        self,
        role_id: str,
        *,
        swap: bool = False,
    ) -> list[str]:
        return [
            target.scene_name
            for target in self.get_role_candidates(role_id, swap=swap)
        ]


def test_role_scope_task_candidates_uses_widest_role_domains():
    scope = RoleScope.from_task(_Task())

    assert (
        scope.entities_for_role("pick"),
        scope.entities_for_role("place"),
    ) == (("apple", "banana"), ("basket", "tray"))


def test_role_scope_repeated_entity_preserves_stable_union():
    scope = RoleScope.from_role_members(
        {
            "pick": ("apple", "mug"),
            "place": ("mug", "tray"),
        }
    )

    assert scope.entities == ("apple", "mug", "tray")
    assert scope.entities_for_role("pick") == ("apple", "mug")


def test_role_scope_relation_pairs_uses_role_domains():
    scope = RoleScope.from_role_members(
        {
            "pick": ("apple", "mug"),
            "place": ("basket", "tray"),
        }
    )

    assert scope.relation_pairs("pick", "place") == (
        ("apple", "basket"),
        ("apple", "tray"),
        ("mug", "basket"),
        ("mug", "tray"),
    )
