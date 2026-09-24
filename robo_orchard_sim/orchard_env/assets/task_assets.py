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

"""Which concrete assets got picked for each role in one build."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from robo_orchard_sim.orchard_env.assets.object_spec import ObjectSpec


@dataclass(frozen=True)
class TaskAssets:
    """The candidates AssetResolver chose for every role of one task.

    Build-time data only. Which candidate a role actually points at
    during an episode is answered by the RoleRegistry, not here.

    Roles are keyed as the task YAML wrote them, so a single
    ``distractors`` entry that expanded into several objects stays one
    role holding several candidates rather than becoming
    ``distractor_0``, ``distractor_1``, ... .

    The views below each serve exactly one caller; they read the same
    ``role_candidates`` in the same order, so two calls never disagree
    about ordering.
    """

    role_candidates: dict[str, list[ObjectSpec]] = field(default_factory=dict)

    @classmethod
    def from_resolved(cls, resolved: Mapping[str, Any]) -> "TaskAssets":
        """Build from AssetResolver output, which is per-role already.

        A role resolves to a bare spec when the YAML asks for one object
        and to a list when it asks for several. Both become a list here
        so downstream task code can handle the shapes uniformly.
        """
        return cls(
            role_candidates={
                role_id: list(value) if isinstance(value, list) else [value]
                for role_id, value in resolved.items()
            }
        )

    def flatten(self) -> dict[str, ObjectSpec]:
        """Scene entities to register, keyed by scene_name.

        Uniqueness of scene_name is guaranteed upstream by AssetResolver.
        """
        return {
            spec.scene_name: spec
            for specs in self.role_candidates.values()
            for spec in specs
        }

    def all_scene_names(self) -> list[str]:
        """Every scene_name in this build, for pose reset to cover.

        The returned names identify the entities that exist in the scene.
        """
        return [
            spec.scene_name
            for specs in self.role_candidates.values()
            for spec in specs
        ]

    def by_role(self, role_id: str) -> list[ObjectSpec]:
        """Candidates offered for one role; empty when the role is absent."""
        return self.role_candidates.get(role_id, [])

    def with_default_namespace(self, namespace: str) -> "TaskAssets":
        """Return a copy namespaced the way ``TaskBase`` namespaces assets.

        ``TaskBase.__init__`` copies each spec to fill in the namespace,
        so a task holding the pre-namespaced assets would hand out scene
        names that do not key ``env.scene``. Applying the same rule here
        keeps the per-role view in step.
        """
        return TaskAssets(
            role_candidates={
                role_id: [
                    cast(
                        ObjectSpec,
                        spec.with_default_namespace(namespace),
                    )
                    for spec in specs
                ]
                for role_id, specs in self.role_candidates.items()
            }
        )
