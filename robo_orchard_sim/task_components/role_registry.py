# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
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

"""Runtime mapping from task role ids to concrete scene targets."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from robo_orchard_sim.contracts.articulated_operation import (
    SUPPORTED_ARTICULATED_OPERATIONS,
    ArticulatedOperation,
)


@dataclass(frozen=True)
class TargetRef:
    """Points at a whole scene entity, or one part of it under an operation.

    Carries identity and intent only. Poses, body indices and other runtime
    facts are derived from the live scene, never stored here.

    ``semantic_name`` names the movable part in asset metadata (for example
    ``lid`` or ``upper drawer``). Metadata resolves it to the physical joint
    and moving body on the live articulation.

    ``operation`` names what the bound role is meant to do with
    ``semantic_name`` — e.g. opening versus closing the same drawer are two
    distinct targets that share a joint. It is only meaningful alongside a
    ``semantic_name``.
    """

    scene_name: str
    semantic_name: str | None = None
    operation: ArticulatedOperation | None = None

    def __post_init__(self) -> None:
        if not self.scene_name:
            raise ValueError("scene_name must be non-empty.")
        if self.semantic_name is not None and not self.semantic_name:
            raise ValueError(
                f"semantic_name must be non-empty when set; use None to "
                f"reference '{self.scene_name}' as a whole."
            )
        if self.operation is None:
            return
        if self.semantic_name is None:
            raise ValueError(
                f"operation '{self.operation}' needs a semantic_name to act "
                f"on; '{self.scene_name}' as a whole has no joint to operate."
            )
        if self.operation not in SUPPORTED_ARTICULATED_OPERATIONS:
            registered = ", ".join(sorted(SUPPORTED_ARTICULATED_OPERATIONS))
            raise ValueError(
                f"Unknown operation '{self.operation}'. "
                f"Registered operations: {registered}."
            )

    def __str__(self) -> str:
        """Render as ``scene_name[/semantic_name[:operation]]``."""
        if self.semantic_name is None:
            return self.scene_name
        if self.operation is None:
            return f"{self.scene_name}/{self.semantic_name}"
        return f"{self.scene_name}/{self.semantic_name}:{self.operation}"


Cardinality = Literal["one", "many"]

_Binding = tuple[Cardinality, TargetRef | list[TargetRef]]


class RoleRegistry:
    """Per-env mapping from role_id to the target(s) it currently names.

    Mutable by design: bindings are rewritten each episode by the
    evaluator after a target is selected. Validators and instruction
    builders read from it to resolve which concrete scene target a role
    currently points at.

    Cardinality is recorded at bind time rather than inferred from the
    stored value, so that ``one`` and ``many`` stay distinguishable no
    matter what the value type is.
    """

    def __init__(self, num_envs: int = 1) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self._table: list[dict[str, _Binding]] = [{} for _ in range(num_envs)]

    @property
    def num_envs(self) -> int:
        """Number of environments this registry tracks bindings for."""
        return len(self._table)

    def bind_one(self, env_idx: int, role_id: str, target: TargetRef) -> None:
        """Bind a ``cardinality='one'`` role to a single target."""
        self._table[env_idx][role_id] = ("one", target)

    def bind_many(
        self,
        env_idx: int,
        role_id: str,
        targets: list[TargetRef],
    ) -> None:
        """Bind a ``cardinality='many'`` role to a list of targets."""
        self._table[env_idx][role_id] = ("many", list(targets))

    def clear(self, env_idx: int | None = None) -> None:
        """Clear bindings for one env, or all envs when env_idx is None."""
        if env_idx is None:
            for table in self._table:
                table.clear()
        else:
            self._table[env_idx].clear()

    def _binding(self, role_id: str, env_idx: int) -> _Binding:
        try:
            return self._table[env_idx][role_id]
        except KeyError:
            raise KeyError(
                f"Role '{role_id}' is not bound for env {env_idx}. "
                f"Bound roles: {sorted(self._table[env_idx])}"
            ) from None

    def resolve(
        self,
        role_id: str,
        env_idx: int = 0,
    ) -> TargetRef | list[TargetRef]:
        """Return the bound target(s) without asserting cardinality."""
        return self._binding(role_id, env_idx)[1]

    def resolve_one(self, role_id: str, env_idx: int = 0) -> TargetRef:
        """Return the single target bound to a ``cardinality='one'`` role."""
        cardinality, value = self._binding(role_id, env_idx)
        if cardinality != "one":
            raise TypeError(
                f"Role '{role_id}' is bound with cardinality='many'; "
                "use resolve_many() instead."
            )
        assert isinstance(value, TargetRef)
        return value

    def resolve_many(self, role_id: str, env_idx: int = 0) -> list[TargetRef]:
        """Return all targets bound to a ``cardinality='many'`` role."""
        cardinality, value = self._binding(role_id, env_idx)
        if cardinality != "many":
            raise TypeError(
                f"Role '{role_id}' is bound with cardinality='one'; "
                "use resolve_one() instead."
            )
        assert isinstance(value, list)
        return value

    def bindings(
        self, env_idx: int = 0
    ) -> dict[str, TargetRef | list[TargetRef]]:
        """Snapshot of current targets (copy, safe to log or record)."""
        return {
            role_id: value
            for role_id, (_, value) in self._table[env_idx].items()
        }
