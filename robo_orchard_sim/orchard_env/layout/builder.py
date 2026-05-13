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

"""Layout consumption: asset resolution + env-level cycling."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from robo_orchard_core.envs.managers.events import EventManagerCfg

from robo_orchard_sim.envs.managers.events.layout_reset import (
    LayoutResetTermCfg,
)
from robo_orchard_sim.envs.managers.events.pool_reset import PoolResetTermCfg
from robo_orchard_sim.envs.managers.events.pose_reset import PoseResetTermCfg
from robo_orchard_sim.orchard_env.assets import ObjectSpec
from robo_orchard_sim.orchard_env.assets.pool_spec import PoolSpec
from robo_orchard_sim.orchard_env.layout.loader import (
    LayoutSequence,
    LayoutValidationError,
)

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
        AssetResolver,
    )


__all__ = ["LayoutBuilder"]


@dataclass(frozen=True)
class LayoutBuilder:
    """A parsed layout bound to its resolved scene-actor names."""

    # Task event-cfg term types that layout takes ownership of.
    _SHADOWS: ClassVar[tuple[type, ...]] = (PoseResetTermCfg, PoolResetTermCfg)

    layouts: LayoutSequence
    role_member_by_category: Mapping[str, Mapping[str, str]]

    @classmethod
    def build(
        cls,
        layouts: LayoutSequence,
        resolver: AssetResolver,
        role_map: Mapping[str, str],
    ) -> tuple[dict[str, ObjectSpec | PoolSpec], LayoutBuilder]:
        """Resolve per (role × unique-category) and return (assets, builder).

        1 unique category per role → ``ObjectSpec``; ≥2 → ``PoolSpec`` named
        ``{slot}_pool_{idx}`` so ``env_base`` auto-attaches ``PoolAliasState``.

        Args:
            layouts (LayoutSequence): Parsed layout sequence.
            resolver (AssetResolver): Asset resolver used to materialise each
                (role, category) pair into an ``ObjectSpec``.
            role_map (Mapping[str, str]): JSON role name → task-slot name
                mapping (e.g. ``{"pick": "pick", "ref": "distractor_0"}``).

        Returns:
            tuple[dict[str, ObjectSpec | PoolSpec], LayoutBuilder]: Keyed by
            task slot — pass straight to the task's assets schema. The
            ``LayoutBuilder`` carries the per-episode cycling state and is
            handed to ``OrchardEnv(layout_builder=...)``.
        """
        layout_roles = list(role_map.keys())
        seen_per_role: dict[str, list[str]] = {r: [] for r in layout_roles}
        seen_sets: dict[str, set[str]] = {r: set() for r in layout_roles}
        for idx, entry in enumerate(layouts.entries):
            missing = [r for r in layout_roles if r not in entry.objects]
            if missing:
                raise LayoutValidationError(
                    f"entry[{idx}] missing role(s) {missing!r}; "
                    f"role_map declared {sorted(layout_roles)}"
                )
            for role in layout_roles:
                cat = entry.objects[role].category
                if cat not in seen_sets[role]:
                    seen_sets[role].add(cat)
                    seen_per_role[role].append(cat)

        def _scene_name(slot: str, idx: int, n: int) -> str:
            return slot if n == 1 else f"{slot}_pool_{idx}"

        asset_configs: dict[str, dict[str, Any]] = {}
        for layout_role, cats in seen_per_role.items():
            slot = role_map[layout_role]
            for i, cat in enumerate(cats):
                key = _scene_name(slot, i, len(cats))
                asset_configs[key] = {
                    "filter": {"category": cat},
                    "prim_name": key,
                }
        resolved = resolver.resolve(asset_configs)

        assets: dict[str, ObjectSpec | PoolSpec] = {}
        role_member_by_category: dict[str, dict[str, str]] = {}
        for layout_role, cats in seen_per_role.items():
            slot = role_map[layout_role]
            specs = [
                resolved[_scene_name(slot, i, len(cats))]
                for i in range(len(cats))
            ]
            for spec in specs:
                if not isinstance(spec, ObjectSpec):
                    raise TypeError(
                        f"resolver returned non-ObjectSpec for slot={slot!r}: "
                        f"{type(spec).__name__}"
                    )
            specs = [s.with_default_namespace("objects") for s in specs]
            # keyed by the layout JSON role; LayoutResetTerm iterates layout
            # entries (which carry JSON role names) and indexes this map.
            role_member_by_category[layout_role] = {
                cat: specs[i].scene_name for i, cat in enumerate(cats)
            }
            assets[slot] = (
                specs[0]
                if len(specs) == 1
                else PoolSpec(role_id=slot, members=specs)
            )
        return assets, cls(
            layouts=layouts, role_member_by_category=role_member_by_category
        )

    @property
    def num_episodes(self) -> int:
        """Cycle length — drives the outer episode loop."""
        return len(self.layouts.entries)

    def apply_to(self, task_event_cfg: EventManagerCfg) -> EventManagerCfg:
        """Replace pose/pool-reset terms with layout's; keep the rest."""
        merged = {
            k: v
            for k, v in task_event_cfg.terms.items()
            if not isinstance(v, self._SHADOWS)
        }
        merged.update(self.event_cfg().terms)
        return EventManagerCfg(terms=merged)

    def event_cfg(self) -> EventManagerCfg:
        """Single-term EventManagerCfg wrapping a ``LayoutResetTerm``."""
        return EventManagerCfg(
            terms={
                "layout_reset": LayoutResetTermCfg(
                    layouts=self.layouts,
                    role_member_by_category={
                        role: dict(members)
                        for role, members in (
                            self.role_member_by_category.items()
                        )
                    },
                ),
            }
        )
