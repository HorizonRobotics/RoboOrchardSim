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
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from robo_orchard_core.envs.managers.events import EventManagerCfg

from robo_orchard_sim.ext.envs.managers.events.layout_reset import (
    LayoutResetTermCfg,
)
from robo_orchard_sim.ext.envs.managers.events.pool_reset import (
    PoolResetTermCfg,
)
from robo_orchard_sim.ext.envs.managers.events.pose_reset import (
    PoseResetTermCfg,
)
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

_DISTRACTOR_N_RE = re.compile(r"^distractor_\d+$")
_EMPTY_OVERLAY: dict[str, Any] = {"filter": {}, "split": None}


@dataclass(frozen=True)
class LayoutBuilder:
    """A parsed layout bound to its resolved scene-actor names."""

    # Task event-cfg term types that layout takes ownership of.
    _SHADOWS: ClassVar[tuple[type, ...]] = (PoseResetTermCfg, PoolResetTermCfg)

    layouts: LayoutSequence
    role_member_by_category: Mapping[str, Mapping[str, str]]

    @staticmethod
    def _validate_slot_overlays(
        overlay: Mapping[str, Mapping[str, Any]] | None,
        named_slots: set[str],
    ) -> dict[str, dict[str, Any]]:
        """Validate overlay; return {role_class: {"filter", "split"}}.

        Keys are task-contract role classes: a named slot (e.g. "pick",
        "anchor", "place") or the literal "distractors", which broadcasts
        to every auto-derived distractor slot. Per-instance
        "distractor_<N>" keys are rejected because their numbering
        depends on layout-instance role order.
        """
        if not overlay:
            return {}
        valid_keys = named_slots | {"distractors"}
        out: dict[str, dict[str, Any]] = {}
        for key, entry in overlay.items():
            if key not in valid_keys:
                hint = (
                    "; use 'distractors' to constrain all auto-derived "
                    "distractor slots"
                    if _DISTRACTOR_N_RE.match(key)
                    else ""
                )
                raise LayoutValidationError(
                    f"asset_configs[{key!r}]: unknown role class; valid "
                    f"keys are {sorted(valid_keys)}{hint}"
                )
            if not isinstance(entry, Mapping):
                raise LayoutValidationError(
                    f"asset_configs[{key!r}] must be a mapping, "
                    f"got {type(entry).__name__}"
                )
            extra = set(entry) - {"filter", "split"}
            if extra:
                raise LayoutValidationError(
                    f"asset_configs[{key!r}]: unexpected key(s) "
                    f"{sorted(extra)}; layout mode only honors 'filter' "
                    f"and 'split' (prim_name / pool_size / uuid / anchor "
                    f"are auto-derived or N/A)"
                )
            if not entry:
                raise LayoutValidationError(
                    f"asset_configs[{key!r}] must contain 'filter' "
                    f"and/or 'split'"
                )
            filter_body: dict[str, Any] = {}
            if "filter" in entry:
                if not isinstance(entry["filter"], Mapping):
                    raise LayoutValidationError(
                        f"asset_configs[{key!r}].filter must be a "
                        f"mapping, got {type(entry['filter']).__name__}"
                    )
                filter_body = dict(entry["filter"])
                if "category" in filter_body:
                    raise LayoutValidationError(
                        f"asset_configs[{key!r}].filter: 'category' is "
                        f"forbidden in layout mode (layout JSON is the "
                        f"authoritative source)"
                    )
            split = entry.get("split")
            if split is not None and (
                not isinstance(split, str) or not split.strip()
            ):
                raise LayoutValidationError(
                    f"asset_configs[{key!r}].split must be a non-empty "
                    f"string, got {split!r}"
                )
            out[key] = {"filter": filter_body, "split": split}
        return out

    @classmethod
    def build(
        cls,
        layouts: LayoutSequence,
        resolver: AssetResolver,
        named_roles: Mapping[str, str],
        slot_filters: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[dict[str, ObjectSpec | PoolSpec], LayoutBuilder]:
        """Resolve per (slot × unique-category) and return (assets, builder).

        ``named_roles`` maps upstream JSON role → task slot (e.g.
        ``{"src": "pick", "dest": "place"}``). Every other upstream role
        found in the layout is assigned, in insertion order, to
        ``distractor_0``, ``distractor_1``, … . 1 unique category per
        slot → ``ObjectSpec``; ≥2 → ``PoolSpec`` named ``{slot}_pool_{idx}``.

        ``slot_filters`` overlays per-role-class ``filter``/``split``
        dicts, keyed by a named slot or the literal ``distractors``
        (broadcast to all auto-derived slots); layout JSON's ``category``
        always wins. ``role_member_by_category`` stays keyed by the
        upstream JSON role, since ``LayoutResetTerm`` indexes it while
        iterating ``layout.objects``.
        """
        if not layouts.entries:
            raise LayoutValidationError("empty layout sequence")

        # Every named role must exist in every entry.
        for idx, entry in enumerate(layouts.entries):
            missing = [r for r in named_roles if r not in entry.objects]
            if missing:
                raise LayoutValidationError(
                    f"entry[{idx}] missing named role(s) {missing!r}; "
                    f"named_roles declared {sorted(named_roles)}"
                )

        # parse_layout already enforces identical role-key sets across
        # entries, but LayoutBuilder.build may be called with a hand-built
        # LayoutSequence — re-check so distractor slot count stays stable.
        first = layouts.entries[0].objects
        first_roles = set(first)
        for idx, entry in enumerate(layouts.entries[1:], start=1):
            if set(entry.objects) != first_roles:
                raise LayoutValidationError(
                    f"entry[{idx}] role keys differ from entry[0]: "
                    f"{sorted(entry.objects)} vs {sorted(first_roles)}"
                )

        # Other roles = entry roles not in named_roles, in insertion order.
        other_roles = [r for r in first if r not in named_roles]
        role_to_slot: dict[str, str] = dict(named_roles)
        for i, role in enumerate(other_roles):
            role_to_slot[role] = f"distractor_{i}"

        named_slots = set(named_roles.values())
        overlay_map = cls._validate_slot_overlays(slot_filters, named_slots)

        layout_roles = list(role_to_slot.keys())
        seen_per_role: dict[str, list[str]] = {r: [] for r in layout_roles}
        seen_sets: dict[str, set[str]] = {r: set() for r in layout_roles}
        for entry in layouts.entries:
            for role in layout_roles:
                cat = entry.objects[role].category
                if cat not in seen_sets[role]:
                    seen_sets[role].add(cat)
                    seen_per_role[role].append(cat)

        def _scene_name(slot: str, idx: int, n: int) -> str:
            return slot if n == 1 else f"{slot}_pool_{idx}"

        asset_configs: dict[str, dict[str, Any]] = {}
        for layout_role, cats in seen_per_role.items():
            slot = role_to_slot[layout_role]
            overlay = overlay_map.get(
                slot if slot in named_slots else "distractors",
                _EMPTY_OVERLAY,
            )
            for i, cat in enumerate(cats):
                key = _scene_name(slot, i, len(cats))
                cfg_entry: dict[str, Any] = {
                    "filter": {**overlay["filter"], "category": cat},
                    "prim_name": key,
                }
                if overlay["split"] is not None:
                    cfg_entry["split"] = overlay["split"]
                asset_configs[key] = cfg_entry
        resolved = resolver.resolve(asset_configs)

        assets: dict[str, ObjectSpec | PoolSpec] = {}
        role_member_by_category: dict[str, dict[str, str]] = {}
        for layout_role, cats in seen_per_role.items():
            slot = role_to_slot[layout_role]
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
