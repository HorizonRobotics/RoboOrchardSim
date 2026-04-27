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

"""Generic asset resolver: config + registry + splits -> AssetSpec dict."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from robo_orchard_sim.asset_manager.registry.errors import (
    EmptyPoolError,
    InsufficientPoolError,
)
from robo_orchard_sim.asset_manager.registry.registry import (
    AssetRegistry,
    AssetSampler,
)
from robo_orchard_sim.asset_manager.registry.types import (
    AssetFilter,
    AssetMeta,
    DistractorSpec,
)
from robo_orchard_sim.asset_manager.splits.splits import AssetSplits

if TYPE_CHECKING:
    from robo_orchard_sim.orchard_env.assets import RigidObjectSpec

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------


class AssetResolverError(Exception):
    """Base class for resolver errors."""


class AssetResolutionError(AssetResolverError):
    """Sampling failed for a specific role."""

    def __init__(self, role: str, filter_repr: str, cause: Exception) -> None:
        self.role = role
        self.filter_repr = filter_repr
        self.cause = cause
        super().__init__(f"role '{role}' — {cause}")


# -----------------------------------------------------------------------
# Split name -> AssetSplits field mapping
# -----------------------------------------------------------------------

_SPLIT_FIELDS = frozenset({"seen", "unseen_category", "unseen_instance"})

# Allowed keys per entry kind. Typos (e.g. ``macth`` instead of ``match``)
# would otherwise silently no-op — validate up-front so authoring errors
# surface on the first resolve call.
_TARGET_ENTRY_KEYS = frozenset({"filter", "prim_name", "split"})
_DISTRACTOR_ENTRY_KEYS = frozenset(
    {
        "anchor",
        "match",
        "differ",
        "filter",
        "min_count",
        "max_count",
        "prim_name_prefix",
        "split",
    }
)


# -----------------------------------------------------------------------
# AssetResolver
# -----------------------------------------------------------------------


class AssetResolver:
    """Resolve per-role asset configs into concrete AssetSpec instances.

    Constructed once per evaluator session with a shared registry,
    optional asset splits, and an rng. Called once per task with the
    task's asset configs.

    The resolver is task-agnostic: it transforms a ``dict[role, config]``
    into a ``dict[role, AssetSpec | list[AssetSpec]]`` and never inspects
    the calling task's schema. Role membership and required/optional
    semantics are owned by the task's ``TaskAssetsBase`` subclass and
    enforced when the caller constructs that dataclass from this
    resolver's output.
    """

    def __init__(
        self,
        registry: AssetRegistry,
        splits: AssetSplits | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._registry = registry
        self._splits = splits
        self._sampler = AssetSampler(registry)
        self._rng = rng or np.random.default_rng()

    def resolve(
        self,
        asset_configs: dict[str, dict],
    ) -> dict[str, Any]:
        """Resolve asset_configs into a dict of AssetSpec instances.

        Entries without an ``anchor`` key are target entries producing a
        single ``AssetSpec``. Entries with ``anchor: <role>`` are
        distractor entries producing a ``list[AssetSpec]`` sampled
        relative to the already-resolved anchor role.

        Args:
            asset_configs: Per-role config dicts.

        Returns:
            Dict mapping role strings to resolved AssetSpec instances
            (single AssetSpec for target entries, list[AssetSpec] for
            distractor entries).
        """
        # Up-front key validation so typos like ``macth`` or ``anhor``
        # fail loudly instead of being silently no-op.
        for role, entry in asset_configs.items():
            allowed = (
                _DISTRACTOR_ENTRY_KEYS
                if "anchor" in entry
                else _TARGET_ENTRY_KEYS
            )
            unknown = frozenset(entry.keys()) - allowed
            if unknown:
                raise AssetResolutionError(
                    role=role,
                    filter_repr=str(entry),
                    cause=ValueError(
                        f"Unknown entry key(s): {sorted(unknown)}. "
                        f"Allowed: {sorted(allowed)}"
                    ),
                )

        # Pass 1: resolve every target entry; cache the meta so
        # distractor entries in Pass 2 can sample relative to it.
        result: dict[str, Any] = {}
        target_metas: dict[str, AssetMeta] = {}
        for role, entry in asset_configs.items():
            if "anchor" not in entry:
                meta, spec = self._resolve_target(role, entry)
                target_metas[role] = meta
                result[role] = spec

        # Pass 2: resolve distractor entries using cached metas.
        for role, entry in asset_configs.items():
            if "anchor" in entry:
                result[role] = self._resolve_distractors(
                    role, entry, target_metas
                )

        return result

    def _resolve_target(
        self, role: str, entry: dict
    ) -> tuple[AssetMeta, "RigidObjectSpec"]:
        """Resolve a single target-kind role. Returns (meta, spec)."""
        try:
            filter_dict = dict(entry["filter"])
        except KeyError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=exc,
            ) from exc
        only_in = self._resolve_split_only_in(
            role, entry, err_repr=str(filter_dict)
        )
        if only_in is not None:
            filter_dict["only_in"] = only_in

        try:
            asset_filter = AssetFilter(**filter_dict)
        except TypeError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(filter_dict),
                cause=exc,
            ) from exc

        try:
            meta = self._sampler.sample_target(asset_filter, rng=self._rng)
        except (EmptyPoolError, InsufficientPoolError) as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=repr(asset_filter),
                cause=exc,
            ) from exc

        try:
            prim_name = entry["prim_name"]
        except KeyError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=repr(asset_filter),
                cause=exc,
            ) from exc

        return meta, self._registry.build_spec(meta, name=prim_name, role=role)

    def _resolve_split_only_in(
        self,
        role: str,
        entry: dict,
        err_repr: str,
    ) -> frozenset[str] | None:
        """Resolve the `split` field to a `only_in` uuid set.

        Returns None when the entry omits ``split`` (no split filter
        applied, full library) or when the resolver has no splits
        configured. Raises AssetResolutionError for unknown split names.
        """
        split_name = entry.get("split")
        if split_name is None:
            return None
        if self._splits is None:
            return None
        if split_name not in _SPLIT_FIELDS:
            raise AssetResolutionError(
                role=role,
                filter_repr=err_repr,
                cause=ValueError(
                    f"Unknown split '{split_name}'. "
                    f"Must be one of: {sorted(_SPLIT_FIELDS)}"
                ),
            )
        return getattr(self._splits, split_name)

    def _resolve_distractors(
        self,
        role: str,
        entry: dict,
        target_metas: dict[str, AssetMeta],
    ) -> list["RigidObjectSpec"]:
        """Resolve a distractor entry relative to a cached anchor."""
        anchor = entry.get("anchor")
        if anchor not in target_metas:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=ValueError(
                    f"anchor '{anchor}' does not refer to an "
                    f"already-resolved target role. Known target roles: "
                    f"{sorted(target_metas.keys())}"
                ),
            )
        anchor_meta = target_metas[anchor]

        try:
            min_count = int(entry["min_count"])
            max_count = int(entry["max_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=exc,
            ) from exc

        only_in = self._resolve_split_only_in(role, entry, err_repr=str(entry))

        try:
            absolute_filter = AssetFilter(**dict(entry.get("filter", {})))
        except TypeError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=exc,
            ) from exc

        try:
            spec = DistractorSpec(
                min_count=min_count,
                max_count=max_count,
                match=tuple(entry.get("match", ())),
                differ=tuple(entry.get("differ", ())),
                absolute_filter=absolute_filter,
                only_in=only_in,
            )
        except ValueError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=exc,
            ) from exc

        try:
            metas = self._sampler.sample_distractors(
                anchor_meta, spec, rng=self._rng
            )
        except (EmptyPoolError, InsufficientPoolError, ValueError) as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(spec),
                cause=exc,
            ) from exc

        prim_name_prefix = entry.get("prim_name_prefix", role)
        return [
            self._registry.build_spec(
                meta,
                name=f"{prim_name_prefix}_{idx}",
                role=role,
            )
            for idx, meta in enumerate(metas)
        ]
