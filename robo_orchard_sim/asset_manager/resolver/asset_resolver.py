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
    UnknownAssetError,
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
    from robo_orchard_sim.orchard_env.assets.pool_spec import PoolSpec

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


def _normalize_filter(entry: dict, role: str) -> dict:
    """Return filter dict; missing key or null -> empty (match-all)."""
    raw = entry.get("filter")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise AssetResolutionError(
            role=role,
            filter_repr=str(raw),
            cause=TypeError(
                f"filter must be a dict (or omitted/null for match-all); "
                f"got {type(raw).__name__}"
            ),
        )
    return dict(raw)


def _synth_meta_from_path(usd_path: str, entry: dict) -> AssetMeta:
    """Build a minimal in-memory AssetMeta for a path-pinned target.

    Only fields consumed by a no-distractor, no-instruction showcase scene
    are meaningful; the rest are deterministic placeholders.
    """
    import hashlib
    import os

    stem = os.path.splitext(os.path.basename(usd_path))[0] or "asset"
    uuid = hashlib.sha1(usd_path.encode("utf-8")).hexdigest()
    mass = float(entry.get("mass", 0.05))
    category = str(entry.get("category", stem))
    interaction_path = str(entry.get("interaction_path", ""))
    return AssetMeta(
        uuid=uuid,
        asset_id=stem,
        relative_path=usd_path,
        domain="showcase",
        super_category="showcase",
        category=category,
        name=stem,
        description=category,
        color=None,
        shape=None,
        material=None,
        real_height=0.0,
        real_mass=mass,
        min_height=0.0,
        max_height=0.0,
        min_mass=mass,
        max_mass=mass,
        usd_path=usd_path,
        urdf_path="",
        interaction_path=interaction_path,
        caption_path="",
    )


# -----------------------------------------------------------------------
# Split name -> AssetSplits field mapping
# -----------------------------------------------------------------------

_SPLIT_FIELDS = frozenset({"seen", "unseen_category", "unseen_instance"})

# Allowed keys per entry kind. Typos (e.g. ``macth`` instead of ``match``)
# would otherwise silently no-op — validate up-front so authoring errors
# surface on the first resolve call.
_TARGET_ENTRY_KEYS = frozenset(
    {
        "filter",
        "prim_name",
        "split",
        "pool_size",
        "uuid",
        "usd_path",
        "interaction_path",
        "mass",
        "category",
    }
)
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
        "pool_size",
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
        *,
        active_snapshot: frozenset[str] | None = None,
    ) -> None:
        self._registry = registry
        self._splits = splits
        self._sampler = AssetSampler(registry)
        self._rng = rng or np.random.default_rng()
        self._active_snapshot = active_snapshot

    def resolve(
        self,
        asset_configs: dict[str, dict],
    ) -> dict[str, Any]:
        """Resolve asset_configs into AssetSpec instances per role."""
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

        committed_uuids: set[str] = set()
        result: dict[str, Any] = {}
        target_metas: dict[str, AssetMeta] = {}

        for role, entry in asset_configs.items():
            if "anchor" not in entry:
                metas, spec_or_specs = self._resolve_target(
                    role,
                    entry,
                    committed_uuids,
                )
                target_metas[role] = metas[0]
                committed_uuids.update(m.uuid for m in metas)
                result[role] = spec_or_specs

        for role, entry in asset_configs.items():
            if "anchor" in entry:
                metas, specs = self._resolve_distractors(
                    role,
                    entry,
                    target_metas,
                    committed_uuids,
                )
                committed_uuids.update(m.uuid for m in metas)
                result[role] = specs

        self._log_capacity_report(asset_configs, result)
        return result

    def _log_capacity_report(
        self,
        asset_configs: dict[str, dict],
        resolved: dict[str, Any],
    ) -> None:
        """Log per-role pool capacity at INFO level."""
        lines = ["asset resolver — pool capacity:"]
        for role, value in resolved.items():
            if isinstance(value, list):
                count = len(value)
            else:
                count = len(value.members) if hasattr(value, "members") else 1
            entry = asset_configs[role]
            requested = entry.get("pool_size", count if count > 1 else 1)
            empty_filter = "uuid" not in entry and not entry.get("filter")
            hint = " [filter=<empty: full registry>]" if empty_filter else ""
            lines.append(
                f"  {role}: requested={requested}, resolved={count}{hint}"
            )
        logger.info("\n".join(lines))

    def _resolve_target(
        self,
        role: str,
        entry: dict,
        committed_uuids: set[str],
    ) -> tuple[list[AssetMeta], "RigidObjectSpec | PoolSpec"]:
        """Resolve a target-kind role.

        Returns (metas, spec_or_specs). Single spec when pool_size <= 1,
        list of specs when pool_size > 1. metas is always a list (length 1
        for classic) so the caller can update committed_uuids uniformly.
        """
        if "usd_path" in entry:
            meta, spec = self._resolve_target_by_path(role, entry)
            return [meta], spec

        if "uuid" in entry:
            meta, spec = self._resolve_target_by_uuid(role, entry)
            return [meta], spec

        filter_dict = _normalize_filter(entry, role)

        only_in = self._resolve_split_only_in(
            role, entry, err_repr=str(filter_dict)
        )
        if only_in is not None:
            filter_dict["only_in"] = only_in
        if committed_uuids:
            filter_dict["exclude"] = frozenset(committed_uuids)

        try:
            asset_filter = AssetFilter(**filter_dict)
        except TypeError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(filter_dict),
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

        pool_size = int(entry.get("pool_size", 1))
        if pool_size < 1:
            raise AssetResolutionError(
                role=role,
                filter_repr=repr(asset_filter),
                cause=ValueError(f"pool_size must be >= 1, got {pool_size}"),
            )

        try:
            if pool_size == 1:
                meta = self._sampler.sample_target(asset_filter, rng=self._rng)
                spec = self._registry.build_spec(
                    meta, name=prim_name, role=role
                )
                return [meta], spec
            metas = self._sampler.sample_target_pool(
                asset_filter,
                k=pool_size,
                rng=self._rng,
            )
            members = [
                self._registry.build_spec(
                    m,
                    name=f"{prim_name}_pool_{idx}",
                    role=role,
                )
                for idx, m in enumerate(metas)
            ]
            from robo_orchard_sim.orchard_env.assets.pool_spec import PoolSpec

            pool_spec = PoolSpec(role_id=prim_name, members=members)
            return metas, pool_spec
        except (EmptyPoolError, InsufficientPoolError) as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=repr(asset_filter),
                cause=exc,
            ) from exc

    def _resolve_target_by_uuid(
        self, role: str, entry: dict
    ) -> tuple[AssetMeta, "RigidObjectSpec"]:
        """Resolve a target pinned by uuid.

        uuid is authoritative: the asset is fetched directly from the
        registry. ``filter`` and ``split`` are still parsed for
        consistency checks but only emit a warning on mismatch — they do
        not override the explicit uuid.
        """
        uuid = entry["uuid"]
        try:
            meta = self._registry.get_meta(uuid)
        except UnknownAssetError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=exc,
            ) from exc

        try:
            prim_name = entry["prim_name"]
        except KeyError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=exc,
            ) from exc

        filter_dict = _normalize_filter(entry, role)
        if filter_dict:
            try:
                filter_check = AssetFilter(**filter_dict)
            except TypeError as exc:
                raise AssetResolutionError(
                    role=role,
                    filter_repr=str(filter_dict),
                    cause=exc,
                ) from exc
            if not filter_check.matches(meta):
                logger.warning(
                    "role %r: pinned uuid %r does not satisfy filter %r; "
                    "uuid takes precedence.",
                    role,
                    uuid,
                    filter_check,
                )

        only_in = self._resolve_split_only_in(role, entry, err_repr=str(entry))
        if only_in is not None and uuid not in only_in:
            split_name = entry.get("split")
            if split_name is not None and self._active_snapshot is not None:
                scope_desc = f"split {split_name!r} ∩ active_snapshot"
            elif split_name is not None:
                scope_desc = f"split {split_name!r}"
            else:
                scope_desc = "active_snapshot"
            logger.warning(
                "role %r: pinned uuid %r is not in %s; uuid takes precedence.",
                role,
                uuid,
                scope_desc,
            )

        return meta, self._registry.build_spec(meta, name=prim_name, role=role)

    def _resolve_target_by_path(
        self, role: str, entry: dict
    ) -> tuple[AssetMeta, "RigidObjectSpec"]:
        """Resolve a target pinned by a direct USD path (no registry lookup).

        For showcase scenes whose assets live as loose directories outside
        any registered library. ``usd_path`` is authoritative and may not be
        combined with registry-driven keys.
        """
        conflicting = sorted({"uuid", "filter", "split"} & set(entry))
        if conflicting:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=ValueError(
                    f"usd_path is mutually exclusive with {conflicting}."
                ),
            )
        try:
            prim_name = entry["prim_name"]
        except KeyError as exc:
            raise AssetResolutionError(
                role=role, filter_repr=str(entry), cause=exc
            ) from exc

        meta = _synth_meta_from_path(str(entry["usd_path"]), entry)
        spec = self._registry.build_spec(meta, name=prim_name, role=role)
        return meta, spec

    def _resolve_split_only_in(
        self,
        role: str,
        entry: dict,
        err_repr: str,
    ) -> frozenset[str] | None:
        """Resolve the `split` field to an `only_in` uuid set.

        Returns None when neither split nor active_snapshot applies.
        Raises AssetResolutionError for unknown split names or when
        the resulting (split ∩ active_snapshot) is empty.
        """
        only_in: frozenset[str] | None = None
        split_name = entry.get("split")
        if split_name is not None and self._splits is not None:
            if split_name not in _SPLIT_FIELDS:
                raise AssetResolutionError(
                    role=role,
                    filter_repr=err_repr,
                    cause=ValueError(
                        f"Unknown split '{split_name}'. "
                        f"Must be one of: {sorted(_SPLIT_FIELDS)}"
                    ),
                )
            only_in = getattr(self._splits, split_name)

        if self._active_snapshot is not None:
            only_in = (
                (only_in & self._active_snapshot)
                if only_in is not None
                else self._active_snapshot
            )
            if not only_in:
                scope_desc = (
                    f"split {split_name!r} ∩ active_snapshot"
                    if split_name is not None
                    else "active_snapshot"
                )
                raise AssetResolutionError(
                    role=role,
                    filter_repr=err_repr,
                    cause=ValueError(
                        f"{scope_desc} is empty — snapshot does not cover "
                        "any assets for this role"
                    ),
                )
        return only_in

    def _resolve_distractors(
        self,
        role: str,
        entry: dict,
        target_metas: dict[str, AssetMeta],
        committed_uuids: set[str],
    ) -> tuple[list[AssetMeta], list["RigidObjectSpec"] | "PoolSpec"]:
        """Resolve a distractor entry. Returns (metas, specs_or_pool)."""
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

        filter_dict = _normalize_filter(entry, role)
        try:
            absolute_filter = AssetFilter(**filter_dict)
        except TypeError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=exc,
            ) from exc

        pool_size = entry.get("pool_size", None)
        if pool_size is not None:
            pool_size = int(pool_size)
            if pool_size < max_count:
                raise AssetResolutionError(
                    role=role,
                    filter_repr=str(entry),
                    cause=ValueError(
                        f"pool_size ({pool_size}) must be >= max_count "
                        f"({max_count}) for distractor pool"
                    ),
                )
            if pool_size > 0 and max_count == 0:
                raise AssetResolutionError(
                    role=role,
                    filter_repr=str(entry),
                    cause=ValueError(
                        f"pool_size={pool_size} with max_count=0 would "
                        f"pre-spawn unused candidates; set max_count>=1 "
                        f"or omit pool_size"
                    ),
                )

        try:
            spec = DistractorSpec(
                min_count=min_count,
                max_count=max_count,
                match=tuple(entry.get("match", ())),
                differ=tuple(entry.get("differ", ())),
                absolute_filter=absolute_filter,
                only_in=only_in,
                exclude=frozenset(committed_uuids),
            )
        except ValueError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=exc,
            ) from exc

        try:
            if pool_size is None:
                metas = self._sampler.sample_distractors(
                    anchor_meta,
                    spec,
                    rng=self._rng,
                )
            else:
                metas = self._sampler.sample_distractor_pool(
                    anchor_meta,
                    spec,
                    pool_size=pool_size,
                    rng=self._rng,
                )
        except (EmptyPoolError, InsufficientPoolError, ValueError) as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(spec),
                cause=exc,
            ) from exc

        prim_name_prefix = entry.get("prim_name_prefix", role)
        # pool_size=None or pool_size<=1: classic list path (no pooling).
        if pool_size is None or pool_size <= 1:
            specs = [
                self._registry.build_spec(
                    meta, name=f"{prim_name_prefix}_{idx}", role=role
                )
                for idx, meta in enumerate(metas)
            ]
            return metas, specs

        # Pool path: wrap in PoolSpec with active_count = max_count.
        from robo_orchard_sim.orchard_env.assets.pool_spec import PoolSpec

        members = [
            self._registry.build_spec(
                meta, name=f"{prim_name_prefix}_pool_{idx}", role=role
            )
            for idx, meta in enumerate(metas)
        ]
        pool_spec = PoolSpec(
            role_id=prim_name_prefix,
            members=members,
            active_count=max_count,
        )
        return metas, pool_spec
