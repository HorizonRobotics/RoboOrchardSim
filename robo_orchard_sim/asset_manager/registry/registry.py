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

"""AssetRegistry + AssetSampler.

Uuid-keyed in-memory asset view plus stateless sampling primitives on top.
"""

from __future__ import annotations
import difflib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import pyarrow.parquet as pq

from robo_orchard_sim.asset_manager.registry.build_index import (
    INDEX_FILENAME,
    SCHEMA_VERSION,
    asset_set_fingerprint,
    build_asset_index,
)
from robo_orchard_sim.asset_manager.registry.errors import (
    AssetIndexNotFoundError,
    AssetIndexVersionError,
    AssetRegistryError,
    CollisionExhaustedError,
    EmptyPoolError,
    InsufficientPoolError,
    UnknownAssetError,
)
from robo_orchard_sim.asset_manager.registry.types import (
    AssetFilter,
    AssetMeta,
    DistractorSpec,
)

if TYPE_CHECKING:
    from robo_orchard_sim.orchard_env.assets import RigidObjectSpec

logger = logging.getLogger(__name__)


def _num(v: Any, default: float = 0.0) -> float:
    """Return float(v) if v is not None, else default."""
    return float(v) if v is not None else default


def _row_set(row: dict[str, Any], key: str) -> frozenset[str] | None:
    """Convert a pyarrow list<string> column value to frozenset, or None."""
    val = row.get(key)
    if val is None:
        return None
    fs = frozenset(val)
    return fs or None


def _row_to_meta(row: dict[str, Any]) -> AssetMeta:
    """Convert a parquet row dict to an AssetMeta.

    Note: `tags` is a pyarrow list<string> column, so
    `row["tags"]` is already a list[str] — do NOT split.
    """
    tags = frozenset(row.get("tags") or [])
    return AssetMeta(
        uuid=row["uuid"],
        asset_id=row["asset_id"],
        relative_path=row["relative_path"],
        domain=row["domain"],
        super_category=row["super_category"],
        category=row["category"],
        name=row.get("name") or "",
        description=row.get("description") or "",
        color=_row_set(row, "color"),
        shape=_row_set(row, "shape"),
        material=_row_set(row, "material"),
        real_height=_num(row.get("real_height")),
        real_mass=_num(row.get("real_mass")),
        min_height=_num(row.get("min_height")),
        max_height=_num(row.get("max_height")),
        min_mass=_num(row.get("min_mass")),
        max_mass=_num(row.get("max_mass")),
        usd_path=row["usd_path"],
        urdf_path=row["urdf_path"],
        interaction_path=row["interaction_path"],
        caption_path=row["caption_path"],
        tags=tags,
        version=row.get("version") or "",
        generate_time=row.get("generate_time") or "",
        aabb_x_min=row.get("aabb_x_min"),
        aabb_x_max=row.get("aabb_x_max"),
        aabb_y_min=row.get("aabb_y_min"),
        aabb_y_max=row.get("aabb_y_max"),
        aabb_z_min=row.get("aabb_z_min"),
        aabb_z_max=row.get("aabb_z_max"),
    )


class AssetRegistry:
    """Pure asset-library view. No benchmark knowledge."""

    def __init__(
        self,
        asset_root: str,
        *,
        index_path: str | Path | None = None,
        auto_build_index: bool = True,
    ) -> None:
        self._asset_root = Path(asset_root)
        self._index_path = (
            Path(index_path)
            if index_path is not None
            else self._asset_root / INDEX_FILENAME
        )
        self._metas: dict[str, AssetMeta] = {}
        self._by_asset_id: dict[str, str] = {}
        self._by_category: dict[str, list[str]] = {}
        self._by_super: dict[str, list[str]] = {}
        self._by_tag: dict[str, list[str]] = {}
        self._load(auto_build=auto_build_index)

    def _check_asset_set_staleness(self, table, schema_meta, *, auto_build):
        """Rebuild (or warn) if the asset set changed since the index build."""
        stored = schema_meta.get(b"asset_set_fingerprint")
        if stored is None:
            logger.info(
                "index at %s has no asset_set_fingerprint (older build); "
                "staleness check skipped until next rebuild",
                self._index_path,
            )
            return table
        current = asset_set_fingerprint(self._asset_root)
        if stored.decode() == current:
            return table
        if auto_build:
            logger.info(
                "asset set changed since index build; rebuilding %s",
                self._index_path,
            )
            build_asset_index(
                str(self._asset_root), output_path=str(self._index_path)
            )
            return pq.read_table(str(self._index_path))
        logger.warning(
            "asset set changed since index build at %s; index may be stale "
            "(new/removed assets invisible). Rebuild with build_asset_index "
            "or pass auto_build_index=True.",
            self._index_path,
        )
        return table

    def _load(self, *, auto_build: bool) -> None:
        index_path = self._index_path
        if not index_path.exists():
            if not auto_build:
                raise AssetIndexNotFoundError(
                    f"No asset index at {index_path}; "
                    "run build_asset_index or pass "
                    "auto_build_index=True"
                )
            logger.info("index missing; building at %s", index_path)
            build_asset_index(
                str(self._asset_root), output_path=str(index_path)
            )

        table = pq.read_table(str(index_path))

        # --- schema-version check (Fix 4) ---
        schema_meta = table.schema.metadata or {}
        raw_version = schema_meta.get(b"schema_version")
        if not isinstance(raw_version, bytes):
            raise AssetIndexVersionError(
                "asset_index.parquet missing or invalid schema_version "
                "metadata; rebuild required"
            )
        try:
            version = raw_version.decode()
        except UnicodeDecodeError as e:
            raise AssetIndexVersionError(
                f"asset_index.parquet schema_version not decodable: {e}; "
                "rebuild required"
            ) from e
        if version != SCHEMA_VERSION:
            if not auto_build:
                raise AssetIndexVersionError(
                    f"asset_index.parquet schema_version={version!r} "
                    f"expected {SCHEMA_VERSION!r}; rebuild required"
                )
            logger.info(
                "index schema version mismatch at %s: found %s expected %s; "
                "rebuilding",
                index_path,
                version,
                SCHEMA_VERSION,
            )
            build_asset_index(
                str(self._asset_root), output_path=str(index_path)
            )
            table = pq.read_table(str(index_path))
            schema_meta = table.schema.metadata or {}
            raw_version = schema_meta.get(b"schema_version")
            if not isinstance(raw_version, bytes):
                raise AssetIndexVersionError(
                    "rebuilt asset_index.parquet missing schema_version "
                    "metadata"
                )
            try:
                version = raw_version.decode()
            except UnicodeDecodeError as e:
                raise AssetIndexVersionError(
                    f"rebuilt asset_index.parquet schema_version not "
                    f"decodable: {e}"
                ) from e
            if version != SCHEMA_VERSION:
                raise AssetIndexVersionError(
                    f"rebuilt asset_index.parquet schema_version={version!r} "
                    f"expected {SCHEMA_VERSION!r}"
                )

        table = self._check_asset_set_staleness(
            table, schema_meta, auto_build=auto_build
        )

        # --- atomic load (Fix 1 + Fix 2) ---
        metas: dict[str, AssetMeta] = {}
        by_asset_id: dict[str, str] = {}
        by_category: dict[str, list[str]] = {}
        by_super: dict[str, list[str]] = {}
        by_tag: dict[str, list[str]] = {}

        for row in table.to_pylist():
            m = _row_to_meta(row)

            if m.uuid in metas:
                existing = metas[m.uuid]
                raise AssetRegistryError(
                    f"Duplicate uuid {m.uuid!r} found for asset_ids "
                    f"{existing.asset_id!r} and {m.asset_id!r}; "
                    "rebuild index"
                )
            if m.asset_id in by_asset_id:
                raise AssetRegistryError(
                    f"Duplicate asset_id {m.asset_id!r} found for uuids "
                    f"{by_asset_id[m.asset_id]!r} and {m.uuid!r}; "
                    "rebuild index"
                )

            metas[m.uuid] = m
            by_asset_id[m.asset_id] = m.uuid
            by_category.setdefault(m.category, []).append(m.uuid)
            by_super.setdefault(m.super_category, []).append(m.uuid)
            for tag in m.tags:
                by_tag.setdefault(tag, []).append(m.uuid)

        # Assign atomically — only reached if all rows parsed successfully.
        self._metas = metas
        self._by_asset_id = by_asset_id
        self._by_category = by_category
        self._by_super = by_super
        self._by_tag = by_tag
        logger.info("loaded %d assets from %s", len(self._metas), index_path)

    @property
    def index_path(self) -> Path:
        """Filesystem path of the parquet index this registry loaded from."""
        return self._index_path

    # ---- core lookups ----
    def get_meta(self, uuid: str) -> AssetMeta:
        try:
            return self._metas[uuid]
        except KeyError:
            close = difflib.get_close_matches(
                uuid, list(self._metas.keys()), n=3
            )
            raise UnknownAssetError(uuid, closest_matches=close)

    def has(self, uuid: str) -> bool:
        return uuid in self._metas

    def __contains__(self, uuid: object) -> bool:
        return isinstance(uuid, str) and uuid in self._metas

    def resolve_asset_id(self, asset_id: str) -> str:
        try:
            return self._by_asset_id[asset_id]
        except KeyError:
            close = difflib.get_close_matches(
                asset_id, list(self._by_asset_id.keys()), n=3
            )
            raise UnknownAssetError(asset_id, closest_matches=close)

    def get_by_asset_id(self, asset_id: str) -> AssetMeta:
        return self.get_meta(self.resolve_asset_id(asset_id))

    def __len__(self) -> int:
        return len(self._metas)

    def __iter__(self) -> Iterator[AssetMeta]:
        return iter(self._metas.values())

    # ---- index accessors ----
    def all_categories(self) -> list[str]:
        return sorted(self._by_category.keys())

    def all_super_categories(self) -> list[str]:
        return sorted(self._by_super.keys())

    def categories_in(self, super_category: str) -> list[str]:
        cats = {
            self._metas[u].category
            for u in self._by_super.get(super_category, [])
        }
        return sorted(cats)

    # ---- to be implemented in later tasks ----
    def query(self, asset_filter: AssetFilter) -> list[AssetMeta]:
        """Return all metas matching asset_filter, sorted by uuid."""
        matches = [m for m in self._metas.values() if asset_filter.matches(m)]
        matches.sort(key=lambda m: m.uuid)
        return matches

    def build_spec(
        self,
        meta: AssetMeta,
        *,
        name: str | None = None,
        role: str,
    ) -> "RigidObjectSpec":
        """Convert AssetMeta into a RigidObjectSpec.

        The spec carries asset-library identity: name (scene/prim name),
        USD + interaction paths, and ``mass`` sourced from the URDF's
        ``<extra_info>`` (a declared asset-library property, not a
        runtime override). Pose (``initial_pos`` / ``initial_rot``) is
        intentionally left unset; runtime placement is owned by
        downstream modules (pose-reset events, etc.). Scene-role
        semantics are injected by the caller via ``role`` and stored as
        ``RigidObjectSpec.actor_type``.

        Imports RigidObjectSpec lazily to keep isaaclab out of the
        registry package's import graph for lightweight callers
        (CLI, tests that don't need specs).
        """
        from robo_orchard_sim.orchard_env.assets import RigidObjectSpec

        return RigidObjectSpec(
            name=name if name is not None else meta.asset_id,
            usd_path=meta.usd_path,
            caption_path=meta.caption_path,
            interaction_path=meta.interaction_path,
            mass=meta.real_mass,
            uuid=meta.uuid,
            category=meta.category,
            actor_type=role,
            attributes={
                "color": tuple(sorted(meta.color or ())),
                "shape": tuple(sorted(meta.shape or ())),
                "material": tuple(sorted(meta.material or ())),
            },
            aabb_z_min=meta.aabb_z_min,
        )


# ---------------------------------------------------------------------------
# AssetSampler — stateless sampling primitives over an AssetRegistry.
# ---------------------------------------------------------------------------

# Fields allowed in DistractorSpec.match / .differ. Excludes tags
# (filtered by AssetFilter, not by match/differ), identity fields
# (uuid/asset_id), and physics floats (fragile equality).
# color/shape/material use set-overlap / set-disjoint via
# _attr_match / _attr_differ; scalar fields use ==/!=.
_MATCH_DIFFER_ALLOWED = frozenset(
    ("super_category", "category", "color", "shape", "material", "size_bucket")
)

_SET_VALUED_FIELDS = frozenset(("color", "shape", "material"))


def _attr_match(a: Any, c: Any, field: str) -> bool:
    """match=(field,) test: set any-overlap, scalar equality.

    None on either side never counts as a match.
    """
    if field in _SET_VALUED_FIELDS:
        if a is None or c is None:
            return False
        return bool(a & c)
    return a == c


def _attr_differ(a: Any, c: Any, field: str) -> bool:
    """differ=(field,) test: set disjoint, scalar inequality.

    None on either side counts as differing.
    """
    if field in _SET_VALUED_FIELDS:
        if a is None or c is None:
            return True
        return not (a & c)
    return a != c


_COMPATIBLE_PAIR_MAX_ATTEMPTS = 10


class AssetSampler:
    """Stateless composable sampling primitives.

    Holds a reference to an AssetRegistry and exposes low-level
    primitives (sample_target, sample_distractors,
    sample_compatible_pair) that upper layers (task definitions,
    benchmark runners) compose to build scenarios.

    All sampling methods accept an np.random.Generator so callers
    can control reproducibility precisely.
    """

    def __init__(self, registry: AssetRegistry) -> None:
        self._reg = registry

    def sample_target(
        self,
        asset_filter: AssetFilter,
        rng: np.random.Generator,
    ) -> AssetMeta:
        """Uniform-random draw from the pool matching asset_filter.

        Raises:
            EmptyPoolError: the pool is empty.
        """
        pool = self._reg.query(asset_filter)
        if not pool:
            raise EmptyPoolError(filter_repr=repr(asset_filter))
        idx = int(rng.integers(0, len(pool)))
        return pool[idx]

    def sample_target_pool(
        self,
        asset_filter: AssetFilter,
        k: int,
        rng: np.random.Generator,
    ) -> list[AssetMeta]:
        """Uniform-random draw of k distinct AssetMeta from matching pool.

        Raises:
            ValueError: k is non-positive.
            InsufficientPoolError: matching pool has fewer than k assets.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        pool = self._reg.query(asset_filter)
        if len(pool) < k:
            raise InsufficientPoolError(
                mode="target_pool",
                available=len(pool),
                requested=k,
            )
        idxs = rng.choice(len(pool), size=k, replace=False)
        return [pool[int(i)] for i in idxs]

    def sample_distractors(
        self,
        anchor: AssetMeta,
        spec: DistractorSpec,
        rng: np.random.Generator,
    ) -> list[AssetMeta]:
        """Sample distractors relative to an anchor asset.

        Pool is built by: (1) assets whose listed ``spec.match`` fields
        are compatible with the anchor (equality for scalar fields;
        set-overlap for color/shape/material), (2) and whose listed
        ``spec.differ`` fields are incompatible (inequality for scalar;
        set-disjoint for color/shape/material), (3) and which passes
        ``spec.absolute_filter``. The anchor itself is always excluded.

        Raises:
            InsufficientPoolError: the pool has fewer unique assets
                than spec.count.
            ValueError: match/differ references an unknown AssetMeta
                field (see ``_MATCH_DIFFER_ALLOWED``).
        """
        invalid = [
            f
            for f in tuple(spec.match) + tuple(spec.differ)
            if f not in _MATCH_DIFFER_ALLOWED
        ]
        if invalid:
            raise ValueError(
                f"Unknown match/differ field(s): {invalid}. "
                f"Allowed: {sorted(_MATCH_DIFFER_ALLOWED)}"
            )

        pool: list[AssetMeta] = []
        for m in self._reg:
            if m.uuid == anchor.uuid:
                continue
            if not all(
                _attr_match(getattr(anchor, f), getattr(m, f), f)
                for f in spec.match
            ):
                continue
            if not all(
                _attr_differ(getattr(anchor, f), getattr(m, f), f)
                for f in spec.differ
            ):
                continue
            if not spec.absolute_filter.matches(m):
                continue
            if spec.exclude and m.uuid in spec.exclude:
                continue
            if spec.only_in is not None and m.uuid not in spec.only_in:
                continue
            pool.append(m)

        # deterministic baseline ordering
        pool.sort(key=lambda m: m.uuid)

        if len(pool) < spec.min_count:
            raise InsufficientPoolError(
                mode="distractor",
                available=len(pool),
                requested=spec.min_count,
            )

        n = min(len(pool), spec.max_count)
        if n == 0:
            return []
        idxs = rng.choice(len(pool), size=n, replace=False)
        return [pool[int(i)] for i in idxs]

    def sample_distractor_pool(
        self,
        anchor: AssetMeta,
        spec: DistractorSpec,
        pool_size: int,
        rng: np.random.Generator,
    ) -> list[AssetMeta]:
        """Uniform-random draw of pool_size distinct distractors.

        Raises:
            ValueError: pool_size is non-positive, or match/differ
                has unknown fields.
            InsufficientPoolError: matching pool has fewer than
                pool_size assets.
        """
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        invalid = [
            f
            for f in tuple(spec.match) + tuple(spec.differ)
            if f not in _MATCH_DIFFER_ALLOWED
        ]
        if invalid:
            raise ValueError(
                f"Unknown match/differ field(s): {invalid}. "
                f"Allowed: {sorted(_MATCH_DIFFER_ALLOWED)}"
            )

        pool: list[AssetMeta] = []
        for m in self._reg:
            if m.uuid == anchor.uuid:
                continue
            if not all(
                getattr(m, f) == getattr(anchor, f) for f in spec.match
            ):
                continue
            if not all(
                getattr(m, f) != getattr(anchor, f) for f in spec.differ
            ):
                continue
            if not spec.absolute_filter.matches(m):
                continue
            if spec.exclude and m.uuid in spec.exclude:
                continue
            if spec.only_in is not None and m.uuid not in spec.only_in:
                continue
            pool.append(m)

        pool.sort(key=lambda m: m.uuid)

        if len(pool) < pool_size:
            raise InsufficientPoolError(
                mode="distractor_pool",
                available=len(pool),
                requested=pool_size,
            )
        idxs = rng.choice(len(pool), size=pool_size, replace=False)
        return [pool[int(i)] for i in idxs]

    def sample_compatible_pair(
        self,
        pick_filter: AssetFilter,
        place_filter: AssetFilter,
        rng: np.random.Generator,
    ) -> tuple[AssetMeta, AssetMeta]:
        """Sample a disjoint (pick, place) pair.

        v1 strategy: independently sample a pick and a place. If the
        two draws happen to be the same asset (dual-role assets can
        satisfy both filters), retry up to _COMPATIBLE_PAIR_MAX_ATTEMPTS
        times before giving up.

        Raises:
            EmptyPoolError: either filter yields an empty pool.
            CollisionExhaustedError: no disjoint pair within
                _COMPATIBLE_PAIR_MAX_ATTEMPTS attempts. Subclass of
                EmptyPoolError.
        """
        for _ in range(_COMPATIBLE_PAIR_MAX_ATTEMPTS):
            pick = self.sample_target(pick_filter, rng)
            place = self.sample_target(place_filter, rng)
            if pick.uuid != place.uuid:
                return pick, place
        raise CollisionExhaustedError(
            pick_filter_repr=repr(pick_filter),
            place_filter_repr=repr(place_filter),
            attempts=_COMPATIBLE_PAIR_MAX_ATTEMPTS,
        )
