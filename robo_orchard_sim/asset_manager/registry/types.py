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

"""AssetMeta and AssetFilter: canonical per-asset metadata + filter DSL."""

from __future__ import annotations
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# AssetMeta — immutable per-asset record loaded from the parquet index.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetMeta:
    """Immutable, hashable metadata record for a single asset."""

    # Identity
    uuid: str
    asset_id: str
    relative_path: str

    # Taxonomy
    domain: str
    super_category: str
    category: str
    name: str
    description: str

    # Attributes (optional — may be None for old assets)
    color: str | None
    shape: str | None
    material: str | None

    # Physics
    real_height: float
    real_mass: float
    min_height: float
    max_height: float
    min_mass: float
    max_mass: float

    # Paths (absolute)
    usd_path: str
    urdf_path: str
    interaction_path: str
    caption_path: str

    # Open-ended capability tags derived from URDF <extra_info><tags>.
    # Callers filter by issubset (AssetFilter.tags).
    tags: frozenset[str] = field(default_factory=frozenset)

    # Provenance
    version: str = ""
    generate_time: str = ""

    @property
    def size_bucket(self) -> str:
        """Bucket height into small/medium/large (thresholds hardcoded)."""
        if self.real_height < 0.05:
            return "small"
        if self.real_height < 0.12:
            return "medium"
        return "large"


# ---------------------------------------------------------------------------
# AssetFilter — composable filter applied against AssetMeta.
# DistractorSpec — arguments for AssetSampler.sample_distractors.
# ---------------------------------------------------------------------------


@dataclass
class AssetFilter:
    """Composable filter applied against an AssetMeta.

    ``tags`` is an AND-match against ``meta.tags``: an asset is kept only
    if every tag in ``self.tags`` appears in the asset's tag set. An
    empty ``tags`` frozenset means "no tag constraint".
    """

    tags: frozenset[str] = field(default_factory=frozenset)
    super_category: str | None = None
    category: str | None = None
    color: str | None = None
    shape: str | None = None
    material: str | None = None
    size_bucket: str | None = None

    # benchmark-injected uuid sets
    only_in: frozenset[str] | None = None
    exclude: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Accept list/tuple/set for convenience; normalize to frozenset.
        if not isinstance(self.tags, frozenset):
            object.__setattr__(self, "tags", frozenset(self.tags))

    def __repr__(self) -> str:
        """Compact repr — only show non-default fields."""
        parts = []
        if self.tags:
            parts.append(f"tags={sorted(self.tags)}")
        for name in (
            "super_category",
            "category",
            "color",
            "shape",
            "material",
            "size_bucket",
        ):
            val = getattr(self, name)
            if val is not None:
                parts.append(f"{name}={val!r}")
        if self.only_in is not None:
            parts.append(f"only_in=<{len(self.only_in)} uuids>")
        if self.exclude:
            parts.append(f"exclude=<{len(self.exclude)} uuids>")
        return f"AssetFilter({', '.join(parts)})"

    def matches(self, meta: AssetMeta) -> bool:
        if self.tags and not self.tags.issubset(meta.tags):
            return False
        if (
            self.super_category is not None
            and meta.super_category != self.super_category
        ):
            return False
        if self.category is not None and meta.category != self.category:
            return False
        if self.color is not None and meta.color != self.color:
            return False
        if self.shape is not None and meta.shape != self.shape:
            return False
        if self.material is not None and meta.material != self.material:
            return False
        if (
            self.size_bucket is not None
            and meta.size_bucket != self.size_bucket
        ):
            return False
        if self.only_in is not None and meta.uuid not in self.only_in:
            return False
        if meta.uuid in self.exclude:
            return False
        return True


@dataclass
class DistractorSpec:
    """Arguments for AssetSampler.sample_distractors.

    Distractor sampling is declarative: the caller specifies which AssetMeta
    fields must ``match`` the anchor's value, which must ``differ`` from
    the anchor's value, and an optional ``absolute_filter`` of hard
    constraints that don't depend on the anchor.

    Count is expressed as a range ``[min_count, max_count]``: fewer than
    ``min_count`` matches raises ``InsufficientPoolError``; otherwise
    ``min(len(pool), max_count)`` distractors are sampled.
    """

    min_count: int
    max_count: int
    match: tuple[str, ...] = ()
    differ: tuple[str, ...] = ()
    absolute_filter: AssetFilter = field(default_factory=AssetFilter)
    only_in: frozenset[str] | None = None
    exclude: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.min_count < 0:
            raise ValueError(
                f"DistractorSpec.min_count must be >= 0, got {self.min_count}"
            )
        if self.max_count < self.min_count:
            raise ValueError(
                f"DistractorSpec requires min_count <= max_count, "
                f"got min_count={self.min_count}, max_count={self.max_count}"
            )
