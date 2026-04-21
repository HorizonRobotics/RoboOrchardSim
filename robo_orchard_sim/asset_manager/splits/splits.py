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

"""AssetSplits dataclass + YAML loader."""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from robo_orchard_sim.asset_manager.registry.errors import UnknownAssetError
from robo_orchard_sim.asset_manager.registry.registry import AssetRegistry
from robo_orchard_sim.asset_manager.splits.errors import (
    DuplicateAssetIdInSplitError,
    EmptySeenSplitError,
    InvalidSplitsYamlError,
    OverlappingSplitsError,
    UnknownAssetIdError,
    UnsupportedSchemaVersionError,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# AssetSplits — immutable uuid-keyed partitioning for a single benchmark.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetSplits:
    """Asset partitioning for a single benchmark.

    All split fields are frozensets of asset uuids (not asset_ids).
    The asset_id -> uuid resolution is done by load_asset_splits at
    load time; this dataclass does not know about the registry.
    """

    name: str
    seen: frozenset[str]
    unseen_category: frozenset[str] = field(default_factory=frozenset)
    unseen_instance: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.seen:
            raise EmptySeenSplitError(self.name)

        overlap_sc = self.seen & self.unseen_category
        overlap_si = self.seen & self.unseen_instance
        overlap_ci = self.unseen_category & self.unseen_instance
        if overlap_sc or overlap_si or overlap_ci:
            raise OverlappingSplitsError(
                name=self.name,
                seen_vs_unseen_category=overlap_sc,
                seen_vs_unseen_instance=overlap_si,
                unseen_category_vs_unseen_instance=overlap_ci,
            )


# ---------------------------------------------------------------------------
# load_asset_splits — YAML -> resolved AssetSplits.
# ---------------------------------------------------------------------------

_KNOWN_KEYS = frozenset(
    {"schema_version", "name", "seen", "unseen_category", "unseen_instance"}
)
_REQUIRED_KEYS = frozenset({"schema_version", "name", "seen"})
_SPLIT_KEYS = ("seen", "unseen_category", "unseen_instance")


def load_asset_splits(
    yaml_path: Path,
    registry: AssetRegistry,
    *,
    strict: bool = True,
) -> AssetSplits:
    """Load a benchmark splits YAML and resolve asset_ids to uuids.

    Args:
        yaml_path: Path to the splits YAML file.
        registry: Asset registry used for asset_id -> uuid resolution.
        strict: When True (default), unknown asset_ids raise
            UnknownAssetIdError. When False, unknown ids are skipped
            with a warning log.

    Returns:
        A validated AssetSplits instance.
    """
    raw = _read_yaml(yaml_path)
    _validate_structure(raw)

    name: str = raw["name"]
    resolved: dict[str, frozenset[str]] = {}
    all_unknown: list[str] = []

    for key in _SPLIT_KEYS:
        ids: list[str] = raw.get(key, [])
        _check_duplicates(key, ids)
        uuids, unknown = _resolve_ids(ids, registry)
        all_unknown.extend(unknown)
        resolved[key] = frozenset(uuids)

    if all_unknown and strict:
        raise UnknownAssetIdError(tuple(all_unknown))
    for uid in all_unknown:
        logger.warning("Skipping unknown asset_id '%s' (strict=False)", uid)

    return AssetSplits(
        name=name,
        seen=resolved["seen"],
        unseen_category=resolved["unseen_category"],
        unseen_instance=resolved["unseen_instance"],
    )


def _read_yaml(yaml_path: Path) -> dict:
    """Read and parse the YAML file, returning the top-level dict."""
    try:
        text = yaml_path.read_text()
    except (OSError, FileNotFoundError) as exc:
        raise InvalidSplitsYamlError(
            f"Cannot read splits file: {yaml_path}"
        ) from exc

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise InvalidSplitsYamlError(
            f"Invalid YAML syntax in {yaml_path}"
        ) from exc

    if not isinstance(data, dict):
        raise InvalidSplitsYamlError(
            f"Expected a YAML mapping at top level, got {type(data).__name__}"
        )
    return data


def _validate_structure(raw: dict) -> None:
    """Validate top-level keys, schema version, and required fields."""
    unknown_keys = set(raw.keys()) - _KNOWN_KEYS
    if unknown_keys:
        raise InvalidSplitsYamlError(
            f"Unknown top-level keys: {sorted(unknown_keys)}"
        )

    missing = _REQUIRED_KEYS - set(raw.keys())
    if missing:
        raise InvalidSplitsYamlError(
            f"Missing required keys: {sorted(missing)}"
        )

    version = raw["schema_version"]
    if not isinstance(version, int):
        raise InvalidSplitsYamlError(
            f"schema_version must be an integer, got {type(version).__name__}"
        )
    if version != SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            found=version, expected=SCHEMA_VERSION
        )

    if not isinstance(raw["name"], str):
        raise InvalidSplitsYamlError(
            f"'name' must be a string, got {type(raw['name']).__name__}"
        )

    for key in _SPLIT_KEYS:
        val = raw.get(key)
        if val is None:
            continue
        if not isinstance(val, list):
            raise InvalidSplitsYamlError(
                f"'{key}' must be a list, got {type(val).__name__}"
            )
        for i, item in enumerate(val):
            if not isinstance(item, str):
                raise InvalidSplitsYamlError(
                    f"'{key}[{i}]' must be a string, "
                    f"got {type(item).__name__}: {item!r}"
                )


def _check_duplicates(split_name: str, ids: list[str]) -> None:
    """Raise on duplicate asset_ids within a single split list."""
    seen: set[str] = set()
    for aid in ids:
        if aid in seen:
            raise DuplicateAssetIdInSplitError(split_name, aid)
        seen.add(aid)


def _resolve_ids(
    ids: list[str],
    registry: AssetRegistry,
) -> tuple[list[str], list[str]]:
    """Resolve asset_ids to uuids, returning (resolved, unknown) lists."""
    resolved: list[str] = []
    unknown: list[str] = []
    for aid in ids:
        try:
            resolved.append(registry.resolve_asset_id(aid))
        except UnknownAssetError:
            unknown.append(aid)
    return resolved, unknown
