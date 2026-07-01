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

from robo_orchard_sim.asset_manager.registry.registry import AssetRegistry
from robo_orchard_sim.asset_manager.splits.errors import (
    DuplicateUuidInSplitError,
    EmptySeenSplitError,
    InvalidSplitsYamlError,
    OverlappingSplitsError,
    UnknownUuidError,
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

    All split fields are frozensets of asset uuids. The YAML stores
    ``{uuid, asset_id}`` entries (uuid authoritative, asset_id a
    human-readable reference); load_asset_splits validates uuids against
    the registry. This dataclass does not know about the registry.
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
_ENTRY_KEYS = frozenset({"uuid", "asset_id"})


def load_asset_splits(
    yaml_path: Path,
    registry: AssetRegistry,
    *,
    strict: bool = True,
) -> AssetSplits:
    """Load a benchmark splits YAML (uuid-authoritative entries).

    Each split entry is a ``{uuid, asset_id}`` mapping. The uuid is the
    key used to populate AssetSplits; asset_id is an optional human-
    readable reference. When asset_id is present but does not match the
    registry's current asset_id for that uuid (e.g. the asset was
    renamed on disk), a warning is logged and the uuid is still used.

    Args:
        yaml_path: Path to the splits YAML file.
        registry: Asset registry used to validate uuids.
        strict: When True (default), uuids absent from the registry raise
            UnknownUuidError. When False, they are skipped with a warning.

    Returns:
        A validated AssetSplits instance.
    """
    raw = _read_yaml(yaml_path)
    _validate_structure(raw)

    name: str = raw["name"]
    resolved: dict[str, frozenset[str]] = {}
    all_unknown: list[str] = []

    for key in _SPLIT_KEYS:
        entries: list[dict] = raw.get(key, [])
        uuids = _collect_uuids(key, entries, registry, all_unknown)
        resolved[key] = frozenset(uuids)

    if all_unknown and strict:
        raise UnknownUuidError(tuple(all_unknown))
    for uid in all_unknown:
        logger.warning("Skipping unknown uuid '%s' (strict=False)", uid)

    return AssetSplits(
        name=name,
        seen=resolved["seen"],
        unseen_category=resolved["unseen_category"],
        unseen_instance=resolved["unseen_instance"],
    )


def _read_yaml(yaml_path: Path) -> dict:
    """Read and parse the YAML file, returning the top-level dict."""
    try:
        text = yaml_path.read_text(encoding="utf-8")
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
            _validate_entry(key, i, item)


def _validate_entry(split_name: str, i: int, item: object) -> None:
    """Validate a single ``{uuid, asset_id}`` split entry."""
    if not isinstance(item, dict):
        raise InvalidSplitsYamlError(
            f"'{split_name}[{i}]' must be a mapping with a 'uuid' key, "
            f"got {type(item).__name__}"
        )
    unknown_keys = set(item) - _ENTRY_KEYS
    if unknown_keys:
        raise InvalidSplitsYamlError(
            f"'{split_name}[{i}]' has unknown keys: {sorted(unknown_keys)}"
        )
    if "uuid" not in item:
        raise InvalidSplitsYamlError(
            f"'{split_name}[{i}]' missing required 'uuid'"
        )
    if not isinstance(item["uuid"], str):
        raise InvalidSplitsYamlError(
            f"'{split_name}[{i}].uuid' must be a string, "
            f"got {type(item['uuid']).__name__}"
        )
    if "asset_id" in item and not isinstance(item["asset_id"], str):
        raise InvalidSplitsYamlError(
            f"'{split_name}[{i}].asset_id' must be a string, "
            f"got {type(item['asset_id']).__name__}"
        )


def _collect_uuids(
    split_name: str,
    entries: list[dict],
    registry: AssetRegistry,
    unknown_out: list[str],
) -> list[str]:
    """Validate uuids against the registry, returning the known ones.

    Duplicate uuids raise; uuids absent from the registry are appended to
    unknown_out and skipped; a stale asset_id (present but not matching
    the registry's current asset_id for that uuid) logs a warning.
    """
    seen_uuids: set[str] = set()
    resolved: list[str] = []
    for entry in entries:
        uuid: str = entry["uuid"]
        if uuid in seen_uuids:
            raise DuplicateUuidInSplitError(split_name, uuid)
        seen_uuids.add(uuid)
        if not registry.has(uuid):
            unknown_out.append(uuid)
            continue
        asset_id = entry.get("asset_id")
        if asset_id is not None:
            current = registry.get_meta(uuid).asset_id
            if current != asset_id:
                logger.warning(
                    "Split '%s': asset_id '%s' is stale; uuid %s now "
                    "resolves to asset_id '%s'",
                    split_name,
                    asset_id,
                    uuid,
                    current,
                )
        resolved.append(uuid)
    return resolved
