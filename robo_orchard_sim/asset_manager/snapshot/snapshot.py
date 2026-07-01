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

"""Snapshot dataclass + save/load + from_registry."""

from __future__ import annotations
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from robo_orchard_sim.asset_manager.registry.errors import UnknownAssetError
from robo_orchard_sim.asset_manager.registry.registry import AssetRegistry
from robo_orchard_sim.asset_manager.snapshot.errors import (
    ChecksumMismatchError,
    DuplicateUuidInSnapshotError,
    InvalidSnapshotYamlError,
    SnapshotNameMismatchError,
    UnknownUuidInSnapshotError,
    UnsupportedSchemaVersionError,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "name",
        "created_at",
        "description",
        "derived_from",
        "assets",
    }
)
_OPTIONAL_KEYS = frozenset(
    {
        "parent_snapshots",
        "inputs_checksums",
        "created_by",
        "tool_version",
    }
)
_KNOWN_KEYS = _REQUIRED_KEYS | _OPTIONAL_KEYS


@dataclass(frozen=True)
class Snapshot:
    """Immutable uuid set with metadata. Not hashable (metadata is dict)."""

    name: str
    uuids: frozenset[str]
    metadata: dict

    __hash__ = None  # type: ignore[assignment]


def save_snapshot(path: Path, snapshot: Snapshot) -> None:
    """Serialize Snapshot to YAML (atomic via tmp file + rename)."""
    body = dict(snapshot.metadata)
    body["schema_version"] = SCHEMA_VERSION
    body["name"] = snapshot.name
    existing = {a["uuid"]: a for a in body.get("assets", [])}
    body["assets"] = [
        existing.get(uuid, {"uuid": uuid, "asset_id": "<unknown>"})
        for uuid in sorted(snapshot.uuids)
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(body, sort_keys=False))
    tmp.replace(path)


def load_snapshot(
    path: Path,
    registry: AssetRegistry,
    *,
    strict: bool = True,
) -> Snapshot:
    """Load and validate a snapshot YAML."""
    raw = _read_yaml(path)
    _validate_structure(raw)
    _validate_name_stem(path, raw["name"])
    declared_assets = _extract_asset_entries(raw["assets"])
    _validate_parent_checksums(
        raw.get("parent_snapshots", []),
        raw.get("inputs_checksums", {}),
    )
    known_uuids, unknown_uuids = _resolve_against_registry(
        declared_assets, registry
    )
    if unknown_uuids:
        if strict:
            raise UnknownUuidInSnapshotError(tuple(sorted(unknown_uuids)))
        for uid in sorted(unknown_uuids):
            logger.warning(
                "Skipping uuid '%s' (not in registry, strict=False)", uid
            )
    _warn_on_asset_id_drift(declared_assets, registry, known_uuids)

    metadata = {k: v for k, v in raw.items() if k != "name"}
    return Snapshot(
        name=raw["name"],
        uuids=frozenset(known_uuids),
        metadata=metadata,
    )


def _read_yaml(path: Path) -> dict:
    """Read and parse the YAML file, returning the top-level dict."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError) as exc:
        raise InvalidSnapshotYamlError(
            f"Cannot read snapshot: {path}"
        ) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise InvalidSnapshotYamlError(f"Invalid YAML in {path}") from exc
    if not isinstance(data, dict):
        raise InvalidSnapshotYamlError(
            f"Expected mapping at top level, got {type(data).__name__}"
        )
    return data


def _validate_structure(raw: dict) -> None:
    """Validate top-level keys, schema version, and required fields."""
    unknown = set(raw.keys()) - _KNOWN_KEYS
    if unknown:
        raise InvalidSnapshotYamlError(
            f"Unknown top-level keys: {sorted(unknown)}"
        )
    missing = _REQUIRED_KEYS - set(raw.keys())
    if missing:
        raise InvalidSnapshotYamlError(
            f"Missing required keys: {sorted(missing)}"
        )
    sv = raw["schema_version"]
    if not isinstance(sv, int):
        raise InvalidSnapshotYamlError(
            f"schema_version must be int, got {type(sv).__name__}"
        )
    if sv != SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(found=sv, expected=SCHEMA_VERSION)
    if not isinstance(raw["name"], str):
        raise InvalidSnapshotYamlError(
            f"'name' must be str, got {type(raw['name']).__name__}"
        )
    derived = raw.get("derived_from")
    if not isinstance(derived, dict) or "op" not in derived:
        raise InvalidSnapshotYamlError(
            "'derived_from' must be a mapping with at least an 'op' key"
        )


def _validate_name_stem(path: Path, yaml_name: str) -> None:
    if path.stem != yaml_name:
        raise SnapshotNameMismatchError(
            file_stem=path.stem, yaml_name=yaml_name
        )


def _extract_asset_entries(assets: list) -> list[dict]:
    """Validate shape + uniqueness; return list of dict entries."""
    if not isinstance(assets, list) or len(assets) == 0:
        raise InvalidSnapshotYamlError("'assets' must be a non-empty list")
    seen: set[str] = set()
    entries: list[dict] = []
    for i, entry in enumerate(assets):
        if not isinstance(entry, dict):
            raise InvalidSnapshotYamlError(f"assets[{i}] must be a mapping")
        if "uuid" not in entry:
            raise InvalidSnapshotYamlError(
                f"assets[{i}] missing required 'uuid'"
            )
        if "asset_id" not in entry:
            raise InvalidSnapshotYamlError(
                f"assets[{i}] missing required 'asset_id'"
            )
        uuid = entry["uuid"]
        if uuid in seen:
            raise DuplicateUuidInSnapshotError(uuid)
        seen.add(uuid)
        entries.append(entry)
    return entries


def _validate_parent_checksums(
    parents: list[str],
    checksums: dict[str, str],
) -> None:
    """Verify each parent sha256 matches the recorded inputs_checksums."""
    for parent in parents:
        if parent not in checksums:
            continue
        expected_raw = checksums[parent]
        if not expected_raw.startswith("sha256:"):
            raise InvalidSnapshotYamlError(
                f"Unsupported checksum format for '{parent}': "
                f"expected 'sha256:<hex>', got {expected_raw!r}"
            )
        expected = expected_raw.removeprefix("sha256:")
        try:
            parent_bytes = Path(parent).read_bytes()
        except (OSError, FileNotFoundError) as exc:
            raise ChecksumMismatchError(
                parent_path=parent,
                expected=expected,
                actual="<file missing or unreadable>",
            ) from exc
        actual = hashlib.sha256(parent_bytes).hexdigest()
        if actual != expected:
            raise ChecksumMismatchError(
                parent_path=parent,
                expected=expected,
                actual=actual,
            )


def _resolve_against_registry(
    entries: list[dict],
    registry: AssetRegistry,
) -> tuple[set[str], set[str]]:
    """Partition entries by whether their uuid exists in the registry."""
    known: set[str] = set()
    unknown: set[str] = set()
    for entry in entries:
        uuid = entry["uuid"]
        if registry.has(uuid):
            known.add(uuid)
        else:
            unknown.add(uuid)
    return known, unknown


def _warn_on_asset_id_drift(
    entries: list[dict],
    registry: AssetRegistry,
    known_uuids: set[str],
) -> None:
    """Log warnings when snapshot asset_id != registry's current asset_id."""
    for entry in entries:
        uuid = entry["uuid"]
        if uuid not in known_uuids:
            continue
        try:
            meta = registry.get_meta(uuid)
        except UnknownAssetError:
            continue
        if meta.asset_id != entry["asset_id"]:
            logger.warning(
                "asset_id drift for uuid '%s': snapshot says '%s', "
                "registry now says '%s'",
                uuid,
                entry["asset_id"],
                meta.asset_id,
            )


def from_registry(
    registry: AssetRegistry,
    name: str,
    description: str,
    *,
    asset_root: Path | None = None,
    created_by: str | None = None,
) -> Snapshot:
    """Build a snapshot from the full registry (in-memory; does not write)."""
    metas = list(registry)
    uuids = frozenset(m.uuid for m in metas)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(tz=timezone.utc).isoformat(
            timespec="seconds"
        ),
        "description": description,
        "derived_from": {
            "op": "registry_dump",
            "asset_root": str(asset_root)
            if asset_root is not None
            else "<unknown>",
        },
        "assets": [{"uuid": m.uuid, "asset_id": m.asset_id} for m in metas],
    }
    if created_by is not None:
        metadata["created_by"] = created_by
    return Snapshot(name=name, uuids=uuids, metadata=metadata)
