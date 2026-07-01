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

"""Compose snapshots via set operations (union, intersect, diff)."""

from __future__ import annotations
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from robo_orchard_sim.asset_manager.snapshot.errors import (
    EmptyComposeResultError,
)
from robo_orchard_sim.asset_manager.snapshot.snapshot import (
    SCHEMA_VERSION,
    Snapshot,
)

logger = logging.getLogger(__name__)

ComposeOp = Literal["union", "intersect", "diff"]


def compose_snapshots(
    snapshots: list[Snapshot],
    op: ComposeOp,
    name: str,
    description: str,
    *,
    parent_paths: list[Path],
    created_by: str | None = None,
) -> Snapshot:
    """Derive a snapshot via union/intersect/diff."""
    if op not in ("union", "intersect", "diff"):
        raise ValueError(
            f"Unknown op '{op}'; must be one of union, intersect, diff"
        )
    if op == "diff":
        if len(snapshots) != 2:
            raise ValueError(
                f"op 'diff' requires exactly 2 inputs, got {len(snapshots)}"
            )
    else:
        if len(snapshots) < 2:
            raise ValueError(
                f"op '{op}' requires at least 2 inputs, got {len(snapshots)}"
            )
    if len(parent_paths) != len(snapshots):
        raise ValueError(
            f"parent_paths length {len(parent_paths)} != "
            f"snapshots length {len(snapshots)}"
        )

    if op == "union":
        result_uuids: frozenset[str] = frozenset()
        for s in snapshots:
            result_uuids = result_uuids | s.uuids
    elif op == "intersect":
        result_uuids = snapshots[0].uuids
        for s in snapshots[1:]:
            result_uuids = result_uuids & s.uuids
    else:  # diff
        result_uuids = snapshots[0].uuids - snapshots[1].uuids

    if not result_uuids:
        raise EmptyComposeResultError(
            op=op,
            input_names=tuple(s.name for s in snapshots),
        )

    entries_by_uuid: dict[str, dict] = {}
    for s in snapshots:
        for entry in s.metadata.get("assets", []):
            entries_by_uuid.setdefault(entry["uuid"], entry)
    assets = [
        entries_by_uuid.get(uuid, {"uuid": uuid, "asset_id": "<unknown>"})
        for uuid in sorted(result_uuids)
    ]

    inputs_checksums: dict[str, str] = {}
    for parent in parent_paths:
        try:
            data = parent.read_bytes()
        except (OSError, FileNotFoundError) as exc:
            logger.warning(
                "Skipping checksum for parent '%s' (unreadable: %s); "
                "provenance will be incomplete",
                parent,
                exc,
            )
            continue
        inputs_checksums[str(parent)] = (
            "sha256:" + hashlib.sha256(data).hexdigest()
        )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(tz=timezone.utc).isoformat(
            timespec="seconds"
        ),
        "description": description,
        "derived_from": {
            "op": op,
            "inputs": [s.name for s in snapshots],
        },
        "parent_snapshots": [str(p) for p in parent_paths],
        "inputs_checksums": inputs_checksums,
        "assets": assets,
    }
    if created_by is not None:
        metadata["created_by"] = created_by

    return Snapshot(name=name, uuids=result_uuids, metadata=metadata)
