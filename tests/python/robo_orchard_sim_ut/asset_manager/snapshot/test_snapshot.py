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

"""Core coverage for asset_manager.snapshot."""

from __future__ import annotations
import hashlib
from pathlib import Path

import pytest
import yaml

from robo_orchard_sim.asset_manager.snapshot import (
    SCHEMA_VERSION,
    ChecksumMismatchError,
    EmptyComposeResultError,
    Snapshot,
    UnknownUuidInSnapshotError,
    compose_snapshots,
    from_registry,
    load_snapshot,
    save_snapshot,
)


def _yaml_body(name: str, uuids: list[str]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "created_at": "2026-05-18T10:00:00",
        "description": "test",
        "derived_from": {"op": "registry_dump"},
        "assets": [
            {"uuid": u, "asset_id": u.replace("u-", "")} for u in uuids
        ],
    }


def _snap(name: str, uuids: set[str]) -> Snapshot:
    return Snapshot(
        name=name,
        uuids=frozenset(uuids),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "created_at": "2026-05-18T00:00:00",
            "description": name,
            "derived_from": {"op": "registry_dump"},
            "assets": [{"uuid": u, "asset_id": u} for u in uuids],
        },
    )


def test_save_load_roundtrip_from_registry_preserves_uuids(
    tmp_path, mini_registry
):
    snap_in = from_registry(
        mini_registry, name="rt", description="d", asset_root=Path("/fake")
    )
    path = tmp_path / "rt.yaml"
    save_snapshot(path, snap_in)
    snap_out = load_snapshot(path, mini_registry)
    assert snap_out.name == "rt"
    assert snap_out.uuids == frozenset(
        {"u-apple-001", "u-apple-002", "u-orange-001", "u-box-001"}
    )
    assert snap_out.metadata["derived_from"]["op"] == "registry_dump"


def test_load_unknown_uuid_strict_raises_error(tmp_path, mini_registry):
    body = _yaml_body("x", ["u-apple-001", "u-does-not-exist"])
    path = tmp_path / "x.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=False))
    with pytest.raises(UnknownUuidInSnapshotError):
        load_snapshot(path, mini_registry, strict=True)


def test_load_parent_checksum_mismatch_raises_error(tmp_path, mini_registry):
    parent = tmp_path / "parent.yaml"
    parent.write_text("anything")
    body = _yaml_body("y", ["u-apple-001"])
    body["parent_snapshots"] = [str(parent)]
    body["inputs_checksums"] = {str(parent): "sha256:" + "0" * 64}
    path = tmp_path / "y.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=False))
    with pytest.raises(ChecksumMismatchError) as exc_info:
        load_snapshot(path, mini_registry)
    assert hashlib.sha256(b"anything").hexdigest() in str(exc_info.value)


def test_compose_union_returns_combined_uuids():
    a, b = _snap("a", {"u1", "u2"}), _snap("b", {"u2", "u3"})
    out = compose_snapshots(
        [a, b],
        op="union",
        name="u",
        description="d",
        parent_paths=[Path("a.yaml"), Path("b.yaml")],
    )
    assert out.uuids == frozenset({"u1", "u2", "u3"})


def test_compose_intersect_returns_common_uuids():
    a, b = _snap("a", {"u1", "u2"}), _snap("b", {"u2", "u3"})
    out = compose_snapshots(
        [a, b],
        op="intersect",
        name="i",
        description="d",
        parent_paths=[Path("a.yaml"), Path("b.yaml")],
    )
    assert out.uuids == frozenset({"u2"})


def test_compose_disjoint_intersect_raises_error():
    a, b = _snap("a", {"u1"}), _snap("b", {"u2"})
    with pytest.raises(EmptyComposeResultError):
        compose_snapshots(
            [a, b],
            op="intersect",
            name="empty",
            description="d",
            parent_paths=[Path("a.yaml"), Path("b.yaml")],
        )


def test_public_api_exports_are_importable():
    from robo_orchard_sim.asset_manager.snapshot import (
        SCHEMA_VERSION,
        ChecksumMismatchError,
        DuplicateUuidInSnapshotError,
        EmptyComposeResultError,
        InvalidSnapshotYamlError,
        Snapshot,
        SnapshotError,
        SnapshotNameMismatchError,
        UnknownUuidInSnapshotError,
        UnsupportedSchemaVersionError,
        from_registry,
        load_snapshot,
        save_snapshot,
    )

    assert SCHEMA_VERSION == 1
    assert callable(save_snapshot)
    assert callable(load_snapshot)
    assert callable(from_registry)
    for cls in (
        ChecksumMismatchError,
        DuplicateUuidInSnapshotError,
        EmptyComposeResultError,
        InvalidSnapshotYamlError,
        SnapshotNameMismatchError,
        UnknownUuidInSnapshotError,
        UnsupportedSchemaVersionError,
    ):
        assert issubclass(cls, SnapshotError)
    assert Snapshot.__name__ == "Snapshot"
