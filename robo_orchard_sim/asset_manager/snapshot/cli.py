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

"""snapshot CLI: dump-registry / compose / inspect subcommands."""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from robo_orchard_sim.asset_manager.registry.registry import AssetRegistry
from robo_orchard_sim.asset_manager.snapshot.compose import compose_snapshots
from robo_orchard_sim.asset_manager.snapshot.errors import (
    EmptyComposeResultError,
    SnapshotError,
)
from robo_orchard_sim.asset_manager.snapshot.snapshot import (
    from_registry,
    load_snapshot,
    save_snapshot,
)


def _default_output(asset_root: Path, name: str) -> Path:
    """Default snapshot output path: <asset_root>/snapshots/<name>.yaml."""
    return asset_root / "snapshots" / f"{name}.yaml"


def _cmd_dump_registry(args: argparse.Namespace) -> int:
    """Freeze the asset registry as a snapshot YAML; exit 1 if file exists."""
    asset_root = Path(args.asset_root).resolve()
    out_path = (
        Path(args.output).resolve()
        if args.output
        else _default_output(asset_root, args.name)
    )
    if out_path.exists():
        print(
            f"ERROR: output file already exists: {out_path}", file=sys.stderr
        )
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    registry = AssetRegistry(str(asset_root))
    created_by = os.environ.get("USER") or os.environ.get("LOGNAME")
    snap = from_registry(
        registry,
        name=args.name,
        description=args.description,
        asset_root=asset_root,
        created_by=created_by,
    )
    save_snapshot(out_path, snap)
    print(f"Wrote snapshot: {out_path} ({len(snap.uuids)} uuids)")
    return 0


def _infer_asset_root(input_path: Path) -> Path | None:
    """Infer asset_root as parent.parent when input is under snapshots/."""
    snapshots_dir = input_path.resolve().parent
    if snapshots_dir.name == "snapshots":
        return snapshots_dir.parent
    return None


def _cmd_compose(args: argparse.Namespace) -> int:
    """Compose snapshots via set op; refuses overwrite, exits 1 on error."""
    inputs = [Path(p).resolve() for p in args.inputs]
    if args.asset_root is not None:
        asset_root = Path(args.asset_root).resolve()
    else:
        inferred = _infer_asset_root(inputs[0])
        if inferred is None:
            print(
                "ERROR: cannot infer --asset-root from "
                f"{inputs[0]} (expected <asset_root>/snapshots/<name>.yaml);"
                " pass --asset-root explicitly.",
                file=sys.stderr,
            )
            return 1
        asset_root = inferred
    out_path = (
        Path(args.output).resolve()
        if args.output
        else inputs[0].parent / f"{args.name}.yaml"
    )
    if out_path.exists():
        print(
            f"ERROR: output file already exists: {out_path}", file=sys.stderr
        )
        return 1

    registry = AssetRegistry(str(asset_root))
    try:
        loaded = [load_snapshot(p, registry, strict=True) for p in inputs]
    except SnapshotError as exc:
        print(f"ERROR loading snapshot: {exc}", file=sys.stderr)
        return 1
    created_by = os.environ.get("USER") or os.environ.get("LOGNAME")
    try:
        snap = compose_snapshots(
            loaded,
            op=args.op,
            name=args.name,
            description=args.description,
            parent_paths=inputs,
            created_by=created_by,
        )
    except (EmptyComposeResultError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_snapshot(out_path, snap)
    print(f"Wrote snapshot: {out_path} ({len(snap.uuids)} uuids)")
    return 0


def _load_snapshot_no_registry(path: Path):
    """Load a snapshot YAML; structural validation only, no registry check."""
    from robo_orchard_sim.asset_manager.snapshot.snapshot import (
        Snapshot,
        _extract_asset_entries,
        _read_yaml,
        _validate_name_stem,
        _validate_structure,
    )

    raw = _read_yaml(path)
    _validate_structure(raw)
    _validate_name_stem(path, raw["name"])
    entries = _extract_asset_entries(raw["assets"])
    uuids = frozenset(e["uuid"] for e in entries)
    metadata = {k: v for k, v in raw.items() if k != "name"}
    return Snapshot(name=raw["name"], uuids=uuids, metadata=metadata)


def _count_unknown_uuids(path: Path, registry: AssetRegistry) -> int:
    """Re-read the snapshot to count uuids not in registry. O(N) extra work."""
    from robo_orchard_sim.asset_manager.snapshot.snapshot import (
        _extract_asset_entries,
        _read_yaml,
    )

    raw = _read_yaml(path)
    entries = _extract_asset_entries(raw["assets"])
    return sum(1 for e in entries if not registry.has(e["uuid"]))


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Print snapshot metadata (omits asset list)."""
    snap_path = Path(args.path).resolve()
    try:
        if args.asset_root is not None:
            registry = AssetRegistry(str(Path(args.asset_root).resolve()))
            snap = load_snapshot(snap_path, registry, strict=False)
            unknown_count = _count_unknown_uuids(snap_path, registry)
        else:
            snap = _load_snapshot_no_registry(snap_path)
            unknown_count = None
    except SnapshotError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    md = snap.metadata
    derived = md.get("derived_from", {})
    derived_str = derived.get("op", "<not recorded>")
    extras = {k: v for k, v in derived.items() if k != "op" and v is not None}
    if extras:
        derived_str += (
            " (" + ", ".join(f"{k}={v}" for k, v in extras.items()) + ")"
        )

    parents = md.get("parent_snapshots") or []
    parents_str = ", ".join(parents) if parents else "(none)"

    created_by = md.get("created_by") or "<not recorded>"
    print(f"Snapshot: {snap.name}")
    print(f"  Path:           {snap_path}")
    print(f"  Created:        {md['created_at']} by {created_by}")
    print(f"  Description:    {md['description']}")
    print(f"  Asset count:    {len(snap.uuids)}")
    if unknown_count is not None:
        print(f"  Unknown uuids:  {unknown_count}")
    print(f"  Derived from:   {derived_str}")
    print(f"  Parents:        {parents_str}")
    print(f"  Schema:         v{md['schema_version']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snapshot",
        description="Asset snapshot CLI: dump-registry / compose / inspect.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    dr = sub.add_parser(
        "dump-registry", help="Freeze the full asset registry as a snapshot."
    )
    dr.add_argument(
        "--asset-root", required=True, help="Path to the asset library root."
    )
    dr.add_argument(
        "--name",
        required=True,
        help="Snapshot name (must equal output file stem).",
    )
    dr.add_argument(
        "--description",
        required=True,
        help="Human-readable description (inline string).",
    )
    dr.add_argument(
        "--output",
        default=None,
        help="Output path; default <asset_root>/snapshots/<name>.yaml",
    )
    dr.set_defaults(func=_cmd_dump_registry)

    co = sub.add_parser(
        "compose",
        help="Derive a snapshot via set op (union/intersect/diff).",
    )
    co.add_argument(
        "--op",
        required=True,
        choices=["union", "intersect", "diff"],
        help="Set operation to apply.",
    )
    co.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Input snapshot YAML paths (>= 2 for union/intersect; "
            "exactly 2 for diff)."
        ),
    )
    co.add_argument(
        "--name",
        required=True,
        help="Output snapshot name (must equal output file stem).",
    )
    co.add_argument(
        "--description",
        required=True,
        help="Human-readable description.",
    )
    co.add_argument(
        "--asset-root",
        default=None,
        help=(
            "Asset root for registry; inferred from inputs[0]'s "
            "parent.parent if omitted."
        ),
    )
    co.add_argument(
        "--output",
        default=None,
        help="Output path; default <dir(inputs[0])>/<name>.yaml",
    )
    co.set_defaults(func=_cmd_compose)

    ins = sub.add_parser(
        "inspect",
        help="Print snapshot metadata (omits asset list).",
    )
    ins.add_argument("path", help="Path to snapshot YAML.")
    ins.add_argument(
        "--asset-root",
        default=None,
        help=(
            "Optional asset root for registry. When provided, uuid "
            "existence is validated against the registry; otherwise "
            "only the YAML structure is parsed."
        ),
    )
    ins.set_defaults(func=_cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
