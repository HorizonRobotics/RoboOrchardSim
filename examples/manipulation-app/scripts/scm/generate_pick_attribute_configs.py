#!/usr/bin/env python3
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

"""Generate per-category / per-UUID attribute task YAML configs."""

from __future__ import annotations
import argparse
import copy
import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from robo_orchard_sim.asset_manager.registry.registry import AssetRegistry
from robo_orchard_sim.asset_manager.registry.types import AssetFilter
from robo_orchard_sim.asset_manager.resolver import (
    AssetResolutionError,
    AssetResolver,
)
from robo_orchard_sim.asset_manager.snapshot import load_snapshot
from robo_orchard_sim.asset_manager.splits import load_asset_splits

_DEFAULT_ASSET_ROOT = (
    "/horizon-bucket/robot_lab/users/chaodong.huang-labs/wuwen_assets/"
    "wuwen_supply_usd_library"
)
_DEFAULT_GRASPABLE_TAG = "is_graspable"
_DEFAULT_CONFIG_LIST_NAME = "yaml_list.txt"
_DEFAULT_SUCCESS_RATE_CSV_NAME = "grasp_success_rate.csv"
_DEFAULT_SELECTED_SPLIT_NAME = "selected_assets_split.yaml"
_SELECTED_SPLIT_NAME = "selected_assets"
_DEFAULT_MIN_SUCCESS_RATE = 0.7
_DEFAULT_TASK = "pick_category"
_DEFAULT_OUTPUT_DIR = "generated_category_attribute_configs"
_CATEGORY_ATTRIBUTES = ("color", "shape")
_ATTRIBUTE_ANCHOR = "attr"

# ---------------------------------------------------------------------------
# YAML anchor helpers — emit instruction.attribute_name and
# distractors.differ[0] as the same ``&attr`` / ``*attr`` node.
# ---------------------------------------------------------------------------


class _AttributeStr(str):
    """Marker str so the custom dumper assigns a fixed anchor to it."""

    __slots__ = ()


class _AnchorAttributeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:  # type: ignore[override]
        return (
            False
            if isinstance(data, _AttributeStr)
            else super().ignore_aliases(data)
        )

    def generate_anchor(self, node: yaml.nodes.Node) -> str:  # type: ignore[override]
        if getattr(node, "_attribute_anchor", False):
            return _ATTRIBUTE_ANCHOR
        return super().generate_anchor(node)


def _represent_attribute_str(
    dumper: yaml.SafeDumper, data: _AttributeStr
) -> yaml.ScalarNode:
    node = dumper.represent_str(str(data))
    node._attribute_anchor = True  # type: ignore[attr-defined]
    return node


_AnchorAttributeDumper.add_representer(_AttributeStr, _represent_attribute_str)


def _dump_attribute_yaml(payload: dict[str, Any], stream: Any) -> None:
    yaml.dump(
        payload,
        stream,
        Dumper=_AnchorAttributeDumper,
        sort_keys=False,
        default_flow_style=False,
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategoryCountSummary:
    """Summary of categories passing the grasp-success threshold."""

    category_count: int
    asset_count: int
    categories: tuple[str, ...]
    selected_assets: tuple[dict[str, str], ...] = ()
    selected_asset_candidates: tuple[dict[str, str], ...] = ()


# ---------------------------------------------------------------------------
# Asset querying / filtering
# ---------------------------------------------------------------------------


def _read_success_rates(csv_path: str) -> dict[str, float]:
    rates: dict[str, float] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} must contain a header")
        missing = {"usd_path", "success"}.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path} missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            usd = row.get("usd_path")
            if not usd:
                continue
            try:
                rates[usd] = float(row["success"])
            except (TypeError, ValueError):
                continue
    return rates


def _filtered_uuids(
    *,
    snapshot_path: str | Path | None,
    split_path: str | Path | None,
    registry: AssetRegistry,
) -> frozenset[str] | None:
    only_in: frozenset[str] | None = None
    if snapshot_path is not None:
        only_in = load_snapshot(Path(snapshot_path), registry).uuids
    if split_path is None:
        return only_in
    split_uuids = load_asset_splits(Path(split_path), registry).seen
    return (only_in & split_uuids) if only_in is not None else split_uuids


def _query_graspable_metas(
    *,
    asset_root: str,
    graspable_tag: str,
    snapshot_path: str | Path | None,
    split_path: str | Path | None,
) -> list[Any]:
    registry = AssetRegistry(asset_root)
    only_in = _filtered_uuids(
        snapshot_path=snapshot_path, split_path=split_path, registry=registry
    )
    asset_filter = AssetFilter(
        tags=frozenset({graspable_tag}),
        **({"only_in": only_in} if only_in is not None else {}),
    )
    return registry.query(asset_filter)


def count_graspable_categories_by_success_rate(
    *,
    asset_root: str = _DEFAULT_ASSET_ROOT,
    success_rate_csv_path: str | None = None,
    min_success_rate: float = _DEFAULT_MIN_SUCCESS_RATE,
    graspable_tag: str = _DEFAULT_GRASPABLE_TAG,
    snapshot_path: str | Path | None = None,
    split_path: str | Path | None = None,
) -> CategoryCountSummary:
    """Count unique graspable asset categories meeting a success threshold."""
    if success_rate_csv_path is None:
        success_rate_csv_path = str(
            Path(asset_root) / _DEFAULT_SUCCESS_RATE_CSV_NAME
        )

    metas = _query_graspable_metas(
        asset_root=asset_root,
        graspable_tag=graspable_tag,
        snapshot_path=snapshot_path,
        split_path=split_path,
    )
    rates = _read_success_rates(success_rate_csv_path)

    categories: set[str] = set()
    selected_assets: list[dict[str, str]] = []
    candidates: list[dict[str, str]] = []
    for meta in metas:
        if rates.get(str(meta.usd_path), -1.0) < min_success_rate:
            continue
        categories.add(str(meta.category))
        rec = {"uuid": str(meta.uuid), "asset_id": str(meta.asset_id)}
        selected_assets.append(rec)
        candidates.append({**rec, "category": str(meta.category)})

    sorted_cats = tuple(sorted(categories))
    return CategoryCountSummary(
        category_count=len(sorted_cats),
        asset_count=len(selected_assets),
        categories=sorted_cats,
        selected_assets=tuple(selected_assets),
        selected_asset_candidates=tuple(candidates),
    )


def _records_by_category(
    candidates: tuple[dict[str, str], ...],
) -> dict[str, list[dict[str, str]]]:
    """Group {uuid, asset_id} records by category."""
    result: dict[str, list[dict[str, str]]] = {}
    for c in candidates:
        result.setdefault(c["category"], []).append(
            {"uuid": c["uuid"], "asset_id": c["asset_id"]}
        )
    return result


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _default_base_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return (
        repo_root
        / "robo_orchard_sim"
        / "task_suite"
        / "manipulation"
        / "semantic_pick"
        / "configs"
        / "pick_category.yaml"
    )


def _safe_path_component(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in value)
    return safe or "category"


def _build_attribute_payload(
    *,
    base_payload: dict[str, Any],
    base_config_path: str | Path,
    attribute: str,
    target_role: str,
    category: str | None = None,
    uuid: str | None = None,
) -> dict[str, Any]:
    """Build a task YAML payload for the given attribute.

    Exactly one of ``category`` or ``uuid`` must be provided:
    - ``category``: filter mode.
      ``asset_configs.<target_role>.filter.category`` is set; the resolver
      picks a matching asset at runtime.
    - ``uuid``: pinned mode.
      ``asset_configs.<target_role>`` is replaced with a minimal dict keyed
      only by ``uuid`` (``filter``/``split``/``pool_size`` are stripped).

    In both modes ``distractors.differ`` and ``instruction.attribute_name``
    are set to the same ``_AttributeStr`` node so they share the YAML
    anchor ``&attr`` / ``*attr`` in the serialised file.
    """
    if (category is None) == (uuid is None):
        raise ValueError("Exactly one of category or uuid must be provided")

    payload = copy.deepcopy(base_payload)
    asset_configs = payload.get("asset_configs")
    if not isinstance(asset_configs, dict):
        raise ValueError(f"{base_config_path} missing asset_configs mapping")
    role_config = asset_configs.get(target_role)
    if not isinstance(role_config, dict):
        raise ValueError(
            f"{base_config_path} missing asset_configs.{target_role} mapping"
        )

    if uuid is not None:
        # Pinned mode: keep non-filter fields, set uuid.
        asset_configs[target_role] = {
            k: v
            for k, v in role_config.items()
            if k not in {"filter", "split", "pool_size"}
        }
        asset_configs[target_role]["uuid"] = uuid
    else:
        # Filter mode: set category inside existing filter.
        filter_cfg = role_config.get("filter")
        if filter_cfg is None:
            filter_cfg = {}
        if not isinstance(filter_cfg, dict):
            raise ValueError(
                f"{base_config_path} asset_configs.{target_role}.filter "
                "must be a mapping"
            )
        role_config["filter"] = {**filter_cfg, "category": category}

    distractors = asset_configs.get("distractors")
    if not isinstance(distractors, dict):
        raise ValueError(
            f"{base_config_path} missing asset_configs.distractors mapping"
        )

    attr_node = _AttributeStr(attribute)
    distractors["match"] = ["category"]
    distractors["differ"] = [attr_node]

    instruction = payload.setdefault("instruction", {})
    if not isinstance(instruction, dict):
        raise ValueError(
            f"{base_config_path} `instruction` must be a mapping if present"
        )
    instruction["attribute_name"] = attr_node

    return payload


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def _attribute_filename(task: str, *parts: str, attribute: str) -> str:
    components = [_safe_path_component(task)]
    components.extend(_safe_path_component(p) for p in parts)
    components.append(_safe_path_component(attribute))
    return "_".join(components) + ".yaml"


def _per_attribute_split_filename(attribute: str) -> str:
    return f"selected_assets_split_{_safe_path_component(attribute)}.yaml"


# ---------------------------------------------------------------------------
# Split writers
# ---------------------------------------------------------------------------


def _split_payload(
    assets: tuple[dict[str, str], ...] | list[dict[str, str]],
    *,
    name: str = _SELECTED_SPLIT_NAME,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": name,
        "seen": list(assets),
        "unseen_category": [],
        "unseen_instance": [],
    }


def write_selected_split_from_summary(
    *,
    output_dir: str | Path,
    summary: CategoryCountSummary,
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    split_path = output_path / _DEFAULT_SELECTED_SPLIT_NAME
    with split_path.open("w", encoding="utf-8") as fw:
        yaml.safe_dump(
            _split_payload(summary.selected_assets), fw, sort_keys=False
        )
    return str(split_path)


def _write_per_attribute_split(
    *,
    output_dir: Path,
    attribute: str,
    asset_records: list[dict[str, str]],
) -> str:
    # Dedup by uuid while preserving insertion order.
    seen: set[str] = set()
    deduped = [
        r
        for r in asset_records
        if not (r["uuid"] in seen or seen.add(r["uuid"]))
    ]  # type: ignore[func-returns-value]
    split_path = output_dir / _per_attribute_split_filename(attribute)
    payload = _split_payload(
        deduped,
        name=f"{_SELECTED_SPLIT_NAME}_{_safe_path_component(attribute)}",
    )
    with split_path.open("w", encoding="utf-8") as fw:
        yaml.safe_dump(payload, fw, sort_keys=False)
    return str(split_path)


# ---------------------------------------------------------------------------
# Distractor validation
# ---------------------------------------------------------------------------


class _ValidationAssetRegistry:
    """Thin wrapper that overrides ``build_spec``.

    It returns a lightweight
    namespace, avoiding the heavy ``isaaclab`` import chain triggered by
    the real ``AssetRegistry.build_spec``.
    """

    def __init__(self, registry: AssetRegistry) -> None:
        self._registry = registry

    def __getattr__(self, name: str) -> Any:
        return getattr(self._registry, name)

    def __iter__(self) -> Any:
        return iter(self._registry)

    def build_spec(self, meta: Any, *, name: str, role: str) -> Any:
        return SimpleNamespace(meta=meta, name=name, role=role)


def _validate_uuids_for_attribute(
    *,
    registry: AssetRegistry,
    base_payload: dict[str, Any],
    base_config_path: str | Path,
    category: str,
    attribute: str,
    target_role: str,
    candidate_uuids: list[str],
    active_snapshot: frozenset[str] | None,
) -> list[str]:
    """Return the subset of UUIDs whose distractor pool satisfies min_count.

    Each UUID is validated independently; failures are logged and skipped
    rather than aborting the whole group.
    """
    passing: list[str] = []
    for uuid in candidate_uuids:
        payload = _build_attribute_payload(
            base_payload=base_payload,
            base_config_path=base_config_path,
            attribute=attribute,
            target_role=target_role,
            category=category,
        )
        payload["asset_configs"][target_role]["uuid"] = uuid
        try:
            AssetResolver(
                registry=registry, active_snapshot=active_snapshot
            ).resolve(payload["asset_configs"])
        except AssetResolutionError as exc:
            print(
                f"Skipped category={category!r} attribute={attribute!r}: "
                f"uuid {uuid!r} failed distractor validation: {exc}"
            )
            continue
        passing.append(uuid)
    return passing


# ---------------------------------------------------------------------------
# Config list writer
# ---------------------------------------------------------------------------


def _write_config_list(
    *, config_list_path: str | Path, generated_paths: list[str]
) -> str:
    list_path = Path(config_list_path)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text(
        "\n".join(generated_paths) + ("\n" if generated_paths else ""),
        encoding="utf-8",
    )
    return str(list_path)


# ---------------------------------------------------------------------------
# Main writer
# ---------------------------------------------------------------------------


def write_category_attribute_configs(
    *,
    asset_root: str = _DEFAULT_ASSET_ROOT,
    snapshot_path: str | Path | None = None,
    split_path: str | Path | None = None,
    base_config_path: str | Path | None = None,
    output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
    config_list_path: str | Path | None = None,
    success_rate_csv_path: str | None = None,
    min_success_rate: float = _DEFAULT_MIN_SUCCESS_RATE,
    graspable_tag: str = _DEFAULT_GRASPABLE_TAG,
    task: str = _DEFAULT_TASK,
    target_role: str = "pick",
    attributes: tuple[str, ...] = _CATEGORY_ATTRIBUTES,
    validate_distractors: bool = False,
) -> list[str]:
    """Write one task YAML per (category, attribute) or per (uuid, attribute).

    When ``validate_distractors=False`` (default): one YAML per
    ``(category, attribute)`` using ``filter.category`` — the resolver
    picks a matching asset at runtime.

    When ``validate_distractors=True``: each candidate UUID is validated
    independently; passing UUIDs get their own pinned YAML (``uuid`` only,
    no category filter). Per-attribute split files are also written.
    """
    if base_config_path is None:
        base_config_path = _default_base_config_path()

    summary = count_graspable_categories_by_success_rate(
        asset_root=asset_root,
        success_rate_csv_path=success_rate_csv_path,
        min_success_rate=min_success_rate,
        graspable_tag=graspable_tag,
        snapshot_path=snapshot_path,
        split_path=split_path,
    )

    with open(base_config_path, encoding="utf-8") as fh:
        base_payload = yaml.safe_load(fh) or {}
    if not isinstance(base_payload, dict):
        raise ValueError(f"{base_config_path} must contain a YAML mapping")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_paths: list[str] = []

    if validate_distractors:
        registry = _ValidationAssetRegistry(AssetRegistry(asset_root))
        active_snapshot: frozenset[str] | None = (
            frozenset(a["uuid"] for a in summary.selected_asset_candidates)
            or None
        )
        records_by_cat = _records_by_category(
            summary.selected_asset_candidates
        )
        per_attr_passing: dict[str, list[dict[str, str]]] = {
            a: [] for a in attributes
        }

        for category in summary.categories:
            records = records_by_cat.get(category, [])
            for attribute in attributes:
                passing = _validate_uuids_for_attribute(
                    registry=registry,
                    base_payload=base_payload,
                    base_config_path=base_config_path,
                    category=category,
                    attribute=attribute,
                    target_role=target_role,
                    candidate_uuids=[r["uuid"] for r in records],
                    active_snapshot=active_snapshot,
                )
                if not passing:
                    continue
                id_by_uuid = {r["uuid"]: r["asset_id"] for r in records}
                for uuid in passing:
                    asset_id = id_by_uuid.get(uuid, "asset")
                    config_path = output_path / _attribute_filename(
                        task, category, asset_id, uuid[:8], attribute=attribute
                    )
                    payload = _build_attribute_payload(
                        base_payload=base_payload,
                        base_config_path=base_config_path,
                        attribute=attribute,
                        target_role=target_role,
                        uuid=uuid,
                    )
                    with config_path.open("w", encoding="utf-8") as fw:
                        _dump_attribute_yaml(payload, fw)
                    generated_paths.append(str(config_path))
                    per_attr_passing[attribute].append(
                        {"uuid": uuid, "asset_id": asset_id}
                    )

        for attribute in attributes:
            _write_per_attribute_split(
                output_dir=output_path,
                attribute=attribute,
                asset_records=per_attr_passing[attribute],
            )
    else:
        for category in summary.categories:
            for attribute in attributes:
                config_path = output_path / _attribute_filename(
                    task, category, attribute=attribute
                )
                payload = _build_attribute_payload(
                    base_payload=base_payload,
                    base_config_path=base_config_path,
                    attribute=attribute,
                    target_role=target_role,
                    category=category,
                )
                with config_path.open("w", encoding="utf-8") as fw:
                    _dump_attribute_yaml(payload, fw)
                generated_paths.append(str(config_path))

    if config_list_path is None:
        config_list_path = output_path / _DEFAULT_CONFIG_LIST_NAME
    _write_config_list(
        config_list_path=config_list_path, generated_paths=generated_paths
    )
    write_selected_split_from_summary(output_dir=output_path, summary=summary)
    return generated_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    default_base = _default_base_config_path()
    parser.add_argument(
        "--asset-root", default=_DEFAULT_ASSET_ROOT, help="Asset library root."
    )
    parser.add_argument(
        "--success-rate-csv",
        default=None,
        help=(
            "Grasp success CSV. Defaults to "
            f"<asset-root>/{_DEFAULT_SUCCESS_RATE_CSV_NAME}."
        ),
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=_DEFAULT_MIN_SUCCESS_RATE,
        help="Inclusive minimum grasp success rate (default: %(default)s).",
    )
    parser.add_argument("--graspable-tag", default=_DEFAULT_GRASPABLE_TAG)
    parser.add_argument(
        "--snapshot",
        dest="snapshot_path",
        type=Path,
        default=None,
        help="Snapshot YAML; restricts to its UUID set before other filters.",
    )
    parser.add_argument(
        "--split",
        dest="split_path",
        type=Path,
        default=None,
        help=(
            "Benchmark split YAML; restricts to `seen` UUIDs only "
            "(`unseen_category`/`unseen_instance` are excluded)."
        ),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=default_base,
        help=f"Base task YAML (default: {default_base}).",
    )
    parser.add_argument("--task", default=_DEFAULT_TASK)
    parser.add_argument("--target-role", default="pick")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(_DEFAULT_OUTPUT_DIR),
        help="Output directory for generated YAML files.",
    )
    parser.add_argument(
        "--attributes",
        nargs="+",
        default=list(_CATEGORY_ATTRIBUTES),
        help="Distractor differ fields to generate (default: color shape).",
    )
    parser.add_argument(
        "--validate-distractors",
        action="store_true",
        help=(
            "Validate each UUID × attribute independently. "
            "Passing UUIDs get a pinned YAML (uuid only, no category filter); "
            "failing UUIDs are dropped. "
            "Also writes selected_assets_split_<attribute>.yaml per attribute."
        ),
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Print the category summary only; do not write YAML files.",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Print matching category names after the summary.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = count_graspable_categories_by_success_rate(
        asset_root=args.asset_root,
        success_rate_csv_path=args.success_rate_csv,
        min_success_rate=args.min_success_rate,
        graspable_tag=args.graspable_tag,
        snapshot_path=args.snapshot_path,
        split_path=args.split_path,
    )
    if args.snapshot_path:
        print(f"Snapshot : {args.snapshot_path}")
    if args.split_path:
        print(f"Split    : {args.split_path}")
    print(f"Matched graspable assets : {summary.asset_count}")
    print(
        f"Categories (success >= {args.min_success_rate:g}) : "
        f"{summary.category_count}"
    )
    if args.list_categories:
        for cat in summary.categories:
            print(cat)
    if args.count_only:
        return
    config_list_path = args.output_dir / _DEFAULT_CONFIG_LIST_NAME
    generated_paths = write_category_attribute_configs(
        asset_root=args.asset_root,
        snapshot_path=args.snapshot_path,
        split_path=args.split_path,
        base_config_path=args.base_config,
        output_dir=args.output_dir,
        config_list_path=config_list_path,
        success_rate_csv_path=args.success_rate_csv,
        min_success_rate=args.min_success_rate,
        graspable_tag=args.graspable_tag,
        task=args.task,
        target_role=args.target_role,
        attributes=tuple(args.attributes),
        validate_distractors=args.validate_distractors,
    )
    print(f"Generated YAML files : {len(generated_paths)}")
    print(f"Wrote list           : {config_list_path}")
    selected_split_path = args.output_dir / _DEFAULT_SELECTED_SPLIT_NAME
    print(f"Wrote selected split : {selected_split_path}")
    if args.validate_distractors:
        for attr in args.attributes:
            split_path = args.output_dir / _per_attribute_split_filename(attr)
            print(f"Wrote per-attr split ({attr}) : {split_path}")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
