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

"""Count high-success graspable asset categories in an asset library."""

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


@dataclass(frozen=True)
class CategoryCountSummary:
    """Summary of categories passing the grasp-success threshold."""

    category_count: int
    asset_count: int
    categories: tuple[str, ...]
    selected_assets: tuple[dict[str, str], ...] = ()
    selected_asset_candidates: tuple[dict[str, str], ...] = ()


def _read_success_rates_by_usd_path(
    success_rate_csv_path: str,
) -> dict[str, float]:
    success_rates: dict[str, float] = {}
    with open(success_rate_csv_path, newline="", encoding="utf-8") as fr:
        reader = csv.DictReader(fr)
        if reader.fieldnames is None:
            raise ValueError(f"{success_rate_csv_path} must contain a header")
        required_fields = {"usd_path", "success"}
        missing_fields = required_fields.difference(reader.fieldnames)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"{success_rate_csv_path} missing required columns: {missing}"
            )
        for row in reader:
            usd_path = row.get("usd_path")
            if not usd_path:
                continue
            try:
                success_rates[usd_path] = float(row["success"])
            except (TypeError, ValueError):
                continue
    return success_rates


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
    safe_value = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    )
    return safe_value or "category"


def _snapshot_uuids(
    *,
    snapshot_path: str | Path | None,
    registry: AssetRegistry,
) -> frozenset[str] | None:
    if snapshot_path is None:
        return None
    return load_snapshot(Path(snapshot_path), registry).uuids


def _filtered_uuids(
    *,
    snapshot_path: str | Path | None,
    split_path: str | Path | None,
    registry: AssetRegistry,
) -> frozenset[str] | None:
    only_in_uuids = _snapshot_uuids(
        snapshot_path=snapshot_path,
        registry=registry,
    )
    if split_path is None:
        return only_in_uuids
    splits = load_asset_splits(Path(split_path), registry)
    split_uuids = splits.seen | splits.unseen_category | splits.unseen_instance
    if only_in_uuids is None:
        return split_uuids
    return only_in_uuids & split_uuids


def _query_graspable_metas(
    *,
    asset_root: str,
    graspable_tag: str,
    snapshot_path: str | Path | None,
    split_path: str | Path | None,
) -> list[Any]:
    registry = AssetRegistry(asset_root)
    only_in = _filtered_uuids(
        snapshot_path=snapshot_path,
        split_path=split_path,
        registry=registry,
    )
    if only_in is None:
        asset_filter = AssetFilter(tags=frozenset({graspable_tag}))
    else:
        asset_filter = AssetFilter(
            tags=frozenset({graspable_tag}),
            only_in=only_in,
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
    success_rates = _read_success_rates_by_usd_path(success_rate_csv_path)

    categories: set[str] = set()
    selected_assets: list[dict[str, str]] = []
    selected_asset_candidates: list[dict[str, str]] = []
    for meta in metas:
        success_rate = success_rates.get(str(meta.usd_path))
        if success_rate is None or success_rate < min_success_rate:
            continue
        categories.add(str(meta.category))
        selected_assets.append(
            {
                "uuid": str(meta.uuid),
                "asset_id": str(meta.asset_id),
            }
        )
        selected_asset_candidates.append(
            {
                "uuid": str(meta.uuid),
                "asset_id": str(meta.asset_id),
                "category": str(meta.category),
            }
        )

    sorted_categories = tuple(sorted(categories))
    return CategoryCountSummary(
        category_count=len(sorted_categories),
        asset_count=len(selected_assets),
        categories=sorted_categories,
        selected_assets=tuple(selected_assets),
        selected_asset_candidates=tuple(selected_asset_candidates),
    )


def _candidate_uuids_by_category(
    selected_asset_candidates: tuple[dict[str, str], ...],
) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {}
    for selected_asset in selected_asset_candidates:
        category = selected_asset["category"]
        candidates.setdefault(category, []).append(selected_asset["uuid"])
    return candidates


class _ValidationAssetRegistry:
    def __init__(self, registry: AssetRegistry) -> None:
        self._registry = registry

    def __getattr__(self, name: str) -> Any:
        return getattr(self._registry, name)

    def __iter__(self) -> Any:
        return iter(self._registry)

    def build_spec(self, meta: Any, *, name: str, role: str) -> Any:
        return SimpleNamespace(meta=meta, name=name, role=role)


def _validate_category_attribute_with_resolver(
    *,
    registry: AssetRegistry,
    base_payload: dict[str, Any],
    base_config_path: str | Path,
    category: str,
    attribute: str,
    target_role: str,
    candidate_uuids: list[str],
    active_snapshot: frozenset[str] | None,
) -> bool:
    for candidate_uuid in candidate_uuids:
        payload = _category_attribute_payload(
            base_payload=base_payload,
            base_config_path=base_config_path,
            category=category,
            attribute=attribute,
            target_role=target_role,
        )
        asset_configs = payload["asset_configs"]
        role_config = asset_configs[target_role]
        role_config["uuid"] = candidate_uuid
        resolver = AssetResolver(
            registry=registry,
            active_snapshot=active_snapshot,
        )
        try:
            resolver.resolve(asset_configs)
        except AssetResolutionError as exc:
            print(
                "Skipped "
                f"category={category!r} attribute={attribute!r}: "
                f"candidate uuid {candidate_uuid!r} failed resolver "
                f"validation: {exc}"
            )
            return False
    return True


def _category_attribute_payload(
    *,
    base_payload: dict[str, Any],
    base_config_path: str | Path,
    category: str,
    attribute: str,
    target_role: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    asset_configs = payload.get("asset_configs")
    if not isinstance(asset_configs, dict):
        raise ValueError(f"{base_config_path} missing asset_configs mapping")
    role_config = asset_configs.get(target_role)
    if not isinstance(role_config, dict):
        raise ValueError(
            f"{base_config_path} missing asset_configs.{target_role} mapping"
        )
    filter_config = role_config.get("filter")
    if filter_config is None:
        filter_config = {}
    if not isinstance(filter_config, dict):
        raise ValueError(
            f"{base_config_path} asset_configs.{target_role}.filter "
            "must be a mapping"
        )
    filter_config = dict(filter_config)
    filter_config["category"] = category
    role_config["filter"] = filter_config

    distractors = asset_configs.get("distractors")
    if not isinstance(distractors, dict):
        raise ValueError(
            f"{base_config_path} missing asset_configs.distractors mapping"
        )
    distractors["match"] = ["category"]
    distractors["differ"] = [attribute]
    return payload


def _category_attribute_filename(
    *,
    task: str,
    category: str,
    attribute: str,
) -> str:
    return (
        f"{_safe_path_component(task)}_"
        f"{_safe_path_component(category)}_"
        f"{_safe_path_component(attribute)}.yaml"
    )


def _write_config_list(
    *,
    config_list_path: str | Path,
    generated_paths: list[str],
) -> str:
    list_path = Path(config_list_path)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text(
        "\n".join(generated_paths) + ("\n" if generated_paths else ""),
        encoding="utf-8",
    )
    return str(list_path)


def _selected_split_payload(
    selected_assets: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": _SELECTED_SPLIT_NAME,
        "seen": list(selected_assets),
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
    selected_split_path = output_path / _DEFAULT_SELECTED_SPLIT_NAME
    with selected_split_path.open("w", encoding="utf-8") as fw:
        yaml.safe_dump(
            _selected_split_payload(summary.selected_assets),
            fw,
            sort_keys=False,
        )
    return str(selected_split_path)


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
    """Write one category task YAML for each requested attribute."""
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

    with open(base_config_path, encoding="utf-8") as fr:
        base_payload = yaml.safe_load(fr) or {}
    if not isinstance(base_payload, dict):
        raise ValueError(f"{base_config_path} must contain a YAML mapping")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    validation_registry = None
    active_snapshot = None
    candidate_uuids: dict[str, list[str]] = {}
    if validate_distractors:
        validation_registry = _ValidationAssetRegistry(
            AssetRegistry(asset_root)
        )
        selected_uuids = frozenset(
            asset["uuid"] for asset in summary.selected_asset_candidates
        )
        active_snapshot = selected_uuids or None
        candidate_uuids = _candidate_uuids_by_category(
            summary.selected_asset_candidates
        )
    generated_paths: list[str] = []
    for category in summary.categories:
        for attribute in attributes:
            if validate_distractors:
                if validation_registry is None:
                    raise RuntimeError("validation registry was not created")
                if not _validate_category_attribute_with_resolver(
                    registry=validation_registry,
                    base_payload=base_payload,
                    base_config_path=base_config_path,
                    category=category,
                    attribute=attribute,
                    target_role=target_role,
                    candidate_uuids=candidate_uuids[category],
                    active_snapshot=active_snapshot,
                ):
                    continue
            config_path = output_path / _category_attribute_filename(
                task=task,
                category=category,
                attribute=attribute,
            )
            payload = _category_attribute_payload(
                base_payload=base_payload,
                base_config_path=base_config_path,
                category=category,
                attribute=attribute,
                target_role=target_role,
            )
            with config_path.open("w", encoding="utf-8") as fw:
                yaml.safe_dump(payload, fw, sort_keys=False)
            generated_paths.append(str(config_path))

    if config_list_path is None:
        config_list_path = output_path / _DEFAULT_CONFIG_LIST_NAME
    _write_config_list(
        config_list_path=config_list_path,
        generated_paths=generated_paths,
    )
    write_selected_split_from_summary(
        output_dir=output_path,
        summary=summary,
    )
    return generated_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser()
    default_base_config = _default_base_config_path()
    parser.add_argument(
        "--asset-root",
        default=_DEFAULT_ASSET_ROOT,
        help="Asset library root.",
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
        help="Inclusive minimum success rate. Defaults to 0.7.",
    )
    parser.add_argument("--graspable-tag", default=_DEFAULT_GRASPABLE_TAG)
    parser.add_argument(
        "--snapshot",
        dest="snapshot_path",
        type=Path,
        default=None,
        help=(
            "Optional snapshot YAML; restricts generated configs to its "
            "uuid set before applying later filters."
        ),
    )
    parser.add_argument(
        "--split",
        dest="split_path",
        type=Path,
        default=None,
        help=(
            "Optional benchmark split YAML; restricts generated configs "
            "to uuids listed in the split file before success-rate "
            "filtering."
        ),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=default_base_config,
        help=f"Base task YAML. Defaults to {default_base_config}.",
    )
    parser.add_argument("--task", default=_DEFAULT_TASK)
    parser.add_argument("--target-role", default="pick")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(_DEFAULT_OUTPUT_DIR),
        help="Directory for generated category attribute YAML files.",
    )
    parser.add_argument(
        "--attributes",
        nargs="+",
        default=list(_CATEGORY_ATTRIBUTES),
        help="Attributes to generate as distractor differ fields.",
    )
    parser.add_argument(
        "--validate-distractors",
        action="store_true",
        help=(
            "Run resolver validation for each generated category/attribute "
            "and skip YAMLs whose pinned pick candidates cannot satisfy "
            "the distractor min_count."
        ),
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only print the category summary; do not write YAML files.",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Print matching category names after the summary.",
    )
    return parser


def main() -> None:
    """Run category counting from CLI arguments."""
    args = build_arg_parser().parse_args()
    summary = count_graspable_categories_by_success_rate(
        asset_root=args.asset_root,
        success_rate_csv_path=args.success_rate_csv,
        min_success_rate=args.min_success_rate,
        graspable_tag=args.graspable_tag,
        snapshot_path=args.snapshot_path,
        split_path=args.split_path,
    )
    if args.snapshot_path is not None:
        print(f"Snapshot: {args.snapshot_path}")
    if args.split_path is not None:
        print(f"Split: {args.split_path}")
    print(f"Matched graspable assets: {summary.asset_count}")
    print(
        "Categories with success_rate "
        f">= {args.min_success_rate:g}: {summary.category_count}"
    )
    if args.list_categories:
        for category in summary.categories:
            print(category)
    if not args.count_only:
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
        print(f"Generated YAML files: {len(generated_paths)}")
        print(f"Wrote list: {config_list_path}")
        print(
            "Wrote selected split: "
            f"{args.output_dir / _DEFAULT_SELECTED_SPLIT_NAME}"
        )
        for path in generated_paths:
            print(path)


if __name__ == "__main__":
    main()
