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

"""Select successful records from a cluster batch summary."""

from __future__ import annotations
import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robo_orchard_sim.asset_manager.registry.registry import AssetRegistry

LOGGER_NAME = "select_successful_records"
LOGGER = logging.getLogger(LOGGER_NAME)
DEFAULT_INPUT_PATH = "logs/multi_data_synthesis/v3"
_ASSET_ROOT_ENV = "ORCHARD_ASSET_LIBRARY"
_DEFAULT_TARGET_ROLE = "pick"


@dataclass(frozen=True)
class SelectionResult:
    """Summary of selected successful record paths and asset records."""

    selected_paths: list[str]
    selected_assets: list[dict[str, Any]]
    kept_assets: int
    removed_assets: int

    @property
    def total_assets(self) -> int:
        """Return the total number of assets considered."""
        return self.kept_assets + self.removed_assets


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _success_rate(asset: dict[str, Any]) -> float:
    try:
        return float(asset["success_rate"])
    except KeyError as exc:
        raise ValueError("asset summary missing key 'success_rate'") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("asset summary success_rate must be numeric") from exc


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_task_root(
    *,
    summary_dir: Path,
    asset: dict[str, Any],
) -> Path | None:
    task_save_root = asset.get("task_save_root")
    if not isinstance(task_save_root, str) or not task_save_root:
        return None
    return _resolve_path(summary_dir, task_save_root)


def _load_config_payload(
    *,
    summary_dir: Path,
    asset: dict[str, Any],
) -> dict[str, Any] | None:
    task_root = _resolve_task_root(summary_dir=summary_dir, asset=asset)
    config_path_value = asset.get("config_path")
    if task_root is None or not isinstance(config_path_value, str):
        return None

    config_path = task_root / "config" / Path(config_path_value).name
    if not config_path.exists():
        return None
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return None
    return payload


def _layout_path_from_config(
    *,
    summary_dir: Path,
    asset: dict[str, Any],
) -> str | None:
    payload = _load_config_payload(summary_dir=summary_dir, asset=asset)
    if payload is None:
        return None
    layout_value = payload.get("layout")
    if not isinstance(layout_value, str) or not layout_value:
        return None
    return layout_value


def _load_layout_payload(layout_path: str) -> dict[str, Any] | None:
    path = Path(layout_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _layout_relation_info(
    layout_payload: dict[str, Any],
) -> dict[str, str]:
    relation_section = layout_payload.get("relation")
    if not isinstance(relation_section, dict):
        return {}
    for key in ("initial_constraints", "spatial_constraints"):
        constraints = relation_section.get(key)
        if not isinstance(constraints, list) or not constraints:
            continue
        first = constraints[0]
        if not isinstance(first, dict):
            continue
        info: dict[str, str] = {}
        for field in ("anchor", "relation", "subject"):
            value = first.get(field)
            if isinstance(value, str):
                info[field] = value
        if info:
            return info
    return {}


def _layout_entry_extras(
    *,
    summary_dir: Path,
    asset: dict[str, Any],
) -> dict[str, str]:
    layout_path = _layout_path_from_config(
        summary_dir=summary_dir, asset=asset
    )
    if layout_path is None:
        return {}
    extras: dict[str, str] = {"layout_path": layout_path}
    layout_payload = _load_layout_payload(layout_path)
    if layout_payload is not None:
        extras.update(_layout_relation_info(layout_payload))
    return extras


def _load_asset_uuid_from_config(
    *,
    summary_dir: Path,
    asset: dict[str, Any],
    target_role: str,
) -> str | None:
    payload = _load_config_payload(summary_dir=summary_dir, asset=asset)
    if payload is None:
        return None
    asset_configs = payload.get("asset_configs")
    if not isinstance(asset_configs, dict):
        return None
    role_config = asset_configs.get(target_role)
    if not isinstance(role_config, dict):
        return None
    uuid = role_config.get("uuid")
    if not isinstance(uuid, str) or not uuid:
        return None
    return uuid


def _asset_attribute_from_config(
    *,
    summary_dir: Path,
    asset: dict[str, Any],
) -> str:
    payload = _load_config_payload(summary_dir=summary_dir, asset=asset)
    if payload is None:
        return ""
    asset_configs = payload.get("asset_configs")
    if not isinstance(asset_configs, dict):
        return ""
    distractors = asset_configs.get("distractors")
    if not isinstance(distractors, dict):
        distractors = asset_configs.get("distrcators")
    if not isinstance(distractors, dict):
        return ""
    differ = distractors.get("differ")
    if isinstance(differ, str):
        return differ
    if not isinstance(differ, list):
        return ""
    attributes = [value for value in differ if isinstance(value, str)]
    return ",".join(attributes)


def _looks_like_uuid_prefix(value: str) -> bool:
    return len(value) >= 8 and all(
        char in "0123456789abcdef" for char in value
    )


def _asset_name_from_config_path(
    *,
    config_path: str,
    task: str | None,
) -> str:
    stem = Path(config_path).stem
    if task and stem.startswith(f"{task}_"):
        stem = stem[len(task) + 1 :]
    parts = stem.split("_")
    if parts and _looks_like_uuid_prefix(parts[-1].lower()):
        parts = parts[:-1]
    return "_".join(parts) or stem


def _strip_task_prefix(
    *,
    asset_name: str,
    task: str | None,
) -> str:
    if task and asset_name.startswith(f"{task}_"):
        return asset_name[len(task) + 1 :]
    return asset_name


def _asset_identity(
    *,
    summary_dir: Path,
    asset: dict[str, Any],
    task: str | None,
    target_role: str,
    asset_registry: Any | None = None,
) -> tuple[str, str]:
    asset_name = (
        asset.get("asset_id")
        or asset.get("object_name")
        or asset.get("name")
        or asset.get("category")
    )
    config_path = asset.get("config_path")
    if not isinstance(asset_name, str) or not asset_name:
        if isinstance(config_path, str) and config_path:
            asset_name = _asset_name_from_config_path(
                config_path=config_path,
                task=task,
            )
        else:
            asset_name = "unknown"
    else:
        asset_name = _strip_task_prefix(asset_name=asset_name, task=task)

    asset_uuid = asset.get("asset_uuid") or asset.get("uuid")
    if not isinstance(asset_uuid, str) or not asset_uuid:
        asset_uuid = _load_asset_uuid_from_config(
            summary_dir=summary_dir,
            asset=asset,
            target_role=target_role,
        )
    if not isinstance(asset_uuid, str) or not asset_uuid:
        asset_uuid = "unknown"

    if asset_registry is not None and asset_uuid != "unknown":
        registry_asset_id = asset_registry.get_meta(asset_uuid).asset_id
        if not isinstance(registry_asset_id, str) or not registry_asset_id:
            raise ValueError(
                f"asset registry returned invalid asset_id for {asset_uuid}"
            )
        asset_name = registry_asset_id

    return asset_name, asset_uuid


def _read_successful_paths(
    *,
    summary_dir: Path,
    record_files: list[Any],
) -> list[str]:
    selected_paths: list[str] = []
    for record_file in record_files:
        if not isinstance(record_file, str):
            raise ValueError("successful_record_files entries must be strings")
        record_path = _resolve_path(summary_dir, record_file)
        if not record_path.exists():
            raise FileNotFoundError(
                f"successful record file not found: {record_path}"
            )
        selected_paths.extend(
            line.strip()
            for line in record_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return selected_paths


def _successful_record_files(asset: dict[str, Any]) -> list[Any]:
    record_files = asset.get("successful_record_files", [])
    if not isinstance(record_files, list):
        raise ValueError(
            "asset summary successful_record_files must be a list"
        )
    return record_files


def _log_keep(
    *,
    group_name: str | None,
    asset_name: str,
    asset_uuid: str,
    rate: float,
    selected_count: int,
) -> None:
    if group_name is None:
        LOGGER.info(
            "KEEP object=%s uuid=%s success_rate=%.4f selected=%d",
            asset_name,
            asset_uuid,
            rate,
            selected_count,
        )
        return
    LOGGER.info(
        "KEEP group=%s object=%s uuid=%s success_rate=%.4f selected=%d",
        group_name,
        asset_name,
        asset_uuid,
        rate,
        selected_count,
    )


def _log_remove(
    *,
    group_name: str | None,
    asset_name: str,
    asset_uuid: str,
    rate: float,
) -> None:
    if group_name is None:
        LOGGER.info(
            "REMOVE object=%s uuid=%s success_rate=%.4f "
            "reason=success_rate_out_of_range",
            asset_name,
            asset_uuid,
            rate,
        )
        return
    LOGGER.info(
        "REMOVE group=%s object=%s uuid=%s success_rate=%.4f "
        "reason=success_rate_out_of_range",
        group_name,
        asset_name,
        asset_uuid,
        rate,
    )


def _select_from_summary(
    *,
    summary_json_path: str,
    min_success_rate: float,
    max_success_rate: float,
    sample_num: int,
    target_role: str = _DEFAULT_TARGET_ROLE,
    group_name: str | None = None,
    asset_root: str | None = None,
    asset_registry: Any | None = None,
) -> SelectionResult:
    """Return sampled successful records from one summary."""
    if sample_num < 0:
        raise ValueError("sample_num must be non-negative")
    if min_success_rate > max_success_rate:
        raise ValueError("min_success_rate must be <= max_success_rate")

    summary_path = Path(summary_json_path)
    summary = _read_json_mapping(summary_path)
    assets = summary.get("assets")
    if not isinstance(assets, list):
        raise ValueError(f"{summary_path} missing assets list")

    summary_dir = summary_path.parent
    task = summary.get("task")
    task_name = task if isinstance(task, str) else None
    selected_paths: list[str] = []
    selected_assets: list[dict[str, Any]] = []
    kept_assets = 0
    removed_assets = 0
    if asset_registry is None and asset_root:
        asset_registry = AssetRegistry(asset_root)

    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("assets entries must be JSON objects")
        asset_name, asset_uuid = _asset_identity(
            summary_dir=summary_dir,
            asset=asset,
            task=task_name,
            target_role=target_role,
            asset_registry=asset_registry,
        )
        asset_attribute = _asset_attribute_from_config(
            summary_dir=summary_dir,
            asset=asset,
        )
        layout_extras = _layout_entry_extras(
            summary_dir=summary_dir,
            asset=asset,
        )
        rate = _success_rate(asset)
        if min_success_rate <= rate <= max_success_rate:
            successful_paths = _read_successful_paths(
                summary_dir=summary_dir,
                record_files=_successful_record_files(asset),
            )
            sampled_paths = successful_paths[:sample_num]
            selected_paths.extend(sampled_paths)
            entry: dict[str, Any] = {
                "asset_id": asset_name,
                "uuid": asset_uuid,
                "mcap_paths": sampled_paths,
                "attribute": asset_attribute,
            }
            entry.update(layout_extras)
            selected_assets.append(entry)
            kept_assets += 1
            _log_keep(
                group_name=group_name,
                asset_name=asset_name,
                asset_uuid=asset_uuid,
                rate=rate,
                selected_count=len(sampled_paths),
            )
        else:
            removed_assets += 1
            _log_remove(
                group_name=group_name,
                asset_name=asset_name,
                asset_uuid=asset_uuid,
                rate=rate,
            )

    return SelectionResult(
        selected_paths=selected_paths,
        selected_assets=selected_assets,
        kept_assets=kept_assets,
        removed_assets=removed_assets,
    )


def _write_selection_json(
    *,
    output_path: str,
    result: SelectionResult,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "assets": result.selected_assets,
        "selected_asset_count": result.kept_assets,
        "total_asset_count": result.total_assets,
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _default_split_output_path(output_path: str) -> str:
    return str(Path(output_path).with_suffix(".yaml"))


def _split_seen_assets(result: SelectionResult) -> list[dict[str, str]]:
    seen_assets: list[dict[str, str]] = []
    for asset in result.selected_assets:
        asset_uuid = asset.get("uuid")
        asset_id = asset.get("asset_id")
        if not isinstance(asset_uuid, str) or not asset_uuid:
            raise ValueError("selected asset missing uuid for split output")
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError(
                "selected asset missing asset_id for split output"
            )
        seen_assets.append({"uuid": asset_uuid, "asset_id": asset_id})
    return seen_assets


def _write_selection_split_yaml(
    *,
    output_path: str,
    result: SelectionResult,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "name": output.stem,
        "seen": _split_seen_assets(result),
        "unseen_category": [],
        "unseen_instance": [],
    }
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _log_summary(result: SelectionResult) -> None:
    LOGGER.info(
        "SUMMARY selected_assets=%d/%d",
        result.kept_assets,
        result.total_assets,
    )


def select_successful_records(
    *,
    summary_json_path: str,
    output_path: str,
    min_success_rate: float,
    max_success_rate: float,
    sample_num: int,
    target_role: str = _DEFAULT_TARGET_ROLE,
    split_output_path: str | None = None,
    asset_root: str | None = None,
) -> SelectionResult:
    """Write sampled successful records and split YAML from one summary."""
    result = _select_from_summary(
        summary_json_path=summary_json_path,
        min_success_rate=min_success_rate,
        max_success_rate=max_success_rate,
        sample_num=sample_num,
        target_role=target_role,
        asset_root=asset_root,
    )
    _write_selection_json(output_path=output_path, result=result)
    _write_selection_split_yaml(
        output_path=split_output_path
        or _default_split_output_path(output_path),
        result=result,
    )
    _log_summary(result)
    return result


def discover_summary_paths(input_path: str) -> list[Path]:
    """Return sorted batch summary paths from a file or run directory."""
    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"input path not found: {path}")

    summary_paths = sorted(path.glob("batch_run_summary_*.json"))
    summary_paths.extend(sorted(path.glob("group_*/batch_run_summary_*.json")))
    if not summary_paths:
        raise FileNotFoundError(
            f"no batch_run_summary_*.json files found under {path}"
        )
    return summary_paths


def select_successful_records_from_input(
    *,
    input_path: str,
    output_path: str,
    min_success_rate: float,
    max_success_rate: float,
    sample_num: int,
    target_role: str = _DEFAULT_TARGET_ROLE,
    split_output_path: str | None = None,
    asset_root: str | None = None,
) -> SelectionResult:
    """Write sampled successful records and split YAML from input."""
    summary_paths = discover_summary_paths(input_path)
    single_file_input = Path(input_path).is_file()
    selected_paths: list[str] = []
    selected_assets: list[dict[str, Any]] = []
    kept_assets = 0
    removed_assets = 0
    asset_registry = AssetRegistry(asset_root) if asset_root else None

    for summary_path in summary_paths:
        group_name = None if single_file_input else summary_path.parent.name
        result = _select_from_summary(
            summary_json_path=str(summary_path),
            min_success_rate=min_success_rate,
            max_success_rate=max_success_rate,
            sample_num=sample_num,
            target_role=target_role,
            group_name=group_name,
            asset_registry=asset_registry,
        )
        selected_paths.extend(result.selected_paths)
        selected_assets.extend(result.selected_assets)
        kept_assets += result.kept_assets
        removed_assets += result.removed_assets

    result = SelectionResult(
        selected_paths=selected_paths,
        selected_assets=selected_assets,
        kept_assets=kept_assets,
        removed_assets=removed_assets,
    )
    _write_selection_json(output_path=output_path, result=result)
    _write_selection_split_yaml(
        output_path=split_output_path
        or _default_split_output_path(output_path),
        result=result,
    )
    _log_summary(result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    env_asset_root = os.environ.get(_ASSET_ROOT_ENV)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help=(
            "Path to one batch_run_summary_<task>.json or a run root "
            "containing group_*/batch_run_summary_<task>.json files."
        ),
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument(
        "--asset-root",
        default=env_asset_root,
        help=(
            "Asset library root used to resolve original asset_id by uuid. "
            f"Defaults to ${_ASSET_ROOT_ENV} when set."
        ),
    )
    parser.add_argument(
        "--split-output",
        help=(
            "Output split YAML path. Defaults to the JSON output path "
            "with a .yaml suffix."
        ),
    )
    parser.add_argument("--sample-num", required=True, type=int)
    parser.add_argument("--min-success-rate", default=0.0, type=float)
    parser.add_argument("--max-success-rate", default=1.0, type=float)
    parser.add_argument("--target-role", default=_DEFAULT_TARGET_ROLE)
    return parser


def main() -> None:
    """Run successful record selection from CLI arguments."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    args = build_arg_parser().parse_args()
    select_successful_records_from_input(
        input_path=args.input_path,
        output_path=args.output,
        min_success_rate=args.min_success_rate,
        max_success_rate=args.max_success_rate,
        sample_num=args.sample_num,
        target_role=args.target_role,
        split_output_path=args.split_output,
        asset_root=args.asset_root,
    )


if __name__ == "__main__":
    main()
