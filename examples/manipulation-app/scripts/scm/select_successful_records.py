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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LOGGER_NAME = "select_successful_records"
LOGGER = logging.getLogger(LOGGER_NAME)
DEFAULT_INPUT_PATH = "logs/multi_data_synthesis/v3"
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


def _load_asset_uuid_from_config(
    *,
    summary_dir: Path,
    asset: dict[str, Any],
    target_role: str,
) -> str | None:
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


def _asset_identity(
    *,
    summary_dir: Path,
    asset: dict[str, Any],
    task: str | None,
    target_role: str,
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

    asset_uuid = asset.get("asset_uuid") or asset.get("uuid")
    if not isinstance(asset_uuid, str) or not asset_uuid:
        asset_uuid = _load_asset_uuid_from_config(
            summary_dir=summary_dir,
            asset=asset,
            target_role=target_role,
        )
    if not isinstance(asset_uuid, str) or not asset_uuid:
        asset_uuid = "unknown"

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

    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("assets entries must be JSON objects")
        asset_name, asset_uuid = _asset_identity(
            summary_dir=summary_dir,
            asset=asset,
            task=task_name,
            target_role=target_role,
        )
        rate = _success_rate(asset)
        if min_success_rate <= rate <= max_success_rate:
            successful_paths = _read_successful_paths(
                summary_dir=summary_dir,
                record_files=_successful_record_files(asset),
            )
            sampled_paths = successful_paths[:sample_num]
            selected_paths.extend(sampled_paths)
            selected_assets.append(
                {
                    "asset_id": asset_name,
                    "uuid": asset_uuid,
                    "mcap_paths": sampled_paths,
                }
            )
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
) -> SelectionResult:
    """Write sampled successful records from one summary."""
    result = _select_from_summary(
        summary_json_path=summary_json_path,
        min_success_rate=min_success_rate,
        max_success_rate=max_success_rate,
        sample_num=sample_num,
        target_role=target_role,
    )
    _write_selection_json(output_path=output_path, result=result)
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
) -> SelectionResult:
    """Write sampled successful records from one file or run root."""
    summary_paths = discover_summary_paths(input_path)
    single_file_input = Path(input_path).is_file()
    selected_paths: list[str] = []
    selected_assets: list[dict[str, Any]] = []
    kept_assets = 0
    removed_assets = 0

    for summary_path in summary_paths:
        group_name = None if single_file_input else summary_path.parent.name
        result = _select_from_summary(
            summary_json_path=str(summary_path),
            min_success_rate=min_success_rate,
            max_success_rate=max_success_rate,
            sample_num=sample_num,
            target_role=target_role,
            group_name=group_name,
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
    _log_summary(result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
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
    )


if __name__ == "__main__":
    main()
