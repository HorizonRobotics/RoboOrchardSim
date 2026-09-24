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

"""Validate a portable SCM batch bundle before local or AIDI execution."""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

_REQUIRED_PLACEHOLDER = "__REQUIRED__"


@dataclass(frozen=True)
class BundleSummary:
    """Observable validation summary for one batch bundle."""

    batch_id: str
    task: str
    scene_count: int
    group_count: int
    data_samples_per_scene: int
    planned_data_count: int
    maximum_group_data_count: int
    expected_upload_path: str
    expected_workspace_batch_plan: str
    submit_template_validated: bool


def _load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _require_non_empty_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _require_nonnegative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _require_within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes {root}: {path}") from exc


def _validate_yaml(path: Path) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"task YAML must contain a mapping: {path}")


def _find_required_placeholders(
    value: Any,
    *,
    field_path: str = "$",
) -> list[str]:
    if isinstance(value, str):
        return [field_path] if _REQUIRED_PLACEHOLDER in value else []
    if isinstance(value, list):
        placeholders: list[str] = []
        for index, item in enumerate(value):
            placeholders.extend(
                _find_required_placeholders(
                    item,
                    field_path=f"{field_path}[{index}]",
                )
            )
        return placeholders
    if isinstance(value, dict):
        placeholders = []
        for key, item in value.items():
            placeholders.extend(
                _find_required_placeholders(
                    item,
                    field_path=f"{field_path}.{key}",
                )
            )
        return placeholders
    return []


def _validate_submit_template(
    *,
    template_path: Path,
    expected_upload_path: str,
) -> None:
    payload = _load_json_mapping(template_path)
    placeholders = _find_required_placeholders(payload)
    if placeholders:
        raise ValueError(
            "submit template has unresolved required fields: "
            + ", ".join(placeholders)
        )

    _require_non_empty_string(payload, "queue_name")
    _require_non_empty_string(payload, "docker_image")
    _require_non_empty_string(payload, "project_id")
    _require_non_empty_string(payload, "job_password")
    if _require_positive_int(payload, "num_workers") != 1:
        raise ValueError("submit template num_workers must equal 1")
    if _require_positive_int(payload, "gpu_per_worker") != 1:
        raise ValueError("submit template gpu_per_worker must equal 1")

    commands = payload.get("cmd")
    if not isinstance(commands, list) or not all(
        isinstance(command, str) for command in commands
    ):
        raise ValueError("submit template cmd must be a list of strings")

    uploads = payload.get("to_upload")
    if not isinstance(uploads, list) or not all(
        isinstance(upload, str) for upload in uploads
    ):
        raise ValueError("submit template to_upload must be a list of strings")
    if expected_upload_path not in uploads:
        raise ValueError(
            "submit template does not upload the complete bundle: "
            f"{expected_upload_path}"
        )


def validate_batch_bundle(
    *,
    batch_plan_path: Path,
    submit_template_path: Path | None = None,
    workspace_batch_plan: str | None = None,
    maximum_group_data_count: int = 500,
) -> BundleSummary:
    """Validate one batch plan, its YAMLs, and optional submit template."""
    if maximum_group_data_count < 1:
        raise ValueError("maximum group data count must be positive")

    plan = _load_json_mapping(batch_plan_path)
    batch_id = _require_non_empty_string(plan, "batch_id")
    task = _require_non_empty_string(plan, "task")
    data_samples_per_scene = _require_positive_int(plan, "episodes_per_config")
    groups = plan.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("batch plan groups must be a non-empty list")

    bundle_root = batch_plan_path.resolve().parent
    config_root = (bundle_root / "configs").resolve()
    if not config_root.is_dir():
        raise FileNotFoundError(f"config directory not found: {config_root}")

    seen_groups: set[str] = set()
    seen_configs: set[str] = set()
    scene_count = 0
    planned_data_count = 0
    observed_group_maximum = 0
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("batch plan group entries must be JSON objects")
        group_id = _require_non_empty_string(group, "group_id")
        if group_id in seen_groups:
            raise ValueError(f"duplicate group_id: {group_id}")
        seen_groups.add(group_id)
        _require_nonnegative_int(group, "seed")

        configs = group.get("configs")
        if not isinstance(configs, list) or not configs:
            raise ValueError(f"{group_id} configs must be a non-empty list")
        group_data_count = len(configs) * data_samples_per_scene
        if group_data_count > maximum_group_data_count:
            raise ValueError(
                f"{group_id} plans {group_data_count} data samples; "
                f"maximum is {maximum_group_data_count}"
            )
        observed_group_maximum = max(observed_group_maximum, group_data_count)
        planned_data_count += group_data_count

        for config_value in configs:
            if not isinstance(config_value, str) or not config_value:
                raise ValueError(f"{group_id} config paths must be strings")
            config_path = Path(config_value)
            if config_path.is_absolute():
                raise ValueError(
                    f"absolute config path is not portable: {config_value}"
                )
            if config_value in seen_configs:
                raise ValueError(f"duplicate config path: {config_value}")
            seen_configs.add(config_value)
            resolved_config = (bundle_root / config_path).resolve()
            _require_within(
                resolved_config,
                config_root,
                label="config path",
            )
            if not resolved_config.is_file():
                raise FileNotFoundError(
                    f"task YAML not found: {resolved_config}"
                )
            _validate_yaml(resolved_config)
            scene_count += 1

    expected_upload_path = Path(
        os.path.relpath(bundle_root, Path.cwd().resolve())
    ).as_posix()
    expected_workspace_batch_plan = (
        f"{bundle_root.name}/{batch_plan_path.name}"
    )
    if workspace_batch_plan is not None:
        workspace_path = PurePosixPath(workspace_batch_plan)
        if workspace_path.is_absolute():
            raise ValueError("workspace batch-plan path must be relative")
        if workspace_path.as_posix() != expected_workspace_batch_plan:
            raise ValueError(
                "workspace batch-plan path does not match basename upload: "
                f"expected {expected_workspace_batch_plan}, "
                f"got {workspace_batch_plan}"
            )

    if submit_template_path is not None:
        _validate_submit_template(
            template_path=submit_template_path,
            expected_upload_path=expected_upload_path,
        )

    return BundleSummary(
        batch_id=batch_id,
        task=task,
        scene_count=scene_count,
        group_count=len(groups),
        data_samples_per_scene=data_samples_per_scene,
        planned_data_count=planned_data_count,
        maximum_group_data_count=observed_group_maximum,
        expected_upload_path=expected_upload_path,
        expected_workspace_batch_plan=expected_workspace_batch_plan,
        submit_template_validated=submit_template_path is not None,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-plan", required=True, type=Path)
    parser.add_argument("--submit-template", type=Path)
    parser.add_argument("--workspace-batch-plan")
    parser.add_argument(
        "--maximum-group-data-count",
        type=int,
        default=500,
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    """Validate a batch bundle from CLI arguments."""
    args = build_arg_parser().parse_args()
    summary = validate_batch_bundle(
        batch_plan_path=args.batch_plan,
        submit_template_path=args.submit_template,
        workspace_batch_plan=args.workspace_batch_plan,
        maximum_group_data_count=args.maximum_group_data_count,
    )
    payload = asdict(summary)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("batch bundle is valid")
    for key, value in payload.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
