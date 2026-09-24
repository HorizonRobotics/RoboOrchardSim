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

"""Prepare one batch-specific AIDI submit template."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any

_SKILL_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BASE_TEMPLATE = _SKILL_ROOT / "assets" / "aidi-submit-template.json"
_REQUIRED_PLACEHOLDER = "__REQUIRED__"


def _load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _validate_template_shape(payload: dict[str, Any]) -> None:
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


def prepare_submit_template(
    *,
    base_template_path: Path,
    output_path: Path,
    queue_name: str,
    bundle_upload_path: Path,
    job_password_env: str,
) -> tuple[str, list[str]]:
    """Write a batch-specific template and return its workspace plan path."""
    if not queue_name.strip():
        raise ValueError("queue_name must not be empty")
    if bundle_upload_path.is_absolute():
        raise ValueError("bundle upload path must be repository-relative")

    bundle_path = bundle_upload_path.resolve()
    if not bundle_path.is_dir():
        raise FileNotFoundError(
            f"batch bundle not found: {bundle_upload_path}"
        )
    if not (bundle_path / "batch_plan.json").is_file():
        raise FileNotFoundError(
            f"batch plan not found under bundle: {bundle_upload_path}"
        )

    payload = _load_json_mapping(base_template_path)
    _validate_template_shape(payload)
    payload["queue_name"] = queue_name
    payload["num_workers"] = 1
    payload["gpu_per_worker"] = 1

    password = os.environ.get(job_password_env)
    if payload.get("job_password") == _REQUIRED_PLACEHOLDER and password:
        payload["job_password"] = password

    normalized_upload = bundle_upload_path.as_posix().rstrip("/")
    uploads = list(payload["to_upload"])
    if normalized_upload not in uploads:
        uploads.append(normalized_upload)
    payload["to_upload"] = uploads

    if output_path.resolve() == base_template_path.resolve():
        raise ValueError("output must not overwrite the base template")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    workspace_batch_plan = f"{bundle_upload_path.name}/batch_plan.json"
    return workspace_batch_plan, _find_required_placeholders(payload)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-template",
        type=Path,
        default=_DEFAULT_BASE_TEMPLATE,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--queue-name", required=True)
    parser.add_argument(
        "--bundle-upload-path",
        required=True,
        type=Path,
        help="Repository-relative portable batch bundle path.",
    )
    parser.add_argument(
        "--job-password-env",
        default="AIDI_JOB_PASSWORD",
        help="Environment variable used to fill a required job password.",
    )
    return parser


def main() -> None:
    """Prepare an AIDI template from CLI arguments."""
    args = build_arg_parser().parse_args()
    workspace_batch_plan, placeholders = prepare_submit_template(
        base_template_path=args.base_template,
        output_path=args.output,
        queue_name=args.queue_name,
        bundle_upload_path=args.bundle_upload_path,
        job_password_env=args.job_password_env,
    )
    print(f"wrote submit template: {args.output}")
    print(f"workspace batch plan: {workspace_batch_plan}")
    if placeholders:
        print("unresolved required fields: " + ", ".join(placeholders))


if __name__ == "__main__":
    main()
