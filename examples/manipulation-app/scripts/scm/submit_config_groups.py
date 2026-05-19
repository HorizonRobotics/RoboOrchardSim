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

"""Render and optionally submit AIDI jobs for batch config groups."""

from __future__ import annotations
import argparse
import copy
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

_DEFAULT_SUBMIT_JSON_DIR_NAME = "submit_jsons"


def shell_quote(value: str | int) -> str:
    """Return a shell-safe string."""
    return shlex.quote(str(value))


def render_submit_jsons(
    *,
    plan: dict[str, Any],
    template: dict[str, Any],
    batch_plan_path: str,
    task_root_dir: str,
    asset_root: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Render one AIDI submit JSON payload per batch group."""
    rendered: list[tuple[str, dict[str, Any]]] = []
    batch_id = str(plan["batch_id"])
    groups = plan["groups"]
    if not isinstance(groups, list):
        raise ValueError("batch plan groups must be a list")
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("batch plan group must be a JSON object")
        group_id = str(group["group_id"])
        submit_json = copy.deepcopy(template)
        job_name = f"{batch_id}_{group_id}"
        submit_json["job_name"] = job_name
        submit_json["num_workers"] = 1
        submit_json["gpu_per_worker"] = 1
        submit_json["execute"] = True
        template_cmd = submit_json.get("cmd", [])
        if not isinstance(template_cmd, list):
            raise ValueError("submit template cmd must be a list")
        submit_json["cmd"] = [
            *template_cmd,
            " ".join(
                [
                    "python3",
                    "examples/manipulation-app/scripts/scm/"
                    "run_multi_task_synthesis.py",
                    "--batch-plan",
                    shell_quote(batch_plan_path),
                    "--group-id",
                    shell_quote(group_id),
                    "--asset-root",
                    shell_quote(asset_root),
                    "--output-root-dir",
                    shell_quote(f"{task_root_dir.rstrip('/')}/{group_id}"),
                ]
            ),
        ]
        rendered.append((f"{job_name}.json", submit_json))
    return rendered


def write_submit_jsons(
    rendered: list[tuple[str, dict[str, Any]]],
    output_dir: str,
) -> list[Path]:
    """Write rendered submit JSON payloads to an output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    seen: set[str] = set()
    for filename, payload in rendered:
        if filename in seen:
            raise ValueError(f"duplicate submit JSON filename: {filename}")
        seen.add(filename)
        path = output_path / filename
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def load_submit_plan(batch_plan_path: str) -> dict[str, Any]:
    """Load the batch plan fields required by submit rendering."""
    payload = json.loads(Path(batch_plan_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"batch plan must be a JSON object: {batch_plan_path}"
        )
    if "batch_id" not in payload:
        raise ValueError(f"batch plan missing batch_id: {batch_plan_path}")
    if not isinstance(payload.get("groups"), list):
        raise ValueError(
            f"batch plan groups must be a list: {batch_plan_path}"
        )
    return payload


def submit_aidi_jsons(json_paths: list[Path]) -> None:
    """Submit rendered AIDI JSON files."""
    for path in json_paths:
        subprocess.run(
            [
                "RoboOrchardJob-AIDISubmit",
                "submit_from_config",
                "--config",
                str(path),
            ],
            check=True,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the submitter CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-plan", required=True)
    parser.add_argument("--submit-template", required=True)
    parser.add_argument(
        "--submit-json-dir",
        default=_DEFAULT_SUBMIT_JSON_DIR_NAME,
        help=(
            "Directory to write rendered submit JSON files. Defaults to "
            f"{_DEFAULT_SUBMIT_JSON_DIR_NAME}."
        ),
    )
    parser.add_argument("--cluster-output-root", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--submit", action="store_true")
    return parser


def main() -> None:
    """Run the submitter CLI."""
    args = build_arg_parser().parse_args()
    plan = load_submit_plan(args.batch_plan)
    template = json.loads(
        Path(args.submit_template).read_text(encoding="utf-8")
    )
    if not isinstance(template, dict):
        raise ValueError(
            f"submit template must be a JSON object: {args.submit_template}"
        )

    rendered = render_submit_jsons(
        plan=plan,
        template=template,
        batch_plan_path=args.batch_plan,
        task_root_dir=args.cluster_output_root,
        asset_root=args.asset_root,
    )
    json_paths = write_submit_jsons(rendered, args.submit_json_dir)
    for path in json_paths:
        print(f"generated: {path}")
    if args.submit:
        submit_aidi_jsons(json_paths)


if __name__ == "__main__":
    main()
