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

"""Generate per-layout task YAML configs from a base config."""

from __future__ import annotations
import argparse
import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_LIST_NAME = "yaml_list.txt"
_LAYOUT_FILE_RE = re.compile(r"layout_[0-9]{4}\.json")
_TASK_DIR_RE = re.compile(r"task_.*")


def _discover_all_layout_paths(*, layout_root: str | Path) -> list[Path]:
    root_path = Path(layout_root)
    layout_paths: list[Path] = []

    with os.scandir(root_path) as task_entries:
        task_paths = sorted(
            (entry.name, Path(entry.path))
            for entry in task_entries
            if _TASK_DIR_RE.fullmatch(entry.name)
        )

    for _, task_path in task_paths:
        try:
            with os.scandir(task_path) as layout_entries:
                layout_candidates = sorted(
                    Path(entry.path)
                    for entry in layout_entries
                    if _LAYOUT_FILE_RE.fullmatch(entry.name)
                    and entry.is_file()
                )
        except NotADirectoryError:
            continue
        layout_paths.extend(layout_candidates)
    return layout_paths


def _select_layout_paths(
    *,
    layout_paths: list[Path],
    num_layouts: int | None,
    layout_range: tuple[int, int] | None,
    last_layouts: int | None,
) -> list[Path]:
    selection_count = sum(
        option is not None
        for option in (num_layouts, layout_range, last_layouts)
    )
    if selection_count != 1:
        raise ValueError(
            "exactly one of num_layouts, layout_range, or last_layouts "
            "must be provided"
        )

    if num_layouts is not None:
        if num_layouts < 0:
            raise ValueError("num_layouts must be non-negative")
        return layout_paths[:num_layouts]

    if layout_range is not None:
        start, end = layout_range
        if start < 0 or end < 0:
            raise ValueError("layout_range indexes must be non-negative")
        if end < start:
            raise ValueError("layout_range end must be >= start")
        return layout_paths[start : end + 1]

    assert last_layouts is not None
    if last_layouts < 0:
        raise ValueError("last_layouts must be non-negative")
    if last_layouts == 0:
        return []
    return layout_paths[-last_layouts:]


def _parse_layout_range(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([0-9]+)-([0-9]+)", value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "layout range must use inclusive START-END syntax"
        )
    start, end = (int(part) for part in match.groups())
    if end < start:
        raise argparse.ArgumentTypeError(
            "layout range end must be greater than or equal to start"
        )
    return start, end


def _config_filename(layout_path: Path) -> str:
    return f"{layout_path.parent.name}_{layout_path.stem}.yaml"


def _layout_payload(
    *,
    base_payload: dict[str, Any],
    layout_path: Path,
) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    payload["layout"] = str(layout_path)
    return payload


def write_layout_task_configs(
    *,
    base_config_path: str | Path,
    layout_root: str | Path,
    num_layouts: int | None = None,
    layout_range: tuple[int, int] | None = None,
    last_layouts: int | None = None,
    output_dir: str | Path,
    config_list_path: str | Path | None = None,
) -> list[str]:
    """Write one task YAML per discovered layout JSON."""
    with open(base_config_path, encoding="utf-8") as fr:
        base_payload = yaml.safe_load(fr) or {}
    if not isinstance(base_payload, dict):
        raise ValueError(f"{base_config_path} must contain a YAML mapping")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_paths: list[str] = []
    seen_paths: set[str] = set()
    layout_paths = _select_layout_paths(
        layout_paths=_discover_all_layout_paths(layout_root=layout_root),
        num_layouts=num_layouts,
        layout_range=layout_range,
        last_layouts=last_layouts,
    )
    for layout_path in layout_paths:
        config_path = output_path / _config_filename(layout_path)
        if str(config_path) in seen_paths:
            raise ValueError(f"duplicate generated config path: {config_path}")
        seen_paths.add(str(config_path))

        payload = _layout_payload(
            base_payload=base_payload,
            layout_path=layout_path,
        )
        with config_path.open("w", encoding="utf-8") as fw:
            yaml.safe_dump(payload, fw, sort_keys=False)
        generated_paths.append(str(config_path))

    if config_list_path is None:
        config_list_path = output_path / _DEFAULT_CONFIG_LIST_NAME
    list_path = Path(config_list_path)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text(
        "\n".join(generated_paths) + ("\n" if generated_paths else ""),
        encoding="utf-8",
    )
    return generated_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--layout-root", required=True)
    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument(
        "--num-layouts",
        type=int,
        help="select the first N discovered layouts",
    )
    selection_group.add_argument(
        "--layout-range",
        type=_parse_layout_range,
        help="select an inclusive global layout index range, e.g. 500-600",
    )
    selection_group.add_argument(
        "--last-layouts",
        type=int,
        help="select the last N discovered layouts",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    """Run layout task config generation from CLI arguments."""
    args = build_arg_parser().parse_args()
    config_list_path = Path(args.output_dir) / _DEFAULT_CONFIG_LIST_NAME
    generated_paths = write_layout_task_configs(
        base_config_path=args.base_config,
        layout_root=args.layout_root,
        num_layouts=args.num_layouts,
        layout_range=args.layout_range,
        last_layouts=args.last_layouts,
        output_dir=args.output_dir,
        config_list_path=config_list_path,
    )
    print(
        f"Generated {len(generated_paths)} configs under {args.output_dir}; "
        f"wrote list {config_list_path}"
    )


if __name__ == "__main__":
    main()
