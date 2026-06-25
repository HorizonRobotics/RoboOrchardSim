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

"""Render one instruction per MCAP path listed in input files."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from urllib import request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robo_orchard_sim.asset_manager.registry import AssetRegistry  # noqa: E402
from robo_orchard_sim.task_components.instructions import (  # noqa: E402
    InstructionActor,
    InstructionWrapper,
    extract_instruction_actor_uuids_from_mcap,
)

_MCAP_SUFFIX = "/episode0/env0_data.mcap"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True)
    parser.add_argument(
        "--input-txt",
        action="append",
        dest="input_txts",
    )
    parser.add_argument(
        "--input-json",
        action="append",
        dest="input_jsons",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--template-name", default="pick_default")
    parser.add_argument(
        "--template-mode",
        choices=("fixed", "variants"),
        default="variants",
    )
    return parser


def _resolve_mcap_path(raw_line: str) -> str:
    line = raw_line.strip()
    if line.endswith(_MCAP_SUFFIX):
        return line
    return f"{line}{_MCAP_SUFFIX}"


def _read_input_txt(txt_path: str) -> str:
    if txt_path.startswith(("http://", "https://")):
        with request.urlopen(txt_path) as response:
            return response.read().decode("utf-8")
    return Path(txt_path).read_text(encoding="utf-8")


def _iter_mcap_paths(txt_paths: list[str]) -> list[str]:
    mcap_paths: list[str] = []
    for txt_path in txt_paths:
        for raw_line in _read_input_txt(txt_path).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            mcap_paths.append(_resolve_mcap_path(line))
    return mcap_paths


def _iter_selected_records_mcap_paths(json_paths: list[str]) -> list[str]:
    mcap_paths: list[str] = []
    for json_path in json_paths:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        for asset in payload["assets"]:
            for raw_path in asset["mcap_paths"]:
                mcap_paths.append(_resolve_mcap_path(raw_path))
    return mcap_paths


def main() -> int:
    """Read input files and print rendered instructions as JSON Lines."""
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.input_txts and not args.input_jsons:
        parser.error("at least one --input-txt or --input-json is required")
    mcap_paths = _iter_mcap_paths(args.input_txts or [])
    mcap_paths.extend(
        _iter_selected_records_mcap_paths(args.input_jsons or [])
    )
    template_name = args.template_name
    template_mode = args.template_mode
    registry = AssetRegistry(args.asset_root)
    instruction_wrapper = InstructionWrapper(
        template_name,
        template_mode=template_mode,
        actor_description_mode="seen",
    )
    actor_cache: dict[str, InstructionActor] = {}
    rows: list[dict[str, object]] = []
    for idx, mcap_path in enumerate(mcap_paths):
        actor_uuids = extract_instruction_actor_uuids_from_mcap(
            mcap_path,
            template_name=template_name,
            template_mode=template_mode,
        )
        actors: dict[str, InstructionActor] = {}
        for actor_name, actor_uuid in actor_uuids.items():
            actor = actor_cache.get(actor_uuid)
            if actor is None:
                actor = InstructionActor.from_registry(
                    actor_uuid,
                    registry,
                    actor_description_mode="raw",
                )
                actor_cache[actor_uuid] = actor
            actors[actor_name] = actor
        instruction = instruction_wrapper.render(
            actors=actors,
            template_seed=idx,
            actor_description_seed=idx,
        )
        row = {
            "mcap_path": mcap_path,
            "instruction": instruction,
        }
        row["actor_uuids"] = actor_uuids
        row["template_seed"] = idx
        row["actor_description_seed"] = idx
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(rows, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
