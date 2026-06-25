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

"""Render attribute-based pick instructions for selected MCAP records."""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Literal, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robo_orchard_sim.asset_manager.registry import AssetRegistry  # noqa: E402
from robo_orchard_sim.task_components.instructions import (  # noqa: E402
    InstructionActor,
    InstructionRenderError,
    InstructionWrapper,
    extract_instruction_actor_uuids_from_mcap,
)

_MCAP_SUFFIX = "/episode0/env0_data.mcap"
_PICK_ACTOR_KEY = "actor1"
_TEMPLATE_NAME = "pick_attribute"
logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True)
    parser.add_argument(
        "--input-json",
        action="append",
        dest="input_jsons",
        required=True,
    )
    parser.add_argument("--output-json")
    parser.add_argument(
        "--template-mode",
        choices=("fixed", "variants"),
        default="variants",
    )
    return parser


def _resolve_mcap_path(raw_path: str) -> str:
    path = raw_path.strip()
    if path.endswith(_MCAP_SUFFIX):
        return path
    return f"{path}{_MCAP_SUFFIX}"


def _iter_selected_record_rows(
    json_paths: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for json_path in json_paths:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        for asset in payload["assets"]:
            asset_id = str(asset["asset_id"])
            attribute = str(asset["attribute"])
            for raw_path in asset["mcap_paths"]:
                rows.append(
                    {
                        "asset_id": asset_id,
                        "attribute": attribute,
                        "mcap_path": _resolve_mcap_path(str(raw_path)),
                    }
                )
    return rows


def _resolve_pick_actor_uuid(
    actor_uuids: dict[str, str],
    *,
    mcap_path: str,
) -> str:
    try:
        return actor_uuids[_PICK_ACTOR_KEY]
    except KeyError as exc:
        raise InstructionRenderError(
            f"MCAP '{mcap_path}' missing '{_PICK_ACTOR_KEY}' actor uuid"
        ) from exc


def main() -> int:
    """Read selected-record JSON files and print instructions as JSON Lines."""
    parser = build_arg_parser()
    args = parser.parse_args()
    template_mode = cast(Literal["fixed", "variants"], args.template_mode)
    registry = AssetRegistry(args.asset_root)
    wrapper = InstructionWrapper(
        _TEMPLATE_NAME,
        template_mode=template_mode,
        actor_description_mode="raw",
    )
    rows: list[dict[str, object]] = []
    skipped_count = 0
    record_rows = _iter_selected_record_rows(args.input_jsons)
    for idx, record in enumerate(record_rows):
        mcap_path = record["mcap_path"]
        attribute = record["attribute"]
        actor_uuids = extract_instruction_actor_uuids_from_mcap(
            mcap_path,
            template_name=_TEMPLATE_NAME,
            template_mode=template_mode,
        )
        actor_uuid = _resolve_pick_actor_uuid(
            actor_uuids,
            mcap_path=mcap_path,
        )
        try:
            actor = InstructionActor.from_registry_with_attribute(
                actor_uuid,
                registry,
                attribute_name=attribute,
                actor_description_mode="raw",
            )
        except InstructionRenderError as exc:
            skipped_count += 1
            logger.warning(
                "Skipping mcap with unsupported attribute value count: "
                "mcap_path=%s actor_uuid=%s attribute=%s error=%s",
                mcap_path,
                actor_uuid,
                attribute,
                exc,
            )
            continue
        instruction = wrapper.render(
            actors={_PICK_ACTOR_KEY: actor},
            template_seed=idx,
            actor_description_seed=idx,
        )
        row = {
            "mcap_path": mcap_path,
            "instruction": instruction,
            "actor_uuids": actor_uuids,
            "actor_uuid": actor_uuid,
            "asset_id": record["asset_id"],
            "attribute": actor.attribute_name,
            "attribute_value": actor.attribute_value,
            "category": actor.category,
            "template_name": _TEMPLATE_NAME,
            "template_seed": idx,
            "actor_description_seed": idx,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(rows, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
    if skipped_count:
        logger.warning(
            "Skipped %d mcap(s) while rendering instructions",
            skipped_count,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
