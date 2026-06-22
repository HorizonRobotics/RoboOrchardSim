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

"""Render spatial-pick instructions from selected MCAP records."""

from __future__ import annotations
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robo_orchard_sim.tasks.instructions import (  # noqa: E402
    InstructionRenderError,
    extract_instruction_actor_uuids_from_mcap,
)

_MCAP_SUFFIX = "/episode0/env0_data.mcap"
_ACTOR1_KEY = "actor1"
_DEFAULT_TEMPLATE_NAME = "pick_default"

PICK_VERBS = [
    "pick up",
    "grasp",
    "grab",
    "lift",
    "take",
]

RELATION_PHRASES = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "front_of": "in front of",
    "behind": "behind",
    "near": "near",
    "far": "far from",
}

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        action="append",
        dest="input_jsons",
        required=True,
    )
    parser.add_argument(
        "--output-json",
        required=True,
    )
    return parser


def _resolve_mcap_path(raw_path: str) -> str:
    path = raw_path.strip()
    if path.endswith(_MCAP_SUFFIX):
        return path
    return f"{path}{_MCAP_SUFFIX}"


def _normalize_object_name(value: str) -> str:
    if value.endswith("_target"):
        value = value[: -len("_target")]
    return value.replace("_", " ")


def _stable_pick_verb(seed_text: str) -> str:
    digest = hashlib.sha1(seed_text.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(PICK_VERBS)
    return PICK_VERBS[index]


def _sentence(text: str) -> str:
    if not text:
        return "."
    return f"{text[0].upper()}{text[1:]}."


def _render_instruction(asset: dict[str, object], mcap_path: str) -> str:
    verb = _stable_pick_verb(mcap_path)
    subject = _normalize_object_name(str(asset["subject"]))
    anchor = _normalize_object_name(str(asset["anchor"]))
    relation_phrase = RELATION_PHRASES[str(asset["relation"])]
    return _sentence(f"{verb} the {subject} {relation_phrase} the {anchor}")


def _iter_selected_record_rows(
    json_paths: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for json_path in json_paths:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        for asset in payload["assets"]:
            for raw_path in asset["mcap_paths"]:
                rows.append(
                    {
                        "asset": asset,
                        "mcap_path": _resolve_mcap_path(str(raw_path)),
                    }
                )
    return rows


def _resolve_actor_uuids(
    mcap_path: str,
    *,
    template_name: str,
) -> dict[str, str] | None:
    try:
        actor_uuids = extract_instruction_actor_uuids_from_mcap(
            mcap_path,
            template_name=template_name,
        )
        actor_uuid = actor_uuids[_ACTOR1_KEY]
    except (InstructionRenderError, KeyError) as exc:
        logger.warning(
            "Skipping mcap without actor1 uuid: mcap_path=%s error=%s",
            mcap_path,
            exc,
        )
        return None
    return {_ACTOR1_KEY: actor_uuid}


def main() -> int:
    """Read selected-record JSON files and write instructions as JSON."""
    parser = build_arg_parser()
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    record_rows = _iter_selected_record_rows(args.input_jsons)
    total_count = len(record_rows)
    for idx, record in enumerate(record_rows, start=1):
        mcap_path = str(record["mcap_path"])
        print(
            f"Processing {idx}/{total_count}: {mcap_path}",
            file=sys.stderr,
        )
        actor_uuids = _resolve_actor_uuids(
            mcap_path,
            template_name=_DEFAULT_TEMPLATE_NAME,
        )
        if actor_uuids is None:
            continue

        asset = record["asset"]
        if not isinstance(asset, dict):
            raise TypeError("selected record asset must be a JSON object")
        row = {
            "mcap_path": mcap_path,
            "instruction": _render_instruction(asset, mcap_path),
            "actor_uuids": actor_uuids,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True))

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(rows, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
