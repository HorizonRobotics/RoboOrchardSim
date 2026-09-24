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

"""Render task instructions for MCAP recordings."""

from __future__ import annotations
import argparse
import json
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

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

DescriptionMode = Literal["raw", "seen", "unseen"]
TemplateMode = Literal["fixed", "variants"]
TemplateTask = Literal["pick-category", "place-a2b"]

TEMPLATE_BY_TASK: dict[TemplateTask, str] = {
    "pick-category": "pick_default",
    "place-a2b": "place_a2b_default",
}


def _add_input_mcap_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-mcap",
        action="append",
        dest="input_mcaps",
        required=True,
    )


def _add_template_mode_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--template-mode",
        choices=("fixed", "variants"),
        default="variants",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the unified instruction renderer argument parser."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="task_name", required=True)

    for task_name in TEMPLATE_BY_TASK:
        task_parser = subparsers.add_parser(task_name)
        _add_input_mcap_argument(task_parser)
        task_parser.add_argument("--asset-root", required=True)
        task_parser.add_argument("--output-json", required=True)
        task_parser.add_argument("--seed", type=int, default=0)
        _add_template_mode_argument(task_parser)
        task_parser.add_argument(
            "--actor-description-mode",
            choices=("raw", "seen", "unseen"),
            default="raw",
        )

    return parser


def resolve_actor_description(
    actor: Any,
    *,
    mode: DescriptionMode,
    seed: int,
) -> str:
    """Resolve the description recorded in the output role contract."""
    if mode == "raw":
        return str(actor.raw_description).replace("_", " ")
    candidates = (
        actor.seen_descriptions
        if mode == "seen"
        else actor.unseen_descriptions
    )
    if not candidates:
        raise InstructionRenderError(
            f"Actor description {mode!r} is empty for uuid {actor.uuid!r}"
        )
    return str(random.Random(seed).choice(candidates))


def build_actor_descriptions(
    actors: Mapping[str, Any],
    *,
    actor_description_mode: DescriptionMode,
    actor_description_seed: int,
) -> dict[str, str]:
    """Build actor keys with the exact rendered description values."""
    return {
        actor_name: resolve_actor_description(
            actor,
            mode=actor_description_mode,
            seed=actor_description_seed,
        )
        for actor_name, actor in actors.items()
    }


def render_template_rows(
    *,
    mcap_paths: Sequence[str],
    task_name: TemplateTask,
    registry: Any,
    template_mode: TemplateMode,
    actor_description_mode: DescriptionMode,
    seed: int,
) -> list[dict[str, object]]:
    """Render registry-backed pick-category or place-a2b rows."""
    template_name = TEMPLATE_BY_TASK[task_name]
    wrapper = InstructionWrapper(
        template_name,
        template_mode=template_mode,
        actor_description_mode=actor_description_mode,
    )
    actor_cache: dict[str, InstructionActor] = {}
    rows: list[dict[str, object]] = []
    for idx, mcap_path in enumerate(mcap_paths):
        row_seed = seed + idx
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
        rows.append(
            {
                "mcap_path": mcap_path,
                "instruction": wrapper.render(
                    actors=actors,
                    template_seed=row_seed,
                    actor_description_seed=row_seed,
                ),
                "actor_descriptions": build_actor_descriptions(
                    actors,
                    actor_description_mode=actor_description_mode,
                    actor_description_seed=row_seed,
                ),
                "actor_uuids": actor_uuids,
                "template_seed": row_seed,
                "actor_description_seed": row_seed,
            }
        )
    return rows


def emit_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    output_json: str,
) -> None:
    """Print JSON Lines and write the same rows as one JSON array."""
    for row in rows:
        print(json.dumps(row, ensure_ascii=True))
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(list(rows), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    """Render instruction manifests for one supported task."""
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        rows = render_template_rows(
            mcap_paths=args.input_mcaps,
            task_name=cast(TemplateTask, args.task_name),
            registry=AssetRegistry(args.asset_root),
            template_mode=cast(TemplateMode, args.template_mode),
            actor_description_mode=cast(
                DescriptionMode,
                args.actor_description_mode,
            ),
            seed=args.seed,
        )
    except InstructionRenderError as exc:
        parser.error(str(exc))
    emit_rows(rows, output_json=args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
