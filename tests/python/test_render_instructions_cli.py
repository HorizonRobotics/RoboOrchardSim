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
# implied. See the License for the specific language governing permissions
# and limitations under the License.

from __future__ import annotations
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
from google.protobuf.struct_pb2 import Struct
from mcap_protobuf.writer import Writer

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scm/render_instructions.py"


@dataclass
class FakeAssetMeta:
    uuid: str
    category: str
    description: str
    caption_path: str


class FakeAssetRegistry:
    """Minimal registry used at the asset lookup boundary."""

    def __init__(self, metas: list[FakeAssetMeta]) -> None:
        self.metas = {meta.uuid: meta for meta in metas}

    def get_meta(self, uuid: str) -> FakeAssetMeta:
        return self.metas[uuid]


def _load_render_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "unified_instruction_renderer",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load renderer from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_meta_mcap(path: Path, payload: dict[str, object]) -> None:
    message = Struct()
    message.update(payload)
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.write_message(
            topic="/meta_data",
            message=message,
            log_time=1,
            publish_time=1,
        )
        writer.finish()


def _write_caption(
    tmp_path: Path,
    *,
    uuid: str,
    raw: str,
    seen: str,
    unseen: str,
) -> str:
    path = tmp_path / f"{uuid}.json"
    path.write_text(
        json.dumps(
            {
                "uuid": uuid,
                "raw": raw,
                "candidates": [seen],
                "unseen": [unseen],
            }
        ),
        encoding="utf-8",
    )
    return str(path)


@pytest.mark.parametrize(
    ("task_name", "seed_args", "expected_seed"),
    [
        ("pick-category", ["--seed", "17"], 17),
        ("place-a2b", [], 0),
    ],
)
def test_build_arg_parser_supported_task_returns_render_options(
    task_name: str,
    seed_args: list[str],
    expected_seed: int,
) -> None:
    render = _load_render_script()

    args = render.build_arg_parser().parse_args(
        [
            task_name,
            "--input-mcap",
            "first.mcap",
            "--input-mcap",
            "second.mcap",
            "--asset-root",
            "/assets",
            "--output-json",
            "/tmp/rendered.json",
            *seed_args,
        ]
    )

    assert vars(args) == {
        "actor_description_mode": "raw",
        "asset_root": "/assets",
        "input_mcaps": ["first.mcap", "second.mcap"],
        "output_json": "/tmp/rendered.json",
        "seed": expected_seed,
        "task_name": task_name,
        "template_mode": "variants",
    }


@pytest.mark.parametrize("task_name", ["pick-category", "place-a2b"])
def test_build_arg_parser_missing_output_json_exits_with_error(
    task_name: str,
) -> None:
    render = _load_render_script()

    with pytest.raises(SystemExit):
        render.build_arg_parser().parse_args(
            [
                task_name,
                "--input-mcap",
                "input.mcap",
                "--asset-root",
                "/assets",
            ]
        )


@pytest.mark.parametrize(
    "args",
    [
        [
            "pick-attribute",
            "--input-mcap",
            "input.mcap",
            "--asset-root",
            "/assets",
            "--output-json",
            "/tmp/rendered.json",
        ],
        [
            "pick-spatial",
            "--input-mcap",
            "input.mcap",
            "--output-json",
            "/tmp/rendered.json",
        ],
    ],
)
def test_build_arg_parser_removed_task_exits_with_error(
    args: list[str],
) -> None:
    render = _load_render_script()

    with pytest.raises(SystemExit):
        render.build_arg_parser().parse_args(args)


def test_render_template_rows_pick_category_seen_mode_increments_seed(
    tmp_path: Path,
) -> None:
    render = _load_render_script()
    mcap_path = tmp_path / "pick_category.mcap"
    _write_meta_mcap(
        mcap_path,
        {
            "actors": {
                "apple": {
                    "actor_type": "pick",
                    "actor_uuid": "apple-uuid",
                }
            }
        },
    )
    registry = FakeAssetRegistry(
        [
            FakeAssetMeta(
                uuid="apple-uuid",
                category="apple",
                description="apple",
                caption_path=_write_caption(
                    tmp_path,
                    uuid="apple-uuid",
                    raw="apple",
                    seen="red apple",
                    unseen="fruit",
                ),
            )
        ]
    )

    rows = render.render_template_rows(
        mcap_paths=[str(mcap_path), str(mcap_path)],
        task_name="pick-category",
        registry=registry,
        template_mode="fixed",
        actor_description_mode="seen",
        seed=17,
    )

    assert rows == [
        {
            "mcap_path": str(mcap_path),
            "instruction": "Pick up red apple",
            "actor_descriptions": {"actor1": "red apple"},
            "actor_uuids": {"actor1": "apple-uuid"},
            "template_seed": 17,
            "actor_description_seed": 17,
        },
        {
            "mcap_path": str(mcap_path),
            "instruction": "Pick up red apple",
            "actor_descriptions": {"actor1": "red apple"},
            "actor_uuids": {"actor1": "apple-uuid"},
            "template_seed": 18,
            "actor_description_seed": 18,
        },
    ]


def test_render_template_rows_place_a2b_default_mode_uses_raw_descriptions(
    tmp_path: Path,
) -> None:
    render = _load_render_script()
    mcap_path = tmp_path / "place_a2b.mcap"
    _write_meta_mcap(
        mcap_path,
        {
            "actors": {
                "apple": {
                    "actor_type": "pick",
                    "actor_uuid": "apple-uuid",
                },
                "bowl": {
                    "actor_type": "place",
                    "actor_uuid": "bowl-uuid",
                },
            }
        },
    )
    registry = FakeAssetRegistry(
        [
            FakeAssetMeta(
                uuid="apple-uuid",
                category="apple",
                description="apple",
                caption_path=_write_caption(
                    tmp_path,
                    uuid="apple-uuid",
                    raw="apple",
                    seen="red apple",
                    unseen="fruit",
                ),
            ),
            FakeAssetMeta(
                uuid="bowl-uuid",
                category="bowl",
                description="bowl",
                caption_path=_write_caption(
                    tmp_path,
                    uuid="bowl-uuid",
                    raw="bowl",
                    seen="white bowl",
                    unseen="container",
                ),
            ),
        ]
    )

    rows = render.render_template_rows(
        mcap_paths=[str(mcap_path)],
        task_name="place-a2b",
        registry=registry,
        template_mode="fixed",
        actor_description_mode="raw",
        seed=0,
    )

    assert rows == [
        {
            "mcap_path": str(mcap_path),
            "instruction": "Pick up apple and place in bowl",
            "actor_descriptions": {
                "actor1": "apple",
                "actor2": "bowl",
            },
            "actor_uuids": {
                "actor1": "apple-uuid",
                "actor2": "bowl-uuid",
            },
            "template_seed": 0,
            "actor_description_seed": 0,
        }
    ]
