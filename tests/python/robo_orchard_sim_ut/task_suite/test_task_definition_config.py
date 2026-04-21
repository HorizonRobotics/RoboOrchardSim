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

"""Tests for TaskDefinition YAML config (asset_configs block).

Covers the resolve_asset_configs() classmethod and sanity-checks the
shipped place_a2b.yaml so its asset_configs block stays well-formed.
"""

from __future__ import annotations
import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from robo_orchard_sim.task_suite.base import TaskDefinition


def _stub_task_def_with_config(yaml_path: str) -> type[TaskDefinition]:
    """Build a minimal TaskDefinition subclass pointing at a YAML."""

    class _Stub(TaskDefinition):
        namespace = "_stub"
        config_path = yaml_path

        @classmethod
        def build(cls, resolver=None, asset_configs=None):  # pragma: no cover
            raise NotImplementedError

    return _Stub


def test_resolve_asset_configs_reads_yaml_block(tmp_path: Path):
    yaml_path = tmp_path / "task.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            asset_configs:
              pick:
                filter: {role: pick, category: apple}
                name: pick_object
              place:
                filter: {role: place}
                name: place_object
                mass: 100.0
            """
        )
    )
    cls = _stub_task_def_with_config(str(yaml_path))
    cfg = cls.resolve_asset_configs()
    assert cfg is not None
    assert set(cfg.keys()) == {"pick", "place"}
    assert cfg["pick"]["filter"] == {"role": "pick", "category": "apple"}
    assert cfg["pick"]["name"] == "pick_object"
    assert cfg["place"]["mass"] == 100.0


def test_resolve_asset_configs_returns_none_when_absent(tmp_path: Path):
    yaml_path = tmp_path / "task.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            scene:
              type: plane_table
            """
        )
    )
    cls = _stub_task_def_with_config(str(yaml_path))
    assert cls.resolve_asset_configs() is None


def test_resolve_asset_configs_returns_none_when_config_path_unset():
    """TaskDefinition subclasses without config_path return None."""

    class _NoConfigTask(TaskDefinition):
        namespace = "_no_config"
        # config_path intentionally left as base class default (None)

        @classmethod
        def build(cls, resolver=None, asset_configs=None):  # pragma: no cover
            raise NotImplementedError

    assert _NoConfigTask.resolve_asset_configs() is None


def test_place_a2b_yaml_ships_valid_asset_configs():
    """The shipped place_a2b.yaml must carry a working asset_configs block.

    This test guards against accidental regressions to the YAML shape that
    `PlaceA2BTaskDefinition.build(resolver=...)` falls back to.
    """
    yaml_path = (
        Path(__file__).resolve().parents[4]
        / "robo_orchard_sim"
        / "task_suite"
        / "manipulation"
        / "place_a2b"
        / "place_a2b.yaml"
    )
    raw = yaml.safe_load(yaml_path.read_text())
    assert "asset_configs" in raw, (
        f"place_a2b.yaml at {yaml_path} is missing the asset_configs block"
    )
    asset_configs = raw["asset_configs"]
    # Keys must match PlaceA2BTaskAssets.required_object_fields
    assert "pick" in asset_configs
    assert "place" in asset_configs
    for role in ("pick", "place"):
        entry = asset_configs[role]
        assert "filter" in entry, f"{role} entry is missing 'filter'"
        assert "prim_name" in entry, f"{role} entry is missing 'prim_name'"
        assert isinstance(entry["filter"], dict)


@pytest.mark.parametrize(
    "bad_yaml",
    [
        # asset_configs not a mapping
        "asset_configs: [not, a, mapping]",
        # entry not a mapping
        "asset_configs:\n  pick: not-a-dict",
    ],
)
def test_resolve_asset_configs_rejects_malformed_shape(
    tmp_path: Path, bad_yaml: str
):
    yaml_path = tmp_path / "task.yaml"
    yaml_path.write_text(bad_yaml)
    cls = _stub_task_def_with_config(str(yaml_path))
    with pytest.raises(ValidationError):
        cls.resolve_asset_configs()
