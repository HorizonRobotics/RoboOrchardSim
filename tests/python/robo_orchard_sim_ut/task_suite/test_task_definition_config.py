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
shipped place-a2b YAML variants so their asset_configs blocks stay
well-formed.
"""

from __future__ import annotations
import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from robo_orchard_sim.task_suite import (
    registration as task_registration,
    registry as task_registry,
)
from robo_orchard_sim.task_suite.base import TaskDefinition


def _stub_task_def_with_config(yaml_path: str) -> type[TaskDefinition]:
    """Build a minimal TaskDefinition subclass pointing at a YAML."""

    class _Stub(TaskDefinition):
        namespace = "_stub"
        config_path = yaml_path

        @classmethod
        def build(cls, resolver=None, config_path=None):  # pragma: no cover
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
        def build(cls, resolver=None, config_path=None):  # pragma: no cover
            raise NotImplementedError

    assert _NoConfigTask.resolve_asset_configs() is None


@pytest.mark.parametrize(
    "yaml_name",
    [
        "place_a2b_easy.yaml",
        "place_a2b_hard.yaml",
    ],
)
def test_place_a2b_yaml_ships_valid_asset_configs(yaml_name: str):
    """The shipped place-a2b YAML variants must carry asset_configs.

    This test guards against accidental regressions to the YAML shape that
    `PlaceA2BEasyTaskDefinition.build(resolver=...)` falls back to.
    """
    yaml_path = (
        Path(__file__).resolve().parents[4]
        / "robo_orchard_sim"
        / "task_suite"
        / "manipulation"
        / "place_a2b"
        / "configs"
        / yaml_name
    )
    raw = yaml.safe_load(yaml_path.read_text())
    assert "asset_configs" in raw, (
        f"{yaml_name} at {yaml_path} is missing the asset_configs block"
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
    "yaml_name",
    [
        "place_a2b_easy.yaml",
        "place_a2b_hard.yaml",
    ],
)
def test_place_a2b_yaml_ships_valid_task_pose_range(yaml_name: str):
    yaml_path = (
        Path(__file__).resolve().parents[4]
        / "robo_orchard_sim"
        / "task_suite"
        / "manipulation"
        / "place_a2b"
        / "configs"
        / yaml_name
    )
    raw = yaml.safe_load(yaml_path.read_text())

    assert "task" in raw, (
        f"{yaml_name} at {yaml_path} is missing the task block"
    )
    params = raw["task"]["params"]
    assert "pose_range" in params
    assert "min_separation" in params
    assert "mode" in params
    assert params["mode"] == "random_non_overlap"
    assert params["min_separation"] == 0.03
    pose_range = params["pose_range"]
    assert set(pose_range) == {"x", "y", "z", "roll", "pitch", "yaw"}
    for axis in pose_range.values():
        assert isinstance(axis, list)
        assert len(axis) == 2


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


def test_registration_build_task_forwards_config_path(monkeypatch) -> None:
    class _Stub(TaskDefinition):
        namespace = "_stub_build_task"

        @classmethod
        def build(cls, resolver=None, config_path=None):  # pragma: no cover
            del cls, resolver, config_path
            raise NotImplementedError

    resolver = object()
    recorded: list[tuple[object, str | None]] = []
    monkeypatch.setitem(
        task_registration._TASK_REGISTRY,
        _Stub.namespace,
        _Stub,
    )
    monkeypatch.setattr(
        _Stub,
        "build",
        classmethod(
            lambda cls, resolver=None, config_path=None: (
                recorded.append((resolver, config_path)) or cls
            )
        ),
    )

    built = task_registration.build_task(
        _Stub.namespace,
        resolver=resolver,
        config_path="/tmp/custom.yaml",
    )

    assert built is _Stub
    assert recorded == [(resolver, "/tmp/custom.yaml")]


def test_task_definition_build_atomic_action_plan_returns_empty_list() -> None:
    class _Stub(TaskDefinition):
        namespace = "_stub_empty_plan"

        @classmethod
        def build(cls, resolver=None, config_path=None):  # pragma: no cover
            del cls, resolver, config_path
            raise NotImplementedError

    plan = _Stub.build_atomic_action_plan(object())

    assert plan == []


def test_registration_build_task_atomic_action_plan_forwards_to_definition() -> (  # noqa: E501
    None
):
    orchard_env = object()
    returned_plan = [object()]
    received_envs: list[object] = []

    class _Stub(TaskDefinition):
        namespace = "_stub_atomic_action_plan"

        @classmethod
        def build(cls, resolver=None, config_path=None):  # pragma: no cover
            del cls, resolver, config_path
            raise NotImplementedError

        @classmethod
        def build_atomic_action_plan(cls, env):
            del cls
            received_envs.append(env)
            return returned_plan

    task_registration.register_task(_Stub)

    plan = task_registration.build_task_atomic_action_plan(
        _Stub.namespace,
        orchard_env=orchard_env,
    )

    assert plan is returned_plan
    assert received_envs == [orchard_env]


def test_registry_build_task_bootstraps_and_forwards_config_path(
    monkeypatch,
) -> None:
    bootstrap_calls: list[str] = []
    forwarded_calls: list[tuple[str, object, str | None]] = []
    resolver = object()

    monkeypatch.setattr(
        task_registry,
        "_bootstrap_task_definitions",
        lambda: bootstrap_calls.append("bootstrapped"),
    )
    monkeypatch.setattr(
        task_registry,
        "_build_task",
        lambda task_name, resolver=None, config_path=None: (
            forwarded_calls.append((task_name, resolver, config_path)) or "env"
        ),
    )

    built = task_registry.build_task(
        "place_a2b_easy",
        resolver=resolver,
        config_path="/tmp/custom.yaml",
    )

    assert built == "env"
    assert bootstrap_calls == ["bootstrapped"]
    assert forwarded_calls == [
        ("place_a2b_easy", resolver, "/tmp/custom.yaml")
    ]
