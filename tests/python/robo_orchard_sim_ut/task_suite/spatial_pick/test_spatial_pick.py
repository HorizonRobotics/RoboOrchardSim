# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""SpatialPickTaskDefinition build path: layout JSON → OrchardEnv."""

from __future__ import annotations
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from robo_orchard_sim.orchard_env.assets.object_spec import RigidObjectSpec
from robo_orchard_sim.orchard_env.assets.pool_spec import PoolSpec
from robo_orchard_sim.task_suite.manipulation.spatial_pick.spatial_pick_env import (  # noqa: E501
    SpatialPickTaskDefinitionBase,
)


def _entry(pick_cat: str, ref_cat: str) -> dict:
    return {
        "asset_bindings": {
            "pick": {
                "category": pick_cat,
                "position": [0.4, -0.1, 0.05],
                "rotation": [1, 0, 0, 0],
            },
            "ref": {
                "category": ref_cat,
                "position": [0.3, 0.1, 0.05],
                "rotation": [1, 0, 0, 0],
            },
        }
    }


def _make_resolver(by_role_cat: dict[str, dict[str, RigidObjectSpec]]):
    resolver = MagicMock()

    def resolve(asset_configs):
        out = {}
        for key, entry in asset_configs.items():
            role = key.split("_pool_", 1)[0] if "_pool_" in key else key
            cat = entry["filter"]["category"]
            out[key] = by_role_cat[role][cat].model_copy(update={"name": key})
        return out

    resolver.resolve.side_effect = resolve
    return resolver


def _write_yaml(tmp_path: Path, **kwargs) -> Path:
    num_envs = kwargs.get("num_envs", 1)
    extra = kwargs.get("extra", "")
    yaml = tmp_path / "task.yaml"
    yaml.write_text(
        f"scene:\n  type: empty\n  num_envs: {num_envs}\n"
        f"embodiment:\n  type: dualarm_piper\n"
        f"layout: layout.json\n{extra}"
    )
    return yaml


def _write_json(tmp_path: Path, entries: list[dict]) -> Path:
    p = tmp_path / "layout.json"
    p.write_text(json.dumps(entries))
    return p


@contextmanager
def _patched_build():
    with (
        patch.object(
            SpatialPickTaskDefinitionBase,
            "resolve_scene",
            return_value=MagicMock(),
        ),
        patch.object(
            SpatialPickTaskDefinitionBase,
            "resolve_embodiment",
            return_value=MagicMock(),
        ),
        patch.object(
            SpatialPickTaskDefinitionBase,
            "resolve_instruction",
            return_value=None,
        ),
        patch("robo_orchard_sim.orchard_env.orchard_env.OrchardEnv") as orch,
    ):
        yield orch


_DUMMY = lambda n: RigidObjectSpec(name=n, usd_path=f"/d/{n}.usd")  # noqa: E731


def test_build_multi_category_pick_yields_pool_spec(tmp_path):
    _write_json(
        tmp_path,
        [_entry("garlic", "thermos"), _entry("potato", "thermos")],
    )
    resolver = _make_resolver(
        {
            "pick": {"garlic": _DUMMY("g"), "potato": _DUMMY("p")},
            "distractor_0": {"thermos": _DUMMY("t")},
        }
    )
    with _patched_build() as orch:
        SpatialPickTaskDefinitionBase.build(
            resolver=resolver, config_path=str(_write_yaml(tmp_path))
        )

    kwargs = orch.call_args.kwargs
    pick_assets = kwargs["task"].assets
    assert isinstance(pick_assets.pick, PoolSpec)
    assert {m.scene_name for m in pick_assets.pick.members} == {
        "objects/pick_pool_0",
        "objects/pick_pool_1",
    }
    layout_builder = kwargs["layout_builder"]
    assert layout_builder.num_episodes == 2
    # role_member_by_category is keyed by LAYOUT-JSON role (what
    # LayoutResetTerm sees), not the task slot — LayoutResetTerm iterates
    # ``layout.objects.items()`` to look up actors per episode.
    assert set(layout_builder.role_member_by_category) == {"pick", "ref"}


def test_build_single_category_pick_yields_object_spec(tmp_path):
    _write_json(
        tmp_path,
        [_entry("garlic", "thermos"), _entry("garlic", "thermos")],
    )
    resolver = _make_resolver(
        {
            "pick": {"garlic": _DUMMY("g")},
            "distractor_0": {"thermos": _DUMMY("t")},
        }
    )
    with _patched_build() as orch:
        SpatialPickTaskDefinitionBase.build(
            resolver=resolver, config_path=str(_write_yaml(tmp_path))
        )
    pick_assets = orch.call_args.kwargs["task"].assets
    assert isinstance(pick_assets.pick, RigidObjectSpec)


@pytest.mark.parametrize(
    "extra,num_envs,match",
    [
        (
            "asset_configs:\n  pick: {filter: {category: x}}\n",
            1,
            "asset_configs",
        ),
        ("", 4, "num_envs"),
    ],
)
def test_build_rejects_invalid_yaml(tmp_path, extra, num_envs, match):
    _write_json(tmp_path, [_entry("garlic", "thermos")])
    yaml = _write_yaml(tmp_path, num_envs=num_envs, extra=extra)
    resolver = _make_resolver(
        {
            "pick": {"garlic": _DUMMY("g")},
            "distractor_0": {"thermos": _DUMMY("t")},
        }
    )
    with pytest.raises(ValueError, match=match):
        SpatialPickTaskDefinitionBase.build(
            resolver=resolver, config_path=str(yaml)
        )
