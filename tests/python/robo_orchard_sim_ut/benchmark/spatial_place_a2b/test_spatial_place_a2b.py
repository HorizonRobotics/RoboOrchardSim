# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""SpatialPlaceA2BTaskDefinition build path: layout JSON -> OrchardEnv."""

from __future__ import annotations
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from robo_orchard_sim.benchmark.manipulation.spatial_place_a2b.spatial_place_a2b_env import (  # noqa: E501
    SpatialPlaceA2BTaskDefinitionBase,
)
from robo_orchard_sim.orchard_env.assets.object_spec import RigidObjectSpec
from robo_orchard_sim.orchard_env.assets.pool_spec import PoolSpec
from robo_orchard_sim.orchard_env.layout.loader import LayoutValidationError


def _entry(src_cat: str, dest_cat: str, ref_cat: str) -> dict:
    return {
        "position": {
            "src": {
                "category": src_cat,
                "position": [0.4, -0.1, 0.05],
                "rotation": [1, 0, 0, 0],
            },
            "ref": {
                "category": ref_cat,
                "position": [0.3, 0.1, 0.05],
                "rotation": [1, 0, 0, 0],
            },
            "dest": {
                "category": dest_cat,
                "position": [0.5, 0.2, 0.05],
                "rotation": [1, 0, 0, 0],
            },
        }
    }


def _make_resolver(by_slot_cat: dict[str, dict[str, RigidObjectSpec]]):
    resolver = MagicMock()

    def resolve(asset_configs):
        out = {}
        for key, entry in asset_configs.items():
            slot = key.split("_pool_", 1)[0] if "_pool_" in key else key
            cat = entry["filter"]["category"]
            out[key] = by_slot_cat[slot][cat].model_copy(update={"name": key})
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
            SpatialPlaceA2BTaskDefinitionBase,
            "resolve_scene",
            return_value=MagicMock(),
        ),
        patch.object(
            SpatialPlaceA2BTaskDefinitionBase,
            "resolve_embodiment",
            return_value=MagicMock(),
        ),
        patch.object(
            SpatialPlaceA2BTaskDefinitionBase,
            "resolve_instruction",
            return_value=None,
        ),
        patch("robo_orchard_sim.orchard_env.orchard_env.OrchardEnv") as orch,
    ):
        yield orch


_DUMMY = lambda n: RigidObjectSpec(name=n, usd_path=f"/d/{n}.usd")  # noqa: E731


def test_build_maps_src_to_pick_dest_to_place_ref_to_distractor(tmp_path):
    _write_json(tmp_path, [_entry("bread", "mug", "gum")])
    resolver = _make_resolver(
        {
            "pick": {"bread": _DUMMY("b")},
            "place": {"mug": _DUMMY("m")},
            "distractor_0": {"gum": _DUMMY("g")},
        }
    )
    with _patched_build() as orch:
        SpatialPlaceA2BTaskDefinitionBase.build(
            resolver=resolver, config_path=str(_write_yaml(tmp_path))
        )

    kwargs = orch.call_args.kwargs
    task_assets = kwargs["task"].assets
    assert isinstance(task_assets.pick, RigidObjectSpec)
    assert isinstance(task_assets.place, RigidObjectSpec)
    layout_builder = kwargs["layout_builder"]
    # role_member_by_category keyed by upstream JSON role.
    assert set(layout_builder.role_member_by_category) == {
        "src",
        "ref",
        "dest",
    }


def test_build_multi_category_src_yields_pool_spec(tmp_path):
    _write_json(
        tmp_path,
        [
            _entry("bread", "mug", "gum"),
            _entry("toast", "mug", "gum"),
        ],
    )
    resolver = _make_resolver(
        {
            "pick": {"bread": _DUMMY("b"), "toast": _DUMMY("t")},
            "place": {"mug": _DUMMY("m")},
            "distractor_0": {"gum": _DUMMY("g")},
        }
    )
    with _patched_build() as orch:
        SpatialPlaceA2BTaskDefinitionBase.build(
            resolver=resolver, config_path=str(_write_yaml(tmp_path))
        )
    task_assets = orch.call_args.kwargs["task"].assets
    assert isinstance(task_assets.pick, PoolSpec)


@pytest.mark.parametrize(
    "extra,num_envs,exc_type,match",
    [
        (
            "asset_configs:\n  pick: {filter: {category: x}}\n",
            1,
            LayoutValidationError,
            "category",
        ),
        ("", 4, ValueError, "num_envs"),
    ],
)
def test_build_rejects_invalid_yaml(
    tmp_path, extra, num_envs, exc_type, match
):
    _write_json(tmp_path, [_entry("bread", "mug", "gum")])
    yaml = _write_yaml(tmp_path, num_envs=num_envs, extra=extra)
    resolver = _make_resolver(
        {
            "pick": {"bread": _DUMMY("b")},
            "place": {"mug": _DUMMY("m")},
            "distractor_0": {"gum": _DUMMY("g")},
        }
    )
    with pytest.raises(exc_type, match=match):
        SpatialPlaceA2BTaskDefinitionBase.build(
            resolver=resolver, config_path=str(yaml)
        )


def test_build_asset_configs_overlay_forwards_tags(tmp_path):
    """asset_configs overlay tags reach the resolver as part of the filter."""
    _write_json(tmp_path, [_entry("bread", "plate", "thermos")])
    resolver = _make_resolver(
        {
            "pick": {"bread": _DUMMY("p")},
            "place": {"plate": _DUMMY("q")},
            "distractor_0": {"thermos": _DUMMY("t")},
        }
    )
    yaml_path = _write_yaml(
        tmp_path,
        extra=(
            "asset_configs:\n"
            "  pick:\n"
            "    filter:\n"
            "      tags: [is_graspable]\n"
        ),
    )
    with _patched_build():
        SpatialPlaceA2BTaskDefinitionBase.build(
            resolver=resolver, config_path=str(yaml_path)
        )
    asset_configs = resolver.resolve.call_args[0][0]
    assert asset_configs["pick"]["filter"]["tags"] == ["is_graspable"]
    assert asset_configs["pick"]["filter"]["category"] == "bread"
    assert asset_configs["place"]["filter"] == {"category": "plate"}
