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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from robo_orchard_sim.benchmark.manipulation.spatial_pick.spatial_pick_env import (  # noqa: E501
    SpatialPickTaskDefinitionBase,
)
from robo_orchard_sim.orchard_env.assets.object_spec import RigidObjectSpec
from robo_orchard_sim.orchard_env.assets.pool_spec import PoolSpec
from robo_orchard_sim.orchard_env.layout.loader import LayoutValidationError


def _entry(
    pick_cat: str,
    ref_cat: str,
    relation: str = "left_of",
) -> dict:
    return {
        "relation": {
            "spatial_constraints": [
                {
                    "relation": relation,
                    "subject": "src",
                    "anchor": "ref",
                }
            ]
        },
        "position": {
            "src": {
                "category": pick_cat,
                "position": [0.4, -0.1, 0.05],
                "rotation": [1, 0, 0, 0],
            },
            "ref": {
                "category": ref_cat,
                "position": [0.3, 0.1, 0.05],
                "rotation": [1, 0, 0, 0],
            },
        },
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


def _write_caption(path: Path, *, uuid: str, raw: str) -> None:
    path.write_text(
        json.dumps(
            {
                "uuid": uuid,
                "raw": raw,
                "seen": [raw],
            }
        ),
        encoding="utf-8",
    )


def _spec_with_caption(
    tmp_path: Path,
    *,
    name: str,
    category: str,
    uuid: str,
) -> RigidObjectSpec:
    caption_path = tmp_path / f"{uuid}_caption_candidates.json"
    _write_caption(caption_path, uuid=uuid, raw=category)
    return RigidObjectSpec(
        name=name,
        usd_path=f"/d/{name}.usd",
        caption_path=str(caption_path),
        uuid=uuid,
        category=category,
    )


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


@contextmanager
def _patched_build_with_instruction():
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
    assert set(layout_builder.role_member_by_category) == {"src", "ref"}


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


def test_build_layout_mode_preserves_light_and_texture_task_params(tmp_path):
    _write_json(tmp_path, [_entry("garlic", "thermos")])
    yaml = _write_yaml(
        tmp_path,
        extra=(
            "task:\n"
            "  params:\n"
            "    light_reset:\n"
            "      enabled: true\n"
            "      asset_names: [background/dis_light]\n"
            "      distant_light:\n"
            "        asset_name: dis_light\n"
            "      randomize_intensity: true\n"
            "      intensity_range:\n"
            "        range: [1000.0, 5000.0]\n"
            "    texture_reset:\n"
            "      enabled: true\n"
            "      asset_names: [background/table]\n"
        ),
    )
    resolver = _make_resolver(
        {
            "pick": {"garlic": _DUMMY("g")},
            "distractor_0": {"thermos": _DUMMY("t")},
        }
    )
    with _patched_build() as orch:
        SpatialPickTaskDefinitionBase.build(
            resolver=resolver, config_path=str(yaml)
        )

    params = orch.call_args.kwargs["task"].params
    assert params.light_reset is not None
    assert params.light_reset.asset_names == ["background/dis_light"]
    assert params.texture_reset is not None
    assert params.texture_reset.asset_names == ["background/table"]


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
    _write_json(tmp_path, [_entry("garlic", "thermos")])
    yaml = _write_yaml(tmp_path, num_envs=num_envs, extra=extra)
    resolver = _make_resolver(
        {
            "pick": {"garlic": _DUMMY("g")},
            "distractor_0": {"thermos": _DUMMY("t")},
        }
    )
    with pytest.raises(exc_type, match=match):
        SpatialPickTaskDefinitionBase.build(
            resolver=resolver, config_path=str(yaml)
        )


def test_build_asset_configs_overlay_forwards_tags(tmp_path):
    """asset_configs overlay tags reach the resolver as part of the filter."""
    _write_json(tmp_path, [_entry("garlic", "thermos")])
    resolver = _make_resolver(
        {
            "pick": {"garlic": _DUMMY("g")},
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
        SpatialPickTaskDefinitionBase.build(
            resolver=resolver, config_path=str(yaml_path)
        )
    asset_configs = resolver.resolve.call_args[0][0]
    assert asset_configs["pick"]["filter"]["tags"] == ["is_graspable"]
    assert asset_configs["pick"]["filter"]["category"] == "garlic"
    assert asset_configs["distractor_0"]["filter"] == {"category": "thermos"}


def test_build_instruction_context_spatial_relation_renders_instruction(
    tmp_path,
):
    _write_json(tmp_path, [_entry("tomato", "apple", relation="left_of")])
    tomato = _spec_with_caption(
        tmp_path,
        name="tomato_object",
        category="tomato",
        uuid="u-tomato-spatial",
    )
    apple = _spec_with_caption(
        tmp_path,
        name="apple_object",
        category="apple",
        uuid="u-apple-spatial",
    )
    resolver = _make_resolver(
        {
            "pick": {"tomato": tomato},
            "distractor_0": {"apple": apple},
        }
    )
    yaml_path = _write_yaml(
        tmp_path,
        extra=(
            "instruction:\n"
            "  template: spatial_pick_default\n"
            "  template_mode: fixed\n"
            "  actor_description_mode: raw\n"
        ),
    )
    with _patched_build_with_instruction() as orch:
        SpatialPickTaskDefinitionBase.build(
            resolver=resolver,
            config_path=str(yaml_path),
        )
    task = orch.call_args.kwargs["task"]
    env = SimpleNamespace(
        scene={
            task.pick_object.scene_name: SimpleNamespace(
                cfg=task.pick_object.to_isaac_cfg()
            ),
            task.distractors[0].scene_name: SimpleNamespace(
                cfg=task.distractors[0].to_isaac_cfg()
            ),
        }
    )

    instruction = task.instruction.render(
        actors=task.build_instruction_context(
            env,
            actor_description_seed=0,
        )
    )

    assert instruction == "Pick up the tomato to the left of the apple."
