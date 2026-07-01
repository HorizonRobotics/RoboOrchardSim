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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from pxr import Usd
from robo_orchard_core.envs.manager_based_env import ResetEvent
from robo_orchard_core.envs.managers.events.event_manager import (
    EventManagerCfg,
)

from robo_orchard_sim.ext.cfg_wrappers.assets_cfg import RigidObjectCfg
from robo_orchard_sim.ext.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.schemas.schemas_cfg import (
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.from_files import (
    UsdFileCfg,
)
from robo_orchard_sim.ext.envs import (
    IsaacEnvContextManager,
    IsaacManagerBasedEnv,
    IsaacManagerBasedEnvCfg,
)
from robo_orchard_sim.ext.envs.managers.events.texture_reset import (
    TextureResetTerm,
    TextureResetTermCfg,
)
from robo_orchard_sim.ext.models.assets.asset_cfg import GroupAssetCfg
from robo_orchard_sim.ext.models.scenes.table_scene import TableSceneCfg

MUG_VARIANTS_USD = (
    "/horizon-bucket/robot_lab2/assets/Test_assets/mug/variants/variants.usd"
)
DEFAULT_VARIANT = "green"
EXPECTED_RESET_VARIANTS = {"beige", "blue"}
SELECT_USD_VARIANTS_PATCH_TARGET = (
    "robo_orchard_sim.ext.envs.managers.events.texture_reset."
    "sim_utils.select_usd_variants"
)


class _FakePath:
    def __init__(self, path_string: str):
        self.pathString = path_string


class _FakeVariantSetCollection:
    def __init__(self, names: list[str]):
        self._names = names

    def GetNames(self):
        return self._names


class _FakeVariantSet:
    def __init__(self, names: list[str]):
        self._names = names

    def GetVariantNames(self):
        return self._names


class _FakePrim:
    def __init__(
        self,
        path: str,
        variant_names: list[str] | None = None,
        variant_set_name: str = "Look",
    ):
        self._path = path
        self._variant_names = variant_names or []
        self._variant_set_name = variant_set_name

    def GetPath(self):
        return _FakePath(self._path)

    def HasVariantSets(self):
        return bool(self._variant_names)

    def GetVariantSets(self):
        return _FakeVariantSetCollection([self._variant_set_name])

    def GetVariantSet(self, _name: str):
        return _FakeVariantSet(self._variant_names)


class _FakeStage:
    def __init__(self, prims: list[_FakePrim]):
        self._prims = prims
        self._prim_map = {prim.GetPath().pathString: prim for prim in prims}

    def Traverse(self):
        return list(self._prims)

    def GetPrimAtPath(self, path: str):
        return self._prim_map[path]


def _make_logic_term() -> TextureResetTerm:
    term = object.__new__(TextureResetTerm)
    term._cfg = SimpleNamespace(
        variant_set_name="Look",
        variant_sort=True,
        variant_index_range=[0, -1],
        asset_cfgs=[],
    )
    term._env = SimpleNamespace(
        num_envs=2,
        scene={},
        sim=SimpleNamespace(stage=None),
    )
    term.generator = torch.Generator(device="cpu")
    return term


def _make_scene_cfg() -> TableSceneCfg:
    scene_cfg = TableSceneCfg(
        num_envs=1,
        env_spacing=2,
        objects=GroupAssetCfg(
            mug=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.5, 0.0, 0.555), rot=(1.0, 0.0, 0.0, 0.0)
                ),
                spawn=UsdFileCfg(
                    usd_path=MUG_VARIANTS_USD,
                    scale=(1.0, 1.0, 1.0),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
                ),
            )
        ),
    )
    return scene_cfg


def _get_selected_look_variant(
    stage: Usd.Stage,
    root_path: str,
) -> str:
    for prim in stage.Traverse():
        path_str = prim.GetPath().pathString
        if not path_str.startswith(root_path):
            continue
        if not prim.HasVariantSets():
            continue
        variant_sets = prim.GetVariantSets().GetNames()
        if "Look" not in variant_sets:
            continue
        return prim.GetVariantSet("Look").GetVariantSelection()

    raise AssertionError(
        f"No prim with Look variant set found under root path: {root_path}"
    )


class TestTextureResetTerm:
    def test_find_variant_prims_under_filters_requested_env_ids(self):
        term = _make_logic_term()
        stage = _FakeStage(
            [
                _FakePrim(
                    "/World/env_0/Object/Looks/material_0",
                    variant_names=["amber"],
                ),
                _FakePrim(
                    "/World/env_1/Object/Looks/material_0",
                    variant_names=["mint"],
                ),
                _FakePrim(
                    "/World/env_1/Object/Looks/material_0",
                    variant_names=["mint"],
                ),
            ]
        )

        prim_paths = term._find_variant_prims_under(
            stage,
            "/World/env_.*/Object",
            env_ids=[1],
        )

        assert prim_paths == ["/World/env_1/Object/Looks/material_0"]

    def test_find_variant_prims_under_without_env_ids_returns_all_envs(self):
        term = _make_logic_term()
        stage = _FakeStage(
            [
                _FakePrim(
                    "/World/env_1/Object/Looks/material_0",
                    variant_names=["mint"],
                ),
                _FakePrim(
                    "/World/env_0/Object/Looks/material_0",
                    variant_names=["amber"],
                ),
                _FakePrim(
                    "/World/env_0/Object/Looks/material_0",
                    variant_names=["amber"],
                ),
            ]
        )

        prim_paths = term._find_variant_prims_under(
            stage,
            "/World/env_.*/Object",
        )

        assert prim_paths == [
            "/World/env_0/Object/Looks/material_0",
            "/World/env_1/Object/Looks/material_0",
        ]

    def test_call_defaults_missing_seed_to_zero_and_forwards_env_ids(
        self,
    ):
        term = _make_logic_term()
        event = ResetEvent(seed=None, env_ids=[1])

        with patch.object(term, "_select_asset_variants") as select_variants:
            term(event)

        select_variants.assert_called_once_with(
            term._env.sim.stage,
            generator=term.generator,
            env_ids=[1],
        )
        expected_generator = torch.Generator(device="cpu")
        expected_generator.manual_seed(0)
        assert (
            torch.randint(
                low=0,
                high=1000,
                size=(1,),
                generator=term.generator,
            ).item()
            == torch.randint(
                low=0,
                high=1000,
                size=(1,),
                generator=expected_generator,
            ).item()
        )

    def test_call_seeds_generator_from_explicit_seed(self):
        term = _make_logic_term()
        event = ResetEvent(seed=123, env_ids=[1])

        with patch.object(term, "_select_asset_variants") as select_variants:
            term(event)

        select_variants.assert_called_once_with(
            term._env.sim.stage,
            generator=term.generator,
            env_ids=[1],
        )
        expected_generator = torch.Generator(device="cpu")
        expected_generator.manual_seed(123)
        assert (
            torch.randint(
                low=0,
                high=1000,
                size=(1,),
                generator=term.generator,
            ).item()
            == torch.randint(
                low=0,
                high=1000,
                size=(1,),
                generator=expected_generator,
            ).item()
        )

    def test_call_propagates_variant_selection_error(self):
        term = _make_logic_term()
        term._select_asset_variants = lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(  # noqa: E501
            RuntimeError("boom")
        )

        with pytest.raises(RuntimeError, match="boom"):
            term(ResetEvent(seed=1, env_ids=[0]))

    def test_select_asset_variants_skips_invalid_variant_range(self):
        term = _make_logic_term()
        term._cfg.asset_cfgs = [SimpleNamespace(name="objects/mug")]
        term._cfg.variant_index_range = [2, 2]
        term._env.scene = {
            "objects/mug": SimpleNamespace(
                cfg=SimpleNamespace(prim_path="/World/env_.*/Object")
            )
        }
        stage = _FakeStage(
            [
                _FakePrim(
                    "/World/env_0/Object/Looks/material_0",
                    variant_names=["amber", "mint"],
                )
            ]
        )

        with patch(SELECT_USD_VARIANTS_PATCH_TARGET) as select_variants:
            term._select_asset_variants(stage, term.generator, env_ids=[0])

        select_variants.assert_not_called()

    def test_select_asset_variants_uses_sorted_filtered_variants(self):
        term = _make_logic_term()
        term._cfg.asset_cfgs = [SimpleNamespace(name="objects/mug")]
        term._cfg.variant_index_range = [1, 3]
        term._env.scene = {
            "objects/mug": SimpleNamespace(
                cfg=SimpleNamespace(prim_path="/World/env_.*/Object")
            )
        }
        stage = _FakeStage(
            [
                _FakePrim(
                    "/World/env_0/Object/Looks/material_0",
                    variant_names=["green", "beige", "blue", "amber"],
                )
            ]
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(7)
        expected_generator = torch.Generator(device="cpu")
        expected_generator.manual_seed(7)
        expected_variants = ["amber", "beige", "blue", "green"][1:3]
        expected_choice = expected_variants[
            torch.randint(
                low=0,
                high=len(expected_variants),
                size=(1,),
                generator=expected_generator,
            ).item()
        ]

        with patch(SELECT_USD_VARIANTS_PATCH_TARGET) as select_variants:
            term._select_asset_variants(stage, generator, env_ids=[0])

        select_variants.assert_called_once_with(
            prim_path="/World/env_0/Object/Looks/material_0",
            variants={"Look": expected_choice},
        )

    def test_select_asset_variants_skips_when_asset_cfgs_is_empty(self):
        term = _make_logic_term()
        stage = _FakeStage([])

        with patch(SELECT_USD_VARIANTS_PATCH_TARGET) as select_variants:
            term._select_asset_variants(stage, term.generator, env_ids=[0])

        select_variants.assert_not_called()

    def test_select_asset_variants_only_updates_requested_envs(self):
        term = _make_logic_term()
        term._cfg.asset_cfgs = [SimpleNamespace(name="objects/mug")]
        term._cfg.variant_index_range = [0, -1]
        term._env.scene = {
            "objects/mug": SimpleNamespace(
                cfg=SimpleNamespace(prim_path="/World/env_.*/Object")
            )
        }
        stage = _FakeStage(
            [
                _FakePrim(
                    "/World/env_0/Object/Looks/material_0",
                    variant_names=["amber", "mint"],
                ),
                _FakePrim(
                    "/World/env_1/Object/Looks/material_0",
                    variant_names=["blue", "green"],
                ),
            ]
        )

        with patch(SELECT_USD_VARIANTS_PATCH_TARGET) as select_variants:
            term._select_asset_variants(stage, term.generator, env_ids=[1])

        select_variants.assert_called_once()
        _, kwargs = select_variants.call_args
        assert kwargs["prim_path"] == "/World/env_1/Object/Looks/material_0"

    def test_texture_reset_changes_real_asset_look_variant_on_reset(self):
        if not Path(MUG_VARIANTS_USD).exists():
            pytest.skip(
                f"mug variants USD not found at {MUG_VARIANTS_USD}; "
                "set ORCHARD_ASSET so that "
                "$ORCHARD_ASSET/Test_assets/mug/variants/variants.usd exists"
            )
        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=_make_scene_cfg(),
            events=EventManagerCfg(
                terms={
                    "texture_reset": TextureResetTermCfg(
                        trigger_topic="reset",
                        asset_cfgs=[SceneEntityCfg(name="objects/mug")],
                        variant_set_name="Look",
                        variant_index_range=[0, 2],
                    )
                },
            ),
        )

        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            _ = env.step()

            object_root_path = env.scene["objects/mug"].cfg.prim_path.replace(
                "env_.*", "env_0"
            )
            initial_variant = _get_selected_look_variant(
                env.sim.stage,
                object_root_path,
            )
            assert initial_variant == DEFAULT_VARIANT

            env.reset(seed=123)
            for _ in range(5):
                _ = env.step()

            reset_variant = _get_selected_look_variant(
                env.sim.stage,
                object_root_path,
            )
            assert reset_variant in EXPECTED_RESET_VARIANTS
            assert reset_variant != initial_variant
