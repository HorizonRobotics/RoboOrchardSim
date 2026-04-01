# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from robo_orchard_core.envs.managers.events.event_manager import (
    EventManagerCfg,
)

import robo_orchard_sim.envs.managers.events.pose_reset as pose_reset_mod
from robo_orchard_sim.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.envs import (
    IsaacEnvContextManager,
    IsaacManagerBasedEnv,
    IsaacManagerBasedEnvCfg,
)
from robo_orchard_sim.envs.managers.events.pose_reset import (
    PoseResetTerm,
    PoseResetTermCfg,
)
from robo_orchard_sim.models.scenes.table_scene import TableSceneCfg
from robo_orchard_sim.utils.usd import get_prim_aabb


def _make_root_state(
    num_envs: int = 1,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
) -> torch.Tensor:
    root_state = torch.zeros((num_envs, 13), dtype=torch.float32)
    root_state[:, 0] = x
    root_state[:, 1] = y
    root_state[:, 2] = z
    root_state[:, 3] = 1.0
    return root_state


class _FakeAsset:
    def __init__(self, tag: str, default_root_state: torch.Tensor):
        self.cfg = SimpleNamespace(
            prim_path=f"/World/env_.*/{tag}",
            spawn=SimpleNamespace(semantic_tags=[("class", tag)]),
        )
        self.data = SimpleNamespace(default_root_state=default_root_state)
        self.pose_write = None
        self.velocity_write = None

    def write_root_pose_to_sim(self, pose: torch.Tensor, env_ids=None):
        env_ids_out = None if env_ids is None else env_ids.clone()
        self.pose_write = (pose.clone(), env_ids_out)

    def write_root_velocity_to_sim(self, velocity: torch.Tensor, env_ids=None):
        env_ids_out = None if env_ids is None else env_ids.clone()
        self.velocity_write = (velocity.clone(), env_ids_out)


def _make_logic_term(
    mode: str,
    *,
    pose_range: dict[str, tuple[float, float]] | None = None,
    absolute_sampling: bool = False,
    min_separation: float = 0.0,
    max_retries: int = 4,
    group_key: str | None = None,
    clear_cross_group_cache: bool = False,
    prefer_stack: bool = False,
) -> PoseResetTerm:
    term = object.__new__(PoseResetTerm)
    term._cfg = SimpleNamespace(
        mode=mode,
        pose_range=pose_range
        or {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        min_separation=min_separation,
        max_retries=max_retries,
        absolute_sampling=absolute_sampling,
        group_key=group_key,
        clear_cross_group_cache=clear_cross_group_cache,
        prefer_stack=prefer_stack,
    )
    term._env = SimpleNamespace(
        device="cpu",
        scene=SimpleNamespace(
            env_origins=torch.zeros((1, 3), dtype=torch.float32),
        ),
    )
    term._mode = mode
    term._count = 0
    term._group_key = group_key
    term._clear_cross_group_cache = clear_cross_group_cache
    term._prefer_stack = prefer_stack
    term._asset_xy_extents = {}
    term._asset_z_half_extents = {}
    term._assets = []
    return term


def _get_asset_half_extents(
    env: IsaacManagerBasedEnv, asset_name: str
) -> tuple[float, float, float]:
    prim_path = env.scene[asset_name].cfg.prim_path.replace("env_.*", "env_0")
    aabb = get_prim_aabb(env.sim.stage, prim_path)
    assert aabb is not None, f"Failed to load AABB for {asset_name}"
    (x_max, x_min), (y_max, y_min), (z_max, z_min) = aabb
    return (
        abs(x_max - x_min) * 0.5,
        abs(y_max - y_min) * 0.5,
        abs(z_max - z_min) * 0.5,
    )


def _get_root_position(
    env: IsaacManagerBasedEnv, asset_name: str
) -> torch.Tensor:
    return env.scene[asset_name].data.root_state_w[0, :3].clone()


def _prepare_non_overlap_scene(
    scene_cfg: TableSceneCfg,
) -> TableSceneCfg:
    ret = scene_cfg.copy()
    ret.objects["cube1"].spawn.rigid_props.disable_gravity = True
    ret.objects["cube1"].spawn.semantic_tags = [("class", "cube1")]
    ret.objects["cube2"].spawn = ret.objects["cube2"].spawn.copy()
    ret.objects["cube2"].spawn.semantic_tags = [("class", "cube2")]
    return ret


class TestResetEventTerm:
    def teardown_method(self):
        pose_reset_mod._CROSS_GROUP_CACHE.clear()

    def test_random_non_overlap_absolute_sampling_ignores_default_root_state(
        self,
    ):
        term = _make_logic_term(
            "random_non_overlap",
            pose_range={"x": (0.4, 0.4), "y": (0.5, 0.5), "z": (0.0, 0.0)},
            absolute_sampling=True,
        )
        asset = _FakeAsset("asset_a", _make_root_state(x=1.0, y=2.0, z=3.0))
        term._assets = [asset]
        term._asset_xy_extents = {"asset_a": (0.1, 0.1)}
        term._uniform_sample_pose = lambda: [0.4, 0.5, 0.9, 1.0, 0.0, 0.0, 0.0]

        term._apply_random_non_overlap(torch.tensor([0]))

        pose, env_ids = asset.pose_write
        assert torch.equal(env_ids, torch.tensor([0]))
        assert torch.allclose(
            pose[0, :3],
            torch.tensor([0.4, 0.5, 0.0]),
        )

    def test_pose_reset_event_random_mode(
        self,
        simple_table_scene_cfg: TableSceneCfg,
    ):
        ret = simple_table_scene_cfg.copy()
        ret.objects["cube1"].spawn.rigid_props.disable_gravity = True

        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=ret,
            events=EventManagerCfg(
                terms={
                    "rest_to_default": PoseResetTermCfg(
                        mode="random",
                        trigger_topic=IsaacManagerBasedEnv.RESET[0],
                        pose_range={"x": [-0.2, 0.2], "y": [-0.2, 0.2]},
                    )
                },
            ),
        )
        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            origin_pose = env.scene["objects/cube1"].data.root_state_w[:, :7]
            _ = env.step()
            env.reset()
            _ = env.step()
            after_pose = env.scene["objects/cube1"].data.root_state_w[:, :7]
            print("origin_pose: ", origin_pose)
            print("after_pose: ", after_pose)
            assert not torch.allclose(
                origin_pose, after_pose, rtol=1e-05, atol=1e-08
            ), "pose_reset_event random mode failed."

    def test_pose_reset_event_default_mode(
        self,
        simple_table_scene_cfg: TableSceneCfg,
    ):
        ret = simple_table_scene_cfg.copy()
        ret.objects["cube1"].spawn.rigid_props.disable_gravity = True

        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=ret,
            events=EventManagerCfg(
                terms={
                    "rest_to_default": PoseResetTermCfg(
                        mode="default",
                        trigger_topic=IsaacManagerBasedEnv.RESET[0],
                    )
                },
            ),
        )
        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            origin_pose = env.scene["objects/cube1"].data.root_state_w[:, :7]
            _ = env.step()
            env.reset()
            _ = env.step()
            after_pose = env.scene["objects/cube1"].data.root_state_w[:, :7]
            print("origin_pose: ", origin_pose)
            print("after_pose: ", after_pose)
            assert torch.allclose(
                origin_pose, after_pose, rtol=1e-02, atol=1e-04
            ), "pose_reset_event default mode failed."

    def test_reset_cube2(
        self,
        simple_two_object_scene_cfg: TableSceneCfg,
    ):
        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=simple_two_object_scene_cfg,
            events=EventManagerCfg(
                terms={
                    "rest_cube2": PoseResetTermCfg(
                        asset_cfgs=[
                            SceneEntityCfg(
                                name="objects/cube2",
                            ),
                        ],
                        mode="default",
                        trigger_topic=IsaacManagerBasedEnv.RESET[0],
                    )
                },
            ),
        )
        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            cube1_origin_pose = env.scene["objects/cube1"].data.root_state_w[
                :, :7
            ]
            cube2_origin_pose = env.scene["objects/cube2"].data.root_state_w[
                :, :7
            ]

            for _ in range(10):
                _ = env.step()

            env.reset()

            cube1_after_pose = env.scene["objects/cube1"].data.root_state_w[
                :, :7
            ]
            cube2_after_pose = env.scene["objects/cube2"].data.root_state_w[
                :, :7
            ]

            assert not torch.allclose(
                cube1_origin_pose, cube1_after_pose, rtol=1e-01, atol=1e-03
            ), "test_reset_cube2 cube1 state failed."

            assert torch.allclose(
                cube2_origin_pose, cube2_after_pose, rtol=1e-01, atol=1e-03
            ), "test_reset_cube2 cube2 reset failed."

    def test_random_non_overlap_mode_keeps_assets_separated(
        self,
        simple_two_object_scene_cfg: TableSceneCfg,
    ):
        ret = _prepare_non_overlap_scene(simple_two_object_scene_cfg)

        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=ret,
            events=EventManagerCfg(
                terms={
                    "rest_non_overlap": PoseResetTermCfg(
                        mode="random_non_overlap",
                        trigger_topic=IsaacManagerBasedEnv.RESET[0],
                        pose_range={
                            "x": [-0.06, 0.06],
                            "y": [-0.06, 0.06],
                            "z": [0.0, 0.0],
                        },
                        min_separation=0.02,
                        max_retries=64,
                        clear_cross_group_cache=True,
                    )
                },
            ),
        )
        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            _ = env.step()

            env.reset(seed=7)
            _ = env.step()

            cube1_pos = _get_root_position(env, "objects/cube1")
            cube2_pos = _get_root_position(env, "objects/cube2")
            cube1_hx, cube1_hy, _ = _get_asset_half_extents(
                env, "objects/cube1"
            )
            cube2_hx, cube2_hy, _ = _get_asset_half_extents(
                env, "objects/cube2"
            )
            dx = abs(float(cube1_pos[0] - cube2_pos[0]))
            dy = abs(float(cube1_pos[1] - cube2_pos[1]))

            assert dx >= (cube1_hx + cube2_hx + 0.02) or dy >= (
                cube1_hy + cube2_hy + 0.02
            ), "random_non_overlap mode placed two assets with XY overlap."

    def test_drop_mode_stacks_assets_when_xy_ranges_overlap(
        self,
        simple_two_object_scene_cfg: TableSceneCfg,
    ):
        ret = _prepare_non_overlap_scene(simple_two_object_scene_cfg)

        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=ret,
            events=EventManagerCfg(
                terms={
                    "rest_drop": PoseResetTermCfg(
                        mode="drop",
                        trigger_topic=IsaacManagerBasedEnv.RESET[0],
                        pose_range={
                            "x": [0.0, 0.0],
                            "y": [0.0, 0.0],
                            "z": [0.0, 0.4],
                        },
                        max_retries=64,
                        clear_cross_group_cache=True,
                    )
                },
            ),
        )
        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            _ = env.step()

            env.reset(seed=11)
            _ = env.step()

            cube1_pos = _get_root_position(env, "objects/cube1")
            cube2_pos = _get_root_position(env, "objects/cube2")
            _, _, cube1_hz = _get_asset_half_extents(env, "objects/cube1")
            _, _, cube2_hz = _get_asset_half_extents(env, "objects/cube2")
            z_gap = abs(float(cube1_pos[2] - cube2_pos[2]))

            assert z_gap >= (cube1_hz + cube2_hz - 1e-3), (
                "drop mode did not stack overlapping assets vertically."
            )

    def test_drop_mode_prefer_stack_stacks_above_existing_assets(
        self,
    ):
        term = _make_logic_term(
            "drop",
            pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.05)},
            prefer_stack=True,
        )
        asset_a = _FakeAsset("asset_a", _make_root_state())
        asset_b = _FakeAsset("asset_b", _make_root_state())
        term._assets = [asset_a, asset_b]
        term._asset_xy_extents = {
            "asset_a": (0.1, 0.1),
            "asset_b": (0.1, 0.1),
        }
        term._asset_z_half_extents = {
            "asset_a": 0.2,
            "asset_b": 0.2,
        }
        term._uniform_sample_pose = lambda: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        with patch.object(
            pose_reset_mod.random,
            "sample",
            return_value=[asset_a, asset_b],
        ):
            term._apply_drop_reset(torch.tensor([0]))

        pose_a = asset_a.pose_write[0]
        pose_b = asset_b.pose_write[0]
        assert torch.isclose(pose_a[0, 2], torch.tensor(0.2))
        assert torch.isclose(pose_b[0, 2], torch.tensor(0.6))

    def test_drop_mode_group_key_coordinates_across_terms(self):
        term_a = _make_logic_term(
            "drop",
            pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.4)},
            group_key="shared_drop_group",
            clear_cross_group_cache=True,
        )
        asset_a = _FakeAsset("asset_a", _make_root_state())
        term_a._assets = [asset_a]
        term_a._asset_xy_extents = {"asset_a": (0.1, 0.1)}
        term_a._asset_z_half_extents = {"asset_a": 0.2}
        term_a._uniform_sample_pose = lambda: [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]

        term_b = _make_logic_term(
            "drop",
            pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.8)},
            group_key="shared_drop_group",
        )
        asset_b = _FakeAsset("asset_b", _make_root_state())
        term_b._assets = [asset_b]
        term_b._asset_xy_extents = {"asset_b": (0.1, 0.1)}
        term_b._asset_z_half_extents = {"asset_b": 0.2}
        term_b._uniform_sample_pose = lambda: [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]

        term_a._apply_drop_reset(torch.tensor([0]))
        term_b._apply_drop_reset(torch.tensor([0]))

        pose = asset_b.pose_write[0]
        assert torch.isclose(pose[0, 2], torch.tensor(0.6))
        assert (
            len(pose_reset_mod._CROSS_GROUP_CACHE["shared_drop_group"][0]) == 2
        )

    def test_group_key_coordinates_non_overlap_across_terms(
        self,
        simple_two_object_scene_cfg: TableSceneCfg,
    ):
        ret = _prepare_non_overlap_scene(simple_two_object_scene_cfg)

        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=ret,
            events=EventManagerCfg(
                terms={
                    "rest_cube1": PoseResetTermCfg(
                        asset_cfgs=[SceneEntityCfg(name="objects/cube1")],
                        mode="random_non_overlap",
                        trigger_topic=IsaacManagerBasedEnv.RESET[0],
                        pose_range={
                            "x": [-0.06, 0.06],
                            "y": [-0.06, 0.06],
                            "z": [0.0, 0.0],
                        },
                        min_separation=0.02,
                        max_retries=64,
                        group_key="shared_reset_group",
                        clear_cross_group_cache=True,
                    ),
                    "rest_cube2": PoseResetTermCfg(
                        asset_cfgs=[SceneEntityCfg(name="objects/cube2")],
                        mode="random_non_overlap",
                        trigger_topic=IsaacManagerBasedEnv.RESET[0],
                        pose_range={
                            "x": [-0.06, 0.06],
                            "y": [-0.06, 0.06],
                            "z": [0.0, 0.0],
                        },
                        min_separation=0.02,
                        max_retries=64,
                        group_key="shared_reset_group",
                    ),
                },
            ),
        )
        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            _ = env.step()

            env.reset(seed=23)
            _ = env.step()

            cube1_pos = _get_root_position(env, "objects/cube1")
            cube2_pos = _get_root_position(env, "objects/cube2")
            cube1_hx, cube1_hy, _ = _get_asset_half_extents(
                env, "objects/cube1"
            )
            cube2_hx, cube2_hy, _ = _get_asset_half_extents(
                env, "objects/cube2"
            )
            dx = abs(float(cube1_pos[0] - cube2_pos[0]))
            dy = abs(float(cube1_pos[1] - cube2_pos[1]))

            assert dx >= (cube1_hx + cube2_hx + 0.02) or dy >= (
                cube1_hy + cube2_hy + 0.02
            ), "group_key did not prevent overlap across separate reset terms."

    def test_group_key_non_overlap_handles_partial_env_ids(self):
        term_a = _make_logic_term(
            "random_non_overlap",
            group_key="shared_group",
        )
        asset_a = _FakeAsset("asset_a", _make_root_state(num_envs=2))
        term_a._env.scene.env_origins = torch.zeros(
            (2, 3), dtype=torch.float32
        )
        term_a._assets = [asset_a]
        term_a._asset_xy_extents = {"asset_a": (0.1, 0.1)}

        term_b = _make_logic_term(
            "random_non_overlap",
            group_key="shared_group",
            max_retries=1,
        )
        asset_b = _FakeAsset("asset_b", _make_root_state(num_envs=2))
        term_b._env.scene.env_origins = torch.zeros(
            (2, 3), dtype=torch.float32
        )
        term_b._assets = [asset_b]
        term_b._asset_xy_extents = {"asset_b": (0.1, 0.1)}

        term_a._apply_random_non_overlap(torch.tensor([1]))
        term_b._apply_random_non_overlap(torch.tensor([0]))

        assert asset_b.pose_write is not None

    def test_drop_mode_group_key_handles_partial_env_ids(self):
        term = _make_logic_term(
            "drop",
            group_key="shared_drop_group",
            pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.4)},
        )
        asset = _FakeAsset("asset_a", _make_root_state(num_envs=2))
        term._env.scene.env_origins = torch.zeros((2, 3), dtype=torch.float32)
        term._assets = [asset]
        term._asset_xy_extents = {"asset_a": (0.1, 0.1)}
        term._asset_z_half_extents = {"asset_a": 0.1}

        term._apply_drop_reset(torch.tensor([1]))
        term._apply_drop_reset(torch.tensor([0, 1]))

    def test_drop_mode_group_key_preserves_stacked_top_height(self):
        term_a = _make_logic_term(
            "drop",
            group_key="stack_group",
            clear_cross_group_cache=True,
            pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 1.5)},
        )
        asset_a = _FakeAsset("asset_a", _make_root_state())
        asset_b = _FakeAsset("asset_b", _make_root_state())
        term_a._assets = [asset_a, asset_b]
        term_a._asset_xy_extents = {
            "asset_a": (0.1, 0.1),
            "asset_b": (0.1, 0.1),
        }
        term_a._asset_z_half_extents = {
            "asset_a": 0.2,
            "asset_b": 0.2,
        }

        term_b = _make_logic_term(
            "drop",
            group_key="stack_group",
            pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 1.5)},
        )
        asset_c = _FakeAsset("asset_c", _make_root_state())
        term_b._assets = [asset_c]
        term_b._asset_xy_extents = {"asset_c": (0.1, 0.1)}
        term_b._asset_z_half_extents = {"asset_c": 0.2}

        with patch.object(
            pose_reset_mod.random,
            "sample",
            side_effect=[[asset_a, asset_b], [asset_c]],
        ):
            term_a._apply_drop_reset(torch.tensor([0]))
            term_b._apply_drop_reset(torch.tensor([0]))

        pose = asset_c.pose_write[0]
        assert torch.isclose(pose[0, 2], torch.tensor(1.0))

    def test_drop_mode_rejects_top_surface_above_z_max(self):
        pose_reset_mod._CROSS_GROUP_CACHE["zmax_group"] = {
            0: [("base", (0.0, 0.0), (0.1, 0.1), 0.15, 0.3)]
        }

        term = _make_logic_term(
            "drop",
            group_key="zmax_group",
            pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.4)},
            max_retries=1,
        )
        asset = _FakeAsset("asset_a", _make_root_state())
        term._assets = [asset]
        term._asset_xy_extents = {"asset_a": (0.1, 0.1)}
        term._asset_z_half_extents = {"asset_a": 0.1}

        with patch.object(
            pose_reset_mod.random,
            "sample",
            return_value=[asset],
        ):
            term._apply_drop_reset(torch.tensor([0]))

        pose = asset.pose_write[0]
        assert torch.isclose(pose[0, 2], torch.tensor(0.0))


if __name__ == "__main__":
    pytest.main(["-s", "test_pose_reset_events.py"])
