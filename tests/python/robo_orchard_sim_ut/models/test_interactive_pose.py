# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
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

import pytest
import torch

from robo_orchard_sim.cfg_wrappers.envs.env_cfg import SimulationCfg
from robo_orchard_sim.cfg_wrappers.sim.schemas.schemas_cfg import (
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.cfg_wrappers.sim.spawners.from_files import (
    UsdFileCfg,
)
from robo_orchard_sim.models.assets.asset_cfg import NV_ISAAC_DIR
from robo_orchard_sim.models.assets.rigid_object import RigidObjectCfg
from robo_orchard_sim.models.scenes.interactive_scene import InteractiveScene
from robo_orchard_sim.models.scenes.table_scene import (
    GroupAssetCfg,
    TableSceneCfg,
)
from robo_orchard_sim.sim_ctx import SimulationContextManager

INTERACTION_JSON_REL_PATH = (
    "tests/python/robo_orchard_sim_ut/models/test_interaction.json"
)
INTERACTION_JSON_ABS_PATH = (
    Path(__file__).parents[4] / INTERACTION_JSON_REL_PATH
)


@pytest.fixture(scope="module")
def scene_with_interactive_cube():
    usd_cfg = UsdFileCfg(
        usd_path=f"{NV_ISAAC_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",  # noqa: E501
        scale=(0.8, 0.8, 0.8),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    )
    scene_cfg = TableSceneCfg(
        num_envs=1,
        env_spacing=2,
        objects=GroupAssetCfg(
            cube1=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.5, 0, 0.555), rot=(1, 0, 0, 0)
                ),
                spawn=usd_cfg,
                interaction_path=str(INTERACTION_JSON_ABS_PATH),
            )
        ),
    )
    return scene_cfg


class TestInteractivePose:
    def _create_scene(self, scene_cfg: TableSceneCfg):
        sim_cfg = SimulationCfg(dt=0.01)
        with SimulationContextManager(
            cfg=sim_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as sim:
            scene = InteractiveScene(scene_cfg)
            sim.reset()
            sim_dt = sim.get_physics_dt()
            for _ in range(2):
                sim.step()
                scene.update(sim_dt)
            yield scene

    def test_get_element_pose_return(self, scene_with_interactive_cube):
        for scene in self._create_scene(scene_with_interactive_cube):
            obj = scene.rigid_objects["objects/cube1"]
            pose_data = obj.get_element_pose(
                mode="passive", action="pick", part="up"
            )

            num_envs = obj.data.root_pos_w.shape[0]
            assert pose_data.pos.shape == (num_envs, 4, 3)
            assert pose_data.quat.shape == (num_envs, 4, 4)
            assert torch.isfinite(pose_data.pos).all()
            assert torch.isfinite(pose_data.quat).all()

    def test_get_axis_returns_unit_vectors(self, scene_with_interactive_cube):
        for scene in self._create_scene(scene_with_interactive_cube):
            obj = scene.rigid_objects["objects/cube1"]
            pose_data = obj.get_element_pose(
                mode="passive", action="pick", part="up"
            )

            x_axis = pose_data.get_axis("x")
            y_axis = pose_data.get_axis("y")
            z_axis = pose_data.get_axis("z")

            assert x_axis.shape == pose_data.pos.shape
            assert y_axis.shape == pose_data.pos.shape
            assert z_axis.shape == pose_data.pos.shape
            assert torch.allclose(
                torch.linalg.norm(x_axis, dim=-1),
                torch.ones_like(torch.linalg.norm(x_axis, dim=-1)),
                atol=1e-5,
            )
            assert torch.allclose(
                torch.linalg.norm(y_axis, dim=-1),
                torch.ones_like(torch.linalg.norm(y_axis, dim=-1)),
                atol=1e-5,
            )
            assert torch.allclose(
                torch.linalg.norm(z_axis, dim=-1),
                torch.ones_like(torch.linalg.norm(z_axis, dim=-1)),
                atol=1e-5,
            )

    def test_get_element_pose_with_repeated_parts(
        self, scene_with_interactive_cube
    ):
        for scene in self._create_scene(scene_with_interactive_cube):
            obj = scene.rigid_objects["objects/cube1"]
            pose_data = obj.get_element_pose(
                mode="passive",
                action="pick",
                part=["up", "up"],
                id=[[0, 1], [2]],
            )

            num_envs = obj.data.root_pos_w.shape[0]
            assert pose_data.pos.shape == (num_envs, 3, 3)
            assert pose_data.quat.shape == (num_envs, 3, 4)

    def test_invalid_id_length(self, scene_with_interactive_cube):
        for scene in self._create_scene(scene_with_interactive_cube):
            obj = scene.rigid_objects["objects/cube1"]
            with pytest.raises(ValueError, match="Part count"):
                obj.get_element_pose(
                    mode="passive",
                    action="pick",
                    part=["up", "up"],
                    id=[[0]],
                )

    def test_missing_part(self, scene_with_interactive_cube):
        for scene in self._create_scene(scene_with_interactive_cube):
            obj = scene.rigid_objects["objects/cube1"]
            with pytest.raises(ValueError, match="Part missing not found"):
                obj.get_element_pose(
                    mode="passive", action="pick", part="missing"
                )

    def test_empty_selection(self, scene_with_interactive_cube):
        for scene in self._create_scene(scene_with_interactive_cube):
            obj = scene.rigid_objects["objects/cube1"]
            with pytest.raises(ValueError, match="No interactive poses"):
                obj.get_element_pose(
                    mode="passive", action="pick", part="up", id=[[99]]
                )
