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

from __future__ import annotations
from pathlib import Path

import pytest
import yaml

from robo_orchard_sim.cfg_wrappers.assets_cfg import ArticulationCfg
from robo_orchard_sim.cfg_wrappers.sim.spawners import UsdFileCfg
from robo_orchard_sim.models.assets.asset_cfg import GroupAssetCfg
from robo_orchard_sim.orchard_env.assets import ArticulationSpec
from robo_orchard_sim.orchard_env.embodiments.dualarm_piper import (
    DualArmPiperEmbodiment,
)
from robo_orchard_sim.orchard_env.embodiments.embodiment_base import (
    EmbodimentBase,
)
from robo_orchard_sim.orchard_env.scene.scene_base import SceneBase
from robo_orchard_sim.task_suite import base as task_base
from robo_orchard_sim.task_suite.base import TaskDefinition
from robo_orchard_sim.tasks.instructions import (
    registry as instruction_registry,
)
from robo_orchard_sim.tasks.instructions.base import InstructionWrapper


class DummyScene(SceneBase):
    def __init__(
        self,
        num_envs: int = 1,
        env_spacing: float = 2.5,
        physics_fps: int = 600,
        render_fps: int = 30,
        step_fps: int = 30,
        kitchen_layout: str = "default",
    ) -> None:
        super().__init__(
            num_envs=num_envs,
            env_spacing=env_spacing,
            physics_fps=physics_fps,
            render_fps=render_fps,
            step_fps=step_fps,
        )
        self.kitchen_layout = kitchen_layout

    def get_assets_cfg(self) -> dict[str, GroupAssetCfg]:
        return {}


class DummyEmbodiment(EmbodimentBase):
    def __init__(self, robot_name: str = "dummy_robot") -> None:
        super().__init__(
            robot=ArticulationSpec(
                name=robot_name,
                namespace="robots",
                template_cfg=ArticulationCfg(
                    prim_path="{ENV_REGEX_NS}/dummy_robot",
                    spawn=UsdFileCfg(usd_path="/tmp/dummy_robot.usd"),
                    init_state=ArticulationCfg.InitialStateCfg(joint_pos={}),
                    actuators={},
                ),
            )
        )


class DummyTaskDefinition(TaskDefinition):
    namespace = "dummy_task"

    @classmethod
    def build(cls):
        raise NotImplementedError


def _write_task_config(tmp_path: Path, config: dict) -> str:
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


def test_resolve_scene_prefers_yaml_over_class_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        task_base.SCENE_REGISTRY,
        "kitchen_scene",
        DummyScene,
    )

    class YamlSceneTaskDefinition(DummyTaskDefinition):
        scene = "plane_table"
        config_path = _write_task_config(
            tmp_path,
            {
                "scene": {
                    "type": "kitchen_scene",
                    "num_envs": 3,
                    "env_spacing": 4.0,
                    "params": {"kitchen_layout": "galley"},
                }
            },
        )

    scene = YamlSceneTaskDefinition.resolve_scene()

    assert isinstance(scene, DummyScene)
    assert scene.get_num_envs() == 3
    assert scene.get_env_spacing() == 4.0
    assert scene.kitchen_layout == "galley"


def test_resolve_scene_rejects_unknown_registered_name() -> None:
    class UnknownSceneTaskDefinition(DummyTaskDefinition):
        scene = "unknown_scene"

    with pytest.raises(ValueError, match="Unknown scene"):
        UnknownSceneTaskDefinition.resolve_scene()


def test_resolve_embodiment_prefers_yaml_over_class_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        task_base.EMBODIMENT_REGISTRY,
        "dummy_embodiment",
        DummyEmbodiment,
    )

    class YamlEmbodimentTaskDefinition(DummyTaskDefinition):
        embodiment = "dualarm_piper"
        config_path = _write_task_config(
            tmp_path,
            {
                "embodiment": {
                    "type": "dummy_embodiment",
                    "params": {"robot_name": "yaml_robot"},
                }
            },
        )

    embodiment = YamlEmbodimentTaskDefinition.resolve_embodiment()

    assert isinstance(embodiment, DummyEmbodiment)
    assert embodiment.name == "yaml_robot"


def test_resolve_embodiment_rejects_unknown_registered_name() -> None:
    class UnknownEmbodimentTaskDefinition(DummyTaskDefinition):
        embodiment = "unknown_embodiment"

    with pytest.raises(ValueError, match="Unknown embodiment"):
        UnknownEmbodimentTaskDefinition.resolve_embodiment()


def test_resolve_embodiment_passes_init_joint_pos_from_yaml(
    tmp_path: Path,
) -> None:
    class YamlEmbodimentTaskDefinition(DummyTaskDefinition):
        config_path = _write_task_config(
            tmp_path,
            {
                "embodiment": {
                    "type": "dualarm_piper",
                    "init_joint_pos": {"left_joint1": 0.1},
                }
            },
        )

    embodiment = YamlEmbodimentTaskDefinition.resolve_embodiment()

    assert isinstance(embodiment, DualArmPiperEmbodiment)
    assert embodiment.init_joint_pos == {"left_joint1": 0.1}


def test_resolve_instruction_prefers_yaml_template_mode_over_class_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        instruction_registry.INSTRUCTION_TEMPLATE_REGISTRY,
        "yaml_template",
        {
            "fixed": "place {pick} into {place}",
            "variants": [],
        },
    )

    class YamlInstructionsTaskDefinition(DummyTaskDefinition):
        instruction = "default_template"
        config_path = _write_task_config(
            tmp_path,
            {
                "instruction": {
                    "template": "yaml_template",
                    "template_mode": "variants",
                }
            },
        )

    instruction = YamlInstructionsTaskDefinition.resolve_instruction()

    assert isinstance(instruction, InstructionWrapper)
    assert instruction.template == "yaml_template"
    assert instruction.template_mode == "variants"


def test_place_a2b_task_definitions_register_easy_and_hard_namespaces() -> (
    None
):
    from robo_orchard_sim.task_suite.manipulation.place_a2b import (
        place_a2b_env,
    )

    assert (
        place_a2b_env.PlaceA2BEasyTaskDefinition.namespace == "place_a2b_easy"
    )
    assert (
        place_a2b_env.PlaceA2BHardTaskDefinition.namespace == "place_a2b_hard"
    )
    assert place_a2b_env.PlaceA2BEasyTaskDefinition.config_path.endswith(
        "place_a2b_easy.yaml"
    )
    assert place_a2b_env.PlaceA2BHardTaskDefinition.config_path.endswith(
        "place_a2b_hard.yaml"
    )


def test_resolve_task_params_reads_yaml_task_section(
    tmp_path: Path,
) -> None:
    class YamlTaskParamsDefinition(DummyTaskDefinition):
        config_path = _write_task_config(
            tmp_path,
            {
                "task": {
                    "params": {
                        "distractor": {
                            "name": "distractor_object",
                            "uuid": "toy_001",
                        }
                    }
                }
            },
        )

    params = YamlTaskParamsDefinition.resolve_task_params()

    assert params == {
        "distractor": {
            "name": "distractor_object",
            "uuid": "toy_001",
        }
    }
