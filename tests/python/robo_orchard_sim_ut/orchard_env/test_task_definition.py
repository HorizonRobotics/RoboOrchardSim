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

from robo_orchard_sim.benchmark import base as task_base
from robo_orchard_sim.benchmark.base import TaskDefinition
from robo_orchard_sim.benchmark.manipulation.affordance import (
    AffordanceTaskDefinition,
)
from robo_orchard_sim.benchmark.manipulation.close import CloseTaskDefinition
from robo_orchard_sim.benchmark.manipulation.joint_direction import (
    JointDirectionTaskDefinition,
)
from robo_orchard_sim.benchmark.manipulation.open import OpenTaskDefinition
from robo_orchard_sim.ext.cfg_wrappers.assets_cfg import ArticulationCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners import UsdFileCfg
from robo_orchard_sim.ext.models.assets.asset_cfg import GroupAssetCfg
from robo_orchard_sim.orchard_env.assets import ArticulationSpec
from robo_orchard_sim.orchard_env.embodiments.dualarm_piper import (
    DualArmPiperEmbodiment,
)
from robo_orchard_sim.orchard_env.embodiments.dualarm_piperx import (
    DualArmPiperXEmbodiment,
)
from robo_orchard_sim.orchard_env.embodiments.embodiment_base import (
    EmbodimentBase,
)
from robo_orchard_sim.orchard_env.embodiments.franka_panda import (
    FrankaPandaEmbodiment,
)
from robo_orchard_sim.orchard_env.embodiments.panda_droid import (
    PandaDroidEmbodiment,
)
from robo_orchard_sim.orchard_env.scene.plane_table_scene import (
    PlaneTableScene,
)
from robo_orchard_sim.orchard_env.scene.scene_base import SceneBase
from robo_orchard_sim.task_components.instructions import (
    registry as instruction_registry,
)
from robo_orchard_sim.task_components.instructions.base import (
    InstructionWrapper,
)


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


def test_resolve_scene_missing_yaml_scene_returns_default_scene() -> None:
    scene = DummyTaskDefinition.resolve_scene()

    assert isinstance(scene, PlaneTableScene)


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


def test_resolve_embodiment_franka_panda_type_returns_franka_embodiment(
    tmp_path: Path,
) -> None:
    class YamlEmbodimentTaskDefinition(DummyTaskDefinition):
        config_path = _write_task_config(
            tmp_path,
            {"embodiment": {"type": "franka_panda"}},
        )

    embodiment = YamlEmbodimentTaskDefinition.resolve_embodiment()

    assert isinstance(embodiment, FrankaPandaEmbodiment)


def test_franka_panda_policy_binding_schema_camera_terms_match_observations():
    embodiment = FrankaPandaEmbodiment(enable_cameras=True)

    observation_cfg = embodiment.get_observation_cfg()
    camera_terms = set(observation_cfg.groups["/camera"].terms)
    schema_terms = {
        binding.obs_term
        for binding in (
            embodiment.get_policy_binding_schema().camera_slots.values()
        )
    }

    assert schema_terms <= camera_terms


def test_resolve_embodiment_panda_droid_type_returns_droid_embodiment(
    tmp_path: Path,
) -> None:
    class YamlEmbodimentTaskDefinition(DummyTaskDefinition):
        config_path = _write_task_config(
            tmp_path,
            {"embodiment": {"type": "panda_droid"}},
        )

    embodiment = YamlEmbodimentTaskDefinition.resolve_embodiment()

    assert isinstance(embodiment, PandaDroidEmbodiment)


def test_resolve_embodiment_panda_droid_camera_override_returns_requested_pose(
    tmp_path: Path,
) -> None:
    expected_xyz = (-0.08, -0.49, 0.48)
    expected_quat = (0.59, -0.62, 0.35, -0.38)

    class YamlEmbodimentTaskDefinition(DummyTaskDefinition):
        config_path = _write_task_config(
            tmp_path,
            {
                "embodiment": {
                    "type": "panda_droid",
                    "params": {
                        "camera_offsets": {
                            "ext2_camera": {
                                "xyz": expected_xyz,
                                "quat": expected_quat,
                                "convention": "ros",
                            }
                        }
                    },
                }
            },
        )

    embodiment = YamlEmbodimentTaskDefinition.resolve_embodiment()
    ext2_offset = embodiment.get_assets_cfg()["cameras"]["ext2_camera"].offset

    assert tuple(ext2_offset.xyz) == expected_xyz
    assert tuple(ext2_offset.quat) == expected_quat


def test_panda_droid_camera_override_after_custom_preserves_default_pose() -> (
    None
):
    overridden = PandaDroidEmbodiment(
        camera_offsets={
            "ext2_camera": {
                "xyz": (-0.08, -0.49, 0.48),
                "quat": (0.59, -0.62, 0.35, -0.38),
            }
        }
    )
    overridden.get_assets_cfg()

    default = PandaDroidEmbodiment()
    ext2_offset = default.get_assets_cfg()["cameras"]["ext2_camera"].offset

    assert tuple(ext2_offset.xyz) == (
        0.2596757315060087,
        -0.36626259649963777,
        0.24849304837972613,
    )


def test_panda_droid_camera_override_unknown_camera_raises_value_error() -> (
    None
):
    with pytest.raises(ValueError, match="unknown_camera"):
        PandaDroidEmbodiment(
            camera_offsets={
                "unknown_camera": {
                    "xyz": (0.0, 0.0, 0.0),
                    "quat": (1.0, 0.0, 0.0, 0.0),
                }
            }
        )


@pytest.mark.parametrize(
    "task_definition",
    [
        OpenTaskDefinition,
        CloseTaskDefinition,
        AffordanceTaskDefinition,
        JointDirectionTaskDefinition,
    ],
)
def test_articulation_task_camera_override_configured_task_returns_medium_pose(
    task_definition: type[TaskDefinition],
) -> None:
    embodiment = task_definition.resolve_embodiment()
    ext2_offset = embodiment.get_assets_cfg()["cameras"]["ext2_camera"].offset

    assert (
        *tuple(ext2_offset.xyz),
        *tuple(ext2_offset.quat),
        ext2_offset.convention,
    ) == (
        -0.084357793,
        -0.4948429153,
        0.4818388331,
        0.5941752707,
        -0.6188753508,
        0.3509019516,
        -0.3752557371,
        "ros",
    )


def test_resolve_embodiment_dualarm_piperx_type_returns_piperx_embodiment(
    tmp_path: Path,
) -> None:
    class YamlEmbodimentTaskDefinition(DummyTaskDefinition):
        config_path = _write_task_config(
            tmp_path,
            {"embodiment": {"type": "dualarm_piperx"}},
        )

    embodiment = YamlEmbodimentTaskDefinition.resolve_embodiment()

    assert isinstance(embodiment, DualArmPiperXEmbodiment)


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


def test_resolve_instruction_yaml_attribute_name_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        instruction_registry.INSTRUCTION_TEMPLATE_REGISTRY,
        "attribute_template",
        {
            "fixed": "Pick {actor1.attribute_value} {actor1.category}",
            "variants": [],
        },
    )

    class YamlAttributeInstructionTaskDefinition(DummyTaskDefinition):
        config_path = _write_task_config(
            tmp_path,
            {
                "instruction": {
                    "template": "attribute_template",
                    "template_mode": "fixed",
                    "attribute_name": "color",
                }
            },
        )

    instruction = YamlAttributeInstructionTaskDefinition.resolve_instruction()

    assert isinstance(instruction, InstructionWrapper)
    assert instruction.attribute_name == "color"


def test_place_a2b_task_definition_registers_namespace_and_config() -> None:
    from robo_orchard_sim.benchmark.manipulation.place_a2b import (
        place_a2b_env,
    )

    task_class = place_a2b_env.PlaceA2BTaskDefinition
    assert task_class.namespace == "place_a2b"
    assert task_class.config_path.endswith("place_a2b.yaml")
    assert not hasattr(place_a2b_env, "PlaceA2BEasyTaskDefinition")
    assert not hasattr(place_a2b_env, "PlaceA2BHardTaskDefinition")


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
