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

"""Regression tests for shipped pick task YAML and instruction template."""

from __future__ import annotations
import sys
import types
from pathlib import Path

import pytest
import yaml

from robo_orchard_sim.benchmark.manipulation.semantic_pick import pick_env
from robo_orchard_sim.orchard_env.task_templates.task_params import (
    TaskLightResetConfig,
)


class FakeResolver:
    def resolve(self, asset_configs):
        del asset_configs
        return {"pick": object()}


@pytest.mark.parametrize(
    (
        "task_definition",
        "yaml_name",
        "expected_template",
        "expected_attribute",
    ),
    [
        (
            pick_env.PickCategoryTaskDefinition,
            "pick_category.yaml",
            "pick_default",
            None,
        ),
        (
            pick_env.PickAttributeTaskDefinition,
            "pick_attribute.yaml",
            "pick_attribute",
            "color",
        ),
        (
            pick_env.PickDisambiguationTaskDefinition,
            "pick_disambiguation.yaml",
            "pick_default",
            None,
        ),
    ],
)
def test_pick_task_definition_yaml_uses_expected_instruction_template(
    task_definition,
    yaml_name: str,
    expected_template: str,
    expected_attribute: str | None,
) -> None:
    yaml_path = (
        Path(__file__).resolve().parents[4]
        / "robo_orchard_sim"
        / "benchmark"
        / "manipulation"
        / "semantic_pick"
        / "configs"
        / yaml_name
    )
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    instruction = task_definition.resolve_instruction()

    assert raw["instruction"]["template"] == expected_template
    assert instruction is not None
    assert instruction.template == expected_template
    assert instruction.attribute_name == expected_attribute
    assert instruction.template_mode == raw["instruction"]["template_mode"]
    assert (
        instruction.actor_description_mode
        == raw["instruction"]["actor_description_mode"]
    )


def test_pick_task_definition_build_assigns_instruction_to_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_orchard_env_module = types.ModuleType(
        "robo_orchard_sim.orchard_env.orchard_env"
    )

    class FakeOrchardEnv:
        def __init__(self, scene, embodiment, task):
            self.scene = scene
            self.embodiment = embodiment
            self.task = task

    fake_orchard_env_module.OrchardEnv = FakeOrchardEnv
    monkeypatch.setitem(
        sys.modules,
        "robo_orchard_sim.orchard_env.orchard_env",
        fake_orchard_env_module,
    )

    fake_pick_task_module = types.ModuleType(
        "robo_orchard_sim.orchard_env.task_templates.pick_task"
    )

    class FakePickAssets:
        def __init__(self, **resolved):
            self.resolved = resolved

        @classmethod
        def from_resolved(cls, resolved):
            return cls(**resolved)

    class FakePickTaskParams:
        def __init__(self, **params):
            self.params = params

    class FakePickTask:
        def __init__(self, assets, params, instruction=None):
            self.assets = assets
            self.params = params
            self.instruction = instruction

    fake_pick_task_module.PickAssets = FakePickAssets
    fake_pick_task_module.PickTask = FakePickTask
    fake_pick_task_module.PickTaskParams = FakePickTaskParams
    monkeypatch.setitem(
        sys.modules,
        "robo_orchard_sim.orchard_env.task_templates.pick_task",
        fake_pick_task_module,
    )

    scene = object()
    embodiment = object()
    monkeypatch.setattr(
        pick_env.PickCategoryTaskDefinition,
        "resolve_scene",
        classmethod(lambda cls, config_path=None: scene),
    )
    monkeypatch.setattr(
        pick_env.PickCategoryTaskDefinition,
        "resolve_embodiment",
        classmethod(lambda cls, config_path=None: embodiment),
    )

    orchard_env = pick_env.PickCategoryTaskDefinition.build(
        resolver=FakeResolver()
    )

    assert orchard_env.scene is scene
    assert orchard_env.embodiment is embodiment
    assert orchard_env.task.instruction is not None
    assert orchard_env.task.instruction.template == "pick_default"
    assert orchard_env.task.instruction.actor_description_mode == "seen"


def test_pick_task_definition_yaml_configures_distant_light() -> None:
    yaml_path = (
        Path(__file__).resolve().parents[4]
        / "robo_orchard_sim"
        / "benchmark"
        / "manipulation"
        / "semantic_pick"
        / "configs"
        / "pick_category.yaml"
    )
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    light_reset = raw["task"]["params"]["light_reset"]

    cfg = TaskLightResetConfig(**light_reset)

    assert light_reset["preset"] == "default_distant_light"
    assert cfg.asset_names == ["background/dis_light"]
    assert cfg.distant_light is not None
    assert cfg.distant_light.asset_name == "dis_light"
