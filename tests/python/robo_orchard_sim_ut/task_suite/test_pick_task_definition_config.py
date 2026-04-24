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
from pathlib import Path

import pytest
import yaml

from robo_orchard_sim.task_suite.manipulation.semantic_pick import pick_env
from robo_orchard_sim.tasks.instructions.base import Actor
from robo_orchard_sim.tasks.instructions.registry import (
    build_instruction_wrapper,
)


@pytest.mark.parametrize(
    ("task_definition", "yaml_name"),
    [
        (
            pick_env.PickCategoryTaskDefinition,
            "pick_category.yaml",
        ),
        (
            pick_env.PickAttributeTaskDefinition,
            "pick_attribute.yaml",
        ),
        (
            pick_env.PickDisambiguationTaskDefinition,
            "pick_disambiguation.yaml",
        ),
    ],
)
def test_pick_task_definition_yaml_uses_pick_instruction_template(
    task_definition,
    yaml_name: str,
) -> None:
    yaml_path = (
        Path(__file__).resolve().parents[4]
        / "robo_orchard_sim"
        / "task_suite"
        / "manipulation"
        / "semantic_pick"
        / "configs"
        / yaml_name
    )
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    assert raw["instruction"]["template"] == "pick_default"
    assert task_definition.resolve_instruction() is not None
    assert task_definition.resolve_instruction().template_mode == "raw"


def test_build_instruction_wrapper_pick_default_raw_renders_pick_only() -> (
    None
):
    instruction = build_instruction_wrapper("pick_default")

    assert (
        instruction.render(actors=[Actor(category="apple", uuid="apple-1")])
        == "Pick up apple"
    )
