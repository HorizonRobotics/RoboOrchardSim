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

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from robo_orchard_core.envs.managers.events import EventManagerCfg

from robo_orchard_sim.task_components.validators.base import Validator


def _load_task_base_with_stubbed_dependencies(
    monkeypatch: pytest.MonkeyPatch,
):
    record_module = types.ModuleType(
        "robo_orchard_sim.ext.envs.managers.record"
    )
    record_module.RecordTermBaseCfg = type("RecordTermBaseCfg", (), {})
    monkeypatch.setitem(
        sys.modules,
        "robo_orchard_sim.ext.envs",
        types.ModuleType("robo_orchard_sim.ext.envs"),
    )
    managers_module_name = "robo_orchard_sim.ext.envs.managers"
    monkeypatch.setitem(
        sys.modules,
        managers_module_name,
        types.ModuleType(managers_module_name),
    )
    monkeypatch.setitem(
        sys.modules,
        "robo_orchard_sim.ext.envs.managers.record",
        record_module,
    )

    asset_cfg_module = types.ModuleType(
        "robo_orchard_sim.ext.models.assets.asset_cfg"
    )
    asset_cfg_module.GroupAssetCfg = type("GroupAssetCfg", (), {})
    monkeypatch.setitem(
        sys.modules,
        "robo_orchard_sim.ext.models",
        types.ModuleType("robo_orchard_sim.ext.models"),
    )
    monkeypatch.setitem(
        sys.modules,
        "robo_orchard_sim.ext.models.assets",
        types.ModuleType("robo_orchard_sim.ext.models.assets"),
    )
    monkeypatch.setitem(
        sys.modules,
        "robo_orchard_sim.ext.models.assets.asset_cfg",
        asset_cfg_module,
    )

    assets_module = types.ModuleType("robo_orchard_sim.orchard_env.assets")
    assets_module.AssetSpec = type("AssetSpec", (), {})
    assets_module.ObjectSpec = type("ObjectSpec", (), {})
    assets_module.PoolSpec = type("PoolSpec", (), {})
    monkeypatch.setitem(
        sys.modules,
        "robo_orchard_sim.orchard_env.assets",
        assets_module,
    )

    task_base_path = (
        Path(__file__).resolve().parents[2]
        / "robo_orchard_sim"
        / "orchard_env"
        / "task_templates"
        / "task_base.py"
    )
    task_base_spec = importlib.util.spec_from_file_location(
        "test_task_base_module",
        task_base_path,
    )
    assert task_base_spec is not None
    assert task_base_spec.loader is not None
    task_base_module = importlib.util.module_from_spec(task_base_spec)
    task_base_spec.loader.exec_module(task_base_module)
    return task_base_module.TaskBase


def test_task_base_subclass_without_instruction_can_be_instantiated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_base = _load_task_base_with_stubbed_dependencies(monkeypatch)

    class _TaskWithoutInstruction(task_base):
        def __init__(self) -> None:
            super().__init__(assets={})

        def get_event_cfg(self) -> EventManagerCfg:
            return EventManagerCfg(terms={})

        def get_validator_actor_names(self) -> list[str]:
            return []

        def build_validator(
            self,
            actors: list[object],
            context=None,
        ) -> Validator:
            del actors, context
            return Validator(
                actors=[],
                criteria=[],
                criteria_name=[],
            )

    task = _TaskWithoutInstruction()

    assert (
        task.build_instruction_context(
            env=None,
            actor_description_seed=7,
        )
        == {}
    )
