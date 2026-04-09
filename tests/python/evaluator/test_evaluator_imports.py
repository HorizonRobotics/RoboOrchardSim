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
import builtins
import importlib
import sys
from typing import Any

import pytest


def test_importing_evaluator_does_not_import_isaac_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reloaded_modules = {
        "robo_orchard_sim.evaluator",
        "robo_orchard_sim.evaluator.base",
        "robo_orchard_sim.evaluator.evaluator",
    }
    blocked_modules = {
        "robo_orchard_sim.envs.env_base",
        "robo_orchard_sim.envs.manager_based_env",
        "robo_orchard_sim.launcher",
    }
    removed_modules = {}
    target_modules = reloaded_modules | blocked_modules
    for module_name in list(sys.modules):
        if module_name not in target_modules:
            continue
        removed_modules[module_name] = sys.modules.pop(module_name)

    attempted_imports: list[str] = []
    real_import = builtins.__import__

    def _guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in blocked_modules:
            attempted_imports.append(name)
            raise AssertionError(f"Unexpected runtime import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    try:
        module = importlib.import_module("robo_orchard_sim.evaluator")

        assert module.EvaluatorCfg is not None
        assert attempted_imports == []
    finally:
        for module_name in target_modules:
            sys.modules.pop(module_name, None)
        sys.modules.update(removed_modules)
