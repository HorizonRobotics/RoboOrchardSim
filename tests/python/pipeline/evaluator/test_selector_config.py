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

"""Selector configuration through public YAML and batch interfaces."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from robo_orchard_sim.pipeline.data_synthesis.batch_synthesis import (
    BatchGroup,
    BatchPlan,
)
from robo_orchard_sim.pipeline.evaluator.batch_evaluation import (
    build_evaluator_cfgs_for_group,
)
from robo_orchard_sim.pipeline.evaluator.evaluator import SwapConfig


@pytest.fixture
def selector_cli():
    root = Path(__file__).resolve().parents[4]
    path = root / "examples/manipulation-app/scripts/eval_policy.py"
    spec = importlib.util.spec_from_file_location("selector_eval_config", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_selector_config_task_overrides_reach_independent_batch_entries(
    tmp_path, selector_cli
):
    task_config = tmp_path / "task.yaml"
    task_config.write_text("{}")
    config = tmp_path / "eval.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "policy": {"model_type": "dummy"},
                "defaults": {
                    "asset_root": "/assets",
                    "swap": {
                        "enabled": True,
                        "selector": "random",
                        "swap_per_scene": 5,
                    },
                },
                "tasks": {
                    name: {"task_type": "pick_category", **fields}
                    for name, fields in {
                        "inherited": {},
                        "empty": {"swap": {}},
                        "override": {"swap": {"selector": "round_robin"}},
                        "disabled": {"swap": {"enabled": False}},
                        "cleared": {"swap": None},
                    }.items()
                },
            },
            sort_keys=False,
        )
    )
    tasks = selector_cli.load_eval_config(config).tasks
    swaps = []
    for task in tasks:
        entries = build_evaluator_cfgs_for_group(
            plan=BatchPlan(
                batch_id="b",
                task=task.task_type,
                episodes_per_config=5,
                base_seed=42,
                groups=[
                    BatchGroup(
                        group_id="g", seed=42, configs=[str(task_config)]
                    )
                ],
            ),
            group_id="g",
            asset_root=task.asset_root,
            output_root_dir=str(tmp_path / "out" / task.name),
            max_steps=1,
            swap=task.swap.to_evaluator_config(SwapConfig),
        )
        swaps.append(entries[0].evaluator_cfg.swap)
    assert [
        (swap.enabled, swap.selector, swap.swap_per_scene) for swap in swaps
    ] == [
        (True, "random", 5),
        (True, "random", 5),
        (True, "round_robin", 5),
        (False, "random", 5),
        (False, "random", 1),
    ]
    swaps[0].selector = "round_robin"
    assert swaps[1].selector == tasks[0].swap.selector == "random"


@pytest.mark.parametrize("selector", ["invalid", None, 1, [], {}])
def test_selector_config_unknown_strategy_raises_error(
    tmp_path, selector_cli, selector
):
    path = tmp_path / "eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "policy": {"model_type": "dummy"},
                "defaults": {"asset_root": "/assets"},
                "tasks": {
                    "pick": {
                        "task_type": "pick_category",
                        "swap": {"selector": selector},
                    }
                },
            }
        )
    )
    with pytest.raises(SystemExit, match="swap.selector"):
        selector_cli.load_eval_config(path)
    with pytest.raises(ValueError, match="selector"):
        SwapConfig(selector=selector)


@pytest.mark.parametrize("swap_fields", [{}, {"swap_per_scene": 1}])
def test_selector_config_omitted_strategy_uses_random_with_one_episode(
    tmp_path, selector_cli, swap_fields
):
    path = tmp_path / "eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "policy": {"model_type": "dummy"},
                "defaults": {"asset_root": "/assets"},
                "tasks": {
                    "pick": {
                        "task_type": "pick_category",
                        "swap": {"enabled": True, **swap_fields},
                    }
                },
            }
        )
    )
    swap = selector_cli.load_eval_config(path).tasks[0].swap
    assert swap.to_evaluator_config(SwapConfig) == SwapConfig(
        enabled=True, selector="random", swap_per_scene=1
    )
    defaults = SwapConfig(enabled=True)
    assert (defaults.selector, defaults.swap_per_scene) == ("random", 1)
