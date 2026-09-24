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

"""Configured selectors use scene seeds and planned scene-local indices."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from test_evaluator import (
    _StepState as StepState,
    _StubOrchardEnv as StubOrchardEnv,
    _StubPolicy as StubPolicy,
    _StubStepEnv as StubStepEnv,
    _StubTask as StubTask,
)

from robo_orchard_sim import benchmark
from robo_orchard_sim.asset_manager import registry, resolver
from robo_orchard_sim.orchard_env.task_spec import RoleSpec
from robo_orchard_sim.pipeline.evaluator import EvaluatorCfg
from robo_orchard_sim.pipeline.evaluator.evaluator import SwapConfig
from robo_orchard_sim.task_components.role_registry import TargetRef
from robo_orchard_sim.task_components.selector import create_selector


@pytest.mark.parametrize("selector", ["random", "round_robin"])
def test_evaluator_selector_multiple_scenes_use_seed_and_local_index(
    monkeypatch, tmp_path, selector
):
    pool = [TargetRef(f"candidate{i}") for i in (4, 1, 8, 0, 9, 2, 7, 3, 6, 5)]

    class CandidateTask(StubTask):
        roles = {
            **StubTask.roles,
            "all": RoleSpec(description="all candidates", cardinality="many"),
        }

        def get_role_candidates(self, role_id, *, swap=False):
            return pool

    module = ModuleType("robo_orchard_sim.ext.envs.env_base")

    class Context:
        def __init__(self, cfg, **kwargs):
            self.env = cfg()

        def __enter__(self):
            return self.env

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        module, "IsaacEnvContextManager", Context, raising=False
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(registry, "AssetRegistry", lambda root: object())
    monkeypatch.setattr(
        resolver, "AssetResolver", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    env = StubStepEnv(episodes=[[StepState()] for _ in range(4)])
    orchard = StubOrchardEnv(env=env, success_steps=[1] * 4)
    orchard.task = CandidateTask(success_steps=[1] * 4)
    monkeypatch.setattr(
        benchmark, "build_task", lambda *args, **kwargs: orchard
    )
    evaluator = EvaluatorCfg(
        task_name="pick_category",
        asset_root=str(tmp_path),
        seed=42,
        episode_num=4,
        max_steps=1,
        max_settle_steps=1,
        settle_streak=1,
        swap=SwapConfig(enabled=True, selector=selector, swap_per_scene=2),
    )()
    evaluator.run_with_runtime(StubPolicy(), sim_app=object())
    assert [reset["role_bindings"] for reset in env.reset_calls] == [
        {
            **{
                role: create_selector(
                    selector, seed=seed, role_id=role
                ).select(pool, 1, index)[0]
                for role in ("pick", "place")
            },
            "all": pool,
        }
        for seed, index in [(42, 0), (42, 1), (43, 0), (43, 1)]
    ]
