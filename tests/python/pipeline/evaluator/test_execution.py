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

import json
import sys
import types
from pathlib import Path

import pytest

from robo_orchard_sim import benchmark
from robo_orchard_sim.pipeline.data_synthesis.batch_synthesis import (
    BatchGroup,
    BatchPlan,
)
from robo_orchard_sim.pipeline.evaluator.distributed import (
    EvaluationManifest,
    build_evaluation_shards,
    validate_shard_result,
)
from robo_orchard_sim.pipeline.evaluator.execution import run_shard


@pytest.mark.parametrize("failure", [RuntimeError, SystemExit])
def test_run_shard_scene_failure_persists_outcome_before_runtime_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text("{}\n")
    plan = BatchPlan(
        batch_id="test",
        task="place_a2b_hard",
        episodes_per_config=2,
        base_seed=100,
        groups=[BatchGroup("group", 100, [str(task_yaml)])],
    )
    shard = build_evaluation_shards(
        plan=plan,
        task_name="place",
        shards_per_config=1,
        episodes_per_scene=2,
    )[0]
    manifest = EvaluationManifest("run", "dummy", (shard,))
    shard_dir = tmp_path / "place" / "shards" / shard.shard_id
    saved_at_shutdown = []

    class Launcher:
        def __init__(self, **kwargs):
            self.app = object()

        def close(self):
            saved_at_shutdown.append(
                json.loads((shard_dir / "shard_summary.json").read_text())
            )

    def build_task(*args, **kwargs):
        raise failure("scene failed")

    launcher_module = types.ModuleType("robo_orchard_sim.launcher")
    monkeypatch.setattr(
        launcher_module, "SimpleIsaacAppLauncher", Launcher, raising=False
    )
    monkeypatch.setitem(sys.modules, launcher_module.__name__, launcher_module)
    monkeypatch.setattr(benchmark, "build_task", build_task)
    result = run_shard(
        manifest=manifest,
        shard=shard,
        task_settings={
            "asset_root": str(tmp_path),
            "max_steps": 1,
            "snapshot": None,
            "splits": None,
            "swap": {"enabled": True, "swap_per_scene": 2},
        },
        policy_config={"policy": "dummy"},
        output_dir=tmp_path,
        enable_recording=False,
    )

    assert (shard_dir / "config" / task_yaml.name).read_text() == "{}\n"
    summary = saved_at_shutdown[0]
    assert (result, summary["configs"][0]["error"]) == (
        (1, "SystemExit: scene failed") if failure is SystemExit else (0, None)
    )
    if failure is RuntimeError:
        payload = json.loads((shard_dir / "eval_result.json").read_text())
        validate_shard_result(shard, payload)
