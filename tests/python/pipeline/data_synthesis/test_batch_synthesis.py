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

"""Tests for batch_synthesis task-config construction (splits/snapshot)."""

from __future__ import annotations
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from robo_orchard_sim.pipeline.data_synthesis import batch_synthesis
from robo_orchard_sim.pipeline.data_synthesis.batch_synthesis import (
    BatchGroup,
    BatchPlan,
    build_task_cfgs_for_group,
    run_group_data_synthesis,
    write_group_outputs,
)
from robo_orchard_sim.pipeline.data_synthesis.multi_task import (
    MultiTaskRunResult,
)
from robo_orchard_sim.pipeline.data_synthesis.single_task import (
    TaskRunResult,
)


def _plan(config_path: str) -> BatchPlan:
    return BatchPlan(
        batch_id="b0",
        task="place_a2b",
        episodes_per_config=2,
        base_seed=0,
        groups=[BatchGroup(group_id="g0", seed=5, configs=[config_path])],
    )


class TestBuildTaskCfgsSplitsSnapshot:
    def test_propagates_splits_and_snapshot(self, tmp_path):
        config_file = tmp_path / "task.yaml"
        config_file.write_text("task: place_a2b\n")

        cfgs = build_task_cfgs_for_group(
            plan=_plan(str(config_file)),
            group_id="g0",
            asset_root="/tmp/assets",
            task_root_dir="/tmp/out",
            splits_path=Path("/x/splits.yaml"),
            snapshot_path=Path("/x/snap.yaml"),
        )

        assert len(cfgs) == 1
        assert cfgs[0].splits_path == Path("/x/splits.yaml")
        assert cfgs[0].snapshot_path == Path("/x/snap.yaml")

    def test_defaults_none_when_omitted(self, tmp_path):
        config_file = tmp_path / "task.yaml"
        config_file.write_text("task: place_a2b\n")

        cfgs = build_task_cfgs_for_group(
            plan=_plan(str(config_file)),
            group_id="g0",
            asset_root="/tmp/assets",
            task_root_dir="/tmp/out",
        )

        assert cfgs[0].splits_path is None
        assert cfgs[0].snapshot_path is None


@pytest.mark.parametrize("swap_enabled", [False, True])
def test_batch_synthesis_swap_option_preserves_seed_and_episode_budget(
    tmp_path, swap_enabled
):
    config_file = tmp_path / "task.yaml"
    config_file.write_text("task: place_a2b\n")
    cfgs = build_task_cfgs_for_group(
        plan=_plan(str(config_file)),
        group_id="g0",
        asset_root=str(tmp_path),
        task_root_dir=str(tmp_path / "out"),
        swap_enabled=swap_enabled,
    )

    assert [(cfg.swap_enabled, cfg.seed, cfg.episode_num) for cfg in cfgs] == [
        (swap_enabled, 5, 2)
    ]


def test_group_outputs_are_written_before_runtime_closes(
    tmp_path, monkeypatch
):
    config_file = tmp_path / "task.yaml"
    config_file.write_text("task: place_a2b\n")
    events = []
    result = MultiTaskRunResult(
        task_results=[],
        total_tasks=0,
        success_tasks=0,
        total_episodes=0,
        success_episodes=0,
        success_rate=0.0,
    )

    class FakeRunner:
        def __init__(self, cfg):
            self.cfg = cfg

        def run_with_runtime(self, runtime):
            events.append(("run", runtime))
            return result

    @contextmanager
    def fake_runtime(launch):
        events.append(("runtime_enter", launch))
        yield "runtime"
        events.append(("runtime_exit", launch))

    def fake_write_group_outputs(**kwargs):
        events.append(("write", kwargs["result"]))

    monkeypatch.setattr(
        batch_synthesis, "MultiTaskDataSynthesisRunner", FakeRunner
    )
    monkeypatch.setattr(
        batch_synthesis, "data_synthesis_runtime", fake_runtime
    )
    monkeypatch.setattr(
        batch_synthesis,
        "write_group_outputs",
        fake_write_group_outputs,
    )

    actual = run_group_data_synthesis(
        plan=_plan(str(config_file)),
        group_id="g0",
        asset_root=str(tmp_path),
        task_root_dir=str(tmp_path / "out"),
    )

    assert actual is result
    assert [event[0] for event in events] == [
        "runtime_enter",
        "run",
        "write",
        "runtime_exit",
    ]


def test_zero_success_group_still_writes_complete_manifests(tmp_path):
    task_result = TaskRunResult(
        task="spatial_near_far",
        config_path="task.yaml",
        task_save_root="g0/config_0000",
        config_dir=str(tmp_path / "g0/config_0000/config"),
        data_dir=str(tmp_path / "g0/config_0000/data"),
        record_dir=str(tmp_path / "g0/config_0000/data"),
        episodes=[],
        total=2,
        success_count=0,
        success_rate=0.0,
    )
    result = MultiTaskRunResult(
        task_results=[task_result],
        total_tasks=1,
        success_tasks=1,
        total_episodes=2,
        success_episodes=0,
        success_rate=0.0,
        mcap_paths=[],
    )
    plan = BatchPlan(
        batch_id="b0",
        task="spatial_near_far",
        episodes_per_config=2,
        base_seed=0,
        groups=[BatchGroup(group_id="g0", seed=0, configs=["task.yaml"])],
    )

    write_group_outputs(
        output_dir=str(tmp_path),
        plan=plan,
        group_id="g0",
        result=result,
    )

    summary_path = tmp_path / "batch_run_summary_spatial_near_far.json"
    records_path = tmp_path / "all_episode_records_spatial_near_far.jsonl"
    mcaps_path = tmp_path / "all_mcap_paths_spatial_near_far.txt"
    summary = json.loads(summary_path.read_text())
    assert summary["total_episodes"] == 2
    assert summary["success_count"] == 0
    assert summary["success_rate"] == 0.0
    assert records_path.read_text() == ""
    assert mcaps_path.read_text() == ""


def test_empty_group_writes_outputs_without_launching_runtime(
    tmp_path, monkeypatch
):
    events = []
    result = MultiTaskRunResult(
        task_results=[],
        total_tasks=0,
        success_tasks=0,
        total_episodes=0,
        success_episodes=0,
        success_rate=0.0,
    )

    class FakeRunner:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            events.append("run")
            return result

    def fail_runtime(_launch):
        raise AssertionError("empty group must not launch Isaac")

    def fake_write_group_outputs(**_kwargs):
        events.append("write")

    monkeypatch.setattr(
        batch_synthesis, "MultiTaskDataSynthesisRunner", FakeRunner
    )
    monkeypatch.setattr(
        batch_synthesis, "data_synthesis_runtime", fail_runtime
    )
    monkeypatch.setattr(
        batch_synthesis,
        "write_group_outputs",
        fake_write_group_outputs,
    )
    plan = BatchPlan(
        batch_id="b0",
        task="spatial_near_far",
        episodes_per_config=2,
        base_seed=0,
        groups=[BatchGroup(group_id="g0", seed=0, configs=[])],
    )

    actual = run_group_data_synthesis(
        plan=plan,
        group_id="g0",
        asset_root=str(tmp_path),
        task_root_dir=str(tmp_path / "out"),
    )

    assert actual is result
    assert events == ["run", "write"]
