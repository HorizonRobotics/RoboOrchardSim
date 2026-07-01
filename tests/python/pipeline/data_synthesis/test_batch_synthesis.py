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
from pathlib import Path

from robo_orchard_sim.pipeline.data_synthesis.batch_synthesis import (
    BatchGroup,
    BatchPlan,
    build_task_cfgs_for_group,
)


def _plan(config_path: str) -> BatchPlan:
    return BatchPlan(
        batch_id="b0",
        task="place_a2b_easy",
        episodes_per_config=2,
        base_seed=0,
        groups=[BatchGroup(group_id="g0", seed=5, configs=[config_path])],
    )


class TestBuildTaskCfgsSplitsSnapshot:
    def test_propagates_splits_and_snapshot(self, tmp_path):
        config_file = tmp_path / "task.yaml"
        config_file.write_text("task: place_a2b_easy\n")

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
        config_file.write_text("task: place_a2b_easy\n")

        cfgs = build_task_cfgs_for_group(
            plan=_plan(str(config_file)),
            group_id="g0",
            asset_root="/tmp/assets",
            task_root_dir="/tmp/out",
        )

        assert cfgs[0].splits_path is None
        assert cfgs[0].snapshot_path is None
