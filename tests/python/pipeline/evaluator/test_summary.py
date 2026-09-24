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

import pytest

from robo_orchard_sim.pipeline.data_synthesis.batch_synthesis import (
    BatchGroup,
    BatchPlan,
)
from robo_orchard_sim.pipeline.evaluator import (
    EpisodeResult,
    EvaluationResult,
    EvaluationSummary,
    ShardSummary,
    TaskSummary,
)
from robo_orchard_sim.pipeline.evaluator.distributed import (
    build_evaluation_shards,
)


def outcome(name: str, successes: list[bool], error: str | None = None):
    plan = BatchPlan(
        batch_id="test",
        task="pick_category",
        episodes_per_config=max(1, len(successes)),
        base_seed=0,
        groups=[BatchGroup("group", 0, ["task.yaml"])],
    )
    shard = build_evaluation_shards(
        plan=plan,
        task_name=name,
        shards_per_config=1,
    )[0]
    result = None
    if successes:
        result = EvaluationResult(
            episode_num=len(successes),
            seed_start=0,
            success_rate=0.0,
            average_progress=0.0,
            episode_results=[
                EpisodeResult(
                    seed=i,
                    success=success,
                    progress=float(success),
                    steps=1,
                    stop_reason="done",
                    metrics={"criteria_reached": {"pick": success}},
                )
                for i, success in enumerate(successes)
            ],
        )
    return ShardSummary(shard, result=result, error=error)


def test_evaluation_summary_unequal_tasks_weights_completed_episodes():
    summary = EvaluationSummary(
        [
            TaskSummary("short", [outcome("short", [True])]),
            TaskSummary("long", [outcome("long", [False, False, False])]),
        ]
    )
    payload = summary.to_payload()

    assert (
        summary.metrics.episodes,
        summary.metrics.success_rate,
        summary.metrics.avg_progress,
        summary.metrics.stage_success_rate,
    ) == (4, 0.25, 0.25, {"pick": 0.25})
    assert payload["tasks"]["short"]["shards"][0]["successes"] == 1


@pytest.mark.parametrize(
    ("successes", "error", "status"),
    [
        ([False], None, "ok"),
        ([True], "interrupted", "partial_error"),
        ([], "missing", "error"),
    ],
)
def test_summary_worker_outcome_preserves_status_and_completed_metrics(
    successes: list[bool],
    error: str | None,
    status: str,
):
    shard = outcome("pick", successes, error)
    task = TaskSummary("pick", [shard])
    overall = EvaluationSummary([task])

    assert (shard.status, task.status, overall.status) == (status,) * 3
    assert overall.metrics.episodes == len(successes)


def test_task_summary_failed_shard_retains_successful_shard_metrics():
    good = outcome("pick", [True])
    bad = outcome("pick", [], "missing")
    from dataclasses import replace

    bad.shard = replace(bad.shard, shard_id="other", shard_index=1)
    task = TaskSummary("pick", [good, bad])

    assert (task.status, task.metrics.episodes, task.metrics.success_rate) == (
        "partial_error",
        1,
        1.0,
    )
    assert task.to_payload()["configs_failed"] == 1


@pytest.mark.parametrize("duplicate", ["task", "shard"])
def test_summary_duplicate_identity_raises_value_error(duplicate: str):
    shard = outcome("pick", [True])
    task = TaskSummary("pick", [shard])
    with pytest.raises(ValueError, match="duplicate"):
        if duplicate == "task":
            EvaluationSummary([task, task])
        else:
            TaskSummary("pick", [shard, shard])


def test_evaluation_summary_empty_run_returns_zero_metrics_and_error():
    result = EvaluationSummary([])
    assert (
        result.status,
        result.metrics.episodes,
        result.metrics.success_rate,
    ) == (
        "error",
        0,
        0.0,
    )
