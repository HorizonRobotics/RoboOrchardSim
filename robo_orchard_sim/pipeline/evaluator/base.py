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

"""Shared evaluator result and configuration models."""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robo_orchard_sim.pipeline.evaluator.distributed.sharding import (
        EvaluationShard,
    )

__all__ = [
    "RESAMPLE_ATTEMPT_MULTIPLIER",
    "EpisodeResult",
    "SkippedEpisode",
    "EvaluationResult",
    "TaskEvaluationResult",
    "MultiEvaluationResult",
    "SummaryMetrics",
    "ShardSummary",
    "TaskSummary",
    "EvaluationSummary",
]

RESAMPLE_ATTEMPT_MULTIPLIER = 3


@dataclass
class EpisodeResult:
    seed: int
    success: bool
    progress: float
    steps: int
    stop_reason: str
    metrics: dict[str, Any]
    # None means no scene checker store was available for this episode.
    checker_summary: dict[str, object] | None = None
    instruction: str = ""
    stage_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class SkippedEpisode:
    seed: int
    reason: str


@dataclass
class EvaluationResult:
    episode_num: int
    seed_start: int
    success_rate: float
    average_progress: float
    episode_results: list[EpisodeResult]
    skipped_episodes: list[SkippedEpisode] = field(default_factory=list)
    attempted_scene_seeds: list[int] = field(default_factory=list)


@dataclass
class TaskEvaluationResult:
    """Observable result for one task-config evaluation run."""

    task_name: str
    config_path: str | None
    result_json_path: str | None = None
    result: EvaluationResult | None = None
    error: str | None = None
    user_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiEvaluationResult:
    """Aggregated result for multiple task-config evaluation runs."""

    task_results: list[TaskEvaluationResult]
    total_tasks: int
    success_tasks: int
    total_episodes: int
    success_episodes: int
    success_rate: float
    average_progress: float
    error: str | None = None
    user_data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SummaryMetrics:
    """Episode-weighted metrics shared by all summary levels."""

    episodes: int
    successes: int
    success_rate: float
    avg_progress: float
    skipped_episodes: int
    stage_success_rate: dict[str, float]

    @classmethod
    def from_results(cls, results: list[EvaluationResult]) -> SummaryMetrics:
        """Calculate metrics from completed episodes, not shard averages."""
        episodes = [ep for result in results for ep in result.episode_results]
        successes = sum(ep.success for ep in episodes)
        stages: dict[str, list[int]] = {}
        for ep in episodes:
            reached = ep.metrics.get("criteria_reached") or {}
            if isinstance(reached, dict):
                for name, value in reached.items():
                    counts = stages.setdefault(name, [0, 0])
                    counts[0] += bool(value)
                    counts[1] += 1
        return cls(
            episodes=len(episodes),
            successes=successes,
            success_rate=successes / len(episodes) if episodes else 0.0,
            avg_progress=(
                sum(ep.progress for ep in episodes) / len(episodes)
                if episodes
                else 0.0
            ),
            skipped_episodes=sum(len(r.skipped_episodes) for r in results),
            stage_success_rate={k: n / d for k, (n, d) in stages.items()},
        )


@dataclass
class ShardSummary:
    """Outcome of one scheduled shard, including failed or missing work."""

    shard: EvaluationShard
    result: EvaluationResult | None = None
    error: str | None = None
    result_json_path: str | None = None

    @property
    def metrics(self) -> SummaryMetrics:
        """Return metrics for completed work in this shard."""
        return SummaryMetrics.from_results(
            [self.result] if self.result is not None else []
        )

    @property
    def status(self) -> str:
        """Distinguish episode failure from an incomplete worker run."""
        if self.error or self.result is None:
            return "partial_error" if self.metrics.episodes else "error"
        return "ok"

    def to_payload(self) -> dict[str, Any]:
        """Serialize metrics and the existing worker result reference."""
        return {
            "shard": self.shard.to_payload(),
            "status": self.status,
            **asdict(self.metrics),
            "error": self.error,
            "configs": [
                {
                    "group_id": self.shard.group_id,
                    "config_index": self.shard.config_index,
                    "config_path": self.shard.config_path,
                    "eval_result_json": self.result_json_path,
                    "error": self.error,
                }
            ],
        }


@dataclass
class TaskSummary:
    """Aggregate all shards belonging to one named evaluation task."""

    task_name: str
    shards: list[ShardSummary]

    def __post_init__(self) -> None:
        if any(s.shard.task_name != self.task_name for s in self.shards):
            raise ValueError("shard belongs to a different task")
        ids = [s.shard.shard_id for s in self.shards]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate shard summary")

    @property
    def metrics(self) -> SummaryMetrics:
        """Aggregate retained episode results across shards."""
        return SummaryMetrics.from_results(
            [s.result for s in self.shards if s.result is not None]
        )

    @property
    def status(self) -> str:
        """Report incomplete work without discarding completed metrics."""
        if not self.shards or any(s.status != "ok" for s in self.shards):
            return "partial_error" if self.metrics.episodes else "error"
        return "ok"

    def to_payload(self) -> dict[str, Any]:
        """Serialize task metrics with shard details and config counts."""
        configs = {
            (s.shard.group_id, s.shard.config_index) for s in self.shards
        }
        failed = {
            (s.shard.group_id, s.shard.config_index)
            for s in self.shards
            if s.status != "ok"
        }
        return {
            "task_name": self.task_name,
            "status": self.status,
            **asdict(self.metrics),
            "error": "; ".join(s.error for s in self.shards if s.error)
            or None,
            "configs": len(configs),
            "configs_total": len(configs),
            "configs_failed": len(failed),
            "configs_succeeded": len(configs) - len(failed),
            "shards": [s.to_payload() for s in self.shards],
        }


@dataclass
class EvaluationSummary:
    """Whole-run summary, retaining the task/shard result hierarchy."""

    tasks: list[TaskSummary]

    def __post_init__(self) -> None:
        names = [task.task_name for task in self.tasks]
        if len(names) != len(set(names)):
            raise ValueError("duplicate task summary")

    @property
    def metrics(self) -> SummaryMetrics:
        """Weight overall metrics by episodes across all tasks."""
        return SummaryMetrics.from_results(
            [
                s.result
                for task in self.tasks
                for s in task.shards
                if s.result is not None
            ]
        )

    @property
    def status(self) -> str:
        """Preserve whole-run error status when any task entirely fails."""
        if not self.tasks or any(t.status == "error" for t in self.tasks):
            return "error"
        return (
            "ok"
            if all(t.status == "ok" for t in self.tasks)
            else "partial_error"
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize overall metrics and all task/shard summaries."""
        tasks = {t.task_name: t.to_payload() for t in self.tasks}
        metrics = self.metrics
        return {
            "tasks": tasks,
            "overall": {
                "status": self.status,
                "ok_tasks": sum(t.status == "ok" for t in self.tasks),
                "partial_error_tasks": sum(
                    t.status == "partial_error" for t in self.tasks
                ),
                "total_tasks": len(self.tasks),
                "total_episodes": metrics.episodes,
                "success_episodes": metrics.successes,
                "skipped_episodes": metrics.skipped_episodes,
                "success_rate": metrics.success_rate,
                "avg_progress": metrics.avg_progress,
                "stage_success_rate": metrics.stage_success_rate,
                **{
                    key: sum(t[key] for t in tasks.values())
                    for key in (
                        "configs_total",
                        "configs_failed",
                        "configs_succeeded",
                    )
                },
            },
        }
