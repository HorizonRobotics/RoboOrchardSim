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

"""Execution-resource configuration for evaluator work items."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationExecutionConfig:
    """Map evaluator work items onto the configured GPU worker slots."""

    gpus: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.gpus:
            raise ValueError("at least one evaluator GPU is required")
        if len(set(self.gpus)) != len(self.gpus):
            raise ValueError("evaluator GPUs must be unique")

    def gpu_slots(self) -> tuple[str, ...]:
        """Return one GPU id for every available worker process slot."""
        return self.gpus

    def concurrency(self, work_item_count: int) -> int:
        """Return the number of work items that may execute concurrently."""
        if work_item_count < 1:
            return 0
        return min(work_item_count, len(self.gpu_slots()))


@dataclass(frozen=True)
class EvaluationWorkload:
    """Estimated work for a task; explicit shards disable auto splitting."""

    episodes_per_config: int
    max_steps: int
    config_count: int = 1
    episodes_per_scene: int = 1
    shards: int | None = None

    def __post_init__(self) -> None:
        if (
            min(
                self.episodes_per_config,
                self.max_steps,
                self.config_count,
                self.episodes_per_scene,
            )
            < 1
        ):
            raise ValueError("workload values must be positive")
        if self.shards is not None and self.shards < 1:
            raise ValueError("shards must be positive")

    def scene_count(self) -> int:
        """Return the maximum number of nonempty shards per config."""
        return -(-self.episodes_per_config // self.episodes_per_scene)


def allocate_task_shards(
    workloads: list[EvaluationWorkload],
    gpu_count: int,
) -> list[int]:
    """Fill GPU slots by splitting the longest estimated automatic shard.

    Start with one shard per config, or the explicit override. While slots
    remain, increase the shard count of the task with the longest estimated
    shard (episodes * max_steps / shards). Scene boundaries cap splitting.
    This is a deterministic greedy heuristic, not an optimality guarantee.
    """
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    counts = [min(w.shards or 1, w.scene_count()) for w in workloads]
    while (
        sum(w.config_count * n for w, n in zip(workloads, counts, strict=True))
        < gpu_count
    ):
        candidates = [
            i
            for i, w in enumerate(workloads)
            if w.shards is None and counts[i] < w.scene_count()
        ]
        if not candidates:
            break
        index = max(
            candidates,
            key=lambda i: (
                workloads[i].episodes_per_config
                * workloads[i].max_steps
                / counts[i]
            ),
        )
        counts[index] += 1
    return counts
