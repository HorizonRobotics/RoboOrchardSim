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
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "EpisodeResult",
    "SkippedEpisode",
    "EvaluationResult",
    "TaskEvaluationResult",
    "MultiEvaluationResult",
]


@dataclass
class EpisodeResult:
    seed: int
    success: bool
    progress: float
    steps: int
    stop_reason: str
    metrics: dict[str, Any]


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
