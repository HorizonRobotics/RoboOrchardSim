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

"""Serial multi-config evaluator."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field
from robo_orchard_core.utils.config import ClassConfig, ClassType_co

from robo_orchard_sim.pipeline.evaluator.base import (
    MultiEvaluationResult,
    TaskEvaluationResult,
)
from robo_orchard_sim.pipeline.evaluator.evaluator import (
    EvaluationRuntime,
    Evaluator,
    EvaluatorCfg,
)


@dataclass
class EvaluationRunEntry:
    """One evaluator config with output metadata for multi-evaluation."""

    evaluator_cfg: EvaluatorCfg
    result_json_path: str | None = None
    user_data: dict[str, Any] = field(default_factory=dict)


class MultiEvaluator:
    """Run multiple evaluator configs serially for one policy."""

    InitFromConfig: bool = True

    cfg: "MultiEvaluatorCfg"

    def __init__(
        self,
        cfg: "MultiEvaluatorCfg",
        *,
        evaluator_class: Any = Evaluator,
    ) -> None:
        self.cfg = cfg
        self.evaluator_class = evaluator_class
        self.entries = self._resolve_entries()

    def _resolve_entries(self) -> list[EvaluationRunEntry]:
        if self.cfg.entries:
            return list(self.cfg.entries)
        return [
            EvaluationRunEntry(evaluator_cfg=evaluator_cfg)
            for evaluator_cfg in self.cfg.evaluator_cfgs
        ]

    def evaluate(self, policy_or_cfg: Any) -> MultiEvaluationResult:
        """Evaluate one policy across every configured task YAML."""
        if not self.entries:
            return self._build_multi_evaluation_result([])

        first_evaluator = self.evaluator_class(self.entries[0].evaluator_cfg)
        if hasattr(first_evaluator, "create_launcher") and hasattr(
            first_evaluator,
            "run_with_runtime",
        ):
            return self._evaluate_with_owned_runtime(
                policy_or_cfg,
                first_evaluator=first_evaluator,
            )

        task_results: list[TaskEvaluationResult] = []
        for entry in self.entries:
            try:
                result = self._evaluate_entry(entry, policy_or_cfg)
            except (Exception, SystemExit) as exc:
                if not self.cfg.continue_on_task_error:
                    raise
                result = TaskEvaluationResult(
                    task_name=entry.evaluator_cfg.task_name,
                    config_path=entry.evaluator_cfg.task_config_path,
                    result_json_path=entry.result_json_path,
                    error=f"{type(exc).__name__}: {exc}",
                    user_data=dict(entry.user_data),
                )
            task_results.append(result)

        return self._build_multi_evaluation_result(task_results)

    def _evaluate_with_owned_runtime(
        self,
        policy_or_cfg: Any,
        *,
        first_evaluator: Any,
    ) -> MultiEvaluationResult:
        launcher = first_evaluator.create_launcher()
        runtime = EvaluationRuntime(sim_app=launcher.app)
        try:
            task_results: list[TaskEvaluationResult] = []
            for entry_index, entry in enumerate(self.entries):
                evaluator = (
                    first_evaluator
                    if entry_index == 0
                    else self.evaluator_class(entry.evaluator_cfg)
                )
                try:
                    result = self._evaluate_entry_with_runtime(
                        entry=entry,
                        evaluator=evaluator,
                        policy_or_cfg=policy_or_cfg,
                        runtime=runtime,
                    )
                except (Exception, SystemExit) as exc:
                    if not self.cfg.continue_on_task_error:
                        raise
                    result = TaskEvaluationResult(
                        task_name=entry.evaluator_cfg.task_name,
                        config_path=entry.evaluator_cfg.task_config_path,
                        result_json_path=entry.result_json_path,
                        error=f"{type(exc).__name__}: {exc}",
                        user_data=dict(entry.user_data),
                    )
                task_results.append(result)
            return self._build_multi_evaluation_result(task_results)
        finally:
            close = getattr(launcher, "close", None)
            if callable(close):
                close()
            else:
                destructor = getattr(launcher, "__del__", None)
                if callable(destructor):
                    destructor()

    def _evaluate_entry_with_runtime(
        self,
        *,
        entry: EvaluationRunEntry,
        evaluator: Any,
        policy_or_cfg: Any,
        runtime: EvaluationRuntime,
    ) -> TaskEvaluationResult:
        result = evaluator.run_with_runtime(
            policy_or_cfg,
            runtime=runtime,
        )
        return TaskEvaluationResult(
            task_name=entry.evaluator_cfg.task_name,
            config_path=entry.evaluator_cfg.task_config_path,
            result_json_path=entry.result_json_path,
            result=result,
            user_data=dict(entry.user_data),
        )

    def _evaluate_entry(
        self,
        entry: EvaluationRunEntry,
        policy_or_cfg: Any,
    ) -> TaskEvaluationResult:
        evaluator = self.evaluator_class(entry.evaluator_cfg)
        if hasattr(evaluator, "__enter__") and hasattr(evaluator, "__exit__"):
            with evaluator as active_evaluator:
                result = active_evaluator.evaluate(policy_or_cfg)
        else:
            try:
                result = evaluator.evaluate(policy_or_cfg)
            finally:
                close = getattr(evaluator, "close", None)
                if callable(close):
                    close()
        return TaskEvaluationResult(
            task_name=entry.evaluator_cfg.task_name,
            config_path=entry.evaluator_cfg.task_config_path,
            result_json_path=entry.result_json_path,
            result=result,
            user_data=dict(entry.user_data),
        )

    def _build_multi_evaluation_result(
        self,
        task_results: list[TaskEvaluationResult],
        *,
        error: str | None = None,
    ) -> MultiEvaluationResult:
        episode_results = [
            episode
            for task_result in task_results
            if task_result.result is not None
            for episode in task_result.result.episode_results
        ]
        total_episodes = len(episode_results)
        success_episodes = sum(episode.success for episode in episode_results)
        average_progress = (
            sum(episode.progress for episode in episode_results)
            / total_episodes
            if total_episodes
            else 0.0
        )
        return MultiEvaluationResult(
            task_results=task_results,
            total_tasks=len(task_results),
            success_tasks=sum(
                task_result.error is None for task_result in task_results
            ),
            total_episodes=total_episodes,
            success_episodes=success_episodes,
            success_rate=(
                success_episodes / total_episodes if total_episodes else 0.0
            ),
            average_progress=average_progress,
            error=error,
            user_data=dict(self.cfg.user_data),
        )


class MultiEvaluatorCfg(ClassConfig[MultiEvaluator]):
    """Configuration for serial multi-config evaluation."""

    class_type: ClassType_co[MultiEvaluator] = MultiEvaluator
    evaluator_cfgs: list[EvaluatorCfg] = Field(default_factory=list)
    entries: list[EvaluationRunEntry] = Field(default_factory=list)
    continue_on_task_error: bool = True
    user_data: dict[str, Any] = Field(default_factory=dict)
