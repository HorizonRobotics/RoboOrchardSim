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

"""Serial multi-task data synthesis runner."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any

from robo_orchard_core.utils.config import ClassConfig, ClassType_co

from robo_orchard_sim.pipeline.data_synthesis.single_task import (
    DataSynthesisRuntime,
    TaskDataSynthesisCfg,
    TaskDataSynthesisRunner,
    TaskRunResult,
)


@dataclass
class MultiTaskRunResult:
    """Observable result for one serial multi-task data synthesis run."""

    task_results: list[TaskRunResult]
    total_tasks: int
    success_tasks: int
    total_episodes: int
    success_episodes: int
    success_rate: float
    error: str | None = None
    mcap_paths: list[str] = field(default_factory=list)
    user_data: dict[str, Any] = field(default_factory=dict)


class MultiTaskDataSynthesisRunner:
    """Run multiple task data synthesis requests serially."""

    InitFromConfig = True

    cfg: "MultiTaskDataSynthesisCfg"
    task_runners: list[TaskDataSynthesisRunner]

    def __init__(
        self,
        cfg: "MultiTaskDataSynthesisCfg",
        *,
        runner_class: Any = TaskDataSynthesisRunner,
    ) -> None:
        self.cfg = cfg
        self.task_runners = [
            runner_class(self._resolve_task_cfg(task_cfg))
            for task_cfg in self.cfg.task_cfgs
        ]

    def _resolve_task_cfg(
        self,
        task_cfg: TaskDataSynthesisCfg,
    ) -> TaskDataSynthesisCfg:
        if self.cfg.task_root_dir is None:
            return task_cfg
        if task_cfg.task_save_root is None:
            raise ValueError(
                "task_save_root is required when task_root_dir is provided"
            )
        if os.path.isabs(task_cfg.task_save_root):
            raise ValueError(
                "task_root_dir requires a relative task_save_root, got "
                f"{task_cfg.task_save_root!r}"
            )

        normalized = os.path.normpath(task_cfg.task_save_root)
        if (
            normalized == os.curdir
            or normalized == os.pardir
            or normalized.startswith(os.pardir + os.sep)
        ):
            raise ValueError(
                "task_root_dir requires a relative task_save_root below the "
                f"root directory, got {task_cfg.task_save_root!r}"
            )

        task_cfg.task_save_root = os.path.join(
            self.cfg.task_root_dir,
            normalized,
        )
        return task_cfg

    def run(self) -> MultiTaskRunResult:
        """Launch Isaac once and run all configured tasks serially."""
        if not self.task_runners:
            return self._build_multi_task_run_result([])

        launcher = self.task_runners[0].create_launcher()
        runtime = DataSynthesisRuntime(sim_app=launcher.app)

        try:
            return self.run_with_runtime(runtime=runtime)
        finally:
            launcher.close()

    def run_with_runtime(
        self,
        runtime: DataSynthesisRuntime | None = None,
        *,
        sim_app: Any | None = None,
    ) -> MultiTaskRunResult:
        """Run all configured tasks with an externally owned runtime."""
        if runtime is None:
            if sim_app is None:
                raise ValueError(
                    "sim_app is required when runtime is not provided"
                )
            runtime = DataSynthesisRuntime(sim_app=sim_app)

        task_results: list[TaskRunResult] = []
        for runner in self.task_runners:
            try:
                task_results.append(
                    self._run_task_with_runtime(runner, runtime)
                )
            except (Exception, SystemExit) as exc:
                if not self.cfg.continue_on_task_error:
                    raise
                task_results.append(
                    self._build_failed_task_result(
                        cfg=runner.cfg,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        return self._build_multi_task_run_result(task_results)

    def _run_task_with_runtime(
        self,
        runner: TaskDataSynthesisRunner,
        runtime: DataSynthesisRuntime,
    ) -> TaskRunResult:
        if hasattr(runner, "run_task_with_runtime"):
            result = runner.run_task_with_runtime(runtime=runtime)
        else:
            result = runner.run_with_runtime(sim_app=runtime.sim_app)
        result.user_data = {**runner.cfg.user_data, **result.user_data}
        return result

    def _build_failed_task_result(
        self,
        *,
        cfg: TaskDataSynthesisCfg,
        error: str,
    ) -> TaskRunResult:
        return TaskRunResult(
            task=cfg.task,
            config_path=cfg.config,
            task_save_root=cfg.task_save_root,
            config_dir=cfg.output_config_dir,
            data_dir=cfg.record_dir,
            record_dir=cfg.record_dir,
            episodes=[],
            total=0,
            success_count=0,
            success_rate=0.0,
            error=error,
            user_data=dict(cfg.user_data),
        )

    def _build_multi_task_run_result(
        self,
        task_results: list[TaskRunResult],
        *,
        error: str | None = None,
    ) -> MultiTaskRunResult:
        total_episodes = sum(result.total for result in task_results)
        success_episodes = sum(result.success_count for result in task_results)
        success_tasks = sum(result.error is None for result in task_results)
        mcap_paths = [
            path
            for task_result in task_results
            for episode in task_result.episodes
            for path in episode.mcap_paths
        ]
        return MultiTaskRunResult(
            task_results=task_results,
            total_tasks=len(task_results),
            success_tasks=success_tasks,
            total_episodes=total_episodes,
            success_episodes=success_episodes,
            success_rate=(
                success_episodes / total_episodes if total_episodes else 0.0
            ),
            error=error,
            mcap_paths=mcap_paths,
            user_data=dict(self.cfg.user_data),
        )


class MultiTaskDataSynthesisCfg(ClassConfig[MultiTaskDataSynthesisRunner]):
    """Configuration for the serial multi-task data synthesis runner."""

    class_type: ClassType_co[MultiTaskDataSynthesisRunner] = (
        MultiTaskDataSynthesisRunner
    )
    task_root_dir: str | None = None
    task_cfgs: list[TaskDataSynthesisCfg]
    continue_on_task_error: bool = True
    user_data: dict[str, Any] = {}
