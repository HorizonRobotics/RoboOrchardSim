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

"""Batch helpers for grouped policy evaluation runs."""

from __future__ import annotations
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from robo_orchard_sim.pipeline.data_synthesis.batch_synthesis import (
    BatchPlan,
    find_group,
)
from robo_orchard_sim.pipeline.evaluator.base import (
    EvaluationResult,
    MultiEvaluationResult,
    TaskEvaluationResult,
)
from robo_orchard_sim.pipeline.evaluator.evaluator import EvaluatorCfg
from robo_orchard_sim.pipeline.evaluator.multi_evaluator import (
    EvaluationRunEntry,
    MultiEvaluator,
    MultiEvaluatorCfg,
)


def build_evaluator_cfgs_for_group(
    *,
    plan: BatchPlan,
    group_id: str,
    asset_root: str,
    output_root_dir: str,
    max_steps: int,
    enable_recording: bool = False,
    snapshot_path: Path | None = None,
    splits_path: Path | None = None,
) -> list[EvaluationRunEntry]:
    """Build evaluator configs for one existing batch-plan group."""
    group = find_group(plan, group_id)
    entries: list[EvaluationRunEntry] = []
    for config_index, config_path in enumerate(group.configs):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"task config not found: {config_path}")

        task_save_root = (
            Path(output_root_dir)
            / group.group_id
            / (f"config_{config_index:04d}")
        )
        user_data = {
            "batch_id": plan.batch_id,
            "group_id": group.group_id,
            "config_index": config_index,
            "config_path": str(path),
        }
        entries.append(
            EvaluationRunEntry(
                evaluator_cfg=EvaluatorCfg(
                    task_name=plan.task,
                    asset_root=asset_root,
                    task_config_path=str(path),
                    enable_recording=enable_recording,
                    record_dir=str(task_save_root / "records"),
                    seed=(
                        group.seed + config_index * plan.episodes_per_config
                    ),
                    episode_num=plan.episodes_per_config,
                    max_steps=max_steps,
                    snapshot_path=snapshot_path,
                    splits_path=splits_path,
                ),
                result_json_path=str(task_save_root / "eval_result.json"),
                user_data=user_data,
            )
        )
    return entries


def copy_task_configs_to_output_dirs(
    *,
    entries: list[EvaluationRunEntry],
    output_root_dir: str,
) -> None:
    """Copy source task YAML files into each per-config output directory."""
    del output_root_dir
    for entry in entries:
        config_path = entry.evaluator_cfg.task_config_path
        result_json_path = entry.result_json_path
        if config_path is None or result_json_path is None:
            continue
        output_config_dir = Path(result_json_path).parent / "config"
        output_config_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            config_path,
            output_config_dir / Path(config_path).name,
        )


def _write_jsonl(output_path: Path, records: list[dict[str, Any]]) -> None:
    with output_path.open("w", encoding="utf-8") as fw:
        for record in records:
            fw.write(json.dumps(record, sort_keys=True))
            fw.write("\n")


def _success_count(result: EvaluationResult | None) -> int:
    if result is None:
        return 0
    return sum(episode.success for episode in result.episode_results)


def _episode_records(
    task_result: TaskEvaluationResult,
) -> list[dict[str, Any]]:
    if task_result.result is None:
        return []
    return [
        {
            **task_result.user_data,
            "episode_index": episode_index,
            "seed": episode.seed,
            "success": episode.success,
            "progress": episode.progress,
            "steps": episode.steps,
            "stop_reason": episode.stop_reason,
            "metrics": episode.metrics,
        }
        for episode_index, episode in enumerate(
            task_result.result.episode_results
        )
    ]


def _config_summary_payload(
    task_result: TaskEvaluationResult,
) -> dict[str, Any]:
    result = task_result.result
    total = len(result.episode_results) if result is not None else 0
    return {
        **task_result.user_data,
        "error": task_result.error,
        "eval_result_json": task_result.result_json_path,
        "seed": result.seed_start if result is not None else None,
        "episode_num": total,
        "success_count": _success_count(result),
        "success_rate": result.success_rate if result is not None else 0.0,
        "average_progress": (
            result.average_progress if result is not None else 0.0
        ),
        "total": total,
    }


def write_group_outputs(
    *,
    output_dir: str,
    plan: BatchPlan,
    group_id: str,
    result: MultiEvaluationResult,
) -> None:
    """Write group-level evaluation summaries and per-config results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for task_result in result.task_results:
        if task_result.result is None or task_result.result_json_path is None:
            continue
        result_path = Path(task_result.result_json_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(asdict(task_result.result), indent=2) + "\n",
            encoding="utf-8",
        )

    episode_records = [
        record
        for task_result in result.task_results
        for record in _episode_records(task_result)
    ]
    _write_jsonl(
        output_path / f"all_episode_eval_records_{plan.task}.jsonl",
        episode_records,
    )

    summary = {
        "batch_id": plan.batch_id,
        "configs": [
            _config_summary_payload(task_result)
            for task_result in result.task_results
        ],
        "group_id": group_id,
        "success_count": result.success_episodes,
        "success_rate": result.success_rate,
        "average_progress": result.average_progress,
        "task": plan.task,
        "total_configs": result.total_tasks,
        "total_episodes": result.total_episodes,
    }
    (output_path / f"batch_eval_summary_{plan.task}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_group_evaluation(
    *,
    plan: BatchPlan,
    group_id: str,
    asset_root: str,
    output_root_dir: str,
    policy_or_cfg: Any,
    max_steps: int,
    enable_recording: bool = False,
    snapshot_path: Path | None = None,
    splits_path: Path | None = None,
    continue_on_task_error: bool = True,
) -> MultiEvaluationResult:
    """Run one batch-plan group through the multi-config evaluator."""
    entries = build_evaluator_cfgs_for_group(
        plan=plan,
        group_id=group_id,
        asset_root=asset_root,
        output_root_dir=output_root_dir,
        max_steps=max_steps,
        enable_recording=enable_recording,
        snapshot_path=snapshot_path,
        splits_path=splits_path,
    )
    copy_task_configs_to_output_dirs(
        entries=entries,
        output_root_dir=output_root_dir,
    )
    result = MultiEvaluator(
        MultiEvaluatorCfg(
            entries=entries,
            continue_on_task_error=continue_on_task_error,
        )
    ).evaluate(policy_or_cfg)
    write_group_outputs(
        output_dir=output_root_dir,
        plan=plan,
        group_id=group_id,
        result=result,
    )
    return result
