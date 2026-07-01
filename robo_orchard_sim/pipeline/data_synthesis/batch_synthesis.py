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

"""Batch helpers for grouped data-synthesis runs."""

from __future__ import annotations
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robo_orchard_sim.pipeline.data_synthesis.multi_task import (
    MultiTaskDataSynthesisCfg,
    MultiTaskDataSynthesisRunner,
    MultiTaskRunResult,
)
from robo_orchard_sim.pipeline.data_synthesis.single_task import (
    EpisodeSummary,
    TaskDataSynthesisCfg,
    TaskRunResult,
)


@dataclass(frozen=True)
class BatchGroup:
    """One job group in a batch plan."""

    group_id: str
    seed: int
    configs: list[str]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this group."""
        return {
            "group_id": self.group_id,
            "seed": self.seed,
            "configs": list(self.configs),
        }


@dataclass(frozen=True)
class BatchPlan:
    """A grouped data-synthesis batch plan."""

    batch_id: str
    task: str
    episodes_per_config: int
    base_seed: int
    groups: list[BatchGroup]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this plan."""
        return {
            "batch_id": self.batch_id,
            "task": self.task,
            "episodes_per_config": self.episodes_per_config,
            "base_seed": self.base_seed,
            "groups": [group.to_payload() for group in self.groups],
        }


def read_config_list(config_list_path: str) -> list[str]:
    """Read a newline-delimited task config list."""
    configs = [
        line.strip()
        for line in Path(config_list_path)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if not configs:
        raise ValueError(f"config list is empty: {config_list_path}")

    seen: set[str] = set()
    for config in configs:
        if config in seen:
            raise ValueError(f"duplicate config path in list: {config}")
        seen.add(config)
    return configs


def schedule_batch_plan(
    *,
    task: str,
    config_list_path: str,
    configs_per_group: int,
    episodes_per_config: int,
    base_seed: int,
    batch_id: str,
) -> BatchPlan:
    """Group explicit task YAML paths into a deterministic batch plan."""
    if configs_per_group < 1:
        raise ValueError("configs_per_group must be >= 1")
    if episodes_per_config < 1:
        raise ValueError("episodes_per_config must be >= 1")

    configs = read_config_list(config_list_path)
    groups: list[BatchGroup] = []
    for group_index, start in enumerate(
        range(0, len(configs), configs_per_group)
    ):
        group_configs = configs[start : start + configs_per_group]
        groups.append(
            BatchGroup(
                group_id=f"group_{group_index:04d}",
                seed=(
                    base_seed
                    + group_index * configs_per_group * episodes_per_config
                ),
                configs=group_configs,
            )
        )

    return BatchPlan(
        batch_id=batch_id,
        task=task,
        episodes_per_config=episodes_per_config,
        base_seed=base_seed,
        groups=groups,
    )


def write_batch_plan(plan: BatchPlan, output_path: str) -> None:
    """Write a batch plan JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plan.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_batch_plan(manifest_path: str) -> BatchPlan:
    """Load a batch plan from JSON.

    Entries in each group's ``configs`` may be absolute paths or relative
    paths; relative paths are resolved against the manifest's directory.
    """
    manifest_dir = Path(manifest_path).resolve().parent
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    def _resolve_config(entry: Any) -> str:
        path = Path(str(entry))
        if not path.is_absolute():
            path = manifest_dir / path
        return str(path.resolve())

    groups = [
        BatchGroup(
            group_id=str(group["group_id"]),
            seed=int(group["seed"]),
            configs=[_resolve_config(config) for config in group["configs"]],
        )
        for group in payload["groups"]
    ]
    return BatchPlan(
        batch_id=str(payload["batch_id"]),
        task=str(payload["task"]),
        episodes_per_config=int(payload["episodes_per_config"]),
        base_seed=int(payload["base_seed"]),
        groups=groups,
    )


def find_group(plan: BatchPlan, group_id: str) -> BatchGroup:
    """Return a group from a batch plan by id."""
    for group in plan.groups:
        if group.group_id == group_id:
            return group
    raise ValueError(f"group_id not found in manifest: {group_id}")


def build_task_cfgs_for_group(
    *,
    plan: BatchPlan,
    group_id: str,
    asset_root: str,
    task_root_dir: str,
    splits_path: Path | None = None,
    snapshot_path: Path | None = None,
) -> list[TaskDataSynthesisCfg]:
    """Build runner task configs for one manifest group."""
    del task_root_dir
    group = find_group(plan, group_id)
    task_cfgs: list[TaskDataSynthesisCfg] = []
    for config_index, config_path in enumerate(group.configs):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"task config not found: {config_path}")
        task_save_root = f"{group.group_id}/config_{config_index:04d}"
        task_cfgs.append(
            TaskDataSynthesisCfg(
                task=plan.task,
                asset_root=asset_root,
                config=str(path),
                seed=group.seed + config_index * plan.episodes_per_config,
                episode_num=plan.episodes_per_config,
                task_save_root=task_save_root,
                snapshot_path=snapshot_path,
                splits_path=splits_path,
                user_data={
                    "batch_id": plan.batch_id,
                    "group_id": group.group_id,
                    "config_index": config_index,
                    "config_path": str(path),
                },
            )
        )
    return task_cfgs


def copy_task_configs_to_output_dirs(
    *,
    task_cfgs: list[TaskDataSynthesisCfg],
    task_root_dir: str,
) -> None:
    """Copy each source task YAML into its task output config directory."""
    for task_cfg in task_cfgs:
        if task_cfg.config is None or task_cfg.task_save_root is None:
            continue
        output_config_dir = (
            Path(task_root_dir) / task_cfg.task_save_root / "config"
        )
        output_config_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            task_cfg.config,
            output_config_dir / Path(task_cfg.config).name,
        )


def _write_jsonl(output_path: Path, records: list[dict[str, Any]]) -> None:
    with output_path.open("w", encoding="utf-8") as fw:
        for record in records:
            fw.write(json.dumps(record, sort_keys=True))
            fw.write("\n")


def _episode_record_payload(
    *,
    task_result: TaskRunResult,
    episode: EpisodeSummary,
) -> dict[str, Any]:
    return {
        **task_result.user_data,
        "config_path": task_result.config_path,
        "episode_index": episode.episode_index,
        "seed": episode.seed,
        "success": episode.success,
        "stop_reason": episode.stop_reason,
        "steps": episode.steps,
        "record_dir": episode.record_dir,
        "mcap_paths": list(episode.mcap_paths),
        "error": episode.error,
    }


def _task_summary_payload(
    *,
    plan: BatchPlan,
    task_result: TaskRunResult,
) -> dict[str, Any]:
    successful_record_files = []
    if task_result.config_dir is not None:
        successful_record_files.append(
            str(
                Path(task_result.config_dir)
                / f"successful_recording_paths_{plan.task}.txt"
            )
        )
    return {
        **task_result.user_data,
        "record_dir": task_result.record_dir,
        "successful_record_files": successful_record_files,
        "task_save_root": task_result.task_save_root,
        "success_count": task_result.success_count,
        "total": task_result.total,
        "success_rate": task_result.success_rate,
    }


def write_group_outputs(
    *,
    output_dir: str,
    plan: BatchPlan,
    group_id: str,
    result: MultiTaskRunResult,
) -> None:
    """Write group-level result summaries and manifests."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    episode_records = [
        _episode_record_payload(
            task_result=task_result,
            episode=episode,
        )
        for task_result in result.task_results
        for episode in task_result.episodes
    ]
    _write_jsonl(
        output_path / f"all_episode_records_{plan.task}.jsonl",
        episode_records,
    )
    (output_path / f"all_mcap_paths_{plan.task}.txt").write_text(
        "\n".join(result.mcap_paths) + ("\n" if result.mcap_paths else ""),
        encoding="utf-8",
    )
    summary = {
        "assets": [
            _task_summary_payload(
                plan=plan,
                task_result=task_result,
            )
            for task_result in result.task_results
        ],
        "batch_id": plan.batch_id,
        "group_id": group_id,
        "task": plan.task,
        "total_assets": result.total_tasks,
        "total_configs": result.total_tasks,
        "total_episodes": result.total_episodes,
        "success_count": result.success_episodes,
        "success_rate": result.success_rate,
    }
    (output_path / f"batch_run_summary_{plan.task}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_group_data_synthesis(
    *,
    plan: BatchPlan,
    group_id: str,
    asset_root: str,
    task_root_dir: str,
    splits_path: Path | None = None,
    snapshot_path: Path | None = None,
) -> MultiTaskRunResult:
    """Run one manifest group through the multi-task synthesis runner."""
    task_cfgs = build_task_cfgs_for_group(
        plan=plan,
        group_id=group_id,
        asset_root=asset_root,
        task_root_dir=task_root_dir,
        splits_path=splits_path,
        snapshot_path=snapshot_path,
    )
    copy_task_configs_to_output_dirs(
        task_cfgs=task_cfgs,
        task_root_dir=task_root_dir,
    )
    result = MultiTaskDataSynthesisRunner(
        MultiTaskDataSynthesisCfg(
            task_root_dir=task_root_dir,
            task_cfgs=task_cfgs,
        )
    ).run()
    write_group_outputs(
        output_dir=task_root_dir,
        plan=plan,
        group_id=group_id,
        result=result,
    )
    return result
