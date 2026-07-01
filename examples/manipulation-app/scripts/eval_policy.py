# ruff: noqa: E402, I001
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

"""Multi-task policy evaluation dispatcher.

One YAML describes the entire evaluation (policy / output / per-task
settings). Every task runs as a `BatchPlan`: either loaded from
`task.batch_plan`, or synthesized from a single task yaml plus
`episode_num`. Each task runs in its own subprocess on one GPU; results
are aggregated into `<output_dir>/summary.json`.

Run:
    PYTHONPATH=$PWD python3 \\
        examples/manipulation-app/scripts/eval_policy.py \\
        --eval-config examples/manipulation-app/configs/eval_example.yaml \\
        --output-dir eval_result/run_001 \\
        --gpus 0,1,2,3

CLI flags:
    --eval-config / --output-dir / --gpus / --enable-recording
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_POLICY_CONFIG_DIR = _REPO_ROOT / "robo_orchard_sim" / "policy" / "configs"
_BENCHMARK_ROOT = _REPO_ROOT / "robo_orchard_sim" / "benchmark"
_SINGLE_TASK_FLAG = "--_single-task"


# --------------------------------------------------------------------------- #
# Config schema (YAML <-> dataclasses)
# --------------------------------------------------------------------------- #


_TASK_FIELDS = {
    "task_type",
    "yaml",
    "batch_plan",
    "splits",
    "snapshot",
    "episode_num",
    "max_steps",
    "seed",
    "asset_root",
}
_DEFAULTS_FIELDS = _TASK_FIELDS - {"task_type", "yaml", "batch_plan"}
_TOP_LEVEL_FIELDS = {"policy", "defaults", "tasks"}


@dataclass(frozen=True)
class PolicyCfg:
    model_type: str
    model_yaml: str | None = None


@dataclass(frozen=True)
class TaskCfg:
    """Fully-resolved per-task settings (defaults already merged in).

    `name` is the user-facing instance name (the YAML key); it identifies
    the run in output paths and logs. `task_type` is the registered task
    name (required) and must match a name registered in
    `robo_orchard_sim/benchmark/registration.py`. Use distinct keys with
    the same `task_type` to run the same registered task multiple times
    with different yaml / splits.
    """

    name: str
    task_type: str
    asset_root: str
    seed: int = 0
    episode_num: int = 20
    max_steps: int = 1000
    splits: str | None = None
    snapshot: str | None = None
    # Mutually exclusive: yaml is a single task config; batch_plan is a
    # path to an existing batch plan JSON.
    yaml: str | None = None
    batch_plan: str | None = None


@dataclass(frozen=True)
class EvalConfig:
    policy: PolicyCfg
    tasks: list[TaskCfg]
    source_path: str = ""

    def task(self, name: str) -> TaskCfg:
        for t in self.tasks:
            if t.name == name:
                return t
        known = ", ".join(t.name for t in self.tasks)
        raise KeyError(f"unknown task {name!r}; known: {known}")


def _expand_env(obj: Any, *, src: Path) -> Any:
    """Recursively expand ${VAR} in string leaves of a parsed config.

    Raises ``RuntimeError`` with a clear hint when a referenced env var is
    unset, instead of silently leaving a literal ``${VAR}`` for downstream
    file IO to fail on.
    """
    if isinstance(obj, str):
        expanded = os.path.expandvars(obj)
        if "${" in expanded:
            raise RuntimeError(
                f"unresolved env var in {src}: {obj!r}. "
                "Set the variable (e.g. "
                "`export ORCHARD_ASSET=/path/to/asset/root`) "
                "or replace the placeholder with an absolute path."
            )
        return expanded
    if isinstance(obj, dict):
        return {k: _expand_env(v, src=src) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v, src=src) for v in obj]
    return obj


def load_eval_config(path: str | Path) -> EvalConfig:
    """Parse and validate an eval-config YAML file."""
    src = Path(path).resolve()
    raw = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"eval config must be a mapping: {src}")
    raw = _expand_env(raw, src=src)

    unknown_top = set(raw) - _TOP_LEVEL_FIELDS
    if unknown_top:
        raise ValueError(
            f"unknown top-level fields {sorted(unknown_top)} in {src}; "
            f"allowed: {sorted(_TOP_LEVEL_FIELDS)} "
            "(output_dir/gpus/enable_recording are CLI flags)"
        )

    policy = _parse_policy(_require(raw, "policy", src, dict), src)
    defaults = _parse_defaults(raw.get("defaults") or {}, src)

    tasks_raw = _require(raw, "tasks", src, dict)
    if not tasks_raw:
        raise ValueError(f"eval config has no tasks: {src}")
    tasks = [
        _parse_task(name=name, raw=spec or {}, defaults=defaults, src=src)
        for name, spec in tasks_raw.items()
    ]

    return EvalConfig(policy=policy, tasks=tasks, source_path=str(src))


def _require(d: dict, key: str, src: Path, ty: type) -> Any:
    if key not in d:
        raise ValueError(f"eval config missing `{key}`: {src}")
    value = d[key]
    if not isinstance(value, ty):
        raise ValueError(
            f"eval config field `{key}` must be {ty.__name__}: {src}"
        )
    return value


def _parse_policy(raw: dict, src: Path) -> PolicyCfg:
    model_type = _require(raw, "model_type", src, str)
    model_yaml = raw.get("model_yaml")
    if model_yaml is not None and not isinstance(model_yaml, str):
        raise ValueError(f"policy.model_yaml must be str: {src}")
    return PolicyCfg(model_type=model_type, model_yaml=model_yaml)


def _parse_defaults(raw: dict, src: Path) -> dict:
    unknown = set(raw) - _DEFAULTS_FIELDS
    if unknown:
        raise ValueError(
            f"unknown defaults fields {sorted(unknown)} in {src}; "
            f"allowed: {sorted(_DEFAULTS_FIELDS)}"
        )
    return dict(raw)


def _parse_task(*, name: str, raw: dict, defaults: dict, src: Path) -> TaskCfg:
    unknown = set(raw) - _TASK_FIELDS
    if unknown:
        raise ValueError(
            f"task {name!r}: unknown fields {sorted(unknown)} in {src}; "
            f"allowed: {sorted(_TASK_FIELDS)}"
        )
    merged = {**defaults, **raw}

    if not raw.get("task_type"):
        raise ValueError(
            f"task {name!r}: `task_type` is required and must match a "
            f"task registered in robo_orchard_sim/benchmark "
            f"({src})"
        )
    if merged.get("yaml") and merged.get("batch_plan"):
        raise ValueError(
            f"task {name!r}: `yaml` and `batch_plan` are mutually exclusive "
            f"({src})"
        )
    if "asset_root" not in merged or not merged["asset_root"]:
        raise ValueError(
            f"task {name!r}: `asset_root` must be set in defaults or task "
            f"({src})"
        )

    return TaskCfg(
        name=name,
        task_type=str(raw["task_type"]),
        asset_root=str(merged["asset_root"]),
        seed=int(merged.get("seed", 0)),
        episode_num=int(merged.get("episode_num", 20)),
        max_steps=int(merged.get("max_steps", 1000)),
        splits=_opt_str(merged.get("splits")),
        snapshot=_opt_str(merged.get("snapshot")),
        yaml=_opt_str(merged.get("yaml")),
        batch_plan=_opt_str(merged.get("batch_plan")),
    )


def _opt_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eval-config",
        type=str,
        required=True,
        help="Path to the eval-config YAML describing this evaluation.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Top-level output directory for this run.",
    )
    p.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Comma-separated GPU ids. Defaults to CUDA_VISIBLE_DEVICES, "
            "or [0] if unset."
        ),
    )
    p.add_argument(
        "--enable-recording",
        action="store_true",
        help="Enable MCAP recording for every task in this run.",
    )
    # Internal: marks the worker subprocess.
    p.add_argument(
        _SINGLE_TASK_FLAG,
        dest="_single_task",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    return p


def _resolve_gpus(cli_arg: str | None) -> list[str]:
    if cli_arg:
        gpus = [g.strip() for g in cli_arg.split(",") if g.strip()]
    elif env := os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        gpus = [g.strip() for g in env.split(",") if g.strip()]
    else:
        gpus = ["0"]
    if not gpus:
        raise SystemExit("No GPUs resolved for dispatch.")
    return gpus


# --------------------------------------------------------------------------- #
# Plan synthesis (the only place "single yaml" vs "batch plan" differ)
# --------------------------------------------------------------------------- #


def _registered_default_yaml(task: str) -> str | None:
    """Find the default yaml by convention: benchmark/**/configs/<task>.yaml.

    Pure filesystem lookup — does NOT import isaacsim or any task module.
    """
    matches = sorted(_BENCHMARK_ROOT.glob(f"**/configs/{task}.yaml"))
    if not matches:
        return None
    if len(matches) > 1:
        raise SystemExit(
            f"ambiguous default yaml for task {task!r}: {matches}"
        )
    return str(matches[0])


def _build_plan(task: TaskCfg):
    """Return a BatchPlan, whether from JSON or synthesized from a yaml."""
    from robo_orchard_sim.pipeline.data_synthesis.batch_synthesis import (
        BatchGroup,
        BatchPlan,
        load_batch_plan,
    )

    if task.batch_plan is not None:
        plan = load_batch_plan(task.batch_plan)
        if plan.task != task.task_type:
            raise SystemExit(
                f"task {task.name!r}: batch plan declares task="
                f"{plan.task!r} but expected {task.task_type!r} "
                f"({task.batch_plan})"
            )
        if not plan.groups:
            raise SystemExit(f"batch plan has no groups: {task.batch_plan}")
        for group in plan.groups:
            if not group.configs:
                raise SystemExit(
                    f"task {task.name!r}: batch plan group "
                    f"{group.group_id!r} has no configs ({task.batch_plan})"
                )
        return plan

    yaml_path = task.yaml or _registered_default_yaml(task.task_type)
    if yaml_path is None:
        raise SystemExit(
            f"task {task.name!r}: no `yaml` set and no default found at "
            f"benchmark/**/configs/{task.task_type}.yaml. Make sure "
            f"`task_type` ({task.task_type!r}) matches a registered task, "
            "or set `yaml`/`batch_plan` explicitly."
        )
    return BatchPlan(
        batch_id=f"{task.name}_{int(time.time())}",
        task=task.task_type,
        episodes_per_config=task.episode_num,
        base_seed=task.seed,
        groups=[
            BatchGroup(
                group_id="group_0000",
                seed=task.seed,
                configs=[yaml_path],
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def _task_dir(output_dir: Path, task_name: str) -> Path:
    return output_dir / task_name


def _task_log(output_dir: Path, task_name: str) -> Path:
    return _task_dir(output_dir, task_name) / "stdout.log"


def _resolve_model_yaml(policy: PolicyCfg) -> Path:
    if policy.model_yaml:
        return Path(policy.model_yaml)
    return _POLICY_CONFIG_DIR / f"{policy.model_type}.yaml"


# --------------------------------------------------------------------------- #
# Worker: --_single-task
# --------------------------------------------------------------------------- #


def _load_model_cfg(policy: PolicyCfg) -> dict:
    path = _resolve_model_yaml(policy)
    if not path.exists():
        raise FileNotFoundError(f"Policy config yaml not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Model yaml must contain a mapping: {path}")
    declared = loaded.get("policy")
    if declared is not None and declared != policy.model_type:
        raise ValueError(
            f"Policy type mismatch: model_type={policy.model_type}, "
            f"but yaml declares policy={declared}"
        )
    cfg = dict(loaded)
    cfg["policy"] = policy.model_type
    return cfg


def _build_policy(policy_cfg: PolicyCfg):
    from robo_orchard_core.policy.base import PolicyConfig
    from robo_orchard_sim.policy.factory import create_policy_from_model_cfg

    obj = create_policy_from_model_cfg(_load_model_cfg(policy_cfg))
    return obj() if isinstance(obj, PolicyConfig) else obj


def _run_single_task(
    eval_cfg: EvalConfig,
    task_name: str,
    output_dir: Path,
    enable_recording: bool,
) -> int:
    """Worker entry: build the plan for one task and execute every group."""
    from robo_orchard_core.utils.logging import LoggerManager
    from robo_orchard_sim.pipeline.evaluator.batch_evaluation import (
        run_group_evaluation,
    )

    logger = LoggerManager().get_child(__name__)
    task = eval_cfg.task(task_name)
    task_out = _task_dir(output_dir, task.name)
    task_out.mkdir(parents=True, exist_ok=True)

    plan = _build_plan(task)
    snapshot = Path(task.snapshot) if task.snapshot else None
    splits = Path(task.splits) if task.splits else None

    policy = _build_policy(eval_cfg.policy)
    try:
        total_eps = ok_eps = 0
        config_summaries: list[dict[str, Any]] = []
        for group in plan.groups:
            result = run_group_evaluation(
                plan=plan,
                group_id=group.group_id,
                asset_root=task.asset_root,
                output_root_dir=str(task_out),
                policy_or_cfg=policy,
                max_steps=task.max_steps,
                enable_recording=enable_recording,
                snapshot_path=snapshot,
                splits_path=splits,
            )
            total_eps += result.total_episodes
            ok_eps += result.success_episodes
            config_summaries.extend(
                _config_summary_payload(task_result)
                for task_result in result.task_results
            )
            logger.info(
                "[%s] group=%s: %d/%d episodes succeeded",
                task.name,
                group.group_id,
                result.success_episodes,
                result.total_episodes,
            )
        logger.info(
            "[%s] done: %d groups, %d/%d episodes succeeded",
            task.name,
            len(plan.groups),
            ok_eps,
            total_eps,
        )
        _write_task_eval_summary(
            task_out=task_out,
            task=task,
            plan_groups=len(plan.groups),
            configs=config_summaries,
        )
        return 0
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #


def _build_worker_cmd(
    task_name: str,
    eval_config_path: str,
    output_dir: Path,
    enable_recording: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--eval-config",
        eval_config_path,
        "--output-dir",
        str(output_dir),
        _SINGLE_TASK_FLAG,
        task_name,
    ]
    if enable_recording:
        cmd.append("--enable-recording")
    return cmd


def _run_worker(
    task_name: str,
    eval_config_path: str,
    output_dir: Path,
    enable_recording: bool,
    gpu_q: Queue,
) -> tuple[str, int, str | None]:
    """Wait for a GPU, exec the worker subprocess, free the GPU."""
    gpu = gpu_q.get()
    try:
        task_out = _task_dir(output_dir, task_name)
        task_out.mkdir(parents=True, exist_ok=True)
        log_path = _task_log(output_dir, task_name)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        cmd = _build_worker_cmd(
            task_name, eval_config_path, output_dir, enable_recording
        )
        print(
            f"[dispatch] task={task_name} gpu={gpu} -> {log_path}",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"# cmd: {' '.join(cmd)}\n")
            log.write(f"# CUDA_VISIBLE_DEVICES={gpu}\n\n")
            log.flush()
            rc = subprocess.run(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        err = None if rc == 0 else f"subprocess returncode={rc}"
        print(f"[done]     task={task_name} gpu={gpu} rc={rc}", flush=True)
        return task_name, rc, err
    finally:
        gpu_q.put(gpu)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _stage_success_rate(
    episodes: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate `criteria_reached` across episodes into per-stage rates."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for ep in episodes:
        reached = (ep.get("metrics") or {}).get("criteria_reached") or {}
        if not isinstance(reached, dict):
            continue
        for name, ok in reached.items():
            counts[name][1] += 1
            if bool(ok):
                counts[name][0] += 1
    return {name: r / t if t else 0.0 for name, (r, t) in counts.items()}


def _success_count(episodes: list[Any]) -> int:
    return sum(bool(episode.success) for episode in episodes)


def _config_summary_payload(task_result: Any) -> dict[str, Any]:
    """Return config-level completion metadata for a task result."""
    result = task_result.result
    episodes = result.episode_results if result is not None else []
    return {
        **dict(task_result.user_data),
        "config_path": task_result.config_path,
        "error": task_result.error,
        "eval_result_json": task_result.result_json_path,
        "seed": result.seed_start if result is not None else None,
        "episode_num": len(episodes),
        "success_count": _success_count(episodes),
        "success_rate": result.success_rate if result is not None else 0.0,
        "average_progress": (
            result.average_progress if result is not None else 0.0
        ),
        "total": len(episodes),
    }


def _write_task_eval_summary(
    *,
    task_out: Path,
    task: TaskCfg,
    plan_groups: int,
    configs: list[dict[str, Any]],
) -> None:
    """Persist all config outcomes so parent aggregation sees failures."""
    failed = [cfg for cfg in configs if cfg.get("error")]
    payload = {
        "task_name": task.name,
        "task_type": task.task_type,
        "groups": plan_groups,
        "configs_total": len(configs),
        "configs_failed": len(failed),
        "configs_succeeded": len(configs) - len(failed),
        "configs": configs,
    }
    (task_out / "task_eval_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_task_eval_summary(task_out: Path) -> dict[str, Any] | None:
    summary_path = task_out / "task_eval_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _load_config_eval_result(
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    result_json = config.get("eval_result_json")
    if not result_json:
        return None, "missing eval_result_json"
    path = Path(result_json)
    if not path.exists():
        return None, f"missing eval_result.json: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"failed to load eval_result.json: {exc}"


def _summarize_config_results(
    configs: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
]:
    episodes: list[dict[str, Any]] = []
    per_config: list[dict[str, Any]] = []
    failed_configs: list[dict[str, Any]] = []
    skipped = 0

    for config in configs:
        config_record = dict(config)
        if config.get("error"):
            failed_configs.append(config_record)
            per_config.append(config_record)
            continue

        data, load_error = _load_config_eval_result(config)
        if load_error is not None:
            config_record["error"] = load_error
            failed_configs.append(config_record)
            per_config.append(config_record)
            continue

        assert data is not None
        eps = data.get("episode_results") or []
        episodes.extend(eps)
        skipped += len(data.get("skipped_episodes") or [])
        config_record.update(
            {
                "episodes": len(eps),
                "success_rate": data.get("success_rate", 0.0),
                "avg_progress": data.get("average_progress", 0.0),
            }
        )
        per_config.append(config_record)

    return episodes, per_config, failed_configs, skipped


def _summarize_task(
    task_out: Path, rc: int, err: str | None
) -> dict[str, Any]:
    """Merge per-config metrics while preserving config-level failures."""
    if rc != 0:
        return {"status": "error", "error": err or f"returncode={rc}"}

    task_summary = _load_task_eval_summary(task_out)
    if task_summary is None:
        return {
            "status": "error",
            "error": f"missing task_eval_summary.json under {task_out}",
        }

    configs = task_summary.get("configs") or []
    episodes, per_config, failed_configs, skipped = _summarize_config_results(
        configs
    )

    total = len(episodes)
    success = sum(1 for ep in episodes if ep.get("success"))
    progress = sum(float(ep.get("progress", 0.0)) for ep in episodes)
    configs_failed = len(failed_configs)
    configs_total = len(configs)
    if configs_total == 0:
        return {
            "status": "error",
            "error": f"task_eval_summary.json has no configs under {task_out}",
        }
    configs_succeeded = configs_total - configs_failed
    if configs_failed:
        status = "partial_error" if configs_succeeded else "error"
    else:
        status = "ok"
    return {
        "status": status,
        "episodes": total,
        "success_rate": success / total if total else 0.0,
        "avg_progress": progress / total if total else 0.0,
        "stage_success_rate": _stage_success_rate(episodes),
        "skipped_episodes": skipped,
        "configs": configs_total,
        "configs_total": configs_total,
        "configs_succeeded": configs_succeeded,
        "configs_failed": configs_failed,
        "failed_configs": failed_configs,
        "per_config": per_config,
    }


def _write_summary(
    output_dir: Path,
    eval_cfg: EvalConfig,
    per_task: dict[str, dict[str, Any]],
    started_at: float,
) -> Path:
    task_names = [t.name for t in eval_cfg.tasks]
    ok = [t for t in task_names if per_task[t].get("status") == "ok"]
    partial = [
        t for t in task_names if per_task[t].get("status") == "partial_error"
    ]
    metric_tasks = [
        t for t in task_names if per_task[t].get("episodes", 0) > 0
    ]
    total_eps = sum(per_task[t]["episodes"] for t in metric_tasks)
    configs_total = sum(
        per_task[t].get("configs_total", 0) for t in task_names
    )
    configs_failed = sum(
        per_task[t].get("configs_failed", 0) for t in task_names
    )

    def _weighted(field: str) -> float:
        if not total_eps:
            return 0.0
        return (
            sum(
                per_task[t][field] * per_task[t]["episodes"]
                for t in metric_tasks
            )
            / total_eps
        )

    status = "ok" if len(ok) == len(task_names) else "partial_error"
    if any(per_task[t].get("status") == "error" for t in task_names):
        status = "error"

    summary = {
        "eval_config": eval_cfg.source_path,
        "policy": {
            "model_type": eval_cfg.policy.model_type,
            "model_yaml": str(_resolve_model_yaml(eval_cfg.policy)),
        },
        "tasks": {t: per_task[t] for t in task_names},
        "overall": {
            "status": status,
            "ok_tasks": len(ok),
            "partial_error_tasks": len(partial),
            "total_tasks": len(task_names),
            "total_episodes": total_eps,
            "success_rate": _weighted("success_rate"),
            "avg_progress": _weighted("avg_progress"),
            "configs_total": configs_total,
            "configs_failed": configs_failed,
            "configs_succeeded": configs_total - configs_failed,
        },
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def _print_console_summary(
    eval_cfg: EvalConfig,
    per_task: dict[str, dict[str, Any]],
    summary_path: Path,
) -> None:
    print(f"\n=== Summary written: {summary_path} ===", flush=True)
    for t in eval_cfg.tasks:
        s = per_task[t.name]
        if s["status"] == "error":
            print(f"  [error] {t.name}: {s.get('error')}", flush=True)
            continue
        label = "partial" if s["status"] == "partial_error" else "ok"
        stages = ", ".join(
            f"{k}={v:.2f}" for k, v in s["stage_success_rate"].items()
        )
        print(
            f"  [{label}] {t.name}: success_rate={s['success_rate']:.4f} "
            f"avg_progress={s['avg_progress']:.4f} "
            f"episodes={s['episodes']} "
            f"configs={s['configs_succeeded']}/{s['configs_total']} "
            f"failed_configs={s['configs_failed']} stages=[{stages}]",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# Dispatch entry
# --------------------------------------------------------------------------- #


def _dispatch(eval_cfg: EvalConfig, args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpus = _resolve_gpus(args.gpus)

    task_names = [t.name for t in eval_cfg.tasks]
    started_at = time.time()

    gpu_q: Queue = Queue()
    for g in gpus:
        gpu_q.put(g)

    results: dict[str, tuple[int, str | None]] = {}
    with ThreadPoolExecutor(
        max_workers=min(len(task_names), len(gpus))
    ) as pool:
        futures = [
            pool.submit(
                _run_worker,
                name,
                eval_cfg.source_path,
                output_dir,
                args.enable_recording,
                gpu_q,
            )
            for name in task_names
        ]
        for fut in futures:
            task, rc, err = fut.result()
            results[task] = (rc, err)

    per_task = {
        name: _summarize_task(_task_dir(output_dir, name), *results[name])
        for name in task_names
    }
    summary_path = _write_summary(
        output_dir=output_dir,
        eval_cfg=eval_cfg,
        per_task=per_task,
        started_at=started_at,
    )
    _print_console_summary(eval_cfg, per_task, summary_path)

    has_worker_error = any(rc != 0 for rc, _ in results.values())
    has_config_error = any(
        task.get("status") != "ok" for task in per_task.values()
    )
    return 1 if has_worker_error or has_config_error else 0


def main() -> None:
    args = _build_parser().parse_args()
    eval_cfg = load_eval_config(args.eval_config)
    if args._single_task is not None:
        sys.exit(
            _run_single_task(
                eval_cfg,
                args._single_task,
                Path(args.output_dir).resolve(),
                args.enable_recording,
            )
        )
    sys.exit(_dispatch(eval_cfg, args))


if __name__ == "__main__":
    main()
