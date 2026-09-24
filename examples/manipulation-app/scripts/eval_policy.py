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
`episode_num`. Every config is compiled into one or more deterministic
episode shards and dispatched through the same GPU worker scheduler.
Execution can be serial or parallel on one or more GPUs. Results are
aggregated into `<output_dir>/summary.json`.
Sharding is automatic; individual tasks may explicitly set `shards`.

Run:
    PYTHONPATH=$PWD python3 \\
        examples/manipulation-app/scripts/eval_policy.py \\
        --eval-config examples/manipulation-app/configs/eval_example.yaml \\
        --output-dir eval_result/run_001 \\
        --gpus 0,1,2,3

CLI flags:
    --eval-config / --output-dir / --gpus / --enable-recording
    --export-video
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import threading
import sys
import tempfile
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robo_orchard_sim.pipeline.evaluator.base import (
    EpisodeResult,
    EvaluationResult,
    EvaluationSummary,
    ShardSummary,
    SkippedEpisode,
    TaskSummary,
)
from robo_orchard_sim.task_components.selector import SelectorName


_POLICY_CONFIG_DIR = _REPO_ROOT / "robo_orchard_sim" / "policy" / "configs"
_BENCHMARK_ROOT = _REPO_ROOT / "robo_orchard_sim" / "benchmark"
_SHARD_MANIFEST_FLAG = "--_shard-manifest"
_SHARD_ID_FLAG = "--_shard-id"


# --------------------------------------------------------------------------- #
# Config schema (YAML <-> dataclasses)
# --------------------------------------------------------------------------- #


_TASK_FIELDS = {
    "task_type",
    "yaml",
    "batch_plan",
    "splits",
    "split_type",
    "snapshot",
    "episode_num",
    "shards",
    "swap",
    "max_steps",
    "seed",
    "asset_root",
}
_DEFAULTS_FIELDS = _TASK_FIELDS - {"task_type", "yaml", "batch_plan", "shards"}
_EXECUTION_FIELDS = {"gpus"}
_TOP_LEVEL_FIELDS = {
    "policy",
    "execution",
    "defaults",
    "tasks",
    "evaluation_id",
    "team_id",
}
_VALID_SPLIT_TYPES = ("seen", "unseen_instance", "unseen_category")


@dataclass(frozen=True)
class PolicyCfg:
    model_type: str
    model_yaml: str | None = None


@dataclass(frozen=True)
class ExecutionCfg:
    """Default local execution resources; CLI options may override them."""

    gpus: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SwapCfg:
    """Whether to rotate the task's target between episodes.

    The dispatcher stores only settings; execution.py builds SwapConfig
    inside the worker process.
    """

    enabled: bool = False
    swap_per_scene: int = 1
    """Episodes one scene serves. Only read when swap is enabled."""
    selector: SelectorName = "random"

    def to_evaluator_config(self, swap_config_cls: Any) -> Any:
        """Hand these settings to the evaluator's own config type."""
        return swap_config_cls(
            enabled=self.enabled,
            swap_per_scene=self.swap_per_scene,
            selector=self.selector,
        )


def _parse_swap(name: str, raw: Any) -> SwapCfg:
    """Read a task entry's optional `swap` block."""
    if raw is None:
        return SwapCfg()
    if not isinstance(raw, dict):
        raise SystemExit(
            f"task {name!r}: `swap` must be a mapping with `enabled`, "
            f"`selector` and `swap_per_scene`, got {type(raw).__name__}"
        )
    unknown = set(raw) - {"enabled", "selector", "swap_per_scene"}
    if unknown:
        raise SystemExit(
            f"task {name!r}: unknown key(s) under `swap`: {sorted(unknown)}"
        )
    defaults = SwapCfg()
    selector = raw.get("selector", defaults.selector)
    if selector not in ("round_robin", "random"):
        raise SystemExit(
            f"task {name!r}: `swap.selector` must be 'round_robin' or "
            f"'random', got {selector!r}"
        )
    enabled = bool(raw.get("enabled", defaults.enabled))
    per_scene = int(raw.get("swap_per_scene", defaults.swap_per_scene))
    if per_scene < 1:
        raise SystemExit(
            f"task {name!r}: `swap.swap_per_scene` must be >= 1, "
            f"got {per_scene}"
        )
    return SwapCfg(
        enabled=enabled, swap_per_scene=per_scene, selector=selector
    )


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
    shards: int | None = None
    swap: SwapCfg = field(default_factory=lambda: SwapCfg())
    max_steps: int = 1000
    splits: str | None = None
    split_type: str | None = None
    snapshot: str | None = None
    # Mutually exclusive: yaml is a single task config; batch_plan is a
    # path to an existing batch plan JSON.
    yaml: str | None = None
    batch_plan: str | None = None


@dataclass(frozen=True)
class EvalConfig:
    policy: PolicyCfg
    execution: ExecutionCfg
    tasks: list[TaskCfg]
    source_path: str = ""
    evaluation_id: str = ""
    team_id: str = ""

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
    execution = _parse_execution(raw.get("execution") or {}, src)
    defaults = _parse_defaults(raw.get("defaults") or {}, src)

    tasks_raw = _require(raw, "tasks", src, dict)
    if not tasks_raw:
        raise ValueError(f"eval config has no tasks: {src}")
    tasks = [
        _parse_task(name=name, raw=spec or {}, defaults=defaults, src=src)
        for name, spec in tasks_raw.items()
    ]

    identifiers = {
        key: _require(raw, key, src, str) if key in raw else ""
        for key in ("evaluation_id", "team_id")
    }
    return EvalConfig(
        policy=policy,
        execution=execution,
        tasks=tasks,
        source_path=str(src),
        **identifiers,
    )


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


def _parse_execution(raw: dict, src: Path) -> ExecutionCfg:
    unknown = set(raw) - _EXECUTION_FIELDS
    if unknown:
        raise ValueError(
            f"unknown execution fields {sorted(unknown)} in {src}; "
            f"allowed: {sorted(_EXECUTION_FIELDS)}"
        )
    raw_gpus = raw.get("gpus")
    if raw_gpus is None:
        gpus = None
    elif isinstance(raw_gpus, (str, int)):
        gpus = tuple(
            gpu.strip() for gpu in str(raw_gpus).split(",") if gpu.strip()
        )
    elif isinstance(raw_gpus, list):
        gpus = tuple(str(gpu) for gpu in raw_gpus)
    else:
        raise ValueError(f"execution.gpus must be a list or string: {src}")
    if gpus == ():
        raise ValueError(f"execution.gpus must not be empty: {src}")

    return ExecutionCfg(gpus=gpus)


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
    if isinstance(defaults.get("swap"), dict) and isinstance(
        raw.get("swap"), dict
    ):
        merged["swap"] = {**defaults["swap"], **raw["swap"]}

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
    split_type = merged.get("split_type")
    if split_type is not None:
        if split_type not in _VALID_SPLIT_TYPES:
            raise ValueError(
                f"task {name!r}: `split_type` must be one of "
                f"{list(_VALID_SPLIT_TYPES)}, got {split_type!r} ({src})"
            )
        if merged.get("batch_plan"):
            raise ValueError(
                f"task {name!r}: `split_type` is incompatible with "
                f"`batch_plan`; the batch plan already fixes the per-config "
                f"splits ({src})"
            )
    if "asset_root" not in merged or not merged["asset_root"]:
        raise ValueError(
            f"task {name!r}: `asset_root` must be set in defaults or task "
            f"({src})"
        )
    shards = merged.get("shards")
    if shards is not None:
        shards = int(shards)
        if shards < 1:
            raise ValueError(
                f"task {name!r}: `shards` must be >= 1, got {shards} ({src})"
            )

    return TaskCfg(
        name=name,
        task_type=str(raw["task_type"]),
        asset_root=str(merged["asset_root"]),
        seed=int(merged.get("seed", 0)),
        episode_num=int(merged.get("episode_num", 20)),
        shards=shards,
        swap=_parse_swap(name, merged.get("swap")),
        max_steps=int(merged.get("max_steps", 1000)),
        splits=_opt_str(merged.get("splits")),
        split_type=_opt_str(merged.get("split_type")),
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
    p.add_argument(
        "--export-video",
        action="store_true",
        help="Export MP4s and backfill summary after evaluation; "
        "requires --enable-recording.",
    )
    # Internal: marks the worker subprocess.
    p.add_argument(
        _SHARD_MANIFEST_FLAG,
        dest="_shard_manifest",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        _SHARD_ID_FLAG,
        dest="_shard_id",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    return p


def _resolve_gpus(
    cli_arg: str | None,
    configured: tuple[str, ...] | None = None,
) -> list[str]:
    if cli_arg:
        gpus = [g.strip() for g in cli_arg.split(",") if g.strip()]
    elif configured is not None:
        gpus = list(configured)
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


def _rewrite_split(yaml_path: str, split_type: str, dest: Path) -> str:
    """Load `yaml_path`, override every asset_configs[*].split, save to dest.

    Only slots that already declare a `split` key get overridden — slots
    without an explicit `split` are left alone (adding one would silently
    change their semantics). Raises if no slot ends up touched.
    Instruction fields (e.g. actor_description_mode) are intentionally
    left untouched — they control language, not the dataset split.

    The rewritten yaml is what actually gets run; the shared execution layer
    copies it into the config's output dir, so no extra durable copy is needed
    here — `dest` is expected to be a scratch path.
    """
    src = Path(yaml_path)
    loaded = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"task yaml must be a mapping: {src}")
    asset_configs = loaded.get("asset_configs")
    if not isinstance(asset_configs, dict) or not asset_configs:
        raise SystemExit(
            f"cannot apply split_type={split_type!r}: {src} has no "
            f"`asset_configs` section to rewrite"
        )
    touched = 0
    for key, slot in asset_configs.items():
        if not isinstance(slot, dict):
            raise SystemExit(
                f"asset_configs[{key!r}] must be a mapping in {src}"
            )
        if "split" not in slot:
            continue
        slot["split"] = split_type
        touched += 1
    if touched == 0:
        raise SystemExit(
            f"cannot apply split_type={split_type!r}: no slot under "
            f"`asset_configs` in {src} declares a `split` field. "
            "Add `split: <value>` to at least one slot, or drop "
            "`split_type` from the eval config."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        yaml.safe_dump(loaded, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return str(dest)


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
    if task.split_type is not None:
        # Batch evaluation copies the used yaml into each config's output
        # dir, so a scratch tmpdir is enough — keep the original filename
        # so downstream artifacts look the same as an unmodified run.
        scratch = Path(tempfile.mkdtemp(prefix=f"{task.name}_split_"))
        yaml_path = _rewrite_split(
            yaml_path,
            task.split_type,
            scratch / Path(yaml_path).name,
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


def _print_worker_error_log(
    output_dir: Path,
    task_name: str,
    reason: str,
    *,
    scan_lines: int = 400,
    fallback_lines: int = 80,
) -> None:
    """Print only failed-worker logs; successful workers stay file-only."""
    log_path = _task_log(output_dir, task_name)
    print(
        f"\n--- worker error: task={task_name}: {reason} ---",
        file=sys.stderr,
        flush=True,
    )
    if not log_path.exists():
        print(f"worker log not found: {log_path}", file=sys.stderr, flush=True)
        return
    with log_path.open("r", encoding="utf-8", errors="replace") as log:
        tail = list(deque(log, maxlen=scan_lines))
    traceback_starts = [
        i
        for i, line in enumerate(tail)
        if line.startswith("Traceback (most recent call last):")
    ]
    excerpt = (
        tail[traceback_starts[-1] :]
        if traceback_starts
        else tail[-fallback_lines:]
    )
    print(
        f"--- error excerpt from {log_path} ---",
        file=sys.stderr,
        flush=True,
    )
    for line in excerpt:
        print(line, end="", file=sys.stderr)
    print("--- end worker error log ---\n", file=sys.stderr, flush=True)


def _resolve_model_yaml(policy: PolicyCfg) -> Path:
    if policy.model_yaml:
        return Path(policy.model_yaml)
    return _POLICY_CONFIG_DIR / f"{policy.model_type}.yaml"


# --------------------------------------------------------------------------- #
# Shared shard worker
# --------------------------------------------------------------------------- #


def _load_model_cfg(policy: PolicyCfg) -> dict:
    path = _resolve_model_yaml(policy)
    if not path.exists():
        raise FileNotFoundError(f"Policy config yaml not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Model yaml must contain a mapping: {path}")
    loaded = _expand_env(loaded, src=path)
    declared = loaded.get("policy")
    if declared is not None and declared != policy.model_type:
        raise ValueError(
            f"Policy type mismatch: model_type={policy.model_type}, "
            f"but yaml declares policy={declared}"
        )
    cfg = dict(loaded)
    cfg["policy"] = policy.model_type
    return cfg


def _stop_process(process: subprocess.Popen) -> None:
    """Stop and reap a process group created by this dispatcher."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _shard_dir(output_dir: Path, task_name: str, shard_id: str) -> Path:
    return _task_dir(output_dir, task_name) / "shards" / shard_id


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    from robo_orchard_sim.pipeline.evaluator.execution import write_json_atomic

    write_json_atomic(path, payload)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #


def _build_shard_worker_cmd(
    *,
    eval_config_path: str,
    output_dir: Path,
    manifest_path: Path,
    shard_id: str,
    enable_recording: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--eval-config",
        eval_config_path,
        "--output-dir",
        str(output_dir),
        _SHARD_MANIFEST_FLAG,
        str(manifest_path),
        _SHARD_ID_FLAG,
        shard_id,
    ]
    if enable_recording:
        cmd.append("--enable-recording")
    return cmd


def _run_shard_worker(
    *,
    task_name: str,
    shard_id: str,
    manifest_path: Path,
    eval_config_path: str,
    output_dir: Path,
    enable_recording: bool,
    gpu_q: Queue,
    cancelled: threading.Event,
) -> tuple[str, str, int, str | None]:
    """Run one shard subprocess on the next available GPU worker slot."""
    gpu = gpu_q.get()
    try:
        shard_out = _shard_dir(output_dir, task_name, shard_id)
        shard_out.mkdir(parents=True, exist_ok=True)
        log_path = shard_out / "stdout.log"
        summary_path = shard_out / "shard_summary.json"
        summary_path.unlink(missing_ok=True)
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
        }
        cmd = _build_shard_worker_cmd(
            eval_config_path=eval_config_path,
            output_dir=output_dir,
            manifest_path=manifest_path,
            shard_id=shard_id,
            enable_recording=enable_recording,
        )
        print(
            f"[dispatch] task={task_name} shard={shard_id} "
            f"gpu={gpu} -> {log_path}",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"# cmd: {' '.join(cmd)}\n")
            log.write(f"# CUDA_VISIBLE_DEVICES={gpu}\n\n")
            log.flush()
            if cancelled.is_set():
                return task_name, shard_id, 1, "evaluation cancelled"
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                while True:
                    if cancelled.is_set():
                        return task_name, shard_id, 1, "evaluation cancelled"
                    try:
                        return_code = process.wait(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                _stop_process(process)
        if return_code != 0:
            error = f"subprocess returncode={return_code}"
        elif not summary_path.exists():
            return_code = 1
            error = "worker exited without shard_summary.json"
        else:
            error = None
        print(
            f"[done] task={task_name} shard={shard_id} "
            f"gpu={gpu} rc={return_code}",
            flush=True,
        )
        return task_name, shard_id, return_code, error
    finally:
        gpu_q.put(gpu)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _parse_episode_result(raw: dict[str, Any]) -> EpisodeResult:
    """Validate a worker episode before it reaches public summaries."""
    seed = raw["seed"]
    success = raw["success"]
    steps = raw.get("steps", 0)
    stop_reason = raw.get("stop_reason", "")
    instruction = raw.get("instruction", "")
    stage_scores = raw.get("stage_scores", {})
    metrics = raw.get("metrics") or {}
    if type(seed) is not int or type(steps) is not int or steps < 0:
        raise ValueError("episode seed/steps must be integers; steps >= 0")
    if type(success) is not bool or not isinstance(stop_reason, str):
        raise ValueError("episode success/stop_reason have invalid types")
    if not isinstance(instruction, str) or not isinstance(stage_scores, dict):
        raise ValueError("episode instruction/stage_scores have invalid types")
    if not isinstance(metrics, dict):
        raise ValueError("episode metrics must be a mapping")
    if any(not isinstance(name, str) for name in stage_scores):
        raise ValueError("stage names must be strings")
    return EpisodeResult(
        seed=seed,
        success=success,
        progress=_score(raw["progress"]),
        steps=steps,
        stop_reason=stop_reason,
        metrics=metrics,
        checker_summary=raw.get("checker_summary"),
        instruction=instruction,
        stage_scores={
            name: _score(value) for name, value in stage_scores.items()
        },
    )


def _read_shard_summary(
    output_dir: Path,
    shard: Any,
    worker_error: str | None,
) -> ShardSummary:
    from robo_orchard_sim.pipeline.evaluator.distributed import (
        validate_shard_result,
    )

    outcome = ShardSummary(shard=shard)
    try:
        if worker_error:
            raise ValueError(worker_error)
        path = _shard_dir(output_dir, shard.task_name, shard.shard_id)
        summary = json.loads((path / "shard_summary.json").read_text())
        if not isinstance(summary, dict):
            raise ValueError("shard summary must be a JSON object")
        if summary.get("shard") != shard.to_payload():
            raise ValueError("manifest metadata mismatch")
        configs = summary.get("configs") or []
        if len(configs) != 1:
            raise ValueError("expected one config result")
        config = configs[0]
        if not isinstance(config, dict):
            raise ValueError("config result must be a JSON object")
        if config.get("error"):
            raise ValueError(config["error"])
        outcome.result_json_path = config.get("eval_result_json")
        if not outcome.result_json_path:
            raise ValueError("missing eval_result.json")
        data = json.loads(Path(outcome.result_json_path).read_text())
        validate_shard_result(shard, data)
        episodes = [
            _parse_episode_result(ep) for ep in data["episode_results"]
        ]
        outcome.result = EvaluationResult(
            episode_num=len(episodes),
            seed_start=shard.seed_start,
            success_rate=sum(ep.success for ep in episodes) / len(episodes)
            if episodes
            else 0.0,
            average_progress=sum(ep.progress for ep in episodes)
            / len(episodes)
            if episodes
            else 0.0,
            episode_results=episodes,
            skipped_episodes=[
                SkippedEpisode(**ep) for ep in data.get("skipped_episodes", [])
            ],
            attempted_scene_seeds=data.get("attempted_scene_seeds", []),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        outcome.error = f"{shard.shard_id}: {exc}"
    return outcome


def _merge_task_shard_summaries(
    *,
    output_dir: Path,
    task: TaskCfg,
    shards: list[Any],
    shard_errors: dict[str, str] | None = None,
) -> TaskSummary:
    """Load validated worker outcomes and persist the public task summary."""
    summary = TaskSummary(
        task.name,
        [
            _read_shard_summary(
                output_dir, shard, (shard_errors or {}).get(shard.shard_id)
            )
            for shard in shards
        ],
    )
    # Keep merged per-config result files for existing result consumers.
    configs: dict[tuple[str, int], list[ShardSummary]] = defaultdict(list)
    for item in summary.shards:
        configs[(item.shard.group_id, item.shard.config_index)].append(item)
    config_payloads = []
    for (group, index), parts in configs.items():
        result = TaskSummary(task.name, parts)
        metrics = result.metrics
        completed = [s.result for s in parts if s.result is not None]
        combined = EvaluationResult(
            episode_num=metrics.episodes,
            seed_start=parts[0].shard.seed_start
            - parts[0].shard.episode_offset
            // parts[0].shard.episodes_per_scene,
            success_rate=metrics.success_rate,
            average_progress=metrics.avg_progress,
            episode_results=[
                ep for r in completed for ep in r.episode_results
            ],
            skipped_episodes=[
                ep for r in completed for ep in r.skipped_episodes
            ],
            attempted_scene_seeds=[
                seed for r in completed for seed in r.attempted_scene_seeds
            ],
        )
        result_path = (
            _task_dir(output_dir, task.name)
            / group
            / f"config_{index:04d}"
            / "eval_result.json"
        )
        _write_json_atomic(result_path, asdict(combined))
        config_payloads.append(
            {
                "group_id": group,
                "config_index": index,
                "config_path": parts[0].shard.config_path,
                "error": result.to_payload()["error"],
                "eval_result_json": str(result_path),
                "seed": combined.seed_start,
                "episode_num": metrics.episodes,
                "success_count": metrics.successes,
                "success_rate": metrics.success_rate,
                "average_progress": metrics.avg_progress,
                "total": metrics.episodes,
            }
        )
    _write_json_atomic(
        _task_dir(output_dir, task.name) / "task_eval_summary.json",
        {
            **summary.to_payload(),
            "task_type": task.task_type,
            "groups": len({shard.group_id for shard in shards}),
            "configs": config_payloads,
        },
    )
    return summary


def _score(value: Any) -> float:
    if type(value) not in (int, float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"expected a finite score in [0, 1], got {value!r}")
    return float(value)


def _preview_episode(episode: EpisodeResult, index: int) -> dict[str, Any]:
    """Project an episode onto the internal leaderboard contract."""
    failed = episode.stop_reason.startswith("episode_error:")
    score = _score(episode.progress)
    stage_scores = {
        name: _score(value) for name, value in episode.stage_scores.items()
    }
    return {
        "episode_index": index,
        "seed": episode.seed,
        "status": "failed" if failed else "succeeded",
        "success": False if failed else episode.success,
        "score": 0.0 if failed else score,
        "steps": episode.steps,
        "stop_reason": episode.stop_reason,
        "instruction": episode.instruction,
        "stage_scores": (
            {name: 0.0 for name in stage_scores} if failed else stage_scores
        ),
        "video": "",
    }


def _leaderboard_task_result(task: TaskSummary) -> dict[str, Any]:
    """Count every planned episode, including missing shard results."""
    total = sum(shard.shard.episode_count for shard in task.shards)
    episodes = [
        _preview_episode(episode, index)
        for index, episode in enumerate(
            episode
            for shard in task.shards
            if shard.result is not None
            for episode in shard.result.episode_results
        )
    ]
    failed = total - sum(ep["status"] == "succeeded" for ep in episodes)
    return {
        "status": "failed" if failed or task.status != "ok" else "succeeded",
        "episode_count": total,
        "failed_episode_count": failed,
        "score": (
            math.fsum(ep["score"] for ep in episodes) / total if total else 0.0
        ),
        "success_rate": (
            sum(ep["success"] for ep in episodes) / len(episodes)
            if episodes
            else 0.0
        ),
        "stage_success_rate": task.metrics.stage_success_rate,
        "preview_episodes": episodes[:10],
    }


def _format_summary_json(data: dict[str, Any], level: int = 0) -> str:
    """Indent leaderboard objects and keep preview episodes on one line."""
    if not data:
        return "{}"
    indent = "  " * level
    fields = []
    for key, value in data.items():
        if isinstance(value, dict):
            rendered = _format_summary_json(value, level + 1)
        elif key == "preview_episodes" and value:
            episodes = ",\n".join(
                f"{indent}    {json.dumps(episode, allow_nan=False)}"
                for episode in value
            )
            rendered = f"[\n{episodes}\n{indent}  ]"
        else:
            rendered = json.dumps(value, allow_nan=False)
        fields.append(f"{indent}  {json.dumps(key)}: {rendered}")
    return "{\n" + ",\n".join(fields) + f"\n{indent}}}"


def _write_summary(
    output_dir: Path,
    eval_cfg: EvalConfig,
    per_task: dict[str, TaskSummary],
    started_at: float,
    execution: Any,
) -> Path:
    distributed = EvaluationSummary(list(per_task.values()))
    _write_json_atomic(
        output_dir / "distributed_summary.json",
        {
            "eval_config": eval_cfg.source_path,
            "policy": {
                "model_type": eval_cfg.policy.model_type,
                "model_yaml": str(_resolve_model_yaml(eval_cfg.policy)),
            },
            "execution": {"gpus": list(execution.gpus)},
            **distributed.to_payload(),
            "elapsed_seconds": round(time.time() - started_at, 2),
        },
    )
    task_results = {
        task.name: _leaderboard_task_result(per_task[task.name])
        for task in eval_cfg.tasks
    }
    task_fields = (
        "episode_count",
        "failed_episode_count",
        "score",
        "success_rate",
        "stage_success_rate",
        "preview_episodes",
    )
    payload = {
        "evaluation_id": eval_cfg.evaluation_id,
        "team_id": eval_cfg.team_id,
        "status": (
            "succeeded"
            if all(
                task["status"] == "succeeded" for task in task_results.values()
            )
            else "failed"
        ),
        "task_results": {
            name: {key: task[key] for key in task_fields}
            for name, task in task_results.items()
        },
        "overall_score": (
            math.fsum(task["score"] for task in task_results.values())
            / len(task_results)
        ),
    }
    path = output_dir / "summary.json"
    path.write_text(_format_summary_json(payload) + "\n", encoding="utf-8")
    return path


def _print_console_summary(
    eval_cfg: EvalConfig,
    per_task: dict[str, TaskSummary],
    summary_path: Path,
) -> None:
    print(f"\n=== Summary written: {summary_path} ===", flush=True)
    for t in eval_cfg.tasks:
        result = _leaderboard_task_result(per_task[t.name])
        print(
            f"  [{result['status']}] {t.name}: "
            f"score={result['score']:.4f} "
            f"episodes={result['episode_count']} "
            f"failed_episodes={result['failed_episode_count']}",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# Dispatch entry
# --------------------------------------------------------------------------- #


def _dispatch_work_items(
    eval_cfg: EvalConfig,
    *,
    output_dir: Path,
    gpus: list[str],
    started_at: float,
    enable_recording: bool,
) -> int:
    from robo_orchard_sim.pipeline.evaluator.distributed import (
        EvaluationExecutionConfig,
        EvaluationManifest,
        build_evaluation_shards,
        write_evaluation_manifest,
    )

    from robo_orchard_sim.pipeline.evaluator.distributed.scheduling import (
        EvaluationWorkload,
        allocate_task_shards,
    )

    execution = EvaluationExecutionConfig(gpus=tuple(gpus))
    plans = [_build_plan(task) for task in eval_cfg.tasks]
    shard_counts = allocate_task_shards(
        [
            EvaluationWorkload(
                episodes_per_config=plan.episodes_per_config,
                max_steps=task.max_steps,
                config_count=sum(len(group.configs) for group in plan.groups),
                episodes_per_scene=(
                    task.swap.swap_per_scene if task.swap.enabled else 1
                ),
                shards=task.shards,
            )
            for task, plan in zip(eval_cfg.tasks, plans, strict=True)
        ],
        len(gpus),
    )
    task_manifests = {}
    jobs = []
    job_costs = {}
    for task, plan, shards_per_config in zip(
        eval_cfg.tasks, plans, shard_counts, strict=True
    ):
        shards = build_evaluation_shards(
            plan=plan,
            task_name=task.name,
            shards_per_config=shards_per_config,
            episodes_per_scene=(
                task.swap.swap_per_scene if task.swap.enabled else 1
            ),
        )
        manifest = EvaluationManifest(
            run_id=output_dir.name,
            policy_id=eval_cfg.policy.model_type,
            shards=tuple(shards),
        )
        manifest_path = write_evaluation_manifest(
            manifest,
            _task_dir(output_dir, task.name) / "evaluation_manifest.json",
        )
        task_manifests[task.name] = (manifest, manifest_path)
        jobs.extend(
            (task.name, shard.shard_id, manifest_path)
            for shard in manifest.shards
        )

        for shard in manifest.shards:
            job_costs[(task.name, shard.shard_id)] = (
                shard.episode_count * task.max_steps
            )

    # Longest estimated processing time first; ties preserve input order.
    jobs.sort(key=lambda job: job_costs[(job[0], job[1])], reverse=True)
    gpu_q: Queue = Queue()
    for gpu in execution.gpu_slots():
        gpu_q.put(gpu)

    worker_results = {}
    max_workers = execution.concurrency(len(jobs))
    print(
        f"[plan] work_items={len(jobs)} "
        f"gpus={list(execution.gpus)} "
        f"concurrency={max_workers}",
        flush=True,
    )
    cancelled = threading.Event()
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = [
            pool.submit(
                _run_shard_worker,
                task_name=task_name,
                shard_id=shard_id,
                manifest_path=manifest_path,
                eval_config_path=eval_cfg.source_path,
                output_dir=output_dir,
                enable_recording=enable_recording,
                gpu_q=gpu_q,
                cancelled=cancelled,
            )
            for task_name, shard_id, manifest_path in jobs
        ]
        for future in futures:
            task_name, shard_id, return_code, error = future.result()
            worker_results[(task_name, shard_id)] = (return_code, error)
    finally:
        cancelled.set()
        pool.shutdown(wait=True, cancel_futures=True)

    per_task = {}
    for task in eval_cfg.tasks:
        manifest, _ = task_manifests[task.name]
        shard_errors = {
            shard.shard_id: error or f"returncode={return_code}"
            for shard in manifest.shards
            for return_code, error in [
                worker_results[(task.name, shard.shard_id)]
            ]
            if return_code != 0
        }
        per_task[task.name] = _merge_task_shard_summaries(
            output_dir=output_dir,
            task=task,
            shards=list(manifest.shards),
            shard_errors=shard_errors,
        )
    for task_summary in per_task.values():
        for shard_summary in task_summary.shards:
            if shard_summary.status != "ok":
                _print_worker_error_log(
                    output_dir,
                    f"{task_summary.task_name}/shards/"
                    f"{shard_summary.shard.shard_id}",
                    shard_summary.error or shard_summary.status,
                )
    summary_path = _write_summary(
        output_dir=output_dir,
        eval_cfg=eval_cfg,
        per_task=per_task,
        started_at=started_at,
        execution=execution,
    )
    _print_console_summary(eval_cfg, per_task, summary_path)
    return int(
        any(task_summary.status != "ok" for task_summary in per_task.values())
    )


def _dispatch(eval_cfg: EvalConfig, args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpus = _resolve_gpus(args.gpus, eval_cfg.execution.gpus)
    started_at = time.time()
    return _dispatch_work_items(
        eval_cfg=eval_cfg,
        output_dir=output_dir,
        gpus=gpus,
        started_at=started_at,
        enable_recording=args.enable_recording,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.export_video and not args.enable_recording:
        parser.error("--export-video requires --enable-recording")
    eval_cfg = load_eval_config(args.eval_config)
    if args._shard_id is not None:
        if args._shard_manifest is None:
            raise SystemExit(f"{_SHARD_MANIFEST_FLAG} is required")
        from robo_orchard_sim.pipeline.evaluator.distributed import (
            read_evaluation_manifest,
        )
        from robo_orchard_sim.pipeline.evaluator.execution import run_shard

        manifest = read_evaluation_manifest(args._shard_manifest)
        if manifest.policy_id != eval_cfg.policy.model_type:
            raise ValueError("manifest policy differs from evaluation config")
        shard = next(
            shard
            for shard in manifest.shards
            if shard.shard_id == args._shard_id
        )
        output_dir = Path(args.output_dir).resolve()
        policy_config = _load_model_cfg(eval_cfg.policy)
        sys.exit(
            run_shard(
                manifest=manifest,
                shard=shard,
                task_settings=asdict(eval_cfg.task(shard.task_name)),
                policy_config=policy_config,
                output_dir=output_dir,
                enable_recording=args.enable_recording,
            )
        )

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        result = _dispatch(eval_cfg, args)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    if args.export_video:
        print(
            "\n=== Exporting MP4 videos and updating summary ===", flush=True
        )
        try:
            export_result = subprocess.run(
                [
                    sys.executable,
                    str(_REPO_ROOT / "tools" / "mcap_to_video.py"),
                    str(Path(args.output_dir).resolve()),
                ],
                check=False,
            ).returncode
        except OSError as exc:
            print(f"[error] Cannot start video export: {exc}", flush=True)
            export_result = 1
        if export_result:
            print(
                "[error] Video export/backfill failed; "
                "evaluation status and scores are unchanged.",
                flush=True,
            )
            result = 1
    sys.exit(result)


if __name__ == "__main__":
    main()
