# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""Multi-process packing fan-out + main-process merge.

Each worker runs the existing single-process pipeline
(``McapEpisodeReader`` -> ``LeRobotEpisodePacker``) into its own temporary
shard directory.  After all workers finish, the main process merges the N
shard dirs into a single final LeRobotDataset on disk.

Merge step does only metadata-level work:
  * Rename mp4 files (`os.rename`, same-fs, atomic).
  * Rename depth PNG dirs (same-fs, atomic), only when depth is included.
  * Read parquet, rewrite three int columns
    (`episode_index`, `index`, `task_index`), write to new path.
  * Concatenate `episodes.jsonl` / `episodes_stats.jsonl` with renumbered
    `episode_index`.
  * Dedup `tasks.jsonl` into a global table; remember per-shard
    local->global task_index remap (used in parquet rewrite).
  * Patch `info.json` totals (episodes / frames / videos / chunks /
    splits).
  * Carry over `robot.urdf` and `cameras` / `extras` from shard 0.
"""

from __future__ import annotations
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# worker payload
# ---------------------------------------------------------------------------


@dataclass
class ShardPayload:
    """Pickleable description of one worker's slice of work."""

    shard_idx: int
    shard_dir: str
    repo_id: str
    embodiment: str
    mcaps: list[str]
    manifest_path: str | None
    image_scale: float
    include_depth: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# A multiprocessing.Queue shared with workers via the pool initializer so they
# can report "one mcap finished" without coupling to a specific progress UI.
# Main process drains the queue and updates a tqdm bar.
_PROGRESS_Q: Any = None


def _worker_init(progress_q: Any) -> None:
    global _PROGRESS_Q
    _PROGRESS_Q = progress_q


def _report_progress(n: int = 1) -> None:
    q = _PROGRESS_Q
    if q is None:
        return
    try:
        q.put_nowait(n)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# worker entry point (runs in child process)
# ---------------------------------------------------------------------------


def _worker_main(payload_dict: dict[str, Any]) -> dict[str, Any]:
    """Run one shard.  Returns a small summary dict for the main process."""
    # Re-import inside the worker so spawn-context starts clean.
    import logging as _logging
    import shutil as _shutil

    # JuiceFS FUSE backends sometimes reject the unlinkat(2) calls used by
    # _rmtree_safe_fd with EACCES.  LeRobot's encode_episode_videos calls
    # shutil.rmtree on the per-episode PNG staging dir; if we leave the
    # default fd-based path enabled, that rmtree blows up on bucket-mounted
    # shard dirs.  Disable the fd-walking path so rmtree uses os.unlink(path).
    if hasattr(_shutil, "rmtree") and getattr(
        _shutil.rmtree, "avoids_symlink_attacks", False
    ):
        _shutil.rmtree.avoids_symlink_attacks = False  # type: ignore[attr-defined]
    _shutil._use_fd_functions = False  # type: ignore[attr-defined]

    from tools.mcap_to_lerobot.config_presets import EMBODIMENT_REGISTRY
    from tools.mcap_to_lerobot.convert import _instruction_for, _load_manifest
    from tools.mcap_to_lerobot.reader import McapEpisodeReader
    from tools.mcap_to_lerobot.writer import LeRobotEpisodePacker

    _logging.basicConfig(
        level=_logging.INFO,
        format=f"[shard {payload_dict['shard_idx']:02d}] "
        "%(asctime)s %(levelname)s %(message)s",
    )
    log = _logging.getLogger("worker")

    p = ShardPayload(**payload_dict)
    if not p.mcaps:
        log.warning("empty shard; nothing to do")
        return {
            "shard_idx": p.shard_idx,
            "shard_dir": p.shard_dir,
            "n_episodes": 0,
            "n_frames": 0,
            "n_ok": 0,
            "n_fail": 0,
            "errors": [],
        }

    cfg = EMBODIMENT_REGISTRY[p.embodiment]
    reader = McapEpisodeReader(
        cfg, image_scale=p.image_scale, include_depth=p.include_depth
    )

    manifest: dict[str, str] = {}
    if p.manifest_path is not None:
        manifest = _load_manifest(Path(p.manifest_path))

    packer = LeRobotEpisodePacker.create(
        repo_id=p.repo_id,
        root=Path(p.shard_dir),
        cfg=cfg,
        first_mcap=Path(p.mcaps[0]),
        use_videos=True,
        image_scale=p.image_scale,
        include_depth=p.include_depth,
    )

    n_ok = n_fail = n_frames = 0
    errors: list[str] = []
    for mcap_str in p.mcaps:
        mcap_path = Path(mcap_str)
        instruction = _instruction_for(mcap_path, manifest)
        try:
            ep = reader.read(
                mcap_path,
                manifest_instruction=instruction,
                default_task="manipulation",
            )
        except Exception as exc:
            log.exception("read failed: %s", mcap_path)
            n_fail += 1
            errors.append(
                f"read {mcap_path.name}: {type(exc).__name__}: {exc}"
            )
            _report_progress(1)
            continue
        if ep.num_frames == 0:
            log.warning("skip %s: 0 retained frames", mcap_path.name)
            n_fail += 1
            errors.append(f"empty {mcap_path.name}")
            _report_progress(1)
            continue
        try:
            packer.add_episode(ep)
        except Exception as exc:
            log.exception("write failed: %s", mcap_path)
            n_fail += 1
            errors.append(
                f"write {mcap_path.name}: {type(exc).__name__}: {exc}"
            )
            _report_progress(1)
            continue
        n_ok += 1
        n_frames += ep.num_frames
        _report_progress(1)

    packer.finalize()
    return {
        "shard_idx": p.shard_idx,
        "shard_dir": p.shard_dir,
        "n_episodes": packer.dataset.meta.total_episodes,
        "n_frames": packer.dataset.meta.total_frames,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# merge: shard dirs -> single final dataset
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_global_task_remap(
    shard_dirs: list[Path],
) -> tuple[list[str], list[dict[int, int]]]:
    """Collect tasks across shards; assign global task_index by first-seen.

    Returns (global_tasks, per_shard_remap) where per_shard_remap[K]
    maps shard-K's local task_index -> global task_index.
    """
    global_tasks: list[str] = []
    desc_to_global: dict[str, int] = {}
    per_shard_remap: list[dict[int, int]] = []
    for shard in shard_dirs:
        rows = _read_jsonl(shard / "meta" / "tasks.jsonl")
        remap: dict[int, int] = {}
        for row in rows:
            local_idx = int(row["task_index"])
            desc = row["task"]
            if desc not in desc_to_global:
                desc_to_global[desc] = len(global_tasks)
                global_tasks.append(desc)
            remap[local_idx] = desc_to_global[desc]
        per_shard_remap.append(remap)
    return global_tasks, per_shard_remap


def _rewrite_parquet(
    src: Path,
    dst: Path,
    new_episode_index: int,
    frame_index_offset: int,
    task_remap: dict[int, int],
) -> int:
    """Read parquet, rewrite 3 int columns, write to dst.

    Returns num_rows.
    """
    t = pq.read_table(src)
    n = t.num_rows
    cols = {name: t.column(name) for name in t.column_names}
    cols["episode_index"] = pa.array([new_episode_index] * n, pa.int64())
    cols["index"] = pa.array(
        np.arange(frame_index_offset, frame_index_offset + n, dtype=np.int64),
        pa.int64(),
    )
    cols["frame_index"] = pa.array(np.arange(n, dtype=np.int64), pa.int64())
    if "task_index" in t.column_names and task_remap:
        old = t.column("task_index").to_pylist()
        cols["task_index"] = pa.array(
            [task_remap.get(int(v), int(v)) for v in old], pa.int64()
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), dst)
    return n


def _move_video_files(
    shard: Path, final: Path, ep_local: int, ep_global: int, chunk_size: int
) -> int:
    """Rename mp4 files for one episode across all camera dirs.

    Returns number of mp4 files moved.
    """
    src_chunk = ep_local // chunk_size
    dst_chunk = ep_global // chunk_size
    src_videos = shard / "videos" / f"chunk-{src_chunk:03d}"
    if not src_videos.exists():
        return 0
    moved = 0
    for cam_dir in src_videos.iterdir():
        if not cam_dir.is_dir():
            continue
        src_mp4 = cam_dir / f"episode_{ep_local:06d}.mp4"
        if not src_mp4.exists():
            continue
        dst_dir = final / "videos" / f"chunk-{dst_chunk:03d}" / cam_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        os.rename(src_mp4, dst_dir / f"episode_{ep_global:06d}.mp4")
        moved += 1
    return moved


def _move_image_dirs(
    shard: Path, final: Path, ep_local: int, ep_global: int
) -> int:
    """Rename per-episode PNG directories.

    LeRobot's image layout is ``images/<feature_key>/episode_NNNNNN/``.
    This covers depth, or color in ``--no-videos`` mode.
    We rename the entire directory: PNG files inside are not touched.
    """
    src_root = shard / "images"
    if not src_root.exists():
        return 0
    moved = 0
    for feat_dir in src_root.iterdir():
        if not feat_dir.is_dir():
            continue
        src_ep_dir = feat_dir / f"episode_{ep_local:06d}"
        if not src_ep_dir.exists():
            continue
        dst_feat = final / "images" / feat_dir.name
        dst_feat.mkdir(parents=True, exist_ok=True)
        os.rename(src_ep_dir, dst_feat / f"episode_{ep_global:06d}")
        moved += 1
    return moved


def _merge_shards(
    shard_dirs: list[Path],
    final_dir: Path,
    chunk_size: int = 1000,
    keep_shards: bool = False,
) -> dict[str, Any]:
    """Merge worker shard datasets into a single LeRobot v2.0 dataset.

    Assumes:
      * All shards have identical features schema, fps, robot_type, cameras
        calib (we copy these from shard 0).
      * episodes.jsonl rows are ordered by local episode_index 0..n-1.
      * tmp shards and final_dir live on the same filesystem so that
        ``os.rename`` is atomic.
    """
    if not shard_dirs:
        raise ValueError("no shard dirs to merge")
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "meta").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("Merge phase: %d shards -> %s", len(shard_dirs), final_dir)

    # 1) Build global task table + per-shard remap.
    global_tasks, per_shard_remap = _build_global_task_remap(shard_dirs)
    logger.info("Global tasks after dedup: %d", len(global_tasks))

    # 2) Pre-scan episodes.jsonl per shard to compute global offsets.
    shard_eps: list[list[dict[str, Any]]] = [
        _read_jsonl(s / "meta" / "episodes.jsonl") for s in shard_dirs
    ]
    shard_eps_stats: list[list[dict[str, Any]]] = [
        _read_jsonl(s / "meta" / "episodes_stats.jsonl") for s in shard_dirs
    ]
    ep_offsets = [0]
    for eps in shard_eps:
        ep_offsets.append(ep_offsets[-1] + len(eps))
    total_eps = ep_offsets[-1]
    if total_eps == 0:
        raise RuntimeError("all shards are empty; nothing to merge")

    # 3) Walk shards, rewrite parquet, move mp4 + image dirs.
    all_eps_meta: list[dict[str, Any]] = []
    all_eps_stats: list[dict[str, Any]] = []
    total_frames = 0
    n_videos = 0

    for k, shard in enumerate(shard_dirs):
        eps = shard_eps[k]
        stats = shard_eps_stats[k]
        remap = per_shard_remap[k]
        # episodes_stats.jsonl is keyed by episode_index too; build lookup.
        stats_by_local: dict[int, dict[str, Any]] = {
            int(row.get("episode_index", -1)): row for row in stats
        }

        for ep_local, ep_row in enumerate(eps):
            ep_global = ep_offsets[k] + ep_local

            # parquet
            src_chunk_local = ep_local // chunk_size
            dst_chunk_global = ep_global // chunk_size
            src_pq = (
                shard
                / "data"
                / f"chunk-{src_chunk_local:03d}"
                / f"episode_{ep_local:06d}.parquet"
            )
            dst_pq = (
                final_dir
                / "data"
                / f"chunk-{dst_chunk_global:03d}"
                / f"episode_{ep_global:06d}.parquet"
            )
            n_rows = _rewrite_parquet(
                src_pq, dst_pq, ep_global, total_frames, remap
            )
            total_frames += n_rows

            # mp4 + image dirs
            n_videos += _move_video_files(
                shard, final_dir, ep_local, ep_global, chunk_size
            )
            _move_image_dirs(shard, final_dir, ep_local, ep_global)

            # episodes meta row (renumbered)
            ep_row_new = dict(ep_row)
            ep_row_new["episode_index"] = ep_global
            all_eps_meta.append(ep_row_new)

            # episodes_stats row
            stat_row = stats_by_local.get(ep_local)
            if stat_row is not None:
                stat_row_new = dict(stat_row)
                stat_row_new["episode_index"] = ep_global
                all_eps_stats.append(stat_row_new)

    # 4) Write merged meta
    _write_jsonl(final_dir / "meta" / "episodes.jsonl", all_eps_meta)
    if all_eps_stats:
        _write_jsonl(
            final_dir / "meta" / "episodes_stats.jsonl", all_eps_stats
        )
    _write_jsonl(
        final_dir / "meta" / "tasks.jsonl",
        [{"task_index": i, "task": d} for i, d in enumerate(global_tasks)],
    )

    # 5) info.json: copy shard 0, patch totals.
    src_info = json.loads(
        (shard_dirs[0] / "meta" / "info.json").read_text(encoding="utf-8")
    )
    src_info["total_episodes"] = total_eps
    src_info["total_frames"] = total_frames
    src_info["total_videos"] = n_videos
    src_info["total_chunks"] = (total_eps + chunk_size - 1) // chunk_size
    src_info["splits"] = {"train": f"0:{total_eps}"}
    src_info["total_tasks"] = len(global_tasks)
    (final_dir / "meta" / "info.json").write_text(
        json.dumps(src_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 6) URDF (copy from shard 0 if present)
    src_urdf = shard_dirs[0] / "meta" / "robot.urdf"
    if src_urdf.exists():
        shutil.copy2(src_urdf, final_dir / "meta" / "robot.urdf")

    # 7) Cleanup
    if not keep_shards:
        for shard in shard_dirs:
            shutil.rmtree(shard, ignore_errors=True)

    dt = time.time() - t0
    logger.info(
        "Merge done in %.1fs: %d episodes, %d frames, %d videos -> %s",
        dt,
        total_eps,
        total_frames,
        n_videos,
        final_dir,
    )
    return {
        "n_episodes": total_eps,
        "n_frames": total_frames,
        "n_videos": n_videos,
        "n_tasks": len(global_tasks),
        "elapsed_s": dt,
    }


# ---------------------------------------------------------------------------
# top-level orchestrator
# ---------------------------------------------------------------------------


def _split_mcaps(mcaps: list[Path], num_workers: int) -> list[list[Path]]:
    """Round-robin split into N equally-sized lists.

    For homogeneous-length episodes this is good enough; LPT scheduling is
    only worth it when episode lengths differ wildly.
    """
    buckets: list[list[Path]] = [[] for _ in range(num_workers)]
    for i, m in enumerate(mcaps):
        buckets[i % num_workers].append(m)
    return [b for b in buckets if b]


def run_parallel(
    *,
    mcaps: list[Path],
    out: Path,
    repo_id: str,
    embodiment: str,
    manifest_path: Path | None,
    image_scale: float,
    include_depth: bool,
    num_workers: int,
    chunk_size: int = 1000,
    keep_shards: bool = False,
    best_effort: bool = False,
    upload_to: str | None = None,
) -> int:
    """Drive parallel packing and merge.

    Returns 0 on success, 2 on partial failure (best-effort), 1 on hard
    failure.
    """
    import multiprocessing as mp

    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if not mcaps:
        logger.error("no mcaps to process")
        return 1
    if out.exists():
        logger.error(
            "out exists: %s; remove it or pick a different --out path", out
        )
        return 1

    tmp_dir = out.with_suffix(out.suffix + ".tmp_shards")
    if tmp_dir.exists():
        logger.error(
            "tmp shard dir exists: %s; remove it before re-running",
            tmp_dir,
        )
        return 1
    tmp_dir.mkdir(parents=True, exist_ok=False)

    # Split & build payloads.
    buckets = _split_mcaps(mcaps, num_workers)
    actual_workers = len(buckets)
    if actual_workers < num_workers:
        logger.info(
            "Reducing workers %d -> %d (more workers than mcaps)",
            num_workers,
            actual_workers,
        )
    payloads: list[dict[str, Any]] = []
    shard_dirs: list[Path] = []
    for i, bucket in enumerate(buckets):
        shard_dir = tmp_dir / f"shard_{i:03d}"
        shard_dirs.append(shard_dir)
        payloads.append(
            ShardPayload(
                shard_idx=i,
                shard_dir=str(shard_dir),
                repo_id=f"{repo_id}/shard_{i:03d}",
                embodiment=embodiment,
                mcaps=[str(p) for p in bucket],
                manifest_path=str(manifest_path) if manifest_path else None,
                image_scale=image_scale,
                include_depth=include_depth,
            ).to_dict()
        )

    logger.info(
        "Parallel pack: %d mcaps over %d workers, tmp=%s, final=%s",
        len(mcaps),
        actual_workers,
        tmp_dir,
        out,
    )

    # Run pool with spawn context (CUDA/torch safe).
    t0 = time.time()
    ctx = mp.get_context("spawn")
    results: list[dict[str, Any]] = []

    progress_q = ctx.Queue()
    total_mcaps = len(mcaps)
    pbar = tqdm(
        total=total_mcaps,
        desc="pack mcaps",
        unit="mcap",
        dynamic_ncols=True,
    )
    stop_evt = threading.Event()

    def _drain() -> None:
        while not stop_evt.is_set():
            try:
                n = progress_q.get(timeout=0.2)
            except Exception:
                continue
            pbar.update(int(n))
        # Drain anything that arrived between last get and stop_evt.set().
        while True:
            try:
                n = progress_q.get_nowait()
            except Exception:
                break
            pbar.update(int(n))

    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()

    try:
        if actual_workers == 1:
            # Single-process path: useful for --num-workers 1 debug,
            # with no pool overhead.  Wire the same queue so the bar still
            # advances.
            _worker_init(progress_q)
            results.append(_worker_main(payloads[0]))
        else:
            with ctx.Pool(
                actual_workers,
                initializer=_worker_init,
                initargs=(progress_q,),
            ) as pool:
                for r in pool.imap_unordered(_worker_main, payloads):
                    results.append(r)
                    tqdm.write(
                        "[shard %02d] done: %d ep, %d frames, %d ok, %d fail"
                        % (
                            r["shard_idx"],
                            r["n_episodes"],
                            r["n_frames"],
                            r["n_ok"],
                            r["n_fail"],
                        )
                    )
    finally:
        stop_evt.set()
        drainer.join(timeout=2.0)
        pbar.close()
    pack_elapsed = time.time() - t0
    logger.info("Worker phase: %.1fs", pack_elapsed)

    # Failure summary.
    total_fail = sum(r["n_fail"] for r in results)
    if total_fail > 0:
        logger.warning("Failed mcaps: %d", total_fail)
        for r in results:
            for err in r["errors"]:
                logger.warning("  shard %02d: %s", r["shard_idx"], err)
        if not best_effort:
            logger.error(
                "%d mcap(s) failed; aborting (use --best-effort to merge "
                "successful shards anyway). Tmp shards retained at %s",
                total_fail,
                tmp_dir,
            )
            return 1

    # Order shard_dirs by shard_idx so global episode order is deterministic
    # (imap_unordered may have shuffled them above).
    results.sort(key=lambda r: r["shard_idx"])
    shard_dirs_sorted = [
        Path(r["shard_dir"]) for r in results if r["n_episodes"] > 0
    ]
    if not shard_dirs_sorted:
        logger.error("no shard produced any episode; nothing to merge")
        return 1

    # Merge.
    summary = _merge_shards(
        shard_dirs_sorted,
        out,
        chunk_size=chunk_size,
        keep_shards=keep_shards,
    )

    # Drop the now-empty tmp_dir if not keeping shards.
    if not keep_shards:
        try:
            tmp_dir.rmdir()
        except OSError:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    total_elapsed = time.time() - t0
    logger.info(
        "Total: %.1fs (worker %.1fs + merge %.1fs). "
        "%d episodes, %d frames -> %s",
        total_elapsed,
        pack_elapsed,
        summary["elapsed_s"],
        summary["n_episodes"],
        summary["n_frames"],
        out,
    )

    # Optional upload to JuiceFS / remote bucket.
    if upload_to is not None:
        _upload_dataset(out, upload_to)

    return 0 if total_fail == 0 else 2


def _upload_dataset(local_dir: Path, dst: str) -> None:
    """Copy a finished dataset to a remote/bucket path.

    Uses plain ``cp -r`` because some bucket backends (JuiceFS pointed at
    write-once object stores) reject ``rename`` and ``unlink`` — which
    ``rsync`` and ``juicefs sync`` both rely on for their staging /
    atomic-replace patterns.  ``cp -r`` only does ``open(O_CREAT|O_WRONLY)``
    + write, which works on append-only mounts.

    The dst path **must not exist** — if it does we abort, because we
    cannot safely overwrite (no rename, no unlink available on the target).
    """
    import shutil as _shutil
    import subprocess as _subprocess

    src = str(local_dir).rstrip("/")
    dst_norm = dst.rstrip("/")

    if Path(dst_norm).exists():
        logger.error(
            "Upload target already exists (cannot safely overwrite on "
            "append-only backends): %s",
            dst_norm,
        )
        return

    cp = _shutil.which("cp")
    if not cp:
        logger.error("cp not available; skipping upload to %s", dst)
        return
    cmd = [cp, "-r", src, dst_norm]
    logger.info("Upload: %s", " ".join(cmd))
    t0 = time.time()
    rc = _subprocess.run(cmd).returncode
    dt = time.time() - t0
    if rc == 0:
        logger.info("Upload done in %.1fs -> %s", dt, dst)
    else:
        logger.error("Upload failed (rc=%d) after %.1fs", rc, dt)
