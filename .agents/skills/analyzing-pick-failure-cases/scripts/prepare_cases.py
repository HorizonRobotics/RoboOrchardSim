## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

from __future__ import annotations
import argparse
import json
from pathlib import Path

from analysis_core import (
    CAMERA_TOPICS,
    AnalysisError,
    discover_mcap,
    extract_evaluation_metrics,
    extract_frames_at_indexes,
    extract_mcap_manifest,
    load_eval_result,
    nearest_timestamp_index,
    reconcile_evaluation_metrics,
    render_candidate_sheet,
    select_episode_results,
)


def _parse_seeds(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _overview_indexes(frame_count: int, stride: int) -> list[int]:
    indexes = list(range(0, frame_count, stride))
    if frame_count and (not indexes or indexes[-1] != frame_count - 1):
        indexes.append(frame_count - 1)
    return indexes


def _close_candidate_indexes(manifest: dict, view: str) -> list[list[int]]:
    view_timestamps = manifest["camera_frames"][view]["timestamps_ns"]
    samples = manifest["gripper"].get("samples", [])
    windows = []
    for interval in manifest["gripper"].get("close_intervals", []):
        start_sample = samples[interval["start_index"]]["timestamp_ns"]
        end_sample = samples[interval["end_index"]]["timestamp_ns"]
        start_frame = nearest_timestamp_index(view_timestamps, start_sample)
        end_frame = nearest_timestamp_index(view_timestamps, end_sample)
        lower = max(0, start_frame - 15)
        upper = min(len(view_timestamps) - 1, end_frame + 15)
        indexes = list(range(lower, upper + 1, 3))
        if indexes[-1] != upper:
            indexes.append(upper)
        windows.append(indexes)
    return windows


def _render_sheets(
    *,
    mcap_path: Path,
    manifest: dict,
    case_dir: Path,
    overview_stride: int,
) -> list[dict]:
    artifacts = []
    candidates_dir = case_dir / "evidence_candidates"
    for view in CAMERA_TOPICS:
        frame_count = len(manifest["camera_frames"][view]["indexes"])
        if not frame_count:
            continue
        indexes = _overview_indexes(frame_count, overview_stride)
        images = extract_frames_at_indexes(mcap_path, view, indexes)
        for page, offset in enumerate(range(0, len(indexes), 20)):
            page_indexes = indexes[offset : offset + 20]
            artifacts.append(
                render_candidate_sheet(
                    frames=[(index, images[index]) for index in page_indexes],
                    view=view,
                    output_path=candidates_dir
                    / f"overview_{view}_{page:03d}.jpg",
                )
            )
        for window, dense_indexes in enumerate(
            _close_candidate_indexes(manifest, view)
        ):
            dense_images = extract_frames_at_indexes(
                mcap_path, view, dense_indexes
            )
            artifacts.append(
                render_candidate_sheet(
                    frames=[
                        (index, dense_images[index]) for index in dense_indexes
                    ],
                    view=view,
                    output_path=candidates_dir
                    / f"close_{window:03d}_{view}.jpg",
                )
            )
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare continuous multi-view evidence for pick failures."
    )
    parser.add_argument("eval_root", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seeds", type=_parse_seeds)
    mode.add_argument("--all-failures", action="store_true")
    parser.add_argument("--include-success", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overview-stride", type=int, default=10)
    parser.add_argument("--close-minimum-delta", type=float, default=0.3)
    parser.add_argument("--close-threshold", type=float, default=0.6)
    parser.add_argument("--close-minimum-samples", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.overview_stride < 1:
        raise SystemExit("--overview-stride must be positive")
    eval_result_path = args.eval_root / "eval_result.json"
    eval_result = load_eval_result(eval_result_path)
    selected = select_episode_results(
        eval_result,
        seeds=args.seeds,
        all_failures=args.all_failures,
        include_success=args.include_success,
    )
    output_dir = args.output_dir or (
        Path("/tmp/pick_failure_analysis") / args.eval_root.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    completed = 0
    for episode_result in selected:
        seed = int(episode_result["seed"])
        case_dir = output_dir / "cases" / f"seed_{seed}"
        try:
            mcap_path = discover_mcap(args.eval_root, seed)
            manifest = extract_mcap_manifest(
                mcap_path,
                minimum_delta=args.close_minimum_delta,
                closed_threshold=args.close_threshold,
                minimum_samples=args.close_minimum_samples,
            )
            manifest["seed"] = seed
            manifest["evaluation"] = reconcile_evaluation_metrics(
                extract_evaluation_metrics(episode_result),
                manifest["metadata"],
            )
            manifest["candidate_artifacts"] = _render_sheets(
                mcap_path=mcap_path,
                manifest=manifest,
                case_dir=case_dir,
                overview_stride=args.overview_stride,
            )
            _write_json(case_dir / "extracted_signals.json", manifest)
            completed += 1
        except (AnalysisError, OSError, ValueError) as error:
            errors.append({"seed": seed, "error": str(error)})
    _write_json(output_dir / "errors.json", errors)
    print(output_dir)
    print(f"prepared={completed} errors={len(errors)}")
    return 1 if errors and not completed else 0


if __name__ == "__main__":
    raise SystemExit(main())
