## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

from __future__ import annotations
import argparse
import json
from pathlib import Path

from analysis_core import (
    extract_frames_at_indexes,
    nearest_timestamp_index,
    render_case_image,
    validate_analysis,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a standardized pick-failure evidence image."
    )
    parser.add_argument("case_dir", nargs="?", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--extracted-signals", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.case_dir:
        analysis_path = args.analysis or args.case_dir / "analysis.json"
        signals_path = (
            args.extracted_signals or args.case_dir / "extracted_signals.json"
        )
        output_path = args.output or args.case_dir / "sequence_analysis.jpg"
    elif args.analysis and args.extracted_signals:
        analysis_path = args.analysis
        signals_path = args.extracted_signals
        output_path = args.output or analysis_path.with_name(
            "sequence_analysis.jpg"
        )
    else:
        raise SystemExit(
            "Provide case_dir or both --analysis and --extracted-signals"
        )

    analysis = _read_json(analysis_path)
    signals = _read_json(signals_path)
    validation = validate_analysis(analysis, signals)
    if not validation["valid"]:
        raise SystemExit(
            "Invalid analysis: " + "; ".join(validation["errors"])
        )

    anchors = analysis["evidence_window"]["frames"]
    anchor_timestamps = signals["camera_frames"]["ext1"]["timestamps_ns"]
    images = {}
    mcap_path = Path(signals["source_mcap"])
    for view in analysis["evidence_window"]["views"]:
        view_timestamps = signals["camera_frames"][view]["timestamps_ns"]
        source_indexes = [
            nearest_timestamp_index(view_timestamps, anchor_timestamps[anchor])
            for anchor in anchors
        ]
        decoded = extract_frames_at_indexes(mcap_path, view, source_indexes)
        for anchor, source_index in zip(anchors, source_indexes, strict=True):
            images[(anchor, view)] = decoded[source_index]
    render_case_image(
        analysis=analysis,
        images=images,
        output_path=output_path,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
