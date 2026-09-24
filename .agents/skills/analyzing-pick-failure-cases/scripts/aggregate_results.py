## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

from __future__ import annotations
import argparse
import json
from pathlib import Path

from analysis_core import (
    aggregate_analyses,
    load_eval_result,
    render_aggregate_overview,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate validated pick-failure analysis records."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("eval_result", type=Path)
    parser.add_argument("--include-invalid", action="store_true")
    args = parser.parse_args()

    analyses = []
    validations = []
    for case_dir in sorted((args.output_dir / "cases").glob("seed_*")):
        analysis_path = case_dir / "analysis.json"
        if not analysis_path.is_file():
            continue
        validation_path = case_dir / "validation.json"
        validation = (
            _read_json(validation_path)
            if validation_path.is_file()
            else {
                "seed": _read_json(analysis_path).get("seed"),
                "valid": False,
            }
        )
        validations.append(validation)
        if validation.get("valid") or args.include_invalid:
            analyses.append(_read_json(analysis_path))

    eval_result = load_eval_result(args.eval_result)
    summary = aggregate_analyses(
        eval_result=eval_result,
        analyses=analyses,
        validation_results=validations,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "failure_analysis.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for analysis in analyses:
            stream.write(json.dumps(analysis, ensure_ascii=False) + "\n")
    (args.output_dir / "aggregate_statistics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    render_aggregate_overview(
        summary, args.output_dir / "aggregate_overview.jpg"
    )
    print(args.output_dir / "aggregate_statistics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
