## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

from __future__ import annotations
import argparse
import json
from pathlib import Path

from analysis_core import validate_analysis


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_dirs(path: Path) -> list[Path]:
    if path.is_file():
        return [path.parent]
    if (path / "analysis.json").is_file():
        return [path]
    cases_dir = path / "cases"
    if cases_dir.is_dir():
        return sorted(
            child
            for child in cases_dir.iterdir()
            if (child / "analysis.json").is_file()
        )
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate pick-failure analysis records and evidence."
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    case_dirs = _case_dirs(args.path)
    if not case_dirs:
        raise SystemExit(f"No analysis.json found below {args.path}")
    invalid = 0
    for case_dir in case_dirs:
        analysis = _read_json(case_dir / "analysis.json")
        signals = _read_json(case_dir / "extracted_signals.json")
        result = validate_analysis(analysis, signals)
        result["seed"] = analysis.get("seed")
        (case_dir / "validation.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        invalid += not result["valid"]
        print(f"seed={analysis.get('seed')} valid={result['valid']}")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
