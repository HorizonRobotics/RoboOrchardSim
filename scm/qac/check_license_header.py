## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

"""Check that Python files carry the repository copyright header."""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable

SKIP_BASENAMES = {"__git_hash__.py"}
SKIP_PREFIXES = ("gen_",)
SKIP_DIRS = {Path("examples/manipulation-app/pick_place/config")}


def should_skip(path: Path) -> bool:
    if path.name in SKIP_BASENAMES:
        return True
    if path.name.startswith(SKIP_PREFIXES):
        return True
    return any(skip_dir in path.parents for skip_dir in SKIP_DIRS)


def has_license_header(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[:5]
    except OSError:
        return False
    return any("Copyright" in line for line in lines)


def iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    for raw_path in paths:
        path = raw_path if raw_path.is_absolute() else Path.cwd() / raw_path
        if path.is_dir():
            yield from (
                child
                for child in path.rglob("*.py")
                if child.is_file() and not should_skip(child)
            )
        elif path.suffix == ".py" and path.is_file() and not should_skip(path):
            yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()

    paths = args.paths or [Path.cwd()]
    missing = sorted(
        {
            path
            for path in iter_python_files(paths)
            if not has_license_header(path)
        }
    )

    if missing:
        for path in missing:
            print(f"{path}: missing copyright header")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
