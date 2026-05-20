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

"""Parse layout JSON descriptions into typed dataclasses."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any  # noqa: UP035


class LayoutValidationError(ValueError):
    """Raised when a layout JSON fails structural validation."""


@dataclass(frozen=True)
class LayoutObject:
    """Spawn parameters for a single object placement."""

    category: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]  # (w, x, y, z) Isaac order


@dataclass(frozen=True)
class Layout:
    """Parsed layout JSON keyed by role."""

    objects: dict[str, LayoutObject]
    raw: dict[str, Any] = field(repr=False)


def normalize_category(value: str) -> str:
    """Lowercase, trim, and replace separators for category matching."""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _parse_binding(role: str, spec: dict[str, Any]) -> LayoutObject:
    for key in ("category", "position", "rotation"):
        if key not in spec:
            raise LayoutValidationError(
                f"position[{role!r}] missing required field {key!r}"
            )
    pos = spec["position"]
    rot = spec["rotation"]
    if len(pos) != 3:
        raise LayoutValidationError(
            f"position[{role!r}].position must be 3 floats [x,y,z], "
            f"got {len(pos)}: {pos}"
        )
    if len(rot) != 4:
        raise LayoutValidationError(
            f"position[{role!r}].rotation must be 4 floats [w,x,y,z], "
            f"got {len(rot)}: {rot}"
        )
    return LayoutObject(
        category=normalize_category(spec["category"]),
        position=(float(pos[0]), float(pos[1]), float(pos[2])),
        rotation=(
            float(rot[0]),
            float(rot[1]),
            float(rot[2]),
            float(rot[3]),
        ),
    )


@dataclass(frozen=True)
class LayoutSequence:
    """Ordered list of Layout entries. All entries share the same role keys."""

    entries: list[Layout]
    raw: list[dict[str, Any]] | dict[str, Any] = field(repr=False)


def parse_layout(path: str | Path) -> LayoutSequence:
    """Load and validate a layout JSON file.

    Accepts a top-level list of layout dicts (one per episode) or a single
    top-level dict (auto-wrapped into a length-1 sequence). All entries must
    declare the same set of role keys; mismatch raises
    ``LayoutValidationError``. Non-``position`` fields are preserved verbatim
    under ``Layout.raw`` / ``LayoutSequence.raw``.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        layout = _parse_one_layout(raw)
        return LayoutSequence(entries=[layout], raw=raw)
    if isinstance(raw, list):
        if len(raw) == 0:
            raise LayoutValidationError("empty layout sequence")
        entries = [
            _parse_one_layout(item, idx=i) for i, item in enumerate(raw)
        ]
        canonical = set(entries[0].objects.keys())
        for i, layout in enumerate(entries[1:], start=1):
            if set(layout.objects.keys()) != canonical:
                raise LayoutValidationError(
                    f"entry[{i}] role keys differ from entry[0]: "
                    f"{sorted(layout.objects.keys())} vs "
                    f"{sorted(canonical)}"
                )
        return LayoutSequence(entries=entries, raw=raw)
    raise LayoutValidationError(
        f"layout JSON top level must be dict or list, got {type(raw).__name__}"
    )


def _parse_one_layout(raw: dict[str, Any], idx: int | None = None) -> Layout:
    """Parse a single layout dict into a ``Layout``."""
    where = f"entry[{idx}] " if idx is not None else ""
    if "position" not in raw:
        raise LayoutValidationError(
            f"{where}missing required field 'position'"
        )
    objects = {
        role: _parse_binding(role, spec)
        for role, spec in raw["position"].items()
    }
    return Layout(objects=objects, raw=raw)
