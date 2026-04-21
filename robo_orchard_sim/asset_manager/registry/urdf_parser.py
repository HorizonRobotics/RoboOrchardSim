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

"""Parse <extra_info> and <inertial> blocks from URDF XML text."""

from __future__ import annotations
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

REQUIRED_FIELDS = ("uuid", "domain", "super_category", "category")
OPTIONAL_ATTR_FIELDS = ("color", "shape", "material")
STRING_FIELDS = ("name", "description", "version", "generate_time")
FLOAT_FIELDS = (
    "min_height",
    "max_height",
    "real_height",
    "min_mass",
    "max_mass",
)


@dataclass
class ParsedUrdf:
    """Flat struct holding everything extracted from a URDF file."""

    uuid: str = ""
    domain: str = ""
    super_category: str = ""
    category: str = ""
    name: str = ""
    description: str = ""
    color: str | None = None
    shape: str | None = None
    material: str | None = None
    real_height: float = 0.0
    min_height: float = 0.0
    max_height: float = 0.0
    real_mass: float = 0.0
    min_mass: float = 0.0
    max_mass: float = 0.0
    version: str = ""
    generate_time: str = ""
    tags: frozenset[str] = field(default_factory=frozenset)
    warnings: list[str] = field(default_factory=list)


def _text(elem: ET.Element | None) -> str | None:
    if elem is None or elem.text is None:
        return None
    return elem.text.strip()


def _float(elem: ET.Element | None) -> float | None:
    txt = _text(elem)
    if txt is None or txt == "":
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def parse_urdf_extra_info(
    urdf_text: str, *, strict: bool = False
) -> ParsedUrdf | None:
    """Parse a URDF string.

    Returns None (or raises ValueError if strict) when the URDF has no
    <extra_info> block or lacks a required field.
    """
    root = ET.fromstring(urdf_text)
    extra = root.find(".//extra_info")
    if extra is None:
        if strict:
            raise ValueError("URDF has no <extra_info> block")
        return None

    out = ParsedUrdf()

    for fname in REQUIRED_FIELDS:
        val = _text(extra.find(fname))
        if val is None or val == "":
            if strict:
                raise ValueError(
                    f"URDF <extra_info> missing required '{fname}'"
                )
            return None
        setattr(out, fname, val)

    for fname in OPTIONAL_ATTR_FIELDS:
        val = _text(extra.find(fname))
        if val is None:
            out.warnings.append(f"missing optional attribute '{fname}'")
        else:
            setattr(out, fname, val)

    for fname in STRING_FIELDS:
        val = _text(extra.find(fname))
        if val is not None:
            setattr(out, fname, val)

    for fname in FLOAT_FIELDS:
        val = _float(extra.find(fname))
        if val is None:
            out.warnings.append(f"missing or unparseable float '{fname}'")
        else:
            setattr(out, fname, val)

    tags_elem = extra.find("tags")
    tags_text = _text(tags_elem)
    if tags_text:
        out.tags = frozenset(
            t.strip() for t in tags_text.split(",") if t.strip()
        )
    elif tags_elem is None:
        out.warnings.append("missing <tags> element")

    mass_elem = root.find(".//inertial/mass")
    if mass_elem is not None and "value" in mass_elem.attrib:
        try:
            out.real_mass = float(mass_elem.attrib["value"])
        except ValueError:
            out.warnings.append("inertial mass unparseable, defaulting 0")
    else:
        out.warnings.append("inertial mass missing, defaulting 0")

    return out
