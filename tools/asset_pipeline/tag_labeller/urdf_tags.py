# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""URDF <tags> element read/write helpers.

The asset registry's AssetFilter consumes a comma-separated <tags>
element nested under <link><extra_info>. This module provides the
minimal I/O surface needed by the tag labeller to insert, read,
and update that element while leaving other URDF content untouched.
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from typing import Iterable


def _find_extra_info(root: ET.Element) -> ET.Element:
    elem = root.find(".//link/extra_info")
    if elem is None:
        raise ValueError("URDF has no <link>/<extra_info> element")
    return elem


def read_extra_info(urdf_path: str) -> dict[str, str]:
    """Return all child elements of <extra_info> as a dict of strings."""
    tree = ET.parse(urdf_path)
    extra = _find_extra_info(tree.getroot())
    out: dict[str, str] = {}
    for child in extra:
        out[child.tag] = (child.text or "").strip()
    return out


def has_tags_element(urdf_path: str) -> bool:
    """Whether <extra_info><tags> exists (regardless of content)."""
    tree = ET.parse(urdf_path)
    extra = _find_extra_info(tree.getroot())
    return extra.find("tags") is not None


def read_tags(urdf_path: str) -> set[str]:
    """Return the set of tags from <extra_info><tags>, empty if missing."""
    tree = ET.parse(urdf_path)
    extra = _find_extra_info(tree.getroot())
    elem = extra.find("tags")
    if elem is None:
        return set()
    text = (elem.text or "").strip()
    if not text:
        return set()
    return {t.strip() for t in text.split(",") if t.strip()}


def write_tags(
    urdf_path: str,
    tags: Iterable[str],
    merge_with_existing: bool = False,
) -> None:
    """Write the <tags> element. Inserts the element if missing.

    Args:
        urdf_path: Path to the URDF file (modified in place).
        tags: Tag names to write.
        merge_with_existing: If True, union with existing tags. If False,
            overwrite.
    """
    tree = ET.parse(urdf_path)
    extra = _find_extra_info(tree.getroot())
    elem = extra.find("tags")

    new_set = {t for t in tags if t}
    if merge_with_existing and elem is not None:
        existing = (elem.text or "").strip()
        if existing:
            new_set |= {t.strip() for t in existing.split(",") if t.strip()}

    payload = ",".join(sorted(new_set))

    if elem is None:
        elem = ET.SubElement(extra, "tags")
    elem.text = payload

    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
