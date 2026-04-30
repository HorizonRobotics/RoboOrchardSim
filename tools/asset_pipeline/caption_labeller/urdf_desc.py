# Project RoboOrchard
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""URDF I/O helpers for the caption_labeller skill.

Pure XML read/write. No GPT or filesystem discovery logic
beyond render-directory listing.
"""

from __future__ import annotations
import os
import xml.etree.ElementTree as ET

CAPTION_ELEMENT_TAG = "caption_candidates"
DEFAULT_RENDER_SUBDIR = "renders"
_REQUIRED_FIELDS = ("uuid", "category")


def _find_extra_info(urdf_path: str) -> tuple[ET.ElementTree, ET.Element]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    extra = root.find(".//link/extra_info")
    if extra is None:
        raise ValueError(
            f"URDF {urdf_path} has no <link>/<extra_info> element"
        )
    return tree, extra


def read_asset_fields(urdf_path: str) -> dict:
    """Read ``{uuid, category}`` from the URDF's ``<extra_info>``.

    Raises:
        KeyError: if either required field is missing or empty.
    """
    _, extra = _find_extra_info(urdf_path)
    out = {}
    for key in _REQUIRED_FIELDS:
        elem = extra.find(key)
        if elem is None or elem.text is None or not elem.text.strip():
            raise KeyError(f"URDF {urdf_path} missing required <{key}> field")
        out[key] = elem.text.strip()
    return out


def has_caption_candidates(urdf_path: str) -> bool:
    """Return True iff ``<extra_info>/<caption_candidates>`` exists."""
    _, extra = _find_extra_info(urdf_path)
    return extra.find(CAPTION_ELEMENT_TAG) is not None


def write_caption_candidates_link(urdf_path: str, relative_path: str) -> None:
    """Insert or update the ``<caption_candidates>`` element.

    The URDF file is rewritten in place. Sibling fields are preserved.
    """
    tree, extra = _find_extra_info(urdf_path)
    elem = extra.find(CAPTION_ELEMENT_TAG)
    if elem is None:
        elem = ET.SubElement(extra, CAPTION_ELEMENT_TAG)
    elem.text = relative_path
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)


_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def find_renders(urdf_path: str) -> list[str]:
    """Return sorted image paths in the sibling ``renders/`` directory.

    Returns an empty list if the directory does not exist.
    """
    renders_dir = os.path.join(
        os.path.dirname(os.path.abspath(urdf_path)), DEFAULT_RENDER_SUBDIR
    )
    if not os.path.isdir(renders_dir):
        return []
    entries = sorted(os.listdir(renders_dir))
    return [
        os.path.join(renders_dir, e)
        for e in entries
        if e.lower().endswith(_IMAGE_EXTS)
    ]
