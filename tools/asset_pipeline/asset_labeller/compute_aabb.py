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

"""AABB helper: read local-frame AABB from a USD and write it into a URDF.

Used by the labeller (Step 2 of the asset batch labelling pipeline) and by
the one-shot backfill script for libraries pre-dating this feature.
"""

from __future__ import annotations
import logging
from pathlib import Path

from lxml import etree
from pxr import Usd, UsdGeom

logger = logging.getLogger(__name__)

Vec3 = tuple[float, float, float]


def compute_aabb_from_usd(
    usd_path: Path | str,
) -> tuple[Vec3, Vec3] | None:
    """Return (min_xyz, max_xyz) in the asset's local frame, or None.

    Uses ComputeUntransformedBound so any parent prim transforms baked
    into the stage do not enter the result — the values are pure mesh
    extents in the default prim's local frame.
    """
    path = Path(usd_path)
    if not path.is_file():
        return None
    try:
        stage = Usd.Stage.Open(str(path))
    except Exception as e:
        logger.debug("Usd.Stage.Open failed: %s", e)
        return None
    if stage is None:
        return None
    prim = stage.GetDefaultPrim()
    if not prim or not prim.IsValid():
        return None
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_],
    )
    try:
        bbox = bbox_cache.ComputeUntransformedBound(prim)
    except Exception as e:
        logger.debug("ComputeUntransformedBound failed: %s", e)
        return None
    rng = bbox.ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    mn = rng.GetMin()
    mx = rng.GetMax()
    return (
        (float(mn[0]), float(mn[1]), float(mn[2])),
        (float(mx[0]), float(mx[1]), float(mx[2])),
    )


def write_aabb_to_urdf(
    urdf_path: Path | str,
    aabb_min: Vec3,
    aabb_max: Vec3,
) -> None:
    """Insert/replace a structured <aabb> element in the URDF's <extra_info>.

    Idempotent: existing <aabb> elements are removed before the new one is
    appended. Preserves all other comments and whitespace in the URDF.
    Raises if the URDF has no <extra_info> element.
    """
    path = Path(urdf_path)
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(str(path), parser)
    extra = tree.find(".//extra_info")
    if extra is None:
        raise ValueError(f"URDF {path} has no <extra_info> element")
    for old in extra.findall("aabb"):
        extra.remove(old)
    aabb_elem = etree.SubElement(extra, "aabb")
    min_elem = etree.SubElement(aabb_elem, "min")
    max_elem = etree.SubElement(aabb_elem, "max")
    min_elem.text = " ".join(f"{v:.6f}" for v in aabb_min)
    max_elem.text = " ".join(f"{v:.6f}" for v in aabb_max)
    tree.write(
        str(path),
        pretty_print=True,
        encoding="utf-8",
        xml_declaration=True,
    )
