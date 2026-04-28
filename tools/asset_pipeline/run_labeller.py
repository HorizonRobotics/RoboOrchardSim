#!/usr/bin/env python3
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

"""Command-line entry script for the AssetLabeller.

Single asset:
    python3 tools/asset_pipeline/run_labeller.py \
        --mesh path/to/mesh.obj \
        --output-root outputs/asset001 \
        --category mug

Batch (folder of assets):
    python3 tools/asset_pipeline/run_labeller.py \
        --input-dir assets_usd/ \
        --output-root outputs/labelled \
        --views 4 \
        --format usd

    Expected folder layout for --input-dir:
        assets_usd/
        |- cup/
        |  |- mesh.usd        (or .obj, .usda, .glb, etc.)
        |  `- texture.png
        |- bottle/
        |  `- mesh.obj
        `- plate/
           `- scene.usda

    The scanner walks recursively. For each directory that contains at
    least one supported mesh, the first matching mesh file is processed.
    The leaf directory name is used as the category hint, while the
    relative directory path is preserved under --output-root to avoid
    name collisions.

Requires:
    - GPT config at tools/asset_pipeline/configs/gpt_config.yaml
      (can be overridden via --config).
    - Dependencies in tools/asset_pipeline/requirements.txt.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_labeller")

SUPPORTED_EXTS = {
    ".obj",
    ".ply",
    ".glb",
    ".gltf",
    ".usd",
    ".usda",
    ".usdc",
    ".usdz",
}

FORMAT_GROUPS = {
    "usd": [".usd", ".usda", ".usdc", ".usdz"],
    "obj": [".obj"],
    "glb": [".glb", ".gltf"],
    "ply": [".ply"],
}

IGNORED_SCAN_DIRS = {
    "mesh",
    "renders",
    "__pycache__",
}


def _add_project_paths() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


_add_project_paths()

from asset_labeller.labeller import normalize_category_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AssetLabeller on one or many 3D assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--mesh",
        type=str,
        help="Path to a single mesh file (.obj / .usd / etc.).",
    )
    source.add_argument(
        "--input-dir",
        type=str,
        help=(
            "Directory of assets. Recursively finds asset folders; each "
            "directory containing a supported mesh is treated as one asset."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output root directory for URDF, mesh and renders.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="unknown",
        help="Category hint (single-asset mode only, default: unknown).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tools/asset_pipeline/configs/gpt_config.yaml",
        help="Path to GPT config YAML file.",
    )
    parser.add_argument(
        "--views",
        type=int,
        default=6,
        help="Number of rendered views (default: 6).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=(512, 512),
        help="Render resolution, width height (default: 512 512).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        choices=list(FORMAT_GROUPS.keys()),
        help=(
            "Preferred mesh format when a subfolder contains multiple "
            "formats (e.g., both .obj and .usd). "
            "Choices: usd, obj, glb, ply. Default: pick first found."
        ),
    )
    parser.add_argument(
        "--no-check-connection",
        action="store_true",
        help="Do not run GPT connection check at startup.",
    )
    parser.add_argument(
        "--strict-textures",
        action="store_true",
        help=(
            "Treat texture audit findings (e.g. USD declares textures "
            "but produced .mtl has no map_Kd) as hard errors. Default "
            "is to log a warning and continue."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Pre-flight texture audit only: scan each input USD and "
            "report assets whose textures would not survive export. "
            "Skips GPT, rendering, URDF generation. Useful before "
            "spending GPT credit on a fresh batch."
        ),
    )
    return parser.parse_args()


def asset_uuid_from_mesh_path(mesh_path: str) -> str:
    """Build a stable UUID from the asset's absolute directory path."""
    asset_abs_dir = os.path.realpath(os.path.dirname(mesh_path))
    return uuid.uuid5(uuid.NAMESPACE_URL, asset_abs_dir).hex


def infer_taxonomy_from_asset_relpath(asset_relpath: str) -> tuple[str, str]:
    """Infer ``(domain, super_category)`` from a relative asset path."""
    parts = [p for p in asset_relpath.split(os.sep) if p and p != "."]
    if len(parts) >= 3:
        return parts[0], parts[1]
    return "unknown", "unknown"


def infer_taxonomy_from_asset_dir(asset_dir: str) -> tuple[str, str]:
    """Infer ``(domain, super_category)`` from an absolute asset directory."""
    parts = [p for p in os.path.normpath(asset_dir).split(os.sep) if p]
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    return "unknown", "unknown"


def scan_assets(
    input_dir: str, prefer_format: str = None
) -> list[tuple[str, str, str]]:
    """Recursively scan *input_dir* and return asset records.

    Returns:
        A list of ``(mesh_path, category, asset_relpath)`` tuples.

    Any directory containing at least one supported mesh is treated as
    one asset. The leaf directory name is normalized and used as the
    category hint.
    ``asset_relpath`` is the path of that asset directory relative to
    ``input_dir`` and is used for the output directory structure.

    When *prefer_format* is set (e.g. ``"usd"``), files matching that
    format group are selected first. If no preferred file is found, the
    first file with any supported extension is used as fallback.
    """
    preferred_exts = set()
    if prefer_format is not None:
        preferred_exts = set(FORMAT_GROUPS.get(prefer_format, []))

    results = []
    for sub_dir, dirnames, filenames in os.walk(input_dir):
        dirnames.sort()
        dirnames[:] = [d for d in dirnames if d not in IGNORED_SCAN_DIRS]
        preferred_hit = None
        fallback_hit = None
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            if fallback_hit is None:
                fallback_hit = fname
            if (
                preferred_exts
                and ext in preferred_exts
                and preferred_hit is None
            ):
                preferred_hit = fname

        chosen = preferred_hit or fallback_hit
        if chosen is not None:
            asset_relpath = os.path.relpath(sub_dir, input_dir)
            category = normalize_category_label(os.path.basename(sub_dir))
            results.append(
                (os.path.join(sub_dir, chosen), category, asset_relpath)
            )

    return results


def label_single(
    labeller,
    mesh_path: str,
    output_root: str,
    category: str,
    domain: str = "unknown",
    super_category: str = "unknown",
    copy_source: bool = False,
) -> dict:
    """Run the labeller on a single asset and return a result dict."""
    input_dir = os.path.abspath(os.path.dirname(mesh_path))
    output_root = os.path.abspath(output_root)
    asset_uuid = asset_uuid_from_mesh_path(mesh_path)
    if output_root != input_dir:
        for generated_name in ("mesh", "renders"):
            generated_path = os.path.join(output_root, generated_name)
            if os.path.isdir(generated_path):
                shutil.rmtree(generated_path)

    urdf_path = labeller(
        mesh_path=mesh_path,
        output_root=output_root,
        category=category,
        uuid=asset_uuid,
        domain=domain,
        super_category=super_category,
    )

    if copy_source and os.path.isfile(mesh_path):
        dest = os.path.join(output_root, os.path.basename(mesh_path))
        if os.path.abspath(mesh_path) != os.path.abspath(dest):
            shutil.copy2(mesh_path, dest)

        src_textures_dir = os.path.join(os.path.dirname(mesh_path), "textures")
        dst_textures_dir = os.path.join(output_root, "textures")
        if os.path.isdir(src_textures_dir) and os.path.abspath(
            src_textures_dir
        ) != os.path.abspath(dst_textures_dir):
            os.makedirs(output_root, exist_ok=True)
            shutil.copytree(
                src_textures_dir,
                dst_textures_dir,
                dirs_exist_ok=True,
            )

    return {
        "mesh": mesh_path,
        "urdf": urdf_path,
        "attrs": dict(labeller.estimated_attrs),
        "uuid": asset_uuid,
        "status": "ok",
    }


def _run_check_only(args: argparse.Namespace) -> None:
    """Pre-flight texture audit; no GPT, no rendering, no URDF generation.

    For each input USD, the audit runs ``export_usd_to_obj_with_materials``
    into a temp directory and inspects the produced .mtl. Findings are
    printed and tallied; no files in the source or output_root are
    written.
    """
    import tempfile

    from asset_labeller.mesh_utils import (
        audit_export_textures,
        export_usd_to_obj_with_materials,
    )

    if args.mesh:
        sources = [(os.path.abspath(args.mesh), os.path.basename(args.mesh))]
    else:
        input_dir = os.path.abspath(args.input_dir)
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        scanned = scan_assets(input_dir, prefer_format=args.format)
        sources = [(mp, rel) for mp, _cat, rel in scanned]

    if not sources:
        logger.warning("No USD assets found to check.")
        return

    usd_exts = {".usd", ".usda", ".usdc", ".usdz"}
    code_counts: dict[str, int] = {}
    findings_by_asset: list[tuple[str, list[tuple[str, str]]]] = []

    with tempfile.TemporaryDirectory(prefix="precheck_") as td:
        for idx, (mesh_path, label) in enumerate(sources, 1):
            ext = os.path.splitext(mesh_path)[1].lower()
            if ext not in usd_exts:
                logger.info(f"[{idx}/{len(sources)}] {label}  (skip: not USD)")
                continue
            tmp_obj = os.path.join(
                td,
                f"{idx:04d}_{os.path.splitext(os.path.basename(mesh_path))[0]}.obj",
            )
            try:
                copied = export_usd_to_obj_with_materials(mesh_path, tmp_obj)
                issues = audit_export_textures(mesh_path, tmp_obj, copied)
            except Exception as e:
                issues = [("EXPORT_RAISED", f"{type(e).__name__}: {e}")]
            for code, _ in issues:
                code_counts[code] = code_counts.get(code, 0) + 1
            if issues:
                findings_by_asset.append((label, issues))
                logger.warning(
                    f"[{idx}/{len(sources)}] {label}  issues="
                    f"{[c for c, _ in issues]}"
                )
            else:
                logger.info(f"[{idx}/{len(sources)}] {label}  clean")

    print()
    print(f"Pre-flight check complete: {len(sources)} USD asset(s) inspected")
    print(f"  clean:    {len(sources) - len(findings_by_asset)}")
    print(f"  with issues: {len(findings_by_asset)}")
    for code, n in sorted(code_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {code}: {n}")
    if findings_by_asset and not args.strict_textures:
        print()
        print("Re-run with --strict-textures to make these block labelling.")


def main() -> None:
    from asset_labeller import AssetLabeller, load_client_from_config

    args = parse_args()
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    # Pre-flight check-only mode: scan inputs for texture audit
    # findings without spending any GPT credit.
    if args.check_only:
        return _run_check_only(args)

    client = load_client_from_config(
        args.config,
        check_connection=not args.no_check_connection,
    )

    copy_source = args.format in ("usd",)

    labeller = AssetLabeller(
        gpt_client=client,
        render_view_num=args.views,
        render_resolution=tuple(args.resolution),
        strict_textures=args.strict_textures,
    )

    # ---- Single-asset mode ----
    if args.mesh:
        mesh_path = os.path.abspath(args.mesh)
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        result = label_single(
            labeller,
            mesh_path,
            output_root,
            normalize_category_label(args.category),
            *infer_taxonomy_from_asset_dir(os.path.dirname(mesh_path)),
            copy_source=copy_source,
        )
        print(f"[AssetLabeller] Generated URDF: {result['urdf']}")
        print(f"[AssetLabeller] Attributes: {result['attrs']}")
        return

    # ---- Batch mode ----
    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    assets = scan_assets(input_dir, prefer_format=args.format)
    if not assets:
        logger.warning(f"No assets found in {input_dir}")
        return

    logger.info(f"Found {len(assets)} assets in {input_dir}")

    all_results = {}
    ok, fail = 0, 0
    for idx, (mesh_path, category, asset_relpath) in enumerate(assets, 1):
        asset_name = asset_relpath
        asset_output = os.path.join(output_root, asset_relpath)
        domain, super_category = infer_taxonomy_from_asset_relpath(
            asset_relpath
        )
        logger.info(f"[{idx}/{len(assets)}] {asset_name}  ({mesh_path})")

        try:
            result = label_single(
                labeller,
                mesh_path,
                asset_output,
                category,
                domain=domain,
                super_category=super_category,
                copy_source=copy_source,
            )
            all_results[asset_name] = result
            ok += 1
            logger.info(f"  -> URDF: {result['urdf']}")
        except Exception as e:
            all_results[asset_name] = {
                "mesh": mesh_path,
                "status": "error",
                "error": str(e),
            }
            fail += 1
            logger.error(f"  -> Failed: {e}")

    summary_path = os.path.join(
        "outputs", "labeller_runs", "label_summary.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Done. {ok} succeeded, {fail} failed. Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
