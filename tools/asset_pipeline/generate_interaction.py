#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

from pxr import Gf, Usd, UsdGeom

SUPPORTED_EXTS = (".usd", ".usdc")


def to_list(v: Gf.Vec3d):
    return [float(v[0]), float(v[1]), float(v[2])]


def default_doc():
    return {
        "interaction": {
            "active": {"place": {"body": []}},
            "passive": {"pick": {"body": []}},
        }
    }


def _make_bbox_cache():
    return UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        useExtentsHint=True,
    )


def get_units_resolve_xform(prim: Usd.Prim) -> Gf.Matrix4d:
    stage = prim.GetStage()
    pseudo = stage.GetPseudoRoot().GetPath()
    chain: List[Usd.Prim] = []
    p = prim
    while p and p.IsValid() and p.GetPath() != pseudo:
        chain.append(p)
        p = p.GetParent()
    chain.reverse()
    M = Gf.Matrix4d(1.0)
    for p in chain:
        order_attr = p.GetAttribute("xformOpOrder")
        tokens = (
            order_attr.Get()
            if order_attr and order_attr.HasAuthoredValueOpinion()
            else []
        )
        if tokens:
            names = [
                str(t)
                for t in tokens
                if str(t).endswith(":unitsResolve")
                or str(t).startswith("xformOp:orient")
            ]
        else:
            names = [
                a.GetName()
                for a in p.GetAttributes()
                if a.GetName().endswith(":unitsResolve")
                or a.GetName().startswith("xformOp:orient")
            ]
        for name in names:
            attr = p.GetAttribute(name)
            if not attr:
                continue
            val = attr.Get()
            if val is None:
                continue
            if name.startswith("xformOp:scale"):
                s = Gf.Vec3d(float(val[0]), float(val[1]), float(val[2]))
                S = Gf.Matrix4d(1.0)
                S.SetScale(s)
                M = M * S
            elif "rotateX:unitsResolve" in name:
                angle = float(val)
                R = Gf.Matrix4d(1.0)
                R.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), angle))
                M = M * R
            elif "rotateY:unitsResolve" in name:
                angle = float(val)
                R = Gf.Matrix4d(1.0)
                R.SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), angle))
                M = M * R
            elif "rotateZ:unitsResolve" in name:
                angle = float(val)
                R = Gf.Matrix4d(1.0)
                R.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), angle))
                M = M * R
            elif name.startswith("xformOp:orient"):
                if isinstance(val, Gf.Quatd):
                    q = val
                else:
                    q = Gf.Quatd(
                        float(val[0]),
                        Gf.Vec3d(float(val[1]), float(val[2]), float(val[3])),
                    )
                R = Gf.Matrix4d(1.0)
                R.SetRotate(Gf.Rotation(q))
                M = M * R
    return M


def get_local_center(prim: Usd.Prim) -> Gf.Vec3d:
    cache = _make_bbox_cache()
    bbox = cache.ComputeLocalBound(prim)
    rng = bbox.ComputeAlignedRange()
    mn = rng.GetMin()
    mx = rng.GetMax()
    center = (mn + mx) * 0.5
    M = get_units_resolve_xform(prim)
    c = M.Transform(center)
    return Gf.Vec3d(c[0], c[1], c[2])


def get_local_extents(prim: Usd.Prim) -> tuple[float, float, float]:
    cache = _make_bbox_cache()
    bbox = cache.ComputeLocalBound(prim)
    rng = bbox.ComputeAlignedRange()
    mn = rng.GetMin()
    mx = rng.GetMax()
    corners = []
    for x in (mn[0], mx[0]):
        for y in (mn[1], mx[1]):
            for z in (mn[2], mx[2]):
                corners.append(Gf.Vec3d(float(x), float(y), float(z)))
    M = get_units_resolve_xform(prim)
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for c in corners:
        t = M.Transform(c)
        xs.append(float(t[0]))
        ys.append(float(t[1]))
        zs.append(float(t[2]))
    dx = max(xs) - min(xs) if xs else 0.0
    dy = max(ys) - min(ys) if ys else 0.0
    dz = max(zs) - min(zs) if zs else 0.0
    return dx, dy, dz


def build_interaction_multi(
    center_local: Gf.Vec3d,
    place_dir_local: Gf.Vec3d,
    place_ref_local: Gf.Vec3d,
    passive_triplets_local: List[Dict[str, Any]],
):
    out = default_doc()
    place_body = {
        "xyz": to_list(center_local),
        "direction": to_list(place_dir_local),
        "ref_frame": to_list(place_ref_local),
    }
    out["interaction"]["active"]["place"]["body"].append(place_body)

    for t in passive_triplets_local:
        d: Gf.Vec3d = t["direction"]
        r: Gf.Vec3d = t["ref"]
        dir_label: str = t.get("dir_label", "")
        ref_label: str = t.get("ref_label", "")
        out["interaction"]["passive"]["pick"]["body"].append(
            {
                "xyz": to_list(center_local),
                "direction": to_list(d),
                "ref_frame": to_list(r),
                "dir_axis": dir_label,
                "ref_axis": ref_label,
            }
        )
    return out


def _collect_mesh_points_local(root: Usd.Prim) -> List[Gf.Vec3d]:
    xform_cache = UsdGeom.XformCache()
    root_world = xform_cache.GetLocalToWorldTransform(root)
    inv_root_world = root_world.GetInverse()
    out: List[Gf.Vec3d] = []

    def _recurse(p: Usd.Prim):
        if p.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(p)
            pts = mesh.GetPointsAttr().Get()
            if pts:
                mesh_world = xform_cache.GetLocalToWorldTransform(p)
                rel = inv_root_world * mesh_world
                for q in pts:
                    v3 = Gf.Vec3d(float(q[0]), float(q[1]), float(q[2]))
                    lv3 = rel.Transform(v3)
                    out.append(lv3)
        for c in p.GetChildren():
            _recurse(c)

    _recurse(root)
    return out


def _mesh_centroid(root: Usd.Prim) -> Gf.Vec3d | None:
    pts = _collect_mesh_points_local(root)
    if not pts:
        return None
    sx = sy = sz = 0.0
    for v in pts:
        sx += v[0]
        sy += v[1]
        sz += v[2]
    n = float(len(pts))
    c = Gf.Vec3d(sx / n, sy / n, sz / n)
    M = get_units_resolve_xform(root)
    t = M.Transform(c)
    return Gf.Vec3d(t[0], t[1], t[2])


def _collect_mesh_points_resolved(root: Usd.Prim) -> List[Gf.Vec3d]:
    pts = _collect_mesh_points_local(root)
    if not pts:
        return []
    M = get_units_resolve_xform(root)
    out: List[Gf.Vec3d] = []
    for p in pts:
        t = M.Transform(p)
        out.append(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
    return out


def _obb_center(root: Usd.Prim) -> Gf.Vec3d | None:
    # Prim-local bound center behaves like an object-aligned bbox center.
    return get_local_center(root)


def _bbox_center(root: Usd.Prim) -> Gf.Vec3d | None:
    pts = _collect_mesh_points_resolved(root)
    if not pts:
        return None
    xs = [float(v[0]) for v in pts]
    ys = [float(v[1]) for v in pts]
    zs = [float(v[2]) for v in pts]
    return Gf.Vec3d(
        (min(xs) + max(xs)) * 0.5,
        (min(ys) + max(ys)) * 0.5,
        (min(zs) + max(zs)) * 0.5,
    )


def _iter_supported_usd_files(path: str) -> List[str]:
    files: List[str] = []
    for ext in SUPPORTED_EXTS:
        pattern = os.path.join(path, f"**/*{ext}")
        for f in glob.glob(pattern, recursive=True):
            if os.path.isfile(f):
                files.append(f)
    return sorted(set(files))


def _is_supported_usd_file(path: str) -> bool:
    lower = path.lower()
    return any(lower.endswith(ext) for ext in SUPPORTED_EXTS)


def _count_meshes(root: Usd.Prim) -> int:
    count = 0
    stack = [root]
    while stack:
        prim = stack.pop()
        if prim.IsA(UsdGeom.Mesh):
            count += 1
        stack.extend(list(prim.GetChildren()))
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "input",
        nargs="+",
        help="USD/USDC file(s) or directory containing asset files",
    )
    ap.add_argument(
        "--root_prim",
        default=None,
        help="Target prim path; defaults to stage's defaultPrim",
    )
    ap.add_argument(
        "--out",
        default=".",
        help=("Output directory or file. Use '.' to output JSON next to USD."),
    )
    ap.add_argument(
        "--gripper_width",
        type=float,
        default=0.09,
        help=(
            "Gripper width. For each approach dir and chosen ref axis, we "
            "check the object's extent along the third, orthogonal axis "
            "(perpendicular to both dir and ref). Only when that extent is "
            "<= width, the annotation will be emitted."
        ),
    )
    ap.add_argument(
        "--center_mode",
        choices=("obb", "bbox", "centroid"),
        default="obb",
        help=(
            "Center used for active.place xyz and passive.pick xyz. "
            "'obb' uses the prim-local oriented box center, "
            "'bbox' uses the resolved axis-aligned bounding-box center, "
            "and 'centroid' uses the mesh-vertex centroid."
        ),
    )

    args = ap.parse_args()

    usd_files = []

    for path in args.input:
        if os.path.isdir(path):
            usd_files.extend(_iter_supported_usd_files(path))
        elif _is_supported_usd_file(path):
            usd_files.append(path)
        else:
            print(f"[WARN] skip unsupported file: {path}")

    usd_files = sorted(set(usd_files))

    if not usd_files:
        print("No .usd/.usdc files found.", file=sys.stderr)
        sys.exit(2)

    total_count = len(usd_files)
    success_count = 0
    fail_count = 0
    total_passive_pick_count = 0

    for usd_path in usd_files:
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print(f"Failed to open USD: {usd_path}", file=sys.stderr)
            fail_count += 1
            continue

        prim = (
            stage.GetDefaultPrim()
            if args.root_prim is None
            else stage.GetPrimAtPath(args.root_prim)
        )
        if not prim or not prim.IsValid():
            print(f"ERROR: invalid root prim for {usd_path}", file=sys.stderr)
            fail_count += 1
            continue

        mesh_count = _count_meshes(prim)

        obb_center = _obb_center(prim)
        dx, dy, dz = get_local_extents(prim)
        center = obb_center or get_local_center(prim)
        if args.center_mode == "obb":
            center = obb_center or center
        elif args.center_mode == "bbox":
            center = _bbox_center(prim) or center
        elif args.center_mode == "centroid":
            center = _mesh_centroid(prim) or center

        Xl = Gf.Vec3d(1, 0, 0)
        Yl = Gf.Vec3d(0, 1, 0)
        Zl = Gf.Vec3d(0, 0, 1)

        passive_triplets: List[Dict[str, Any]] = []

        extents_by_axis: Dict[str, float] = {"x": dx, "y": dy, "z": dz}
        axis_vec: Dict[str, Gf.Vec3d] = {"x": Xl, "y": Yl, "z": Zl}

        approach_map = {
            "-x": {
                "vec": Gf.Vec3d(-Xl[0], -Xl[1], -Xl[2]),
                "refs": ["y", "z"],
            },
            "-y": {
                "vec": Gf.Vec3d(-Yl[0], -Yl[1], -Yl[2]),
                "refs": ["x", "z"],
            },
            "-z": {
                "vec": Gf.Vec3d(-Zl[0], -Zl[1], -Zl[2]),
                "refs": ["x", "y"],
            },
        }

        def _third_axis(a: str, b: str) -> str:
            for c_axis in ("x", "y", "z"):
                if c_axis != a and c_axis != b:
                    return c_axis
            return "x"

        for dir_label, cfg in approach_map.items():
            approach_vec = cfg["vec"]
            dir_axis = dir_label[-1]
            for ref_axis in cfg["refs"]:
                grip_axis = _third_axis(dir_axis, ref_axis)
                extent = extents_by_axis[grip_axis]
                if extent <= args.gripper_width:
                    ref_v = axis_vec[ref_axis]
                    ref_label = f"+{ref_axis}"
                    passive_triplets.append(
                        {
                            "direction": approach_vec,
                            "ref": ref_v,
                            "dir_label": dir_label,
                            "ref_label": ref_label,
                        }
                    )

        place_dir = Gf.Vec3d(-Zl[0], -Zl[1], -Zl[2])
        if dy <= args.gripper_width:
            place_ref = Xl
        elif dx <= args.gripper_width:
            place_ref = Yl
        else:
            place_ref = Xl if dy <= dx else Yl

        data = build_interaction_multi(
            center_local=center,
            place_dir_local=place_dir,
            place_ref_local=place_ref,
            passive_triplets_local=passive_triplets,
        )

        if args.out.endswith(".json"):
            out_path = args.out
        else:
            usd_dir = os.path.dirname(os.path.abspath(usd_path))
            out_path = os.path.join(usd_dir, "interaction.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        passive_pick_count = len(
            data["interaction"]["passive"]["pick"]["body"]
        )
        total_passive_pick_count += passive_pick_count

        if passive_pick_count > 0:
            success_count += 1
            status = "OK"
        else:
            fail_count += 1
            status = "FAIL"

        print(
            f"[{status}] wrote {out_path} "
            f"(input={usd_path}, root={prim.GetPath()}, meshes={mesh_count}, "
            f"passive_pick_body={passive_pick_count})"
        )

    print(
        "[SUMMARY] "
        f"processed={total_count}, "
        f"succeeded={success_count}, "
        f"failed={fail_count}, "
        f"passive_pick_body_total={total_passive_pick_count}"
    )


if __name__ == "__main__":
    main()
