# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""Rich visualization for a packed LeRobot v2 dataset.

What this renders per frame
---------------------------
Top row    : RGB from every color camera (horizontally concatenated).
Bottom row : depth from the same cameras (BWR colormap, [0.01, 1.2] m), same
             horizontal order so they line up with the RGB row.

If a URDF was copied into the dataset (meta/robot.urdf) and per-frame camera
extrinsics are available, the RGB row additionally draws each robot joint's
local XYZ coordinate frame (red=X, green=Y, blue=Z) projected into the
camera image.  End-effector joints get a slightly larger axis and the
current gripper value drawn next to the origin.

Implementation notes
--------------------
* URDF parsing is hand-rolled with xml.etree.ElementTree, no external
  ``urdfpy`` / ``kinpy`` dependency.  Only FK-relevant fields are read;
  meshes are ignored.
* FK is a tiny DFS from the URDF root link.
* Depth colorize uses a hand-built 256-entry BWR LUT (no matplotlib).
* mp4 output via cv2.VideoWriter.
"""

from __future__ import annotations
import json
import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URDF parsing + FK
# ---------------------------------------------------------------------------


@dataclass
class _Joint:
    name: str
    type: str  # "revolute" | "continuous" | "prismatic" | "fixed"
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray  # unit vector


@dataclass
class KinematicTree:
    root_link: str
    links: list[str] = field(default_factory=list)
    joints: list[_Joint] = field(default_factory=list)
    joints_by_name: dict[str, _Joint] = field(default_factory=dict)
    children_of_link: dict[str, list[_Joint]] = field(default_factory=dict)


def _parse_xyz(text: str | None) -> np.ndarray:
    if text is None:
        return np.zeros(3, dtype=np.float64)
    return np.array([float(x) for x in text.strip().split()], dtype=np.float64)


def parse_urdf(urdf_text: str) -> KinematicTree:
    """Parse a URDF XML string into a KinematicTree."""
    root = ET.fromstring(urdf_text)
    links: list[str] = [L.get("name") or "" for L in root.findall("link")]

    joints: list[_Joint] = []
    for j in root.findall("joint"):
        name = j.get("name") or ""
        jtype = j.get("type") or "fixed"
        parent_el = j.find("parent")
        child_el = j.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.get("link") or ""
        child = child_el.get("link") or ""

        origin = j.find("origin")
        origin_xyz = _parse_xyz(
            origin.get("xyz") if origin is not None else None
        )
        origin_rpy = _parse_xyz(
            origin.get("rpy") if origin is not None else None
        )

        axis_el = j.find("axis")
        axis = _parse_xyz(axis_el.get("xyz") if axis_el is not None else None)
        if np.linalg.norm(axis) < 1e-12:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = axis / np.linalg.norm(axis)

        joints.append(
            _Joint(
                name=name,
                type=jtype,
                parent=parent,
                child=child,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis=axis,
            )
        )

    # Root link = a link that is never a child of any joint.
    child_set = {jj.child for jj in joints}
    roots = [L for L in links if L not in child_set]
    if not roots:
        raise RuntimeError("URDF has no root link.")
    root_link = roots[0]
    if len(roots) > 1:
        logger.warning(
            "URDF has multiple root links %s; using %r.", roots, root_link
        )

    tree = KinematicTree(root_link=root_link, links=links, joints=joints)
    tree.joints_by_name = {jj.name: jj for jj in joints}
    for jj in joints:
        tree.children_of_link.setdefault(jj.parent, []).append(jj)
    return tree


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + ax * ax * C, ax * ay * C - az * s, ax * az * C + ay * s],
            [ay * ax * C + az * s, c + ay * ay * C, ay * az * C - ax * s],
            [az * ax * C - ay * s, az * ay * C + ax * s, c + az * az * C],
        ]
    )


def _joint_transform(j: _Joint, value: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(j.origin_rpy)
    T[:3, 3] = j.origin_xyz
    if j.type in ("revolute", "continuous"):
        Rj = _axis_angle_to_matrix(j.axis, value)
        Tj = np.eye(4)
        Tj[:3, :3] = Rj
        T = T @ Tj
    elif j.type == "prismatic":
        Tj = np.eye(4)
        Tj[:3, 3] = j.axis * value
        T = T @ Tj
    return T


def forward_kinematics(
    tree: KinematicTree,
    joint_values: dict[str, float],
    base_link: str | None = None,
) -> dict[str, np.ndarray]:
    """Compute the 4x4 transform of every link relative to base_link."""
    base = base_link or tree.root_link
    poses: dict[str, np.ndarray] = {base: np.eye(4)}
    stack: list[str] = [base]
    while stack:
        link = stack.pop()
        T_world_parent = poses[link]
        for j in tree.children_of_link.get(link, []):
            value = float(joint_values.get(j.name, 0.0))
            T_parent_child = _joint_transform(j, value)
            T_world_child = T_world_parent @ T_parent_child
            poses[j.child] = T_world_child
            stack.append(j.child)
    return poses


# ---------------------------------------------------------------------------
# projection helpers
# ---------------------------------------------------------------------------


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """[x, y, z, w] quaternion -> 3x3 rotation matrix."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )


def _cam_inverse(xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Mcap FrameTransform (parent->child) -> T_cam_parent (4x4).

    Points in parent frame, left-multiplied by this matrix, end up in the
    camera frame.
    """
    R = _quat_xyzw_to_matrix(quat_xyzw)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ xyz
    return T


def _project_points(
    points_world: np.ndarray,
    T_cam_world: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) world points -> ((N, 2) int pixels, (N,) depth)."""
    N = points_world.shape[0]
    ones = np.ones((N, 1), dtype=points_world.dtype)
    homog = np.concatenate([points_world, ones], axis=1)
    cam = (T_cam_world @ homog.T).T[:, :3]
    depth = cam[:, 2]
    safe = np.where(np.abs(depth) > 1e-6, depth, 1e-6)
    u = K[0, 0] * (cam[:, 0] / safe) + K[0, 2]
    v = K[1, 1] * (cam[:, 1] / safe) + K[1, 2]
    pts2 = np.stack([u, v], axis=1).astype(np.int32)
    return pts2, depth


# ---------------------------------------------------------------------------
# overlay drawing
# ---------------------------------------------------------------------------

_AXIS_BGR = [
    (0, 0, 255),  # X = red
    (0, 255, 0),  # Y = green
    (255, 0, 0),  # Z = blue
]


def _draw_link_axes(
    img_bgr: np.ndarray,
    T_world_link: np.ndarray,
    T_cam_world: np.ndarray,
    K: np.ndarray,
    axis_len: float,
    label: str | None = None,
) -> None:
    """Draw the link's local XYZ axes onto img_bgr in-place."""
    origin_w = T_world_link[:3, 3]
    R = T_world_link[:3, :3]
    pts_world = np.stack(
        [
            origin_w + R @ np.array([axis_len, 0, 0]),
            origin_w + R @ np.array([0, axis_len, 0]),
            origin_w + R @ np.array([0, 0, axis_len]),
            origin_w,
        ],
        axis=0,
    )
    pts2, depth = _project_points(pts_world, T_cam_world, K)
    if depth[3] < 0.02:
        return  # origin behind camera

    h, w = img_bgr.shape[:2]
    o = tuple(pts2[3])
    if not (0 <= o[0] < w and 0 <= o[1] < h):
        return

    for ax in range(3):
        tip = tuple(pts2[ax])
        cv2.line(img_bgr, o, tip, _AXIS_BGR[ax], 2)
        cv2.circle(img_bgr, tip, 4, _AXIS_BGR[ax], -1)

    if label:
        cv2.putText(
            img_bgr,
            label,
            (o[0] + 5, o[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )


def _draw_joint_table(
    img_rgb: np.ndarray,
    state: np.ndarray,
    rows: list[tuple[str, int, int]],
    origin: tuple[int, int] = (10, 60),
) -> None:
    """Overlay a joint-state table in-place.

    ``rows`` is a list of ``(row_label, state_offset, state_length)``. The
    last column of each row is rendered as the gripper ('g'); preceding
    columns are 'j1', 'j2', ...  Useful for both single-arm (e.g. franka:
    ``[("arm", 0, 8)]``) and dual-arm (e.g. piperx:
    ``[("L", 0, 7), ("R", 7, 7)]``) layouts.

    Renders nothing if ``rows`` is empty or the state is too short.
    """
    if not rows or state.size == 0:
        return
    max_len = max(length for _, _, length in rows)
    if max_len <= 0:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    color = (255, 255, 255)
    x0, y0 = origin
    col_w = 72
    row_h = 22
    # header: j1, j2, ..., j(max_len-1), g
    headers = [f"j{i + 1}" for i in range(max_len - 1)] + ["g"]
    for i, lab in enumerate(headers):
        cv2.putText(
            img_rgb,
            lab,
            (x0 + 30 + i * col_w, y0),
            font,
            scale,
            color,
            thick,
            cv2.LINE_AA,
        )
    # rows
    for r, (tag, off, length) in enumerate(rows):
        if off + length > state.size:
            continue
        y = y0 + (r + 1) * row_h
        cv2.putText(
            img_rgb,
            tag,
            (x0, y),
            font,
            scale,
            color,
            thick,
            cv2.LINE_AA,
        )
        for i in range(length):
            v = float(state[off + i])
            cv2.putText(
                img_rgb,
                f"{v:+.2f}",
                (x0 + 30 + i * col_w, y),
                font,
                scale,
                color,
                thick,
                cv2.LINE_AA,
            )


def _wrap_text_by_pixel_width(
    text: str,
    max_px: int,
    font: int,
    scale: float,
    thick: int,
) -> list[str]:
    """Greedy word-wrap into lines that fit within ``max_px`` pixels."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        candidate = cur + " " + w
        (cw, _), _ = cv2.getTextSize(candidate, font, scale, thick)
        if cw <= max_px:
            cur = candidate
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _draw_instruction(
    img_rgb: np.ndarray,
    instruction: str,
    bottom_margin: int = 12,
    side_margin: int = 10,
) -> None:
    """Draw a wrapped instruction string at the bottom-left in-place.

    Adds a translucent dark band behind the text so it stays readable on top
    of any scene background.
    """
    if not instruction:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 1
    h, w = img_rgb.shape[:2]
    max_px = w - 2 * side_margin
    lines = _wrap_text_by_pixel_width(instruction, max_px, font, scale, thick)
    if not lines:
        return
    (_, line_h), baseline = cv2.getTextSize("Ay", font, scale, thick)
    line_step = line_h + baseline + 4
    n = len(lines)
    band_top = h - bottom_margin - n * line_step - 4
    band_bot = h - bottom_margin + baseline
    band_top = max(0, band_top)
    overlay = img_rgb.copy()
    cv2.rectangle(
        overlay,
        (0, band_top),
        (w, band_bot),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.55, img_rgb, 0.45, 0, dst=img_rgb)
    for i, line in enumerate(lines):
        y = band_top + (i + 1) * line_step - baseline
        cv2.putText(
            img_rgb,
            line,
            (side_margin, y),
            font,
            scale,
            (255, 255, 255),
            thick,
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# depth colorize
# ---------------------------------------------------------------------------


def _bwr_lut() -> np.ndarray:
    """256x3 BGR lookup table approximating matplotlib 'bwr' reversed."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            k = t / 0.5
            b = 255
            g = int(255 * k)
            r = int(255 * k)
        else:
            k = (t - 0.5) / 0.5
            b = int(255 * (1 - k))
            g = int(255 * (1 - k))
            r = 255
        lut[i] = (b, g, r)
    return lut[::-1]


_BWR_LUT = _bwr_lut()


def _depth_to_bgr(
    depth: np.ndarray, vmin: float = 0.01, vmax: float = 1.2
) -> np.ndarray:
    """Colorize depth -> (H, W, 3) BGR uint8.

    Accepts (H, W), (H, W, 1) (HWC), or (1, H, W) (CHW from LeRobot).
    """
    if depth.ndim == 3:
        if depth.shape[0] == 1 and depth.shape[-1] != 1:
            depth = depth[0]  # CHW (1, H, W) -> (H, W)
        else:
            depth = depth[:, :, 0]  # HWC (H, W, 1) -> (H, W)
    if depth.dtype in (np.uint16, np.int16):
        depth = depth.astype(np.float32) / 1000.0  # mm -> m
    elif depth.dtype != np.float32:
        depth = depth.astype(np.float32)

    mask = depth > 0
    norm = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)
    idx = (norm * 255).astype(np.int32)
    out = _BWR_LUT[idx]
    out[~mask] = 0
    return out


# ---------------------------------------------------------------------------
# frame assembly
# ---------------------------------------------------------------------------


def _to_uint8_rgb(img: Any) -> np.ndarray:
    """LeRobot image (tensor/ndarray/PIL) -> HxWx3 uint8 RGB."""
    try:
        import torch

        if isinstance(img, torch.Tensor):
            arr = img.cpu().numpy()
        else:
            arr = np.asarray(img)
    except ImportError:
        arr = np.asarray(img)

    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        if float(arr.max()) <= 1.001:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def _compose_grid(
    rgb_row: list[np.ndarray], depth_row: list[np.ndarray]
) -> np.ndarray:
    """Stack RGB row above depth row.

    If ``depth_row`` is empty, return the RGB row alone (used when a
    dataset was packed without depth features).
    """
    top = np.concatenate(rgb_row, axis=1)
    if not depth_row:
        return top
    bot = np.concatenate(depth_row, axis=1)
    if top.shape[1] != bot.shape[1]:
        w = min(top.shape[1], bot.shape[1])
        top = top[:, :w]
        bot = bot[:, :w]
    return np.concatenate([top, bot], axis=0)


# ---------------------------------------------------------------------------
# obs.state -> URDF joint mapping for dualarm_piperx
# ---------------------------------------------------------------------------

# state[i] -> list of (urdf_joint_name, factor).
# Gripper scalar fans out to BOTH finger prismatic joints; URDF axis already
# encodes the sign (joint7 axis = -Z, joint8 axis = +Z), so both fingers get
# the same magnitude (factor=0.5 of the un-scaled scalar; reader pre-divides
# by gripper_scale before this point).
PIPERX_STATE_TO_JOINTS: list[list[tuple[str, float]]] = [
    [("left_joint1", 1.0)],
    [("left_joint2", 1.0)],
    [("left_joint3", 1.0)],
    [("left_joint4", 1.0)],
    [("left_joint5", 1.0)],
    [("left_joint6", 1.0)],
    [("left_joint7", 0.5), ("left_joint8", 0.5)],
    [("right_joint1", 1.0)],
    [("right_joint2", 1.0)],
    [("right_joint3", 1.0)],
    [("right_joint4", 1.0)],
    [("right_joint5", 1.0)],
    [("right_joint6", 1.0)],
    [("right_joint7", 0.5), ("right_joint8", 0.5)],
]


def _state_to_joint_values(
    state: np.ndarray,
    state_to_joints: list[list[tuple[str, float]]],
    gripper_scale: float = 2.0,
) -> dict[str, float]:
    """Convert obs.state vector to {urdf_joint_name: value}.

    Gripper entries (len(mapping) > 1) are first divided by gripper_scale to
    undo the x2 applied at pack time, then distributed to finger joints.
    """
    out: dict[str, float] = {}
    for i, mapping in enumerate(state_to_joints):
        v = float(state[i])
        if len(mapping) > 1:
            v = v / gripper_scale
        for joint_name, factor in mapping:
            out[joint_name] = v * factor
    return out


PIPERX_EE_LINKS: tuple[str, ...] = ("left_ee_link", "right_ee_link")

PIPERX_DRAWN_LINKS: tuple[str, ...] = (
    "left_link1",
    "left_link2",
    "left_link3",
    "left_link4",
    "left_link5",
    "left_link6",
    "left_ee_link",
    "right_link1",
    "right_link2",
    "right_link3",
    "right_link4",
    "right_link5",
    "right_link6",
    "right_ee_link",
)


def _ee_gripper_value(state: np.ndarray, link: str) -> float:
    if "left" in link:
        return float(state[6]) if len(state) > 6 else 0.0
    if "right" in link:
        return float(state[13]) if len(state) > 13 else 0.0
    return 0.0


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def render_episode(
    ds,
    episode_index: int | list[int],
    out_path: Path,
    fps: int | None = None,
    with_overlay: bool = True,
    state_to_joints: list[list[tuple[str, float]]] | None = None,
    ee_links: tuple[str, ...] | None = None,
    drawn_links: tuple[str, ...] | None = None,
    base_link: str | None = None,
    gripper_scale: float | None = None,
) -> None:
    """Render one or more episodes of a LeRobotDataset into a single rich mp4.

    ``episode_index`` accepts either a single integer or a list of integers;
    when a list is given, all episodes are concatenated back-to-back through
    the same writer with no inter-episode separator.

    ``base_link`` is the frame in which the stored
    ``camera_pose.{cam}.xyz/quat``
    are expressed (mcap's normalized parent_frame_id, typically
    ``left_base_link``).  FK is always rooted at the URDF root; link poses are
    then transformed into this frame for projection.  If ``None``, the frame
    is read from
    ``info.json -> cameras.<first_cam>.extrinsic.parent_frame_id`` (last
    path segment).
    """
    root = Path(ds.root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    cameras_info = info.get("cameras", {})

    # Resolve viz config from the dataset's robot_type (writer/lerobot stores
    # this in info.json), then let any explicit kwargs override.  Falls back
    # to PIPERX_* legacy constants if the embodiment is unknown.
    robot_type = info.get("robot_type", "")
    try:
        from tools.mcap_to_lerobot.config_presets import EMBODIMENT_REGISTRY

        viz_cfg = EMBODIMENT_REGISTRY.get(robot_type)
    except ImportError:
        viz_cfg = None

    if state_to_joints is None:
        state_to_joints = (
            viz_cfg.viz_state_to_joints if viz_cfg else PIPERX_STATE_TO_JOINTS
        )
    if ee_links is None:
        ee_links = tuple(viz_cfg.viz_ee_links if viz_cfg else PIPERX_EE_LINKS)
    if drawn_links is None:
        drawn_links = tuple(
            viz_cfg.viz_drawn_links if viz_cfg else PIPERX_DRAWN_LINKS
        )
    if gripper_scale is None:
        gripper_scale = viz_cfg.viz_gripper_scale if viz_cfg else 2.0
    state_table_rows: list[tuple[str, int, int]] = (
        list(viz_cfg.viz_state_table_rows) if viz_cfg else []
    )
    logger.info(
        "viz config: robot_type=%r, %d state mappings, %d drawn links, "
        "%d ee links, %d table row(s)",
        robot_type,
        len(state_to_joints),
        len(drawn_links),
        len(ee_links),
        len(state_table_rows),
    )

    cam_names: list[str] = sorted(
        k.split(".")[2]
        for k, v in ds.features.items()
        if k.startswith("observation.images.")
        and not k.endswith("_depth")
        and v["dtype"] in ("image", "video")
    )
    has_depth = any(
        k.endswith("_depth") and k.startswith("observation.images.")
        for k in ds.features
    )
    logger.info(
        "Rendering %d cameras: %s (depth=%s)",
        len(cam_names),
        cam_names,
        has_depth,
    )

    tree: KinematicTree | None = None
    if with_overlay:
        urdf_rel = info.get("extras", {}).get("urdf_relative_path")
        if urdf_rel:
            urdf_text = (root / urdf_rel).read_text()
            tree = parse_urdf(urdf_text)
            logger.info(
                "URDF loaded: %d links, %d joints, root=%s",
                len(tree.links),
                len(tree.joints),
                tree.root_link,
            )
        else:
            logger.warning(
                "No urdf_relative_path in info.json; overlay disabled."
            )
            with_overlay = False

    # Resolve the frame in which the stored camera_pose is expressed.
    # In sim mcaps it is e.g. "robots/dualarm_piperx/left_base_link"; we
    # only need the trailing link name.
    if base_link is None and cam_names:
        first = cameras_info.get(cam_names[0], {})
        parent = (first.get("extrinsic", {}) or {}).get("parent_frame_id", "")
        base_link = parent.rsplit("/", 1)[-1] if parent else None
    if with_overlay and tree is not None:
        if base_link is None or base_link not in tree.links:
            logger.warning(
                "Could not resolve a valid base_link (got %r); "
                "falling back to URDF root %r.",
                base_link,
                tree.root_link,
            )
            base_link = tree.root_link
        logger.info(
            "Camera-pose frame = %r (URDF root = %r).",
            base_link,
            tree.root_link,
        )

    K_per_cam: dict[str, np.ndarray] = {}
    for cam in cam_names:
        K_flat = cameras_info.get(cam, {}).get("intrinsic_K")
        if K_flat is None or len(K_flat) != 9:
            logger.warning("No intrinsic K for %s; overlay disabled.", cam)
            with_overlay = False
            continue
        K_per_cam[cam] = np.array(K_flat, dtype=np.float64).reshape(3, 3)

    ep_frames = ds.episode_data_index
    episode_indices: list[int] = (
        [int(episode_index)]
        if isinstance(episode_index, int)
        else [int(i) for i in episode_index]
    )
    if not episode_indices:
        raise RuntimeError("episode_index list is empty.")
    ranges: list[tuple[int, int, int]] = []  # (ep_idx, start, end)
    for ep_idx in episode_indices:
        s = int(ep_frames["from"][ep_idx])
        e = int(ep_frames["to"][ep_idx])
        if e <= s:
            raise RuntimeError(f"Episode {ep_idx} is empty.")
        logger.info("Episode %d: frames %d..%d", ep_idx, s, e - 1)
        ranges.append((ep_idx, s, e))

    fps = fps or int(getattr(ds, "fps", 30))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None

    total_written = 0
    for ep_idx, start, end in ranges:
        for t in range(start, end):
            sample = ds[t]

            rgb_row: list[np.ndarray] = []
            depth_row: list[np.ndarray] = []

            # FK once per frame (shared across cameras).
            # Always root FK at the URDF root so every arm is visited.
            # Then rebase all link poses into ``base_link``.
            # ``base_link`` is the frame in which camera_pose is stored.
            # ``T_base_root`` is the inverse root-frame pose of base_link.
            poses: dict[str, np.ndarray] | None = None
            state_np: np.ndarray = np.zeros(0)
            if with_overlay and tree is not None:
                state = sample["observation.state"]
                try:
                    import torch

                    if isinstance(state, torch.Tensor):
                        state = state.cpu().numpy()
                except ImportError:
                    pass
                state_np = np.asarray(state)
                joint_values = _state_to_joint_values(
                    state_np, state_to_joints, gripper_scale=gripper_scale
                )
                poses_root = forward_kinematics(
                    tree, joint_values, base_link=tree.root_link
                )
                T_root_base = poses_root.get(
                    base_link or tree.root_link, np.eye(4)
                )
                T_base_root = np.eye(4)
                T_base_root[:3, :3] = T_root_base[:3, :3].T
                T_base_root[:3, 3] = (
                    -T_root_base[:3, :3].T @ T_root_base[:3, 3]
                )
                poses = {k: T_base_root @ v for k, v in poses_root.items()}

            for cam in cam_names:
                rgb = _to_uint8_rgb(sample[f"observation.images.{cam}"])
                depth_bgr: np.ndarray | None = None
                if has_depth:
                    depth = sample[f"observation.images.{cam}_depth"]
                    try:
                        import torch

                        if isinstance(depth, torch.Tensor):
                            depth = depth.cpu().numpy()
                    except ImportError:
                        pass
                    depth = np.asarray(depth)
                    depth_bgr = _depth_to_bgr(depth)

                # Per-frame extrinsic overlay.
                if with_overlay and poses is not None:
                    try:
                        cam_xyz = sample[f"observation.camera_pose.{cam}.xyz"]
                        cam_quat = sample[
                            f"observation.camera_pose.{cam}.quat"
                        ]
                        try:
                            import torch

                            if isinstance(cam_xyz, torch.Tensor):
                                cam_xyz = cam_xyz.cpu().numpy()
                            if isinstance(cam_quat, torch.Tensor):
                                cam_quat = cam_quat.cpu().numpy()
                        except ImportError:
                            pass
                        cam_xyz_np = np.asarray(cam_xyz, dtype=np.float64)
                        cam_quat_np = np.asarray(cam_quat, dtype=np.float64)
                        T_cam_base = _cam_inverse(cam_xyz_np, cam_quat_np)
                        K = K_per_cam[cam]

                        overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        for link in drawn_links:
                            if link not in poses:
                                continue
                            is_ee = link in ee_links
                            axis_len = 0.06 if is_ee else 0.025
                            label = None
                            if is_ee:
                                label = (
                                    f"{link} G="
                                    f"{_ee_gripper_value(state_np, link):.2f}"
                                )
                            _draw_link_axes(
                                overlay,
                                poses[link],
                                T_cam_base,
                                K,
                                axis_len,
                                label=label,
                            )
                        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    except KeyError as exc:
                        logger.warning(
                            "Missing camera_pose feature for %s: %s; "
                            "overlay skipped this cam.",
                            cam,
                            exc,
                        )

                if cam == cam_names[0]:
                    cv2.putText(
                        rgb,
                        f"Ep {ep_idx} Frame {t - start}/{end - start - 1}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    if state_np.size > 0 and state_table_rows:
                        _draw_joint_table(rgb, state_np, state_table_rows)
                    task_str = sample.get("task")
                    if isinstance(task_str, str) and task_str:
                        _draw_instruction(rgb, task_str)

                rgb_row.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if depth_bgr is not None:
                    depth_row.append(depth_bgr)

            frame = _compose_grid(rgb_row, depth_row)

            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                logger.info(
                    "Writing %dx%d @ %d fps -> %s", w, h, fps, out_path
                )
            writer.write(frame)
            total_written += 1

    if writer is not None:
        writer.release()
    logger.info(
        "Rendered %d frames from %d episode(s) to %s",
        total_written,
        len(ranges),
        out_path,
    )
