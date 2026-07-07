# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""Write RawEpisode objects into a LeRobotDataset on disk.

Key responsibilities
--------------------
* ``build_features``: derive the LeRobot ``features`` dict from an
  ``EmbodimentConfig`` (joint dims, camera shapes, depth shapes).
* ``LeRobotEpisodePacker``: thin wrapper around ``LeRobotDataset.create /
  add_frame / save_episode`` that also patches ``meta/info.json`` with
  camera calibration data after all episodes are written.

Depth-image note
----------------
``LeRobotDataset._save_image`` calls ``image_array_to_pil_image`` which only
handles 3-channel uint8 arrays.  For uint16 single-channel depth we convert
to a ``PIL.Image`` (mode "I;16" saved as PNG) **before** handing it to
``add_frame``, which bypasses the array-to-PIL path entirely (LeRobot accepts
``PIL.Image`` objects directly).
"""

from __future__ import annotations
import glob
import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import av
import av.logging
import numpy as np
import PIL.Image
from lerobot.datasets import lerobot_dataset as _ld_mod
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from tools.mcap_to_lerobot.config import EmbodimentConfig
from tools.mcap_to_lerobot.data_types import CameraCalib, RawEpisode

logger = logging.getLogger(__name__)

# Suppress PyAV / libav messages globally. Combined with the fd-level
# redirect inside _encode_h264, this also silences libx264's init banner
# and end-of-stream stats which are emitted from C and bypass Python logging.
av.logging.set_level(av.logging.PANIC)
logging.getLogger("libav").setLevel(logging.CRITICAL)


@contextmanager
def _silence_stderr_fd():
    """Redirect file descriptor 2 to /dev/null.

    Why: libx264 prints its end-of-stream stats and init banner directly to
    stderr from C, bypassing Python's logging and PyAV's log callback. The
    only reliable way to suppress them is at the OS file-descriptor level.
    """
    sys.stderr.flush()
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


def _encode_h264(imgs_dir, video_path, fps, overwrite=False, **_):
    """Encode PNG frames into an H.264 mp4 with libx264 stats silenced."""
    imgs_dir = Path(imgs_dir)
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    template = "frame_" + ("[0-9]" * 6) + ".png"
    input_list = sorted(
        glob.glob(str(imgs_dir / template)),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    if not input_list:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")

    with PIL.Image.open(input_list[0]) as dummy:
        width, height = dummy.size

    video_options = {"g": "2", "crf": "23"}

    with _silence_stderr_fd():
        with av.open(str(video_path), "w") as output:
            stream = output.add_stream("h264", fps, options=video_options)
            stream.pix_fmt = "yuv420p"
            stream.width = width
            stream.height = height
            for frame_path in input_list:
                with PIL.Image.open(frame_path) as img:
                    rgb = img.convert("RGB")
                    frame = av.VideoFrame.from_image(rgb)
                packet = stream.encode(frame)
                if packet:
                    output.mux(packet)
            packet = stream.encode()
            if packet:
                output.mux(packet)

    if not video_path.exists():
        raise OSError(
            f"Video encoding did not work. File not found: {video_path}."
        )


_ld_mod.encode_video_frames = _encode_h264


# ---------------------------------------------------------------------------
# feature builder
# ---------------------------------------------------------------------------


def build_features(
    cfg: EmbodimentConfig,
    color_shape: tuple[int, int, int],
    depth_hw: tuple[int, int] | None,
    use_videos: bool = True,
) -> dict:
    """Build the LeRobot ``features`` dict from an EmbodimentConfig.

    Parameters
    ----------
    cfg:
        EmbodimentConfig (from a preset or custom).
    color_shape:
        ``(H, W, C)`` of the RGB color images.  All cameras share the same
        shape in the current implementation.
    depth_hw:
        ``(H, W)`` of depth images.  Stored as single-channel ``dtype=image``
        (PNG, not mp4).  Pass ``None`` to disable depth packing entirely.
    use_videos:
        If True, color images are stored as mp4 (``dtype="video"``); otherwise
        as per-frame PNG (``dtype="image"``).
    """
    # total joint dim per embodiment
    total_joint_dim = sum(
        arm.arm_dim + 1 for arm in cfg.arms
    )  # arm_dim + 1 gripper each

    # joint names: concatenate across arms
    obs_names: list[str] = []
    act_names: list[str] = []
    for arm in cfg.arms:
        obs_names.extend(arm.obs_joint_names)
        act_names.extend(arm.action_arm_joint_names)

    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (total_joint_dim,),
            "names": obs_names or None,
        },
        "action": {
            "dtype": "float32",
            "shape": (total_joint_dim,),
            "names": act_names or None,
        },
        "observation.velocity": {
            "dtype": "float32",
            "shape": (total_joint_dim,),
            "names": obs_names or None,
        },
        "observation.effort": {
            "dtype": "float32",
            "shape": (total_joint_dim,),
            "names": obs_names or None,
        },
    }

    color_dtype = "video" if use_videos else "image"
    h_c, w_c, c_c = color_shape

    for cam_name in cfg.cameras:
        features[f"observation.images.{cam_name}"] = {
            "dtype": color_dtype,
            "shape": (h_c, w_c, c_c),
            "names": ["height", "width", "channels"],
        }
        # depth: always dtype=image (PNG), single channel stored as HxWx1.
        # Skipped entirely when depth_hw is None (depth packing disabled).
        if depth_hw is not None:
            h_d, w_d = depth_hw[0], depth_hw[1]
            features[f"observation.images.{cam_name}_depth"] = {
                "dtype": "image",
                "shape": (h_d, w_d, 1),
                "names": ["height", "width", "channels"],
            }
        # Per-frame camera extrinsics relative to parent_frame_id recorded in
        # the mcap (typically left_base_link).  Split into xyz (3,) + quat (4,)
        # for clarity; quat ordering is [x, y, z, w] (matches mcap message).
        features[f"observation.camera_pose.{cam_name}.xyz"] = {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x", "y", "z"],
        }
        features[f"observation.camera_pose.{cam_name}.quat"] = {
            "dtype": "float32",
            "shape": (4,),
            "names": ["x", "y", "z", "w"],
        }

    return features


# ---------------------------------------------------------------------------
# depth → PIL conversion (bypasses LeRobot's 3-channel-only path)
# ---------------------------------------------------------------------------


def _depth_to_pil(depth: np.ndarray) -> PIL.Image.Image:
    """Convert HxWx1 or HxW uint16 ndarray to a PIL Image saved as 16-bit PNG.

    LeRobot accepts PIL.Image objects in add_frame and saves them directly
    via PIL.Image.save, which correctly preserves 16-bit depth in PNG.
    """
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    # PIL mode "I;16" is 16-bit unsigned grayscale, saved as PNG without loss.
    return PIL.Image.fromarray(depth.astype(np.uint16), mode="I;16")


def _prepare_frame(frame: dict[str, Any]) -> dict[str, Any]:
    """Convert depth ndarrays to PIL.Image before handing to add_frame."""
    prepared = {}
    for k, v in frame.items():
        if k.endswith("_depth") and isinstance(v, np.ndarray):
            prepared[k] = _depth_to_pil(v)
        else:
            prepared[k] = v
    return prepared


# ---------------------------------------------------------------------------
# info.json patcher
# ---------------------------------------------------------------------------


def patch_info_json(
    dataset_root: Path,
    cameras: dict[str, CameraCalib],
    image_scale: float = 1.0,
    urdf_relative_path: str | None = None,
) -> None:
    """Write camera calibration + extras into meta/info.json.

    Adds extension keys to the JSON object (not part of the LeRobot v2 spec):
        * ``cameras``: per-camera intrinsics / extrinsics / shape.
        * ``extras.image_scale``: the scale that was applied at convert time
          (1.0 means none).  ``verify.py`` reads this to know whether to
          rescale when cross-checking against the source mcap.
        * ``extras.urdf_relative_path``: where ``robot.urdf`` was copied
          relative to the dataset root, if any.
    Skipped with a warning if ``meta/info.json`` does not exist.
    """
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        logger.warning(
            "patch_info_json: %s not found; skipping calib patch.", info_path
        )
        return

    with open(info_path, "r") as f:
        info = json.load(f)

    info["cameras"] = {
        name: calib.to_dict() for name, calib in cameras.items()
    }
    extras = info.get("extras", {})
    extras["image_scale"] = float(image_scale)
    if urdf_relative_path is not None:
        extras["urdf_relative_path"] = urdf_relative_path
    info["extras"] = extras

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(
        "Patched calibration + extras (image_scale=%.3f, urdf=%s) into %s.",
        image_scale,
        urdf_relative_path or "<none>",
        info_path,
    )


def _copy_urdf_into_dataset(
    dataset_root: Path, urdf_path: Path
) -> Path | None:
    """Copy the URDF file into ``dataset_root/meta/robot.urdf``.

    Returns the destination Path on success or None on failure (missing
    source / unreadable / unwritable).  Failures are logged as warnings;
    they never raise so finalize() can still complete with calib written.
    """
    if not urdf_path.exists():
        logger.warning("URDF source not found: %s; skipping copy.", urdf_path)
        return None
    dest = dataset_root / "meta" / "robot.urdf"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = urdf_path.read_text(encoding="utf-8")
        dest.write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "Failed to copy URDF %s -> %s: %s", urdf_path, dest, exc
        )
        return None
    logger.info("Copied URDF (%d bytes) into %s.", len(text), dest)
    return dest


# ---------------------------------------------------------------------------
# LeRobotEpisodePacker
# ---------------------------------------------------------------------------


class LeRobotEpisodePacker:
    """Write RawEpisode objects into a LeRobotDataset on disk.

    Usage::

        packer = LeRobotEpisodePacker.create(
            repo_id="robo_orchard/piperx_pick",
            root=Path("/tmp/dataset"),
            cfg=DUALARM_PIPERX_CONFIG,
            first_mcap=Path("episode0/env0_data.mcap"),
            use_videos=True,
        )
        for ep in episodes:
            packer.add_episode(ep)
        packer.finalize()
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        cfg: EmbodimentConfig,
        image_scale: float = 1.0,
    ) -> None:
        self._ds = dataset
        self._cfg = cfg
        self._calib: dict[str, CameraCalib] = {}
        self._image_scale = float(image_scale)

    # ------------------------------------------------------------------
    # factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        repo_id: str,
        root: Path,
        cfg: EmbodimentConfig,
        first_mcap: Path,
        use_videos: bool = True,
        image_scale: float = 1.0,
        include_depth: bool = True,
    ) -> "LeRobotEpisodePacker":
        """Create a brand-new dataset on disk and return a ready packer.

        ``first_mcap`` is used only to probe image/depth shapes; it must be
        readable.  When ``image_scale < 1.0`` the probed shapes (and therefore
        the dataset's feature schema) are the post-resize shapes.

        Set ``include_depth=False`` to skip depth packing entirely (no depth
        features in the schema, no PNG depth files written).  This is
        recommended when the destination filesystem (e.g. JuiceFS-backed
        bucket storage) handles small files poorly — depth would otherwise
        dominate the file count.
        """
        if not (0.0 < image_scale <= 1.0):
            raise ValueError(
                f"image_scale must be in (0, 1], got {image_scale}"
            )

        from tools.mcap_to_lerobot.reader import (
            probe_depth_shape,
            probe_image_shape,
        )

        # Probe shapes from the first mcap so features are correct.
        first_cam_name = next(iter(cfg.cameras))
        first_cam_topics = cfg.cameras[first_cam_name]
        color_shape = probe_image_shape(
            first_mcap, first_cam_topics.color_image, image_scale=image_scale
        )
        depth_hw: tuple[int, int] | None = None
        if include_depth:
            depth_hw = probe_depth_shape(
                first_mcap,
                first_cam_topics.depth_image,
                image_scale=image_scale,
            )

        features = build_features(
            cfg, color_shape, depth_hw, use_videos=use_videos
        )
        logger.info(
            (
                "Building dataset with %d features "
                "(color %s, depth %s, scale %.3f)"
            ),
            len(features),
            color_shape,
            depth_hw if include_depth else "<disabled>",
            image_scale,
        )

        if root.exists():
            raise FileExistsError(
                f"Dataset root already exists: {root}. "
                "Delete it or choose a different --out path."
            )

        ds = LeRobotDataset.create(
            repo_id=repo_id,
            fps=cfg.fps,
            features=features,
            root=root,
            robot_type=cfg.robot_type,
            use_videos=use_videos,
        )
        # Async PNG writer: LeRobot's default `_save_image` blocks on PIL +
        # write for every frame of every camera.  Start a background thread
        # pool to overlap those writes with reader / joint accumulation.
        # 4 threads/cam is LeRobot's own recommendation; scale up to 12 for
        # 3-camera dual-arm to keep the queue drained.
        n_cam = len(cfg.cameras)
        ds.start_image_writer(num_processes=0, num_threads=max(2, n_cam * 2))
        return cls(ds, cfg=cfg, image_scale=image_scale)

    # ------------------------------------------------------------------
    # write one episode
    # ------------------------------------------------------------------

    def add_episode(self, ep: RawEpisode) -> None:
        """Push all frames of ep into the dataset, then flush via save_episode.

        Workaround: LeRobot only mkdir's the image directory on frame_index==0
        of each ``image`` feature.  Empirically, it sometimes deletes the
        directory mid-episode (e.g. between episodes 2 and 3 on this dataset),
        causing PIL.Image.save to fail with FileNotFoundError when LeRobot
        next tries to write a frame to it.  Rather than rely on LeRobot's
        per-key mkdir, we explicitly mkdir every image directory once per
        frame before delegating to add_frame.  Cheap and idempotent.
        """
        if ep.num_frames == 0:
            logger.warning(
                "add_episode: skipping episode with 0 frames (task=%r).",
                ep.task,
            )
            return

        # Compute the LeRobot episode_index assigned to this episode.
        next_ep_idx = self._ds.meta.total_episodes

        for frame in ep.frames:
            self._ensure_image_dirs(next_ep_idx, frame)
            prepared = _prepare_frame(frame)
            self._ds.add_frame(prepared, task=ep.task)
        self._ds.save_episode()

        # Cache first episode's calib (assumed static across episodes).
        if not self._calib and ep.calib:
            self._calib = ep.calib

    def _ensure_image_dirs(
        self, episode_index: int, frame: dict[str, Any]
    ) -> None:
        """Make sure every image-feature directory for this episode exists.

        Mirrors LeRobot's DEFAULT_IMAGE_PATH layout
        (``images/{image_key}/episode_{ep:06d}/frame_{frame:06d}.png``).
        Idempotent (``exist_ok=True``) so calling per-frame is safe and cheap.
        """
        for key, feat in self._ds.features.items():
            if feat.get("dtype") not in ("image", "video"):
                continue
            if key not in frame:
                continue
            img_dir = (
                Path(self._ds.root)
                / "images"
                / key
                / f"episode_{episode_index:06d}"
            )
            img_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # finalize
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Copy URDF (if any) and patch meta/info.json with calib + extras."""
        # Drain and shut down the async image-writer pool so all pending
        # PNG writes flush before we start finalizing meta.
        try:
            self._ds.stop_image_writer()
        except Exception:
            logger.exception("stop_image_writer failed (continuing)")

        root = Path(self._ds.root)

        urdf_rel: str | None = None
        if self._cfg.urdf_path is not None:
            dest = _copy_urdf_into_dataset(root, self._cfg.urdf_path)
            if dest is not None:
                urdf_rel = str(dest.relative_to(root))

        patch_info_json(
            root,
            self._calib,
            image_scale=self._image_scale,
            urdf_relative_path=urdf_rel,
        )
        logger.info(
            "Dataset finalized: %d episodes, %d frames -> %s",
            self._ds.meta.total_episodes,
            self._ds.meta.total_frames,
            self._ds.root,
        )

    @property
    def dataset(self) -> LeRobotDataset:
        return self._ds
