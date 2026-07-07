# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""Decode one robo_orchard_sim mcap into an aligned RawEpisode.

Pipeline
--------
1. _read_buckets  — iterate mcap, bucket every wanted topic by log_time_ns.
2. _pick_base_time — choose master clock
   (config.master_clock_topic or arms[0].action_arm).
3. _time_sync     — argmin-nearest align every stream to the master timestamps.
4. _filter_frames — static-frame + head/tail trim mask.
5. _build_frames  — assemble per-frame dicts:
   obs/action/velocity/effort/color/depth.
6. _extract_calib — pull CameraCalib from color_info + tf buckets.
7. _resolve_task  — manifest instruction > /meta_data > default.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from mcap_protobuf.reader import read_protobuf_messages

from tools.mcap_to_lerobot.config import EmbodimentConfig
from tools.mcap_to_lerobot.data_types import CameraCalib, RawEpisode

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _decode_jpg(data: bytes) -> np.ndarray:
    """JPEG bytes → HxWx3 uint8 RGB."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed on a color CompressedImage")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _decode_depth(data: bytes) -> np.ndarray:
    """PNG bytes → HxWx1 uint16 (single channel, preserves 16-bit depth)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("cv2.imdecode failed on a depth CompressedImage")
    if img.ndim == 2:
        return img[:, :, np.newaxis]  # HxW → HxWx1
    return img[:, :, :1]  # keep only first channel, shape (H, W, 1)


def _argmin_sync(msg_ts: list[int], base_ts: np.ndarray) -> list[int]:
    """For each base timestamp, find the index of the nearest msg timestamp."""
    msg_arr = np.array(msg_ts, dtype=np.int64)
    # shape: (len_base, len_msg)
    diff = np.abs(base_ts[:, None] - msg_arr[None, :])
    return diff.argmin(axis=1).tolist()


def _scaled_hw(h: int, w: int, scale: float) -> tuple[int, int]:
    """Truncated integer (H, W) at the target scale."""
    return int(h * scale), int(w * scale)


def _maybe_resize_color(img: np.ndarray, scale: float) -> np.ndarray:
    """Resize HxWx3 uint8 RGB using cv2.INTER_AREA.

    Short-circuits at scale==1.0.
    """
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    new_h, new_w = _scaled_hw(h, w, scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _maybe_resize_depth(depth: np.ndarray, scale: float) -> np.ndarray:
    """Resize HxWx1 uint16 depth using cv2.INTER_NEAREST.

    Short-circuits at scale==1.0.

    Nearest-neighbour is mandatory: linear/area interpolation at object
    boundaries averages foreground and background depths into a fake midpoint
    that does not correspond to any physical surface.
    """
    if scale == 1.0:
        return depth
    h, w = depth.shape[:2]
    new_h, new_w = _scaled_hw(h, w, scale)
    # cv2.resize wants 2D for single-channel; squeeze + re-expand.
    resized = cv2.resize(
        depth[:, :, 0], (new_w, new_h), interpolation=cv2.INTER_NEAREST
    )
    return resized[:, :, np.newaxis]


def _scale_intrinsic_K(K: list[float], scale: float) -> list[float]:
    """Scale a 9-element flat K matrix (row-major 3x3) by `scale`.

    Multiplies the first two rows (fx, 0, cx) / (0, fy, cy) by scale.  The
    third row stays [0, 0, 1].  Returns a fresh list.
    """
    if scale == 1.0:
        return list(K)
    out = list(K)
    for i in range(6):  # first 6 entries == first two rows
        out[i] = out[i] * scale
    return out


def _scale_intrinsic_P(P: list[float], scale: float) -> list[float]:
    """Scale a 12-element flat P matrix (row-major 3x4) by `scale`.

    Multiplies the first two rows by scale; the third row stays [0, 0, 1, 0].
    """
    if scale == 1.0:
        return list(P)
    out = list(P)
    for i in range(8):  # first 8 entries == first two rows
        out[i] = out[i] * scale
    return out


def _meta_to_dict(struct_msg) -> dict:
    """google.protobuf.Struct → plain python dict, without MessageToDict."""
    # Struct supports dict-like iteration directly
    result = {}
    for key in struct_msg:
        val = struct_msg[key]
        # val is a google.protobuf.struct_pb2.Value wrapper
        # Access the underlying python value via its kind
        kind = val.WhichOneof("kind")
        if kind == "number_value":
            result[key] = val.number_value
        elif kind == "string_value":
            result[key] = val.string_value
        elif kind == "bool_value":
            result[key] = val.bool_value
        elif kind == "struct_value":
            result[key] = _meta_to_dict(val.struct_value)
        elif kind == "list_value":
            result[key] = [_extract_list_val(v) for v in val.list_value.values]
        else:
            result[key] = None
    return result


def _extract_list_val(v) -> Any:
    kind = v.WhichOneof("kind")
    if kind == "number_value":
        return v.number_value
    if kind == "string_value":
        return v.string_value
    if kind == "bool_value":
        return v.bool_value
    if kind == "struct_value":
        return _meta_to_dict(v.struct_value)
    if kind == "list_value":
        return [_extract_list_val(i) for i in v.list_value.values]
    return None


# ---------------------------------------------------------------------------
# McapEpisodeReader
# ---------------------------------------------------------------------------


class McapEpisodeReader:
    """Read a single robo_orchard_sim mcap into a cleaned, aligned RawEpisode.

    Parameters
    ----------
    cfg:
        EmbodimentConfig from config_presets (e.g. DUALARM_PIPERX_CONFIG).
    image_scale:
        Downscale factor applied to color and depth images.  ``1.0`` (default)
        means no scaling and the resize path is short-circuited.  Must be in
        ``(0, 1]``.  When set < 1.0:
            * color is resized with cv2.INTER_AREA (best for downsampling).
            * depth is resized with cv2.INTER_NEAREST (preserves discrete
              depth values; avoids fake midpoints at object edges).
            * camera intrinsic K and projection P are scaled accordingly so
              downstream projection math stays valid.
    """

    def __init__(
        self,
        cfg: EmbodimentConfig,
        image_scale: float = 1.0,
        include_depth: bool = True,
    ) -> None:
        if not (0.0 < image_scale <= 1.0):
            raise ValueError(
                f"image_scale must be in (0, 1], got {image_scale}"
            )
        self.cfg = cfg
        self._image_scale = float(image_scale)
        self._include_depth = bool(include_depth)
        self._wanted_topics: set[str] = self._build_topic_whitelist()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def read(
        self,
        mcap_path: str | Path,
        manifest_instruction: str | None = None,
        default_task: str = "manipulation",
    ) -> RawEpisode:
        """Decode one mcap file → RawEpisode.

        Args:
            mcap_path: Path to ``env*_data.mcap``.
            manifest_instruction: Instruction string from an external JSON
                manifest.  Takes priority over /meta_data.
            default_task: Fallback task string if neither manifest nor
                /meta_data provides one.
        """
        mcap_path = Path(mcap_path)
        buckets, meta_dict = self._read_buckets(mcap_path)
        base_ts = self._pick_base_time(buckets)
        if not base_ts:
            raise RuntimeError(f"{mcap_path}: master clock has 0 messages.")

        synced = self._time_sync(buckets, base_ts)
        mask = self._filter_frames(synced, base_ts)
        frames = self._build_frames(synced, base_ts, mask)
        calib = self._extract_calib(synced)
        task = self._resolve_task(
            meta_dict, manifest_instruction, default_task
        )

        return RawEpisode(
            frames=frames, task=task, meta=meta_dict, calib=calib
        )

    # ------------------------------------------------------------------
    # step 1: read_buckets
    # ------------------------------------------------------------------

    def _build_topic_whitelist(self) -> set[str]:
        topics: set[str] = {self.cfg.meta_topic}
        for arm in self.cfg.arms:
            topics.add(arm.obs_topic)
            topics.add(arm.action_arm_topic)
            topics.add(arm.action_gripper_topic)
        for cam_topics in self.cfg.cameras.values():
            topics.add(cam_topics.color_image)
            topics.add(cam_topics.color_info)
            topics.add(cam_topics.tf)
            if self._include_depth:
                topics.add(cam_topics.depth_image)
                topics.add(cam_topics.depth_info)
        return topics

    def _read_buckets(
        self, mcap_path: Path
    ) -> tuple[dict[str, list[tuple[int, Any]]], dict]:
        """Read mcap into per-topic lists of (log_time_ns, payload).

        For joint topics the payload is the raw proto message.
        For image topics the payload is raw bytes (data field).
        For camera_info / tf topics the payload is the proto message.
        """
        buckets: dict[str, list[tuple[int, Any]]] = {
            t: [] for t in self._wanted_topics if t != self.cfg.meta_topic
        }
        meta_dict: dict[str, Any] = {}

        for m in read_protobuf_messages(
            str(mcap_path), topics=list(self._wanted_topics)
        ):
            topic = m.topic
            ts = m.log_time_ns
            proto = m.proto_msg

            if topic == self.cfg.meta_topic:
                try:
                    meta_dict = _meta_to_dict(proto)
                except Exception:
                    meta_dict = {}
                continue

            if topic not in buckets:
                continue

            # image raw topics: store raw bytes only
            if topic.endswith("/image_raw"):
                buckets[topic].append((ts, proto.data))
            else:
                buckets[topic].append((ts, proto))

        return buckets, meta_dict

    # ------------------------------------------------------------------
    # step 2: pick_base_time
    # ------------------------------------------------------------------

    def _pick_base_time(self, buckets: dict[str, list]) -> list[int]:
        master_topic = self.cfg.master_clock_topic
        if master_topic is None:
            master_topic = self.cfg.arms[0].action_arm_topic
        entries = buckets.get(master_topic, [])
        return [ts for ts, _ in entries]

    # ------------------------------------------------------------------
    # step 3: time_sync
    # ------------------------------------------------------------------

    def _time_sync(
        self,
        buckets: dict[str, list[tuple[int, Any]]],
        base_ts: list[int],
    ) -> dict[str, list[Any]]:
        """Align every topic to base_ts via argmin nearest-neighbour.

        Returns a dict mapping topic → list of payloads (one per base_ts tick).
        """
        base_arr = np.array(base_ts, dtype=np.int64)
        synced: dict[str, list[Any]] = {}
        for topic, entries in buckets.items():
            if not entries:
                synced[topic] = []
                continue
            msg_ts = [ts for ts, _ in entries]
            payloads = [p for _, p in entries]
            indices = _argmin_sync(msg_ts, base_arr)
            synced[topic] = [payloads[i] for i in indices]
        return synced

    # ------------------------------------------------------------------
    # step 4: filter_frames
    # ------------------------------------------------------------------

    def _filter_frames(
        self,
        synced: dict[str, list[Any]],
        base_ts: list[int],
    ) -> np.ndarray:
        """Return a keep-mask: trim leading + trailing static frames only.

        Static decision uses **observation** joints (real robot state, not
        master/action targets) and includes the gripper finger, so a frame
        where only the gripper moves counts as moving.

        Inner static frames are preserved — the goal is to clip the idle
        head/tail of an episode, not to drop intermediate pauses.
        """
        n = len(base_ts)
        mask = np.ones(n, dtype=bool)
        cfg = self.cfg

        # --- static filter: based on observation (arm + gripper finger) ---
        if cfg.static_threshold > 0:
            obs_cols = []
            for arm in cfg.arms:
                entries = synced.get(arm.obs_topic, [])
                if not entries:
                    continue
                # arm_dim arm joints + 1 first-finger gripper joint
                cols = arm.arm_dim + 1
                positions = np.array(
                    [
                        [s.position for s in msg.states[:cols]]
                        if len(msg.states) >= cols
                        else [s.position for s in msg.states]
                        + [0.0] * (cols - len(msg.states))
                        for msg in entries
                    ],
                    dtype=np.float32,
                )
                obs_cols.append(positions)
            if obs_cols:
                joint_pos = np.concatenate(obs_cols, axis=1)
                # frame-to-frame absolute deltas
                diff = np.abs(np.diff(joint_pos, axis=0))
                is_moving = np.any(diff > cfg.static_threshold, axis=1)
                # A frame counts as "moving" if it is either side of a
                # detected transition — this avoids clipping the very first
                # or very last moving frame at each boundary.
                moving_frame = np.zeros(n, dtype=bool)
                moving_frame[1:] |= is_moving
                moving_frame[:-1] |= is_moving
                if moving_frame.any():
                    first = int(np.argmax(moving_frame))
                    last = n - 1 - int(np.argmax(moving_frame[::-1]))
                    keep = np.zeros(n, dtype=bool)
                    keep[first : last + 1] = True
                    mask &= keep
                # else: entire episode is static — keep everything as-is.

        # --- head / tail time trim (additive constraint) ---
        if (
            cfg.head_time_to_filter_s is not None
            or cfg.tail_time_to_filter_s is not None
        ):
            ts_arr = np.array(base_ts, dtype=np.int64)
            time_mask = np.ones(n, dtype=bool)
            if cfg.head_time_to_filter_s is not None:
                elapsed = (ts_arr - ts_arr[0]) / 1e9
                time_mask[elapsed < cfg.head_time_to_filter_s] = False
            if cfg.tail_time_to_filter_s is not None:
                remaining = (ts_arr[-1] - ts_arr) / 1e9
                time_mask[remaining < cfg.tail_time_to_filter_s] = False
            mask &= time_mask

        return mask

    # ------------------------------------------------------------------
    # step 5: build_frames
    # ------------------------------------------------------------------

    def _build_frames(
        self,
        synced: dict[str, list[Any]],
        base_ts: list[int],  # noqa: ARG002  kept for API symmetry with _filter_frames
        mask: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Assemble per-frame dicts for all retained frames."""
        retained = np.where(mask)[0]

        # Pre-decode all color + depth images (only retained indices).
        # When image_scale < 1.0, resize on the fly:
        #   color: INTER_AREA (best for downsampling)
        #   depth: INTER_NEAREST (preserves discrete depth values)
        scale = self._image_scale
        color_arrays: dict[str, list[np.ndarray]] = {}
        depth_arrays: dict[str, list[np.ndarray]] = {}
        for cam_name, cam_topics in self.cfg.cameras.items():
            color_raw = synced.get(cam_topics.color_image, [])
            if color_raw:
                color_arrays[cam_name] = [
                    _maybe_resize_color(_decode_jpg(color_raw[i]), scale)
                    for i in retained
                ]
            if self._include_depth:
                depth_raw = synced.get(cam_topics.depth_image, [])
                if depth_raw:
                    depth_arrays[cam_name] = [
                        _maybe_resize_depth(_decode_depth(depth_raw[i]), scale)
                        for i in retained
                    ]

        frames: list[dict[str, Any]] = []
        for out_idx, in_idx in enumerate(retained):
            frame: dict[str, Any] = {}

            # ---- observation.state / velocity / effort ----
            #
            # Layout per arm: [arm_dim arm joints, 1 gripper scalar].
            #
            # obs gripper source = obs joint topic's FIRST FINGER state (the
            # state right after the arm_dim arm joints in the same message).
            # This mirrors arrow packer: slave/obs reflects the real robot
            # state (finger1 position scaled by gripper_scale, default x2 to
            # match piper's full-open-width convention).  Velocity / effort
            # of that finger are likewise scaled to keep units consistent.
            obs_pos_parts, obs_vel_parts, obs_eff_parts = [], [], []
            for arm in self.cfg.arms:
                obs_msgs = synced.get(arm.obs_topic, [])
                if not obs_msgs:
                    break

                obs_msg = obs_msgs[in_idx]

                # arm joints (first arm_dim entries of obs)
                arm_pos = np.array(
                    [s.position for s in obs_msg.states[: arm.arm_dim]],
                    dtype=np.float32,
                )
                arm_vel = np.array(
                    [s.velocity for s in obs_msg.states[: arm.arm_dim]],
                    dtype=np.float32,
                )
                arm_eff = np.array(
                    [s.effort for s in obs_msg.states[: arm.arm_dim]],
                    dtype=np.float32,
                )

                # obs gripper = obs_msg.states[arm_dim] (first finger),
                # scaled by gripper_scale. Skip the frame gracefully if
                # the message is shorter than expected.
                if len(obs_msg.states) <= arm.arm_dim:
                    break
                obs_grip_state = obs_msg.states[arm.arm_dim]
                grip_pos = np.array(
                    [obs_grip_state.position * arm.gripper_scale],
                    dtype=np.float32,
                )
                grip_vel = np.array(
                    [obs_grip_state.velocity * arm.gripper_scale],
                    dtype=np.float32,
                )
                grip_eff = np.array(
                    [obs_grip_state.effort * arm.gripper_scale],
                    dtype=np.float32,
                )

                obs_pos_parts.extend([arm_pos, grip_pos])
                obs_vel_parts.extend([arm_vel, grip_vel])
                obs_eff_parts.extend([arm_eff, grip_eff])
            else:
                frame["observation.state"] = np.concatenate(
                    obs_pos_parts
                ).astype(np.float32)
                frame["observation.velocity"] = np.concatenate(
                    obs_vel_parts
                ).astype(np.float32)
                frame["observation.effort"] = np.concatenate(
                    obs_eff_parts
                ).astype(np.float32)

            # ---- action ----
            act_pos_parts = []
            for arm in self.cfg.arms:
                act_arm_msgs = synced.get(arm.action_arm_topic, [])
                act_grip_msgs = synced.get(arm.action_gripper_topic, [])
                if not act_arm_msgs or not act_grip_msgs:
                    break

                act_arm_msg = act_arm_msgs[in_idx]
                act_grip_msg = act_grip_msgs[in_idx]

                arm_pos = np.array(
                    [s.position for s in act_arm_msg.states[: arm.arm_dim]],
                    dtype=np.float32,
                )
                grip_state = next(
                    (
                        s
                        for s in act_grip_msg.states
                        if s.name == arm.gripper_joint_name
                    ),
                    None,
                )
                if grip_state is None:
                    grip_state = act_grip_msg.states[0]
                grip_pos = np.array(
                    [grip_state.position * arm.gripper_scale], dtype=np.float32
                )
                act_pos_parts.extend([arm_pos, grip_pos])
            else:
                frame["action"] = np.concatenate(act_pos_parts).astype(
                    np.float32
                )

            # ---- images ----
            for cam_name in self.cfg.cameras:
                if cam_name in color_arrays:
                    frame[f"observation.images.{cam_name}"] = color_arrays[
                        cam_name
                    ][out_idx]
                if cam_name in depth_arrays:
                    frame[f"observation.images.{cam_name}_depth"] = (
                        depth_arrays[cam_name][out_idx]
                    )

            # ---- per-frame camera extrinsics (xyz, quat) ----
            # tf_msg is the FrameTransform proto at this frame's master clock.
            # parent_frame_id is preserved at the dataset level via info.json;
            # here we only export translation + rotation.
            for cam_name, cam_topics in self.cfg.cameras.items():
                tf_seq = synced.get(cam_topics.tf, [])
                if not tf_seq:
                    raise RuntimeError(
                        f"Missing TF stream for {cam_name} "
                        f"(topic {cam_topics.tf}); cannot fill camera_pose."
                    )
                tf_msg = tf_seq[in_idx]
                frame[f"observation.camera_pose.{cam_name}.xyz"] = np.array(
                    [
                        tf_msg.translation.x,
                        tf_msg.translation.y,
                        tf_msg.translation.z,
                    ],
                    dtype=np.float32,
                )
                frame[f"observation.camera_pose.{cam_name}.quat"] = np.array(
                    [
                        tf_msg.rotation.x,
                        tf_msg.rotation.y,
                        tf_msg.rotation.z,
                        tf_msg.rotation.w,
                    ],
                    dtype=np.float32,
                )

            frames.append(frame)

        return frames

    # ------------------------------------------------------------------
    # step 6: extract_calib
    # ------------------------------------------------------------------

    def _extract_calib(
        self, synced: dict[str, list[Any]]
    ) -> dict[str, CameraCalib]:
        """Extract CameraCalib from first-frame camera_info and tf topics.

        When ``self._image_scale < 1.0``, K / P matrices and the orded
        width / height are scaled to match the resized images.  Extrinsics
        (translation / rotation) stay unchanged since they describe the
        camera pose in world coordinates and are scale-invariant.
        """
        scale = self._image_scale
        calib: dict[str, CameraCalib] = {}
        for cam_name, cam_topics in self.cfg.cameras.items():
            info_msgs = synced.get(cam_topics.color_info, [])
            tf_msgs = synced.get(cam_topics.tf, [])
            if not info_msgs or not tf_msgs:
                continue

            info = info_msgs[0]
            tf = tf_msgs[0]

            scaled_h, scaled_w = _scaled_hw(info.height, info.width, scale)
            calib[cam_name] = CameraCalib(
                cam_name=cam_name,
                intrinsic_K=_scale_intrinsic_K(list(info.K), scale),
                intrinsic_P=_scale_intrinsic_P(list(info.P), scale),
                distortion_model=getattr(
                    info, "distortion_model", "plumb_bob"
                ),
                distortion_D=list(info.D),
                width=scaled_w,
                height=scaled_h,
                parent_frame_id=tf.parent_frame_id,
                child_frame_id=tf.child_frame_id,
                translation={
                    "x": tf.translation.x,
                    "y": tf.translation.y,
                    "z": tf.translation.z,
                },
                rotation={
                    "x": tf.rotation.x,
                    "y": tf.rotation.y,
                    "z": tf.rotation.z,
                    "w": tf.rotation.w,
                },
            )
        return calib

    # ------------------------------------------------------------------
    # step 7: resolve_task
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_task(
        meta_dict: dict,
        manifest_instruction: str | None,
        default_task: str,
    ) -> str:
        # Priority 1: explicit manifest instruction
        if manifest_instruction and isinstance(manifest_instruction, str):
            return manifest_instruction.strip() or default_task

        if not meta_dict:
            return default_task

        # Priority 2: instruction field in /meta_data
        instr = meta_dict.get("instruction")
        if isinstance(instr, str) and instr.strip():
            return instr.strip()

        # Priority 3: first actor_category in actors dict
        actors = meta_dict.get("actors")
        if isinstance(actors, dict):
            for entry in actors.values():
                if isinstance(entry, dict):
                    cat = entry.get("actor_category")
                    if isinstance(cat, str) and cat.strip():
                        return cat.strip()

        return default_task


def probe_image_shape(
    mcap_path: str | Path,
    color_topic: str,
    image_scale: float = 1.0,
) -> tuple[int, int, int]:
    """Decode the first color frame of a topic to discover (H, W, 3).

    Useful for building LeRobot features before creating the dataset.
    If ``image_scale < 1.0`` the returned shape is the post-resize shape.
    """
    for m in read_protobuf_messages(str(mcap_path), topics=[color_topic]):
        h, w, c = _decode_jpg(m.proto_msg.data).shape
        new_h, new_w = _scaled_hw(int(h), int(w), image_scale)
        return (new_h, new_w, int(c))
    raise RuntimeError(f"{mcap_path}: no messages on {color_topic}.")


def probe_depth_shape(
    mcap_path: str | Path,
    depth_topic: str,
    image_scale: float = 1.0,
) -> tuple[int, int]:
    """Decode the first depth frame to discover (H, W).

    If ``image_scale < 1.0`` the returned (H, W) is post-resize.
    """
    for m in read_protobuf_messages(str(mcap_path), topics=[depth_topic]):
        depth = _decode_depth(m.proto_msg.data)  # (H, W, 1)
        h, w = depth.shape[:2]
        new_h, new_w = _scaled_hw(int(h), int(w), image_scale)
        return (new_h, new_w)
    raise RuntimeError(f"{mcap_path}: no messages on {depth_topic}.")
