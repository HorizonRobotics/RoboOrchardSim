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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export RGB previews and backfill summary.json; --all exports all MCAPs."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from PIL import Image, ImageDraw, ImageFont

FPS = 30
NS = 1_000_000_000


@dataclass
class VideoSpec:
    topics: list[str]
    sizes: list[tuple[int, int]]
    start: int
    frames: int
    identity: str

    @property
    def width(self) -> int:
        return sum(width for width, _ in self.sizes)

    @property
    def height(self) -> int:
        return self.sizes[0][1]


def _is_rgb(topic: str) -> bool:
    return topic.endswith(("/color_image/image_raw", "/rgb/image_raw", "/rgb"))


def _decode(data: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Cannot decode RGB image")
    return frame


def inspect_mcap(path: Path) -> VideoSpec:
    """Read RGB timing, dimensions and fingerprint."""
    dimensions: dict[str, tuple[int, int]] = {}
    digest = hashlib.sha256(b"mcap-rgb-mosaic-v2-fps30")
    start = end = None
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()
        topics = (
            [c.topic for c in summary.channels.values() if _is_rgb(c.topic)]
            if summary is not None
            else None
        )
        for schema, channel, message, decoded in reader.iter_decoded_messages(
            topics=topics
        ):
            if not _is_rgb(channel.topic):
                continue
            if schema.name != "foxglove.CompressedImage":
                raise ValueError(f"Unsupported RGB schema: {schema.name}")
            if channel.topic not in dimensions:
                dimensions[channel.topic] = _decode(decoded.data).shape[:2]
            if start is None:
                start = message.log_time
            end = message.log_time
            digest.update(channel.topic.encode() + b"\0")
            digest.update(message.log_time.to_bytes(8, "little"))
            digest.update(message.data)
    if start is None or end is None:
        raise ValueError("No RGB images found")
    topics = sorted(dimensions)
    height = max(2, min(h for h, _ in dimensions.values()) // 2 * 2)
    sizes = [
        (
            max(
                2, round(dimensions[t][1] * height / dimensions[t][0] / 2) * 2
            ),
            height,
        )
        for t in topics
    ]
    return VideoSpec(
        topics,
        sizes,
        start,
        (end - start) * FPS // NS + 1,
        "mcap-rgb-v2:" + digest.hexdigest(),
    )


def _view_name(topic: str) -> str:
    for suffix in ("/color_image/image_raw", "/rgb/image_raw", "/rgb"):
        if topic.endswith(suffix):
            return topic[: -len(suffix)].rsplit("/", 1)[-1] or "rgb"
    return topic


@lru_cache(maxsize=128)
def _label_badge(label: str, height: int, width: int) -> np.ndarray:
    factor = 3
    font_size = max(10, round(height / 32))
    padding = max(3, round(height / 80))
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            font = ImageFont.truetype(name, font_size * factor)
            break
        except OSError:
            continue
    else:
        font = ImageFont.load_default(size=font_size * factor)
    left, top, right, bottom = font.getbbox(label)
    pad = padding * factor
    badge = Image.new("RGBA", (right - left + 2 * pad, bottom - top + 2 * pad))
    draw = ImageDraw.Draw(badge)
    draw.rounded_rectangle(
        (0, 0, badge.width - 1, badge.height - 1),
        radius=max(3, round(height / 120)) * factor,
        fill=(20, 24, 30, 180),
    )
    draw.text(
        (pad - left, pad - top), label, font=font, fill=(245, 247, 250, 255)
    )
    badge.thumbnail(
        (
            max(1, min(width, badge.width // factor)),
            max(1, min(height, badge.height // factor)),
        ),
        Image.Resampling.LANCZOS,
    )
    return np.asarray(badge)


def _tile(frame: np.ndarray | None, topic: str, size: tuple[int, int]):
    width, height = size
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    if frame is not None:
        scale = min(width / frame.shape[1], height / frame.shape[0])
        w = max(1, round(frame.shape[1] * scale))
        h = max(1, round(frame.shape[0] * scale))
        x, y = (width - w) // 2, (height - h) // 2
        tile[y : y + h, x : x + w] = cv2.resize(frame, (w, h))
    label = _view_name(topic) + (" (no frame)" if frame is None else "")
    margin = min(max(2, round(height / 60)), (min(width, height) - 1) // 2)
    badge = _label_badge(label, height - 2 * margin, width - 2 * margin)
    h, w = badge.shape[:2]
    region = tile[margin : margin + h, margin : margin + w]
    alpha = badge[:, :, 3:4].astype(np.float32) / 255
    region[:] = (
        badge[:, :, :3][:, :, ::-1] * alpha + region * (1 - alpha)
    ).astype(np.uint8)
    return tile


def validate_video(path: Path, spec: VideoSpec) -> bool:
    """Verify source identity and complete video decoding."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode or probe.stderr:
        return False
    try:
        info = json.loads(probe.stdout)
        streams = [s for s in info["streams"] if s["codec_type"] == "video"]
        if len(streams) != 1:
            return False
        video = streams[0]
        if (video["width"], video["height"]) != (spec.width, spec.height):
            return False
        if info["format"].get("tags", {}).get("comment") != spec.identity:
            return False
    except (KeyError, TypeError, ValueError):
        return False
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-f",
            "null",
            "-",
            "-progress",
            "pipe:1",
        ],
        capture_output=True,
        text=True,
    )
    counts = re.findall(r"^frame=(\d+)\s*$", decoded.stdout, re.MULTILINE)
    return (
        decoded.returncode == 0
        and not decoded.stderr
        and bool(counts)
        and int(counts[-1]) == spec.frames
    )


def _encode(path: Path, output: Path, spec: VideoSpec) -> None:
    with path.open("rb") as source, output.open("wb") as destination:
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{spec.width}x{spec.height}",
            "-r",
            str(FPS),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(FPS),
            "-metadata",
            f"comment={spec.identity}",
            "-movflags",
            "frag_keyframe+empty_moov+default_base_moof",
            "-f",
            "mp4",
            "pipe:1",
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=destination,
        )
        assert process.stdin is not None
        try:
            reader = make_reader(source, decoder_factories=[DecoderFactory()])
            messages = iter(reader.iter_decoded_messages(topics=spec.topics))
            pending = next(messages, None)
            frames: dict[str, np.ndarray] = {}
            for index in range(spec.frames):
                timestamp = spec.start + index * NS // FPS
                while pending is not None and pending[2].log_time <= timestamp:
                    _, channel, _, decoded = pending
                    frames[channel.topic] = _decode(decoded.data)
                    pending = next(messages, None)
                canvas = np.concatenate(
                    [
                        _tile(frames.get(topic), topic, size)
                        for topic, size in zip(
                            spec.topics, spec.sizes, strict=True
                        )
                    ],
                    axis=1,
                )
                process.stdin.write(canvas.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("FFmpeg encoding failed")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def export_video(path: Path, output: Path) -> tuple[Path, bool]:
    """Export or reuse a verified video; preserve incomplete previous files."""
    spec = inspect_mcap(path)
    folder = output.parent
    stem = output.stem
    folder.mkdir(parents=True, exist_ok=True)
    candidates = [output, *sorted(folder.glob(f"{stem}__retry*.mp4"))]
    for candidate in candidates:
        if candidate.is_file() and validate_video(candidate, spec):
            return candidate, False
    retry = 1
    while output.exists():
        output = folder / f"{stem}__retry{retry}.mp4"
        retry += 1
    _encode(path, output, spec)
    if not validate_video(output, spec):
        raise RuntimeError(f"Video validation failed: {output}")
    return output, True


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_episodes(
    task_dir: Path,
    limit: int | None,
) -> list[tuple[Path, int, dict]]:
    manifest = _read_json(task_dir / "task_eval_summary.json")
    episodes = []
    for config in manifest["configs"]:
        config_dir = (
            task_dir
            / config["group_id"]
            / f"config_{config['config_index']:04d}"
        )
        result_path = config_dir / "eval_result.json"
        if not result_path.is_file():
            if config.get("error"):
                continue
            raise FileNotFoundError(result_path)
        result = _read_json(result_path)
        for index, episode in enumerate(result["episode_results"]):
            episodes.append((config_dir, index, episode))
            if limit is not None and len(episodes) >= limit:
                return episodes
    return episodes


def _matching_recordings(
    config_dir: Path,
    index: int,
    episode: dict,
    paths: list[Path],
) -> list[Path]:
    record_dir = (episode.get("metrics") or {}).get("record_dir")
    if record_dir:
        exact = [p for p in paths if str(p.parent) == record_dir]
        if exact:
            return exact
    # Swap episodes can share a seed; match config and ordinal as well.
    directory = f"episode_{index:04d}_seed_{episode['seed']}"
    legacy = [
        p
        for p in paths
        if p.is_relative_to(config_dir / "records")
        and p.parent.name == directory
    ]
    if legacy:
        return legacy
    # Older shard results lack record_dir and store MCAPs below shards/.
    # Only accept a unique seed match; repeated seeds remain unresolved.
    shard_dir = config_dir.parent.parent / "shards"
    return [
        p
        for p in paths
        if p.is_relative_to(shard_dir)
        and p.parent.name.startswith("episode_")
        and p.parent.name.endswith(f"_seed_{episode['seed']}")
    ]


def collect_preview_recordings(
    root: Path,
    summary: dict[str, Any],
) -> tuple[list[tuple[dict, Path]], int]:
    """Match summary previews to recordings."""
    if not isinstance(summary.get("task_results"), dict):
        raise ValueError("summary.json does not use the leaderboard format")
    selected = []
    unresolved = 0
    for name, task in summary["task_results"].items():
        previews = task.get("preview_episodes", [])
        if not previews:
            continue
        try:
            indexes = [preview["episode_index"] for preview in previews]
            if any(type(index) is not int or index < 0 for index in indexes):
                raise ValueError("invalid preview episode_index")
            episodes = _task_episodes(root / name, max(indexes) + 1)
            matches = []
            for preview in previews:
                index = preview["episode_index"]
                if type(index) is not int or not 0 <= index < len(episodes):
                    raise ValueError("preview episode_index is out of range")
                config_dir, local_index, episode = episodes[index]
                keys = ("seed", "steps", "stop_reason")
                if any(preview[k] != episode[k] for k in keys):
                    raise ValueError("preview does not match episode result")
                if preview.get("instruction", "") != episode.get(
                    "instruction", ""
                ):
                    raise ValueError("preview instruction does not match")
                matches.append((preview, config_dir, local_index, episode))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"[warning] Cannot map task {name}: {exc}", flush=True)
            unresolved += len(previews)
            continue
        paths = sorted((root / name).rglob("*.mcap"))
        for preview, config_dir, local_index, episode in matches:
            candidates = _matching_recordings(
                config_dir,
                local_index,
                episode,
                paths,
            )
            if len(candidates) != 1:
                unresolved += 1
                print(
                    f"[warning] {name} episode {preview['episode_index']}: "
                    "no unique recording",
                    flush=True,
                )
                continue
            selected.append((preview, candidates[0]))
    return selected, unresolved


def video_targets(
    root: Path, summary: dict[str, Any] | None, *, convert_all: bool
) -> dict[Path, Path]:
    """Name videos by task-wide episode index."""
    targets = {}
    for name, task in (summary or {}).get("task_results", {}).items():
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid task directory name: {name!r}")
        previews = task.get("preview_episodes", [])
        if not convert_all and not previews:
            continue
        try:
            limit = (
                None
                if convert_all
                else max(p["episode_index"] for p in previews) + 1
            )
            episodes = _task_episodes(root / name, limit)
            indices = (
                range(len(episodes))
                if convert_all
                else [p["episode_index"] for p in previews]
            )
            paths = sorted((root / name).rglob("*.mcap"))
            for index in indices:
                config_dir, local_index, episode = episodes[index]
                matches = _matching_recordings(
                    config_dir, local_index, episode, paths
                )
                if len(matches) == 1:
                    seed = int(episode["seed"])
                    targets[matches[0]] = (
                        root
                        / "videos"
                        / name
                        / f"episode_{index:04d}_seed_{seed}.mp4"
                    )
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            print(f"[warning] Cannot name task {name}: {exc}", flush=True)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_root", type=Path, help="Evaluation output root to scan"
    )
    parser.add_argument(
        "--all",
        dest="convert_all",
        action="store_true",
        help="Convert all MCAPs; summary.json is optional in this mode.",
    )
    args = parser.parse_args()
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            parser.error(f"{command} is required")
    root = args.output_root.resolve()
    if not root.is_dir():
        parser.error(f"Output root is not a directory: {root}")
    summary_path = root / "summary.json"
    if not summary_path.is_file() and not args.convert_all:
        parser.error(f"Summary not found: {summary_path}")
    summary = _read_json(summary_path) if summary_path.is_file() else None
    selected, unresolved = [], 0
    if summary is not None:
        if not isinstance(summary.get("task_results"), dict):
            parser.error(
                "Summary must use the leaderboard task_results format"
            )
        selected, unresolved = collect_preview_recordings(root, summary)
    paths = (
        sorted(root.rglob("*.mcap"))
        if args.convert_all
        else sorted({path for _, path in selected})
    )
    if args.convert_all and not paths:
        parser.error("No MCAP files found")
    targets = video_targets(root, summary, convert_all=args.convert_all)
    videos: dict[Path, Path] = {}
    created = skipped = failed = 0
    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {path}", flush=True)
        try:
            target = targets.get(path)
            if target is None:
                relative = path.relative_to(root)
                task = (
                    relative.parts[0] if len(relative.parts) > 1 else "_root"
                )
                digest = hashlib.sha256(str(relative).encode()).hexdigest()[
                    :16
                ]
                target = root / "videos" / task / "unmapped" / f"{digest}.mp4"
            output, written = export_video(path, target)
            videos[path] = output
            created += int(written)
            skipped += int(not written)
            print(
                f"  {'created' if written else 'reused'}: {output}", flush=True
            )
        except Exception as exc:
            failed += 1
            print(f"[error] {path}: {exc}", flush=True)
    updated = 0
    for preview, path in selected:
        if path not in videos:
            unresolved += 1
            continue
        video = os.path.relpath(videos[path], root)
        if preview.get("video") != video:
            preview["video"] = video
            updated += 1
    if updated:
        # Bucket mounts may not support atomic replacement.
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        f"Created={created}, reused={skipped}, failed={failed}; "
        f"preview videos updated={updated}, unresolved={unresolved}"
    )
    return 1 if failed or unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
