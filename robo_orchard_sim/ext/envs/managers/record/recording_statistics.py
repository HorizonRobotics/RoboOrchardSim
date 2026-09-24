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

"""Thread-safe performance counters for parallel recording."""

from __future__ import annotations
import threading
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecordingStatisticsSnapshot:
    """Immutable view of one recording episode's counters."""

    enqueued_frames: int
    written_frames: int
    dropped_queue_frames: int
    incomplete_camera_frames: int
    max_queue_depth: int
    average_snapshot_ms: float
    average_worker_ms: float


class RecordingStatistics:
    """Collect counters updated by both producer and worker threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Reset all counters for a new episode."""
        with getattr(self, "_lock", threading.Lock()):
            self._enqueued_frames = 0
            self._written_frames = 0
            self._dropped_queue_frames = 0
            self._incomplete_camera_frames = 0
            self._max_queue_depth = 0
            self._snapshot_total_ms = 0.0
            self._worker_total_ms = 0.0

    def record_enqueued(self, depth: int, snapshot_ms: float) -> None:
        """Record one accepted frame and the current queue depth."""
        with self._lock:
            self._enqueued_frames += 1
            self._snapshot_total_ms += snapshot_ms
            self._max_queue_depth = max(self._max_queue_depth, depth)

    def record_written(self, duration_ms: float) -> None:
        """Record one frame completed by the worker."""
        with self._lock:
            self._written_frames += 1
            self._worker_total_ms += duration_ms

    def record_queue_drop(self) -> None:
        """Record a frame rejected because the queue was full."""
        with self._lock:
            self._dropped_queue_frames += 1

    def record_incomplete_camera(self) -> None:
        """Record a sample skipped because a camera frame was incomplete."""
        with self._lock:
            self._incomplete_camera_frames += 1

    def snapshot(self) -> RecordingStatisticsSnapshot:
        """Return an immutable copy of the current counters."""
        with self._lock:
            snapshot_count = max(1, self._enqueued_frames)
            worker_count = max(1, self._written_frames)
            return RecordingStatisticsSnapshot(
                enqueued_frames=self._enqueued_frames,
                written_frames=self._written_frames,
                dropped_queue_frames=self._dropped_queue_frames,
                incomplete_camera_frames=self._incomplete_camera_frames,
                max_queue_depth=self._max_queue_depth,
                average_snapshot_ms=(self._snapshot_total_ms / snapshot_count),
                average_worker_ms=self._worker_total_ms / worker_count,
            )
