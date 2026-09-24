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

"""Build independently owned CPU snapshots for parallel recording."""

from __future__ import annotations
import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel
from robo_orchard_core.datatypes.tf_graph import BatchFrameTransformGraph


class CpuSnapshotCopies:
    """Clone nested observations into reusable CPU buffers."""

    def __init__(self, buffers: dict[str, torch.Tensor]) -> None:
        self._buffers = buffers
        self._copy_requests: dict[
            torch.device,
            list[tuple[torch.Tensor, torch.Tensor]],
        ] = {}

    def clone(self, value: Any, path: str = "root") -> Any:
        """Return an independently owned CPU representation of ``value``."""
        if isinstance(value, torch.Tensor):
            return self._clone_tensor(path, value)
        if isinstance(value, BatchFrameTransformGraph):
            return self._clone_transform_graph(value, path)
        if isinstance(value, BaseModel):
            return self._clone_model(value, path)
        if isinstance(value, Mapping):
            return {
                key: self.clone(item, f"{path}/{key}")
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(
                self.clone(item, f"{path}/{index}")
                for index, item in enumerate(value)
            )
        if isinstance(value, list):
            return [
                self.clone(item, f"{path}/{index}")
                for index, item in enumerate(value)
            ]
        if isinstance(value, np.ndarray):
            return value.copy()
        return copy.deepcopy(value)

    def finalize(self) -> tuple[Any, ...]:
        """Schedule CUDA-to-CPU copies and return completion events."""
        ready_events = []
        for device, requests in self._copy_requests.items():
            producer_stream = torch.cuda.current_stream(device)
            copy_stream = torch.cuda.Stream(device=device)
            copy_stream.wait_stream(producer_stream)
            with torch.cuda.stream(copy_stream):
                for destination, source in requests:
                    destination.copy_(source, non_blocking=True)
                    source.record_stream(copy_stream)
            ready_event = torch.cuda.Event()
            ready_event.record(copy_stream)
            producer_stream.wait_event(ready_event)
            ready_events.append(ready_event)
        return tuple(ready_events)

    def _clone_tensor(self, path: str, value: torch.Tensor) -> torch.Tensor:
        source = value.detach()
        pin_memory = source.device.type == "cuda"
        destination = self._buffer_for(
            path,
            source,
            pin_memory=pin_memory,
        )
        if not pin_memory:
            destination.copy_(source)
            return destination
        self._copy_requests.setdefault(source.device, []).append(
            (destination, source)
        )
        return destination

    def _clone_model(self, value: BaseModel, path: str) -> BaseModel:
        fields = {
            name: self.clone(getattr(value, name), f"{path}/{name}")
            for name in value.__class__.model_fields
        }
        if value.__pydantic_extra__:
            fields.update(
                {
                    name: self.clone(item, f"{path}/{name}")
                    for name, item in value.__pydantic_extra__.items()
                }
            )
        return value.__class__.model_construct(**fields)

    def _clone_transform_graph(
        self,
        value: BatchFrameTransformGraph,
        path: str,
    ) -> BatchFrameTransformGraph:
        transforms = []
        static_transforms = []
        for parent_frame, child_edges in value.edges.items():
            for child_frame, transform in child_edges.items():
                if value.is_mirrored_tf(parent_frame, child_frame):
                    continue
                transforms.append(
                    self.clone(
                        transform,
                        f"{path}/{parent_frame}/{child_frame}",
                    )
                )
                static_transforms.append(
                    value.is_static_tf(parent_frame, child_frame)
                )
        return BatchFrameTransformGraph(
            tf_list=transforms,
            bidirectional=True,
            static_tf=static_transforms,
        )

    def _buffer_for(
        self,
        path: str,
        source: torch.Tensor,
        *,
        pin_memory: bool,
    ) -> torch.Tensor:
        destination = self._buffers.get(path)
        reusable = (
            destination is not None
            and destination.shape == source.shape
            and destination.dtype == source.dtype
        )
        if not reusable:
            destination = torch.empty_like(
                source,
                device="cpu",
                pin_memory=pin_memory,
            )
            self._buffers[path] = destination
        return destination
