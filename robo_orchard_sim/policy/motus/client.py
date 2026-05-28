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

"""Client for Motus/DreamZero msgpack websocket policy servers."""

from __future__ import annotations
from typing import Any

import numpy as np

PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600


class MotusClient:
    """Thin client for the Motus/DreamZero websocket protocol."""

    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._ws = None
        self._packer = self._build_packer()
        self._metadata: dict[str, Any] = {}
        self._ensure_connected()

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def infer(self, obs: dict[str, Any]) -> np.ndarray:
        request = dict(obs)
        request["endpoint"] = "infer"
        self._ensure_connected()
        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Motus inference server error:\n{response}")
        return self._unpack(response)

    def reset(self, reset_info: dict[str, Any] | None = None) -> None:
        request = dict(reset_info or {})
        request["endpoint"] = "reset"
        self._ensure_connected()
        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, str) and response != "reset successful":
            raise RuntimeError(f"Motus reset failed:\n{response}")

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None

    def __enter__(self) -> "MotusClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _ensure_connected(self) -> None:
        if self._ws is not None:
            return

        try:
            import websockets.sync.client
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "websockets is required to use MotusClient"
            ) from exc

        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
        )
        self._metadata = self._unpack(self._ws.recv())

    @staticmethod
    def _build_packer():
        try:
            from openpi_client import msgpack_numpy
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "openpi_client is required to use MotusClient"
            ) from exc
        return msgpack_numpy.Packer()

    @staticmethod
    def _unpack(payload):
        from openpi_client import msgpack_numpy

        return msgpack_numpy.unpackb(payload)
