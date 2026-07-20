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

# INTERNAL

"""Client for openpi-protocol websocket policy servers (Cosmos)."""

from __future__ import annotations
from typing import Any

PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600


class CosmosWsClient:
    """openpi websocket protocol: metadata on connect, msgpack obs/action."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._ws = None
        self._packer = self._build_packer()
        self._metadata: dict[str, Any] = {}
        self._ensure_connected()

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def infer(self, request: dict[str, Any]) -> Any:
        self._ensure_connected()
        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Cosmos inference server error:\n{response}")
        return self._unpack(response)

    def reset(self) -> None:
        self.close()
        self._ensure_connected()

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None

    def __enter__(self) -> "CosmosWsClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ensure_connected(self) -> None:
        if self._ws is not None:
            return

        try:
            import websockets.sync.client
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "websockets is required to use CosmosWsClient"
            ) from exc

        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
            proxy=None,
        )
        self._metadata = self._unpack(self._ws.recv())

    @staticmethod
    def _build_packer():
        try:
            from openpi_client import msgpack_numpy
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "openpi_client is required to use CosmosWsClient"
            ) from exc
        return msgpack_numpy.Packer()

    @staticmethod
    def _unpack(payload):
        from openpi_client import msgpack_numpy

        return msgpack_numpy.unpackb(payload)
