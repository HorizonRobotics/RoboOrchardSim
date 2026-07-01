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

"""Thin ZMQ client for the GR00T inference server."""

from __future__ import annotations
from typing import Any


class GrootZmqClient:
    """Query a GR00T policy server over ZMQ REQ/REP."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ) -> None:
        import msgpack_numpy as mnp
        import zmq

        self._zmq = zmq
        self._mnp = mnp
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._context = zmq.Context()
        self._socket = None
        self._connect()

    def _connect(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(self._zmq.REQ)
        self._socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        *,
        requires_input: bool = True,
    ) -> Any:
        """Send one request and return the deserialized response."""
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token
        assert self._socket is not None
        try:
            self._socket.send(self._mnp.packb(request))
            message = self._socket.recv()
        except self._zmq.error.Again:
            # Timeout leaves the REQ socket unusable; rebuild for next call.
            self._connect()
            raise
        response = self._mnp.unpackb(message, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return ``(action, info)`` for one observation."""
        response = self.call_endpoint(
            "get_action",
            {"observation": observation, "options": options},
        )
        action, info = response
        return action, info

    def reset(self, options: dict[str, Any] | None = None) -> Any:
        """Reset server-side policy state."""
        return self.call_endpoint("reset", {"options": options})

    def ping(self) -> bool:
        """Return True if the server responds."""
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except self._zmq.error.ZMQError:
            self._connect()
            return False

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._context.term()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
