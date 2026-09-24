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

"""DreamZero websocket client."""

from robo_orchard_sim.policy.motus.client import MotusClient


class DreamZeroClient(MotusClient):
    """Motus-compatible client using DreamZero's websocket transport."""

    def _ensure_connected(self) -> None:
        if self._ws is not None:
            return

        try:
            import websockets.sync.client
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "websockets is required to use DreamZeroClient"
            ) from exc

        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
        )
        self._metadata = self._unpack(self._ws.recv())


__all__ = ["DreamZeroClient"]
