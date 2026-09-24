#
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

"""sessions-v1 framing shared by independently installed peer packages.

Only the JSON header changes; tensor block indices and bytes stay untouched.
Keep this small wire codec identical on both sides of the connection.
"""

import json
import struct

PROTOCOL = "sessions-v1"
MAX_SIZE = 200 * 1024 * 1024
ROUTE_KEY = "__supervisor__"


def _header(message: bytes) -> tuple[dict, bytes]:
    if not isinstance(message, bytes) or not 8 <= len(message) <= MAX_SIZE:
        raise ValueError("Expected a bounded binary message")
    size = struct.unpack_from("!Q", message)[0]
    if size > 1024 * 1024 or size + 8 > len(message):
        raise ValueError("Invalid routing header size")
    header = json.loads(message[8 : 8 + size])
    if not isinstance(header, dict):
        raise ValueError("Message header must be an object")
    return header, message[8 + size :]


def _encode(header: dict, blocks: bytes) -> bytes:
    data = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    result = struct.pack("!Q", len(data)) + data + blocks
    if len(data) > 1024 * 1024 or len(result) > MAX_SIZE:
        raise ValueError("Routed message exceeds size limit")
    return result


def _identity(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise ValueError("Invalid routing identity")
    return value


def wrap(message: bytes, session_id: str, request_id: str) -> bytes:
    """Add request routing without decoding tensor data."""
    header, blocks = _header(message)
    if ROUTE_KEY in header:
        raise ValueError("Message already contains routing metadata")
    header[ROUTE_KEY] = {
        "protocol": PROTOCOL,
        "session_id": _identity(session_id),
        "request_id": _identity(request_id),
    }
    return _encode(header, blocks)


def unwrap(message: bytes) -> tuple[str, str, bytes]:
    """Validate routing and return the original policy frame."""
    header, blocks = _header(message)
    route = header.pop(ROUTE_KEY, None)
    if (
        not isinstance(route, dict)
        or set(route) != {"protocol", "session_id", "request_id"}
        or route["protocol"] != PROTOCOL
    ):
        raise ValueError("Missing or unsupported routing metadata")
    return (
        _identity(route["session_id"]),
        _identity(route["request_id"]),
        _encode(header, blocks),
    )
