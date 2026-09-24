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

"""Canonical WebSocket policy service using only NumPy and websockets."""

from __future__ import annotations
import inspect
import json
import logging
import struct
import threading
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect
from websockets.sync.connection import Connection
from websockets.sync.server import serve

from robo_orchard_server.server.multiplex import PROTOCOL, unwrap, wrap
from robo_orchard_server.server.policy import BasePolicy, PolicyFactory
from robo_orchard_server.server.types import JointAction, Observation

LOGGER = logging.getLogger(__name__)
MAX_MESSAGE_BYTES = 200 * 1024 * 1024


def _decode_message(message: bytes | str) -> dict[str, Any]:
    """Decode the existing client's length-prefixed JSON and binary blocks."""
    if not isinstance(message, bytes):
        raise ValueError("Expected a binary WebSocket message")
    cursor = 0

    def read_block() -> bytes:
        nonlocal cursor
        if len(message) - cursor < 8:
            raise ValueError("Truncated block length")
        size = struct.unpack_from("!Q", message, cursor)[0]
        cursor += 8
        if size > len(message) - cursor:
            raise ValueError("Truncated binary block")
        block = message[cursor : cursor + size]
        cursor += size
        return block

    header = json.loads(read_block())
    blocks = []
    while cursor < len(message):
        blocks.append(read_block())

    def block_at(index: int) -> bytes:
        if type(index) is not int or not 0 <= index < len(blocks):
            raise ValueError("Invalid binary block index")
        return blocks[index]

    def restore(value: Any) -> Any:
        if isinstance(value, list):
            return [restore(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "__bytes_idx__" in value:
            return block_at(value["__bytes_idx__"])
        if "__tensor__" in value:
            return _decode_array(
                {
                    **value["__tensor__"],
                    "data": block_at(value["__tensor_idx__"]),
                }
            )
        restored = {key: restore(item) for key, item in value.items()}
        # Camera arrays have bare tensor payloads; state arrays have markers.
        if {"dtype", "shape", "data"} <= restored.keys():
            return _decode_array(restored)
        return restored

    request = restore(header)
    if not isinstance(request, dict):
        raise ValueError("Request must be a dictionary")
    return request


def _decode_array(payload: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(payload["dtype"])
    if dtype.kind not in "biufc":
        raise ValueError(f"Unsupported array dtype: {dtype}")
    shape = payload["shape"]
    if not isinstance(shape, list) or any(
        type(size) is not int or size < 0 for size in shape
    ):
        raise ValueError("Array shape must contain nonnegative integers")
    # Copy gives adapters writable arrays independent of the received bytes.
    return np.frombuffer(payload["data"], dtype=dtype).reshape(shape).copy()


def _encode_message(payload: dict[str, Any]) -> bytes:
    blocks: list[bytes] = []

    def extract(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            index = len(blocks)
            blocks.append(value.tobytes(order="C"))
            return {
                "__tensor__": {
                    "dtype": value.dtype.name,
                    "shape": list(value.shape),
                },
                "__tensor_idx__": index,
            }
        if isinstance(value, dict):
            return {key: extract(item) for key, item in value.items()}
        if isinstance(value, list):
            return [extract(item) for item in value]
        return value

    header = json.dumps(extract(payload), separators=(",", ":")).encode()
    return b"".join(
        struct.pack("!Q", len(block)) + block for block in [header, *blocks]
    )


def _action_contract(obs: Observation) -> tuple[list[str], int]:
    layout = obs["action_layout"]
    order = layout["manipulator_order"]
    if not order or len(set(order)) != len(order):
        raise ValueError("Layout must declare unique manipulator slots")
    if set(order) != set(layout["manipulators"]) or set(order) != set(
        obs["manipulators"]
    ):
        raise ValueError("Observation and layout manipulator slots differ")
    names: list[str] = []
    batches: set[int] = set()
    for slot in order:
        spec = layout["manipulators"][slot]
        names.extend(spec["arm_joint_names"])
        names.extend(spec["gripper_joint_names"])
        positions = obs["manipulators"][slot]["joint_position"]
        if not isinstance(positions, np.ndarray) or positions.ndim != 2:
            raise ValueError("joint_position must be a NumPy array [B, N]")
        batches.add(positions.shape[0])
    if not names or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("Layout must declare nonempty joint names")
    if len(names) != len(set(names)):
        raise ValueError("Layout joint names must be unique")
    if len(batches) != 1 or 0 in batches:
        raise ValueError("Manipulators must share a positive batch size")
    return names, batches.pop()


def _encode_action(
    action: JointAction, expected_names: list[str], batch_size: int
) -> dict[str, Any]:
    if not isinstance(action, dict):
        raise TypeError("Action must be a dictionary")
    names = action.get("joint_names")
    if not isinstance(names, list) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("joint_names must be a list of nonempty strings")
    if len(names) != len(set(names)) or set(names) != set(expected_names):
        raise ValueError("joint_names must contain every layout joint once")
    values = action.get("values")
    if not isinstance(values, np.ndarray) or values.dtype != np.float32:
        raise TypeError("Action values must be a float32 NumPy array")
    if values.shape != (batch_size, len(names)):
        raise ValueError(
            f"Action values must have shape [{batch_size}, {len(names)}], "
            f"got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Action values must be finite")
    # This marker is required for PolicyClient to build UnifiedJointCommand.
    return {"__joint_command__": {"joint_names": names, "values": values}}


def _validate_callable(function: object, label: str, *args: object) -> None:
    """Check synchronous call compatibility without executing model code."""
    if not callable(function):
        raise TypeError(f"{label} must be callable")
    try:
        implementation = inspect.unwrap(function)
        if not inspect.isroutine(implementation) and not inspect.isclass(
            implementation
        ):
            implementation = implementation.__call__
        for target in (function, implementation):
            if (
                inspect.iscoroutinefunction(target)
                or inspect.isasyncgenfunction(target)
                or inspect.isgeneratorfunction(target)
            ):
                raise TypeError(f"{label} must be synchronous")
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{label} requires a synchronous, inspectable Python callable: "
            f"{exc}"
        ) from exc
    try:
        signature.bind(*args)
    except TypeError as exc:
        raise TypeError(
            f"{label} must accept {len(args)} positional argument(s) "
            f"without additional required arguments: {exc}"
        ) from exc


def _call_sync(
    function: Callable[..., Any],
    label: str,
    *args: object,
    require_none: bool = False,
) -> Any:
    """Reject deferred results, including synchronous wrappers' results."""
    result = _synchronous_result(function(*args), label)
    if require_none and result is not None:
        raise TypeError(f"{label} must return None")
    return result


def _synchronous_result(result: Any, label: str) -> Any:
    """Reject deferred values without executing them."""
    if (
        inspect.isawaitable(result)
        or inspect.isasyncgen(result)
        or inspect.isgenerator(result)
    ):
        # Dispose of native coroutines without executing their bodies, so a
        # rejected adapter doesn't emit unawaited-coroutine warnings.
        if inspect.iscoroutine(result) or inspect.isgenerator(result):
            result.close()
        raise TypeError(f"{label} must return a synchronous result")
    return result


def _validate_policy(policy: object) -> BasePolicy:
    if not isinstance(policy, BasePolicy):
        raise TypeError(
            f"policy {type(policy).__name__} must inherit BasePolicy"
        )
    prefix = type(policy).__name__
    _validate_callable(policy.reset, f"{prefix}.reset()")
    _validate_callable(policy.act, f"{prefix}.act(obs)", object())
    _validate_callable(
        policy.act_sequence, f"{prefix}.act_sequence(obs)", object()
    )
    _validate_callable(policy.close, f"{prefix}.close()")
    return policy


def _close_candidate(candidate: object) -> None:
    """Attempt cleanup of factory results, including invalid adapters."""
    close = (
        candidate.close
        if isinstance(candidate, BasePolicy)
        else getattr(candidate, "close", None)
    )
    if close is not None:
        label = f"{type(candidate).__name__}.close()"
        _validate_callable(close, label)
        _call_sync(close, label, require_none=True)


class PolicyWebsocketServer:
    """Serve evaluators with a shared policy or independent Client sessions.

    Supply exactly one of policy (stateless shared mode) or policy_factory
    (a synchronous callable returning a fresh, initialized session). Policies
    inherit BasePolicy and implement reset() and act(obs). Override
    act_sequence(obs) for multi-step predictions that reduce round trips.
    Only factory-created sessions are closed by the server. Session state must
    live in the adapter, separately from any shared model weights.

    Factory calls, policy calls, validation, encoding and session cleanup run
    serially. Network I/O remains independent, without cross-connection FIFO.
    Errors are returned to the caller; later requests remain allowed.
    """

    def __init__(
        self,
        policy: BasePolicy | None = None,
        host: str = "0.0.0.0",
        port: int = 8765,
        logging_tag: str = "robo-orchard-server",
        *,
        policy_factory: PolicyFactory | None = None,
    ) -> None:
        """Configure the server with exactly one policy source.

        Args:
            policy: Initialized, stateless policy shared by all connections.
                Mutually exclusive with policy_factory.
            host: Interface address to listen on.
            port: WebSocket port to listen on.
            logging_tag: Tag included in responses.
            policy_factory: Synchronous callable invoked without arguments
                once per Client session, returning a fresh BasePolicy with
                reset() and act(obs). Pass the function itself, for example
                policy_factory=create_policy, not create_policy(). Bind
                model/config dependencies in a closure or functools.partial.
                The result must be ready to use: the server does not reset it
                on creation. Session caches must be independent, although
                model weights may be shared. On session closure it calls
                the session's close() method. Mutually exclusive
                with policy.

        Raises:
            ValueError: Both policy sources or neither are supplied.
            TypeError: The shared policy does not inherit BasePolicy, or a
                method/factory is asynchronous or has an incompatible call
                signature. Factory results are validated per session.
        """
        if (policy is None) == (policy_factory is None):
            raise ValueError("Provide exactly one of policy or policy_factory")
        if policy_factory is None:
            policy = _validate_policy(policy)
        else:
            _validate_callable(policy_factory, "policy_factory()")
        self.policy = policy
        self.policy_factory = policy_factory
        self.host = host
        self.port = port
        self.logging_tag = logging_tag
        self._connection_lock = threading.Condition()
        self._inference_lock = threading.Lock()
        self._connections: set[Connection] = set()
        self._stopping = False
        # Only live policy identities are tracked, under the inference lock.
        self._active_policies: set[int] = set()

    def _encode_response(self, response: dict[str, Any]) -> bytes:
        return _encode_message({**response, "logging_tag": self.logging_tag})

    def _encode_error(self, exc: Exception) -> bytes:
        return self._encode_response({"error": f"{type(exc).__name__}: {exc}"})

    def _cleanup_session(
        self, candidate: object, websocket: Connection
    ) -> None:
        """Release a session while the caller holds the inference lock."""
        try:
            _close_candidate(candidate)
        except Exception:
            LOGGER.exception(
                "Policy session cleanup failed id=%s", websocket.id
            )
        finally:
            self._active_policies.discard(id(candidate))

    def _dispatch(
        self, request: dict[str, Any], policy: BasePolicy
    ) -> dict[str, Any]:
        request_type = request.get("type")
        prefix = type(policy).__name__
        if request_type == "reset":
            _call_sync(policy.reset, f"{prefix}.reset()", require_none=True)
            return {"ok": True}
        if request_type not in ("act", "act_sequence"):
            raise ValueError(f"Unsupported request type: {request_type!r}")
        data = request["obs_data"]
        if data.get("format") != "canonical":
            raise ValueError("This server requires canonical observations")
        obs: Observation = {
            "instruction": data.get("instruction"),
            "cameras": data.get("cameras", {}),
            "manipulators": data["manipulators"],
            "action_layout": data["action_layout"],
        }
        if request.get("instruction") is not None:
            obs["instruction"] = request["instruction"]
        names, batch_size = _action_contract(obs)
        if request_type == "act":
            action = _call_sync(policy.act, f"{prefix}.act(obs)", obs)
            actions = _encode_action(action, names, batch_size)
        else:
            sequence = _call_sync(
                policy.act_sequence, f"{prefix}.act_sequence(obs)", obs
            )
            if not isinstance(sequence, list) or not sequence:
                raise ValueError("act_sequence must return a nonempty list")
            actions = [
                _encode_action(
                    _synchronous_result(
                        action, f"{prefix}.act_sequence(obs) item"
                    ),
                    names,
                    batch_size,
                )
                for action in sequence
            ]
        return {"actions": actions}

    def handle_client(self, websocket: Connection) -> None:
        """Create, serve and release this connection's policy session."""
        with self._connection_lock:
            stopping = self._stopping
            if not stopping:
                self._connections.add(websocket)
            connection_count = len(self._connections)
        if stopping:
            websocket.close(code=1001, reason="Server stopping")
            return
        LOGGER.info(
            "Evaluator connected id=%s active_connections=%d",
            websocket.id,
            connection_count,
        )
        policy: BasePolicy | None = None
        cleanup_candidate: object | None = None
        try:
            creation_error = None
            with self._inference_lock:
                with self._connection_lock:
                    if self._stopping:
                        return
                try:
                    if self.policy_factory is None:
                        policy = self.policy
                    else:
                        candidate = _call_sync(
                            self.policy_factory, "policy_factory()"
                        )
                        if id(candidate) in self._active_policies:
                            raise ValueError(
                                "Factory policy is already in use"
                            )
                        cleanup_candidate = candidate
                        policy = _validate_policy(candidate)
                        self._active_policies.add(id(policy))
                        LOGGER.info(
                            "Policy session created id=%s", websocket.id
                        )
                except Exception as exc:
                    LOGGER.exception(
                        "Policy session creation failed id=%s", websocket.id
                    )
                    creation_error = self._encode_error(exc)
                    if cleanup_candidate is not None:
                        # Invalid instances don't need to wait for a failed
                        # client's closing handshake to release resources.
                        self._cleanup_session(cleanup_candidate, websocket)
                        cleanup_candidate = None
            if creation_error is not None:
                websocket.send(creation_error)
                # Older sync clients can lose the error if the close frame
                # arrives during their opening handshake. A control-frame
                # round trip after the data frame lets them finish opening
                # before close, without changing the application protocol.
                websocket.ping().wait(timeout=5)
                websocket.close(
                    code=1011, reason="Policy session creation failed"
                )
                return
            assert policy is not None
            for message in websocket:
                try:
                    request = _decode_message(message)
                    with self._inference_lock:
                        with self._connection_lock:
                            if self._stopping:
                                return
                        # Encode before another policy call can reuse buffers.
                        encoded = self._encode_response(
                            self._dispatch(request, policy)
                        )
                except Exception as exc:
                    LOGGER.exception(
                        "Policy request failed id=%s", websocket.id
                    )
                    with self._inference_lock:
                        encoded = self._encode_error(exc)
                websocket.send(encoded)
        except ConnectionClosed:
            pass
        finally:
            if cleanup_candidate is not None:
                with self._inference_lock:
                    self._cleanup_session(cleanup_candidate, websocket)
            with self._connection_lock:
                self._connections.discard(websocket)
                connection_count = len(self._connections)
                self._connection_lock.notify_all()
            LOGGER.info(
                "Evaluator disconnected id=%s active_connections=%d",
                websocket.id,
                connection_count,
            )

    def handle_sessions(self, websocket: Connection) -> None:
        """Serve logical sessions in receive order on one connection.

        Each factory-created policy belongs to exactly one logical Client.
        Every request finishes before the next is read, including reset and
        close_session; a slow inference cannot interleave model state.
        """
        sessions: dict[str, BasePolicy] = {}
        with self._connection_lock:
            if self._stopping:
                websocket.close(code=1001, reason="Server stopping")
                return
            self._connections.add(websocket)
        try:
            for message in websocket:
                session_id, request_id, payload = unwrap(message)
                with self._inference_lock:
                    with self._connection_lock:
                        if self._stopping:
                            return
                    try:
                        request = _decode_message(payload)
                        kind = request.get("type")
                        if kind == "close_session":
                            policy = sessions.pop(session_id)
                            if self.policy_factory is not None:
                                self._cleanup_session(policy, websocket)
                            response = {"ok": True}
                        else:
                            if session_id not in sessions:
                                if kind != "reset":
                                    raise ValueError("New session must reset")
                                if len(sessions) >= 64:
                                    raise ValueError(
                                        "Session capacity exceeded"
                                    )
                                candidate = self.policy
                                if self.policy_factory is not None:
                                    candidate = _call_sync(
                                        self.policy_factory, "policy_factory()"
                                    )
                                    if id(candidate) in self._active_policies:
                                        raise ValueError(
                                            "Policy already in use"
                                        )
                                    try:
                                        policy = _validate_policy(candidate)
                                    except Exception:
                                        self._cleanup_session(
                                            candidate, websocket
                                        )
                                        raise
                                    self._active_policies.add(id(policy))
                                else:
                                    policy = candidate
                                sessions[session_id] = policy
                            response = self._dispatch(
                                request, sessions[session_id]
                            )
                        encoded = self._encode_response(response)
                    except Exception as exc:
                        LOGGER.exception("Policy session request failed")
                        encoded = self._encode_error(exc)
                    response_frame = wrap(encoded, session_id, request_id)
                websocket.send(response_frame)
        except ConnectionClosed:
            pass
        finally:
            with self._inference_lock:
                if self.policy_factory is not None:
                    for policy in sessions.values():
                        self._cleanup_session(policy, websocket)
            with self._connection_lock:
                self._connections.discard(websocket)
                self._connection_lock.notify_all()

    def run_reverse(
        self,
        uri: str,
        *,
        team_id: str,
        evaluation_id: str,
    ) -> None:
        """Authenticate one socket and serve sessions-v1 requests serially."""
        if urlsplit(uri).scheme not in {"ws", "wss"}:
            raise ValueError("Expected a supervisor WebSocket URL")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (team_id, evaluation_id)
        ):
            raise ValueError("Nonempty team_id and evaluation_id are required")

        def receive_control(websocket: Connection, expected: dict) -> None:
            message = websocket.recv()
            if not isinstance(message, str) or len(message.encode()) > 4096:
                raise ValueError("Invalid supervisor control message")
            try:
                payload = json.loads(message)
            except (ValueError, RecursionError):
                raise ValueError("Invalid supervisor control JSON") from None
            if payload != expected:
                raise ValueError("Unsupported supervisor control message")

        with connect(
            uri,
            max_size=MAX_MESSAGE_BYTES,
            compression=None,
            open_timeout=10,
            close_timeout=1,
        ) as websocket:
            try:
                receive_control(websocket, {"type": "authenticate"})
                websocket.send(
                    json.dumps(
                        {
                            "type": "authenticate",
                            "team_id": team_id,
                            "evaluation_id": evaluation_id,
                        }
                    )
                )
                receive_control(websocket, {"type": "authenticated"})
                receive_control(
                    websocket,
                    {
                        "type": "policy_protocol",
                        "protocol": PROTOCOL,
                    },
                )
                websocket.send(
                    json.dumps(
                        {
                            "type": "policy_protocol_ready",
                            "protocol": PROTOCOL,
                        }
                    )
                )
            except Exception:
                websocket.close(
                    code=1008, reason="Authentication or protocol failed"
                )
                raise
            self.handle_sessions(websocket)

    def run(self) -> None:
        """Listen until interrupted, then close and drain all sessions."""
        with serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=MAX_MESSAGE_BYTES,
            compression=None,
        ) as server:
            LOGGER.info("Listening on ws://%s:%s", self.host, self.port)
            try:
                server.serve_forever()
            finally:
                # Wake idle handlers without holding the connection lock
                # during I/O. In-progress model calls finish naturally; queued
                # factory/action calls must not start after this boundary.
                with self._connection_lock:
                    self._stopping = True
                    connections = tuple(self._connections)
                for connection in connections:
                    connection.close(code=1001, reason="Server stopping")
                with self._connection_lock:
                    self._connection_lock.wait_for(
                        lambda: not self._connections
                    )
