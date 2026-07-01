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

"""WebSocket-based policy server and client for remote model inference.

The server loads a policy (for example Holobrain) and exposes it over
WebSocket. The remote client is wrapped in ``ServerPolicy`` so the evaluator
can use a remote model through the same ``PolicyMixin`` interface as a local
policy.

Server usage::

    python -m robo_orchard_sim.policy.server \
        --model-type holobrain \
        --model-yaml path/to/holobrain.yaml \
        --host 0.0.0.0 --port 8765

Client usage (in eval yaml)::

    model_cfg:
      policy: server
      server_host: <host>
      server_port: 8765
"""

import asyncio
import json
import logging
import struct
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import yaml
from robo_orchard_core.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_core.utils.config import ClassType
from robo_orchard_core.utils.logging import LoggerManager

from robo_orchard_sim.contracts.joint_command import UnifiedJointCommand
from robo_orchard_sim.contracts.policy_binding import (
    CanonicalPolicyInput,
    PolicyRequirement,
)
from robo_orchard_sim.policy.action_layout import CompiledActionLayout

logger = LoggerManager().get_child(__name__)

_MAX_MSG = 200 * 1024 * 1024  # 200 MB – large enough for image payloads
_POLICY_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
_HEADER_SIZE_BYTES = 8


# ---------------------------------------------------------------------------
# Proxy objects – reconstruct observations on the server side so that
# existing policy preprocessing code works unchanged.
# ---------------------------------------------------------------------------


class _SensorProxy:
    """Mimics an Isaac sensor with .sensor_data / .intrinsic_matrices."""

    def __init__(self, sensor_data, intrinsic_matrices=None, pose=None):
        self.sensor_data = sensor_data
        self.intrinsic_matrices = intrinsic_matrices
        self.pose = pose


class _PoseProxy:
    def __init__(self, xyz, quat):
        self.xyz = xyz
        self.quat = quat


def _rebuild_observations(obs_data: dict) -> dict:
    """Rebuild wire-format observations into policy input structure."""
    if obs_data.get("format") == "canonical":
        return _rebuild_canonical_observations(obs_data)
    remote_policy_type = obs_data.get("format", "full")
    if remote_policy_type != "full":
        return _rebuild_profiled_observations(
            obs_data,
            observation_fields=_resolve_observation_fields(remote_policy_type),
        )

    observations: dict[str, Any] = {}
    if "cameras" in obs_data:
        cam_dict: dict[str, Any] = {}
        for term, d in obs_data["cameras"].items():
            pose_data = d.get("pose")
            pose = None
            if pose_data is not None:
                pose = _PoseProxy(
                    xyz=_decode_tensor_payload(pose_data["xyz"]),
                    quat=_decode_tensor_payload(pose_data["quat"]),
                )
            rgb = _SensorProxy(
                _decode_tensor_payload(d["rgb"]),
                _decode_tensor_payload(d["intrinsic_matrices"])
                if "intrinsic_matrices" in d
                else None,
                pose=pose,
            )
            depth = _SensorProxy(_decode_tensor_payload(d["depth"]))
            cam_dict[term] = {"rgb": rgb, "depth": depth}
        observations["/camera"] = cam_dict
    if "robot" in obs_data:
        observations["/robot"] = _decode_value(obs_data["robot"])
    return observations


def _rebuild_profiled_observations(
    obs_data: dict,
    *,
    observation_fields: dict[str, Any],
) -> dict:
    observations: dict[str, Any] = {}
    if "cameras" in obs_data:
        cam_dict: dict[str, Any] = {}
        for term, d in obs_data["cameras"].items():
            camera_obs: dict[str, Any] = {}
            pose = None
            if observation_fields["include_pose"] and "pose" in d:
                pose = _PoseProxy(
                    xyz=_decode_tensor_payload(d["pose"]["xyz"]),
                    quat=_decode_tensor_payload(d["pose"]["quat"]),
                )
            if observation_fields["include_rgb"] and "rgb" in d:
                camera_obs["rgb"] = _SensorProxy(
                    _decode_tensor_payload(d["rgb"]),
                    _decode_tensor_payload(d["intrinsic_matrices"])
                    if observation_fields["include_intrinsic"]
                    and "intrinsic_matrices" in d
                    else None,
                    pose=pose,
                )
            if observation_fields["include_depth"] and "depth" in d:
                camera_obs["depth"] = _SensorProxy(
                    _decode_tensor_payload(d["depth"])
                )
            cam_dict[term] = camera_obs
        observations["/camera"] = cam_dict
    if "robot" in obs_data:
        observations["/robot"] = _decode_value(obs_data["robot"])
    return observations


def _resolve_observation_fields(remote_policy_type: str) -> dict[str, Any]:
    match remote_policy_type:
        case "openpi":
            from robo_orchard_sim.policy.openpi.adapter import OpenPiAdapter

            return OpenPiAdapter.required_observation_fields()
        case _:
            raise ValueError(
                "Profiled remote observation extraction is unsupported "
                f"for remote policy type: {remote_policy_type}"
            )


def _rebuild_canonical_observations(obs_data: dict) -> CanonicalPolicyInput:
    cameras: dict[str, dict[str, Any]] = {}
    for slot, d in obs_data.get("cameras", {}).items():
        pose = None
        if "pose" in d:
            pose = _PoseProxy(
                xyz=_decode_tensor_payload(d["pose"]["xyz"]),
                quat=_decode_tensor_payload(d["pose"]["quat"]),
            )
        camera_obs: dict[str, Any] = {}
        if "rgb" in d:
            camera_obs["rgb"] = _SensorProxy(
                _decode_tensor_payload(d["rgb"]),
                _decode_tensor_payload(d["intrinsic_matrices"])
                if "intrinsic_matrices" in d
                else None,
                pose=pose,
            )
        if "depth" in d:
            camera_obs["depth"] = _SensorProxy(
                _decode_tensor_payload(d["depth"])
            )
        cameras[slot] = camera_obs

    manipulators = {
        slot: _decode_value(manipulator_obs)
        for slot, manipulator_obs in obs_data.get("manipulators", {}).items()
    }
    return CanonicalPolicyInput(
        instruction=obs_data.get("instruction"),
        cameras=cameras,
        manipulators=manipulators,
        action_layout=(
            CompiledActionLayout.from_payload(obs_data["action_layout"])
            if "action_layout" in obs_data
            else None
        ),
    )


def _encode_tensor_payload(value: torch.Tensor) -> dict[str, Any]:
    cpu_value = value.detach().cpu()
    return {
        "dtype": str(cpu_value.dtype).removeprefix("torch."),
        "shape": list(cpu_value.shape),
        "data": memoryview(cpu_value.contiguous().numpy()).tobytes(),
    }


def _decode_tensor_payload(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    if isinstance(payload["data"], (bytes, bytearray, memoryview)):
        return torch.frombuffer(
            bytearray(payload["data"]),
            dtype=dtype,
        ).reshape(payload["shape"])
    return torch.tensor(payload["data"], dtype=dtype).reshape(payload["shape"])


def _extract_binary_tensors(value: Any, tensors: list[bytes]) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        tensor_idx = len(tensors)
        tensors.append(bytes(value))
        return {"__bytes_idx__": tensor_idx}
    if isinstance(value, dict):
        if "__tensor__" in value:
            tensor_payload = dict(value["__tensor__"])
            tensor_data = tensor_payload.pop("data")
            if isinstance(tensor_data, (bytes, bytearray, memoryview)):
                tensor_bytes = bytes(tensor_data)
            else:
                dtype = getattr(torch, tensor_payload["dtype"])
                tensor_bytes = (
                    torch.tensor(tensor_data, dtype=dtype)
                    .contiguous()
                    .numpy()
                    .tobytes()
                )
            tensor_idx = len(tensors)
            tensors.append(tensor_bytes)
            return {"__tensor__": tensor_payload, "__tensor_idx__": tensor_idx}
        return {
            key: _extract_binary_tensors(item, tensors)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_extract_binary_tensors(item, tensors) for item in value]
    return value


def _inject_binary_tensors(value: Any, tensors: list[bytes]) -> Any:
    if isinstance(value, dict):
        if "__bytes_idx__" in value:
            return tensors[value["__bytes_idx__"]]
        if "__tensor__" in value:
            tensor_payload = dict(value["__tensor__"])
            tensor_payload["data"] = tensors[value["__tensor_idx__"]]
            return {"__tensor__": tensor_payload}
        return {
            key: _inject_binary_tensors(item, tensors)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_inject_binary_tensors(item, tensors) for item in value]
    return value


def _encode_binary_message(payload: dict[str, Any]) -> bytes:
    tensors: list[bytes] = []
    header_payload = _extract_binary_tensors(payload, tensors)
    header_json = json.dumps(header_payload, separators=(",", ":")).encode(
        "utf-8"
    )
    tensor_blob = b"".join(
        struct.pack("!Q", len(tensor_bytes)) + tensor_bytes
        for tensor_bytes in tensors
    )
    return struct.pack("!Q", len(header_json)) + header_json + tensor_blob


def _decode_binary_message(message: bytes) -> dict[str, Any]:
    header_len = struct.unpack("!Q", message[:_HEADER_SIZE_BYTES])[0]
    header_start = _HEADER_SIZE_BYTES
    header_end = header_start + header_len
    header_payload = json.loads(
        message[header_start:header_end].decode("utf-8")
    )
    tensors: list[bytes] = []
    cursor = header_end
    while cursor < len(message):
        tensor_len = struct.unpack("!Q", message[cursor : cursor + 8])[0]
        cursor += 8
        tensors.append(message[cursor : cursor + tensor_len])
        cursor += tensor_len
    return _inject_binary_tensors(header_payload, tensors)


def _encode_value(value: Any) -> Any:
    if isinstance(value, UnifiedJointCommand):
        return {
            "__joint_command__": {
                "values": {"__tensor__": _encode_tensor_payload(value.values)},
                "joint_names": list(value.joint_names),
            }
        }
    if isinstance(value, torch.Tensor):
        return {"__tensor__": _encode_tensor_payload(value)}
    if isinstance(value, dict):
        return {key: _encode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, tuple):
        return [_encode_value(item) for item in value]
    return value


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "__joint_command__" in value:
            payload = value["__joint_command__"]
            return UnifiedJointCommand(
                values=_decode_tensor_payload(payload["values"]["__tensor__"]),
                joint_names=tuple(payload["joint_names"]),
            )
        if "__tensor__" in value:
            return _decode_tensor_payload(value["__tensor__"])
        return {key: _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _move_to_device(value: Any, device: str) -> Any:
    if isinstance(value, UnifiedJointCommand):
        return UnifiedJointCommand(
            values=value.values.to(device),
            joint_names=value.joint_names,
        )
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {
            key: _move_to_device(item, device) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _extract_obs_data(
    observations: dict | CanonicalPolicyInput,
    *,
    remote_policy_type: str = "full",
) -> dict:
    """Convert live observations into a pickle-safe dict of CPU tensors."""
    if isinstance(observations, CanonicalPolicyInput):
        return _extract_canonical_obs_data(observations)
    if remote_policy_type != "full":
        raise ValueError(
            "Remote policy type "
            f"{remote_policy_type!r} requires CanonicalPolicyInput "
            "observations."
        )

    obs: dict[str, Any] = {}
    if "/camera" in observations:
        cameras: dict[str, dict] = {}
        for term, td in observations["/camera"].items():
            out = td.get("output", td)
            cam: dict[str, Any] = {
                "rgb": _encode_tensor_payload(out["rgb"].sensor_data),
                "depth": _encode_tensor_payload(out["depth"].sensor_data),
            }
            m = getattr(out["rgb"], "intrinsic_matrices", None)
            if m is not None:
                cam["intrinsic_matrices"] = _encode_tensor_payload(m)
            pose = getattr(out["rgb"], "pose", None)
            if pose is not None:
                cam["pose"] = {
                    "xyz": _encode_tensor_payload(pose.xyz),
                    "quat": _encode_tensor_payload(pose.quat),
                }
            cameras[term] = cam
        obs["cameras"] = cameras
    if "/robot" in observations:
        robot: dict[str, Any] = {}
        for k, v in observations["/robot"].items():
            robot[k] = _encode_value(v)
        obs["robot"] = robot
    return obs


def _extract_canonical_obs_data(
    observations: CanonicalPolicyInput,
) -> dict[str, Any]:
    obs: dict[str, Any] = {
        "format": "canonical",
        "instruction": observations.instruction,
    }
    if observations.action_layout is not None:
        obs["action_layout"] = observations.action_layout.to_payload()
    cameras: dict[str, dict[str, Any]] = {}
    for slot, camera_obs in observations.cameras.items():
        cam: dict[str, Any] = {}
        if "rgb" in camera_obs:
            cam["rgb"] = _encode_tensor_payload(camera_obs["rgb"].sensor_data)
            intrinsic_matrices = getattr(
                camera_obs["rgb"],
                "intrinsic_matrices",
                None,
            )
            if intrinsic_matrices is not None:
                cam["intrinsic_matrices"] = _encode_tensor_payload(
                    intrinsic_matrices
                )
            pose = getattr(camera_obs["rgb"], "pose", None)
            if pose is not None:
                cam["pose"] = {
                    "xyz": _encode_tensor_payload(pose.xyz),
                    "quat": _encode_tensor_payload(pose.quat),
                }
        if "depth" in camera_obs:
            cam["depth"] = _encode_tensor_payload(
                camera_obs["depth"].sensor_data
            )
        cameras[slot] = cam
    obs["cameras"] = cameras

    manipulators: dict[str, Any] = {}
    for slot, manipulator_obs in observations.manipulators.items():
        manipulators[slot] = _encode_value(manipulator_obs)
    obs["manipulators"] = manipulators
    return obs


def _extract_profiled_obs_data(
    observations: dict,
    *,
    remote_policy_type: str,
    observation_fields: dict[str, Any],
) -> dict:
    obs: dict[str, Any] = {"format": remote_policy_type}
    if "/camera" in observations:
        cameras: dict[str, dict] = {}
        for term in observation_fields["camera_terms"]:
            td = observations["/camera"][term]
            out = td.get("output", td)
            cam: dict[str, Any] = {}
            if observation_fields["include_rgb"]:
                cam["rgb"] = _encode_tensor_payload(out["rgb"].sensor_data)
            if observation_fields["include_depth"]:
                cam["depth"] = _encode_tensor_payload(out["depth"].sensor_data)
            if observation_fields["include_intrinsic"]:
                m = getattr(out["rgb"], "intrinsic_matrices", None)
                if m is not None:
                    cam["intrinsic_matrices"] = _encode_tensor_payload(m)
            if observation_fields["include_pose"]:
                pose = getattr(out["rgb"], "pose", None)
                if pose is not None:
                    cam["pose"] = {
                        "xyz": _encode_tensor_payload(pose.xyz),
                        "quat": _encode_tensor_payload(pose.quat),
                    }
            cameras[term] = cam
        obs["cameras"] = cameras
    if "/robot" in observations:
        robot: dict[str, Any] = {}
        for k in observation_fields["robot_keys"]:
            v = observations["/robot"][k]
            robot[k] = _encode_value(v)
        obs["robot"] = robot
    return obs


def _resolve_policy_requirement(
    remote_policy_type: str,
) -> PolicyRequirement | None:
    match remote_policy_type:
        case "holobrain":
            from robo_orchard_sim.policy.holobrain.policy import (
                HolobrainPolicy,
            )

            return HolobrainPolicy.policy_requirement()
        case _:
            return None


def _apply_instruction(
    observations: dict[str, Any] | CanonicalPolicyInput,
    instruction: str | None,
) -> dict[str, Any] | CanonicalPolicyInput:
    if instruction is None:
        return observations
    if isinstance(observations, CanonicalPolicyInput):
        return observations.model_copy(update={"instruction": instruction})
    observations = dict(observations)
    observations["instruction"] = instruction
    return observations


class PolicyWebsocketServer:
    """WebSocket server that hosts a policy for remote inference."""

    def __init__(
        self,
        policy: PolicyMixin,
        host: str = "0.0.0.0",
        port: int = 8765,
        logging_tag: str | None = None,
    ):
        self.policy = policy
        self.host = host
        self.port = port
        self.logging_tag = logging_tag or f"{host}:{port}"

    async def _handle(self, websocket):
        await self.handle_client(websocket)

    async def handle_client(self, websocket):
        """Serve one connected websocket client."""
        remote = websocket.remote_address
        logger.info(
            "[%s] Client connected from %s",
            self.logging_tag,
            remote,
        )
        try:
            async for message in websocket:
                try:
                    req = _decode_binary_message(message)
                    req_type = req.get("type", "act")
                    if req_type == "reset":
                        self.policy.reset()
                        await websocket.send(
                            _encode_binary_message(
                                {
                                    "ok": True,
                                    "logging_tag": self.logging_tag,
                                }
                            )
                        )
                        continue

                    if req_type not in {"act", "act_sequence"}:
                        raise ValueError(
                            f"Unsupported request type: {req_type}"
                        )

                    obs = _rebuild_observations(req["obs_data"])
                    obs = _apply_instruction(
                        obs,
                        req.get("instruction"),
                    )
                    if req_type == "act_sequence":
                        actions = self._act_sequence(obs)
                    else:
                        actions = self.policy.act(obs)
                    response = _encode_binary_message(
                        {
                            "actions": _encode_value(actions),
                            "logging_tag": self.logging_tag,
                        }
                    )
                    await websocket.send(response)
                except Exception:
                    logger.exception(
                        "[%s] Inference error",
                        self.logging_tag,
                    )
                    await websocket.send(
                        _encode_binary_message(
                            {
                                "error": "inference failed",
                                "logging_tag": self.logging_tag,
                            }
                        )
                    )
        finally:
            logger.info(
                "[%s] Client disconnected from %s",
                self.logging_tag,
                remote,
            )

    async def serve_async(self):
        try:
            import websockets
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "websockets is required to run the policy server"
            ) from exc

        async with websockets.serve(
            self.handle_client, self.host, self.port, max_size=_MAX_MSG
        ):
            logger.info(
                "[%s] Policy server listening on ws://%s:%s",
                self.logging_tag,
                self.host,
                self.port,
            )
            await asyncio.Future()

    def run(self):
        """Blocking entry-point."""
        asyncio.run(self.serve_async())

    def _act_sequence(
        self,
        obs: dict[str, Any],
    ) -> list["RemoteAction"]:
        act_sequence = getattr(self.policy, "act_sequence", None)
        if callable(act_sequence):
            actions = act_sequence(obs)
            if not isinstance(actions, list):
                raise TypeError("Policy act_sequence must return a list")
            return actions
        return [self.policy.act(obs)]


class PolicyClient:
    """WebSocket client for remote policy inference via server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        logging_tag: str | None = None,
        remote_policy_type: str = "full",
    ):
        self._url = f"ws://{host}:{port}"
        self._ws = None
        self.logging_tag = logging_tag or f"client->{host}:{port}"
        self._remote_policy_type = remote_policy_type

    def _ensure_connected(self):
        if self._ws is None:
            try:
                from websockets.sync.client import connect
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "websockets is required to use the server policy client"
                ) from exc

            self._ws = connect(self._url, max_size=_MAX_MSG)
            logger.info(
                "[%s] Connected to policy server at %s",
                self.logging_tag,
                self._url,
            )

    def _send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        self._ensure_connected()
        self._ws.send(_encode_binary_message(request))
        response = _decode_binary_message(self._ws.recv())
        if "logging_tag" in response:
            self.logging_tag = response["logging_tag"]
        if "error" in response:
            raise RuntimeError(f"Remote policy error: {response['error']}")
        return response

    @staticmethod
    def _resolve_instruction(
        observations: dict[str, Any] | CanonicalPolicyInput,
        instruction: str | None,
    ) -> str | None:
        if instruction is not None:
            return instruction
        if isinstance(observations, CanonicalPolicyInput):
            return observations.instruction
        return observations.get("instruction")

    def request_action(
        self,
        observations: dict[str, Any] | CanonicalPolicyInput,
        *,
        instruction: str | None = None,
    ) -> "RemoteAction":
        """Send observations to server and return predicted actions."""
        instruction = self._resolve_instruction(observations, instruction)
        req = {
            "type": "act",
            "obs_data": _extract_obs_data(
                observations,
                remote_policy_type=self._remote_policy_type,
            ),
            "instruction": instruction,
        }
        resp = self._send_request(req)
        actions = _decode_value(resp["actions"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        actions = _move_to_device(actions, device)
        return actions

    def request_action_sequence(
        self,
        observations: dict[str, Any] | CanonicalPolicyInput,
        *,
        instruction: str | None = None,
    ) -> list["RemoteAction"]:
        """Send observations to server and return predicted actions."""
        instruction = self._resolve_instruction(observations, instruction)
        req = {
            "type": "act_sequence",
            "obs_data": _extract_obs_data(
                observations,
                remote_policy_type=self._remote_policy_type,
            ),
            "instruction": instruction,
        }
        resp = self._send_request(req)
        actions = _decode_value(resp["actions"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        actions = _move_to_device(actions, device)
        if isinstance(actions, list):
            return actions
        return [actions]

    def reset_remote_policy(self) -> None:
        """Reset remote policy state such as cached action horizons."""
        self._send_request({"type": "reset"})

    def close(self):
        if self._ws:
            self._ws.close()
            self._ws = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


RemoteAction = UnifiedJointCommand | dict[str, torch.Tensor] | torch.Tensor


class ServerPolicy(PolicyMixin[dict[str, Any], RemoteAction]):
    """Policy adapter that forwards inference to a remote websocket server."""

    cfg: "ServerPolicyCfg"

    def __init__(
        self,
        cfg: "ServerPolicyCfg",
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            observation_space=observation_space,
            action_space=action_space,
        )
        self._client = self._build_client(cfg)
        self._cached_actions: list[RemoteAction] = []
        self._cached_index = 0

    def reset(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._cached_actions = []
        self._cached_index = 0
        self._client.reset_remote_policy()

    def act(self, obs: dict[str, Any]) -> RemoteAction:
        if self._cached_index >= len(self._cached_actions):
            self._cached_actions = self._client.request_action_sequence(obs)
            self._cached_index = 0
        action = self._cached_actions[self._cached_index]
        self._cached_index += 1
        return action

    @property
    def logging_tag(self) -> str:
        """Return the best-known logging tag for diagnostics."""
        return self._client.logging_tag

    @staticmethod
    def _build_client(cfg: "ServerPolicyCfg") -> PolicyClient:
        return PolicyClient(
            host=cfg.host,
            port=cfg.port,
            logging_tag=cfg.logging_tag,
            remote_policy_type=cfg.remote_policy_type,
        )

    def policy_requirement(self) -> PolicyRequirement | None:
        return _resolve_policy_requirement(self.cfg.remote_policy_type)

    def close(self) -> None:
        self._client.close()

    def __del__(self):
        self.close()


class ServerPolicyCfg(PolicyConfig[ServerPolicy]):
    """Config for :class:`ServerPolicy`."""

    class_type: ClassType[ServerPolicy] = ServerPolicy

    host: str = "localhost"
    port: int = 8765
    logging_tag: str | None = None
    remote_policy_type: str = "full"


def _build_server_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Policy WebSocket Server")
    parser.add_argument(
        "--model-type",
        required=True,
        help="Policy type to load, for example holobrain or dummy.",
    )
    parser.add_argument(
        "--model-yaml",
        default=None,
        help=(
            "Optional policy-specific yaml. If omitted, the server loads "
            "robo_orchard_sim/policy/configs/<model-type>.yaml."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def _resolve_policy_cfg_yaml(
    model_type: str,
    model_yaml: str | None,
) -> Path:
    if model_yaml is not None:
        return Path(model_yaml)
    return _POLICY_CONFIG_DIR / f"{model_type}.yaml"


def _load_policy_model_cfg(args) -> dict[str, Any]:
    config_path = _resolve_policy_cfg_yaml(
        model_type=args.model_type,
        model_yaml=args.model_yaml,
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Policy config yaml not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Policy config yaml must contain a mapping: {config_path}"
        )
    model_cfg = dict(loaded)
    model_cfg["policy"] = args.model_type
    return model_cfg


def _build_server_logging_tag(
    *,
    host: str,
    port: int,
    model_cfg: dict[str, Any],
) -> str:
    logging_tag = model_cfg.pop("logging_tag", None)
    if logging_tag:
        return f"{host}:{port}:{logging_tag}"
    return f"{host}:{port}"


def _normalize_local_policy(policy_or_cfg: Any) -> PolicyMixin:
    if isinstance(policy_or_cfg, PolicyConfig):
        return policy_or_cfg()
    return policy_or_cfg


# ---------------------------------------------------------------------------
# Standalone server entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from robo_orchard_sim.policy.factory import create_policy_from_model_cfg

    parser = _build_server_parser()
    args = parser.parse_args()
    model_cfg = _load_policy_model_cfg(args)
    logging_tag = _build_server_logging_tag(
        host=args.host,
        port=args.port,
        model_cfg=model_cfg,
    )

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [{logging_tag}] %(message)s",
    )
    logger.info("model_cfg: %s", model_cfg)

    policy = _normalize_local_policy(create_policy_from_model_cfg(model_cfg))
    server = PolicyWebsocketServer(
        policy=policy,
        host=args.host,
        port=args.port,
        logging_tag=logging_tag,
    )
    server.run()
