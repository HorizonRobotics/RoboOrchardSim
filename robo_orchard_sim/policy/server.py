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
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import yaml
from robo_orchard_core.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_core.utils.config import ClassType

logger = logging.getLogger(__name__)

_MAX_MSG = 200 * 1024 * 1024  # 200 MB – large enough for image payloads
_POLICY_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


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


def _encode_tensor_payload(value: torch.Tensor) -> dict[str, Any]:
    cpu_value = value.detach().cpu()
    return {
        "dtype": str(cpu_value.dtype).removeprefix("torch."),
        "shape": list(cpu_value.shape),
        "data": cpu_value.tolist(),
    }


def _decode_tensor_payload(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=dtype).reshape(payload["shape"])


def _encode_value(value: Any) -> Any:
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
        if "__tensor__" in value:
            return _decode_tensor_payload(value["__tensor__"])
        return {key: _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _move_to_device(value: Any, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {
            key: _move_to_device(item, device) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _extract_obs_data(observations: dict) -> dict:
    """Convert live observations into a pickle-safe dict of CPU tensors."""
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
        remote = websocket.remote_address
        logger.info(
            "[%s] Client connected from %s",
            self.logging_tag,
            remote,
        )
        try:
            async for message in websocket:
                try:
                    req = json.loads(message)
                    req_type = req.get("type", "act")
                    if req_type == "reset":
                        self.policy.reset()
                        await websocket.send(
                            json.dumps(
                                {
                                    "ok": True,
                                    "logging_tag": self.logging_tag,
                                }
                            )
                        )
                        continue

                    if req_type != "act":
                        raise ValueError(
                            f"Unsupported request type: {req_type}"
                        )

                    obs = _rebuild_observations(req["obs_data"])
                    instruction = req.get("instruction")
                    if instruction is not None:
                        obs["instruction"] = instruction
                    actions = self.policy.act(obs)
                    await websocket.send(
                        json.dumps(
                            {
                                "actions": _encode_value(actions),
                                "logging_tag": self.logging_tag,
                            }
                        )
                    )
                except Exception:
                    logger.exception(
                        "[%s] Inference error",
                        self.logging_tag,
                    )
                    await websocket.send(
                        json.dumps(
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
            self._handle, self.host, self.port, max_size=_MAX_MSG
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


class PolicyClient:
    """WebSocket client for remote policy inference via server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        logging_tag: str | None = None,
    ):
        self._url = f"ws://{host}:{port}"
        self._ws = None
        self.logging_tag = logging_tag or f"client->{host}:{port}"

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
        self._ws.send(json.dumps(request))
        response = json.loads(self._ws.recv())
        if "logging_tag" in response:
            self.logging_tag = response["logging_tag"]
        if "error" in response:
            raise RuntimeError(f"Remote policy error: {response['error']}")
        return response

    def request_action(
        self,
        observations: dict[str, Any],
        *,
        instruction: str | None = None,
    ):
        """Send observations to server and return predicted actions."""
        if instruction is None:
            instruction = observations.get("instruction")
        req = {
            "type": "act",
            "obs_data": _extract_obs_data(observations),
            "instruction": instruction,
        }
        resp = self._send_request(req)
        actions = _decode_value(resp["actions"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return _move_to_device(actions, device)

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


RemoteAction = dict[str, torch.Tensor] | torch.Tensor


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

    def reset(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._client.reset_remote_policy()

    def act(self, obs: dict[str, Any]) -> RemoteAction:
        return self._client.request_action(obs)

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
        )

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
    logging.info("model_cfg: %s", model_cfg)

    policy = _normalize_local_policy(create_policy_from_model_cfg(model_cfg))
    server = PolicyWebsocketServer(
        policy=policy,
        host=args.host,
        port=args.port,
        logging_tag=logging_tag,
    )
    server.run()
