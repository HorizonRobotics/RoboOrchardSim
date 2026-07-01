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

from typing import Any

from robo_orchard_sim.policy.dummy.policy import DummyPolicyCfg
from robo_orchard_sim.policy.groot.policy import GrootArmMapCfg, GrootPolicyCfg
from robo_orchard_sim.policy.holobrain.policy import HolobrainPolicyCfg
from robo_orchard_sim.policy.server import ServerPolicyCfg


def _cfg_get(model_cfg: Any, name: str, default=None):
    if isinstance(model_cfg, dict):
        return model_cfg.get(name, default)
    return getattr(model_cfg, name, default)


def _cfg_get_non_null(model_cfg: Any, name: str, default=None):
    value = _cfg_get(model_cfg, name, default)
    if value is None:
        return default
    return value


def create_policy_from_model_cfg(
    model_cfg: Any,
    *,
    embodiment_type: str | None = None,
):
    """Create a simulator policy from a minimal config object or dict."""
    policy_name = _cfg_get(model_cfg, "policy")
    if policy_name == "dummy":
        return DummyPolicyCfg()
    if policy_name == "holobrain":
        return HolobrainPolicyCfg(
            model_dir=_cfg_get(model_cfg, "model_dir"),
            logging_tag=_cfg_get(model_cfg, "logging_tag"),
            inference_prefix=_cfg_get(model_cfg, "inference_prefix"),
            embodiment_type=_cfg_get_non_null(
                model_cfg, "embodiment_type", embodiment_type
            ),
            device=_cfg_get(model_cfg, "device"),
            valid_action_step=_cfg_get(model_cfg, "valid_action_step"),
        )
    if policy_name == "server":
        remote_policy_type = _cfg_get(model_cfg, "remote_policy_type")
        return ServerPolicyCfg(
            host=_cfg_get(model_cfg, "host", "127.0.0.1"),
            port=_cfg_get(model_cfg, "port", 8765),
            logging_tag=_cfg_get(model_cfg, "logging_tag"),
            remote_policy_type=remote_policy_type or "full",
        )
    if policy_name == "groot":
        groot_kwargs: dict[str, Any] = dict(
            host=_cfg_get(model_cfg, "host", "127.0.0.1"),
            port=_cfg_get(model_cfg, "port", 5555),
            timeout_ms=_cfg_get(model_cfg, "timeout_ms", 15000),
            open_loop_horizon=_cfg_get(model_cfg, "open_loop_horizon"),
            instruction=_cfg_get(model_cfg, "instruction"),
            logging_tag=_cfg_get(model_cfg, "logging_tag"),
            api_token=_cfg_get(model_cfg, "api_token"),
        )
        language_key = _cfg_get(model_cfg, "language_key")
        if language_key is not None:
            groot_kwargs["language_key"] = language_key
        video_map = _cfg_get(model_cfg, "video_map")
        if video_map is not None:
            groot_kwargs["video_map"] = video_map
        arms = _cfg_get(model_cfg, "arms")
        if arms is not None:
            groot_kwargs["arms"] = [GrootArmMapCfg(**arm) for arm in arms]
        return GrootPolicyCfg(**groot_kwargs)
    raise ValueError(f"Invalid policy: {policy_name}")
