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
from robo_orchard_sim.policy.holobrain.policy import HolobrainPolicyCfg
from robo_orchard_sim.policy.openpi.policy import OpenPiPolicyCfg
from robo_orchard_sim.policy.server import ServerPolicyCfg


def _cfg_get(model_cfg: Any, name: str, default=None):
    if isinstance(model_cfg, dict):
        return model_cfg.get(name, default)
    return getattr(model_cfg, name, default)


def create_policy_from_model_cfg(model_cfg: Any):
    """Create a simulator policy from a minimal config object or dict."""
    policy_name = _cfg_get(model_cfg, "policy")
    if policy_name == "dummy":
        return DummyPolicyCfg()
    if policy_name == "holobrain":
        return HolobrainPolicyCfg(
            model_dir=_cfg_get(model_cfg, "model_dir"),
            logging_tag=_cfg_get(model_cfg, "logging_tag"),
            inference_prefix=_cfg_get(model_cfg, "inference_prefix"),
            joint_num=_cfg_get(model_cfg, "joint_num", 7),
            device=_cfg_get(model_cfg, "device"),
            valid_action_step=_cfg_get(model_cfg, "valid_action_step"),
        )
    if policy_name == "openpi":
        openpi_kwargs = dict(
            model=_cfg_get(model_cfg, "model"),
            model_dir=_cfg_get(model_cfg, "model_dir"),
            logging_tag=_cfg_get(model_cfg, "logging_tag"),
            joint_num=_cfg_get(model_cfg, "joint_num", 7),
            valid_action_step=_cfg_get(model_cfg, "valid_action_step", 50),
            enable_intrinsic_remap=_cfg_get(
                model_cfg,
                "enable_intrinsic_remap",
                True,
            ),
        )
        inference = _cfg_get(model_cfg, "inference")
        if inference is not None:
            openpi_kwargs["inference"] = inference
        cameras = _cfg_get(model_cfg, "cameras")
        if cameras is not None:
            openpi_kwargs["cameras"] = cameras
        return OpenPiPolicyCfg(**openpi_kwargs)
    if policy_name == "server":
        return ServerPolicyCfg(
            host=_cfg_get(model_cfg, "host", "127.0.0.1"),
            port=_cfg_get(model_cfg, "port", 8765),
            logging_tag=_cfg_get(model_cfg, "logging_tag"),
        )
    raise ValueError(f"Invalid policy: {policy_name}")
