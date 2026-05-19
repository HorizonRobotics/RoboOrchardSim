#!/usr/bin/env python3
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

"""Generate per-graspable-asset task YAML configs from a base config."""

from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
from typing import Any

import yaml

from robo_orchard_sim.asset_manager.registry.registry import AssetRegistry
from robo_orchard_sim.asset_manager.registry.types import AssetFilter

_ASSET_ROOT_ENV = "ORCHARD_ASSET_LIBRARY"
_DEFAULT_GRASPABLE_TAG = "is_graspable"
_DEFAULT_CONFIG_LIST_NAME = "yaml_list.txt"


def discover_graspable_assets(
    *,
    asset_root: str,
    graspable_tag: str = _DEFAULT_GRASPABLE_TAG,
) -> list[dict[str, Any]]:
    """Return JSON-serializable records for graspable registry assets."""
    registry = AssetRegistry(asset_root)
    asset_filter = AssetFilter(tags=frozenset({graspable_tag}))
    metas = registry.query(asset_filter)

    return [
        {
            "uuid": meta.uuid,
            "asset_id": meta.asset_id,
            "usd_path": meta.usd_path,
            "interaction_path": meta.interaction_path,
            "super_category": meta.super_category,
            "category": meta.category,
        }
        for meta in metas
    ]


def _safe_path_component(value: str) -> str:
    safe_value = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    )
    return safe_value or "asset"


def _config_filename(
    *,
    task: str,
    asset_id: str,
    uuid: str,
) -> str:
    return (
        f"{_safe_path_component(task)}_"
        f"{_safe_path_component(asset_id)}_{uuid[:8]}.yaml"
    )


def _asset_record_identity(record: dict[str, Any]) -> tuple[str, str]:
    try:
        return str(record["asset_id"]), str(record["uuid"])
    except KeyError as exc:
        raise ValueError(f"asset record missing key {exc.args[0]!r}") from exc


def _pinned_payload(
    *,
    base_payload: dict[str, Any],
    target_role: str,
    uuid: str,
    base_config_path: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    asset_configs = payload.get("asset_configs")
    if not isinstance(asset_configs, dict):
        raise ValueError(f"{base_config_path} missing asset_configs mapping")
    role_config = asset_configs.get(target_role)
    if not isinstance(role_config, dict):
        raise ValueError(
            f"{base_config_path} missing asset_configs.{target_role} mapping"
        )

    pinned_role_config = {
        key: value
        for key, value in role_config.items()
        if key not in {"filter", "split", "pool_size"}
    }
    pinned_role_config["uuid"] = uuid
    asset_configs[target_role] = pinned_role_config
    return payload


def write_asset_configs_from_records(
    *,
    base_config_path: str,
    output_dir: str,
    config_list_path: str,
    task: str,
    target_role: str,
    records: list[dict[str, Any]],
) -> list[str]:
    """Write one task YAML per asset record and a newline config list."""
    with open(base_config_path, encoding="utf-8") as fr:
        base_payload = yaml.safe_load(fr) or {}
    if not isinstance(base_payload, dict):
        raise ValueError(f"{base_config_path} must contain a YAML mapping")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_paths: list[str] = []
    seen_paths: set[str] = set()
    for record in records:
        asset_id, uuid = _asset_record_identity(record)
        config_path = output_path / _config_filename(
            task=task,
            asset_id=asset_id,
            uuid=uuid,
        )
        if str(config_path) in seen_paths:
            raise ValueError(f"duplicate generated config path: {config_path}")
        seen_paths.add(str(config_path))

        payload = _pinned_payload(
            base_payload=base_payload,
            target_role=target_role,
            uuid=uuid,
            base_config_path=base_config_path,
        )
        with config_path.open("w", encoding="utf-8") as fw:
            yaml.safe_dump(payload, fw, sort_keys=False)
        generated_paths.append(str(config_path))

    list_path = Path(config_list_path)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text(
        "\n".join(generated_paths) + ("\n" if generated_paths else ""),
        encoding="utf-8",
    )
    return generated_paths


def write_graspable_asset_configs(
    *,
    asset_root: str,
    base_config_path: str,
    output_dir: str,
    config_list_path: str,
    task: str,
    target_role: str = "pick",
    graspable_tag: str = _DEFAULT_GRASPABLE_TAG,
) -> list[str]:
    """Discover graspable assets and write per-asset task YAML configs."""
    records = discover_graspable_assets(
        asset_root=asset_root,
        graspable_tag=graspable_tag,
    )
    return write_asset_configs_from_records(
        base_config_path=base_config_path,
        output_dir=output_dir,
        config_list_path=config_list_path,
        task=task,
        target_role=target_role,
        records=records,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    env_default = os.environ.get(_ASSET_ROOT_ENV)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--asset-root",
        default=env_default,
        required=env_default is None,
        help=(
            f"Asset library root. Defaults to the ${_ASSET_ROOT_ENV} env "
            "var and is required if that env var is not set."
        ),
    )
    parser.add_argument("--target-role", default="pick")
    parser.add_argument("--graspable-tag", default=_DEFAULT_GRASPABLE_TAG)
    return parser


def main() -> None:
    """Run graspable asset config generation from CLI arguments."""
    args = build_arg_parser().parse_args()
    config_list_path = str(Path(args.output_dir) / _DEFAULT_CONFIG_LIST_NAME)
    generated_paths = write_graspable_asset_configs(
        asset_root=args.asset_root,
        base_config_path=args.base_config,
        output_dir=args.output_dir,
        config_list_path=config_list_path,
        task=args.task,
        target_role=args.target_role,
        graspable_tag=args.graspable_tag,
    )
    print(
        f"Generated {len(generated_paths)} configs under {args.output_dir}; "
        f"wrote list {config_list_path}"
    )


if __name__ == "__main__":
    main()
