# ruff: noqa: E402
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

"""Example: assemble any registered task as an ``OrchardEnv`` from YAML.

Selects the task definition from the task-suite registry by ``--task``
namespace (e.g. ``place_a2b_easy``, ``place_a2b_hard``, ``pick_category``,
``pick_attribute``, ``pick_disambiguation``) and builds it through
``build_task(...)``. The selected task definition reads scene, embodiment,
instruction, ``asset_configs``, and task params from its default YAML.

``--config`` optionally overrides that YAML for this build only. It does not
change which task definition is selected, and it does not mutate the
registered task definition class.

Usage::

    # Via env var
    export ORCHARD_ASSET_LIBRARY=test_assets/wuwen_0411_labelled_usd
    python examples/manipulation-app/scripts/simple_orchard_env_example.py \\
        --task place_a2b_easy

    # Use a different registered task
    python examples/manipulation-app/scripts/simple_orchard_env_example.py \\
        --task pick_category --asset-root test_assets/wuwen_0411_labelled_usd

    # Override the task's default YAML with a custom one
    python examples/manipulation-app/scripts/simple_orchard_env_example.py \\
        --task place_a2b_easy \\
        --config path/to/my_place_a2b.yaml

First run on a fresh asset root auto-builds ``asset_index.parquet`` in
the library directory; subsequent runs reuse it.
"""

from __future__ import annotations

from robo_orchard_sim.launcher import SimpleIsaacAppLauncher

launcher = SimpleIsaacAppLauncher(
    headless=True,
    enable_cameras=True,
    virtual_display=False,
)
sim_app = launcher.app  # keep application alive

import argparse
import os

import numpy as np

from robo_orchard_sim.asset_manager.registry import AssetRegistry
from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
    AssetResolver,
    AssetResolverError,
)
from robo_orchard_sim.envs.env_base import IsaacEnvContextManager
from robo_orchard_sim.task_suite.registry import (
    build_task,
)

_ASSET_ROOT_ENV = "ORCHARD_ASSET_LIBRARY"


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the task-agnostic orchard env example."""
    env_default = os.environ.get(_ASSET_ROOT_ENV)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help=(
            "Registered task namespace to build (e.g. pick_category, "
            "pick_attribute, pick_disambiguation)."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional YAML path overriding the selected task's default "
            "config_path. Does NOT change which task class is used; "
            "that is always determined by --task."
        ),
    )
    parser.add_argument(
        "--asset-root",
        type=str,
        default=env_default,
        required=env_default is None,
        help=(
            f"Asset library root. Defaults to the ${_ASSET_ROOT_ENV} env "
            "var; required if that env var is not set. Auto-builds "
            "asset_index.parquet on first run."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/orchard_env_example.json",
        help="Output JSON path for serialized env cfg.",
    )
    parser.add_argument(
        "--enable-recording",
        action="store_true",
        help="Enable MCAP recording for the example run.",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default="logs/records",
        help="Output directory for MCAP recording files.",
    )
    return parser


def main() -> None:
    """Build the OrchardEnv example and optionally validate it at runtime."""
    parser = build_arg_parser()

    args = parser.parse_args()

    registry = AssetRegistry(args.asset_root)
    resolver = AssetResolver(
        registry=registry,
        splits=None,  # TODO: support splits
        rng=np.random.default_rng(args.seed),
    )

    config_path = None
    if args.config is not None:
        config_path = os.path.abspath(args.config)
        if not os.path.isfile(config_path):
            raise SystemExit(
                f"\nERROR: --config path does not exist: {config_path}"
            )

    try:
        # TODO: Decouple build_task from resolver
        env = build_task(
            task_name=args.task,
            resolver=resolver,
            config_path=config_path,
        )
    except KeyError as exc:
        raise SystemExit(f"\nERROR: {exc}") from exc
    except AssetResolverError as exc:
        raise SystemExit(
            f"\nERROR resolving assets from registry: {exc}\n"
            f"Check the asset_configs: block in the task YAML, or pass "
            f"a different --config."
        )
    if args.enable_recording:
        env.configure_recording(file_path=args.record_dir)

    env_cfg = env.to_isaac_env_cfg()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(env_cfg.to_str(format="json", indent=4))

    print(f"OrchardEnv for task {args.task!r} assembled successfully.")
    print(f"Output: {args.output}")
    print(f"Config YAML: {config_path or 'task default'}")
    print(f"Scene type: {type(env.scene).__name__}")
    print(f"Embodiment: {env.embodiment.scene_name}")
    print(f"Task type: {type(env.task).__name__}")
    print(f"Scene assets: {sorted(env_cfg.scene.assets.keys())}")
    print(f"Event terms: {sorted(env_cfg.events.terms.keys())}")
    if env_cfg.records is not None:
        print(f"Record terms: {sorted(env_cfg.records.terms.keys())}")
        print(f"Record dir: {env_cfg.records.file_path}")

    env_manager = IsaacEnvContextManager(
        env_cfg,
        with_new_stage=True,
        disable_exit_on_stop=False,
    )
    with env_manager as env:
        _ = env.reset()
        print("Runtime reset done.")
        print(f"Available entities: {list(env.scene.keys())}")

        for i in range(5):
            if not sim_app.is_running():
                break
            _ = env.step()
            print(f"Step {i + 1} done.")

    print("Example finished successfully.")


if __name__ == "__main__":
    main()
