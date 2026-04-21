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

"""Example: assemble a registry-driven PlaceA2B ``OrchardEnv``.

Builds an ``OrchardEnv`` where task assets are sampled from an
``AssetRegistry`` via an ``AssetResolver``. The task's scene /
embodiment / instruction and its default ``asset_configs`` live in
``place_a2b.yaml`` next to the task definition; this script only
constructs the registry + resolver and delegates to
``PlaceA2BTaskDefinition.build``.

Usage::

    # Via env var
    export ORCHARD_ASSET_LIBRARY=test_assets/wuwen_0411_labelled_usd
    python examples/manipulation-app/scripts/simple_orchard_env_example.py

    # Or via CLI flag
    python examples/manipulation-app/scripts/simple_orchard_env_example.py \\
        --asset-root test_assets/wuwen_0411_labelled_usd

First run on a fresh asset root auto-builds ``asset_index.parquet`` in
the library directory; subsequent runs reuse it.

To customize which assets are sampled, edit the ``asset_configs:``
block in the task YAML (``robo_orchard_sim/task_suite/manipulation/
place_a2b/place_a2b.yaml``) — no Python change required.
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
from robo_orchard_sim.task_suite.manipulation.place_a2b import (
    PlaceA2BTaskDefinition,
)

_ASSET_ROOT_ENV = "ORCHARD_ASSET_LIBRARY"


def main() -> None:
    """Build a registry-driven PlaceA2B env and run reset + a few steps."""
    env_default = os.environ.get(_ASSET_ROOT_ENV)
    parser = argparse.ArgumentParser()
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
        default="configs/place_a2b_orchard_env_example.json",
        help="Output JSON path for serialized env cfg.",
    )
    args = parser.parse_args()

    registry = AssetRegistry(args.asset_root)
    resolver = AssetResolver(
        registry=registry,
        splits=None,
        rng=np.random.default_rng(args.seed),
    )

    try:
        place_a2b_env = PlaceA2BTaskDefinition.build(resolver=resolver)
    except AssetResolverError as exc:
        raise SystemExit(
            f"\nERROR resolving assets from registry: {exc}\n"
            f"Check the asset_configs: block in the task YAML, or pass "
            f"asset_configs=... explicitly."
        )

    env_cfg = place_a2b_env.to_isaac_env_cfg()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(env_cfg.to_str(format="json", indent=4))

    print("placeA2B OrchardEnv assembled successfully.")
    print(f"Output: {args.output}")
    print(f"Scene type: {type(place_a2b_env.scene).__name__}")
    print(f"Embodiment: {place_a2b_env.embodiment.scene_name}")
    print(f"Task type: {type(place_a2b_env.task).__name__}")
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
