# ruff: noqa: E402
# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
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

"""Example showing how to assemble a placeA2B ``OrchardEnv``.

This script demonstrates the user-facing `orchard_env` API:
1) define pick/place/table assets
2) assemble ``PlaneTableScene + DualArmPiperEmbodiment + PlaceA2BTask``
3) return an ``OrchardEnv``
4) optionally convert it to ``IsaacManagerBasedEnvCfg`` and run a reset
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

from robo_orchard_sim.envs.env_base import IsaacEnvContextManager


def main() -> None:
    """Build the OrchardEnv example and optionally validate it at runtime."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="configs/place_a2b_orchard_env_example.json",
        help="Output json path for serialized env cfg.",
    )

    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--env_spacing", type=float, default=2.5)
    parser.add_argument("--physics_fps", type=int, default=600)
    parser.add_argument("--render_fps", type=int, default=30)
    parser.add_argument("--action_fps", type=int, default=30)
    args = parser.parse_args()

    from robo_orchard_sim.task_suite.manipulation.place_a2b import (
        PlaceA2BTaskDefinition,
    )

    place_a2b_env = PlaceA2BTaskDefinition.get_env(
        num_envs=args.num_envs,
        env_spacing=args.env_spacing,
        physics_fps=args.physics_fps,
        render_fps=args.render_fps,
        step_fps=args.action_fps,
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
    print(f"Scene asset: {sorted(env_cfg.scene.assets.keys())}")
    print(f"Event terms: {sorted(env_cfg.events.terms.keys())}")

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
