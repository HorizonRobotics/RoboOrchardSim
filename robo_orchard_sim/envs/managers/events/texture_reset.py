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

from collections.abc import Sequence
from typing import List

import isaaclab.sim as sim_utils
import torch
from pxr import Usd
from robo_orchard_core.envs.manager_based_env import ResetEvent
from robo_orchard_core.envs.managers.events.event_term import (
    EventTermBase,
    EventTermBaseCfg,
)

from robo_orchard_sim.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg as LabSceneEntityCfg,
)
from robo_orchard_sim.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.utils.config import ClassType_co

__all__ = ["TextureResetTerm", "TextureResetTermCfg"]


class TextureResetTerm(
    EventTermBase[ResetEvent, IsaacEnvType_co, "TextureResetTerm"],
):
    """Select asset texture variants during reset."""

    def __init__(self, cfg: "TextureResetTermCfg", env: IsaacEnvType_co):
        super().__init__(cfg, env)
        self._cfg = cfg
        self._env = env
        self.generator = torch.Generator(device="cpu")

    def __call__(self, event_msg: ResetEvent):
        # Seed the RNG directly from the event payload.
        seed_value = getattr(event_msg, "seed", None)

        # Reset only the requested environments when env_ids is provided.
        env_ids = getattr(event_msg, "env_ids", None)
        if seed_value is None:
            print(
                "[TextureReset] event.seed is None; defaulting RNG seed to 0"
            )
            seed_value = 0
        self.generator.manual_seed(int(seed_value))
        self._select_asset_variants(
            self._env.sim.stage, generator=self.generator, env_ids=env_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        pass

    def _find_variant_prims_under(
        self,
        stage: Usd.Stage,
        root_path: str,
        env_ids: Sequence[int] | None = None,
    ) -> List[str]:
        """Find prims under the root path that expose the target VariantSet.

        The target variant set name is configured by ``cfg.variant_set_name``.
        """
        # Resolve environment-specific root paths from the regex template.
        resolved_roots: List[str]
        if "env_.*" not in root_path:
            resolved_roots = [root_path]
        elif env_ids is not None:
            resolved_roots = [
                root_path.replace("env_.*", f"env_{int(env_id)}")
                for env_id in env_ids
            ]
        else:
            resolved_roots = [
                root_path.replace("env_.*", f"env_{env_id}")
                for env_id in range(self._env.num_envs)
            ]
        variant_paths: List[str] = []
        for prim in stage.Traverse():
            path_str = prim.GetPath().pathString
            if not any(path_str.startswith(rr) for rr in resolved_roots):
                continue
            if prim.HasVariantSets():
                vs_names = prim.GetVariantSets().GetNames()
                if self._cfg.variant_set_name in vs_names:
                    # Collect leaf material prims to avoid duplicates.
                    # TODO: Make this more robust for different layouts.
                    if path_str.endswith("material_0") or "Looks" in path_str:
                        variant_paths.append(path_str)
        # Return unique, sorted paths for deterministic traversal order.
        return sorted(set(variant_paths))

    def _select_asset_variants(
        self,
        stage: Usd.Stage,
        generator: torch.Generator,
        env_ids: Sequence[int] | None = None,
    ):
        """Select variants for assets listed in ``cfg.asset_cfgs``.

        For each asset, find material prims under its root prim path that have
        the target VariantSet, and select a variant.
        """
        if not getattr(self._cfg, "asset_cfgs", None):
            print(
                "[TextureReset] No asset_cfgs provided; "
                "skipping variant selection."
            )
            return

        for _asset_idx, asset_cfg in enumerate(self._cfg.asset_cfgs):
            # Resolve the asset root prim path from the scene configuration.
            scene_item = self._env.scene[asset_cfg.name]
            root_path = scene_item.cfg.prim_path

            prim_paths = self._find_variant_prims_under(
                stage,
                root_path,
                env_ids=env_ids,
            )
            if not prim_paths:
                print(
                    f"[TextureReset] Asset '{asset_cfg.name}' under "
                    f"'{root_path}' has no '{self._cfg.variant_set_name}' "
                    "variant prims. "
                    "Skipping."
                )
                continue
            print(
                f"[TextureReset] Asset '{asset_cfg.name}' variant prims: "
                f"{prim_paths}"
            )

            for _prim_idx, prim_path in enumerate(prim_paths):
                prim = stage.GetPrimAtPath(prim_path)
                vs = prim.GetVariantSet(self._cfg.variant_set_name)

                # Gather the available variant names.
                variants = vs.GetVariantNames()

                # Sort variant names when deterministic ordering is requested.
                if self._cfg.variant_sort:
                    variants = sorted(variants)
                # Apply index-range filtering.
                start_idx, end_idx = self._cfg.variant_index_range
                if end_idx == -1:
                    end_idx = len(variants)

                # Clamp out-of-range indices before slicing.
                start_idx = max(0, min(start_idx, len(variants)))
                end_idx = max(0, min(end_idx, len(variants)))
                if start_idx >= end_idx:
                    print(
                        f"[TextureReset] Prim {prim_path} has invalid "
                        "variant_index_range="
                        f"{self._cfg.variant_index_range}. Skipping."
                    )
                    continue
                variants = variants[start_idx:end_idx]

                if not variants:
                    print(
                        f"[TextureReset] Prim {prim_path} has empty "
                        f"'{self._cfg.variant_set_name}' variant set."
                    )
                    continue

                # Sample a variant index using the event-scoped RNG.
                chosen_index = int(
                    torch.randint(
                        low=0,
                        high=len(variants),
                        size=(1,),
                        generator=generator,
                    ).item()
                )
                chosen_variant = variants[chosen_index]

                sim_utils.select_usd_variants(
                    prim_path=prim_path,
                    variants={self._cfg.variant_set_name: chosen_variant},
                )
                print(
                    f"[TextureReset] Prim {prim_path} total variants: "
                    f"{len(variants)} textures -> chosen "
                    f"'{chosen_variant}' (index {chosen_index})"
                )


class TextureResetTermCfg(
    EventTermBaseCfg[TextureResetTerm, LabSceneEntityCfg]
):
    """Configuration for texture variant selection reset term."""

    class_type: ClassType_co[TextureResetTerm] = TextureResetTerm

    variant_set_name: str = "Look"
    """Name of the VariantSet to operate on."""

    variant_sort: bool = True
    """Sort variant names before sampling."""

    variant_index_range: List[int] = [0, -1]
    """Inclusive start and exclusive end index range used for sampling."""
