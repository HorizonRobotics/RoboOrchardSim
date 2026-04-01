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


import os
import warnings
from typing import Any

from isaaclab.assets import (
    Articulation,
    AssetBase,
    RigidObject,
)
from isaaclab.scene import InteractiveScene as InteractiveSceneBase
from isaaclab.sensors import (
    SensorBase,
)

from robo_orchard_sim.cfg_wrappers.scenes_cfg import InteractiveSceneCfg
from robo_orchard_sim.models.assets.asset_cfg import GroupAssetCfg
from robo_orchard_sim.models.assets.xform_asset import XFormPrimAsset
from robo_orchard_sim.models.prim import PrimClassCfg, PrimType

__all__ = [
    "InteractiveScene",
    "InteractiveSceneCfg",
]


class InteractiveScene(InteractiveSceneBase):
    """The Scene class extended from isaac lab InteractiveScene class.

    It supports config class of GroupAssetCfg. The asset names in the
    GroupAssetCfg are concatenated with the group name separated by '/'.
    For example, if the group name is 'group1' and the asset name is 'asset1',
    the asset name in the scene will be 'group1/asset1'.

    """

    @staticmethod
    def _is_group_asset_mapping(asset_cfg: Any) -> bool:
        return isinstance(asset_cfg, dict) and all(
            isinstance(value, dict) for value in asset_cfg.values()
        )

    def _add_asset(self, asset_name: str, asset_cfg: PrimClassCfg[PrimType]):
        """Add orchard wrappered asset to the scene."""
        asset_cfg.prim_path_format(ENV_REGEX_NS=self.env_regex_ns)
        inst = asset_cfg.create_instance_by_cfg()
        if isinstance(inst, SensorBase):
            self._sensors[asset_name] = inst
        elif isinstance(inst, Articulation):
            if asset_name in self._articulations:
                warnings.warn(
                    f"Articulation {asset_name} already exists, will be overwritten",  # noqa
                    stacklevel=2,
                )
            self._articulations[asset_name] = inst
        elif isinstance(inst, RigidObject):
            if asset_name in self._rigid_objects:
                warnings.warn(
                    f"Rigid object {asset_name} already exists, will be overwritten",  # noqa
                    stacklevel=2,
                )
            self._rigid_objects[asset_name] = inst
        elif isinstance(inst, XFormPrimAsset):
            self._extras[asset_name] = inst
        else:
            raise ValueError(
                f"Unknown asset type for {asset_name}: {asset_cfg}"
            )

    def _add_entities_from_cfg(self):
        """Add entities from the cfg.

        This method extends the base class method to support GroupAssetCfg.
        """

        namespace_asset_cfgs = self.cfg.__dict__.get("assets")
        has_namespace_assets = self._is_group_asset_mapping(
            namespace_asset_cfgs
        )

        # pop all group asset cfgs from the cfg
        group_asset_cfgs = {}  # all the group config
        for asset_name, asset_cfg in self.cfg.__dict__.items():
            if asset_name == "assets" and has_namespace_assets:
                continue
            if isinstance(asset_cfg, GroupAssetCfg):
                group_asset_cfgs[asset_name] = asset_cfg

        if has_namespace_assets:
            self.cfg.__dict__.pop("assets")
            for group_name, group_asset_cfg in namespace_asset_cfgs.items():
                group_asset_cfgs[group_name] = GroupAssetCfg(**group_asset_cfg)
        # rename each asset in the group asset cfgs with new name:
        # new_name = `group_asset_cfg_name/asset_name`
        flattened_group_assets = {}
        for group_asset_cfg_name in group_asset_cfgs:
            group_asset_cfg: GroupAssetCfg = group_asset_cfgs[
                group_asset_cfg_name
            ]
            if group_asset_cfg_name in self.cfg.__dict__:
                self.cfg.__dict__.pop(group_asset_cfg_name)
            for asset_name, asset_cfg in group_asset_cfg.items():
                new_asset_name = os.path.join(group_asset_cfg_name, asset_name)
                if new_asset_name in self.cfg.__dict__:
                    warnings.warn(
                        f"Asset {new_asset_name} already exists in the cfg, "
                        "will be overwritten",
                        stacklevel=2,
                    )
                self.cfg.__dict__[new_asset_name] = asset_cfg
                flattened_group_assets[new_asset_name] = new_asset_name

        # Filter out All config classes that defined by robo_orchard_sim.
        # Those config will be handled by refactored method.
        ro_configs: dict[str, PrimClassCfg] = {}
        for asset_name, asset_cfg in self.cfg.__dict__.items():
            if isinstance(asset_cfg, PrimClassCfg):
                ro_configs[asset_name] = asset_cfg
        for asset_name in ro_configs.keys():
            self.cfg.__dict__.pop(asset_name)

        # call the base class method

        super()._add_entities_from_cfg()

        for asset_name, asset_cfg in ro_configs.items():
            self._add_asset(asset_name, asset_cfg)

        # add robo_orchard_sim assets config back
        for asset_name, asset_cfg in ro_configs.items():
            self.cfg.__dict__[asset_name] = asset_cfg

        # push back the group asset cfgs to the cfg
        # and remove flattened_group_assets from the cfg
        for asset_name in flattened_group_assets:
            self.cfg.__dict__.pop(asset_name)
        for name, cfg in group_asset_cfgs.items():
            self.cfg.__dict__[name] = cfg
        if has_namespace_assets:
            self.cfg.__dict__["assets"] = namespace_asset_cfgs

    def delete_all_assets(self):
        """Explicitly delete all assets in the scene.

        Assets in isaac lab usually implement __del__ method to unsubscribe
        from the simulation. This method calls the __del__ method of the
        assets in the scene to unsubscribe them all.

        This method is necessary to avoid closing the simulation stage without
        unsubscribing the assets from the simulation. Isaac sim seems to have
        a bug that does not release asset object referenced by simulation
        timeline.

        """

        # for asset_key in self.keys():
        #     asset = self[asset_key]
        #     if hasattr(asset, "__del__"):
        #         asset.__del__()

        # clear and delete all assets
        for asset_family in [
            self._articulations,
            self._deformable_objects,
            self._rigid_objects,
            self._rigid_object_collections,
            self._extras,
            self._sensors,
        ]:
            for k in list(asset_family.keys()):
                asset = asset_family.pop(k)
                # just to make sure that __del__ is called.
                # for camera and AssetBase, it seems that the __del__ method
                # is not called
                # Cycle reference???
                # explicitly. So, we call it here.
                if isinstance(asset, (AssetBase, SensorBase)):
                    asset.__del__()
                try:
                    del asset
                except Exception as _:
                    pass

            asset_family.clear()

        del self._terrain
        self._terrain = None
