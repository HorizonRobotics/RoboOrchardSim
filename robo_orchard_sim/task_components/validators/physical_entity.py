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

"""Physical scene-entity identities and runtime resolution."""

from __future__ import annotations
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robo_orchard_sim.asset_manager.metadata import (
    ArticulationOperationMeta,
    JointOperationMeta,
    get_joint,
    get_operation,
)
from robo_orchard_sim.task_components.role_registry import TargetRef

if TYPE_CHECKING:
    import numpy as np
    import torch

    from robo_orchard_sim.task_components.validators.role_scope import (
        RoleScope,
    )


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty.")


@dataclass(frozen=True, slots=True)
class SceneJointKey:
    """Identify one physical joint within a scene articulation."""

    scene_name: str
    joint_name: str

    def __post_init__(self) -> None:
        _require_non_empty(self.scene_name, "scene_name")
        _require_non_empty(self.joint_name, "joint_name")


@dataclass(frozen=True, slots=True)
class SceneBodyKey:
    """Identify one physical body within a scene asset."""

    scene_name: str
    body_name: str

    def __post_init__(self) -> None:
        _require_non_empty(self.scene_name, "scene_name")
        _require_non_empty(self.body_name, "body_name")


SceneEntityKey = str | SceneBodyKey | SceneJointKey


@dataclass(frozen=True, slots=True)
class SceneArticulationJointComponent:
    """One metadata-declared articulation joint in the live scene."""

    articulation: Any
    metadata: JointOperationMeta
    joint_key: SceneJointKey
    joint_id: int


@dataclass(frozen=True, slots=True)
class SceneArticulationBodyComponent:
    """One metadata-declared articulation body in the live scene."""

    articulation: Any
    body_key: SceneBodyKey
    body_id: int


@dataclass(frozen=True, slots=True)
class BoundOperationComponents:
    """Physical scene keys and goal metadata selected by one TargetRef."""

    joint_key: SceneJointKey
    interaction_body_keys: tuple[SceneBodyKey, ...]
    outcome_body_key: SceneBodyKey
    operation: ArticulationOperationMeta

    @property
    def interaction_body_key(self) -> SceneBodyKey:
        """Return the interaction body for consumers requiring one body."""
        if len(self.interaction_body_keys) != 1:
            raise ValueError(
                "This selector requires a single interaction body"
            )
        return self.interaction_body_keys[0]


def scene_name_of(key: SceneEntityKey) -> str:
    """Return the owning scene asset name for any entity key."""
    return key if isinstance(key, str) else key.scene_name


def has_articulation_components(scene_asset: Any) -> bool:
    """Return whether an asset declares semantic articulation components."""
    try:
        return bool(scene_asset.cfg.joint_operations)
    except AttributeError:
        return False


def _exact_component_id(
    *,
    expected_name: str,
    matched_ids: list[int],
    matched_names: list[str],
    component_kind: str,
    scene_name: str,
) -> int:
    exact_ids = [
        component_id
        for component_id, matched_name in zip(
            matched_ids,
            matched_names,
            strict=True,
        )
        if matched_name == expected_name
    ]
    if len(exact_ids) != 1:
        raise ValueError(
            f"Expected exactly one {component_kind} named '{expected_name}' "
            f"on '{scene_name}', found {matched_names}."
        )
    return int(exact_ids[0])


def iter_articulation_joints(
    env: Any,
    scope: "RoleScope",
) -> Iterator[SceneArticulationJointComponent]:
    """Yield every metadata-declared physical joint in scene scope."""
    scene_names = dict.fromkeys(scene_name_of(key) for key in scope.entities)
    for scene_name in scene_names:
        articulation = env.scene[scene_name]
        try:
            joint_operations = articulation.cfg.joint_operations
        except AttributeError:
            continue
        for metadata in joint_operations:
            joint_ids, joint_names = articulation.find_joints(
                metadata.joint_name
            )
            yield SceneArticulationJointComponent(
                articulation=articulation,
                metadata=metadata,
                joint_key=SceneJointKey(scene_name, metadata.joint_name),
                joint_id=_exact_component_id(
                    expected_name=metadata.joint_name,
                    matched_ids=joint_ids,
                    matched_names=joint_names,
                    component_kind="joint",
                    scene_name=scene_name,
                ),
            )


def iter_articulation_bodies(
    env: Any,
    scope: "RoleScope",
) -> Iterator[SceneArticulationBodyComponent]:
    """Yield every distinct metadata-declared body in scene scope."""
    scene_names = dict.fromkeys(scene_name_of(key) for key in scope.entities)
    for scene_name in scene_names:
        articulation = env.scene[scene_name]
        try:
            joint_operations = articulation.cfg.joint_operations
        except AttributeError:
            continue
        body_names = dict.fromkeys(
            body_name
            for metadata in joint_operations
            for body_name in (
                metadata.outcome_link,
                *(
                    body_name
                    for operation in metadata.operations.values()
                    for body_name in operation.effective_interaction_links
                ),
            )
        )
        for body_name in body_names:
            body_ids, matched_names = articulation.find_bodies(body_name)
            yield SceneArticulationBodyComponent(
                articulation=articulation,
                body_key=SceneBodyKey(scene_name, body_name),
                body_id=_exact_component_id(
                    expected_name=body_name,
                    matched_ids=body_ids,
                    matched_names=matched_names,
                    component_kind="body",
                    scene_name=scene_name,
                ),
            )


def resolve_bound_operation(
    env: Any,
    target: TargetRef,
) -> BoundOperationComponents:
    """Map task intent to the corresponding physical scene component keys."""
    if target.semantic_name is None or target.operation is None:
        raise ValueError(
            f"Target '{target}' must include semantic_name and operation."
        )
    articulation = env.scene[target.scene_name]
    try:
        joint_operations = articulation.cfg.joint_operations
    except AttributeError:
        raise ValueError(
            f"Target '{target.scene_name}' has no joint operation metadata."
        ) from None
    joint = get_joint(joint_operations, target.semantic_name)
    operation = get_operation(
        joint_operations,
        target.semantic_name,
        target.operation,
    )
    return BoundOperationComponents(
        joint_key=SceneJointKey(target.scene_name, joint.joint_name),
        interaction_body_keys=tuple(
            SceneBodyKey(target.scene_name, name)
            for name in operation.effective_interaction_links
        ),
        outcome_body_key=SceneBodyKey(
            target.scene_name,
            joint.outcome_link,
        ),
        operation=operation,
    )


def resolve_env_prim_path(scene_asset: Any, env_idx: int) -> str:
    """Resolve an asset's configured prim path for one environment."""
    return re.sub(
        r"env_\.\*",
        f"env_{env_idx}",
        scene_asset.cfg.prim_path,
        count=1,
    )


def resolve_rigid_body_prim_path(scene_asset: Any, env_idx: int) -> str:
    """Resolve a rigid asset to the prim PhysX treats as its body.

    The configured ``prim_path`` may name a container Xform whose rigid
    body lives on a descendant. PhysX filter patterns match collision
    shapes, so filtering on the container matches nothing.
    """
    view = getattr(scene_asset, "root_physx_view", None)
    prim_paths = None
    if view is not None:
        prim_paths = getattr(view, "prim_paths", None)
    if prim_paths is not None:
        try:
            return str(prim_paths[env_idx])
        except (IndexError, TypeError):
            pass
    return resolve_env_prim_path(scene_asset, env_idx)


def resolve_body_id(env: Any, key: SceneBodyKey) -> int:
    """Resolve one exact scene body key to its runtime body index."""
    asset = env.scene[key.scene_name]
    body_ids, body_names = asset.find_bodies(key.body_name)
    return _exact_component_id(
        expected_name=key.body_name,
        matched_ids=body_ids,
        matched_names=body_names,
        component_kind="body",
        scene_name=key.scene_name,
    )


def entity_position(
    env: Any,
    entity_id: SceneEntityKey,
    env_idx: int,
) -> "torch.Tensor":
    """Return one root or body position in world coordinates."""
    if isinstance(entity_id, str):
        return env.scene[entity_id].data.root_pos_w[env_idx]
    if isinstance(entity_id, SceneBodyKey):
        body_id = resolve_body_id(env, entity_id)
        return env.scene[entity_id.scene_name].data.body_pos_w[
            env_idx, body_id
        ]
    raise TypeError(f"Joint '{entity_id}' has no world position.")


def entity_quaternion(
    env: Any,
    entity_id: SceneEntityKey,
    env_idx: int,
) -> "torch.Tensor":
    """Return one root or body orientation in world coordinates."""
    if isinstance(entity_id, str):
        return env.scene[entity_id].data.root_quat_w[env_idx]
    if isinstance(entity_id, SceneBodyKey):
        body_id = resolve_body_id(env, entity_id)
        return env.scene[entity_id.scene_name].data.body_quat_w[
            env_idx, body_id
        ]
    raise TypeError(f"Joint '{entity_id}' has no world orientation.")


def entity_pose_array(env: Any, entity_id: SceneEntityKey) -> "np.ndarray":
    """Return root or body poses for every parallel environment."""
    import torch

    if isinstance(entity_id, str):
        return env.scene[entity_id].data.root_state_w[:, :7].cpu().numpy()
    if isinstance(entity_id, SceneBodyKey):
        body_id = resolve_body_id(env, entity_id)
        data = env.scene[entity_id.scene_name].data
        return (
            torch.cat(
                [
                    data.body_pos_w[:, body_id],
                    data.body_quat_w[:, body_id],
                ],
                dim=-1,
            )
            .cpu()
            .numpy()
        )
    raise TypeError(f"Joint '{entity_id}' has no world pose.")


def entity_prim_path(
    env: Any,
    entity_id: SceneEntityKey,
    env_idx: int,
) -> str:
    """Return one root or body PhysX prim path."""
    if isinstance(entity_id, str):
        return resolve_env_prim_path(env.scene[entity_id], env_idx)
    if isinstance(entity_id, SceneBodyKey):
        body_id = resolve_body_id(env, entity_id)
        asset = env.scene[entity_id.scene_name]
        try:
            return str(asset.root_physx_view.link_paths[env_idx][body_id])
        except (AttributeError, IndexError, TypeError):
            raise ValueError(
                f"Could not resolve body '{entity_id.body_name}' on "
                f"'{entity_id.scene_name}' to a PhysX path."
            ) from None
    raise TypeError(f"Joint '{entity_id}' has no prim path.")


class CollisionBodyBounds:
    """Cache collider AABBs in each rigid body's scaled local frame.

    Bounds enclose enabled collision geometry, including empty space between
    colliders. Live body poses, rather than USD poses, follow articulation
    motion. Geometry/scale changes require reset; rigid motion does not.
    """

    def __init__(self) -> None:
        self._bounds: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def distances(
        self, env, body: SceneBodyKey, points_w: torch.Tensor, env_idx: int = 0
    ) -> torch.Tensor:
        """Return distances in meters from world points to the local box."""
        import torch
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = env.sim.stage
        path = entity_prim_path(env, body, env_idx)
        key = (stage, path, points_w.device, points_w.dtype)
        if key not in self._bounds:
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError(f"Missing collision body prim '{path}'.")
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                ["default", "render", "proxy", "guide"],
                useExtentsHint=False,
                ignoreVisibility=True,
            )
            scale = Gf.Transform(
                UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
            ).GetScale()
            lower, upper = [], []
            iterator = iter(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
            for child in iterator:
                if child != prim and child.HasAPI(UsdPhysics.RigidBodyAPI):
                    iterator.PruneChildren()
                    continue
                if not child.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if (
                    not UsdPhysics.CollisionAPI(child)
                    .GetCollisionEnabledAttr()
                    .Get()
                ):
                    continue
                box = cache.ComputeRelativeBound(
                    child, prim
                ).ComputeAlignedBox()
                if box.IsEmpty():
                    continue
                a = [box.GetMin()[i] * scale[i] for i in range(3)]
                b = [box.GetMax()[i] * scale[i] for i in range(3)]
                lower.append([min(x, y) for x, y in zip(a, b, strict=True)])
                upper.append([max(x, y) for x, y in zip(a, b, strict=True)])
            if not lower:
                raise ValueError(f"No enabled collision bounds for '{path}'.")
            lo = points_w.new_tensor(lower).amin(dim=0)
            hi = points_w.new_tensor(upper).amax(dim=0)
            if not torch.isfinite(torch.stack((lo, hi))).all():
                raise ValueError(f"Non-finite collision bounds for '{path}'.")
            self._bounds[key] = lo, hi
        lo, hi = self._bounds[key]
        delta = points_w - entity_position(env, body, env_idx)
        quat = entity_quaternion(env, body, env_idx)
        # Inverse rotation, quaternion order wxyz.
        vector = -quat[1:].expand_as(delta)
        cross = 2 * torch.linalg.cross(vector, delta)
        local = delta + quat[0] * cross + torch.linalg.cross(vector, cross)
        outside = torch.maximum(lo - local, local - hi).clamp_min(0)
        return torch.linalg.vector_norm(outside, dim=-1)

    def reset(self) -> None:
        """Invalidate cached bounds after scene or geometry changes."""
        self._bounds.clear()


__all__ = [
    "CollisionBodyBounds",
    "BoundOperationComponents",
    "SceneArticulationBodyComponent",
    "SceneArticulationJointComponent",
    "SceneBodyKey",
    "SceneEntityKey",
    "SceneJointKey",
    "entity_pose_array",
    "entity_position",
    "entity_prim_path",
    "entity_quaternion",
    "has_articulation_components",
    "iter_articulation_bodies",
    "iter_articulation_joints",
    "resolve_body_id",
    "resolve_bound_operation",
    "resolve_env_prim_path",
    "resolve_rigid_body_prim_path",
    "scene_name_of",
]
