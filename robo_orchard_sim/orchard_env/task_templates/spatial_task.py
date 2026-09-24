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

"""Pick task whose target is named by where it sits, not by what it is."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import Field
from robo_orchard_core.envs.managers.events import EventManagerCfg
from robo_orchard_core.utils.config import Config
from typing_extensions import Literal

from robo_orchard_sim.ext.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.ext.envs.managers.events.light_reset import (
    LightResetTermCfg,
)
from robo_orchard_sim.ext.envs.managers.events.polar_reset import (
    PolarResetTermCfg,
    RegionCfg,
)
from robo_orchard_sim.ext.envs.managers.events.texture_reset import (
    TextureResetTermCfg,
)
from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets
from robo_orchard_sim.orchard_env.task_spec import RoleSpec
from robo_orchard_sim.orchard_env.task_templates.task_base import TaskBase
from robo_orchard_sim.orchard_env.task_templates.task_params import (
    TaskLightResetConfig,
    TaskTextureResetConfig,
)
from robo_orchard_sim.task_components.instructions.base import (
    InstructionActor,
    InstructionWrapper,
)
from robo_orchard_sim.task_components.role_registry import TargetRef
from robo_orchard_sim.task_components.validators.base import Validator
from robo_orchard_sim.task_components.validators.checkers import (
    ContactChecker,
    LiftChecker,
    ReachChecker,
    SceneCheckerSuite,
)
from robo_orchard_sim.task_components.validators.context import (
    ValidatorContext,
)
from robo_orchard_sim.task_components.validators.metrics import (
    AllMetricSelector,
    BoundRoleMetricSelector,
    DwellMetricSelector,
    MetricStore,
)
from robo_orchard_sim.task_components.validators.role_scope import RoleScope

PICK_ROLE = "pick"
REF_ROLE = "ref"

CANDIDATE_PREFIX = "pick"
"""Assets whose role name starts with this may fill the pick role.

Anything else the YAML declares is spawned and arranged like the rest
but is never asked for, which is how clutter gets into the scene without
becoming something an instruction could name.
"""

SpatialRelation = Literal[
    "left_of", "right_of", "front_of", "behind", "near", "far"
]

_RELATION_PHRASES: dict[str, str] = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "front_of": "in front of",
    "behind": "behind",
    "near": "near",
    "far": "far from",
}

# Which axis each direction word means. Directions are read from the
# arm outward: +x leads away from it, +y to its left, so an object "in
# front of" the reference is the one further from the arm rather than
# the one nearer the viewer. Used to check that a config's `axis` says
# the same thing its `relation` does, not to supply one.
_DIRECTIONAL_AXES: dict[str, float] = {
    "front_of": 0.0,
    "left_of": 90.0,
    "behind": 180.0,
    "right_of": 270.0,
}

_DISTANCE_RELATIONS = frozenset({"near", "far"})


def relation_phrase(relation: str) -> str:
    """Render a relation as the words an instruction uses for it."""
    return _RELATION_PHRASES.get(
        relation, relation.strip().lower().replace("_", " ")
    )


class SpatialCandidate(Config):
    """Where one object goes, and what an instruction may call it.

    Every object is placed at a distance along one of four directions
    around the reference, and ``axis`` decides whether its direction is
    part of what makes it that object. Naming one holds the object
    there, which is what an object described as being to the left needs.
    Leaving it out hands the object whichever direction is free that
    episode, so only its distance identifies it -- otherwise a policy
    could find it by direction and never read the word.

    An object with no ``relation`` is clutter: placed like the rest, but
    nothing describes it, so no episode can ask for it.
    """

    radius: tuple[float, float]
    relation: SpatialRelation | None = None
    axis: float | None = None


class SpatialPoseResetConfig(Config):
    """How the reference and everything around it get placed."""

    anchor_range: RegionCfg
    workspace: RegionCfg
    candidates: dict[str, SpatialCandidate] = Field(default_factory=dict)
    axis_jitter: float = 40.0
    min_separation: float = 0.03
    relation_margin: float = 0.05
    """Clear water between the near band and the far one.

    Bands that merely fail to overlap would leave the two nearly the
    same distance away on some episodes, where calling one "near" and
    the other "far" is defensible on paper and unreadable in the scene.
    """


class SpatialTaskParams(Config):
    """Task-level parameters for spatial pick."""

    pose_reset: SpatialPoseResetConfig
    light_reset: TaskLightResetConfig | None = None
    texture_reset: TaskTextureResetConfig | None = None
    reach_dwell_steps: int = 15


class SpatialTask(TaskBase):
    """Pick one of several objects, told apart by where each one sits.

    Every candidate is spawned at once and an episode binds one of them.
    What holds still about a candidate is whatever its own description
    claims, and nothing else: an object called "to the left" keeps its
    direction while its distance varies, and one called "near" keeps its
    distance while its direction is redealt. Either way the remaining
    freedom is resampled every reset, so the only thing that reliably
    identifies the target is the wording the instruction uses for it.

    Because both kinds of description come out of the same arrangement,
    a scene can mix them, and a task changes shape by changing the YAML
    rather than the code.
    """

    roles: ClassVar[dict[str, RoleSpec]] = {
        PICK_ROLE: RoleSpec(
            description="the object to pick up",
            cardinality="one",
            required_traits=("is_graspable",),
        ),
        REF_ROLE: RoleSpec(
            description="the object the target is described against",
            cardinality="one",
        ),
    }

    def __init__(
        self,
        assets: TaskAssets,
        params: SpatialTaskParams,
        instruction: InstructionWrapper | None = None,
    ) -> None:
        self.params = params
        super().__init__(assets, instruction=instruction)

        self._validate_candidates(self.assets, params)
        self.ref_object = self.assets.by_role(REF_ROLE)[0]
        self._relation_by_scene_name = {
            self.assets.by_role(key)[0].scene_name: candidate.relation
            for key, candidate in params.pose_reset.candidates.items()
            if candidate.relation is not None
        }

    @classmethod
    def candidate_keys(cls, assets: TaskAssets) -> list[str]:
        """Asset roles that may fill the pick role, sorted."""
        return sorted(
            key
            for key in assets.role_candidates
            if key.startswith(CANDIDATE_PREFIX)
        )

    @classmethod
    def _placed_keys(cls, assets: TaskAssets) -> list[str]:
        """Everything arranged around the reference, sorted."""
        return sorted(key for key in assets.role_candidates if key != REF_ROLE)

    @staticmethod
    def axis_for(candidate: SpatialCandidate) -> float | None:
        """The direction this candidate must hold, if any.

        Read straight off the config: a direction is something the YAML
        states, not something inferred from the wording, so the file
        shows where each object goes without knowing this class.
        """
        return candidate.axis

    @classmethod
    def _validate_roles(cls, assets: TaskAssets) -> None:
        """Check the pick role by its candidates, not by its own name.

        No asset is filed under ``pick`` itself: the candidates live
        under ``pick_left``, ``pick_near`` and so on, and the role draws
        from all of them.
        """
        if not cls.candidate_keys(assets):
            raise ValueError(
                f"{cls.__name__} needs at least one asset whose role name "
                f"starts with {CANDIDATE_PREFIX!r} to fill the "
                f"{PICK_ROLE!r} role, but the task YAML provides "
                f"{sorted(assets.role_candidates)}."
            )
        if not assets.by_role(REF_ROLE):
            raise ValueError(
                f"{cls.__name__} needs an asset for role {REF_ROLE!r}, but "
                f"the task YAML provides {sorted(assets.role_candidates)}."
            )

    @classmethod
    def _validate_candidates(
        cls,
        assets: TaskAssets,
        params: SpatialTaskParams,
    ) -> None:
        """Check the placement means what the instruction will say."""
        cfg = params.pose_reset
        declared = set(cfg.candidates)
        present = set(cls._placed_keys(assets))
        if declared != present:
            raise ValueError(
                f"{cls.__name__}: pose_reset.candidates describes "
                f"{sorted(declared)} but the task YAML spawns "
                f"{sorted(present)}; everything placed around the "
                "reference needs a placement, and every placement needs "
                "an object."
            )

        if not any(
            candidate.relation is not None
            for candidate in cfg.candidates.values()
        ):
            raise ValueError(
                f"{cls.__name__}: no candidate carries a relation, so no "
                "episode could ask for anything. Give at least one of "
                f"{sorted(declared)} a relation."
            )

        cls._validate_directions(cfg, declared)
        cls._validate_distances(cfg, declared)

    @classmethod
    def _validate_directions(
        cls,
        cfg: SpatialPoseResetConfig,
        declared: set[str],
    ) -> None:
        """A direction word and a pinned axis must agree."""
        for key in sorted(declared):
            candidate = cfg.candidates[key]
            expected = _DIRECTIONAL_AXES.get(candidate.relation or "")
            if expected is None:
                if candidate.relation in _DISTANCE_RELATIONS and (
                    candidate.axis is not None
                ):
                    raise ValueError(
                        f"{cls.__name__}: {key!r} is described by distance "
                        f"({candidate.relation!r}) but pins the "
                        f"{candidate.axis} degree axis. Holding its "
                        "direction still would let that direction identify "
                        "it, which is the thing the wording is meant to "
                        "make a policy read past."
                    )
                continue
            if candidate.axis is None:
                raise ValueError(
                    f"{cls.__name__}: {key!r} claims "
                    f"{candidate.relation!r} but states no axis. A "
                    "direction word is only true of an object that is "
                    f"held there, so say 'axis: {expected}' -- without "
                    "it the object is dealt a fresh direction each "
                    "episode and the wording stops matching the scene."
                )
            if candidate.axis != expected:
                raise ValueError(
                    f"{cls.__name__}: {key!r} claims "
                    f"{candidate.relation!r}, which is the {expected} "
                    f"degree direction, but pins {candidate.axis} degrees. "
                    "An episode would place it somewhere its own "
                    "description does not fit."
                )

    @classmethod
    def _validate_distances(
        cls,
        cfg: SpatialPoseResetConfig,
        declared: set[str],
    ) -> None:
        """Distance words must stay tellable apart, clutter in between."""
        bands: dict[str, list[tuple[float, float]]] = {}
        for key in sorted(declared):
            candidate = cfg.candidates[key]
            if candidate.relation in _DISTANCE_RELATIONS:
                bands.setdefault(candidate.relation, []).append(
                    candidate.radius
                )
        near_bands = bands.get("near", [])
        far_bands = bands.get("far", [])
        if not (near_bands and far_bands):
            return

        near_max = max(band[1] for band in near_bands)
        far_min = min(band[0] for band in far_bands)
        if far_min - near_max < cfg.relation_margin:
            raise ValueError(
                f"{cls.__name__}: the far band starts at {far_min}m but "
                f"the near one reaches {near_max}m, leaving "
                f"{far_min - near_max:.3f}m between them, under the "
                f"{cfg.relation_margin}m margin. An episode could put the "
                "two at nearly the same distance, where neither word "
                "describes the scene."
            )

        # Clutter has to stay in the gap. Outside it, a clutter object
        # could come out nearer than the near candidate or further than
        # the far one, quietly making the instruction false.
        for key in sorted(declared):
            candidate = cfg.candidates[key]
            if candidate.relation is not None:
                continue
            low, high = candidate.radius
            if low < near_max or high > far_min:
                raise ValueError(
                    f"{cls.__name__}: clutter {key!r} sits at "
                    f"{candidate.radius}, outside the "
                    f"[{near_max}, {far_min}] gap between the near and far "
                    "bands. It could come out nearer than the near "
                    "candidate or further than the far one, making the "
                    "instruction describe the wrong object."
                )

    def get_role_candidates(
        self,
        role_id: str,
        *,
        swap: bool = False,
    ) -> list[TargetRef]:
        """Offer the objects this role may point at.

        The pick role draws from every candidate the task YAML spawned,
        each of which carries its own relation and so describes itself
        regardless of ``swap``. The reference is never one of them -- it
        is what the others are described against.
        """
        del swap
        if role_id == PICK_ROLE:
            return [
                TargetRef(spec.scene_name)
                for key in self.candidate_keys(self.assets)
                for spec in self.assets.by_role(key)
            ]
        return [
            TargetRef(spec.scene_name) for spec in self.assets.by_role(role_id)
        ]

    def get_event_cfg(self) -> EventManagerCfg:
        """Arrange the whole scene with a single term.

        One term places the reference and everything around it, so the
        objects are compared against each other as they go down and
        there is no placement state to share between terms.
        """
        cfg = self.params.pose_reset
        scene_name_of = {
            key: self.assets.by_role(key)[0].scene_name
            for key in cfg.candidates
        }
        terms: dict[str, Any] = {
            "pose_reset_event": PolarResetTermCfg(
                asset_cfgs=[
                    SceneEntityCfg(name=name)
                    for name in self.assets.all_scene_names()
                ],
                trigger_topic="reset",
                anchor_name=self.ref_object.scene_name,
                anchor_region=cfg.anchor_range,
                workspace=cfg.workspace,
                radius={
                    scene_name_of[key]: candidate.radius
                    for key, candidate in cfg.candidates.items()
                },
                axis={
                    scene_name_of[key]: axis
                    for key, candidate in cfg.candidates.items()
                    if (axis := self.axis_for(candidate)) is not None
                },
                axis_jitter=cfg.axis_jitter,
                min_separation=cfg.min_separation,
            )
        }

        light_reset_cfg = self.params.light_reset
        if light_reset_cfg is not None and light_reset_cfg.enabled:
            terms["light_reset_event"] = LightResetTermCfg(
                asset_cfgs=[
                    SceneEntityCfg(name=name)
                    for name in light_reset_cfg.asset_names
                ],
                trigger_topic="reset",
                randomize_color=light_reset_cfg.randomize_color,
                color_temperature_range=(
                    light_reset_cfg.color_temperature_range
                ),
                rgb_noise=light_reset_cfg.rgb_noise,
                randomize_intensity=light_reset_cfg.randomize_intensity,
                intensity_range=light_reset_cfg.intensity_range,
                randomize_position=light_reset_cfg.randomize_position,
                position_cfg=light_reset_cfg.position_cfg,
                crazy_randomization_rate=(
                    light_reset_cfg.crazy_randomization_rate
                ),
            )
        texture_reset_cfg = self.params.texture_reset
        if texture_reset_cfg is not None and texture_reset_cfg.enabled:
            terms["texture_reset_event"] = TextureResetTermCfg(
                asset_cfgs=[
                    SceneEntityCfg(name=name)
                    for name in texture_reset_cfg.asset_names
                ],
                trigger_topic="reset",
                variant_set_name=texture_reset_cfg.variant_set_name,
                variant_sort=texture_reset_cfg.variant_sort,
                variant_index_range=texture_reset_cfg.variant_index_range,
            )
        return EventManagerCfg(terms=terms)

    def build_validator(
        self,
        context: ValidatorContext | None = None,
    ) -> Validator:
        """Score whichever candidate the pick role names right now."""
        if context is None or context.robot is None:
            raise ValueError(
                "SpatialTask.build_validator() requires ValidatorContext "
                "with robot data."
            )
        metric_store = MetricStore(RoleScope.from_task(self))
        checker_suite = SceneCheckerSuite(
            (
                ReachChecker(threshold=0.2),
                ContactChecker(force_threshold=0.1),
                LiftChecker(threshold=0.03),
            )
        )
        reach_pick = BoundRoleMetricSelector(
            metric_store=metric_store,
            context=context,
            role_id=PICK_ROLE,
            metric="reached",
        )
        contact_pick = BoundRoleMetricSelector(
            metric_store=metric_store,
            context=context,
            role_id=PICK_ROLE,
            metric="contacted",
        )
        grasp_pick = BoundRoleMetricSelector(
            metric_store=metric_store,
            context=context,
            role_id=PICK_ROLE,
            metric="grasped",
        )
        lift_pick = BoundRoleMetricSelector(
            metric_store=metric_store,
            context=context,
            role_id=PICK_ROLE,
            metric="lifted",
        )
        lifted_while_grasped = AllMetricSelector((lift_pick, grasp_pick))
        return Validator(
            context=context,
            checker_suite=checker_suite,
            metric_store=metric_store,
            criteria=[
                DwellMetricSelector(
                    reach_pick,
                    self.params.reach_dwell_steps,
                ),
                (
                    contact_pick,
                    [0],
                ),
                (
                    grasp_pick,
                    [1],
                ),
                (
                    lifted_while_grasped,
                    [2],
                ),
            ],
            criteria_name=[
                "reach_pick",
                "contact_pick",
                "grasp_pick",
                "lift_pick",
            ],
        )

    def build_instruction_context(
        self,
        env: Any,
        *,
        actor_description_seed: int,
        context: ValidatorContext | None = None,
    ) -> Mapping[str, Any]:
        """Describe the bound candidate by its relation to the reference."""
        if context is None:
            raise ValueError(
                "SpatialTask.build_instruction_context() requires a "
                "ValidatorContext to resolve role bindings."
            )
        if self.instruction is None:
            return {}

        target = context.role_registry.resolve_one(PICK_ROLE)
        try:
            relation = self._relation_by_scene_name[target.scene_name]
        except KeyError:
            raise KeyError(
                f"Role {PICK_ROLE!r} is bound to {target.scene_name!r}, "
                "which carries no relation and so cannot be described. "
                f"Describable: {sorted(self._relation_by_scene_name)}."
            ) from None

        mode = self.instruction.actor_description_mode
        pick_actor = InstructionActor.from_rigid_object(
            env.scene[target.scene_name],
            actor_description_mode=mode,
            actor_description_seed=actor_description_seed,
        )
        ref_actor = InstructionActor.from_rigid_object(
            env.scene[self.ref_object.scene_name],
            actor_description_mode=mode,
            actor_description_seed=actor_description_seed,
        )
        return {
            "actor1": pick_actor,
            "obj": pick_actor,
            "actor2": ref_actor,
            "ref_obj": ref_actor,
            "spatial_relation": relation_phrase(relation),
        }


__all__ = [
    "PICK_ROLE",
    "REF_ROLE",
    "CANDIDATE_PREFIX",
    "SpatialRelation",
    "SpatialCandidate",
    "SpatialPoseResetConfig",
    "SpatialTaskParams",
    "SpatialTask",
    "relation_phrase",
]
