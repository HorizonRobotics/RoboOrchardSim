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

from __future__ import annotations
from typing import Dict, List, Optional

import curobo.util.usd_helper as usd_helper

# Standard Library
import torch

# Third Party
import torch.autograd.profiler as profiler
from curobo.geom.types import (
    WorldConfig,
)

# CuRobo
from curobo.rollout.rollout_base import Goal
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.tensor_util import tensor_repeat_seeds
from curobo.util.trajectory import InterpolateType
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.reacher.ik_solver import IKResult
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenPlanConfig,
    MotionGenResult,
    MotionGenStatus,
)
from curobo.wrap.reacher.types import ReacherSolveState, ReacherSolveType
from pxr import UsdGeom


class MotionGenEx(MotionGen):
    """Extended version of MotionGen.

    Only override the internal functions that enable multi
    IK solutions in batch mode.
    """

    def _plan_from_solve_state_batch(
        self,
        solve_state: ReacherSolveState,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> MotionGenResult:
        """Plan from a given reacher solve state in batch mode.

        Args:
            solve_state: Reacher solve state for planning.
            start_state: Start joint state for planning.
            goal_pose: Goal poses to reach for end-effector.
            plan_config: Planning parameters for motion generation.
            link_poses: Goal poses for other links in the robot.

        Returns:
            MotionGenResult: Result of planning.
        """
        if plan_config is None:
            plan_config = MotionGenPlanConfig()

        self._trajopt_goal_config[:] = self.get_retract_config().view(
            1, 1, self._dof
        )
        trajopt_seed_traj = None
        trajopt_seed_success = None
        trajopt_newton_iters = None
        graph_success = 0

        # plan ik:
        ik_result = self._solve_ik_from_solve_state(
            goal_pose=goal_pose,
            solve_state=solve_state,
            start_state=start_state,
            use_nn_seed=plan_config.use_nn_ik_seed,
            partial_ik_opt=plan_config.partial_ik_opt,
            link_poses=link_poses,
        )

        # -----------------Start of Post Processing of IK Selection------------
        batch_dim, ik_dim, dof = ik_result.solution.shape  # [B,20,dof]
        start_pos = start_state.position[:, None, :].expand(
            batch_dim, ik_dim, dof
        )
        delta_norm = torch.norm(
            ik_result.solution - start_pos, dim=2
        )  # [B,20]
        valid_mask = ik_result.success.bool()  # [B,20]

        # Sort each batch: prioritize success, then ascending order of norm
        sorted_indices = []
        for b in range(batch_dim):
            # Indices of successful solutions
            success_idx = torch.where(valid_mask[b])[0]
            fail_idx = torch.where(~valid_mask[b])[0]
            # Sort successful solutions by norm
            if len(success_idx) > 0:
                success_norm = delta_norm[b, success_idx]
                success_sorted = success_idx[torch.argsort(success_norm)]
            else:
                success_sorted = torch.tensor(
                    [],
                    dtype=torch.long,
                    device=delta_norm.device,
                )
            # Sort failed solutions by norm
            if len(fail_idx) > 0:
                fail_norm = delta_norm[b, fail_idx]
                fail_sorted = fail_idx[torch.argsort(fail_norm)]
            else:
                fail_sorted = torch.tensor(
                    [],
                    dtype=torch.long,
                    device=delta_norm.device,
                )

            indices = torch.cat([success_sorted, fail_sorted], dim=0)
            sorted_indices.append(indices)
        sorted_indices = torch.stack(sorted_indices, dim=0)  # [B, ik_dim]

        def batch_gather(data, idx):
            # data: [B, ik_dim, ...], idx: [B, ik_dim]
            B = data.shape[0]
            out = []
            for b in range(B):
                out.append(data[b][idx[b]])
            return torch.stack(out, dim=0)

        ik_result.solution = batch_gather(ik_result.solution, sorted_indices)
        ik_result.success = batch_gather(ik_result.success, sorted_indices)
        ik_result.position_error = batch_gather(
            ik_result.position_error, sorted_indices
        )
        ik_result.rotation_error = batch_gather(
            ik_result.rotation_error, sorted_indices
        )
        # --------------End of Post Processing of IK Selection----------------

        if not plan_config.enable_graph and plan_config.partial_ik_opt:
            ik_result.success[:] = True

        # check for success:
        result = MotionGenResult(
            ik_result.success,
            position_error=ik_result.position_error,
            rotation_error=ik_result.rotation_error,
            ik_time=ik_result.solve_time,
            solve_time=ik_result.solve_time,
            debug_info={},
            # goalset_index=ik_result.goalset_index,
        )

        ik_success = torch.count_nonzero(ik_result.success)
        if ik_success == 0:
            result.status = MotionGenStatus.IK_FAIL
            result.success = result.success[:, 0]
            return result

        # do graph search:
        # ik_out_seeds = solve_state.num_trajopt_seeds
        # if plan_config.enable_graph:
        #     ik_out_seeds = min(solve_state.num_trajopt_seeds, ik_success)

        # if not plan_config.enable_opt and plan_config.enable_graph:
        #    self.graph_planner.interpolation_steps = self.interpolation_steps
        #    self.graph_planner.interpolation_type = self.interpolation_type
        # elif plan_config.enable_graph:
        #    self.graph_planner.interpolation_steps = self.trajopt_solver.traj_tsteps # noqa: E501
        #    self.graph_planner.interpolation_type = InterpolateType.LINEAR
        goal_config = ik_result.solution[ik_result.success].view(
            -1, self.ik_solver.dof
        )

        # get shortest path
        if plan_config.enable_graph:
            interpolation_steps = None
            if plan_config.enable_opt:
                interpolation_steps = self.trajopt_solver.action_horizon

            # Get the actual number of IK seeds used (from ik_result shape)
            _, actual_ik_seeds, _ = ik_result.solution.shape

            # Create expanded start configs for all IK seeds
            start_config_expanded = tensor_repeat_seeds(
                start_state.position, actual_ik_seeds
            )
            # Filter to only successful IK solutions
            start_config = start_config_expanded[ik_result.success.view(-1)]
            graph_result = self.graph_search(
                start_config, goal_config, interpolation_steps
            )
            graph_success = torch.count_nonzero(graph_result.success).item()

            result.graph_time = graph_result.solve_time
            result.solve_time += graph_result.solve_time
            if graph_success > 0:
                # path = graph_result.interpolated_plan
                result.graph_plan = graph_result.interpolated_plan
                result.interpolated_plan = graph_result.interpolated_plan
                result.used_graph = True

                if plan_config.enable_opt:
                    # Extract graph plan positions
                    graph_positions = result.graph_plan.position.view(
                        graph_success, interpolation_steps, self._dof
                    )

                    trajopt_seed_traj = torch.zeros(
                        (
                            solve_state.num_trajopt_seeds,
                            solve_state.batch_size,
                            self.trajopt_solver.action_horizon,
                            self._dof,
                        ),
                        device=self.tensor_args.device,
                        dtype=self.tensor_args.dtype,
                    )

                    max_seeds = (
                        solve_state.num_trajopt_seeds * solve_state.batch_size
                    )
                    for i in range(min(graph_success, max_seeds)):
                        seed_idx = i % solve_state.num_trajopt_seeds
                        batch_idx = i // solve_state.num_trajopt_seeds
                        if batch_idx < solve_state.batch_size:
                            trajopt_seed_traj[
                                seed_idx, batch_idx, :interpolation_steps, :
                            ] = graph_positions[i]

                    trajopt_seed_success = torch.zeros(
                        (
                            solve_state.batch_size,
                            solve_state.num_trajopt_seeds,
                        ),
                        dtype=torch.bool,
                        device=ik_result.success.device,
                    )

                    if graph_success > 0:
                        for i in range(min(graph_success, max_seeds)):
                            seed_idx = i % solve_state.num_trajopt_seeds
                            batch_idx = i // solve_state.num_trajopt_seeds
                            if batch_idx < solve_state.batch_size:
                                trajopt_seed_success[batch_idx, seed_idx] = (
                                    True
                                )

                    trajopt_newton_iters = self.graph_trajopt_iters

                else:
                    ik_success = ik_result.success.view(-1).clone()

                    g_dim = torch.nonzero(ik_success).view(-1)[
                        graph_result.success
                    ]

                    self._batch_graph_search_buffer.copy_at_index(
                        graph_result.interpolated_plan, g_dim
                    )

                    # result.graph_plan = JointState.from_position(
                    #    self._batch_graph_search_buffer,
                    #    joint_names=graph_result.interpolated_plan.joint_names, # noqa: E501
                    # )
                    result.interpolated_plan = self._batch_graph_search_buffer
                    g_dim = g_dim.cpu().squeeze().tolist()
                    if isinstance(g_dim, int):
                        g_dim = [g_dim]
                    for x, x_val in enumerate(g_dim):
                        self._batch_path_buffer_last_tstep[x_val] = (
                            graph_result.path_buffer_last_tstep[x]
                        )
                    result.path_buffer_last_tstep = (
                        self._batch_path_buffer_last_tstep
                    )
                    result.optimized_plan = result.interpolated_plan
                    result.optimized_dt = torch.as_tensor(
                        [
                            self.interpolation_dt
                            for i in range(
                                result.interpolated_plan.position.shape[0]
                            )
                        ],
                        device=self.tensor_args.device,
                        dtype=self.tensor_args.dtype,
                    )
                    result.success = result.success.view(-1).clone()
                    result.success[ik_success][graph_result.success] = True
                    return result

            else:
                result.success[:] = False
                result.success = result.success[:, 0]
                result.status = MotionGenStatus.GRAPH_FAIL
                if not graph_result.valid_query:
                    result.valid_query = False
                    if self.store_debug_in_result:
                        result.debug_info = {
                            "graph_debug": graph_result.debug_info
                        }
                    return result

        if plan_config.enable_opt:
            batch_size = ik_result.success.shape[0]

            successful_configs = []
            for b in range(batch_size):
                batch_success = ik_result.success[b]
                if batch_success.any():
                    first_success_idx = torch.where(batch_success)[0][0]
                    successful_configs.append(
                        ik_result.solution[b, first_success_idx]
                    )
                else:
                    successful_configs.append(ik_result.solution[b, 0])

            goal_config_batch = torch.stack(successful_configs, dim=0)

            # Update trajopt_goal_config properly
            if len(goal_config_batch.shape) == 2:
                # Reshape to [batch, 1, dof] to match trajopt_goal_config
                self._trajopt_goal_config[:batch_size, 0, :] = (
                    goal_config_batch
                )
            else:
                self._trajopt_goal_config[:batch_size] = goal_config_batch

            goal_config = self._trajopt_goal_config  # batch index == 0

            goal = Goal(
                goal_pose=goal_pose,
                current_state=start_state,
                links_goal_pose=link_poses,
            )
            # generate seeds:
            if trajopt_seed_traj is None or (
                plan_config.enable_graph
                and graph_success < solve_state.batch_size
            ):
                seed_link_poses = None
                if link_poses is not None:
                    seed_link_poses = {}

                    for k in link_poses.keys():
                        seed_link_poses[k] = link_poses[k].repeat_seeds(
                            solve_state.num_trajopt_seeds
                        )
                seed_goal = Goal(
                    goal_pose=goal_pose.repeat_seeds(
                        solve_state.num_trajopt_seeds
                    ),
                    current_state=start_state.repeat_seeds(
                        solve_state.num_trajopt_seeds
                    ),
                    goal_state=JointState.from_position(
                        goal_config.view(-1, self._dof)
                    ),
                    links_goal_pose=seed_link_poses,
                )
                if trajopt_seed_traj is not None:
                    trajopt_seed_traj = trajopt_seed_traj.transpose(
                        0, 1
                    ).contiguous()

                # create seeds here:
                trajopt_seed_traj = self.trajopt_solver.get_seed_set(
                    seed_goal,
                    trajopt_seed_traj,  # batch, num_seeds, h, dof
                    num_seeds=1,
                    batch_mode=solve_state.batch_mode,
                    seed_success=trajopt_seed_success,
                )
                trajopt_seed_traj = trajopt_seed_traj.view(
                    solve_state.num_trajopt_seeds,
                    solve_state.batch_size,
                    self.trajopt_solver.action_horizon,
                    self._dof,
                ).contiguous()
            if plan_config.enable_finetune_trajopt:
                og_value = self.trajopt_solver.interpolation_type
                self.trajopt_solver.interpolation_type = (
                    InterpolateType.LINEAR_CUDA
                )

            traj_result = self._solve_trajopt_from_solve_state(
                goal,
                solve_state,
                trajopt_seed_traj,
                newton_iters=trajopt_newton_iters,
                return_all_solutions=plan_config.enable_finetune_trajopt,
            )

            # output of traj result will have 1 solution per batch

            # run finetune opt on 1 solution per batch:
            if plan_config.enable_finetune_trajopt:
                self.trajopt_solver.interpolation_type = og_value
            if self.store_debug_in_result:
                result.debug_info["trajopt_result"] = traj_result

            # run finetune
            if (
                plan_config.enable_finetune_trajopt
                and torch.count_nonzero(traj_result.success) > 0
            ):
                with profiler.record_function("motion_gen/finetune_trajopt"):
                    seed_traj = (
                        traj_result.raw_action.clone()
                    )  # solution.position.clone()
                    seed_traj = seed_traj.contiguous()
                    og_solve_time = traj_result.solve_time

                    scaled_dt = torch.clamp(
                        torch.max(
                            traj_result.optimized_dt[traj_result.success]
                        )
                        * self.finetune_dt_scale,
                        self.trajopt_solver.minimum_trajectory_dt,
                    )
                    self.finetune_trajopt_solver.update_solver_dt(
                        scaled_dt.item()
                    )

                    traj_result = self._solve_trajopt_from_solve_state(
                        goal,
                        solve_state,
                        seed_traj,
                        trajopt_instance=self.finetune_trajopt_solver,
                        num_seeds_override=solve_state.num_trajopt_seeds,
                    )

                result.finetune_time = traj_result.solve_time

                traj_result.solve_time = og_solve_time
                if self.store_debug_in_result:
                    result.debug_info["finetune_trajopt_result"] = traj_result
            elif (
                plan_config.enable_finetune_trajopt
                and len(traj_result.success.shape) == 2
            ):
                traj_result.success = traj_result.success[:, 0]

            result.success = traj_result.success

            result.interpolated_plan = traj_result.interpolated_solution
            result.solve_time += traj_result.solve_time
            result.trajopt_time = traj_result.solve_time
            result.position_error = traj_result.position_error
            result.rotation_error = traj_result.rotation_error
            result.cspace_error = traj_result.cspace_error
            result.goalset_index = traj_result.goalset_index
            result.path_buffer_last_tstep = traj_result.path_buffer_last_tstep
            result.optimized_plan = traj_result.solution
            result.optimized_dt = traj_result.optimized_dt
            if torch.count_nonzero(traj_result.success) == 0:
                result.status = MotionGenStatus.TRAJOPT_FAIL
                result.success[:] = False
            if self.store_debug_in_result:
                result.debug_info = {"trajopt_result": traj_result}
        return result

    @profiler.record_function("motion_gen/ik")
    def _solve_ik_from_solve_state(
        self,
        goal_pose: Pose,
        solve_state: ReacherSolveState,
        start_state: JointState,
        use_nn_seed: bool,
        partial_ik_opt: bool,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> IKResult:
        """Solve inverse kinematics from solve state.

        used by motion generation planning call.
        """
        newton_iters = self.partial_ik_iters if partial_ik_opt else None

        if (
            solve_state.solve_type == ReacherSolveType.BATCH
            or solve_state.solve_type == ReacherSolveType.BATCH_ENV
            or solve_state.solve_type == ReacherSolveType.BATCH_ENV_GOALSET
        ):
            # trajopt_seeds = solve_state.num_trajopt_seeds
            trajopt_seeds = 20
        else:
            trajopt_seeds = solve_state.num_trajopt_seeds

        ik_result = self.ik_solver.solve_any(
            solve_state.solve_type,
            goal_pose,
            start_state.position.view(-1, self._dof),
            start_state.position.view(-1, 1, self._dof),
            trajopt_seeds,
            solve_state.num_ik_seeds,
            use_nn_seed,
            newton_iters,
            link_poses,
        )
        return ik_result

    def update_world(self, worlds: list[WorldConfig] | WorldConfig):
        """Over write update_world with batch world.

        This allows for updating the world representation as long as the new
        world representation does not have a larger number of obstacles than
        the :attr:`MotionGen.collision_cache` as created during initialization
        of :class:`MotionGenConfig`. Updating the world also invalidates the
        cached roadmaps in the graph planner. See :ref:`world_collision` for
        more details.

        Args:
            world: New world configuration for collision checking.
        """
        if isinstance(worlds, WorldConfig):
            worlds = [worlds]

        self.world_coll_checker.load_batch_collision_model(worlds)
        self.graph_planner.reset_buffer()


class UsdHelperExt(UsdHelper):
    def _object_classification(self, prim_path, obstacles: dict, r_T_w):
        x = prim_path
        if x.IsA(UsdGeom.Cube):
            if obstacles["cuboid"] is None:
                obstacles["cuboid"] = []
            cube = usd_helper.get_cube_attrs(
                x, cache=self._xform_cache, transform=r_T_w
            )
            obstacles["cuboid"].append(cube)
        elif x.IsA(UsdGeom.Sphere):
            if obstacles["sphere"] is None:
                obstacles["sphere"] = []
            obstacles["sphere"].append(
                usd_helper.get_sphere_attrs(
                    x, cache=self._xform_cache, transform=r_T_w
                )
            )
        elif x.IsA(UsdGeom.Mesh):
            if obstacles["mesh"] is None:
                obstacles["mesh"] = []
            m_data = usd_helper.get_mesh_attrs(
                x, cache=self._xform_cache, transform=r_T_w
            )
            if m_data is not None:
                obstacles["mesh"].append(m_data)
        elif x.IsA(UsdGeom.Cylinder):
            if obstacles["cylinder"] is None:
                obstacles["cylinder"] = []
            cube = usd_helper.get_cylinder_attrs(
                x, cache=self._xform_cache, transform=r_T_w
            )
            obstacles["cylinder"].append(cube)
        elif x.IsA(UsdGeom.Capsule):
            if obstacles["capsule"] is None:
                obstacles["capsule"] = []
            cap = usd_helper.get_capsule_attrs(
                x, cache=self._xform_cache, transform=r_T_w
            )
            obstacles["capsule"].append(cap)

    def get_obstacles_from_stage(
        self,
        only_paths: Optional[List[str]] = None,
        ignore_paths: Optional[List[str]] = None,
        only_substring: Optional[List[str]] = None,
        ignore_substring: Optional[List[str]] = None,
        reference_prim_path: Optional[str] = None,
        timecode: float = 0,
    ) -> WorldConfig:
        # read obstacles from usd by iterating through all prims:
        obstacles = {
            "cuboid": [],
            "sphere": None,
            "mesh": None,
            "cylinder": None,
            "capsule": None,
        }
        r_T_w = None
        self._xform_cache.Clear()
        self._xform_cache.SetTime(timecode)
        if reference_prim_path is not None:
            reference_prim = self.stage.GetPrimAtPath(reference_prim_path)
            r_T_w, _ = usd_helper.get_prim_world_pose(
                self._xform_cache, reference_prim, inverse=True
            )
        all_items = self.stage.Traverse()
        for x in all_items:
            if only_paths is not None:
                if not any(
                    [str(x.GetPath()).startswith(k) for k in only_paths]
                ):
                    continue
            if ignore_paths is not None:
                if any([str(x.GetPath()).startswith(k) for k in ignore_paths]):
                    continue
            if only_substring is not None:
                if not any([k in str(x.GetPath()) for k in only_substring]):
                    continue
            if ignore_substring is not None:
                if any([k in str(x.GetPath()) for k in ignore_substring]):
                    continue
            # if in instance do extra process
            if x.IsInstance():
                master = x.GetPrototype()  # 注意是 GetPrototype
                instance_path = x.GetPath()
                master_root = master.GetPath()
                # 遍历 master tree
                for m in self._traverse_prim_tree(master):
                    # 计算相对路径
                    rel = m.GetPath().MakeRelativePath(master_root)
                    # 拼出实例路径下的“逻辑” prim 路径
                    inst_path = instance_path.AppendPath(rel)
                    # 拿回真正 Prim
                    inst_prim = self.stage.GetPrimAtPath(inst_path)
                    if not inst_prim:
                        continue
                    # 传给分类逻辑
                    self._object_classification(inst_prim, obstacles, r_T_w)
            else:
                self._object_classification(x, obstacles, r_T_w)

        world_model = WorldConfig(**obstacles)
        return world_model

    def _traverse_prim_tree(self, prim):
        yield prim
        for child in prim.GetChildren():
            yield from self._traverse_prim_tree(child)
