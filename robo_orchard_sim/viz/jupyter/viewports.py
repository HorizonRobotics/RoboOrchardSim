# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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
import time
import weakref
from typing import TYPE_CHECKING, Any

import torch
from robo_orchard_core.datatypes.geometry import BatchPose6D
from robo_orchard_core.utils.math.coord_convention import CoordAxisOpenGL
from robo_orchard_core.viz.jupyter import IpyFPVCameraViz, draw_image

from robo_orchard_sim.ext.sim.context import SimulationContext
from robo_orchard_sim.ext.sim.viewports import ViewportCamera

if TYPE_CHECKING:
    from ipycanvas import Canvas


class IsaacIpyViewportViz(IpyFPVCameraViz):
    """A first-person viewport visualization class for Isaac.

    Camera control:
        - Left click and drag to rotate the camera.
        - Right click and drag to move the camera.
        - Scroll to move forward/backward.
        - Ctrl + scroll to change move scale.

    This class will add a render callback to the simulation context to capture
    the viewport image. When a event (e.g., mouse move) triggers the render,
    the image will be captured and displayed on the canvas.


    Args:
        height (int): The height of the canvas.
        width (int): The width of the canvas.
        sim_ctx (SimulationContext): The simulation context.
        additional_watched_events (list[str], optional): The additional watched
            events. Default is None.
        canvas (Canvas, optional): The canvas to draw the image. If not
            provided, a new canvas will be created. Default is None.
        view_camera (ViewportCamera, optional): The viewport camera. If not
            provided, use the current active viewport camera. Default is None.
        add_render_callback: (bool, optional): Whether to add a render callback
            to the simulation context. If True, the ipywidgets will be updated
            when simulation context renders. Default is False.
        max_fps (int, optional): The maximum frames per second. Default is 20.
        format (str, optional): The format of the image. Default is "jpeg".
        quality (int, optional): The quality of the image. Default is 80.
        initial_move_scale (float, optional): The initial move scale.
            Default is 2.5.
        forward_sensitivity (float, optional): The forward sensitivity.
            Default is 0.001.
        translation_sensitivity (float, optional): The translation sensitivity.
            Default is 1.
        rotation_sensitivity (float, optional): The rotation sensitivity.
            Default is 0.5.
    """

    def __init__(
        self,
        height: int,
        width: int,
        sim_ctx: SimulationContext,
        additional_watched_events: list[str] | None = None,
        canvas: Canvas | None = None,
        view_camera: ViewportCamera | None = None,
        add_render_callback: bool = False,
        max_fps: int = 20,
        format="jpeg",
        quality=80,
        initial_move_scale: float = 2.5,
        forward_sensitivity: float = 0.001,
        translation_sensitivity: float = 1,
        rotation_sensitivity: float = 0.5,
    ):
        if view_camera is None:
            view_camera = ViewportCamera()
        self._sim_ctx: SimulationContext = weakref.proxy(sim_ctx)
        self._view_camera: ViewportCamera = view_camera
        self._format = format
        self._quality = quality

        if self._sim_ctx._viewport_context is None:
            raise ValueError(
                "The simulation context does not have a viewport context. "
                "Please disable headless mode, or set livestream to 1 in the "
                "isaac simulation app configuration."
            )

        self._view_buffer_ready = False
        self._buffer_img: torch.Tensor = torch.Tensor()

        super().__init__(
            height=height,
            width=width,
            local_coord_axis=CoordAxisOpenGL(),
            additional_watched_events=additional_watched_events,
            canvas=canvas,
            max_fps=max_fps,
            initial_move_scale=initial_move_scale,
            forward_sensitivity=forward_sensitivity,
            translation_sensitivity=translation_sensitivity,
            rotation_sensitivity=rotation_sensitivity,
        )
        self._add_render_callback = add_render_callback
        if self._add_render_callback:
            sim_ctx.add_render_callback(
                f"{id(self)}_render_callback", self._on_capture
            )
        self._render()

    def __del__(self):
        if self._add_render_callback:
            self._sim_ctx.remove_render_callback(f"{id(self)}_render_callback")

    def get_pose_view_world(self) -> BatchPose6D:
        return BatchPose6D(
            xyz=torch.asarray(self._view_camera.position_world).reshape(1, 3),
            quat=torch.asarray(self._view_camera.rotation_quat_world).reshape(
                1, 4
            ),
        )

    def _apply_to_view_local(
        self, translation: torch.Tensor | None, quat: torch.Tensor | None
    ):
        cam2world = self.get_pose_view_world()
        if [translation, quat].count(None) != 1:
            raise ValueError("Either translation or quat must be provided.")

        new_cam2cam = BatchPose6D.identity(batch_size=1, device="cpu")

        if translation is not None:
            new_cam2cam.xyz[0, ...] = translation
            new_cam2world = new_cam2cam.compose(cam2world)

            self._view_camera.set_position_world(
                pos=(
                    new_cam2world.xyz[0, 0].item(),
                    new_cam2world.xyz[0, 1].item(),
                    new_cam2world.xyz[0, 2].item(),
                ),
                rotate=False,
            )
            return

        if quat is not None:
            new_cam2cam.quat[0, ...] = quat
            new_cam2world = new_cam2cam.compose(cam2world)
            self._view_camera.set_rotation_quat_world(
                quat=(
                    new_cam2world.quat[0, 0].item(),
                    new_cam2world.quat[0, 1].item(),
                    new_cam2world.quat[0, 2].item(),
                    new_cam2world.quat[0, 3].item(),
                ),
            )
            return

    def get_rendered_image(
        self, update_view: bool = True, sleep_time: float = 0.03
    ) -> torch.Tensor:
        """Get the rendered image.

        Args:
            update_view (bool, optional): Whether to update the view.
                Default is True.
            sleep_time (float, optional): The sleep time after updating
                the view. Default is 0.03.

        """
        if update_view:
            self._render()
            if sleep_time > 0:
                time.sleep(sleep_time)
        return self.captured_image

    def _on_capture(self, event: Any = None):
        weak_ref_self: IsaacIpyViewportViz = weakref.proxy(self)

        def on_capture_fn(img: torch.Tensor):
            weak_ref_self._buffer_img = img.clone()
            draw_image(
                weak_ref_self.canvas,
                weak_ref_self._buffer_img.numpy(),
                format=weak_ref_self._format,
                quality=weak_ref_self._quality,
            )

        # if weak_ref_self._sim_ctx._app.get_update_number() % 2 == 0:
        self._view_camera.capture_viewport_to_buffer(on_capture_fn)

    def _render(self):
        # it seems that need to render twice to get the correct image
        for _ in range(2):
            if not self._add_render_callback:
                self._on_capture()
            self._sim_ctx.render()

    @property
    def captured_image(self) -> torch.Tensor:
        return self._buffer_img

    def set_pose_world(self, pos: tuple[float, float, float]):
        """Set the camera position in the world frame.

        Args:
            pos (torch.Tensor): The position in the world frame.
        """
        self._view_camera.set_position_world(pos, rotate=False)

    def set_rotation_quat_world(self, quat: tuple[float, float, float, float]):
        """Set the camera rotation in the world frame.

        Args:
            quat (torch.Tensor): The quaternion in the world frame.
        """
        self._view_camera.set_rotation_quat_world(quat)

    def set_rotation_by_look_at(self, target: tuple[float, float, float]):
        """Set the camera rotation by looking at the target position.

        Args:
            target (torch.Tensor): The target position.
        """
        self._view_camera.set_look_target_world(target=target, rotate=True)
