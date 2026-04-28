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

"""Six-direction mesh renderer using nvdiffrast for headless GPU rendering.

Replaces the previous pyrender-based renderer. nvdiffrast uses CUDA directly
for rasterization, so it works without any display server (X11, EGL, etc.).

Requirements:
    - PyTorch with CUDA support
    - nvdiffrast  (pip install nvdiffrast)
"""

import logging
import math
import os

import numpy as np
import torch
import trimesh
from PIL import Image

logger = logging.getLogger(__name__)

# Camera view definitions: (azimuth_degrees, elevation_degrees)
# Z-up convention with semantic "front" aligned to viewing from +X to -X:
# - front: eye at +X, looking toward -X
# - right: eye at +Y, looking toward -Y
# - back: eye at -X, looking toward +X
# - left: eye at -Y, looking toward +Y
# - top: eye at +Z
# - bottom: eye at -Z
CAMERA_VIEWS = {
    "front": (-90, 0),
    "right": (0, 0),
    "back": (90, 0),
    "left": (180, 0),
    "top": (0, 90),
    "bottom": (0, -90),
}

VIEW_PRESETS = {
    4: ["front", "right", "back", "left"],
    6: ["front", "right", "back", "left", "top", "bottom"],
}


def select_views(num_views: int) -> dict[str, tuple[float, float]]:
    """Select a subset of camera views by count.

    Args:
        num_views: Number of views to select (4 or 6 supported as presets).
            Falls back to first *num_views* entries from CAMERA_VIEWS.

    Returns:
        Ordered dict of {view_name: (azimuth, elevation)}.
    """
    if num_views in VIEW_PRESETS:
        names = VIEW_PRESETS[num_views]
    else:
        names = list(CAMERA_VIEWS.keys())[:num_views]
    return {name: CAMERA_VIEWS[name] for name in names}


# Lazily initialised rasterization context (reused across calls)
_glctx = None


def _get_glctx():
    """Get or create a reusable nvdiffrast rasterization context.

    nvdiffrast 0.2.x exposes only a GL-based context
    (RasterizeGLContext). This still runs fully on the GPU but uses a
    headless OpenGL context under the hood (EGL/GLX), so we do not need
    a windowing system in Python code.
    """
    global _glctx
    if _glctx is None:
        import nvdiffrast.torch as dr

        _glctx = dr.RasterizeGLContext()
    return _glctx


def render_views(
    mesh: trimesh.Trimesh | trimesh.Scene,
    output_dir: str,
    views: dict[str, tuple[float, float]] = None,
    resolution: tuple[int, int] = (512, 512),
    background_color: tuple[int, ...] = (255, 255, 255, 0),
) -> list[str]:
    """Render a mesh from multiple camera viewpoints.

    Uses nvdiffrast for headless GPU-accelerated rendering with texture
    support.  No display server (X11 / EGL) is required.

    Args:
        mesh: A trimesh.Trimesh or trimesh.Scene object.
        output_dir: Directory to save rendered images.
        views: Dict of {view_name: (azimuth_deg, elevation_deg)}.
            Defaults to 6-direction views (front/back/left/right/top/bottom).
        resolution: Output image resolution as (width, height).
        background_color: RGBA background color.

    Returns:
        List of saved image file paths, ordered by view name.
    """
    try:
        import nvdiffrast.torch as dr
    except ImportError:
        raise ImportError(
            "Rendering requires 'nvdiffrast'. "
            "Install with: pip install nvdiffrast"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "nvdiffrast renderer requires CUDA, but torch.cuda.is_available() "
            "returned False.  Make sure you are running on a GPU node with "
            "the correct CUDA drivers."
        )

    if views is None:
        views = CAMERA_VIEWS

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda")
    glctx = _get_glctx()

    # ------------------------------------------------------------------
    # Prepare mesh data as torch tensors
    # ------------------------------------------------------------------
    mesh_batches = _prepare_render_batches(mesh, device)
    vertices = mesh_batches["bbox_vertices"]

    # Force-centre geometry for consistent framing.
    if vertices.numel() > 0:
        vmin = vertices.min(dim=0).values
        vmax = vertices.max(dim=0).values
        center = (vmin + vmax) * 0.5
        vertices = vertices - center
        mesh_batches["bbox_vertices"] = vertices
        for batch in mesh_batches["submeshes"]:
            batch["vertices"] = batch["vertices"] - center

    # Camera distance from bounding box
    distance = _compute_camera_distance(vertices)

    width, height = resolution
    aspect = width / height
    bg_rgb = torch.tensor(
        [c / 255.0 for c in background_color[:3]],
        dtype=torch.float32,
        device=device,
    )

    output_paths = []

    with torch.no_grad():
        for view_name, (azimuth, elevation) in views.items():
            # Build Model-View-Projection matrix
            mvp = _build_mvp(
                azimuth, elevation, distance, aspect, fov_deg=45.0
            ).to(device)

            color = bg_rgb.view(1, 1, 3).repeat(height, width, 1)
            alpha = torch.zeros(
                (height, width, 1), dtype=torch.float32, device=device
            )
            depth = torch.full(
                (height, width, 1),
                float("inf"),
                dtype=torch.float32,
                device=device,
            )

            for batch in mesh_batches["submeshes"]:
                verts_clip = _transform_vertices(
                    batch["vertices"], mvp, device
                )
                rast_out, _ = dr.rasterize(
                    glctx,
                    verts_clip,
                    batch["faces"],
                    resolution=[height, width],
                )

                if batch["has_texture"]:
                    uv_attr = batch["tex_coords"].unsqueeze(0).contiguous()
                    uv_interp, _ = dr.interpolate(
                        uv_attr, rast_out, batch["faces"]
                    )
                    tex = batch["texture_map"].unsqueeze(0)
                    albedo = dr.texture(tex, uv_interp, filter_mode="linear")
                elif batch["vertex_colors"] is not None:
                    vc_attr = batch["vertex_colors"].unsqueeze(0).contiguous()
                    albedo, _ = dr.interpolate(
                        vc_attr, rast_out, batch["faces"]
                    )
                else:
                    albedo = torch.full(
                        (1, height, width, 3),
                        0.7,
                        dtype=torch.float32,
                        device=device,
                    )

                n_attr = batch["normals"].unsqueeze(0).contiguous()
                normals_interp, _ = dr.interpolate(
                    n_attr, rast_out, batch["faces"]
                )
                normals_interp = torch.nn.functional.normalize(
                    normals_interp, dim=-1
                )

                shaded = _apply_lighting(albedo[0], normals_interp[0])
                shaded = dr.antialias(
                    shaded.unsqueeze(0).contiguous(),
                    rast_out,
                    verts_clip,
                    batch["faces"],
                )[0]

                mask = (rast_out[0, :, :, 3:4] > 0).float()
                sub_depth = rast_out[0, :, :, 2:3]
                update = (mask > 0) & (sub_depth < depth)
                color = torch.where(update.expand_as(color), shaded, color)
                alpha = torch.where(update, mask, alpha)
                depth = torch.where(update, sub_depth, depth)

            # ----------------------------------------------------------
            # Save image  (flip vertically: OpenGL row-0 = bottom)
            # ----------------------------------------------------------
            img_rgb = color.cpu().numpy()[::-1]
            img_rgb = np.clip(img_rgb, 0.0, 1.0)
            alpha_np = alpha.cpu().numpy()[::-1]
            alpha_np = np.clip(alpha_np, 0.0, 1.0)

            rgba = np.concatenate([img_rgb, alpha_np], axis=-1)
            img_np = (rgba * 255).astype(np.uint8)

            img = Image.fromarray(img_np, mode="RGBA")
            img_path = os.path.join(output_dir, f"{view_name}.png")
            img.save(img_path)
            output_paths.append(img_path)

            logger.info(f"Rendered view '{view_name}' -> {img_path}")

    return output_paths


# ======================================================================
# Mesh preparation
# ======================================================================


def _prepare_render_batches(
    mesh: trimesh.Trimesh | trimesh.Scene,
    device: torch.device,
) -> dict:
    """Convert a trimesh.Trimesh or trimesh.Scene to render batches."""
    if isinstance(mesh, trimesh.Scene):
        geometries = [
            geom
            for geom in mesh.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ]
    elif isinstance(mesh, trimesh.Trimesh):
        geometries = [mesh]
    else:
        raise ValueError(
            f"Unexpected mesh type: {type(mesh)}. Expected Trimesh or Scene."
        )

    if not geometries:
        raise ValueError("Scene contains no valid Trimesh geometry.")

    submeshes = [_prepare_single_mesh(geom, device) for geom in geometries]
    bbox_vertices = torch.cat(
        [submesh["vertices"] for submesh in submeshes], dim=0
    )
    return {
        "submeshes": submeshes,
        "bbox_vertices": bbox_vertices,
    }


def _prepare_single_mesh(
    mesh: trimesh.Trimesh,
    device: torch.device,
) -> dict:
    """Convert one trimesh into tensors needed by nvdiffrast."""
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
    normals = _compute_vertex_normals(vertices, faces)

    has_texture = False
    tex_coords = None
    texture_map = None
    vertex_colors = None
    visual = mesh.visual

    if isinstance(visual, trimesh.visual.TextureVisuals):
        uv = getattr(visual, "uv", None)
        mat_image = None
        if hasattr(visual, "material") and visual.material is not None:
            mat_image = getattr(visual.material, "image", None)

        if uv is not None and mat_image is not None:
            uv = uv.copy().astype(np.float32)
            uv[:, 1] = 1.0 - uv[:, 1]
            tex_coords = torch.tensor(uv, dtype=torch.float32, device=device)
            img_arr = (
                np.array(mat_image.convert("RGB")).astype(np.float32) / 255.0
            )
            texture_map = torch.tensor(
                img_arr, dtype=torch.float32, device=device
            )
            has_texture = True

    if not has_texture:
        if isinstance(visual, trimesh.visual.ColorVisuals):
            vc = visual.vertex_colors
            if vc is not None and len(vc) == len(mesh.vertices):
                vc_rgb = vc[:, :3].astype(np.float32) / 255.0
                vertex_colors = torch.tensor(
                    vc_rgb, dtype=torch.float32, device=device
                )
        elif isinstance(visual, trimesh.visual.TextureVisuals):
            try:
                colors = visual.to_color().vertex_colors
                if colors is not None:
                    vc_rgb = colors[:, :3].astype(np.float32) / 255.0
                    vertex_colors = torch.tensor(
                        vc_rgb, dtype=torch.float32, device=device
                    )
            except Exception:
                pass

    return {
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "tex_coords": tex_coords,
        "texture_map": texture_map,
        "vertex_colors": vertex_colors,
        "has_texture": has_texture,
    }


def _transform_vertices(
    vertices: torch.Tensor,
    mvp: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Transform vertices to clip space for rasterization."""
    ones = torch.ones(vertices.shape[0], 1, dtype=torch.float32, device=device)
    return (
        (mvp @ torch.cat([vertices, ones], dim=-1).T)
        .T.unsqueeze(0)
        .contiguous()
    )


def _compute_vertex_normals(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Compute area-weighted per-vertex normals.

    Args:
        vertices: ``(V, 3)`` vertex positions.
        faces: ``(F, 3)`` face indices (int32).

    Returns:
        ``(V, 3)`` unit-length per-vertex normals.
    """
    fi = faces.long()
    v0 = vertices[fi[:, 0]]
    v1 = vertices[fi[:, 1]]
    v2 = vertices[fi[:, 2]]

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

    normals = torch.zeros_like(vertices)
    normals.index_add_(0, fi[:, 0], face_normals)
    normals.index_add_(0, fi[:, 1], face_normals)
    normals.index_add_(0, fi[:, 2], face_normals)

    normals = torch.nn.functional.normalize(normals, dim=-1)
    return normals


# ======================================================================
# Camera / projection utilities
# ======================================================================


def _compute_camera_distance(
    vertices: torch.Tensor,
    fov_deg: float = 45.0,
) -> float:
    """Compute camera distance that fits the whole mesh in the viewport.

    Args:
        vertices: ``(V, 3)`` vertex positions.
        fov_deg: Camera vertical field of view in degrees.

    Returns:
        Camera distance from the mesh centre.
    """
    vmin = vertices.min(dim=0).values
    vmax = vertices.max(dim=0).values
    extent = (vmax - vmin).max().item()

    half_fov = math.radians(fov_deg / 2)
    distance = (extent / 2) / math.tan(half_fov)
    distance *= 1.5  # Padding
    return max(distance, 0.1)


def _build_mvp(
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    aspect: float,
    fov_deg: float = 45.0,
    near: float = 0.01,
    far: float = 100.0,
) -> torch.Tensor:
    """Build a Model-View-Projection matrix for a given viewpoint.

    Args:
        azimuth_deg: Horizontal rotation in degrees (0 = right / +Y).
        elevation_deg: Vertical rotation in degrees (0 = horizontal).
        distance: Distance from the origin.
        aspect: Width / height.
        fov_deg: Vertical FOV in degrees.
        near: Near clipping plane.
        far: Far clipping plane.

    Returns:
        ``(4, 4)`` MVP matrix (float32).
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    x = distance * math.cos(el) * (-math.sin(az))
    y = distance * math.cos(el) * (math.cos(az))
    z = distance * math.sin(el)

    eye = np.array([x, y, z], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    view = _look_at_matrix(eye, target, up)
    proj = _perspective_matrix(fov_deg, aspect, near, far)
    mvp = proj @ view

    return torch.tensor(mvp, dtype=torch.float32)


def _look_at_matrix(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """Compute a look-at view matrix (OpenGL convention: -Z forward).

    Args:
        eye: Camera position ``(3,)``.
        target: Look-at point ``(3,)``.
        up: World up vector ``(3,)``.

    Returns:
        ``(4, 4)`` view matrix.
    """
    forward = eye - target
    fwd_len = np.linalg.norm(forward)
    if fwd_len < 1e-8:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        forward = forward / fwd_len

    right = np.cross(up, forward)
    right_len = np.linalg.norm(right)
    if right_len < 1e-8:
        # Degenerate case — looking straight up or down.
        alt_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        right = np.cross(alt_up, forward)
        right_len = np.linalg.norm(right)
        if right_len < 1e-8:
            alt_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            right = np.cross(alt_up, forward)
            right_len = np.linalg.norm(right)
    right = right / right_len

    true_up = np.cross(forward, right)

    # Rotation part  (world → camera)
    R = np.eye(4, dtype=np.float32)
    R[0, :3] = right
    R[1, :3] = true_up
    R[2, :3] = forward

    # Translation part
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye

    return R @ T


def _perspective_matrix(
    fov_deg: float,
    aspect: float,
    near: float,
    far: float,
) -> np.ndarray:
    """Compute an OpenGL-style perspective projection matrix.

    Args:
        fov_deg: Vertical field of view in degrees.
        aspect: Width / height.
        near: Near clip distance.
        far: Far clip distance.

    Returns:
        ``(4, 4)`` projection matrix.
    """
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)

    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


# ======================================================================
# Lighting
# ======================================================================


def _apply_lighting(
    albedo: torch.Tensor,
    normals: torch.Tensor,
) -> torch.Tensor:
    """Apply bright, near-uniform Lambertian lighting.

    We intentionally use a symmetric six-direction light rig plus a
    stronger ambient term so all canonical views stay readable and no
    single face becomes noticeably darker than the others.

    Args:
        albedo: ``(H, W, 3)`` surface albedo / colour.
        normals: ``(H, W, 3)`` interpolated surface normals.

    Returns:
        ``(H, W, 3)`` lit colour image clamped to [0, 1].
    """
    device = albedo.device

    # Symmetric lighting from all cardinal directions keeps the rendered
    # preview bright and reduces view-dependent brightness imbalance.
    lights = [
        {"dir": [1.0, 0.0, 0.0], "intensity": 0.18},
        {"dir": [-1.0, 0.0, 0.0], "intensity": 0.18},
        {"dir": [0.0, 1.0, 0.0], "intensity": 0.18},
        {"dir": [0.0, -1.0, 0.0], "intensity": 0.18},
        {"dir": [0.0, 0.0, 1.0], "intensity": 0.22},
        {"dir": [0.0, 0.0, -1.0], "intensity": 0.12},
    ]
    ambient = 0.42

    color = albedo * ambient

    for light in lights:
        ld = torch.tensor(light["dir"], dtype=torch.float32, device=device)
        ld = ld / ld.norm()
        ndotl = (normals * ld.view(1, 1, 3)).sum(dim=-1, keepdim=True)
        ndotl = ndotl.clamp(min=0.0)
        color = color + albedo * ndotl * light["intensity"]

    return color.clamp(0.0, 1.0)


# ======================================================================
# Image grid utility  (display-only, no GPU dependency)
# ======================================================================


def combine_images_to_grid(
    images: list[str | Image.Image],
    grid_shape: tuple[int, int] = None,
    target_size: tuple[int, int] = (512, 512),
) -> Image.Image:
    """Combine multiple images into a single grid image.

    Args:
        images: List of image file paths or PIL Images.
        grid_shape: (rows, cols) for the grid. Auto-computed if None.
        target_size: Size to resize each image to (width, height).

    Returns:
        A single PIL Image containing all images in a grid.
    """
    n = len(images)
    if n == 0:
        raise ValueError("No images to combine.")

    if grid_shape is None:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    else:
        rows, cols = grid_shape

    loaded = []
    for img in images:
        if isinstance(img, str):
            loaded.append(Image.open(img).convert("RGB").resize(target_size))
        elif isinstance(img, Image.Image):
            loaded.append(img.convert("RGB").resize(target_size))
        else:
            raise ValueError(f"Unexpected image type: {type(img)}")

    w, h = target_size
    grid = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))

    for idx, img in enumerate(loaded):
        row, col = divmod(idx, cols)
        grid.paste(img, (col * w, row * h))

    return grid
