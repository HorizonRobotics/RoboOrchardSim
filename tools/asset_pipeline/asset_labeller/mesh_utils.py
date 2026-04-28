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

"""Mesh loading utilities with texture support for OBJ and USD formats."""

import logging
import os
import re
import shutil

import numpy as np
import trimesh
from PIL import Image

logger = logging.getLogger(__name__)


def load_mesh(file_path: str) -> trimesh.Trimesh:
    """Load a mesh file with texture support.

    Supports OBJ (with .mtl and texture images) and USD/USDA/USDC/USDZ
    (with referenced texture images).

    Args:
        file_path: Path to the mesh file.

    Returns:
        A trimesh.Trimesh object with visual/texture data attached.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mesh file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".obj", ".ply", ".glb", ".gltf"):
        return _load_trimesh(file_path)
    elif ext in (".usd", ".usda", ".usdc", ".usdz"):
        return _load_usd_mesh(file_path)
    else:
        raise ValueError(
            f"Unsupported mesh format: '{ext}'. "
            f"Supported: .obj, .ply, .glb, .gltf, .usd, .usda, .usdc, .usdz"
        )


def load_render_asset(file_path: str) -> trimesh.Trimesh | trimesh.Scene:
    """Load a renderable asset while preserving multi-material structure."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mesh file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".obj", ".ply", ".glb", ".gltf"):
        return _load_trimesh_render_asset(file_path)
    if ext in (".usd", ".usda", ".usdc", ".usdz"):
        return _load_usd_render_asset(file_path)

    raise ValueError(
        f"Unsupported mesh format: '{ext}'. "
        f"Supported: .obj, .ply, .glb, .gltf, .usd, .usda, .usdc, .usdz"
    )


def _load_trimesh(file_path: str) -> trimesh.Trimesh:
    """Load a mesh using trimesh with automatic texture/material loading.

    For OBJ files, trimesh will automatically parse the .mtl file and
    load referenced texture images from the same directory.

    Args:
        file_path: Path to the mesh file.

    Returns:
        A trimesh.Trimesh with visual data.
    """
    loaded = trimesh.load(file_path, process=True)

    # trimesh.load may return a Scene if the file has multiple meshes
    if isinstance(loaded, trimesh.Scene):
        mesh = _scene_to_single_mesh(loaded)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(
            f"Unexpected trimesh type: {type(loaded)}. "
            "Expected Trimesh or Scene."
        )

    logger.info(
        f"Loaded mesh from {file_path}: "
        f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces, "
        f"visual type: {type(mesh.visual).__name__}"
    )
    return mesh


def _load_trimesh_render_asset(
    file_path: str,
) -> trimesh.Trimesh | trimesh.Scene:
    """Load a mesh for rendering while preserving scene/material boundaries."""
    loaded = trimesh.load(file_path, process=False)
    if isinstance(loaded, (trimesh.Trimesh, trimesh.Scene)):
        return loaded
    raise ValueError(
        f"Unexpected trimesh type: {type(loaded)}. Expected Trimesh or Scene."
    )


def _scene_to_single_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    """Merge all meshes in a trimesh Scene into a single Trimesh.

    Attempts to preserve texture from the largest mesh if available.

    Args:
        scene: A trimesh.Scene containing one or more meshes.

    Returns:
        A single trimesh.Trimesh.
    """
    meshes = []
    for _name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            meshes.append(geom)

    if not meshes:
        raise ValueError("Scene contains no valid Trimesh geometry.")

    if len(meshes) == 1:
        return meshes[0]

    # Try to concatenate; texture may be lost for multi-mesh scenes
    combined = trimesh.util.concatenate(meshes)
    logger.warning(
        f"Scene contains {len(meshes)} meshes, merged into one. "
        "Texture may be from the first mesh only."
    )
    return combined


def _load_usd_mesh(usd_path: str) -> trimesh.Trimesh:
    """Load a mesh from a USD file with texture support.

    Extracts vertex positions, face indices, UV coordinates, and texture
    file paths from USD materials. Texture paths are resolved relative to
    the USD file's directory.

    Args:
        usd_path: Path to the USD file.

    Returns:
        A trimesh.Trimesh with TextureVisuals if texture is found.
    """
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        raise ImportError(
            "USD support requires the 'usd-core' package. "
            "Install it with: pip install usd-core"
        )

    usd_dir = os.path.dirname(os.path.abspath(usd_path))
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise ValueError(f"Failed to open USD stage: {usd_path}")

    # Convert authored units to meters so that exported OBJ matches
    try:
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    except Exception:
        meters_per_unit = 1.0
    if meters_per_unit <= 0:
        meters_per_unit = 1.0

    # Apply xform ops so meshes in scenes are placed correctly.
    xform_cache = UsdGeom.XformCache()

    # Collect all mesh prims
    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_vertex_uvs: list[np.ndarray] = []
    all_vertex_colors: list[np.ndarray] = []
    vertex_offset = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh_data = _extract_usd_mesh_data(
            prim, meters_per_unit=meters_per_unit, xform_cache=xform_cache
        )
        if mesh_data is None:
            continue
        vertices = mesh_data["vertices"]
        triangles = mesh_data["triangles"]
        triangle_uvs = mesh_data["triangle_uvs"]
        triangle_materials = mesh_data["triangle_materials"]

        material_records = {
            key: _build_material_record(
                stage=stage,
                usd_dir=usd_dir,
                material_path=key,
                textures_dir=os.path.join(
                    usd_dir, "__unused_render_textures__"
                ),
                copied_textures=[],
                copy_textures=False,
            )
            for key in _stable_unique(triangle_materials)
        }

        has_multi_material = len(material_records) > 1
        has_any_texture = any(
            record.get("texture_image") is not None
            for record in material_records.values()
        )

        if has_multi_material and has_any_texture:
            flat_vidx = triangles.reshape(-1)
            expanded_vertices = vertices[flat_vidx]
            expanded_faces = (
                np.arange(expanded_vertices.shape[0], dtype=np.int32).reshape(
                    -1, 3
                )
                + vertex_offset
            )
            expanded_colors = []
            for tri_idx, material_key in enumerate(triangle_materials):
                record = material_records[material_key]
                for uv in triangle_uvs[tri_idx]:
                    expanded_colors.append(_sample_material_color(record, uv))

            all_vertices.append(expanded_vertices)
            all_faces.append(expanded_faces)
            all_vertex_colors.append(
                np.asarray(expanded_colors, dtype=np.uint8)
            )
            vertex_offset += expanded_vertices.shape[0]
        else:
            tri_faces = triangles + vertex_offset
            all_vertices.append(vertices)
            all_faces.append(tri_faces)

            if len(triangle_uvs) > 0:
                vertex_uvs = np.zeros((vertices.shape[0], 2), dtype=np.float64)
                flat_vidx = triangles.reshape(-1)
                flat_uvs = triangle_uvs.reshape(-1, 2)
                vertex_uvs[flat_vidx] = flat_uvs
                all_vertex_uvs.append(vertex_uvs)

            vertex_offset += vertices.shape[0]

    if not all_vertices:
        raise ValueError(f"No mesh geometry found in USD file: {usd_path}")

    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if all_vertex_colors:
        vertex_colors = np.vstack(all_vertex_colors)
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=vertex_colors,
        )
    else:
        texture_image = _extract_usd_texture(stage, usd_dir)
        # Attach texture if available
        if texture_image is not None and all_vertex_uvs:
            vertex_uvs = np.vstack(all_vertex_uvs)
            if len(vertex_uvs) == len(mesh.vertices):
                from trimesh.visual.texture import TextureVisuals

                material = trimesh.visual.material.SimpleMaterial(
                    image=texture_image
                )
                mesh.visual = TextureVisuals(
                    uv=vertex_uvs.astype(np.float64), material=material
                )
            else:
                logger.warning(
                    "USD texture found but UV/vertex count mismatch "
                    "after conversion: "
                    f"uvs={len(vertex_uvs)} verts={len(mesh.vertices)}"
                )

    if not all_vertex_colors and texture_image is not None and all_vertex_uvs:
        vertex_uvs = np.vstack(all_vertex_uvs)
        logger.info(
            f"Loaded USD texture visuals from {usd_path}: "
            f"UV count={len(vertex_uvs)}"
        )

    logger.info(
        f"Loaded USD mesh from {usd_path}: "
        f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
    )
    return mesh


def _load_usd_render_asset(usd_path: str) -> trimesh.Trimesh | trimesh.Scene:
    """Load a USD asset as a renderable scene with per-material submeshes."""
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        raise ImportError(
            "USD support requires the 'usd-core' package. "
            "Install it with: pip install usd-core"
        )

    usd_dir = os.path.dirname(os.path.abspath(usd_path))
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise ValueError(f"Failed to open USD stage: {usd_path}")

    try:
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    except Exception:
        meters_per_unit = 1.0
    if meters_per_unit <= 0:
        meters_per_unit = 1.0

    xform_cache = UsdGeom.XformCache()
    scene = trimesh.Scene()
    geom_count = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh_data = _extract_usd_mesh_data(
            prim, meters_per_unit=meters_per_unit, xform_cache=xform_cache
        )
        if mesh_data is None:
            continue

        vertices = mesh_data["vertices"]
        triangles = mesh_data["triangles"]
        triangle_uvs = mesh_data["triangle_uvs"]
        triangle_materials = mesh_data["triangle_materials"]

        for material_key in _stable_unique(triangle_materials):
            record = _build_material_record(
                stage=stage,
                usd_dir=usd_dir,
                material_path=material_key,
                textures_dir=os.path.join(
                    usd_dir, "__unused_render_textures__"
                ),
                copied_textures=[],
                copy_textures=False,
            )
            tri_ids = np.where(triangle_materials == material_key)[0]
            if len(tri_ids) == 0:
                continue

            render_mesh = _build_render_submesh(
                vertices=vertices,
                triangles=triangles[tri_ids],
                triangle_uvs=triangle_uvs[tri_ids],
                material_record=record,
            )
            geom_name = (
                f"{_sanitize_mtl_name(str(prim.GetPath()))}__{record['name']}"
            )
            scene.add_geometry(render_mesh, geom_name=geom_name)
            geom_count += 1

    if geom_count == 0:
        raise ValueError(f"No mesh geometry found in USD file: {usd_path}")

    if geom_count == 1:
        return next(iter(scene.geometry.values()))
    return scene


def export_usd_to_obj_with_materials(
    usd_path: str,
    obj_output_path: str,
) -> list[str]:
    """Export a USD mesh to OBJ/MTL while preserving per-part materials.

    The generated OBJ keeps one ``usemtl`` section per USD material binding,
    and referenced base-color textures are copied into the same
    directory as the OBJ file.

    Args:
        usd_path: Path to the input USD/USD[A|C|Z] file.
        obj_output_path: Target OBJ path.

    Returns:
        List of copied texture file paths.
    """
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        raise ImportError(
            "USD export requires the 'usd-core' package. "
            "Install it with: pip install usd-core"
        )

    usd_dir = os.path.dirname(os.path.abspath(usd_path))
    obj_output_path = os.path.abspath(obj_output_path)
    obj_dir = os.path.dirname(obj_output_path)
    os.makedirs(obj_dir, exist_ok=True)

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise ValueError(f"Failed to open USD stage: {usd_path}")

    try:
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    except Exception:
        meters_per_unit = 1.0
    if meters_per_unit <= 0:
        meters_per_unit = 1.0

    xform_cache = UsdGeom.XformCache()
    mtl_name = os.path.splitext(os.path.basename(obj_output_path))[0] + ".mtl"
    mtl_path = os.path.join(obj_dir, mtl_name)
    textures_dir = obj_dir

    material_map: dict[str, dict] = {}
    material_order: list[str] = []
    copied_textures: list[str] = []
    obj_lines = [f"mtllib {mtl_name}"]
    vertex_index = 1
    uv_index = 1

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh_data = _extract_usd_mesh_data(
            prim, meters_per_unit=meters_per_unit, xform_cache=xform_cache
        )
        if mesh_data is None:
            continue

        vertices = mesh_data["vertices"]
        triangles = mesh_data["triangles"]
        triangle_uvs = mesh_data["triangle_uvs"]
        triangle_materials = mesh_data["triangle_materials"]

        if len(vertices) == 0 or len(triangles) == 0:
            continue

        obj_lines.append(f"o {_sanitize_mtl_name(str(prim.GetPath()))}")
        for vertex in vertices:
            obj_lines.append(
                f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}"
            )

        for material_name in _stable_unique(triangle_materials):
            if material_name not in material_map:
                material_map[material_name] = _build_material_record(
                    stage=stage,
                    usd_dir=usd_dir,
                    material_path=material_name,
                    textures_dir=textures_dir,
                    copied_textures=copied_textures,
                )
                material_order.append(material_name)

            obj_lines.append(f"usemtl {material_map[material_name]['name']}")
            material_triangles = np.where(triangle_materials == material_name)[
                0
            ]
            for tri_idx in material_triangles:
                tri = triangles[tri_idx]
                tri_uv = triangle_uvs[tri_idx]
                face_tokens = []
                for corner in range(3):
                    uv = tri_uv[corner]
                    obj_lines.append(f"vt {uv[0]:.8f} {uv[1]:.8f}")
                    face_tokens.append(
                        f"{int(tri[corner]) + vertex_index}/{uv_index}"
                    )
                    uv_index += 1
                obj_lines.append(f"f {' '.join(face_tokens)}")

        vertex_index += len(vertices)

    if vertex_index == 1:
        raise ValueError(f"No mesh geometry found in USD file: {usd_path}")

    with open(obj_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(obj_lines) + "\n")

    _write_mtl_file(mtl_path, material_order, material_map)
    return copied_textures


def _triangulate_faces(
    face_counts: np.ndarray, face_indices: np.ndarray
) -> np.ndarray:
    """Triangulate polygon faces into triangle faces.

    Args:
        face_counts: Number of vertices per face.
        face_indices: Flattened vertex indices for all faces.

    Returns:
        Array of triangle face indices, shape (N, 3).
    """
    triangles = []
    idx = 0
    for count in face_counts:
        if count == 3:
            triangles.append(face_indices[idx : idx + 3])
        elif count == 4:
            # Quad -> 2 triangles
            triangles.append(face_indices[[idx, idx + 1, idx + 2]])
            triangles.append(face_indices[[idx, idx + 2, idx + 3]])
        else:
            # Fan triangulation for n-gons
            for i in range(1, count - 1):
                triangles.append(face_indices[[idx, idx + i, idx + i + 1]])
        idx += count

    return np.array(triangles, dtype=np.int32)


def _extract_usd_mesh_data(prim, meters_per_unit: float, xform_cache):
    """Extract triangulated mesh data and per-triangle material bindings."""
    from pxr import UsdGeom

    mesh_prim = UsdGeom.Mesh(prim)
    points = mesh_prim.GetPointsAttr().Get()
    face_counts = mesh_prim.GetFaceVertexCountsAttr().Get()
    face_indices = mesh_prim.GetFaceVertexIndicesAttr().Get()

    if points is None or face_counts is None or face_indices is None:
        return None

    vertices = np.array(points, dtype=np.float64)
    try:
        xf = xform_cache.GetLocalToWorldTransform(prim)
        xf_np = np.array(xf, dtype=np.float64)
        ones = np.ones((vertices.shape[0], 1), dtype=np.float64)
        vertices = (np.concatenate([vertices, ones], axis=1) @ xf_np.T)[:, :3]
    except Exception:
        pass

    if meters_per_unit != 1.0:
        vertices = vertices * meters_per_unit

    face_counts = np.array(face_counts, dtype=np.int32)
    face_indices = np.array(face_indices, dtype=np.int32)
    uvs, uv_indices, uv_interp = _extract_raw_uvs(mesh_prim)
    face_materials = _extract_face_material_bindings(prim, len(face_counts))

    triangles = []
    triangle_uvs = []
    triangle_materials = []

    offset = 0
    for face_idx, count in enumerate(face_counts):
        polygon_vertices = face_indices[offset : offset + count]
        polygon_uvs = _polygon_uv_indices(
            uv_indices=uv_indices,
            interpolation=uv_interp,
            polygon_vertices=polygon_vertices,
            face_index=face_idx,
            offset=offset,
            count=count,
        )
        mat_name = face_materials[face_idx]

        for corner in range(1, count - 1):
            triangles.append(
                [
                    polygon_vertices[0],
                    polygon_vertices[corner],
                    polygon_vertices[corner + 1],
                ]
            )
            triangle_uvs.append(
                [
                    _lookup_uv(uvs, polygon_uvs[0]),
                    _lookup_uv(uvs, polygon_uvs[corner]),
                    _lookup_uv(uvs, polygon_uvs[corner + 1]),
                ]
            )
            triangle_materials.append(mat_name)
        offset += count

    if not triangles:
        return None

    return {
        "vertices": vertices,
        "triangles": np.asarray(triangles, dtype=np.int32),
        "triangle_uvs": np.asarray(triangle_uvs, dtype=np.float64),
        "triangle_materials": np.asarray(triangle_materials, dtype=object),
    }


def _extract_raw_uvs(mesh_prim):
    """Extract raw UV values and indices from a USD mesh prim."""
    from pxr import UsdGeom

    primvar_api = UsdGeom.PrimvarsAPI(mesh_prim)
    uv_primvar = None
    for name in ["st", "UVMap", "uv", "texCoord"]:
        pv = primvar_api.GetPrimvar(name)
        if pv and pv.HasValue():
            uv_primvar = pv
            break

    if uv_primvar is None:
        return None, None, "none"

    uvs = np.array(uv_primvar.Get(), dtype=np.float64)
    indices = uv_primvar.GetIndices()
    interp = str(uv_primvar.GetInterpolation() or "none")

    if indices is not None and len(indices) > 0:
        uv_indices = np.array(indices, dtype=np.int32)
    else:
        uv_indices = np.arange(len(uvs), dtype=np.int32)

    return uvs, uv_indices, interp


def _extract_uvs(mesh_prim, face_counts):
    """Extract UV coordinates from a USD mesh prim.

    Args:
        mesh_prim: A UsdGeom.Mesh prim.
        face_counts: Number of vertices per face.

    Returns:
        Tuple of (uvs, face_uvs, interpolation) where interpolation is
        "vertex" or e.g. "faceVarying". Returns (None, None, "none") if
        not found.
    """
    uvs, uv_indices, interp = _extract_raw_uvs(mesh_prim)
    if uvs is None:
        return None, None, "none"

    # Triangulate UV indices same as face indices
    face_counts = np.array(face_counts, dtype=np.int32)
    face_uvs = _triangulate_faces(face_counts, uv_indices)

    return uvs, face_uvs, interp


def _extract_face_material_bindings(prim, face_count: int) -> list[str]:
    """Return one material path string per polygon face."""
    from pxr import UsdGeom

    default_binding = _get_bound_material_path(prim)
    face_materials = [default_binding or "__default__"] * face_count

    for child in prim.GetChildren():
        if not child.IsA(UsdGeom.Subset):
            continue
        material_path = _get_bound_material_path(child)
        if material_path is None:
            continue
        indices = child.GetAttribute("indices").Get()
        if indices is None:
            continue
        for face_idx in indices:
            if 0 <= int(face_idx) < face_count:
                face_materials[int(face_idx)] = material_path

    return face_materials


def _get_bound_material_path(prim) -> str | None:
    current = prim
    while current and current.IsValid():
        rel = current.GetRelationship("material:binding")
        if rel is not None:
            targets = rel.GetTargets()
            if targets:
                return str(targets[0])
        current = current.GetParent()
    return None


def _polygon_uv_indices(
    uv_indices,
    interpolation: str,
    polygon_vertices: np.ndarray,
    face_index: int,
    offset: int,
    count: int,
) -> np.ndarray:
    """Resolve UV indices for one polygon face."""
    if uv_indices is None:
        return np.full(count, -1, dtype=np.int32)

    if interpolation == "vertex":
        return polygon_vertices.astype(np.int32)

    if interpolation in ("faceVarying", "varying"):
        return uv_indices[offset : offset + count].astype(np.int32)

    if interpolation == "uniform":
        if face_index < len(uv_indices):
            return np.full(count, int(uv_indices[face_index]), dtype=np.int32)
        return np.full(count, -1, dtype=np.int32)

    if interpolation == "constant":
        if len(uv_indices) > 0:
            return np.full(count, int(uv_indices[0]), dtype=np.int32)
        return np.full(count, -1, dtype=np.int32)

    return polygon_vertices.astype(np.int32)


def _lookup_uv(uvs: np.ndarray | None, uv_idx: int) -> np.ndarray:
    if uvs is None or uv_idx < 0 or uv_idx >= len(uvs):
        return np.array([0.0, 0.0], dtype=np.float64)
    return np.asarray(uvs[uv_idx], dtype=np.float64)


def _extract_usd_texture(stage, usd_dir: str):
    """Extract the first texture image from USD materials.

    Looks for UsdShade materials and extracts the diffuse/albedo
    texture file path, resolving it relative to the USD file directory.

    Args:
        stage: An opened USD stage.
        usd_dir: Directory of the USD file for resolving relative paths.

    Returns:
        A PIL.Image or None if no texture is found.
    """
    from pxr import UsdShade

    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue

        material = UsdShade.Material(prim)

        # Look for surface shader
        surface_output = material.GetSurfaceOutput()
        if not surface_output:
            continue

        # Traverse connected shaders to find texture
        for connection in surface_output.GetConnectedSources():
            if not connection:
                continue
            source_prim = connection[0].source.GetPrim()
            shader = UsdShade.Shader(source_prim)
            if not shader:
                continue

            texture_path = _find_texture_in_shader(shader, stage, usd_dir)
            if texture_path is not None:
                return texture_path

    # Fallback: search for any shader with texture inputs
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        texture_path = _find_texture_in_shader(shader, stage, usd_dir)
        if texture_path is not None:
            return texture_path

    return None


def _build_material_record(
    stage,
    usd_dir: str,
    material_path: str,
    textures_dir: str,
    copied_textures: list[str],
    copy_textures: bool = True,
) -> dict:
    """Create an OBJ/MTL material record from a USD material binding."""
    record = {
        "name": _sanitize_mtl_name(material_path),
        "texture_relpath": None,
        "diffuse_color": (0.8, 0.8, 0.8),
        "texture_image": None,
    }

    if material_path == "__default__":
        return record

    prim = stage.GetPrimAtPath(material_path)
    if not prim or not prim.IsValid():
        return record

    texture_path, diffuse_color = _extract_usd_material_inputs(prim, usd_dir)
    if diffuse_color is not None:
        record["diffuse_color"] = diffuse_color

    if texture_path is not None:
        try:
            record["texture_image"] = Image.open(texture_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load texture {texture_path}: {e}")
            record["texture_image"] = None

        if copy_textures:
            dst = os.path.join(textures_dir, os.path.basename(texture_path))
            if not os.path.exists(dst):
                shutil.copy2(texture_path, dst)
                copied_textures.append(dst)
            record["texture_relpath"] = os.path.basename(texture_path)

    return record


def _sample_material_color(record: dict, uv: np.ndarray) -> np.ndarray:
    """Sample one RGB color from a material record at the given UV."""
    image = record.get("texture_image")
    if image is None:
        diffuse = np.asarray(record["diffuse_color"], dtype=np.float64)
        rgb = np.clip(np.round(diffuse * 255.0), 0, 255).astype(np.uint8)
        return rgb

    width, height = image.size
    u = float(np.clip(uv[0], 0.0, 1.0))
    v = float(np.clip(uv[1], 0.0, 1.0))
    x = min(int(round(u * (width - 1))), width - 1)
    y = min(int(round((1.0 - v) * (height - 1))), height - 1)
    return np.asarray(image.getpixel((x, y))[:3], dtype=np.uint8)


def _build_render_submesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_uvs: np.ndarray,
    material_record: dict,
) -> trimesh.Trimesh:
    """Build a renderable trimesh submesh for one material group."""
    flat_vidx = triangles.reshape(-1)
    expanded_vertices = vertices[flat_vidx]
    expanded_faces = np.arange(
        expanded_vertices.shape[0], dtype=np.int32
    ).reshape(-1, 3)
    mesh = trimesh.Trimesh(
        vertices=expanded_vertices,
        faces=expanded_faces,
        process=False,
    )

    texture_image = material_record.get("texture_image")
    if texture_image is not None:
        from trimesh.visual.texture import TextureVisuals

        mesh.visual = TextureVisuals(
            uv=triangle_uvs.reshape(-1, 2).astype(np.float64),
            material=trimesh.visual.material.SimpleMaterial(
                image=texture_image
            ),
        )
    else:
        diffuse = np.asarray(
            material_record["diffuse_color"], dtype=np.float64
        )
        rgb = np.clip(np.round(diffuse * 255.0), 0, 255).astype(np.uint8)
        rgba = np.tile(np.append(rgb, 255), (len(mesh.vertices), 1))
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=rgba,
        )

    return mesh


def _extract_usd_material_inputs(material_prim, usd_dir: str):
    """Extract base-color texture and fallback diffuse color from material."""
    try:
        from pxr import UsdShade
    except ImportError:
        return None, None

    texture_path = None
    diffuse_color = None

    for child in material_prim.GetChildren():
        shader = UsdShade.Shader(child)
        if not shader:
            continue

        if texture_path is None:
            texture_path = _find_texture_path_in_shader(shader, usd_dir)
        if diffuse_color is None:
            diffuse_color = _find_diffuse_color_in_shader(shader)

        if texture_path is not None and diffuse_color is not None:
            break

    return texture_path, diffuse_color


def _find_texture_path_in_shader(shader, usd_dir: str) -> str | None:
    """Find a base-color texture file path referenced by a USD shader."""
    try:
        from pxr import UsdShade
    except ImportError:
        return None

    shader_id = shader.GetIdAttr().Get()
    if shader_id and "UsdUVTexture" in str(shader_id):
        file_input = shader.GetInput("file")
        if file_input and file_input.GetAttr().HasValue():
            asset_path = file_input.Get()
            if asset_path:
                return _resolve_texture_path(str(asset_path), usd_dir)

    for input_name in [
        "diffuse_texture",
        "baseColorTexture",
        "diffuseColor",
        "albedo",
        "baseColor",
        "color",
        "file",
    ]:
        inp = shader.GetInput(input_name)
        if not inp:
            continue

        sources = inp.GetConnectedSources()
        if sources:
            for conn_info in sources:
                if not conn_info:
                    continue
                connected_prim = conn_info[0].source.GetPrim()
                connected_shader = UsdShade.Shader(connected_prim)
                if connected_shader:
                    result = _find_texture_path_in_shader(
                        connected_shader, usd_dir
                    )
                    if result is not None:
                        return result

        if inp.GetAttr().HasValue():
            val = inp.Get()
            if hasattr(val, "resolvedPath") or isinstance(val, str):
                resolved = _resolve_texture_path(str(val), usd_dir)
                if resolved is not None:
                    return resolved

    return None


def _find_diffuse_color_in_shader(shader) -> tuple[float, float, float] | None:
    for input_name in ["diffuse_color_constant", "diffuseColor", "base_color"]:
        inp = shader.GetInput(input_name)
        if not inp or not inp.GetAttr().HasValue():
            continue
        value = inp.Get()
        if value is None:
            continue
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if arr.size >= 3:
            return tuple(float(x) for x in arr[:3])
    return None


def _find_texture_in_shader(shader, stage, usd_dir: str):
    """Find a texture file referenced by a USD shader.

    Checks common input names for texture file references.

    Args:
        shader: A UsdShade.Shader prim.
        stage: The USD stage.
        usd_dir: Base directory for resolving relative paths.

    Returns:
        A PIL.Image or None.
    """
    from pxr import UsdShade

    # Check if this shader is a texture reader
    shader_id = shader.GetIdAttr().Get()
    if shader_id and "UsdUVTexture" in str(shader_id):
        file_input = shader.GetInput("file")
        if file_input and file_input.GetAttr().HasValue():
            asset_path = file_input.Get()
            if asset_path:
                resolved = _resolve_texture_path(str(asset_path), usd_dir)
                if resolved:
                    try:
                        return Image.open(resolved).convert("RGB")
                    except Exception as e:
                        logger.warning(
                            f"Failed to load texture {resolved}: {e}"
                        )
        return None

    # Check connected shaders for texture inputs
    texture_input_names = [
        "diffuseColor",
        "albedo",
        "baseColor",
        "color",
        "diffuse_texture",
        "file",
    ]
    for input_name in texture_input_names:
        inp = shader.GetInput(input_name)
        if not inp:
            continue

        # Check for connected texture reader
        sources = inp.GetConnectedSources()
        if sources:
            for conn_info in sources:
                if not conn_info:
                    continue
                connected_prim = conn_info[0].source.GetPrim()
                connected_shader = UsdShade.Shader(connected_prim)
                if connected_shader:
                    result = _find_texture_in_shader(
                        connected_shader, stage, usd_dir
                    )
                    if result is not None:
                        return result

        # Check for direct asset path value
        if inp.GetAttr().HasValue():
            val = inp.Get()
            if hasattr(val, "resolvedPath") or isinstance(val, str):
                resolved = _resolve_texture_path(str(val), usd_dir)
                if resolved:
                    try:
                        return Image.open(resolved).convert("RGB")
                    except Exception as e:
                        logger.warning(
                            f"Failed to load texture {resolved}: {e}"
                        )

    return None


def _sanitize_mtl_name(name: str) -> str:
    safe = []
    for char in str(name):
        if char.isalnum() or char in ("_", "-", "."):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "material"


def _stable_unique(items) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _write_mtl_file(
    mtl_path: str, material_order: list[str], material_map: dict[str, dict]
) -> None:
    lines = []
    for material_key in material_order:
        record = material_map[material_key]
        kd = record["diffuse_color"]
        lines.append(f"newmtl {record['name']}")
        lines.append(f"Kd {kd[0]:.6f} {kd[1]:.6f} {kd[2]:.6f}")
        lines.append("Ka 0.000000 0.000000 0.000000")
        lines.append("Ks 0.000000 0.000000 0.000000")
        lines.append("d 1.000000")
        if record["texture_relpath"] is not None:
            lines.append(f"map_Kd {record['texture_relpath']}")
        lines.append("")

    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _resolve_texture_path(raw_path: str, base_dir: str) -> str | None:
    """Resolve a texture file path from USD to an absolute path.

    Handles paths like '@./textures/albedo.png@' and relative paths.

    Args:
        raw_path: Raw texture path string from USD.
        base_dir: Base directory for relative path resolution.

    Returns:
        Absolute path to the texture file, or None if not found.
    """
    # Strip USD asset path decorators
    path = raw_path.strip().strip("@").strip("'").strip('"')

    if not path:
        return None

    # Try absolute path first
    if os.path.isabs(path) and os.path.isfile(path):
        return path

    # Resolve relative to USD file directory
    abs_path = os.path.normpath(os.path.join(base_dir, path))
    if os.path.isfile(abs_path):
        return abs_path

    # Try stripping leading ./ or ./
    stripped = path.lstrip("./")
    abs_path = os.path.normpath(os.path.join(base_dir, stripped))
    if os.path.isfile(abs_path):
        return abs_path

    logger.warning(
        f"Texture file not found: {raw_path} (resolved: {abs_path})"
    )
    return None


def _apply_texture_to_mesh(
    mesh: trimesh.Trimesh,
    uvs: np.ndarray,
    face_uvs: np.ndarray,
    texture_image: Image.Image,
) -> trimesh.Trimesh:
    """Apply a texture image to a trimesh using UV coordinates.

    Args:
        mesh: The target trimesh.
        uvs: UV coordinate array, shape (N, 2).
        face_uvs: Face UV index array, shape (F, 3).
        texture_image: PIL Image to use as texture.

    Returns:
        The mesh with TextureVisuals applied.
    """
    from trimesh.visual.texture import TextureVisuals

    try:
        # Build per-vertex UVs from face UVs
        # If face_uvs indices map into uvs, construct per-vertex UV
        vertex_uvs = np.zeros((len(mesh.vertices), 2), dtype=np.float64)
        for fi in range(len(mesh.faces)):
            for vi in range(3):
                vertex_idx = mesh.faces[fi, vi]
                uv_idx = face_uvs[fi, vi]
                if uv_idx < len(uvs):
                    vertex_uvs[vertex_idx] = uvs[uv_idx]

        material = trimesh.visual.material.SimpleMaterial(image=texture_image)
        mesh.visual = TextureVisuals(uv=vertex_uvs, material=material)
    except Exception as e:
        logger.warning(f"Failed to apply texture to mesh: {e}")

    return mesh


def normalize_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, float]:
    """Normalize mesh to fit within [-0.5, 0.5] range.

    Args:
        mesh: Input mesh.

    Returns:
        Tuple of (normalized_mesh, scale_factor).
    """
    scale = np.ptp(mesh.vertices, axis=0).max()
    if scale > 0:
        mesh.vertices /= scale
    return mesh, scale


def scale_mesh_to_height(
    mesh: trimesh.Trimesh, target_height: float
) -> tuple[trimesh.Trimesh, float]:
    """Scale a normalized mesh to a target real-world height.

    The height is measured along the Y-axis (vertical).

    Args:
        mesh: A normalized mesh (should be in [-0.5, 0.5] range).
        target_height: Target height in meters.

    Returns:
        Tuple of (scaled_mesh, applied_scale).
    """
    raw_height = np.ptp(mesh.vertices, axis=0)[1]  # Y-axis height
    if raw_height <= 0:
        logger.warning("Mesh has zero height along Y-axis, using unit scale.")
        return mesh, 1.0

    scale = round(target_height / raw_height, 6)
    mesh = mesh.apply_scale(scale)
    return mesh, scale


# ---------------------------------------------------------------------
# Texture audit
# ---------------------------------------------------------------------

# Issue codes returned from audit_export_textures().
TEXTURE_AUDIT_USD_HAS_TEXTURES_BUT_NO_MAP_KD = "USD_HAS_TEXTURES_BUT_NO_MAP_KD"
TEXTURE_AUDIT_MAP_KD_FILE_MISSING = "MAP_KD_FILE_MISSING"
TEXTURE_AUDIT_PARTIAL_TEXTURE_BINDING = "PARTIAL_TEXTURE_BINDING"

_IMG_EXTS_TUP = (".png", ".jpg", ".jpeg")
_MTL_MAP_KD_RE = re.compile(r"^map_Kd\s+(\S+)", re.MULTILINE)


def _list_source_texture_candidates(usd_path: str) -> list[str]:
    """Return absolute paths of png/jpg referenced by USD shader inputs.

    Walks every shader prim inside any Material prim of ``usd_path`` and
    collects the resolved absolute paths of asset-typed inputs that point
    to existing png/jpg files. Used by :func:`audit_export_textures` to
    decide whether the source USD declares textures that the export
    pipeline ought to surface.
    """
    try:
        from pxr import Usd, UsdShade
    except ImportError:
        return []

    if not os.path.isfile(usd_path):
        return []

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        return []

    usd_dir = os.path.dirname(os.path.abspath(usd_path))
    candidates: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        # Only count shaders inside Material prims (excludes one-off
        # Shader prims used for, e.g., light or render settings).
        anc = prim.GetParent()
        in_material = False
        while anc and anc.IsValid():
            if anc.IsA(UsdShade.Material):
                in_material = True
                break
            anc = anc.GetParent()
        if not in_material:
            continue

        for attr in prim.GetAttributes():
            if str(attr.GetTypeName()) != "asset":
                continue
            if not attr.HasValue():
                continue
            val = attr.Get()
            if val is None:
                continue
            raw = getattr(val, "path", "") or str(val)
            if not raw.lower().endswith(_IMG_EXTS_TUP):
                continue
            resolved = getattr(
                val, "resolvedPath", ""
            ) or _resolve_texture_path(raw, usd_dir)
            if (
                resolved
                and os.path.isfile(resolved)
                and resolved not in candidates
            ):
                candidates.append(resolved)
    return candidates


def audit_export_textures(
    usd_path: str,
    obj_path: str,
    copied_textures: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Validate that ``export_usd_to_obj_with_materials`` preserved textures.

    Checked invariants (issue codes returned on violation):

    * ``USD_HAS_TEXTURES_BUT_NO_MAP_KD`` — the source USD declares one
      or more asset-typed shader inputs pointing to png/jpg files that
      exist on disk, yet the produced ``.mtl`` has no ``map_Kd`` line.
      This is the regression that left ``pear_002`` and ``candy_001``
      untextured in past pipeline runs.

    * ``MAP_KD_FILE_MISSING`` — the produced ``.mtl`` references a file
      via ``map_Kd`` that does not exist relative to the OBJ directory.

    * ``PARTIAL_TEXTURE_BINDING`` — source declared multiple texture
      candidates (e.g. base-color + normal + roughness PBR) but ``.mtl``
      only got a strict subset. Informational only; OBJ/MTL has no
      native slot for normal or roughness maps so this is often expected
      when the source uses a PBR workflow.

    Args:
        usd_path: Path to the source USD file passed to export.
        obj_path: Path to the produced ``.obj`` (its sibling ``.mtl`` is
            inspected and ``os.path.dirname(obj_path)`` is used as the
            search root for ``map_Kd`` references).
        copied_textures: Optional list of texture file paths returned by
            ``export_usd_to_obj_with_materials``. When supplied, used to
            assess ``PARTIAL_TEXTURE_BINDING``.

    Returns:
        List of ``(code, detail)`` tuples. Empty list = audit clean.
    """
    issues: list[tuple[str, str]] = []
    obj_dir = os.path.dirname(os.path.abspath(obj_path))
    mtl_path = obj_path[:-4] + ".mtl"

    source_candidates = _list_source_texture_candidates(usd_path)

    mtl_refs: list[str] = []
    if os.path.isfile(mtl_path):
        try:
            text = open(mtl_path, encoding="utf-8").read()
            mtl_refs = _MTL_MAP_KD_RE.findall(text)
        except Exception:
            pass

    if source_candidates and not mtl_refs:
        issues.append(
            (
                TEXTURE_AUDIT_USD_HAS_TEXTURES_BUT_NO_MAP_KD,
                f"source declares {len(source_candidates)} texture "
                f"candidate(s) "
                f"({[os.path.basename(p) for p in source_candidates[:3]]}...) "
                f"but produced .mtl has no map_Kd line",
            )
        )

    for ref in mtl_refs:
        target = os.path.normpath(os.path.join(obj_dir, ref))
        if not os.path.isfile(target):
            issues.append(
                (
                    TEXTURE_AUDIT_MAP_KD_FILE_MISSING,
                    f"map_Kd {ref!r} not found at {target}",
                )
            )

    if copied_textures and source_candidates:
        if len(copied_textures) < len(source_candidates) and len(mtl_refs) > 0:
            issues.append(
                (
                    TEXTURE_AUDIT_PARTIAL_TEXTURE_BINDING,
                    f"source had {len(source_candidates)} texture "
                    f"candidate(s), export copied {len(copied_textures)}; "
                    f"normal/roughness/etc. maps may have been dropped "
                    f"(expected: OBJ has no native slot for non-diffuse maps)",
                )
            )

    return issues
