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

"""Asset labelling tool with physical and semantic attributes.

Uses GPT vision models to estimate object properties from rendered
views, and generates a URDF file with mesh, physics, and metadata.
"""

import logging
import os
import re
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from difflib import get_close_matches
from xml.dom.minidom import parseString

import numpy as np
from scipy.spatial.transform import Rotation

from asset_labeller.convex_decomposer import decompose_convex_mesh
from asset_labeller.gpt_client import GPTClient
from asset_labeller.mesh_utils import (
    audit_export_textures,
    export_usd_to_obj_with_materials,
    load_mesh,
    load_render_asset,
)
from asset_labeller.renderer import render_views, select_views

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "v0.1.0"

__all__ = ["AssetLabeller"]


SHAPE_CHOICES = [
    "sphere",
    "hemisphere",
    "ellipsoid",
    "cylinder",
    "cone",
    "cube",
    "cuboid",
    "disk",
    "ring",
    "L-shape",
    "T-shape",
    "flat",
    "elongated",
    "irregular",
    "unknown",
]

COLOR_CHOICES = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "silver",
    "golden",
    "beige",
    "transparent",
    "unknown",
]

# Maximum number of dominant colors to keep in <color>. The field stores
# a comma-separated list (e.g. "red,yellow") so that an asset with two
# equally prominent hues can still be retrieved by either color.
MAX_COLORS = 2

MATERIAL_CHOICES = [
    "plastic",
    "metal",
    "wood",
    "ceramic",
    "glass",
    "fabric",
    "rubber",
    "paper",
    "silicone",
    "leather",
    "stone",
    "organic",
    "wicker",
    "unknown",
]


URDF_TEMPLATE = """
<robot name="template_robot">
    <link name="template_link">
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="mesh.obj" scale="1.0 1.0 1.0"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="mesh.obj" scale="1.0 1.0 1.0"/>
            </geometry>
            <gazebo>
                <mu1>0.8</mu1>
                <mu2>0.6</mu2>
            </gazebo>
        </collision>
        <inertial>
            <mass value="1.0"/>
            <origin xyz="0 0 0"/>
            <inertia ixx="1.0" ixy="0.0" ixz="0.0"
                     iyy="1.0" iyz="0.0" izz="1.0"/>
        </inertial>
        <extra_info>
            <scale>1.0</scale>
            <uuid></uuid>
            <domain>unknown</domain>
            <super_category>unknown</super_category>
            <category>unknown</category>
            <name>unknown</name>
            <color>unknown</color>
            <shape>unknown</shape>
            <material>unknown</material>
            <description>unknown</description>
            <min_height>0.0</min_height>
            <max_height>0.0</max_height>
            <real_height>0.0</real_height>
            <min_mass>0.0</min_mass>
            <max_mass>0.0</max_mass>
            <version>"0.0.0"</version>
            <generate_time>"-1"</generate_time>
            <gs_model>""</gs_model>
        </extra_info>
    </link>
</robot>
"""

DEFAULT_PROMPT_TEMPLATE = (  # noqa: E501
    "{view_desc}of the 3D object asset,\n"
    "category: {category}.\n"
    "You are an expert in 3D object analysis and "
    "physical property estimation.\n"
    "Give the category of this object asset (within 3 words), "
    "(if category is\n"
    "already provided, use it directly), accurately describe "
    "this 3D object asset (within 15 words),\n"
    "Determine the pose of the object in the first image based "
    "on all views and estimate the true vertical height\n"
    "(vertical projection) range of the object (in meters), "
    "i.e., how tall the object appears from top\n"
    "to bottom in the first image. also weight range "
    "(unit: kilogram), the average\n"
    "static friction coefficient of the object relative to "
    "rubber and the average dynamic friction\n"
    "coefficient of the object relative to rubber. "
    "Return response in format as shown in Output Example.\n"
    "\n"
    "Additional fields (for structured metadata):\n"
    "- Super Category: broad super-category (1 word), e.g., "
    "fruit, furniture, tool, kitchenware, stationery, "
    "electronics, toy, vehicle, clothing.\n"
    "- Name: specific name for this object instance "
    "(within 3 words).\n"
    "- Color: list 1 OR 2 dominant colors. Pick 1 when a single hue "
    "clearly dominates (>~70% of visible surface). Pick 2 (in order of "
    "prominence) only when two colors are roughly equally prominent "
    "AND each occupies at least ~30% of visible surface. Never list "
    "more than 2. Each color must come from [{color_choices}]. "
    "Format: a single comma-separated value, e.g. 'red' or 'red,yellow' "
    "(no spaces around commas).\n"
    "- Shape: choose exactly ONE from [{shape_choices}].\n"
    "- Material: pick the single primary material by mass or by the "
    "functional/structural part. Choose exactly ONE from "
    "[{material_choices}].\n"
    "\n"
    "Output Example:\n"
    "Super Category: kitchenware\n"
    "Category: cup\n"
    "Name: golden cup\n"
    "Description: shiny golden cup with floral design\n"
    "Color: golden,red\n"
    "Shape: cylinder\n"
    "Material: metal\n"
    "Pose: <short_description_within_10_words>\n"
    "Height: 0.10-0.15 m\n"
    "Weight: 0.3-0.6 kg\n"
    "Static friction coefficient: 0.6\n"
    "Dynamic friction coefficient: 0.5\n"
    "\n"
    "IMPORTANT: Estimating Vertical Height from the First "
    "(Front View) Image and pose estimation based on "
    "all views.\n"
    '- The "vertical height" refers to the real-world '
    "vertical size of the object\n"
    "as projected in the first image, aligned with the "
    "image's vertical axis.\n"
    "- For flat objects like plates or disks or book, if "
    "their face is visible in the front view,\n"
    "use the diameter as the vertical height. If the edge "
    "is visible, use the thickness instead.\n"
    "- This is not necessarily the full length of the object, "
    "but how tall it appears\n"
    "in the first image vertically, based on its pose and "
    "orientation estimation on all views.\n"
    "- Distinguish whether the entire objects such as plates, "
    "books, pens, spoons, fork are placed\n"
    "    horizontally or vertically based on pictures from "
    "left, right views.\n"
    "\n"
    "Estimate the vertical projection of their real length "
    "based on its pose.\n"
    "For example:\n"
    "  - A pen standing upright in the first image "
    "(aligned with the image's vertical axis)\n"
    "    full body visible in the first and other image: "
    "-> vertically -> vertical height = 0.14-0.20 m\n"
    "  - A pen lying flat in the first image or either "
    "the tip or the tail is facing the image\n"
    "    (showing thickness or as a circle), left/right "
    "view can show the full body\n"
    "    -> horizontally -> vertical height "
    "= 0.018-0.025 m\n"
    "  - Tilted pen in the first image (e.g., ~45 angle): "
    "vertical height = 0.07-0.12 m\n"
    "- Use the rest views to help determine the object's "
    "3D pose and orientation.\n"
    "Assume the object is in real-world scale and estimate "
    "the approximate vertical height\n"
    "based on the pose estimation and how large it appears "
    "vertically in the first image.\n"
)


def normalize_category_label(category: str) -> str:
    """Normalize category labels and strip instance suffixes like ``_001``."""
    value = (category or "unknown").strip().lower().replace(" ", "_")
    value = re.sub(r"_\d+$", "", value)
    return value or "unknown"


class AssetLabeller:
    """Generate URDF files for 3D assets with attributes.

    Uses GPT vision models to estimate object properties from
    rendered views, and generates a URDF file with mesh, physics,
    and extended metadata including shape, color, material,
    super_category, and name.

    Supports OBJ, PLY, GLB, GLTF, USD, USDA, USDC, USDZ as
    input formats via the built-in nvdiffrast renderer. For
    USD-native rendering (e.g. via Blender), pass a custom
    ``render_fn`` to the constructor.

    Args:
        gpt_client: GPT client instance for vision queries.
        mesh_file_list: Additional files to copy alongside
            the mesh (e.g., texture images, .mtl files).
        prompt_template: Custom prompt template with
            ``{category}``, ``{shape_choices}``, and
            ``{view_desc}`` placeholders.
        attrs_name: List of attribute names for extra_info.
            Defaults to EXTRA_INFO_ATTRS.
        render_dir: Subdirectory name for rendered images.
        render_view_num: Number of views to render (default 4).
        render_resolution: Resolution for rendered images.
        render_fn: Optional custom render function with
            signature ``(mesh_path, output_root,
            num_images=, output_subdir=) -> list[str]``.
            When provided, this is called instead of the
            built-in nvdiffrast renderer.
        rotate_xyzw: Quaternion (x,y,z,w) for mesh rotation
            (e.g., to align Blender export with sim axes).

    Example:
        ```python
        from asset_labeller.gpt_client import load_client_from_config
        from asset_labeller.labeller import AssetLabeller

        client = load_client_from_config("configs/gpt_config.yaml")
        labeller = AssetLabeller(client, render_view_num=4)
        urdf_path = labeller(
            mesh_path="path/to/mesh.obj",
            output_root="output_dir",
            category="apple",
        )

        # With a custom render function:
        labeller = AssetLabeller(client, render_fn=my_custom_render_fn)
        urdf_path = labeller(
            mesh_path="path/to/scene.usd",
            output_root="output_dir",
        )
        ```
    """

    EXTRA_INFO_ATTRS = [
        "uuid",
        "domain",
        "super_category",
        "category",
        "name",
        "color",
        "shape",
        "material",
        "description",
        "min_height",
        "max_height",
        "real_height",
        "min_mass",
        "max_mass",
        "version",
        "generate_time",
        "gs_model",
    ]

    def __init__(
        self,
        gpt_client: GPTClient,
        mesh_file_list: list[str] = None,
        prompt_template: str = None,
        attrs_name: list[str] = None,
        render_dir: str = "renders",
        render_view_num: int = 6,
        render_resolution: tuple[int, int] = (512, 512),
        render_fn: callable = None,
        decompose_convex: bool = True,
        rotate_xyzw: list[float] = None,
        strict_textures: bool = False,
    ) -> None:
        self.gpt_client = gpt_client
        self.mesh_file_list = (
            mesh_file_list if mesh_file_list is not None else []
        )
        self.render_resolution = render_resolution
        self.render_view_num = render_view_num
        self.render_fn = render_fn
        self.decompose_convex = decompose_convex
        self.rotate_xyzw = rotate_xyzw
        # When True, audit_export_textures issues raise instead of just
        # logging a warning. Default off so a single broken asset does
        # not abort a batch; opt in for CI / curated source data.
        self.strict_textures = strict_textures

        if render_view_num == 4:
            view_desc = (
                "This is an orthographic projection showing the "
                "front(1st image), right(2nd), back(3rd), and left(4th) views "
            )
        elif render_view_num == 6:
            view_desc = (
                "This is rendered views showing the "
                "front(1st), right(2nd), back(3rd), left(4th), "
                "top(5th), and bottom(6th) views "
            )
        else:
            view_desc = "This is the rendered views "

        if prompt_template is None:
            prompt_template = DEFAULT_PROMPT_TEMPLATE
        self.prompt_template = prompt_template
        self._view_desc = view_desc

        if attrs_name is not None:
            self.attrs_name = attrs_name
        else:
            self.attrs_name = list(self.EXTRA_INFO_ATTRS)

        self.output_mesh_dir = "mesh"
        self.output_render_dir = render_dir
        self.estimated_attrs = {}

    @staticmethod
    def match_shape(raw: str) -> str:
        """Match a raw shape string to the closest SHAPE_CHOICES entry."""
        raw_lower = raw.lower().strip()
        choices_lower = [s.lower() for s in SHAPE_CHOICES]
        if raw_lower in choices_lower:
            return SHAPE_CHOICES[choices_lower.index(raw_lower)]
        match = get_close_matches(raw_lower, choices_lower, n=1, cutoff=0.5)
        if match:
            matched = SHAPE_CHOICES[choices_lower.index(match[0])]
            logger.warning(
                f"Shape '{raw}' not in SHAPE_CHOICES, matched to '{matched}'"
            )
            return matched
        logger.warning(
            f"Shape '{raw}' not in SHAPE_CHOICES and no close match, "
            "defaulting to 'irregular'"
        )
        return "irregular"

    @staticmethod
    def match_single_color(raw: str) -> str:
        """Match one raw color phrase to the closest COLOR_CHOICES entry.

        For compound phrases (e.g. "dark blue"), the last word is taken
        as the dominant hue. Falls back to fuzzy match, then 'unknown'.
        """
        raw_lower = raw.lower().strip()
        tokens = raw_lower.split()
        if tokens:
            raw_lower = tokens[-1]
        choices_lower = [c.lower() for c in COLOR_CHOICES]
        if raw_lower in choices_lower:
            return COLOR_CHOICES[choices_lower.index(raw_lower)]
        match = get_close_matches(raw_lower, choices_lower, n=1, cutoff=0.5)
        if match:
            return COLOR_CHOICES[choices_lower.index(match[0])]
        return "unknown"

    @staticmethod
    def match_color(raw: str) -> str:
        """Canonicalize a raw color value into <color> field form.

        The <color> field stores up to ``MAX_COLORS`` (2) dominant colors
        as a comma-separated list, e.g. "red" or "red,yellow". This helper
        accepts either a single phrase or a separator-delimited list (any
        of ``,``, ``/``, ``+``, ``&``, or the word ``and``) and returns
        the canonical comma-joined string. Each component is normalized
        via :meth:`match_single_color`. Duplicates are removed; result
        is capped at MAX_COLORS. ``unknown`` components are dropped when
        there is at least one valid color; if none survive, returns
        ``"unknown"``.
        """
        if raw is None:
            return "unknown"
        # Split on comma, slash, plus, ampersand, or the literal word 'and'.
        parts = [
            p.strip()
            for p in re.split(r"[,/+&]|\band\b", raw, flags=re.IGNORECASE)
            if p.strip()
        ]
        if not parts:
            logger.warning(
                f"Color '{raw}' produced no parsable tokens, "
                "defaulting to 'unknown'"
            )
            return "unknown"
        canonical: list[str] = []
        for p in parts:
            c = AssetLabeller.match_single_color(p)
            if c not in canonical:
                canonical.append(c)
        # Drop 'unknown' if at least one real color was matched.
        real = [c for c in canonical if c != "unknown"]
        result = real[:MAX_COLORS] if real else ["unknown"]
        joined = ",".join(result)
        if joined != raw.lower().strip():
            logger.warning(f"Color '{raw}' canonicalized to '{joined}'")
        return joined

    @staticmethod
    def match_material(raw: str) -> str:
        """Match a raw material string to the closest MATERIAL_CHOICES entry.

        For multi-material inputs (e.g. "plastic and metal",
        "metal-rubber"), tokenize and return the first token that matches
        a known material as the primary material. Falls back to fuzzy
        matching and finally to 'plastic'.
        """
        raw_lower = raw.lower().strip()
        choices_lower = [m.lower() for m in MATERIAL_CHOICES]
        if raw_lower in choices_lower:
            return MATERIAL_CHOICES[choices_lower.index(raw_lower)]
        tokens = [t for t in re.split(r"[\s,\-/]+|\band\b", raw_lower) if t]
        for tok in tokens:
            if tok in choices_lower:
                matched = MATERIAL_CHOICES[choices_lower.index(tok)]
                logger.warning(
                    f"Material '{raw}' not in MATERIAL_CHOICES, "
                    f"matched to '{matched}'"
                )
                return matched
        match = get_close_matches(raw_lower, choices_lower, n=1, cutoff=0.5)
        if match:
            matched = MATERIAL_CHOICES[choices_lower.index(match[0])]
            logger.warning(
                f"Material '{raw}' not in MATERIAL_CHOICES, "
                f"fuzzy-matched to '{matched}'"
            )
            return matched
        logger.warning(
            f"Material '{raw}' not in MATERIAL_CHOICES and no close "
            "match, defaulting to 'unknown'"
        )
        return "unknown"

    def parse_response(self, response: str) -> dict[str, any]:
        """Parse GPT response to extract asset attributes.

        Uses robust key-value parsing instead of positional line indexing.

        Args:
            response: GPT response string.

        Returns:
            Dictionary of parsed attributes.
        """
        raw_lines = response.strip().split("\n")
        lines = {}
        for line in raw_lines:
            line = line.strip()
            if not line or line.startswith("```") or ":" not in line:
                continue
            key, _, value = line.partition(":")
            lines[key.strip().lower()] = value.strip()

        super_category = lines.get("super category", "unknown")
        category = lines.get("category", "unknown")
        name = lines.get("name", "unknown")
        description = lines.get("description", "unknown")
        color = self.match_color(lines.get("color", "unknown"))
        material = self.match_material(lines.get("material", "unknown"))
        shape = self.match_shape(lines.get("shape", "irregular"))

        min_height, max_height = self.parse_range(
            lines.get("height", "0.1-0.2 m")
        )
        min_mass, max_mass = self.parse_range(
            lines.get("weight", "0.1-0.5 kg")
        )
        mu1 = self.parse_float(lines.get("static friction coefficient", "0.6"))
        mu2 = self.parse_float(
            lines.get("dynamic friction coefficient", "0.5")
        )

        return {
            "super_category": super_category.lower().replace(" ", "_"),
            "category": normalize_category_label(category),
            "name": name.lower(),
            "description": description.lower(),
            "color": color.lower(),
            "shape": shape,
            "material": material,
            "min_height": round(min_height, 4),
            "max_height": round(max_height, 4),
            "min_mass": round(min_mass, 4),
            "max_mass": round(max_mass, 4),
            "mu1": round(mu1, 2),
            "mu2": round(mu2, 2),
            "version": VERSION,
            "generate_time": datetime.now().strftime("%Y%m%d%H%M%S"),
        }

    @staticmethod
    def parse_range(value: str) -> tuple[float, float]:
        """Parse a range string like '0.07-0.10 m' into (min, max).

        Args:
            value: Range string.

        Returns:
            Tuple of (min_value, max_value).
        """
        try:
            # Remove unit suffixes
            cleaned = value.replace(",", "").strip()
            for unit in ["m", "kg", "cm", "g", "lb", "lbs"]:
                cleaned = cleaned.replace(unit, "").strip()

            parts = cleaned.split("-")
            if len(parts) == 2:
                return float(parts[0].strip()), float(parts[1].strip())
            else:
                v = float(parts[0].strip())
                return v, v
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse range: '{value}', using defaults")
            return 0.1, 0.2

    @staticmethod
    def parse_float(value: str) -> float:
        """Parse a float value from a string, stripping non-numeric chars.

        Args:
            value: String containing a float value.

        Returns:
            Parsed float value.
        """
        try:
            cleaned = value.replace(",", "").strip()
            # Take the first number-like token
            for token in cleaned.split():
                try:
                    return float(token)
                except ValueError:
                    continue
            return float(cleaned)
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse float: '{value}', using 0.5")
            return 0.5

    def generate_urdf(
        self,
        input_mesh: str,
        output_dir: str,
        attr_dict: dict,
        output_name: str = None,
    ) -> str:
        """Generate a URDF file for a given mesh with attributes.

        Steps:
            1. Load and normalize the mesh.
            2. Scale to real height.
            3. Save visual mesh as OBJ and GLB.
            4. Optionally generate convex-decomposed collision mesh.
            5. Fill URDF template with attributes.
            6. Write URDF file.

        Args:
            input_mesh: Path to the input mesh file.
            output_dir: Directory to store URDF and mesh files.
            attr_dict: Dictionary of asset attributes.
            output_name: Optional name for the URDF and robot.

        Returns:
            Path to the generated URDF file.
        """
        # 1. Load mesh
        mesh = load_mesh(input_mesh)

        # 2. Scaling strategy
        # - USD: keep authored geometry scale (already
        #   converted to meters in load_mesh), so exported
        #   OBJ matches the USD viewer (metersPerUnit baked).
        # - Non-USD: optionally scale to GPT real_height.
        ext = os.path.splitext(input_mesh)[1].lower()
        usd_exts = {".usd", ".usda", ".usdc", ".usdz"}

        if ext in usd_exts:
            scale = 1.0
            # Record the geometric height as metadata.
            # USD scenes may be Z-up; store max extent.
            try:
                geom_height = float(np.ptp(mesh.vertices, axis=0).max())
            except Exception:
                geom_height = 0.0
            if geom_height > 0:
                attr_dict["real_height"] = round(geom_height, 4)
        else:
            # Keep previous behavior for non-USD inputs.
            raw_height = float(
                np.ptp(mesh.vertices, axis=0)[1]
            )  # Y-up convention
            real_height = float(
                attr_dict.get("real_height", raw_height or 0.1)
            )
            if raw_height > 0:
                scale = round(real_height / raw_height, 6)
                mesh = mesh.apply_scale(scale)
            else:
                scale = 1.0

        # 3. Prepare output directories and save visual mesh
        mesh_folder = os.path.join(output_dir, self.output_mesh_dir)
        os.makedirs(mesh_folder, exist_ok=True)

        obj_name = os.path.basename(input_mesh)
        # Always export as .obj for URDF compatibility
        if not obj_name.endswith(".obj"):
            obj_name = os.path.splitext(obj_name)[0] + ".obj"
        mesh_output_path = os.path.join(mesh_folder, obj_name)
        glb_name = os.path.splitext(obj_name)[0] + ".glb"
        glb_output_path = os.path.join(mesh_folder, glb_name)
        if ext in usd_exts:
            copied_textures = export_usd_to_obj_with_materials(
                input_mesh, mesh_output_path
            )
            tex_issues = audit_export_textures(
                input_mesh, mesh_output_path, copied_textures
            )
            for code, detail in tex_issues:
                logger.warning(
                    "texture audit %s on %s: %s",
                    code,
                    input_mesh,
                    detail,
                )
            if tex_issues and self.strict_textures:
                raise RuntimeError(
                    f"texture audit failed for {input_mesh}: "
                    f"{[c for c, _ in tex_issues]}"
                )
        else:
            mesh.export(mesh_output_path)
        mesh.export(glb_output_path)

        collision_mesh_filename = os.path.join(self.output_mesh_dir, obj_name)
        if self.decompose_convex:
            try:
                d_params = dict(
                    threshold=0.05, max_convex_hull=100, verbose=False
                )
                collision_name = (
                    f"{os.path.splitext(obj_name)[0]}_collision.obj"
                )
                collision_output_path = os.path.join(
                    mesh_folder, collision_name
                )
                decompose_convex_mesh(
                    mesh_output_path, collision_output_path, **d_params
                )
                collision_mesh_filename = os.path.join(
                    self.output_mesh_dir, collision_name
                )
            except Exception as e:
                logger.warning(
                    "Convex decomposition failed for %s, %s. "
                    "Use original mesh for collision computation.",
                    mesh_output_path,
                    e,
                )

        # 4. Copy additional mesh files (textures, .mtl, etc.)
        input_dir = os.path.dirname(input_mesh)
        if ext not in usd_exts:
            for file in self.mesh_file_list:
                src_file = os.path.join(input_dir, file)
                dest_file = os.path.join(mesh_folder, file)
                if os.path.isfile(src_file):
                    shutil.copy(src_file, dest_file)

        # 5. Determine output name
        if output_name is None:
            output_name = os.path.splitext(obj_name)[0]

        # 6. Fill URDF template
        robot = ET.fromstring(URDF_TEMPLATE)
        robot.set("name", output_name)

        link = robot.find("link")
        if link is None:
            raise ValueError("URDF template is missing 'link' element.")
        link.set("name", output_name)

        # Apply rotation if specified
        if self.rotate_xyzw is not None:
            rpy = Rotation.from_quat(self.rotate_xyzw).as_euler(
                "xyz", degrees=False
            )
            rpy_str = " ".join(str(round(num, 4)) for num in rpy)
            link.find("visual/origin").set("rpy", rpy_str)
            link.find("collision/origin").set("rpy", rpy_str)

        # Update visual geometry
        visual_mesh = link.find("visual/geometry/mesh")
        if visual_mesh is not None:
            visual_mesh.set(
                "filename", os.path.join(self.output_mesh_dir, obj_name)
            )
            visual_mesh.set("scale", "1.0 1.0 1.0")

        # Update collision geometry
        collision_mesh = link.find("collision/geometry/mesh")
        if collision_mesh is not None:
            collision_mesh.set("filename", collision_mesh_filename)
            collision_mesh.set("scale", "1.0 1.0 1.0")

        # Update friction coefficients
        gazebo = link.find("collision/gazebo")
        if gazebo is not None:
            mu1_elem = gazebo.find("mu1")
            mu2_elem = gazebo.find("mu2")
            if mu1_elem is not None:
                mu1_elem.text = f"{attr_dict.get('mu1', 0.8):.2f}"
            if mu2_elem is not None:
                mu2_elem.text = f"{attr_dict.get('mu2', 0.6):.2f}"

        # Update mass
        mass_elem = link.find("inertial/mass")
        if mass_elem is not None:
            mass_value = (
                attr_dict.get("min_mass", 1.0) + attr_dict.get("max_mass", 1.0)
            ) / 2
            mass_elem.set("value", f"{mass_value:.4f}")

        # Update scale in extra_info
        scale_elem = link.find("extra_info/scale")
        if scale_elem is not None:
            scale_elem.text = f"{scale:.6f}"

        # Update all extra_info attributes
        for key in self.attrs_name:
            elem = link.find(f"extra_info/{key}")
            if elem is not None and key in attr_dict:
                elem.text = f"{attr_dict[key]}"

        # 7. Write URDF file
        os.makedirs(output_dir, exist_ok=True)
        urdf_path = os.path.join(output_dir, f"{output_name}.urdf")
        tree = ET.ElementTree(robot)
        tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

        logger.info(f"URDF file saved to {urdf_path}")
        return urdf_path

    @staticmethod
    def get_attr_from_urdf(
        urdf_path: str,
        attr_root: str = ".//link/extra_info",
        attr_name: str = "scale",
    ):
        """Extract an attribute value from a URDF file.

        Args:
            urdf_path: Path to the URDF file.
            attr_root: XML path to the attribute root element.
            attr_name: Name of the attribute to extract.

        Returns:
            The attribute value (float or string), or None if not found.
        """
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        tree = ET.parse(urdf_path)
        root = tree.getroot()
        extra_info = root.find(attr_root)
        if extra_info is not None:
            elem = extra_info.find(attr_name)
            if elem is not None:
                value = elem.text
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return value
        return None

    @staticmethod
    def add_quality_tag(
        urdf_path: str, results: list, output_path: str = None
    ) -> None:
        """Add quality check results as a tag in the URDF file.

        Args:
            urdf_path: Path to the URDF file.
            results: List of [checker_name, result] pairs.
            output_path: Output path (defaults to overwriting input).
        """
        if output_path is None:
            output_path = urdf_path

        tree = ET.parse(urdf_path)
        root = tree.getroot()
        custom_data = ET.SubElement(root, "custom_data")
        quality = ET.SubElement(custom_data, "quality")
        for key, value in results:
            tag = ET.SubElement(quality, key)
            tag.text = str(value)

        rough_string = ET.tostring(root, encoding="utf-8")
        formatted = parseString(rough_string).toprettyxml(indent="   ")
        cleaned = "\n".join(
            line for line in formatted.splitlines() if line.strip()
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        logger.info(f"Quality tag added to {output_path}")

    def get_estimated_attributes(self, asset_attrs: dict) -> dict:
        """Calculate summary attributes from parsed asset properties.

        Args:
            asset_attrs: Full asset attribute dictionary.

        Returns:
            Summary dict with height, mass, mu, category, etc.
        """
        return {
            "height": round(
                (asset_attrs["min_height"] + asset_attrs["max_height"]) / 2, 4
            ),
            "mass": round(
                (asset_attrs["min_mass"] + asset_attrs["max_mass"]) / 2, 4
            ),
            "mu": round((asset_attrs["mu1"] + asset_attrs["mu2"]) / 2, 4),
            "super_category": asset_attrs.get("super_category", "unknown"),
            "category": asset_attrs.get("category", "unknown"),
            "name": asset_attrs.get("name", "unknown"),
            "color": asset_attrs.get("color", "unknown"),
            "shape": asset_attrs.get("shape", "unknown"),
            "material": asset_attrs.get("material", "unknown"),
        }

    def _render_views(self, mesh_path: str, output_root: str) -> list[str]:
        """Render multi-view images of an asset.

        If a custom ``render_fn`` was provided at init time, it is used
        directly with the file path (supports any format the function
        handles, e.g. USD).  Otherwise the built-in nvdiffrast pipeline
        is used after loading the mesh via ``load_mesh`` (which already
        supports OBJ, PLY, GLB, GLTF, USD, USDA, USDC, USDZ).

        Args:
            mesh_path: Path to the mesh / USD file.
            output_root: Base output directory.

        Returns:
            List of rendered image file paths.
        """
        render_dir = os.path.join(output_root, self.output_render_dir)

        if self.render_fn is not None:
            image_paths = self.render_fn(
                mesh_path,
                output_root,
                num_images=self.render_view_num,
                output_subdir=self.output_render_dir,
            )
        else:
            mesh = load_render_asset(mesh_path)
            views = select_views(self.render_view_num)
            image_paths = render_views(
                mesh,
                output_dir=render_dir,
                views=views,
                resolution=self.render_resolution,
            )

        logger.info(f"Rendered {len(image_paths)} views to {render_dir}")
        return image_paths

    def __call__(
        self,
        mesh_path: str,
        output_root: str,
        text_prompt: str = None,
        category: str = "unknown",
        **kwargs,
    ) -> str:
        """Generate a URDF file for a mesh asset.

        Full pipeline:
            1. Build prompt from template.
            2. Render multi-view images (OBJ, USD, etc.).
            3. Query GPT for attribute estimation.
            4. Parse response and merge with overrides.
            5. Generate URDF with mesh and attributes.

        Args:
            mesh_path: Path to the mesh file (.obj, .usd, .usda, etc.).
            output_root: Directory for all outputs.
            text_prompt: Custom prompt (uses default template if None).
            category: Category hint for GPT prompt.
            **kwargs: Attribute overrides (e.g., min_height=0.1).

        Returns:
            Path to the generated URDF file.
        """
        if text_prompt is None or len(text_prompt) == 0:
            normalized_category = normalize_category_label(category)
            text_prompt = self.prompt_template.format(
                category=normalized_category,
                shape_choices=", ".join(SHAPE_CHOICES),
                color_choices=", ".join(COLOR_CHOICES),
                material_choices=", ".join(MATERIAL_CHOICES),
                view_desc=self._view_desc,
            )

        image_paths = self._render_views(mesh_path, output_root)

        response = self.gpt_client.query(text_prompt, image_paths)

        if response is None:
            logger.warning("API returned None, using default attributes.")
            normalized_category = normalize_category_label(category)
            asset_attrs = {
                "super_category": "unknown",
                "category": normalized_category,
                "name": normalized_category,
                "description": normalized_category,
                "color": "unknown",
                "shape": "unknown",
                "material": "unknown",
                "min_height": 1,
                "max_height": 1,
                "min_mass": 1,
                "max_mass": 1,
                "mu1": 0.8,
                "mu2": 0.6,
                "version": VERSION,
                "generate_time": datetime.now().strftime("%Y%m%d%H%M%S"),
            }
        else:
            asset_attrs = self.parse_response(response)

        if category != "unknown":
            kwargs = dict(kwargs)
            kwargs["category"] = normalize_category_label(category)
        for key in self.attrs_name:
            if key in kwargs:
                asset_attrs[key] = kwargs[key]

        asset_attrs["real_height"] = round(
            (asset_attrs["min_height"] + asset_attrs["max_height"]) / 2, 4
        )

        self.estimated_attrs = self.get_estimated_attributes(asset_attrs)

        urdf_path = self.generate_urdf(mesh_path, output_root, asset_attrs)

        logger.info(f"response: {response}")

        return urdf_path
