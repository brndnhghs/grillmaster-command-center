"""three.js 3D node definitions.

Extracted from ``core/graph.py`` (TD-07 / ROADMAP R8) so the graph-execution
module stays focused on execution rather than client-side 3D node metadata.
These defs are pure serialisable NodeDef dicts consumed by
``graph.get_all_node_defs()``; they carry no execution logic.

Nothing here imports from ``graph.py`` — ``graph.py`` imports the public names
(``THREEJS_3D_NODE_DEFS``, ``THREEJS_POSTFX_PARAMS``, ``MODEL_PLACEMENT_PARAMS``)
from here. That one-way dependency keeps the extraction mechanically safe.
"""

from __future__ import annotations


def threejs_node_def(method_id: str, name: str, *,
                      category: str = "client_3d",
                      tags: list[str] | None = None,
                      inputs: dict[str, str] | None = None,
                      outputs: dict[str, str] | None = None,
                      params: dict[str, dict] | None = None,
                      description: str = "",
                      deprecated: bool = False) -> dict:
    """Build a serialisable NodeDef dict for a three.js 3D node."""
    return {
        "method_id": method_id, "name": name,
        "category": category, "tags": tags or [],
        "inputs": inputs or {}, "outputs": outputs or {},
        "param_ports": [], "description": description,
        "version": 1, "deprecated": deprecated,
        "start_frame": 0, "end_frame": 0, "prebake": 0,
        "params": params or {},
    }


# Shared placement params for model nodes (__gltf__ / __usd__) — lets the 3D
# viewport editor's transform gizmo write positions/rotations back to them.
MODEL_PLACEMENT_PARAMS: dict[str, dict] = {
    "pos_x": {"description": "position X", "min": -5, "max": 5, "default": 0},
    "pos_y": {"description": "position Y", "min": -5, "max": 5, "default": 0},
    "pos_z": {"description": "position Z", "min": -5, "max": 5, "default": 0},
    "rot_x": {"description": "rotation X (deg)", "min": -180, "max": 180, "default": 0},
    "rot_y": {"description": "rotation Y (deg)", "min": -180, "max": 180, "default": 0},
    "rot_z": {"description": "rotation Z (deg)", "min": -180, "max": 180, "default": 0},
}

# Shared post-processing params for the 3D Scene Render / Scene (legacy) nodes.
# The headless sidecar runs a core-three.js-only multi-pass stack (bloom +
# grade + vignette + FXAA + radial chromatic aberration); all values sit at
# their neutral default so the default render path is unchanged unless the
# user opts in.
THREEJS_POSTFX_PARAMS: dict[str, dict] = {
    "bloom":            {"description": "bloom glow strength (0 = off)", "min": 0, "max": 2, "default": 0},
    "bloom_threshold":  {"description": "bloom luminance threshold", "min": 0, "max": 2, "default": 0.8},
    "bloom_knee":       {"description": "bloom knee softness", "min": 0, "max": 1, "default": 0.2},
    "bloom_intensity":  {"description": "bloom additive intensity", "min": 0, "max": 3, "default": 0.6},
    "bloom_radius":     {"description": "bloom blur radius (px)", "min": 0.25, "max": 4, "default": 1.0},
    "bloom_passes":     {"description": "bloom blur passes", "min": 1, "max": 16, "default": 4},
    "fx_brightness":    {"description": "grade brightness", "min": 0, "max": 2, "default": 1.0},
    "fx_contrast":      {"description": "grade contrast", "min": 0, "max": 2, "default": 1.0},
    "fx_saturation":    {"description": "grade saturation", "min": 0, "max": 2, "default": 1.0},
    "vignette":         {"description": "vignette strength (0 = off)", "min": 0, "max": 1, "default": 0},
    "vignette_radius":  {"description": "vignette radius", "min": 0.2, "max": 1.2, "default": 0.85},
    "vignette_softness": {"description": "vignette edge softness", "min": 0.05, "max": 1, "default": 0.5},
    "fxaa":             {"description": "FXAA edge anti-alias (0 = off)", "min": 0, "max": 1, "default": 0},
    "grain":            {"description": "film grain strength (0 = off); blue-noise/IGN-dithered ISO noise", "min": 0, "max": 1.5, "default": 0},
    "grain_size":       {"description": "grain dot size in px (1 = per-pixel fine grain)", "min": 1, "max": 4, "default": 1.0},
    "chromatic":        {"description": "radial chromatic aberration strength (0 = off)", "min": 0, "max": 1, "default": 0},
    "chromatic_scale":  {"description": "chromatic aberration radial falloff power", "min": 0.25, "max": 4, "default": 1.0},
    "radial_blur":      {"description": "radial (dolly-zoom) blur strength (0 = off)", "min": 0, "max": 1, "default": 0},
    "radial_blur_falloff": {"description": "radial blur sharpness at center (higher = tighter focus point)", "min": 0.25, "max": 4, "default": 1.0},
    "lens_distortion": {"description": "lens distortion (barrel/pincushion): +wide-angle, -telephoto, 0 = off", "min": -1, "max": 1, "default": 0},
    "lens_distortion_scale": {"description": "lens distortion radial falloff power (1 = quadratic, 2 = cubic bulge)", "min": 0.5, "max": 3, "default": 1.0},
    "lens_distortion_anim": {"description": "lens breathing amplitude (slow sinusoidal zoom — makes a static scene read as alive; 0 = off)", "min": 0, "max": 1, "default": 0},
    # Screen-Space Ambient Occlusion (SSAO) — depth-aware crevice darkening.
    # The headless sidecar renders the scene a second time with a normal+depth
    # override material and compares each pixel's depth against its
    # hemisphere-sampled neighbours; crevices darken. 0 = off (direct path).
    "ssao": {"description": "screen-space ambient occlusion strength (0 = off)", "min": 0, "max": 2, "default": 0},
    "ssao_radius": {"description": "SSAO sampling radius (fraction of screen)", "min": 0.05, "max": 0.6, "default": 0.3},
    "ssao_bias": {"description": "SSAO depth bias (occlusion threshold — larger skips near-self)", "min": 0.0, "max": 0.1, "default": 0.01},
    "ssao_power": {"description": "SSAO falloff exponent (higher = tighter, darker contact shadows)", "min": 0.5, "max": 4.0, "default": 1.5},
}

THREEJS_3D_NODE_DEFS: dict[str, dict] = {
    '__geometry__': threejs_node_def('__geometry__', '3D Geometry', tags=['3d', 'client'],
        outputs={"geometry": "geometry"},
        description='Emits a geometry for a Mesh node.',
        params={
            "shape":  {"description": "geometry shape",
                "choices": ["box","sphere","torus","torusknot","cone","cylinder",
                            "icosahedron","dodecahedron","plane"],
                "default": "torusknot"},
            "size":   {"description": "overall size", "min": 0.1, "max": 3, "default": 1},
            "detail": {"description": "tessellation detail", "min": 0, "max": 1, "default": 0.5},
        }),
    '__material__': threejs_node_def('__material__', '3D Material', tags=['3d', 'client', 'pbr'],
        outputs={"material": "material"},
        description='PBR material for a Mesh node.',
        params={
            "color":              {"description": "base color", "default": "#4a9eff"},
            "metalness":          {"description": "metalness", "min": 0, "max": 1, "default": 0.4},
            "roughness":          {"description": "roughness", "min": 0, "max": 1, "default": 0.35},
            "emissive":           {"description": "emissive color", "default": "#000000"},
            "emissive_intensity": {"description": "emissive intensity", "min": 0, "max": 4, "default": 1},
            "flat_shading":       {"description": "flat shading (0/1)", "min": 0, "max": 1, "default": 0},
        }),
    '__mesh3d__': threejs_node_def('__mesh3d__', '3D Mesh', tags=['3d', 'client'],
        inputs={"geometry": "geometry", "material": "material"},
        outputs={"object": "object3d"},
        description='Geometry + Material → transformable object (keyframeable).',
        params={
            "pos_x": {"description": "position X", "min": -5, "max": 5, "default": 0},
            "pos_y": {"description": "position Y", "min": -5, "max": 5, "default": 0},
            "pos_z": {"description": "position Z", "min": -5, "max": 5, "default": 0},
            "rot_x": {"description": "rotation X (deg)", "min": -180, "max": 180, "default": 0},
            "rot_y": {"description": "rotation Y (deg)", "min": -180, "max": 180, "default": 0},
            "rot_z": {"description": "rotation Z (deg)", "min": -180, "max": 180, "default": 0},
            "spin_speed": {"description": "auto Y-spin (rad/s)", "min": 0, "max": 4, "default": 0.6},
            "scale": {"description": "uniform scale", "min": 0.1, "max": 3, "default": 1},
        }),
    '__group3d__': threejs_node_def('__group3d__', '3D Group', tags=['3d', 'client'],
        inputs={"object_a": "object3d", "object_b": "object3d"},
        outputs={"object": "object3d"},
        description='Combine two objects into one.'),
    '__light3d__': threejs_node_def('__light3d__', '3D Light', tags=['3d', 'client'],
        outputs={"light": "light"},
        description='Light for a Scene Render node.',
        params={
            "type":      {"description": "light type",
                "choices": ["point","directional","spot"], "default": "point"},
            "pos_x":     {"description": "position X", "min": -10, "max": 10, "default": 3},
            "pos_y":     {"description": "position Y", "min": -10, "max": 10, "default": 4},
            "pos_z":     {"description": "position Z", "min": -10, "max": 10, "default": 5},
            "color":     {"description": "light color", "default": "#ffffff"},
            "intensity": {"description": "intensity", "min": 0, "max": 500, "default": 60},
        }),
    '__camera3d__': threejs_node_def('__camera3d__', '3D Camera', tags=['3d', 'client'],
        outputs={"camera": "camera"},
        description='Camera for a Scene Render node.',
        params={
            "pos_x": {"description": "camera X", "min": -12, "max": 12, "default": 0},
            "pos_y": {"description": "camera Y", "min": -12, "max": 12, "default": 0},
            "pos_z": {"description": "camera Z (dolly)", "min": 0.5, "max": 16, "default": 4},
            "look_x": {"description": "look-at X", "min": -5, "max": 5, "default": 0},
            "look_y": {"description": "look-at Y", "min": -5, "max": 5, "default": 0},
            "look_z": {"description": "look-at Z", "min": -5, "max": 5, "default": 0},
            "fov":    {"description": "field of view", "min": 15, "max": 110, "default": 50},
        }),
    '__scene_render__': threejs_node_def('__scene_render__', '3D Scene Render',
        tags=['3d', 'client', 'render'],
        inputs={"object": "object3d", "light": "light", "camera": "camera"},
        outputs={"image": "image", "luminance": "field"},
        description='Assemble object(s) + light + camera → rendered image.',
        params={
            "bg_color": {"description": "background color", "default": "#0a0e18"},
            "bg_mode":  {"description": "background mode",
                "choices": ["color", "transparent"], "default": "color"},
            "ambient":  {"description": "ambient light", "min": 0, "max": 1, "default": 0.35},
            "exposure": {"description": "tone-map exposure", "min": 0.1, "max": 3, "default": 1.0},
            "tone_map": {"description": "tone mapping operator",
                "choices": ["aces", "agx", "neutral", "reinhard", "cineon", "linear", "none"],
                "default": "aces"},
            "env_preset": {"description": "procedural environment preset",
                "choices": ["studio", "warm", "cool", "none"], "default": "studio"},
            "env_intensity": {"description": "environment/reflection strength",
                "min": 0, "max": 3, "default": 1.0},
            "shadows": {"description": "cast render shadows (0/1)", "min": 0, "max": 1, "default": 0},
            **THREEJS_POSTFX_PARAMS,
        }),
    '__scene3d__': threejs_node_def('__scene3d__', '3D Scene (legacy)',
        tags=['3d', 'client', 'deprecated'],
        inputs={"object": "object3d", "light": "light", "camera": "camera"},
        outputs={"image": "image", "luminance": "field"},
        description='Legacy 3D scene node (deprecated, use Scene Render).',
        deprecated=True,
        params={
            "bg_color": {"description": "background color", "default": "#0a0e18"},
            "bg_mode":  {"description": "background mode",
                "choices": ["color", "transparent"], "default": "color"},
            "ambient":  {"description": "ambient light", "min": 0, "max": 1, "default": 0.35},
            "exposure": {"description": "tone-map exposure", "min": 0.1, "max": 3, "default": 1.0},
            "tone_map": {"description": "tone mapping operator",
                "choices": ["aces", "agx", "neutral", "reinhard", "cineon", "linear", "none"],
                "default": "aces"},
            "env_preset": {"description": "procedural environment preset",
                "choices": ["studio", "warm", "cool", "none"], "default": "studio"},
            "env_intensity": {"description": "environment/reflection strength",
                "min": 0, "max": 3, "default": 1.0},
            "shadows": {"description": "cast render shadows (0/1)", "min": 0, "max": 1, "default": 0},
            **THREEJS_POSTFX_PARAMS,
        }),
    '__gltf__': threejs_node_def('__gltf__', '3D Model (GLTF)',
        tags=['3d', 'client', 'gltf'],
        outputs={"object": "object3d"},
        description='Load a .gltf/.glb model as an object.',
        params={
            "url":        {"description": "model URL (.gltf/.glb)",
                "default": "https://raw.githubusercontent.com/mrdoob/three.js/r160/examples/models/gltf/DamagedHelmet/glTF/DamagedHelmet.gltf"},
            **MODEL_PLACEMENT_PARAMS,
            "scale":      {"description": "uniform scale", "min": 0.05, "max": 5, "default": 1},
            "spin_speed": {"description": "auto Y-spin (rad/s)", "min": 0, "max": 4, "default": 0.6},
        }),
    '__usd__': threejs_node_def('__usd__', '3D Model (USD)',
        tags=['3d', 'client', 'usd'],
        outputs={"object": "object3d"},
        description='Load a Universal Scene Description model (.usdz or ASCII '
                    '.usda/.usd; binary .usdc crate files are not supported). '
                    'Upload via the node panel or reference a URL.',
        params={
            "url":        {"description": "model URL (.usdz/.usda/.usd — upload via node panel)",
                "default": ""},
            **MODEL_PLACEMENT_PARAMS,
            "scale":      {"description": "uniform scale", "min": 0.05, "max": 5, "default": 1},
            "spin_speed": {"description": "auto Y-spin (rad/s)", "min": 0, "max": 4, "default": 0.6},
        }),
}
