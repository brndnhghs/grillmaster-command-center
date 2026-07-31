"""truchet_sdf_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── 426 Smooth Truchet (SDF) ──
_register("truchet_sdf_gpu", "Smooth Truchet (client-GPU twin of node 426)", "procedural", '''
void main() {
    float tile = max(8.0, u_tile_size);
    vec2 uv = v_uv * u_resolution;
    vec2 cell = floor(uv / tile);
    vec2 local = fract(uv / tile);                       // [0,1]
    local = rot(u_time * u_anim_speed) * (local - 0.5) + 0.5;   // flow
    float rnd = hash21(cell);
    if (rnd > 0.5) local.x = 1.0 - local.x;
    float d1 = abs(length(local) - 0.5);                 // arc at corner (0,0)
    float d2 = abs(length(local - 1.0) - 0.5);           // arc at corner (1,1)
    float d = min(d1, d2);
    float aa = 2.0 / tile;
    float lineMask = smoothstep(u_stroke * 0.5 + aa, u_stroke * 0.5 - aa, abs(d));
    float glow = u_edge_glow * smoothstep(0.12, 0.0, abs(d));
    float hue = fract(rnd + u_time * 0.05);
    vec3 base = 0.5 + 0.5 * cos(6.2831853 * (hue + vec3(0.0, 0.33, 0.67)));
    vec3 col = mix(vec3(0.05), base, lineMask);
    col += glow * base;
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
        "tile_size": {"glsl": "float", "min": 24.0, "max": 200.0, "default": 56.0, "description": "tile size (px)"},
        "stroke": {"glsl": "float", "min": 0.04, "max": 0.4, "default": 0.13, "description": "tube width (frac of tile)"},
        "edge_glow": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.25, "description": "outer glow strength"},
        "anim_speed": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 0.6, "description": "animation speed"},
    })