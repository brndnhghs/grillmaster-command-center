"""wallpaper_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO




# ═══════════════════════════════════════════════════════════════════════════
#  P1 — GPU reaction-diffusion sim shaders (client ping-pong; server untouched)
# ═══════════════════════════════════════════════════════════════════════════
# Three-shader set for a WebGL2 ping-pong float-state simulation. The client
# (client3d.js) owns the {a,b} RGBA-float state pair and the substep loop; these
# GLSL bodies define seed → step → display. State packs U in .r, V in .g.
# The CPU node (methods/simulations/gray_scott.py, id 155) stays the
# authoritative export — nothing here is rendered server-side.

_register("wallpaper_gpu",
            "Wallpaper-group tiling (client-GPU twin of node 06)",
            "procedural", _INFERNO + """
void main() {
    // u_params.x = tile size (log-scaled), .y = color variation, .z = rotation noise
    float ts = mix(8.0, 64.0, clamp(u_scale_variation, 0.0, 1.0));
    float cv = u_color_variation;
    vec2 st = v_uv * u_resolution / ts;
    vec2 g = floor(st), f = fract(st);
    // per-tile slight rotation + hue offset (echoes rotation_noise/color_variation)
    float r = (hash21(g) - 0.5) * 6.2831853 * cv;
    float hue = hash21(g + 3.17) * cv;
    vec2 p = (f - 0.5) * rot(r);
    float d = abs(p.x) + abs(p.y);            // diamond motif
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (hue + vec3(0.0, 0.33, 0.67)));
    col *= smoothstep(0.5, 0.45, d);
    f_color = vec4(inferno(clamp(length(col) * 0.6 + d * 0.4, 0.0, 1.0)), 1.0);
}
""",
    uniforms={
    "scale_variation": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "tile size variation"},
    "color_variation": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "color variation"}
}
    )