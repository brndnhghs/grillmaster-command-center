"""truchet_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("truchet_gpu", "Truchet arc tiling (typed procedural)",
          "procedural", '''
void main() {
    vec2 uv = v_uv * max(u_scale, 1.0);
    uv += u_time * u_drift * vec2(0.1, 0.06);
    vec2 g = floor(uv), f = fract(uv);
    float flip = step(0.5, hash21(g));
    if (flip > 0.5) f.x = 1.0 - f.x;
    float d1 = length(f - vec2(0.0, 0.0));
    float d2 = length(f - vec2(1.0, 1.0));
    float lw = u_width * 0.5;
    float arc = min(abs(d1 - 0.5), abs(d2 - 0.5));
    float line = smoothstep(lw, lw - 0.06, arc);
    vec3 col = mix(u_bg, u_fg, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale": {"glsl": "float", "min": 2.0, "max": 40.0, "default": 10.0,
              "description": "tile density"},
    "width": {"glsl": "float", "min": 0.05, "max": 0.6, "default": 0.25,
              "description": "arc line width"},
    "drift": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.0,
              "description": "animated drift"},
    "bg":    {"glsl": "color", "default": "#0a0a14", "description": "background"},
    "fg":    {"glsl": "color", "default": "#ffd166", "description": "arc color"},
})