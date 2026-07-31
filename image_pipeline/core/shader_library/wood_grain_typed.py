"""wood_grain_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("wood_grain_typed", "Wood grain rings with typed rings/scale/colors (node 256)",
          "procedural", '''void main() {
    vec2 uv = v_uv - 0.5;
    float d = length(uv) * max(u_scale, 1.0);
    float grain = sin(d * max(u_rings, 1.0) + fbm(uv * 10.0) * u_turb) * 0.5 + 0.5;
    vec3 col = mix(u_dark, u_light, grain);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "rings": {"glsl": "float", "min": 1.0, "max": 40.0, "default": 8.0,
              "description": "ring frequency"},
    "scale": {"glsl": "float", "min": 1.0, "max": 30.0, "default": 10.0,
              "description": "ring spread"},
    "turb":  {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.5,
              "description": "grain turbulence"},
    "dark":  {"glsl": "color", "default": "#3a1d0a", "description": "dark wood"},
    "light": {"glsl": "color", "default": "#9a5a2a", "description": "light wood"},
})