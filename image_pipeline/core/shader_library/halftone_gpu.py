"""halftone_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("halftone_gpu", "Halftone dot-screen of the input (typed)",
          "filter", '''
void main() {
    vec2 uv = v_uv;
    float ang = radians(u_angle);
    vec2 rp = rot(ang) * (uv - 0.5) + 0.5;
    float scale = max(u_scale, 4.0);
    vec2 cell = fract(rp * scale) - 0.5;
    float d = length(cell);
    float lum = dot(texture(u_texture, uv).rgb, vec3(0.299, 0.587, 0.114));
    float radius = (1.0 - lum) * 0.7 * u_dot;
    float dot_ = smoothstep(radius, radius - 0.08, d);
    vec3 col = mix(u_bg, u_ink, dot_);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale": {"glsl": "float", "min": 8.0, "max": 120.0, "default": 48.0,
              "description": "dot grid density"},
    "angle": {"glsl": "float", "min": 0.0, "max": 90.0, "default": 15.0,
              "description": "screen angle (deg)"},
    "dot":   {"glsl": "float", "min": 0.2, "max": 2.0, "default": 1.0,
              "description": "dot size multiplier"},
    "bg":    {"glsl": "color", "default": "#ffffff", "description": "paper color"},
    "ink":   {"glsl": "color", "default": "#101010", "description": "ink color"},
})