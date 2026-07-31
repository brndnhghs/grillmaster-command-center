"""weave_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("weave_typed", "Woven fabric: over/under threads on a typed checker grid (node 291)",
          "procedural", '''void main() {
    vec2 uv = v_uv * u_scale;
    uv += u_time * u_speed * 0.05;
    vec2 g = floor(uv);
    vec2 fv = fract(uv);
    float parity = mod(g.x + g.y, 2.0);
    float bulge;
    float along;
    if (parity < 0.5) { bulge = sin(fv.y * 3.14159265); along = fv.x; }
    else              { bulge = sin(fv.x * 3.14159265); along = fv.y; }
    float thread = smoothstep(0.0, 0.5, bulge) * smoothstep(1.0, 0.5, bulge);
    float shade = 0.5 + 0.5 * sin(along * 3.14159265);
    vec3 base = (parity < 0.5) ? u_color_a : u_color_b;
    vec3 col = base * (0.45 + 0.6 * shade);
    col *= (0.35 + 0.65 * thread);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale":   {"glsl": "float", "min": 2.0, "max": 40.0, "default": 12.0,
                "description": "thread count"},
    "speed":   {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                "description": "drift speed"},
    "color_a": {"glsl": "color", "default": "#b3421f", "description": "weft color"},
    "color_b": {"glsl": "color", "default": "#1f5ab3", "description": "warp color"},
})