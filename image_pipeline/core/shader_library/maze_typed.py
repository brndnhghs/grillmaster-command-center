"""maze_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("maze_typed", "Hash maze: procedural wall grid (typed, node 272)",
          "procedural", '''void main() {
    vec2 uv = v_uv * max(u_scale, 1.0);
    uv += vec2(u_time * u_drift * 0.05, 0.0);
    vec2 g = floor(uv);
    vec2 f = fract(uv);
    float hw = max(u_wall, 0.02) * 0.5;
    float h1 = hash21(g);
    float h2 = hash21(g + 17.3);
    float vwall = step(1.0 - u_density, h1) * step(f.x, hw);
    float hwall = step(1.0 - u_density, h2) * step(f.y, hw);
    float wall = clamp(vwall + hwall, 0.0, 1.0);
    vec3 col = mix(u_bg, u_fg, wall);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale":   {"glsl": "float", "min": 4.0, "max": 60.0, "default": 18.0,
                "description": "grid density"},
    "wall":    {"glsl": "float", "min": 0.04, "max": 0.5, "default": 0.18,
                "description": "wall thickness"},
    "density": {"glsl": "float", "min": 0.02, "max": 0.6, "default": 0.25,
                "description": "wall probability"},
    "drift":   {"glsl": "float", "min": 0.0, "max": 4.0, "default": 0.0,
                "description": "scroll drift"},
    "bg":      {"glsl": "color", "default": "#0a0a14", "description": "background"},
    "fg":      {"glsl": "color", "default": "#9be7ff", "description": "wall color"},
})