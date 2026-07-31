"""flow_field_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("flow_field_typed", "Curl-noise flow field streamlines (typed, node 281)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.1 * u_speed;
    vec2 q = p * u_zoom;
    float ang = fbm(q + vec2(t, -t)) * 6.28318530 * u_swirl;
    vec2 dir = vec2(cos(ang), sin(ang));
    float stripe = sin(dot(p, dir) * u_freq * 6.28318530 + t * 4.0);
    float line = smoothstep(1.0 - u_density, 1.0, abs(stripe));
    vec3 col = mix(u_bg, u_fg, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "zoom":    {"glsl": "float", "min": 0.5, "max": 6.0, "default": 2.5,
                "description": "noise zoom"},
    "swirl":   {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0,
                "description": "flow curl amount"},
    "freq":    {"glsl": "float", "min": 4.0, "max": 40.0, "default": 16.0,
                "description": "streamline density"},
    "density": {"glsl": "float", "min": 0.05, "max": 0.8, "default": 0.35,
                "description": "line coverage"},
    "speed":   {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                "description": "animation speed"},
    "bg":      {"glsl": "color", "default": "#070510", "description": "background"},
    "fg":      {"glsl": "color", "default": "#8affc1", "description": "streamlines"},
})