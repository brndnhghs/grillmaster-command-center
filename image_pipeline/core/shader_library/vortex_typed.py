"""vortex_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("vortex_typed", "Spiral vortex / galaxy field with typed arms/twist/falloff (node 290)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float r = length(p);
    float a = atan(p.y, p.x);
    float t = u_time * u_speed;
    float swirl = a + r * u_twist - t;
    float bands = 0.5 + 0.5 * sin(swirl * u_arms);
    float density = exp(-r * u_falloff);
    vec3 col = mix(u_bg, mix(u_color_a, u_color_b, bands), density);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":   {"glsl": "float", "min": 0.0, "max": 4.0, "default": 0.8,
                "description": "spin speed"},
    "arms":    {"glsl": "float", "min": 1.0, "max": 24.0, "default": 4.0,
                "description": "spiral arm count"},
    "twist":   {"glsl": "float", "min": -8.0, "max": 8.0, "default": 3.0,
                "description": "winding tightness"},
    "falloff": {"glsl": "float", "min": 0.5, "max": 8.0, "default": 2.6,
                "description": "core brightness falloff"},
    "color_a": {"glsl": "color", "default": "#1b2a6b", "description": "arm color A"},
    "color_b": {"glsl": "color", "default": "#ffd27a", "description": "arm color B"},
    "bg":      {"glsl": "color", "default": "#050308", "description": "background"},
})