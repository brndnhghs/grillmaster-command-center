"""contours_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("contours_gpu", "Marching Squares Contours (client-GPU twin of node 441)", "procedural",
'''void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    float t = u_time;

    // scalar field: fbm noise blended with a radial wave
    vec2 flow = vec2(t * u_flow_amp, -t * u_flow_amp * 0.5);
    float n = fbm(uv * 3.0 + flow);
    float radial = 0.5 + 0.5 * sin(length(uv) * 6.0 - t * 0.3);
    float field = mix(n, radial, 0.4) * u_noise_amp + (1.0 - u_noise_amp) * n;
    field = clamp(field, 0.0, 1.0);

    int N = int(clamp(u_n_levels, 3.0, 24.0));
    float lv = field * float(N);
    float f = fract(lv);
    float d = min(f, 1.0 - f);                 // distance to nearest iso-level
    float w = fwidth(lv) * 1.5 + 0.015;
    float line = 1.0 - smoothstep(0.0, w, d);

    // faint reference grid (grid_step ~ pixels per cell)
    vec2 g = abs(fract(uv * (10.0 / max(u_grid_step, 1.0))) - 0.5);
    float grid = 1.0 - smoothstep(0.0, 0.04, min(g.x, g.y));
    line = max(line * u_line_alpha, grid * 0.12);

    // color by level (level mode)
    float lev = floor(lv) / float(N);
    vec3 cmap = 0.5 + 0.5 * cos(6.2831853 * (lev + vec3(0.0, 0.33, 0.67)));
    vec3 col = mix(vec3(0.05, 0.06, 0.08), cmap, line);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
uniforms={
    "n_levels": {"glsl": "float", "min": 3.0, "max": 24.0, "default": 10.0, "description": "number of contour levels"},
    "grid_step": {"glsl": "float", "min": 2.0, "max": 16.0, "default": 5.0, "description": "reference grid cell size"},
    "line_alpha": {"glsl": "float", "min": 0.1, "max": 1.0, "default": 0.9, "description": "contour line opacity"},
    "flow_amp": {"glsl": "float", "min": 0.0, "max": 0.5, "default": 0.2, "description": "animated flow amplitude"},
    "noise_amp": {"glsl": "float", "min": 0.1, "max": 1.0, "default": 0.6, "description": "noise contribution to field"},
})