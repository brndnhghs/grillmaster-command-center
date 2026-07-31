"""autostereogram_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("autostereogram_gpu", "Autostereogram (client-GPU twin of node 954)", "procedural",
'''void main() {
    vec2 res = u_resolution;
    float px = gl_FragCoord.x;
    float py = gl_FragCoord.y;
    vec2 uv = (vec2(px, py) - 0.5 * res) / min(res.x, res.y);
    float t = u_time;
    float r2 = dot(uv, uv);
    float depth = clamp(1.0 - r2 * 2.5 + 0.06 * sin(t + py * 0.01), 0.0, 1.0);
    depth *= u_depth_scale;
    float shift = depth * u_separation;
    float sx = px - shift;
    vec2 gp = vec2(sx, py) / u_tile_size;
    vec2 cell = fract(gp) - 0.5;
    float dotm = smoothstep(0.35, 0.3, length(cell));
    vec3 base = vec3(0.82);
    vec3 col = mix(base, vec3(0.08), dotm);
    f_color = vec4(col, 1.0);
}
''',
uniforms={
    "separation": {"glsl": "float", "min": 4.0, "max": 80.0, "default": 40.0, "description": "stereo separation (px)"},
    "depth_scale": {"glsl": "float", "min": 0.1, "max": 1.5, "default": 1.0, "description": "depth relief"},
    "tile_size": {"glsl": "float", "min": 4.0, "max": 48.0, "default": 16.0, "description": "dot tile size (px)"},
})