"""siren_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("siren_gpu", "SIREN Field (client-GPU twin of node 512)", "procedural",
'''void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    float t = u_time;
    vec2 p = uv * u_coord_scale;
    float v = 0.0;
    for (int k = 0; k < 6; k++) {
        float fk = float(k);
        vec2 w = vec2(cos(fk * 1.7), sin(fk * 2.3));
        float b = fk * 0.9;
        float om = mix(u_omega0, u_omega, fract(fk * 0.37));
        float s = sin(om * dot(w, p + 0.08 * t) + b + t * 0.3);
        v += s * (0.5 + 0.5 * sin(fk * 1.1));
    }
    v *= u_weight_scale / 6.0;
    v = v * 0.5 + 0.5;
    // cosine palette (Inigo Quilez) — no dependency on late helpers
    vec3 cmap = 0.5 + 0.5 * cos(6.2831853 * (v + vec3(0.0, 0.33, 0.67)));
    f_color = vec4(clamp(cmap, 0.0, 1.0), 1.0);
}
''',
uniforms={
    "omega0": {"glsl": "float", "min": 1.0, "max": 60.0, "default": 30.0, "description": "base frequency"},
    "omega": {"glsl": "float", "min": 1.0, "max": 60.0, "default": 30.0, "description": "top frequency"},
    "weight_scale": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0, "description": "output gain"},
    "coord_scale": {"glsl": "float", "min": 0.5, "max": 12.0, "default": 3.0, "description": "coordinate scale"},
})