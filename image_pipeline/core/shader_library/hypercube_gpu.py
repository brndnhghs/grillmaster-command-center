"""hypercube_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("hypercube_gpu", "4D Hypercube (client-GPU twin of node 108)", "procedural",
'''float _hc_pc(int x) {
    int c = 0;
    for (int k = 0; k < 4; k++) { c += (x >> k) & 1; }
    return float(c);
}
float _hc_distSeg(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    float t = u_time;

    float a1 = t * (0.2 + u_speed_xw * 0.5);
    float a2 = t * (0.2 + u_speed_yw * 0.5);
    float cx = cos(a1), sx = sin(a1);
    float cy = cos(a2), sy = sin(a2);

    vec2 pos[16];
    vec3 colw[16];
    for (int i = 0; i < 16; i++) {
        float x = ((i >> 0) & 1) == 1 ? 1.0 : -1.0;
        float y = ((i >> 1) & 1) == 1 ? 1.0 : -1.0;
        float z = ((i >> 2) & 1) == 1 ? 1.0 : -1.0;
        float w = ((i >> 3) & 1) == 1 ? 1.0 : -1.0;
        float rx = x * cx - w * sx;
        float rw = x * sx + w * cx;
        float ry = y * cy - z * sy;
        float rz = y * sy + z * cy;
        float k = 1.0 / (u_proj_radius - rw);
        vec3 p3 = vec3(rx, ry, rz) * k * u_proj_radius * 0.32;
        pos[i] = p3.xy;
        float hue = 0.5 * (rw + 1.0);
        colw[i] = vec3(0.5 + 0.5 * cos(6.2831853 * (u_inner_hue + hue)),
                       0.5 + 0.5 * cos(6.2831853 * (u_outer_hue + hue + 0.33)),
                       0.5 + 0.5 * cos(6.2831853 * (u_inner_hue + hue + 0.67)));
    }

    float dmin = 1e9;
    vec3 edgeCol = vec3(0.0);
    for (int i = 0; i < 16; i++) {
        for (int j = i + 1; j < 16; j++) {
            if (_hc_pc(i ^ j) == 1.0) {
                float d = _hc_distSeg(uv, pos[i], pos[j]);
                if (d < dmin) { dmin = d; edgeCol = 0.5 * (colw[i] + colw[j]); }
            }
        }
    }
    float lw = u_line_width * 0.0025;
    float line = 1.0 - smoothstep(0.0, lw, dmin);
    vec3 col = line * edgeCol + vec3(0.02, 0.02, 0.03) * (1.0 - line);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
uniforms={
    "speed_xw": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 0.5, "description": "XW-plane rotation speed"},
    "speed_yw": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 0.3, "description": "YW-plane rotation speed"},
    "proj_radius": {"glsl": "float", "min": 2.0, "max": 6.0, "default": 3.5, "description": "4D perspective radius"},
    "line_width": {"glsl": "float", "min": 1.0, "max": 4.0, "default": 3.0, "description": "edge thickness"},
    "inner_hue": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.55, "description": "inner vertex hue"},
    "outer_hue": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.08, "description": "outer vertex hue"},
})