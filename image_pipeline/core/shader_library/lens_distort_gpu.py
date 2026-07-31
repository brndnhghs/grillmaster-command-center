"""lens_distort_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("lens_distort_gpu", "Lens Distortion — Brown–Conrady radial (barrel/pincushion) + chromatic split (client-GPU twin of node 480)",
          "filter", '''
void main() {
    vec2 ctr = vec2(u_center_x, u_center_y);
    vec2 d = v_uv - ctr;
    d.x *= u_aspect;
    float r = length(d);
    float r2 = r * r;
    float rd = r * (1.0 + u_amount * r2 + u_k2 * r2 * r2);
    vec2 dir = d / max(r, 1e-4);
    vec2 base = ctr + (dir * rd) / max(u_aspect, 1e-4);
    vec3 col = texture(u_texture, clamp(base, 0.0, 1.0)).rgb;
    if (u_chromatic > 0.0) {
        float k = u_chromatic * 0.02 * rd;
        float rR = texture(u_texture, clamp(ctr + (dir * (rd + k)) / max(u_aspect, 1e-4), 0.0, 1.0)).r;
        float bB = texture(u_texture, clamp(ctr + (dir * (rd - k)) / max(u_aspect, 1e-4), 0.0, 1.0)).b;
        col = vec3(rR, col.g, bB);
    }
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "amount":     {"glsl": "float", "min": -0.6, "max": 0.6, "default": 0.25,
                   "description": "Brown–Conrady k1 radial distortion (barrel<0 / pincushion>0)"},
    "k2":         {"glsl": "float", "min": -0.3, "max": 0.3, "default": 0.0,
                   "description": "higher-order k2 radial term"},
    "center_x":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "distortion centre X (uv)"},
    "center_y":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "distortion centre Y (uv)"},
    "aspect":     {"glsl": "float", "min": 0.3, "max": 3.0, "default": 1.0,
                   "description": "aspect correction (elliptical distortion)"},
    "chromatic":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                   "description": "radial chromatic-aberration split"},
})