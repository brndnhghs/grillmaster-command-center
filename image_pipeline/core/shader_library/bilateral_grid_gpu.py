"""bilateral_grid_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("bilateral_grid_gpu", "Edge-preserving bilateral smoothing (typed twin of node 345)",
          "filter", '''
void main() {
    vec3 center = texture(u_texture, v_uv).rgb;
    vec2 px = 1.0 / u_resolution;
    float R = max(u_sigma_s, 0.5) * max(u_grid_scale, 1.0);
    const int N = 24;
    float ga = 2.39996323;
    float invR2 = 1.0 / (2.0 * max(u_sigma_r, 0.5) * max(u_sigma_r, 0.5));
    float sp2 = max(R * R * 0.25, 1.0);
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    // Joint bilateral: weight neighbours by BOTH spatial Gaussian and range
    // (color) similarity to the center, so smooth regions melt while silhouettes
    // survive. A genuine single-pass approximation of the bilateral grid.
    for (int i = 0; i < N; i++) {
        float fi = float(i);
        float rad = sqrt((fi + 0.5) / float(N)) * R;
        float ang = fi * ga;
        vec2 off = vec2(cos(ang), sin(ang)) * rad;
        vec3 s = texture(u_texture, v_uv + off * px).rgb;
        float ws = exp(-(rad * rad) / sp2);
        float dc = distance(s, center);
        float wr = exp(-(dc * dc) * invR2);
        float w = ws * wr;
        acc += s * w;
        wsum += w;
    }
    vec3 bilat = acc / max(wsum, 0.001);
    vec3 col = mix(bilat, center, u_blend);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "grid_scale": {"glsl": "float", "min": 1.0, "max": 8.0, "default": 4.0,
                   "description": "spatial cell size in px (smoother when larger)"},
    "sigma_s":    {"glsl": "float", "min": 0.5, "max": 8.0, "default": 2.0,
                   "description": "spatial blur radius in grid cells"},
    "sigma_r":    {"glsl": "float", "min": 0.5, "max": 8.0, "default": 2.0,
                   "description": "range (intensity) blur radius — smaller = sharper edges"},
    "blend":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                   "description": "blend original source back in (1=original)"},
})