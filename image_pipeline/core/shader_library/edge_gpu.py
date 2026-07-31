"""edge_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("edge_gpu", "Sobel edge detection on the input (typed)",
          "filter", '''
float _lum(vec2 uv) {
    return dot(texture(u_texture, uv).rgb, vec3(0.299, 0.587, 0.114));
}
void main() {
    vec2 px = u_thickness / u_resolution;
    float tl = _lum(v_uv + px * vec2(-1.0,  1.0));
    float  l = _lum(v_uv + px * vec2(-1.0,  0.0));
    float bl = _lum(v_uv + px * vec2(-1.0, -1.0));
    float  t = _lum(v_uv + px * vec2( 0.0,  1.0));
    float  b = _lum(v_uv + px * vec2( 0.0, -1.0));
    float tr = _lum(v_uv + px * vec2( 1.0,  1.0));
    float  r = _lum(v_uv + px * vec2( 1.0,  0.0));
    float br = _lum(v_uv + px * vec2( 1.0, -1.0));
    float gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
    float gy =  tl + 2.0 * t + tr - bl - 2.0 * b - br;
    float e = clamp(length(vec2(gx, gy)) * u_strength, 0.0, 1.0);
    vec3 col = mix(u_bg, u_edge, e);   // edges in u_edge over u_bg
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "strength":   {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.5,
                   "description": "edge gain"},
    "thickness":  {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.5,
                   "description": "edge kernel thickness (px)"},
    "bg":         {"glsl": "color", "default": "#000000", "description": "background color"},
    "edge":       {"glsl": "color", "default": "#39ff88", "description": "edge color"},
})