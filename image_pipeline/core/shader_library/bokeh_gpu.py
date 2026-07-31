"""bokeh_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("bokeh_gpu", "Bokeh lens blur with shaped aperture (typed twin of node 420)",
          "filter", '''
void main() {
    vec3 src = texture(u_texture, v_uv).rgb;
    vec2 px = 1.0 / u_resolution;
    float R = max(u_radius, 1.0);
    const int N = 24;
    float ga = 2.39996323;
    float rot_ang = radians(u_rotation);
    mat2 ROT = rot(rot_ang);
    float seg = 3.14159265 / max(u_blades, 3.0);
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    // Disc sampling weighted by a regular-N-gon aperture SDF (the iris shape);
    // horizontal anamorphic stretch bakes the cinematic streak into highlights.
    for (int i = 0; i < N; i++) {
        float fi = float(i);
        float rad = sqrt((fi + 0.5) / float(N)) * R;
        float ang = fi * ga;
        vec2 off = vec2(cos(ang) * u_anamorphic, sin(ang)) * rad;
        vec2 pr = ROT * off;
        float a = atan(pr.y, pr.x);
        float rr = length(pr);
        float m = mod(a + 3.14159265, 2.0 * seg) - seg;
        float edge = R * cos(seg - abs(m));
        float w = (rr <= edge) ? 1.0 : 0.0;
        vec3 s = texture(u_texture, v_uv + off * px).rgb;
        acc += s * w;
        wsum += w;
    }
    vec3 blurred = acc / max(wsum, 0.001);
    float lum = dot(src, vec3(0.2126, 0.7152, 0.0722));
    vec3 hot = src * smoothstep(0.6, 1.0, lum) * u_highlight;
    vec3 col = (blurred + hot) * u_brightness;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "radius":     {"glsl": "float", "min": 2.0, "max": 48.0, "default": 16.0,
                   "description": "bokeh radius in px (defocus amount)"},
    "blades":     {"glsl": "float", "min": 3.0, "max": 12.0, "default": 6.0,
                   "description": "iris blade count (polygon sides)"},
    "anamorphic": {"glsl": "float", "min": 1.0, "max": 8.0, "default": 4.0,
                   "description": "horizontal streak stretch"},
    "rotation":   {"glsl": "float", "min": 0.0, "max": 360.0, "default": 0.0,
                   "description": "aperture rotation (deg)"},
    "brightness": {"glsl": "float", "min": 0.2, "max": 2.5, "default": 1.0,
                   "description": "output brightness gain"},
    "highlight":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.35,
                   "description": "re-add hot cores to out-of-focus lights"},
})