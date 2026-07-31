"""nematic_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("nematic_display",
          "Active-nematic display: director-hue schlieren + order brightness + defect glow (node 99)",
          "procedural", '''
float _dir(vec4 q) { return 0.5 * atan(2.0 * q.g, q.r + 1e-6); }
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float Qxx = s.r, Qxy = s.g;
    float S = 2.0 * sqrt(Qxx * Qxx + Qxy * Qxy);
    float theta = _dir(s);
    float hue = fract(theta / 3.14159265 * 2.0);     // 2 cycles per π (nematic)
    float val = clamp(abs(S) * 1.5 + 0.2, 0.2, 1.0);
    float sat = clamp(abs(S) * 2.0 + 0.2, 0.2, 1.0);
    float phi = hue * 6.2831853;
    vec3 col = 0.5 + 0.5 * vec3(cos(phi), cos(phi - 2.094), cos(phi + 2.094));
    col = (1.0 - sat) * 0.3 + sat * col;
    col *= val;
    // Defect glow: wrapped director-gradient magnitude (bend) → warm cores.
    float tl = _dir(texture(u_texture, v_uv + vec2(-texel.x, 0.0)));
    float tr = _dir(texture(u_texture, v_uv + vec2( texel.x, 0.0)));
    float tt = _dir(texture(u_texture, v_uv + vec2(0.0,  texel.y)));
    float tb = _dir(texture(u_texture, v_uv + vec2(0.0, -texel.y)));
    float bend = sqrt(sin(tr - tl) * sin(tr - tl) + sin(tt - tb) * sin(tt - tb));
    float glow = clamp((bend - 0.3) * 2.0, 0.0, 1.0);
    col = clamp(col + glow * vec3(1.0, 0.85, 0.3) * 0.5, 0.0, 1.0);
    f_color = vec4(col, 1.0);
}
''')