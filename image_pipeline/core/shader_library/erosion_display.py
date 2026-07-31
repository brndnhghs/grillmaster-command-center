"""erosion_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("erosion_display",
          "Hydraulic-erosion display: grayscale hillshade + water-channel brightening (node 156)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 c = texture(u_texture, v_uv);
    float h = c.r, w = c.g;
    float hL = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float hR = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float hU = texture(u_texture, v_uv + vec2(0.0,  texel.y)).r;
    float hD = texture(u_texture, v_uv + vec2(0.0, -texel.y)).r;
    float dx = (hR - hL) * 0.5, dy = (hU - hD) * 0.5;
    float slope  = atan(sqrt(dx * dx + dy * dy) * 8.0);
    float aspect = atan(dy, -dx);
    float az = radians(315.0), alt = radians(45.0);
    float shade = clamp(sin(alt) * cos(slope) + cos(alt) * sin(slope) * cos(az - aspect), 0.0, 1.0);
    float hn = clamp(h * 0.7 + 0.5, 0.0, 1.0);
    float g = 0.55 * shade + 0.35 * hn;
    float wn = clamp(w * 6.0, 0.0, 1.0);          // water channels brighten
    g = clamp(max(g, wn * 0.3) + wn * 0.15, 0.0, 1.0);
    f_color = vec4(g, g, g, 1.0);
}
''')