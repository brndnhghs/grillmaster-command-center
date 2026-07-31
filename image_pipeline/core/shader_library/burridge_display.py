"""burridge_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("burridge_display",
          "Burridge-Knopoff display (tectonic): stress field + edge-detected crack lines (grayscale)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    float st = texture(u_texture, v_uv).r;
    // Contrast-stretched stress background (matches CPU tectonic: (s-0.2)/0.6 ^0.8).
    float s = clamp(st, 0.0, 1.0);
    float ss = clamp((s - 0.2) / 0.6, 0.0, 1.0);
    float bg = pow(ss, 0.8) * 0.59 + 0.08;
    // 4-directional stress gradient → bright crack edges.
    float c  = st;
    float gl = abs(texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r - c);
    float gr = abs(texture(u_texture, v_uv + vec2( texel.x, 0.0)).r - c);
    float gu = abs(texture(u_texture, v_uv + vec2(0.0,  texel.y)).r - c);
    float gd = abs(texture(u_texture, v_uv + vec2(0.0, -texel.y)).r - c);
    float grad = max(max(gl, gr), max(gu, gd));
    float edges = clamp(grad * 3.0, 0.0, 1.0) * 0.86;
    // Faint permanent damage scars.
    float dmg = texture(u_texture, v_uv).g;
    float scar = clamp(log(1.0 + dmg) / 3.0, 0.0, 1.0) * 0.08;
    float g = max(max(bg, edges), scar);
    f_color = vec4(vec3(g), 1.0);
}
''')