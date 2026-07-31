"""dendrite_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("dendrite_display",
          "Dendritic display: φ → grayscale + thin interface outline (node 122)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    float phi = texture(u_texture, v_uv).r;
    float pl = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float pr = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float pt = texture(u_texture, v_uv + vec2(0.0,  texel.y)).r;
    float pb = texture(u_texture, v_uv + vec2(0.0, -texel.y)).r;
    float pa = texture(u_texture, v_uv + vec2(-texel.x,  texel.y)).r;
    float pc = texture(u_texture, v_uv + vec2( texel.x,  texel.y)).r;
    float pe = texture(u_texture, v_uv + vec2(-texel.x, -texel.y)).r;
    float pf = texture(u_texture, v_uv + vec2( texel.x, -texel.y)).r;
    // 3×3 Gaussian-ish smooth of φ (suppresses the explicit-scheme checkerboard,
    // mirroring the CPU node's light periodic blur).
    float phiS = (phi * 4.0 + (pl + pr + pt + pb) * 2.0 + (pa + pc + pe + pf)) / 16.0;
    float gray = (phiS + 1.0) * 0.5;
    float gmag = length(vec2((pr - pl) * 0.5, (pt - pb) * 0.5));
    if (abs(phi) < 0.2 && gmag > 0.15) gray = 1.0;   // thin interface outline
    f_color = vec4(gray, gray, gray, 1.0);
}
''')