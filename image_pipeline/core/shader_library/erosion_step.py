"""erosion_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("erosion_step",
          "Hydraulic-erosion step: rain + surface-gradient flow + stream-power erosion/deposition (node 156)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 c = texture(u_texture, v_uv);
    float h = c.r, w = c.g, s = c.b;
    vec4 cl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 cr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 cu = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 cd = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float rain  = clamp(u_params.x, 0.0, 0.05);
    float K_e   = clamp(u_params.y, 0.001, 0.5);
    float K_d   = clamp(u_params.z, 0.01, 1.0);
    float theta = clamp(u_params.w, 0.02, 0.5);
    // Rainfall.
    w += rain;
    // Water flows to lower total surface (h+w): Laplacian of the surface pools
    // water into valleys and drains it off ridges.
    float surfC = h + w;
    float lapSurf = (cl.r + cl.g) + (cr.r + cr.g) + (cu.r + cu.g) + (cd.r + cd.g) - 4.0 * surfC;
    w = max(w + 0.25 * lapSurf, 0.0);
    // Terrain slope → stream-power erosion; sediment picked up.
    float dx = (cr.r - cl.r) * 0.5, dy = (cu.r - cd.r) * 0.5;
    float slope = sqrt(dx * dx + dy * dy);
    float erode = K_e * w * slope;
    h -= erode; s += erode;
    // Deposition where flow slackens (low slope).
    float dep = K_d * s * (1.0 - clamp(slope * 6.0, 0.0, 1.0));
    h += dep; s = max(s - dep, 0.0);
    // Thermal creep toward the angle of repose.
    float lapH = cl.r + cr.r + cu.r + cd.r - 4.0 * h;
    h += theta * 0.15 * lapH;
    // Evaporation + tiny continuous noise (keeps drainage networks reorganizing).
    w *= 0.97;
    h += (hash21(v_uv * 331.0 + vec2(h, w) * 57.0) - 0.5) * 0.003;
    f_color = vec4(h, w, s, 1.0);
}
''')