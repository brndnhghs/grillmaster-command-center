"""dbm_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("dbm_step",
          "Dielectric Breakdown step: relax Laplace potential, grow frontier proportional to |grad(phi)|^eta (node 106 twin)",
          "procedural", '''
float dbmNbPhi(vec2 off) {
    vec4 n = texture(u_texture, v_uv + off);
    if (n.g > 0.5) return 1.0;    // tree electrode
    if (n.g < -0.5) return 0.0;   // far-field boundary
    return n.r;
}

void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float occ = s.g;
    float temp = s.b;

    float eta  = u_params.x;
    float grow = u_params.y;   // growth_rate
    float cool = u_params.z;   // cool_rate
    float diel = u_params.w;   // dielectric strength

    // Far-field boundary: frozen.
    if (occ < -0.5) { f_color = vec4(0.0, -1.0, 0.0, 1.0); return; }
    // Tree electrode: hold phi=1, cool the temperature each step.
    if (occ > 0.5)  { f_color = vec4(1.0, 1.0, temp * cool, 1.0); return; }

    // One Jacobi relaxation sweep of the harmonic potential.
    float pl = dbmNbPhi(vec2(-texel.x, 0.0));
    float pr = dbmNbPhi(vec2( texel.x, 0.0));
    float pd = dbmNbPhi(vec2(0.0, -texel.y));
    float pu = dbmNbPhi(vec2(0.0,  texel.y));
    float phiNew = 0.25 * (pl + pr + pu + pd);

    // Gradient magnitude of the potential at this cell (tip vs flat front).
    float gx = pr - pl;
    float gy = pu - pd;
    float grad = length(vec2(gx, gy)) * 0.5;

    bool frontier = (pl > 0.5 || pr > 0.5 || pu > 0.5 || pd > 0.5);

    float newOcc = 0.0;
    float newTemp = 0.0;
    if (frontier) {
        // Per-cell dielectric weakness (stable hash) — weak spots grow easier.
        float weak = hash21(floor(v_uv * u_resolution) + 3.17);
        float dieMul = mix(1.0, weak, diel);
        float w = pow(max(grad, 0.0), eta);
        float prob = clamp(w * grow * 0.06 * dieMul, 0.0, 1.0);
        float rng = hash21(floor(v_uv * u_resolution) + 0.5 + phiNew * 53.0);
        if (rng < prob) { newOcc = 1.0; newTemp = 1.0; }
    }

    f_color = vec4(phiNew, newOcc, newTemp, 1.0);
}
''')