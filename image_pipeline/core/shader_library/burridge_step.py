"""burridge_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("burridge_step",
          "Burridge-Knopoff one step: load + threshold slip + 4-neighbor stress redistribution (toroidal)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float stress   = s.r;
    float damage   = s.g;
    float strength = s.b;

    float rate  = clamp(u_params.x, 0.001, 0.1);
    float thr   = clamp(u_params.y, 0.5, 5.0);
    float resid = clamp(u_params.z, 0.0, 0.5);
    float alpha = clamp(u_params.w, 0.0, 0.25);

    // Slow tectonic loading + tiny per-cell noise (breaks symmetry / nucleates).
    float nz = (hash21(v_uv * 311.7 + fract(stress * 53.13)) - 0.5) * 0.008;
    stress += rate + nz;

    // Neighbor slip stress: a neighbor over its own effective threshold released
    // its stress; we receive alpha * that released stress from each of 4 sides.
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 su = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sd = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float rel_l = (sl.r > thr * sl.b) ? sl.r : 0.0;
    float rel_r = (sr.r > thr * sr.b) ? sr.r : 0.0;
    float rel_u = (su.r > thr * su.b) ? su.r : 0.0;
    float rel_d = (sd.r > thr * sd.b) ? sd.r : 0.0;
    stress += alpha * (rel_l + rel_r + rel_u + rel_d);

    // This block's own slip: if over effective threshold, reset to residual and
    // record a damage event (permanent scar for the fracture render).
    float eff = thr * strength;
    bool over = stress > eff;
    float new_stress = over ? resid : stress;
    float new_damage = damage + (over ? 1.0 : 0.0);

    f_color = vec4(clamp(new_stress, 0.0, 8.0), new_damage, strength, 1.0);
}
''')