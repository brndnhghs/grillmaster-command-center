"""cml_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("cml_step",
          "Coupled logistic one step: logistic map + diffusive coupling + EMA trail",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float x = s.r, accum = s.g;
    float r = clamp(u_params.x, 3.5, 4.0);
    float eps = clamp(u_params.y, 0.05, 0.5);
    float decay = clamp(u_params.z, 0.5, 0.99);
    // f(x) at this cell and its 4 toroidal neighbours
    float fx  = r * x * (1.0 - x);
    float xu  = texture(u_texture, v_uv + vec2(0.0, -texel.y)).r;
    float xd  = texture(u_texture, v_uv + vec2(0.0,  texel.y)).r;
    float xl  = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float xr  = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float fsum = r*xu*(1.0-xu) + r*xd*(1.0-xd) + r*xl*(1.0-xl) + r*xr*(1.0-xr);
    float xn = (1.0 - eps) * fx + (eps * 0.25) * fsum;
    xn = clamp(xn, 0.0, 1.0);
    // Exponential moving-average trail to suppress discrete-time strobing
    float an = decay * accum + (1.0 - decay) * xn;
    f_color = vec4(xn, clamp(an, 0.0, 1.0), 0.0, 1.0);
}
''')