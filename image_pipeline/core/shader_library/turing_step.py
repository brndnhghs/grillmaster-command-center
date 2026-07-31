"""turing_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# Turing / Schnakenberg (node 169): ru = g*(a - u + u^2 v); rv = g*(b - u^2 v).
# Diffusion Du, Dv; p1=a, p2=b, p3=g, p4=Du.
_register("turing_step",
          "Schnakenberg/Turing RD step (5-pt toroidal Laplacian)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float U = s.r, V = s.g;
    float lu = texture(u_texture, v_uv + vec2(-texel.x,0.0)).r
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).r
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0*U;
    float lv = texture(u_texture, v_uv + vec2(-texel.x,0.0)).g
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).g
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).g
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).g - 4.0*V;
    float a = u_params.x, b = u_params.y, g = u_params.z, Du = u_params.w;
    float Dv = 0.5;
    float u2v = U*U*V;
    float nU = U + 0.02 * (g*(a - U + u2v) + Du*lu);
    float nV = V + 0.02 * (g*(b - u2v) + Dv*lv);
    f_color = vec4(clamp(nU,0.0,1.0), clamp(nV,0.0,1.0), 0.0, 1.0);
}
''')