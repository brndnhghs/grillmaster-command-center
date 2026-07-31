"""fhn_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# FitzHugh-Nagumo (node 133): du = (u - u^3/3 - v)/e + Du*Lap(u);
# dv = e*(u + a - b*v) + Dv*Lap(v). p1=e, p2=a, p3=b, p4=Du.
_register("fhn_step",
          "FitzHugh-Nagumo step (5-pt toroidal Laplacian, excitable media)",
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
    float e = max(u_params.x, 1e-3), a = u_params.y, b = u_params.z, Du = u_params.w;
    float Dv = 0.0;
    float nU = U + 0.08 * ((U - U*U*U/3.0 - V)/e + Du*lu);
    float nV = V + 0.08 * (e*(U + a - b*V) + Dv*lv);
    f_color = vec4(clamp(nU,-1.0,1.0), clamp(nV,-1.0,1.0), 0.0, 1.0);
}
''')