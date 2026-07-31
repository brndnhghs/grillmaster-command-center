"""lv3_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("lv3_step",
          "3-species Lotka-Volterra step (5-pt toroidal Laplacian on 3 channels)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float U = s.r, V = s.g, W = s.b;
    float lu = texture(u_texture, v_uv + vec2(-texel.x,0.0)).r
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).r
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0*U;
    float lv = texture(u_texture, v_uv + vec2(-texel.x,0.0)).g
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).g
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).g
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).g - 4.0*V;
    float lw = texture(u_texture, v_uv + vec2(-texel.x,0.0)).b
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).b
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).b
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).b - 4.0*W;
    float k1 = u_params.x, k2 = u_params.y, k3 = u_params.z, k4 = u_params.w;
    float nU = U + 0.15 * (U - k1*U*V + 0.08*lu);
    float nV = V + 0.15 * (k2*U*V - k3*V*W + 0.08*lv);
    float nW = W + 0.15 * (k4*V*W - W + 0.08*lw);
    f_color = vec4(clamp(nU,0.0,1.0), clamp(nV,0.0,1.0), clamp(nW,0.0,1.0), 1.0);
}
''')