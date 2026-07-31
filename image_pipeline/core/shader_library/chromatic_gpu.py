"""chromatic_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("chromatic_gpu", "Chromatic aberration RGB split of the input (typed)",
          "filter", '''
void main() {
    vec2 uv = v_uv;
    vec2 dir = (uv - 0.5);
    float amt = u_amount * 0.05;
    float ph = u_time * u_pulse;
    float k = amt * (1.0 + 0.3 * sin(ph));
    float rC = texture(u_texture, uv + dir * k).r;
    float gC = texture(u_texture, uv).g;
    float bC = texture(u_texture, uv - dir * k).b;
    f_color = vec4(rC, gC, bC, 1.0);
}
''', uniforms={
    "amount": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.4,
               "description": "aberration amount"},
    "pulse":  {"glsl": "float", "min": 0.0, "max": 4.0, "default": 0.0,
               "description": "animated pulse speed"},
})