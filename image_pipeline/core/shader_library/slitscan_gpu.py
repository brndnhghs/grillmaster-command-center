"""slitscan_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("slitscan_gpu",
          "Slit-scan displacement (client-GPU twin of node 57)",
          "procedural", '''
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
void main() {
    // u_params.x = amplitude (0.5 -> ~0.275), u_params.y = frequency
    // (0.5 -> ~0.25), u_params.z = slit_type (0=vertical,1=horizontal,
    // 2=radial,3=spiral,4=angular,5=diagonal).
    int mode = int(floor(u_slit_type * 6.999));
    float amp  = 0.05 + u_amplitude * 0.45;
    float freq = 0.005 + u_frequency * 0.495;
    float t = u_time * 0.05;

    vec2 uv = v_uv;
    vec2 c = uv - 0.5;
    float r = length(c);
    float a = atan(c.y, c.x);

    float disp;
    if (mode == 1)      disp = sin(freq * uv.y * 40.0 + t);
    else if (mode == 2) disp = sin(freq * r * 40.0 - t);
    else if (mode == 3) disp = sin(freq * r * 40.0 + a * 4.0 + t);
    else if (mode == 4) disp = sin(freq * a * 6.0 + t);
    else if (mode == 5) disp = sin(freq * (uv.x + uv.y) * 40.0 + t);
    else                disp = sin(freq * uv.x * 40.0 + t);

    vec2 suv = fract(uv + amp * disp);
    float n = fbm(suv * 5.0);
    float hue = fract(n + t * 0.1);
    vec3 col = mix(vec3(n), hsv2rgb(vec3(hue, 0.7, 0.9)), 0.5);
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
    "amplitude": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "displacement amplitude"},
    "frequency": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "slit frequency"},
    "slit_type": {"glsl": "float", "min": 0.0, "max": 5.0, "default": 0.0, "description": "slit pattern (0-5)"}
}
    )