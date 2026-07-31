"""fxaa_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_shader



# 350 FXAA Anti-Aliasing — Fast Approximate Anti-Aliasing (Lottes, NVIDIA
# 2009/2011). Real-time screen-space AA: 3x3 luma neighbourhood, edge early-out,
# edge-tangent direction blend (the learnopengl normalize variant, matched by
# the CPU node 350 export). u_params.x = edge_threshold with 0.5 = neutral
# medium strength (pitfall #15: 0.5 must not be a degenerate extreme).
_register("fxaa_gpu", "GPU FXAA anti-aliasing (edge-tangent luma blend)", "filter", _filter_shader('''
    vec3 rgbNW = texture(u_texture, uv + vec2(-step.x,  step.y)).rgb;
    vec3 rgbNE = texture(u_texture, uv + vec2( step.x,  step.y)).rgb;
    vec3 rgbSW = texture(u_texture, uv + vec2(-step.x, -step.y)).rgb;
    vec3 rgbSE = texture(u_texture, uv + vec2( step.x, -step.y)).rgb;
    vec3 rgbM  = orig.rgb;
    float lNW = dot(rgbNW, vec3(0.299, 0.587, 0.114));
    float lNE = dot(rgbNE, vec3(0.299, 0.587, 0.114));
    float lSW = dot(rgbSW, vec3(0.299, 0.587, 0.114));
    float lSE = dot(rgbSE, vec3(0.299, 0.587, 0.114));
    float lM  = dot(rgbM,  vec3(0.299, 0.587, 0.114));

    float lumaMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
    float lumaMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));

    float et = 0.04 + u_params.x * 0.46;            // edge_threshold: 0.04..0.50
    if ((lumaMax - lumaMin) < max(0.001, lumaMax * et)) {
        f_color = orig;                            // flat region — pass through
        return;
    }

    float dirX = -lNW - lNE + lSW + lSE;
    float dirY = -lNW - lSW + lNE + lSE;
    float dirL = max(1e-6, sqrt(dirX * dirX + dirY * dirY));
    vec2 dir = vec2(dirX, dirY) / dirL;

    vec3 rgbA = (texture(u_texture, uv + dir * (1.0/6.0) * step).rgb +
                 texture(u_texture, uv - dir * (1.0/6.0) * step).rgb) * 0.5;
    vec3 rgbB = (rgbA * 0.5) + (texture(u_texture, uv + dir * (1.0/2.0) * step).rgb +
                                texture(u_texture, uv - dir * (1.0/2.0) * step).rgb) * 0.25;
    float lB = dot(rgbB, vec3(0.299, 0.587, 0.114));
    vec3 rgb = ((lB < lumaMin) || (lB > lumaMax)) ? rgbA : rgbB;   // anti-ringing clamp
    f_color = vec4(rgb, 1.0);
'''))