"""cel_shading_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# ── 462 Cel Shading (banded Lambert + Fresnel rim + outline) ──
_register("cel_shading_gpu", "Cel Shading (client-GPU twin of node 462)", "filter", _filter_typed('''
    float l0 = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float lx = dot(texture(u_texture, uv + vec2(step.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
    float ly = dot(texture(u_texture, uv + vec2(0.0, step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float relief = 40.0;
    float gx = (lx - l0) * relief;
    float gy = (ly - l0) * relief;
    vec3 N = normalize(vec3(-gx, -gy, 1.0));
    float az = radians(u_light_azimuth);
    float el = radians(u_light_elevation);
    vec3 L = normalize(vec3(cos(el) * cos(az), cos(el) * sin(az), sin(el)));
    float ndl = clamp(dot(N, L), 0.0, 1.0);
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 halfv = normalize(L + V);
    float ndh = clamp(dot(N, halfv), 0.0, 1.0);
    float spec_disc = (ndh > u_spec_threshold ? 1.0 : 0.0) * u_specular;   // hard toon-specular disc
    float bands = max(2.0, floor(u_bands + 0.5));
    float lit = clamp(floor(ndl * bands) / max(1.0, bands - 1.0), 0.0, 1.0);
    float ambient = 0.18;
    vec3 albedo = 0.5 + 0.5 * cos(6.2831853 * (u_base_hue + vec3(0.0, 0.33, 0.67)));
    float shade = ambient + (1.0 - ambient) * lit;
    vec3 outc = albedo * shade + spec_disc;
    float rimf = pow(clamp(1.0 - N.z, 0.0, 1.0), 2.0);
    vec3 rim_col = 0.5 + 0.5 * cos(6.2831853 * ((u_base_hue + 0.5) + vec3(0.0, 0.33, 0.67)));
    outc += rimf * u_rim * rim_col;
    float edge = sqrt(gx * gx + gy * gy);
    float omask = edge > u_outline ? 1.0 : 0.0;
    outc *= (1.0 - 0.75 * omask);
    f_color = vec4(clamp(outc, 0.0, 1.0), 1.0);
'''), uniforms={
    "light_azimuth": {"glsl": "float", "min": 0.0, "max": 360.0, "default": 135.0, "description": "light azimuth (deg)"},
    "light_elevation": {"glsl": "float", "min": 5.0, "max": 85.0, "default": 45.0, "description": "light elevation (deg)"},
    "bands": {"glsl": "float", "min": 2.0, "max": 8.0, "default": 4.0, "description": "toon light levels"},
    "specular": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.7, "description": "toon specular disc intensity"},
    "spec_threshold": {"glsl": "float", "min": 0.30, "max": 0.95, "default": 0.75, "description": "specular disc half-vector threshold (higher=smaller disc)"},
    "rim": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.6, "description": "Fresnel rim strength"},
    "outline": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.22, "description": "outline slope threshold"},
    "base_hue": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.58, "description": "albedo base hue"},
})