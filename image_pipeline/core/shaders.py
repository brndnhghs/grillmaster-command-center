"""
GPU-accelerated shader pipeline using ModernGL + GLSL fragment shaders.

Two modes:
  - Procedural (method #82): generates imagery from scratch
  - Filter (--filter shader): processes an input image

Runs headlessly on Apple M1 Metal backend (GL 4.1 core profile).
"""

from __future__ import annotations
from pathlib import Path
import math
import random
import re

import numpy as np
from PIL import Image


# ═══════════════════════════════════════════════
#  GL CONTEXT (lazy singleton)
# ═══════════════════════════════════════════════

_CTX = None


def _get_ctx():
    global _CTX
    if _CTX is None:
        import moderngl
        _CTX = moderngl.create_context(standalone=True, require=330)
    return _CTX


# ═══════════════════════════════════════════════
#  QUAD GEOMETRY (full-screen triangle strip)
# ═══════════════════════════════════════════════

_QUAD_VERTICES = np.array([
    -1, -1,  0, 0,
     1, -1,  1, 0,
     1,  1,  1, 1,
    -1,  1,  0, 1,
], dtype='f4')

_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype='i4')

# Shared vertex shader
_VERTEX_SHADER = '''
#version 330
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_uv = in_uv;
}
'''


# ═══════════════════════════════════════════════
#  SHADER LIBRARY
# ═══════════════════════════════════════════════

# Each entry: name, description, type (procedural|filter|both), fragment source

SHADERS = {}

def _register(name: str, description: str, shader_type: str, source: str):
    SHADERS[name] = {
        "name": name,
        "description": description,
        "type": shader_type,
        "source": source,
    }


# ── COMMON PROLOGUE (injected into every shader) ──

_PROLOGUE = '''
#version 330
precision highp float;

in vec2 v_uv;
out vec4 f_color;

uniform vec2 u_resolution;
uniform float u_time;
uniform vec4 u_params;   // xyzw = 4 generic float params
uniform sampler2D u_texture;  // input image (for filter mode)

// 2D rotation
mat2 rot(float a) { float c=cos(a), s=sin(a); return mat2(c,-s,s,c); }

// 2D noise helpers
float hash21(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i); float b = hash21(i + vec2(1, 0));
    float c = hash21(i + vec2(0, 1)); float d = hash21(i + vec2(1, 1));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) {
        v += a * noise(p); p *= 2.0; a *= 0.5;
    }
    return v;
}
'''


# ═══════════════════════════════════════════════
#  REGISTER SHADERS
# ═══════════════════════════════════════════════

# ── PROCEDURAL (generate from scratch) ──

_register("mandelbrot", "Mandelbrot set zoom region", "procedural", '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    float zoom = exp(u_params.x * 3.0);
    vec2 c = vec2(-0.5, 0.0) + uv * zoom;
    vec2 z = vec2(0.0);
    int n = 0;
    for (int i = 0; i < 100; i++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        if (dot(z, z) > 4.0) break;
        n++;
    }
    float t = float(n) / 100.0;
    f_color = vec4(0.5 + 0.5 * cos(t * 6.28 + vec3(0, 2, 4)), 1.0);
}
''')

_register("julia", "Julia set fractal", "procedural", '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    vec2 c = vec2(-0.7269 + u_params.x * 0.1, 0.1889 + u_params.y * 0.1);
    vec2 z = uv * exp(u_params.z * 2.0);
    int n = 0;
    for (int i = 0; i < 100; i++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        if (dot(z, z) > 4.0) break;
        n++;
    }
    float t = float(n) / 100.0;
    f_color = vec4(0.5 + 0.5 * cos(t * 6.28 + vec3(0, 2, 4)), 1.0);
}
''')

_register("plasma", "Multi-octave colored plasma", "procedural", '''
void main() {
    vec2 uv = v_uv;
    float t = u_time * 0.1;
    float v = sin(uv.x * 8.0 + t) * cos(uv.y * 6.0 + t * 0.7);
    v += sin(uv.x * 16.0 - t * 1.2) * cos(uv.y * 12.0 + t * 0.5) * 0.5;
    v += sin((uv.x + uv.y) * 24.0 + t * 0.3) * 0.25;
    v = v * 0.5 + 0.5;
    f_color = vec4(0.5 + 0.5 * cos(v * 6.28 + vec3(0, 2, 4)), 1.0);
}
''')

_register("domain_warp", "Domain-warped fractal noise", "procedural", '''
void main() {
    vec2 uv = v_uv * 3.0;
    float t = u_time * 0.05;
    float warp = 2.0 + u_params.x * 3.0;
    vec2 q = vec2(fbm(uv + t), fbm(uv + vec2(5.2, 1.3) + t * 0.7));
    vec2 r = vec2(fbm(uv + warp * q + vec2(1.7, 9.2) + t * 0.3),
                  fbm(uv + warp * q + vec2(8.3, 2.8) + t * 0.4));
    float v = fbm(uv + warp * r);
    f_color = vec4(0.5 + 0.5 * cos(v * 6.28 + vec3(0, 2, 4)), 1.0);
}
''')

_register("voronoi", "Voronoi/Worley noise cells", "procedural", '''
void main() {
    vec2 uv = v_uv * (5.0 + u_params.x * 5.0);
    vec2 i = floor(uv); vec2 f = fract(uv);
    float md = 1.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 n = vec2(float(x), float(y));
            vec2 p = hash21(i + n) * vec2(1.0);
            float d = length(n + p - f);
            md = min(md, d);
        }
    }
    f_color = vec4(md, md * 0.5, 1.0 - md, 1.0);
}
''')

_register("voronoise", "Smooth voronoi layers", "procedural", '''
void main() {
    vec2 uv = v_uv * 4.0;
    float t = u_time * 0.02;
    vec2 q = vec2(fbm(uv + t), fbm(uv + vec2(3.7, 1.2) + t));
    vec2 r = vec2(fbm(uv + 4.0 * q + vec2(1.7, 9.2)),
                  fbm(uv + 4.0 * q + vec2(8.3, 2.8)));
    float v = fbm(uv + 4.0 * r);
    f_color = vec4(0.5 + 0.5 * cos(v * 4.0 + vec3(0, 2, 4)), 1.0);
}
''')

_register("ripples", "Concentric ripple pattern", "procedural", '''
void main() {
    vec2 uv = v_uv - 0.5;
    float d = length(uv);
    float r = sin(d * 30.0 - u_time * 2.0) * 0.5 + 0.5;
    float g = sin(d * 30.0 - u_time * 2.0 + 2.0) * 0.5 + 0.5;
    float b = sin(d * 30.0 - u_time * 2.0 + 4.0) * 0.5 + 0.5;
    f_color = vec4(r, g, b, 1.0) * (1.0 - d);
}
''')

_register("cells", "Cellular growth simulation", "procedural", '''
void main() {
    vec2 uv = v_uv * 8.0;
    vec2 i = floor(uv); vec2 f = fract(uv);
    float md = 8.0;
    vec2 mp = vec2(0.0);
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 n = vec2(float(x), float(y));
            vec2 p = hash21(i + n) * vec2(1.0);
            float d = length(n + p - f);
            if (d < md) { md = d; mp = n + p; }
        }
    }
    float c = hash21(i + mp);
    vec3 col = 0.5 + 0.5 * cos(c * 6.28 + vec3(0, 2, 4));
    col *= 1.0 - md * 1.2;
    col += vec3(0.05) / (md * md + 0.01);
    f_color = vec4(col, 1.0);
}
''')

_register("bubble_chamber", "Simulated bubble chamber trails", "procedural", '''
void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    float t = u_time * 0.3;
    float v = 0.0;
    for (int i = 0; i < 20; i++) {
        float fi = float(i);
        vec2 p = vec2(sin(fi * 1.7 + t * 0.5), cos(fi * 2.3 + t * 0.7)) * 0.8;
        float d = length(uv - p) - 0.03;
        v += 0.005 / (d * d + 0.001);
    }
    f_color = vec4(v * 0.5, v * 0.8, v, 1.0);
}
''')

_register("stars", "Starfield with parallax", "procedural", '''
void main() {
    vec2 uv = v_uv;
    float t = u_time * 0.05;
    vec3 col = vec3(0.0);
    for (int i = 0; i < 50; i++) {
        float fi = float(i);
        vec2 p = fract(vec2(sin(fi * 127.1 + t), cos(fi * 311.7 + t * 0.7)));
        float d = length(uv - p);
        float brightness = 0.003 / (d * d);
        vec3 star_col = 0.5 + 0.5 * cos(fi + vec3(0, 2, 4));
        col += brightness * star_col;
    }
    f_color = vec4(col, 1.0);
}
''')

_register("lightning_fractal", "Fractal lightning branching", "procedural", '''
void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    float t = u_time * 0.2;
    vec2 p = vec2(0.0);
    float v = 0.0;
    for (int i = 0; i < 64; i++) {
        float fi = float(i);
        p += vec2(sin(fi * 0.3 + t), cos(fi * 0.7 + t * 0.5)) * 0.02;
        float d = length(uv - p);
        v += 0.02 / (d + 0.01);
    }
    f_color = vec4(v * 0.3, v * 0.5, v, 1.0);
}
''')

_register("spiral", "Logarithmic spiral galaxy", "procedural", '''
void main() {
    vec2 uv = v_uv - 0.5;
    float a = atan(uv.y, uv.x);
    float r = length(uv);
    float spiral = sin(a * 4.0 - r * 15.0 + u_time * 0.5) * 0.5 + 0.5;
    float fade = exp(-r * 3.0);
    float col = spiral * fade;
    f_color = vec4(col * 1.2, col * 0.8, col * fade + 0.1, 1.0);
}
''')

_register("dendritic", "Dendritic / tree-like branching", "procedural", '''
void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    float t = u_time * 0.1;
    float d = length(uv);
    float a = atan(uv.y, uv.x) * 3.0;
    float branch = sin(a * 8.0 + log(d + 0.001) * 10.0 + t) * 0.5 + 0.5;
    float v = branch * exp(-d * 2.0);
    f_color = vec4(v * 0.3, v * 0.6, v * 0.2, 1.0);
}
''')

_register("barnsley", "Barnsley fern approximation", "procedural", '''
void main() {
    vec2 uv = v_uv * 3.0 - 1.5;
    float t = u_time * 0.1;
    float v = 0.0;
    for (int i = 0; i < 100; i++) {
        float fi = float(i);
        vec2 p = vec2(sin(fi * 0.5 + t), cos(fi * 0.3 + t * 0.7));
        float dx = uv.x - p.x * 0.5;
        float dy = uv.y - p.y * 0.8 - 0.5;
        v += 0.001 / (dx*dx + dy*dy + 0.001);
    }
    f_color = vec4(v * 0.2, v * 0.8, v * 0.2, 1.0);
}
''')

_register("spectral", "Spectral / rainbow interference", "procedural", '''
void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    float t = u_time * 0.1;
    float a = atan(uv.y, uv.x);
    float r = length(uv);
    float v = sin(r * 20.0 - t) + cos(a * 5.0 + t * 0.5);
    v = v * 0.25 + 0.5;
    f_color = vec4(0.5 + 0.5 * cos(v * 6.28 + vec3(0, 2, 4)), 1.0);
}
''')

_register("truchet", "Truchet tile pattern", "procedural", '''
void main() {
    vec2 uv = v_uv * 6.0;
    vec2 i = floor(uv); vec2 f = fract(uv) - 0.5;
    float flip = hash21(i) > 0.5 ? 1.0 : -1.0;
    float d = length(f * flip);
    float v = smoothstep(0.4, 0.5, d);
    float c = hash21(i + vec2(1.0));
    vec3 col = mix(vec3(0.9, 0.9, 0.95), 0.5 + 0.5 * cos(c * 6.28 + vec3(0, 2, 4)), v);
    f_color = vec4(col, 1.0);
}
''')

_register("kaleidoscope_fractal", "Kaleidoscope IFS fractal", "procedural", '''
void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    float t = u_time * 0.1;
    for (int i = 0; i < 10; i++) {
        uv = abs(uv);
        float a = sin(t + float(i) * 0.5);
        uv = rot(a) * uv;
        uv = uv * 1.5 - vec2(0.5);
    }
    float v = length(uv);
    f_color = vec4(0.5 + 0.5 * cos(v * 10.0 + vec3(0, 2, 4)), 1.0);
}
''')

_register("waves_3d", "3D wave interference", "procedural", '''
void main() {
    vec2 uv = v_uv * 4.0 - 2.0;
    float t = u_time * 0.5;
    float v = 0.0;
    for (int i = 0; i < 10; i++) {
        float fi = float(i);
        vec2 p = vec2(sin(fi * 1.3 + t), cos(fi * 1.7 + t * 0.8));
        v += sin(dot(uv, p) * 3.0 + t) * 0.1;
    }
    vec3 col = 0.5 + 0.5 * cos(v * 4.0 + vec3(0, 2, 4));
    f_color = vec4(col, 1.0);
}
''')

_register("pixel_sort_gpu", "Edge-directed pixel sorting on GPU", "procedural", '''
void main() {
    vec2 uv = v_uv;
    float v = fbm(uv * 4.0 + u_time * 0.02);
    vec2 step = vec2(1.0 / u_resolution.x, 1.0 / u_resolution.y);
    float dx = fbm((uv + vec2(step.x, 0)) * 4.0) - v;
    float dy = fbm((uv + vec2(0, step.y)) * 4.0) - v;
    float edge = abs(dx) + abs(dy);
    float bands = floor(uv.x * 20.0 + v * 10.0) / 20.0 + v * 0.1;
    vec3 col = 0.5 + 0.5 * cos(bands * 6.28 + vec3(0, 2, 4));
    col = mix(col, vec3(0.1), edge * 5.0);
    f_color = vec4(col, 1.0);
}
''')

_register("ocean", "Procedural ocean waves", "procedural", '''
void main() {
    vec2 uv = v_uv * 3.0;
    float t = u_time * 0.3;
    float v = sin(uv.x * 5.0 + t) * cos(uv.y * 3.0 + t * 0.7);
    v += sin(uv.x * 8.0 - t * 1.3) * sin(uv.y * 6.0 + t) * 0.5;
    v += sin((uv.x + uv.y) * 12.0 + t * 0.5) * 0.25;
    v = v * 0.5 + 0.5;
    vec3 col = mix(vec3(0.0, 0.2, 0.5), vec3(0.1, 0.6, 0.8), v);
    col += vec3(0.3, 0.4, 0.5) * pow(v, 4.0);
    f_color = vec4(col, 1.0);
}
''')

_register("nebula_gpu", "Space nebula gas clouds", "procedural", '''
void main() {
    vec2 uv = v_uv * 2.0;
    float t = u_time * 0.03;
    vec2 q = vec2(fbm(uv + t), fbm(uv + vec2(5.2, 1.3) + t * 0.7));
    vec2 r = vec2(fbm(uv + 3.0 * q + vec2(1.7, 9.2) + t * 0.3),
                  fbm(uv + 3.0 * q + vec2(8.3, 2.8) + t * 0.4));
    float v = fbm(uv + 3.0 * r);
    float mask = 1.0 - abs(v_uv.y - 0.5) * 2.0;
    vec3 col = 0.3 + 0.7 * (0.5 + 0.5 * cos(v * 4.0 + vec3(0, 1, 2)));
    col *= mask;
    f_color = vec4(col, 1.0);
}
''')

_register("terrain", "Procedural terrain heightmap", "procedural", '''
void main() {
    vec2 uv = v_uv * 3.0;
    float t = u_time * 0.02;
    float h = fbm(uv + t);
    float h2 = fbm(uv * 2.0 + t * 1.5) * 0.5;
    float h3 = fbm(uv * 4.0 + t * 2.0) * 0.25;
    h = h * 0.6 + h2 * 0.3 + h3 * 0.1;
    vec3 col;
    if (h < 0.3) col = vec3(0.1, 0.3, 0.6);
    else if (h < 0.45) col = vec3(0.2, 0.5, 0.2);
    else if (h < 0.6) col = vec3(0.3, 0.3, 0.1);
    else if (h < 0.75) col = vec3(0.4, 0.25, 0.1);
    else col = vec3(0.8, 0.8, 0.9);
    float shade = 0.5 + 0.5 * cos(h * 20.0);
    f_color = vec4(col * shade, 1.0);
}
''')

_register("wood_grain_gpu", "Concentric wood grain rings", "procedural", '''
void main() {
    vec2 uv = v_uv - 0.5;
    float d = length(uv) * 10.0;
    float grain = sin(d * 8.0 + fbm(uv * 10.0) * 0.5) * 0.5 + 0.5;
    vec3 col = mix(vec3(0.3, 0.15, 0.05), vec3(0.6, 0.3, 0.1), grain);
    f_color = vec4(col, 1.0);
}
''')

_register("fire_gpu", "Animated fire/flame", "procedural", '''
void main() {
    vec2 uv = v_uv;
    float t = u_time * 0.5;
    float v = fbm(vec2(uv.x * 3.0, (1.0 - uv.y) * 5.0 + t));
    v = v * (1.0 - uv.y);
    vec3 col = mix(vec3(1.0, 0.9, 0.4), vec3(0.8, 0.2, 0.0), v);
    col = mix(col, vec3(0.1, 0.0, 0.0), 1.0 - v);
    f_color = vec4(col, 1.0);
}
''')

_register("smoke_gpu", "Rising smoke / steam", "procedural", '''
void main() {
    vec2 uv = v_uv;
    float t = u_time * 0.1;
    float v = fbm(uv * 3.0 + vec2(0.0, t));
    v = v * (1.0 - uv.y) * 0.8;
    vec3 col = mix(vec3(0.8, 0.8, 0.85), vec3(0.2, 0.2, 0.25), v);
    f_color = vec4(col, 1.0);
}
''')


# ── FILTER SHADERS (process input image) ──

def _filter_shader(source: str) -> str:
    """Wrap a filter shader body with the full image processing prologue."""
    return f'''
{_PROLOGUE}
vec4 sample(vec2 uv) {{ return texture(u_texture, uv); }}

void main() {{
    vec2 uv = v_uv;
    vec2 step = 1.0 / u_resolution;
    vec4 orig = sample(uv);
    {source}
}}
'''

_register("shader_bloom", "GPU bloom/glow from bright areas", "filter", _filter_shader('''
    float brightness = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    vec4 glow = vec4(0.0);
    for (int x = -3; x <= 3; x++) {
        for (int y = -3; y <= 3; y++) {
            vec2 off = vec2(float(x), float(y)) * step * 2.0;
            float b = dot(texture(u_texture, uv + off).rgb, vec3(0.299, 0.587, 0.114));
            if (b > 0.7) glow += texture(u_texture, uv + off) * exp(-float(x*x + y*y) / 4.0);
        }
    }
    glow /= 8.0;
    f_color = orig + glow * u_params.x;
'''))

_register("shader_emboss", "GPU emboss / bump mapping", "filter", _filter_shader('''
    float gx = dot(texture(u_texture, uv + vec2(step.x, 0)).rgb, vec3(0.299, 0.587, 0.114));
    float gy = dot(texture(u_texture, uv + vec2(0, step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float gz = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float dx = gx - gz; float dy = gy - gz;
    float bump = (dx + dy) * 2.0 + 0.5;
    f_color = vec4(mix(orig.rgb, vec3(bump), u_params.x), 1.0);
'''))

_register("shader_kaleidoscope", "GPU kaleidoscope mirror", "filter", _filter_shader('''
    vec2 p = uv - 0.5;
    float a = atan(p.y, p.x);
    float r = length(p);
    float seg = 3.14159 * 2.0 / max(4.0, 8.0 - u_params.x * 4.0);
    a = mod(a, seg);
    a = abs(a - seg * 0.5);
    vec2 q = vec2(cos(a), sin(a)) * r + 0.5;
    f_color = texture(u_texture, q);
'''))

_register("shader_water_ripple", "GPU water ripple distortion", "filter", _filter_shader('''
    vec2 off = vec2(
        sin(uv.y * 50.0 + u_time * 2.0) * 0.01 * u_params.x,
        cos(uv.x * 50.0 + u_time * 1.5) * 0.01 * u_params.x
    );
    f_color = texture(u_texture, uv + off);
'''))

_register("shader_heat_shimmer", "GPU heat haze / shimmer", "filter", _filter_shader('''
    float haze = sin(uv.x * 30.0 + uv.y * 20.0 + u_time * 3.0) * u_params.x * 0.02;
    vec2 off = vec2(0.0, haze * (1.0 - uv.y));
    f_color = texture(u_texture, uv + off);
'''))

_register("shader_pixelate_gpu", "GPU pixelation with edge preservation", "filter", _filter_shader('''
    float block = max(4.0, 64.0 - u_params.x * 60.0);
    vec2 q = floor(uv * u_resolution / block) * block / u_resolution;
    f_color = texture(u_texture, q);
'''))

_register("shader_ink_bleed", "GPU ink bleed / watercolor spread", "filter", _filter_shader('''
    vec3 sum = vec3(0.0);
    float count = 0.0;
    for (int x = -4; x <= 4; x++) {
        for (int y = -4; y <= 4; y++) {
            vec2 off = vec2(float(x), float(y)) * step * u_params.x;
            float w = exp(-float(x*x + y*y) / (4.0 * u_params.x));
            sum += texture(u_texture, uv + off).rgb * w;
            count += w;
        }
    }
    f_color = vec4(sum / count, 1.0);
'''))

_register("shader_halftone_gpu", "GPU halftone dot screen", "filter", _filter_shader('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float cell = 8.0 + u_params.x * 20.0;
    vec2 q = fract(uv * u_resolution / cell);
    float d = length(q - 0.5);
    float dot_r = (1.0 - gray) * 0.5;
    float v = d < dot_r ? 0.0 : 1.0;
    f_color = vec4(vec3(v), 1.0);
'''))

_register("shader_crt_gpu", "GPU CRT scanlines + bloom", "filter", _filter_shader('''
    float scan = sin(uv.y * u_resolution.y * 3.14159) * 0.5 + 0.5;
    float scanline = 1.0 - (1.0 - scan) * 0.3;
    // chromatic shift at edges
    vec2 r_uv = uv + vec2(0.001 * pow(abs(uv.x - 0.5) * 2.0, 2.0), 0.0);
    vec2 b_uv = uv - vec2(0.001 * pow(abs(uv.x - 0.5) * 2.0, 2.0), 0.0);
    vec3 col;
    col.r = texture(u_texture, r_uv).r;
    col.g = texture(u_texture, uv).g;
    col.b = texture(u_texture, b_uv).b;
    col *= scanline;
    f_color = vec4(col, 1.0);
'''))

_register("shader_hologram", "GPU hologram / scan effect", "filter", _filter_shader('''
    float scan = sin(uv.y * u_resolution.y * 0.5 + u_time * 5.0) * 0.5 + 0.5;
    float scanline = 1.0 - pow(scan, 4.0) * 0.4;
    float edge = abs(uv.x - 0.5) * 2.0;
    float vignette = 1.0 - pow(edge, 3.0) * 0.5;
    float shift = sin(uv.x * 50.0 + u_time * 3.0) * 0.02;
    vec2 q = uv + vec2(0.0, shift);
    vec3 col = texture(u_texture, q).rgb * scanline * vignette;
    float hue = sin(uv.y * 20.0 + u_time * 2.0) * 0.1 + 0.1;
    col += vec3(hue, hue * 0.3, hue * 0.8);
    f_color = vec4(col, 1.0);
'''))

_register("shader_mosaic_gpu", "GPU stained glass mosaic", "filter", _filter_shader('''
    float cell = 20.0 + u_params.x * 40.0;
    vec2 cell_uv = floor(uv * u_resolution / cell) * cell / u_resolution + cell / u_resolution * 0.5;
    f_color = texture(u_texture, cell_uv);
'''))

_register("shader_edge_detect_gpu", "GPU Sobel edge detection", "filter", _filter_shader('''
    float tl = dot(texture(u_texture, uv + vec2(-step.x, -step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float t  = dot(texture(u_texture, uv + vec2(0, -step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float tr = dot(texture(u_texture, uv + vec2(step.x, -step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float l  = dot(texture(u_texture, uv + vec2(-step.x, 0)).rgb, vec3(0.299, 0.587, 0.114));
    float r  = dot(texture(u_texture, uv + vec2(step.x, 0)).rgb, vec3(0.299, 0.587, 0.114));
    float bl = dot(texture(u_texture, uv + vec2(-step.x, step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float b  = dot(texture(u_texture, uv + vec2(0, step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float br = dot(texture(u_texture, uv + vec2(step.x, step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy = -tl - 2.0*t - tr + bl + 2.0*b + br;
    float edge = sqrt(gx*gx + gy*gy);
    f_color = vec4(mix(orig.rgb, vec3(edge), u_params.x), 1.0);
'''))

_register("shader_warhol", "GPU Warhol 4-panel duotone", "filter", _filter_shader('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    vec2 p = floor(uv * 2.0);
    vec3 c1, c2;
    if (p.x < 1.0 && p.y < 1.0) { c1 = vec3(0.8, 0.2, 0.2); c2 = vec3(1.0, 1.0, 0.4); }
    else if (p.x >= 1.0 && p.y < 1.0) { c1 = vec3(0.2, 0.4, 0.8); c2 = vec3(0.6, 0.2, 0.6); }
    else if (p.x < 1.0 && p.y >= 1.0) { c1 = vec3(0.2, 0.8, 0.2); c2 = vec3(0.4, 0.2, 0.8); }
    else { c1 = vec3(0.8, 0.6, 0.2); c2 = vec3(0.8, 0.2, 0.2); }
    f_color = vec4(mix(c1, c2, gray), 1.0);
'''))

_register("shader_duotone_gpu", "GPU duotone with color controls", "filter", _filter_shader('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    vec3 c1 = vec3(u_params.x, u_params.y, u_params.z);
    vec3 c2 = vec3(u_params.w, 0.2, 0.8);
    f_color = vec4(mix(c1, c2, gray), 1.0);
'''))

_register("shader_rgb_split", "GPU RGB channel separation", "filter", _filter_shader('''
    float shift = 0.02 * u_params.x;
    vec2 r_uv = uv + vec2(shift, 0.0);
    vec2 b_uv = uv - vec2(shift, 0.0);
    float r = texture(u_texture, r_uv).r;
    float g = orig.g;
    float b = texture(u_texture, b_uv).b;
    f_color = vec4(r, g, b, 1.0);
'''))

_register("shader_caustics_gpu", "GPU caustic light overlay", "filter", _filter_shader('''
    float caustic = sin(uv.x * 30.0 + u_time) * cos(uv.y * 25.0 + u_time * 0.7);
    caustic += sin(uv.x * 50.0 - u_time * 1.3) * sin(uv.y * 40.0 + u_time * 0.5) * 0.5;
    caustic = max(0.0, caustic) * u_params.x * 0.8;
    vec3 light = vec3(0.8, 0.9, 1.0) * caustic;
    f_color = vec4(orig.rgb + light, 1.0);
'''))

_register("shader_glitch_gpu", "GPU digital glitch artifacts", "filter", _filter_shader('''
    float band = floor(uv.y * 40.0 * u_params.x);
    float shift = sin(band * 7.0 + u_time * 5.0) * 0.05 * u_params.x;
    float noise = fract(sin(dot(uv * u_resolution, vec2(12.9898, 78.233))) * 43758.5453);
    float glitch = noise > (1.0 - u_params.x * 0.1) ? 1.0 : 0.0;
    vec2 q = uv + vec2(shift + glitch * 0.1, 0.0);
    f_color = texture(u_texture, q);
'''))

_register("shader_posterize_gpu", "GPU posterization / color reduction", "filter", _filter_shader('''
    float levels = max(2.0, 16.0 - u_params.x * 14.0);
    vec3 col = floor(orig.rgb * levels) / levels;
    f_color = vec4(col, 1.0);
'''))

_register("shader_oil_gpu", "GPU oil painting simulation", "filter", _filter_shader('''
    float radius = 2.0 + u_params.x * 4.0;
    vec3 sum = vec3(0.0); float total = 0.0;
    for (int x = -3; x <= 3; x++) {
        for (int y = -3; y <= 3; y++) {
            vec2 off = vec2(float(x), float(y)) * step;
            float w = exp(-float(x*x + y*y) / (radius * radius));
            sum += texture(u_texture, uv + off).rgb * w;
            total += w;
        }
    }
    f_color = vec4(sum / total, 1.0);
'''))

_register("shader_neon_gpu", "GPU neon glow on edges", "filter", _filter_shader('''
    float gx = 0.0, gy = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 off = vec2(float(x), float(y)) * step;
            float v = dot(texture(u_texture, uv + off).rgb, vec3(0.299, 0.587, 0.114));
            gx += float(x) * v; gy += float(y) * v;
        }
    }
    float edge = sqrt(gx*gx + gy*gy);
    float glow = edge * u_params.x * 3.0;
    vec3 neon = vec3(glow * 0.8, glow * 0.3, glow);
    f_color = vec4(orig.rgb + neon, 1.0);
'''))

_register("shader_pencil_gpu", "GPU pencil sketch", "filter", _filter_shader('''
    float gx = 0.0, gy = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 off = vec2(float(x), float(y)) * step;
            float v = dot(texture(u_texture, uv + off).rgb, vec3(0.299, 0.587, 0.114));
            gx += float(x) * v; gy += float(y) * v;
        }
    }
    float edge = sqrt(gx*gx + gy*gy);
    float sketch = 1.0 - edge * 4.0;
    f_color = vec4(mix(orig.rgb, vec3(sketch), u_params.x), 1.0);
'''))

_register("shader_motion_blur_gpu", "GPU directional motion blur", "filter", _filter_shader('''
    float angle = u_params.x * 6.2832;
    float dist = 10.0 + u_params.y * 20.0;
    vec2 dir = vec2(cos(angle), sin(angle)) * step * dist;
    vec3 col = vec3(0.0);
    for (int i = -5; i <= 5; i++) {
        float t = float(i) / 5.0;
        col += texture(u_texture, uv + dir * t).rgb * (1.0 - abs(t));
    }
    f_color = vec4(col / 3.5, 1.0);
'''))


# ═══════════════════════════════════════════════
#  RENDER ENGINE
# ═══════════════════════════════════════════════

def _create_vao(ctx, prog):
    """Create full-screen quad VAO."""
    vbo = ctx.buffer(_QUAD_VERTICES.tobytes())
    ibo = ctx.buffer(_QUAD_INDICES.tobytes())
    vao = ctx.vertex_array(prog, [
        (vbo, '2f 2f', 'in_vert', 'in_uv'),
    ], ibo)
    return vao


def render_shader(shader_name: str, resolution: tuple[int, int] = (512, 512),
                   params: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5),
                   time: float = 0.0,
                   input_image: np.ndarray | None = None) -> Image.Image:
    """Render a shader to an image.

    Args:
        shader_name: Name in SHADERS dict
        resolution: (width, height) output size
        params: 4 float uniforms mapped to u_params
        time: Time value for u_time animation
        input_image: Optional numpy array (H,W,3) for filter shaders

    Returns: PIL Image
    """
    if shader_name not in SHADERS:
        raise ValueError(f"Unknown shader: {shader_name}. Available: {list(SHADERS.keys())}")

    info = SHADERS[shader_name]
    ctx = _get_ctx()

    w, h = resolution

    # Build fragment shader
    if info["type"] == "filter":
        # Filter shaders already have full wrapper
        frag_src = info["source"]
    else:
        # Procedural shaders get the prologue
        frag_src = _PROLOGUE + info["source"]

    try:
        prog = ctx.program(vertex_shader=_VERTEX_SHADER, fragment_shader=frag_src)
    except Exception as e:
        raise RuntimeError(f"Shader compilation failed for '{shader_name}': {e}")

    vao = _create_vao(ctx, prog)

    # Create framebuffer
    fbo = ctx.simple_framebuffer((w, h))
    fbo.use()

    # Set uniforms (some may be optimized out)
    for uniform_name, uniform_value in [('u_resolution', (float(w), float(h))),
                                         ('u_time', time),
                                         ('u_params', params)]:
        if uniform_name in prog:
            prog[uniform_name].value = uniform_value

    # Handle input texture
    if input_image is not None and 'u_texture' in prog:
        # Upload input texture in RGB order
        tex_data = input_image[:, :, ::-1].tobytes()  # RGB -> BGR
        texture = ctx.texture((input_image.shape[1], input_image.shape[0]), 3, tex_data)
        texture.use(0)
        prog['u_texture'].value = 0

    ctx.clear(0.0, 0.0, 0.0)
    vao.render()
    data = fbo.read()

    # Convert to PIL
    img = Image.frombytes('RGB', (w, h), data, 'raw', 'BGR')

    # Cleanup
    fbo.release()
    vao.release()
    prog.release()

    return img


def _have_cv2():
    """Check if OpenCV is available."""
    try:
        import cv2
        return True
    except ImportError:
        return False


# ═══════════════════════════════════════════════
#  HANDY HELPERS
# ═══════════════════════════════════════════════

def list_shaders(shader_type: str | None = None) -> list[dict]:
    """List all available shaders, optionally filtered by type."""
    if shader_type:
        return [v for v in SHADERS.values() if v["type"] == shader_type]
    return list(SHADERS.values())


def render_procedural(shader_name: str, resolution=(512, 512), params=(0.5, 0.5, 0.5, 0.5),
                       time=0.0) -> Image.Image:
    """Render a procedural shader (no input image needed)."""
    info = SHADERS.get(shader_name)
    if info and info["type"] == "filter":
        raise ValueError(f"'{shader_name}' is a filter shader, use render_filter() instead")
    return render_shader(shader_name, resolution, params, time)


def render_filter(shader_name: str, input_image: np.ndarray,
                   params=(0.5, 0.5, 0.5, 0.5), time=0.0) -> Image.Image:
    """Apply a filter shader to an input image."""
    info = SHADERS.get(shader_name)
    if info and info["type"] == "procedural":
        raise ValueError(f"'{shader_name}' is a procedural shader, use render_procedural() instead")
    h, w = input_image.shape[:2]
    return render_shader(shader_name, (w, h), params, time, input_image)