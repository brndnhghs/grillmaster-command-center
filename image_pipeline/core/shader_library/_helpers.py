"""Shared GLSL helpers for the shader library.

Split out of core/shaders.py: the prologue plus reusable GLSL fragments and
registration-time wrapper functions that shader_library modules reference.
"""
from __future__ import annotations

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




#  P0.3 — Escape-time / deterministic fractal CPU-twin shaders (client-GPU live
#  preview of nodes 33/51/52/66/67/69). These are ADDITIVE: the server's CPU
#  numpy path stays the authoritative export; these only drive the browser live
#  preview. They reuse the prologue helpers (rot/hash21/noise/fbm) and the
#  inferno colormap where a fire-style look suits the node's default.

# ── Reusable fractal coloring + escape helper (consumed by the twins below) ──
_FRACTAL_HELPERS = '''
vec3 fractal_palette(float t) {
    // Smooth cosine palette (matches the CPU 'sine' color mode's character).
    return 0.5 + 0.5 * cos(6.28318 * (vec3(1.0, 0.75, 0.5) * t) + vec3(0.0, 2.0, 4.0));
}

// Smooth iteration count → [0,1] using the standard normalized-iteration trick.
float smooth_iter(float n, float last_z2, float max_iter) {
    float nu = n + 1.0 - log(max(log(last_z2) * 0.5, 1.0001)) / log(2.0);
    return clamp(nu / max(max_iter, 1.0), 0.0, 1.0);
}
'''


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


def _filter_typed(body: str) -> str:
    """Wrap a TYPED filter shader body (uses named u_<var> uniforms).

    Unlike _filter_shader, this does NOT embed the prologue — render_shader's
    _assemble_gl330 injects the shared _PROLOGUE + typed-uniform declarations
    for filter shaders that declare `uniforms=`. The body uses v_uv, u_texture,
    u_resolution, u_time (all from the prologue) and the local `step`/`sample`.
    """
    return f'''
vec4 sample(vec2 uv) {{ return texture(u_texture, uv); }}

void main() {{
    vec2 uv = v_uv;
    vec2 step = 1.0 / u_resolution;
    vec4 orig = sample(uv);
    {body}
}}
'''



# ── Horizon-Based Ambient Occlusion (node 425) ──────────────────────────────
# Closed-form client-GPU twin of the CPU HBAO node. Given a procedural fbm
# height field h(uv), AO at each pixel is the fraction of the sky hemisphere
# NOT blocked by higher neighbours: walk N azimuth rays, record the max horizon
# silhouette angle phi = atan2((h(q)-h(p))*k, dist) along each ray, sum the
# visible fraction 0.5*(1+cos(phi)). No inter-frame state -> pure f(uv,t), the
# P0.6 field-eval family. Helpers are inlined (late _INFERNO_GPU is below this
# region — pitfall #17). Shader-only knobs (EXAG/jitter hash) are documented;
# the node's REAL params are routed by name via param_map in gpu_shaders.py.
_HBAO_HELPERS = '''
float _hbao_hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float _hbao_vnoise(vec2 x) {
    vec2 xi = floor(x); vec2 xf = fract(x);
    vec2 u = xf * xf * (3.0 - 2.0 * xf);
    float a = _hbao_hash(xi);
    float b = _hbao_hash(xi + vec2(1.0, 0.0));
    float c = _hbao_hash(xi + vec2(0.0, 1.0));
    float d = _hbao_hash(xi + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2.0 - 1.0;
}
float _hbao_fbm(vec2 x, int oct) {
    float outv = 0.0, amp = 1.0, freq = 1.0, norm = 0.0;
    for (int o = 0; o < 6; o++) {
        if (o >= oct) break;
        float a = 2.39996323 * float(o + 1);
        float ca = cos(a), sa = sin(a);
        vec2 r = vec2(x.x * freq * ca - x.y * freq * sa,
                      x.x * freq * sa + x.y * freq * ca);
        outv += amp * _hbao_vnoise(r);
        norm += amp; amp *= 0.5; freq *= 2.0;
    }
    return outv / max(norm, 1e-6);
}
vec3 _hbao_inferno(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(0.00021894, 0.00165100, -0.01948090);
    const vec3 c1 = vec3(0.10651342, 0.56395644, 3.93271239);
    const vec3 c2 = vec3(11.60249308, -3.97285397, -15.94239411);
    const vec3 c3 = vec3(-41.70399613, 17.43639888, 44.35414520);
    const vec3 c4 = vec3(77.16293570, -33.40235894, -81.80730926);
    const vec3 c5 = vec3(-71.31942824, 32.62606426, 73.20951986);
    const vec3 c6 = vec3(25.13112622, -12.24266895, -23.07032500);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}
vec3 _hbao_hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
'''



# ── Thin-Film Interference (node 1004) ──────────────────────────────────────
# Closed-form client-GPU twin of the CPU spectral thin-film node. A film of
# refractive index n and thickness d reflects a two-beam interference spectrum
# with phase delta = 4*pi*n*d*cos(theta_t)/lambda + pi (single phase reversal at
# the air->film interface, Hecht Optics 4ed §9.5). The reflected spectrum is
# integrated against the CIE 1931 2-deg colour-matching functions (Wyman, Sloan
# & Shirley 2013 analytic gaussian fit) and converted XYZ->linear-sRGB, so the
# full violet->magenta->red band wrap is reproduced rather than a naive RGB pick.
# A procedural fbm thickness field paints the bands; a drainage term thins the
# film toward the top like a real draining bubble. No inter-frame state -> pure
# f(uv,t), the P0.6 field-eval family. CPU numpy node 1004 stays authoritative
# for export; this twin is the live-preview approximation (35-sample spectrum vs
# the CPU's 69, single fbm scale vs the CPU's rng-jittered scale).
_TF_HELPERS = '''
float _tf_hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float _tf_vnoise(vec2 x) {
    vec2 xi = floor(x); vec2 xf = fract(x);
    vec2 u = xf * xf * (3.0 - 2.0 * xf);
    float a = _tf_hash(xi);
    float b = _tf_hash(xi + vec2(1.0, 0.0));
    float c = _tf_hash(xi + vec2(0.0, 1.0));
    float d = _tf_hash(xi + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2.0 - 1.0;
}
float _tf_fbm(vec2 x) {
    float outv = 0.0, amp = 1.0, freq = 1.0, norm = 0.0;
    for (int o = 0; o < 5; o++) {
        outv += amp * _tf_vnoise(x * freq);
        norm += amp; amp *= 0.5; freq *= 2.0;
    }
    return outv / max(norm, 1e-6);
}
// One-sided gaussian (Wyman et al. 2013 CIE-CMF building block).
float _tf_g(float x, float mu, float s1, float s2) {
    float s = x < mu ? s1 : s2;
    float t = (x - mu) / s;
    return exp(-0.5 * t * t);
}
vec3 _tf_cmf(float lam) {
    float xb = 1.056 * _tf_g(lam, 599.8, 37.9, 31.0)
             + 0.362 * _tf_g(lam, 442.0, 16.0, 26.7)
             - 0.065 * _tf_g(lam, 501.1, 20.4, 26.2);
    float yb = 0.821 * _tf_g(lam, 568.8, 46.9, 40.5)
             + 0.286 * _tf_g(lam, 530.9, 16.3, 31.1);
    float zb = 1.217 * _tf_g(lam, 437.0, 11.8, 36.0)
             + 0.681 * _tf_g(lam, 459.0, 26.0, 13.8);
    return vec3(xb, yb, zb);
}
vec3 _tf_xyz2srgb(vec3 xyz) {
    mat3 M = mat3( 3.2404542, -0.9692660,  0.0556434,
                  -1.5371385,  1.8760108, -0.2040259,
                  -0.4985314,  0.0415560,  1.0572252);
    vec3 lin = max(M * xyz, 0.0);
    vec3 a = vec3(0.055);
    bvec3 hi = greaterThan(lin, vec3(0.0031308));
    vec3 srgb = mix(12.92 * lin, 1.055 * pow(lin, vec3(1.0/2.4)) - a, vec3(hi));
    return clamp(srgb, 0.0, 1.0);
}
'''


# ── P0.6 field-eval client-GPU twins (continued) ──────────────────────────
# Nodes 53 / 43 / 57 are closed-form functions of (uv, t) — exact parity
# previews (no seeded-layout divergence), same family as 125/164/172. The
# CPU numpy nodes stay the authoritative export (two-tier precision). Placed
# here (before the _INFERNO block) so they only use the prologue helpers
# (hash21/fbm/rot) plus a self-contained inline colormap. pitfall #15: encode
# 0.5 as NEUTRAL so the default u_params yields each node's canonical view.

def _inferno_local(t):
    # Compact inferno polynomial (duplicated locally; each twin is a separate
    # shader program so no symbol collision). Kept as a Python string so it can
    # be inlined into the twin bodies below.
    return '''vec3 inferno(float t){
    t = clamp(t, 0.0, 1.0);
    const vec3 c0=vec3(0.00021894,0.00016488,-0.01907227);
    const vec3 c1=vec3(0.10651034,0.56396050,3.93279110);
    const vec3 c2=vec3(11.6028830,-3.9781129,-15.9420510);
    const vec3 c3=vec3(-41.703996,17.4360890,44.3541450);
    const vec3 c4=vec3(77.1629350,-33.402243,-81.8094230);
    const vec3 c5=vec3(-71.319421,32.6260640,73.2095190);
    const vec3 c6=vec3(25.1311300,-12.242810,-23.0709590);
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}'''



# ═══════════════════════════════════════════════
#  P0 client-GPU parity shaders for existing CPU nodes
# ═══════════════════════════════════════════════
# Render EXISTING CPU pattern nodes (04 Worley, 02 Quasicrystal) on the browser
# GPU for the live preview (see methods/gpu_shaders.py CLIENT_GPU_SHIMS). The CPU
# numpy node stays the authoritative export (two-tier precision).
#
# NOTE (determinism): both CPU nodes seed feature-point positions / per-wave
# phases with numpy PCG64 (np.random.default_rng), which GLSL cannot reproduce.
# These shaders replicate the STRUCTURE via a GLSL hash, so the live look matches
# the node's character but not the exact seeded layout. High/exact parity needs a
# derived-uniforms path (compute the RNG values server-side -> uniforms) - deferred.

_INFERNO = """
vec3 inferno(float t){ t = clamp(t, 0.0, 1.0);
  const vec3 c0=vec3(0.00021894,0.00016488,-0.01907227);
  const vec3 c1=vec3(0.10651034,0.56396050, 3.93279110);
  const vec3 c2=vec3(11.6028830,-3.9781129,-15.9420510);
  const vec3 c3=vec3(-41.703996,17.4360890, 44.3541450);
  const vec3 c4=vec3(77.1629350,-33.402243,-81.8094230);
  const vec3 c5=vec3(-71.319421,32.6260640, 73.2095190);
  const vec3 c6=vec3(25.1311300,-12.242810,-23.0709590);
  return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
"""


# ── Typed escape-time fractal nodes (ids 238-243) ───────────────────────
# Categorical coverage for the signature fractal family (Mandelbrot / Julia /
# Burning Ship / Newton / Sierpinski / Lyapunov). These expose NAMED, typed
# controls (zoom, center, iteration count, palette, colors) + wireable SCALAR
# ports, replacing the opaque p1..p4 shims for these nodes. CPU fns stay
# authoritative; these are an additive typed-uniform convenience layer.
_TYPED_FRACTAL_HELPERS = _FRACTAL_HELPERS + '''
vec3 inferno_l(float t){
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(0.00021894, 0.00016488, -0.01907227);
    const vec3 c1 = vec3(0.10651034, 0.56396050, 3.93279110);
    const vec3 c2 = vec3(11.6028830, -3.9781129, -15.9420510);
    const vec3 c3 = vec3(-41.703996, 17.4360890, 44.3541450);
    const vec3 c4 = vec3(77.1629350, -33.402243, -81.8094230);
    const vec3 c5 = vec3(-71.319421, 32.6260640, 73.2095190);
    const vec3 c6 = vec3(25.1311300, -12.242810, -23.0709590);
    return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}
vec3 _fractalColor(float t, int mode, vec3 ca, vec3 cb, float shift){
    if (mode == 1) return inferno_l(t);
    if (mode == 2) return mix(ca, cb, clamp(t, 0.0, 1.0));
    return fractal_palette(t + shift);
}
'''


# ── Typed-uniform closed-form field-eval twins (250-257) ──────────────────────
# Each exposes its key visual parameters as named u_* uniforms so the GPU node
# is fully editable AND wireable (data-typed SCALAR inputs) per the typed-uniform
# contract. All are pure functions of (uv, t) -> exact parity preview, no seeded
# layout divergence (same family as 125 Chladni / 164 Moiré / 172 Dunes).

# Shared inferno colormap (each _register is a separate program).
_INFERNO_GPU = '''
vec3 inferno(float t){
    t = clamp(t, 0.0, 1.0);
    const vec3 c0=vec3(0.00021894,0.00016488,-0.01907227);
    const vec3 c1=vec3(0.10651034,0.56396050,3.93279110);
    const vec3 c2=vec3(11.6028830,-3.9781129,-15.9420510);
    const vec3 c3=vec3(-41.703996,17.4360890,44.3541450);
    const vec3 c4=vec3(77.1629350,-33.402243,-81.8094230);
    const vec3 c5=vec3(-71.319421,32.6260640,73.2095190);
    const vec3 c6=vec3(25.1311300,-12.242810,-23.0709590);
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
'''


# ── Typed-uniform derivative-field nodes (258-264, 2026-07-11) ───────────────
# Single-input IMAGE filters that derive a FIELD from the upstream frame:
# Sobel magnitude / direction, Laplacian, Scharr, normal map, gradient
# orientation flow, emboss. Every variable is a real node param + wireable
# SCALAR port; filters take image_in: IMAGE (same contract as 244-249).
# Bodies stay in the GL330/ES300 parity subset. NOTE: the pixel-step uniform is
# named `u_texel` (NOT `step`, which is reserved in every filter twin — pitfall
# #15b). The 3x3 stencil is shared via _DERIV_GPU.

_DERIV_GPU = '''
float _dlum(vec2 uv) {
    return dot(texture(u_texture, uv).rgb, vec3(0.299, 0.587, 0.114));
}
vec3 _dfetch(vec2 uv) { return texture(u_texture, uv).rgb; }
'''
