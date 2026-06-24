from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, get_font, BG_DEFAULT, W, H
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

@method(
    id="65",
    name="Waveform",
    category="math_art",
    tags=["waveform", "audio", "expanded", "animation"],
    params={
        "wave_type": {"description": "waveform type: sine, sawtooth, square, triangle, pulse, am_modulated, fm_modulated, noise_floor, lissajous, harmonic_series, interference, phase_space, wavetable, granular", "default": "sine"},
        "freq1": {"description": "base frequency", "min": 1, "max": 50, "default": 5},
        "freq2": {"description": "secondary frequency", "min": 1, "max": 50, "default": 3},
        "freq3": {"description": "tertiary frequency", "min": 1, "max": 50, "default": 7},
        "noise_level": {"description": "noise level (0-1)", "min": 0.0, "max": 1.0, "default": 0.05},
        "amplitude_ratio": {"description": "amplitude ratio (0-1)", "min": 0.0, "max": 1.0, "default": 0.8},
        "layout": {"description": "layout: single, multi_track, stereo_pair, circular, equalizer, waterfall", "default": "single"},
        "style": {"description": "render style: line, gradient_fill, oscilloscope, neon_tube, heat_wave, particle_trace, filled_wave", "default": "line"},
        "palette": {"description": "PALETTES name for palette quantization", "default": ""},
        "bg_style": {"description": "background: dark, light, grid, gradient, scanline", "default": "dark"},
        "num_tracks": {"description": "number of tracks (multi_track layout)", "min": 1, "max": 20, "default": 4},
        "pulse_width": {"description": "pulse width for pulse wave (0-1)", "min": 0.0, "max": 1.0, "default": 0.5},
        "mod_freq": {"description": "modulation frequency", "min": 1, "max": 20, "default": 2},
        "mod_depth": {"description": "modulation depth (0-1)", "min": 0.0, "max": 1.0, "default": 0.5},
        "decay_rate": {"description": "decay rate (0-1)", "min": 0.0, "max": 1.0, "default": 0.9},
        "line_width": {"description": "line width in pixels", "min": 1, "max": 10, "default": 2},
        "fill_alpha": {"description": "fill alpha (0-1)", "min": 0.0, "max": 1.0, "default": 0.3},
        "num_bars": {"description": "number of bars (equalizer layout)", "min": 5, "max": 200, "default": 50},"anim_mode": {"description": "animation mode", "choices": ["none", "freq_sweep", "phase_drift", "modulation_cycle", "layout_morph"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    }
)
def method_waveform(out_dir: Path, seed: int, params=None):
    """Generate waveform visualizations with various wave types, layouts, and render styles.

    Renders audio-style waveforms using mathematical wave functions (sine, sawtooth,
    square, triangle, pulse, AM/FM modulated, noise floor, lissajous, harmonic series,
    interference, phase space, wavetable, granular). Supports 6 layouts (single,
    multi_track, stereo_pair, circular, equalizer, waterfall) and 7 render styles
    (line, gradient_fill, oscilloscope, neon_tube, heat_wave, particle_trace,
    filled_wave). Animation modes: freq_sweep (frequency oscillation), phase_drift
    (phase offset drift), modulation_cycle (modulation depth oscillation),
    layout_morph (parameter cross-fade).

    Params:
        wave_type: waveform type (sine, sawtooth, square, triangle, pulse, ...)
        freq1: base frequency (1-50, default 5)
        freq2: secondary frequency (1-50, default 3)
        freq3: tertiary frequency (1-50, default 7)
        noise_level: noise level 0-1 (default 0.05)
        amplitude_ratio: amplitude ratio 0-1 (default 0.8)
        layout: layout (single, multi_track, stereo_pair, circular, equalizer, waterfall)
        style: render style (line, gradient_fill, oscilloscope, neon_tube, ...)
        palette: PALETTES name for palette quantization
        bg_style: background (dark, light, grid, gradient, scanline)
        num_tracks: number of tracks for multi_track layout (1-20, default 4)
        pulse_width: pulse width for pulse wave 0-1 (default 0.5)
        mod_freq: modulation frequency (1-20, default 2)
        mod_depth: modulation depth 0-1 (default 0.5)
        decay_rate: decay rate 0-1 (default 0.9)
        line_width: line width in pixels (1-10, default 2)
        fill_alpha: fill alpha 0-1 (default 0.3)
        num_bars: number of bars for equalizer layout (5-200, default 50)
        time: animation time (0-6.28)
        anim_mode: animation mode (none, freq_sweep, phase_drift, modulation_cycle, layout_morph)
        anim_speed: animation speed multiplier (0.1-3.0, default 1.0)
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    wt = str(params.get("wave_type", "sine"))
    f1 = float(params.get("freq1", 5))
    f2 = float(params.get("freq2", 3))
    f3 = float(params.get("freq3", 7))
    nl = float(params.get("noise_level", 0.05))
    ar = float(params.get("amplitude_ratio", 0.8))
    layout = str(params.get("layout", "single"))
    style = str(params.get("style", "line"))
    pal_name = str(params.get("palette", ""))
    bg_style = str(params.get("bg_style", "dark"))
    nt = int(params.get("num_tracks", 4))
    pw = float(params.get("pulse_width", 0.5))
    mf = float(params.get("mod_freq", 2))
    md = float(params.get("mod_depth", 0.5))
    decay = float(params.get("decay_rate", 0.9))
    lw = int(params.get("line_width", 2))
    fa = float(params.get("fill_alpha", 0.3))
    nb = int(params.get("num_bars", 50))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    # ── cv2 guard ──
    if not _has_cv2:
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[:, :, :] = 0.05
        capture_frame("65", img)
        save(img, mn(65, "Waveform"), out_dir)
        return

    # ── Animation: modulate params ──
    if anim_mode == "freq_sweep":
        f1 = f1 * (0.5 + 0.5 * math.sin(t * 0.3 * anim_speed))
        f2 = f2 * (0.5 + 0.5 * math.cos(t * 0.4 * anim_speed))
        f3 = f3 * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed + 1.0))
    elif anim_mode == "phase_drift":
        t = t + 0.5 * math.sin(t * 0.2 * anim_speed)  # phase offset drift
    elif anim_mode == "modulation_cycle":
        md = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 0.3 * anim_speed))
        mf = mf * (0.5 + 0.5 * math.cos(t * 0.4 * anim_speed))
    elif anim_mode == "layout_morph":
        # Cross-fade between wave types
        wt_idx = int(t * 0.2 * anim_speed) % 14
        wt_list = ["sine", "sawtooth", "square", "triangle", "pulse", "am_modulated",
                    "fm_modulated", "noise_floor", "lissajous", "harmonic_series",
                    "interference", "phase_space", "wavetable", "granular"]
        wt = wt_list[wt_idx]

    from ...core.utils import PALETTES, quantize_to_palette
    pal = PALETTES.get(pal_name, [])
    img = np.zeros((H, W, 3), dtype=np.float32)
    if bg_style == "dark":
        img[:, :, :] = 0.05
    elif bg_style == "light":
        img[:, :, :] = 0.9
    elif bg_style == "grid":
        img[:, :, :] = 0.05
        if _has_cv2:
            for x in range(0, W, 20):
                cv2.line(img, (x, 0), (x, H), (0.1, 0.1, 0.1), 1)
            for y in range(0, H, 20):
                cv2.line(img, (0, y), (W, y), (0.1, 0.1, 0.1), 1)
    elif bg_style == "gradient":
        yy, xx = np.ogrid[:H, :W]
        img[:, :, 0] = yy / H * 0.1
        img[:, :, 1] = xx / W * 0.08
        img[:, :, 2] = 0.05
    elif bg_style == "scanline":
        img[:, :, :] = 0.05
        for y in range(0, H, 3):
            img[y:y + 1, :] = 0.08
    def wave_val(x,t):
        if wt=="sine": return math.sin(x*f1*0.1+t)*0.5+math.sin(x*f2*0.1+t*1.3)*0.3+math.sin(x*f3*0.1+t*0.5)*0.2
        if wt=="sawtooth": return 2*((x*f1*0.01+t)%1)-1
        if wt=="square": return 1 if math.sin(x*f1*0.1+t)>0 else -1
        if wt=="triangle": return 2*abs(2*((x*f1*0.01+t)%1)-1)-1
        if wt=="pulse": return 1 if (x*f1*0.01+t)%1<pw else -1
        if wt=="am_modulated": return math.sin(x*f1*0.1+t)*(1+md*math.sin(x*mf*0.1+t))
        if wt=="fm_modulated": return math.sin(x*f1*0.1+t+md*math.sin(x*mf*0.1+t))
        if wt=="noise_floor": return rng.uniform(-1,1)*nl+math.sin(x*f1*0.1+t)*0.3
        if wt=="lissajous": return math.sin(x*f1*0.01+t)*0.5+math.cos(x*f2*0.01+t)*0.5
        if wt=="harmonic_series":
            v=0
            for h in range(1,9): v+=math.sin(x*f1*0.1*h+t)/h
            return v*0.5
        if wt=="interference": return math.sin(x*f1*0.1+t)*math.cos(x*(f1+0.5)*0.1+t)
        if wt=="phase_space": return math.sin(x*f1*0.1+t)*math.sin((x+10)*f1*0.1+t)
        if wt=="wavetable":
            s = math.sin(x*f1*0.1+t); sw = 2*((x*f1*0.01+t)%1)-1
            return s*(1-md)+sw*md
        if wt=="granular":
            env = max(0,math.sin((x%20)/20*math.pi))
            return env*math.sin(x*f1*0.1+t)
        return math.sin(x*f1*0.1+t)
    if layout=="single":
        pts = []
        for x in range(W):
            v = wave_val(x,t)+rng.uniform(-nl,nl)
            y = int(H/2+v*ar*H/2)
            pts.append((x,y))
        if style=="line":
            for i in range(len(pts)-1): cv2.line(img,pts[i],pts[i+1],(0.8,0.6,0.1),lw)
        elif style=="gradient_fill":
            for i in range(len(pts)-1): cv2.line(img,pts[i],pts[i+1],(0.8,0.6,0.1),lw)
            pts2 = [(0,H//2)]+pts+[(W-1,H//2)]
            cv2.fillPoly(img,[np.array(pts2,dtype=np.int32)],(0.8,0.6,0.1,fa) if False else (0.8*fa,0.6*fa,0.1*fa))
        elif style=="oscilloscope":
            for i in range(len(pts)-1): cv2.line(img,pts[i],pts[i+1],(0.2,0.8,0.2),lw)
            cv2.GaussianBlur(img,(0,0),sigmaX=3,dst=img)
        elif style=="neon_tube":
            for i in range(len(pts)-1): cv2.line(img,pts[i],pts[i+1],(0.8,0.6,0.2),lw+2)
            for i in range(len(pts)-1): cv2.line(img,pts[i],pts[i+1],(1,0.9,0.5),1)
        elif style=="heat_wave":
            for i in range(len(pts)-1):
                v = pts[i][1]/H; cv2.line(img,pts[i],pts[i+1],(v,0.3,1-v),lw)
        elif style=="particle_trace":
            for x in range(0,W,3):
                v = wave_val(x,t)+rng.uniform(-nl,nl); y = int(H/2+v*ar*H/2)
                cv2.circle(img,(x,y),1,(0.8,0.6,0.1),-1)
        elif style=="filled_wave":
            pts2 = [(0,H)]+pts+[(W-1,H)]
            cv2.fillPoly(img,[np.array(pts2,dtype=np.int32)],(0.8*fa,0.6*fa,0.1*fa))
    elif layout=="multi_track":
        for tr in range(nt):
            f = f1+tr*2; pts = []
            for x in range(W):
                v = math.sin(x*f*0.1+t+tr*0.5)+rng.uniform(-nl,nl)
                y = int(H*(tr+0.5)/nt+v*ar*H/(nt*2))
                pts.append((x,y))
            col = [0.8-tr*0.1,0.6-tr*0.05,0.1+tr*0.1] if not pal else [c/255.0 for c in pal[tr%len(pal)]]
            for i in range(len(pts)-1): cv2.line(img,pts[i],pts[i+1],col,lw)
    elif layout=="stereo_pair":
        for ch in range(2):
            pts = []
            for x in range(W):
                v = math.sin(x*(f1+ch*2)*0.1+t)+rng.uniform(-nl,nl)
                y = int(H*(ch+0.5)/2+v*ar*H/4)
                pts.append((x,y))
            col = [0.8,0.6,0.1] if ch==0 else [0.2,0.6,0.8]
            for i in range(len(pts)-1): cv2.line(img,pts[i],pts[i+1],col,lw)
        cv2.line(img,(0,H//2),(W-1,H//2),(0.3,0.3,0.3),1)
    elif layout=="circular":
        cx,cy = W//2,H//2; r = min(W,H)//3
        for a in range(360):
            th = a*math.pi/180; v = wave_val(a*W//360,t)+rng.uniform(-nl,nl)
            rr = r*(1+v*ar*0.5); x = int(cx+rr*math.cos(th)); y = int(cy+rr*math.sin(th))
            if 0<=x<W and 0<=y<H: img[y,x] = [0.8,0.6,0.1]
    elif layout=="equalizer":
        for b in range(nb):
            v = abs(wave_val(b*W//nb,t))+rng.uniform(-nl,nl)
            h = int(v*ar*H/2); x0 = b*W//nb; x1 = (b+1)*W//nb
            col = [b/nb,0.3,1-b/nb]
            img[H//2-h:H//2+h, x0:x1] = col
    elif layout=="waterfall":
        for tr in range(20):
            pts = []
            for x in range(W):
                v = math.sin(x*(f1+tr*0.5)*0.1+t+tr*0.3)+rng.uniform(-nl,nl)
                y = int(H*(1-(tr/20)**0.7)+v*ar*H/20)
                pts.append((x,y))
            col = [0.8-tr*0.04,0.6-tr*0.02,0.1+tr*0.02]
            for i in range(len(pts)-1): cv2.line(img,pts[i],pts[i+1],col,1)
    if pal_name and pal_name in PALETTES: img = quantize_to_palette(img.clip(0,1),pal_name)
    capture_frame('65', img); save(img.clip(0,1), mn(65,"Waveform"), out_dir)

