"""
ML model methods — Stable Diffusion 1.5 (diffusers) and ComfyUI.
These require GPU or running services and may be slow/skip if unavailable.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, load_input, write_scalars, write_field, write_mask


# ── CLIP Semantic Palette (appended 2026-07-16) ──
@method(
    id="__clip_palette__",
    name="CLIP Palette",
    category="ml_models",
    tags=["ml", "clip", "vision-language", "recolor", "palette", "utility"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "palette": "SCALAR", "score": "SCALAR",
             "clip_discriminated": "SCALAR", "mask": "MASK", "weights": "FIELD"},
    params={
        "palettes": {
            "description": "one palette per line: 'Name: #hex1,#hex2,#hex3' (3-5 hex colors, dark→light). Named presets (fire/ocean/forest/neon/noir/sunset) are used when no hex is given.",
            "default": "warm fire: #2b0a02,#ff3b00,#ffd000,#fff3b0\n"
                        "cool ocean: #001b2e,#0077be,#00c2ff,#d6f7ff\n"
                        "forest: #0b1f0a,#2e7d32,#9ccc65,#e8f5c8\n"
                        "neon night: #0a0014,#bd00ff,#00fff2,#ffffff\n"
                        "film noir: #000000,#3a3a3a,#9e9e9e,#ffffff",
        },
        "prompt_prefix": {
            "description": "text prepended to each palette name before CLIP scoring",
            "default": "a photo of",
        },
        "recolor_mode": {
            "description": "how to remap the image to the chosen palette",
            "default": "ramp",
            "choices": ["ramp", "nearest", "tint"],
        },
        "device": {
            "description": "torch device for CLIP (cpu/mps/cuda)",
            "default": "cpu",
        },
        "model_name": {
            "description": "CLIP model id (RN50, ViT-B/32, ViT-B/16, ViT-L/14)",
            "default": "ViT-B/32",
        },
    },
    is_time_varying=False,
    timeout=120,
)
def method_clip_palette(out_dir: Path, seed: int, params=None):
    """Semantic palette assignment driven by CLIP zero-shot vision-language scoring.

    Takes an upstream IMAGE wire (or a wired input image) and a set of named
    palettes. CLIP scores the image against each palette's *name* (a text
    prompt), and the best-matching palette is used to recolor the image — a
    luminance-preserving remap (duotone/tritone style) chosen by what the
    image *means*, not by its raw pixels.

    This is the natural creative companion to ``__clip_score__`` (which only
    scores/labels) and ``__clip_sam__`` (which segments): here the CLIP
    understanding *drives a graphics operation*. Useful for auto-mood-matching a
    generative clip, or for building a CLIP-selected color story in the graph.

    Outputs:
      - ``image``   (IMAGE): the source image recolored to the winning palette
      - ``palette`` (SCALAR): index of the chosen palette (0..N-1)
      - ``score``   (SCALAR): CLIP softmax probability of the winning palette
      - ``mask``    (MASK):   the image luminance (reusable downstream)
      - ``weights`` (FIELD):  per-palette CLIP probabilities broadcast over canvas

    CLIP's ViT-B/32 weights are cached at ``~/.cache/clip/`` and load on
    first use. If CLIP cannot be imported or the image is missing, a clean
    fallback (gray passthrough + uniform weights) is written so the graph
    keeps flowing.
    """
    if params is None:
        params = {}
    seed_all(seed)

    import numpy as _np
    from PIL import Image as _PILImage

    # ── Named palette presets (used when a line gives no hex colors) ──
    _PRESETS = {
        "fire":   [(0.17, 0.04, 0.01), (1.0, 0.23, 0.0), (1.0, 0.82, 0.0), (1.0, 0.95, 0.69)],
        "ocean":  [(0.0, 0.11, 0.18), (0.0, 0.47, 0.74), (0.0, 0.76, 1.0), (0.84, 0.97, 1.0)],
        "forest": [(0.04, 0.12, 0.04), (0.18, 0.49, 0.20), (0.61, 0.80, 0.40), (0.91, 0.96, 0.78)],
        "neon":   [(0.04, 0.0, 0.08), (0.74, 0.0, 1.0), (0.0, 1.0, 0.95), (1.0, 1.0, 1.0)],
        "noir":   [(0.0, 0.0, 0.0), (0.23, 0.23, 0.23), (0.62, 0.62, 0.62), (1.0, 1.0, 1.0)],
        "sunset": [(0.10, 0.02, 0.20), (0.85, 0.18, 0.45), (1.0, 0.55, 0.20), (1.0, 0.88, 0.55)],
        "default": [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)],
    }

    def _parse_palettes(raw):
        out = []
        for ln in str(raw).splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ":" in ln:
                name, rest = ln.split(":", 1)
                name = name.strip()
                hexes = [h.strip() for h in rest.split(",") if h.strip()]
            else:
                name, hexes = ln, []
            colors = []
            for h in hexes:
                h = h.lstrip("#")
                if len(h) == 6:
                    try:
                        colors.append((int(h[0:2], 16) / 255.0,
                                     int(h[2:4], 16) / 255.0,
                                     int(h[4:6], 16) / 255.0))
                    except ValueError:
                        pass
            if not colors:
                key = next((k for k in _PRESETS if k in name.lower()), "default")
                colors = _PRESETS[key]
            out.append((name, colors))
        return out

    palettes_raw = params.get("palettes",
        "warm fire: #2b0a02,#ff3b00,#ffd000,#fff3b0\n"
        "cool ocean: #001b2e,#0077be,#00c2ff,#d6f7ff\n"
        "forest: #0b1f0a,#2e7d32,#9ccc65,#e8f5c8\n"
        "neon night: #0a0014,#bd00ff,#00fff2,#ffffff\n"
        "film noir: #000000,#3a3a3a,#9e9e9e,#ffffff")
    palette_list = _parse_palettes(palettes_raw)
    if not palette_list:
        palette_list = [("default", _PRESETS["default"])]

    prompt_prefix = params.get("prompt_prefix", "a photo of")
    recolor_mode = params.get("recolor_mode", "ramp")
    device = params.get("device", "cpu")
    model_name = params.get("model_name", "ViT-B/32")

    names = [n for n, _ in palette_list]
    texts = [f"{prompt_prefix} {n}".strip() for n in names]

    # ── Load the input image (wired input ALWAYS overrides internal gen) ──
    wired = params.get("input_image", "")
    arr = None
    if wired:
        try:
            arr = load_input(wired, int(W), int(H))
        except (FileNotFoundError, OSError, ValueError):
            arr = None
    if arr is None:
        arr = _np.full((int(H), int(W), 3), 0.5, dtype=_np.float32)

    base_u8 = (_np.clip(arr, 0.0, 1.0) * 255).astype(_np.uint8)
    lum = _np.clip(arr, 0.0, 1.0).mean(axis=-1)  # (H,W) luminance

    # ── Default outputs (overwritten on success) ──
    probs = _np.full(len(names), 1.0 / len(names), dtype=_np.float32)
    top_idx = 0
    top_score = float(probs.max())
    top_name = names[top_idx]
    # Honesty flag: True only when CLIP genuinely executed AND discriminated
    # (peaked above the uniform baseline). Mirrors __clip_score__'s
    # clip_discriminated. See clip skill best-practice #8.
    clip_discriminated = False

    try:
        import clip
        import torch
        dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        model, preprocess = clip.load(model_name, device=dev)
        pil_img = _PILImage.fromarray(base_u8)
        img_tensor = preprocess(pil_img).unsqueeze(0).to(dev)
        text_tokens = clip.tokenize(texts).to(dev)
        with torch.no_grad():
            image_features = model.encode_image(img_tensor)
            text_features = model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            logits = (image_features @ text_features.T).squeeze(0)
            probs_t = logits.softmax(dim=-1)
        probs = probs_t.cpu().numpy().astype(_np.float32)
        top_idx = int(probs.argmax())
        top_score = float(probs.max())
        top_name = names[top_idx]
        # Discrimination check (clip skill best-practice #8): a genuine CLIP
        # run must peak ABOVE the uniform baseline 1/n_cand. See __clip_score__.
        n_cand = probs.shape[-1]
        peak = float(probs.max())
        clip_discriminated = bool(peak > 1.0 / n_cand + 1e-3)
    except Exception as e:
        print(f"  ✗ CLIP Palette: {e}")

    # ── Recolor the image to the winning palette ──
    _, colors = palette_list[top_idx]
    C = _np.array(colors, dtype=_np.float32)  # (n, 3)
    n = C.shape[0]
    pos = _np.linspace(0.0, 1.0, n)
    out = _np.zeros((int(H), int(W), 3), dtype=_np.float32)
    if recolor_mode == "nearest":
        idx = _np.clip(_np.round(lum * (n - 1)).astype(int), 0, n - 1)
        out = C[idx]  # fancy-index → (H,W,3)
    elif recolor_mode == "tint":
        ramp = _np.zeros((int(H), int(W), 3), dtype=_np.float32)
        for c in range(3):
            ramp[..., c] = _np.interp(lum, pos, C[:, c])
        out = 0.45 * arr + 0.55 * ramp
    else:  # ramp
        for c in range(3):
            out[..., c] = _np.interp(lum, pos, C[:, c])
    out = _np.clip(out, 0.0, 1.0)
    vis = (out * 255).astype(_np.uint8)

    # ── Write scalar + field + mask outputs ──
    write_scalars(out_dir, palette=float(top_idx), score=top_score,
                  n_palettes=float(len(names)),
                  clip_discriminated=1.0 if clip_discriminated else 0.0)
    col = probs.reshape(1, 1, -1)  # (1, 1, N)
    weights_field = _np.repeat(_np.repeat(col, int(W), axis=1), int(H), axis=0)
    write_field(out_dir, weights_field.astype(_np.float32))
    write_mask(out_dir, lum.astype(_np.float32))

    save(vis, mn("__clip_palette__", "CLIP Palette"), out_dir)
    print(f"  ✓ __clip_palette__ palette='{top_name}' ({top_score:.3f}) "
          f"mode={recolor_mode} n={len(names)}")
