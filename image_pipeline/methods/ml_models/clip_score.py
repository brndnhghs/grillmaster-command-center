"""
ML model methods — Stable Diffusion 1.5 (diffusers) and ComfyUI.
These require GPU or running services and may be slow/skip if unavailable.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, load_input, write_scalars, write_field


@method(
    id="__clip_score__",
    name="CLIP Score",
    category="ml_models",
    tags=["ml", "clip", "vision-language", "scoring", "utility"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "score": "SCALAR", "clip_ran": "SCALAR", "clip_discriminated": "SCALAR", "weights": "FIELD"},
    params={
        "labels": {
            "description": "one candidate label per line; the image is scored against each",
            "default": "a cat\na dog\na sunset\na cityscape\na fractal pattern",
        },
        "prompt_prefix": {
            "description": "text prepended to each label (CLIP prompt template)",
            "default": "a photo of",
        },
        "visualization": {
            "description": "how to visualize the CLIP weights over the image",
            "default": "heatmap",
            "choices": ["heatmap", "bars", "none"],
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
)
def method_clip_score(out_dir: Path, seed: int, params=None):
    """Score an input image against text labels with OpenAI CLIP.

    Takes an upstream IMAGE wire (or a wired input image) and a set of text
    labels. Runs zero-shot CLIP classification, normalizes the per-label
    logits to a probability distribution, and exposes:

      - ``score``   (SCALAR): the top (winning) label probability in [0, 1]
      - ``weights`` (FIELD): the per-label probability column, broadcast over
        the canvas so it can be wired into MASK/FIELD downstream ports
      - ``image``   (IMAGE): the source image annotated with a color-coded
        CLIP heatmap / label bars showing how strongly each label matches

    No training data is needed — CLIP is run directly from its pretrained
    weights (downloaded on first use). If CLIP cannot be imported or the
    image is missing, a clean fallback (gray field + uniform weights) is
    written so the graph keeps flowing.

    Params:
        labels: newline-separated candidate labels
        prompt_prefix: template prepended to each label
        visualization: heatmap | bars | none
        device: cpu/mps/cuda
        model_name: CLIP architecture id
    """
    if params is None:
        params = {}
    seed_all(seed)

    import numpy as _np

    labels_raw = params.get("labels", "a cat\na dog\na sunset\na cityscape\na fractal pattern")
    prompt_prefix = params.get("prompt_prefix", "a photo of")
    visualization = params.get("visualization", "heatmap")
    device = params.get("device", "cpu")
    model_name = params.get("model_name", "ViT-B/32")

    # Normalize labels to a list
    label_list = [ln.strip() for ln in str(labels_raw).splitlines() if ln.strip()]
    if not label_list:
        label_list = ["a photo"]

    # Build the per-label text prompts
    texts = [f"{prompt_prefix} {lb}".strip() for lb in label_list]

    # ── Load the input image (wired input ALWAYS overrides internal gen) ──
    wired = params.get("input_image", "")
    arr = None
    if wired:
        try:
            arr = load_input(wired, int(W), int(H))
        except (FileNotFoundError, OSError, ValueError):
            arr = None
    if arr is None:
        # No wired image — emit a neutral gray canvas so the node still produces
        # a valid (if meaningless) output rather than breaking the graph.
        arr = _np.full((int(H), int(W), 3), 0.5, dtype=_np.float32)

    # ── Default outputs (overwritten on success) ──
    probs = _np.full(len(label_list), 1.0 / len(label_list), dtype=_np.float32)
    top_score = float(probs.max())
    clip_ran = False  # set True only when CLIP genuinely executed
    clip_discriminated = False  # set True only when CLIP genuinely discriminated
    top_label = label_list[int(probs.argmax())]

    try:
        import clip
        import torch

        dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        model, preprocess = clip.load(model_name, device=dev)

        # Image → CLIP tensor
        from PIL import Image as _PILImage
        pil_img = _PILImage.fromarray((_np.clip(arr, 0.0, 1.0) * 255).astype(_np.uint8))
        img_tensor = preprocess(pil_img).unsqueeze(0).to(dev)
        text_tokens = clip.tokenize(texts).to(dev)

        with torch.no_grad():
            image_features = model.encode_image(img_tensor)
            text_features = model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # Cosine similarity per label
            logits = (image_features @ text_features.T).squeeze(0)
            probs_t = logits.softmax(dim=-1)
        probs = probs_t.cpu().numpy().astype(_np.float32)
        top_score = float(probs.max())
        top_idx = int(probs.argmax())
        top_label = label_list[top_idx]
        clip_ran = True
        # Discrimination check (clip skill best-practice #8): a genuine CLIP
        # run must peak ABOVE the uniform baseline 1/n_cand. A near-uniform
        # distribution means the model ran but expressed no real preference —
        # downstream consumers must be able to tell "ran + discriminated" from
        # "ran but effectively guessing".
        n_cand = probs.shape[-1]
        peak = float(probs.max())
        clip_discriminated = bool(peak > 1.0 / n_cand + 1e-3)
    except Exception as e:
        # Honest fallback: keep a valid (uniform) output so the graph keeps
        # flowing, but DO NOT claim CLIP scored the image. A uniform
        # distribution is the exact fingerprint of "model did not run" and
        # downstream consumers must be able to detect it via `clip_ran` == 0.
        print(f"  ✗ CLIP Score: {e} — emitting uniform fallback (clip_ran=0)")
        top_label = label_list[int(probs.argmax())]

    # ── Write scalar + field outputs ──
    # `clip_ran` is the honesty flag that lets a downstream node (or test)
    # distinguish a real CLIP embedding from the silent uniform fallback:
    # a genuine run always peaks ABOVE 1/n_labels; the fallback sits at exactly
    # 1/n_labels. See grillmaster-image-pipeline Pitfalls #18-#20 + CLIP skill
    # best-practice #8.
    write_scalars(
        out_dir,
        score=top_score,
        n_labels=float(len(label_list)),
        clip_ran=1.0 if clip_ran else 0.0,
        clip_discriminated=1.0 if clip_discriminated else 0.0,
    )
    # Field = per-label probability column broadcast over the canvas (H×W FIELD)
    col = probs.reshape(1, 1, -1)  # (1, 1, n_labels)
    weights_field = _np.repeat(_np.repeat(col, int(W), axis=1), int(H), axis=0)  # (H, W, n_labels)
    write_field(out_dir, weights_field.astype(_np.float32))

    # ── Build the annotated visualization image ──
    base = (_np.clip(arr, 0.0, 1.0) * 255).astype(_np.uint8).copy()
    if visualization in ("heatmap", "bars"):
        # Color map: per-label vivid palette, blended by probability
        rng = _np.random.default_rng(1234)
        palette = rng.uniform(0.2, 1.0, size=(len(label_list), 3)).astype(_np.float32)
        palette = palette / (palette.sum(axis=1, keepdims=True) + 1e-6)
        win = probs.reshape(1, 1, -1, 1)  # (1,1,n,1)
        tint = (win * palette[None, None, :, :]).sum(axis=2)  # (1,1,3)
        tint = _np.repeat(_np.repeat(tint, int(W), axis=1), int(H), axis=0)  # (H,W,3)

        if visualization == "heatmap":
            # Overlay tint at 55% opacity over the source image
            vis = (0.45 * base.astype(_np.float32) / 255.0 + 0.55 * tint) * 255.0
        else:  # bars
            vis = base.astype(_np.float32)
            # Draw a label-bar strip across the bottom 18% of the frame
            bh = max(8, int(int(H) * 0.18))
            bar_y0 = int(H) - bh
            n = len(label_list)
            bw = int(W) // n
            for i, p in enumerate(probs):
                x0 = i * bw
                x1 = (i + 1) * bw if i < n - 1 else int(W)
                c = (palette[i] * 255).astype(_np.uint8)
                vis[bar_y0:int(H), x0:x1] = c
                # Fill height by probability
                fill = int(bh * float(p))
                vis[bar_y0 + (bh - fill):int(H), x0:x1] = (vis[bar_y0 + (bh - fill):int(H), x0:x1] * 0.4).astype(_np.uint8)
        vis = _np.clip(vis, 0, 255).astype(_np.uint8)
    else:
        vis = base

    save(vis, mn("__clip_score__", "CLIP Score"), out_dir)
    print(f"  ✓ __clip_score__ top='{top_label}' ({top_score:.3f})  labels={len(label_list)}")
