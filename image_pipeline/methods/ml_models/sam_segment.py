"""
ML model methods — Stable Diffusion 1.5 (diffusers) and ComfyUI.
These require GPU or running services and may be slow/skip if unavailable.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, load_input, write_scalars, write_mask


@method(
    id="__sam_segment__",
    name="SAM Segment",
    category="ml_models",
    tags=["ml", "sam", "segmentation", "mask", "utility"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "mask": "MASK", "score": "SCALAR"},
    params={
        "prompt": {
            "description": "text prompt for SAM 2 freetext / Grounding-style prompting (currently used as a fallback label)",
            "default": "an object",
        },
        "point_x": {
            "description": "click x (fraction 0..1 of width) for a foreground point prompt; -1 disables point prompting",
            "min": -1.0, "max": 1.0, "default": -1.0,
        },
        "point_y": {
            "description": "click y (fraction 0..1 of height) for a foreground point prompt; -1 disables point prompting",
            "min": -1.0, "max": 1.0, "default": -1.0,
        },
        "box": {
            "description": "bounding box [x0,y0,x1,y1] as fractions of width/height; 'none' disables box prompting",
            "default": "none",
        },
        "mode": {
            "description": "segmentation mode",
            "default": "automatic",
            "choices": ["automatic", "point", "box"],
        },
        "checkpoint": {
            "description": "SAM ViT checkpoint to download/cache on first use (vit_b / vit_l / vit_h)",
            "default": "vit_b",
            "choices": ["vit_b", "vit_l", "vit_h"],
        },
        "points_per_side": {
            "description": "SAM automatic mask density (more = more candidates, slower)",
            "min": 8, "max": 64, "default": 32,
        },
        "max_masks": {
            "description": "cap on candidate masks kept for the output (automatic mode)",
            "min": 1, "max": 100, "default": 40,
        },
        "device": {
            "description": "torch device for SAM (cpu/mps/cuda)",
            "default": "cpu",
        },
    },
    is_time_varying=False,
)
def method_sam_segment(out_dir: Path, seed: int, params=None):
    """Segment an input image into masks with Meta's Segment Anything Model (SAM).

    Takes an upstream IMAGE wire (or a wired input image) and produces a MASK
    plus a visualized IMAGE and a SCALAR coverage score:

      - ``mask``   (MASK):   the largest/strongest segmentation mask in [0, 1]
      - ``image``  (IMAGE):  the source image annotated with the mask outline
      - ``score``  (SCALAR): the IoU prediction of the chosen mask in [0, 1]
                            (automatic mode reports the covered-pixel fraction)

    Two prompting modes are supported:
      - ``automatic``: SamAutomaticMaskGenerator emits all masks; the largest
        by area is returned.
      - ``point`` / ``box``: a single prompted prediction via SamPredictor.

    The SAM ViT-B/L/H checkpoint is auto-downloaded and cached in
    ``~/.cache/sam_segment/`` on first use (mirrors how CLIP self-bootstraps).
    Requires the ``segment-anything`` package (already installed). If SAM or the
    checkpoint cannot be loaded, a clean fallback (solid-gray mask + gray image,
    score 0) is written so the graph keeps flowing.

    Params:
        prompt: fallback label / freetext (kept for parity with text prompts)
        point_x / point_y: foreground click as a fraction of the canvas (-1 = off)
        box: bounding box as fractions [x0,y0,x1,y1] or 'none'
        mode: automatic | point | box
        checkpoint: vit_b | vit_l | vit_h
        device: cpu/mps/cuda
    """
    if params is None:
        params = {}
    seed_all(seed)

    import numpy as _np

    ckpt_choice = params.get("checkpoint", "vit_b")
    device = params.get("device", "cpu")
    mode = params.get("mode", "automatic")
    prompt = params.get("prompt", "an object")
    px = float(params.get("point_x", -1.0))
    py = float(params.get("point_y", -1.0))
    box_raw = params.get("box", "none")
    # `points_per_side` is the SAM automatic-mode candidate density. It is a
    # declared, user-facing param (schema default 32). MUST be read here and
    # threaded into SamAutomaticMaskGenerator -- a prior hardcode of 32 made
    # the param a dead control AND forced the slow default (1024 candidates)
    # even when callers asked for less. On CPU this is the difference between
    # a ~seconds regression run and a multi-minute hang that stalls pytest.
    points_per_side = int(params.get("points_per_side", 32))

    # Checkpoint URLs (Meta's official hosted weights)
    _CKPT_URLS = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }
    _CKPT_FILES = {"vit_b": "sam_vit_b_01ec64.pth",
                   "vit_l": "sam_vit_l_0b3195.pth",
                   "vit_h": "sam_vit_h_4b8939.pth"}

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
    from PIL import Image as _PILImage
    rgb = base_u8[:, :, ::-1].copy()  # BGR for OpenCV-style SAM image

    # Default outputs (overwritten on success)
    out_mask = _np.zeros((int(H), int(W)), dtype=_np.float32)
    out_score = 0.0
    out_img = base_u8
    # Honesty flag: True ONLY when SAM genuinely produced a segmentation from a
    # real model run. Mirrors __clip_score__'s clip_ran. A silent failure (the
    # except branch below) leaves this False so a downstream consumer can tell
    # "model ran and found nothing" from "model fell back" — see SAM skill
    # ("verify the model actually ran") and grillmaster Pitfalls #18-#20.
    sam_ran = False

    try:
        import os
        import urllib.request
        import torch
        from segment_anything import (
            sam_model_registry,
            SamPredictor,
            SamAutomaticMaskGenerator,
        )

        cache_dir = os.path.expanduser("~/.cache/sam_segment")
        os.makedirs(cache_dir, exist_ok=True)
        ckpt_path = os.path.join(cache_dir, _CKPT_FILES[ckpt_choice])
        if not os.path.exists(ckpt_path):
            print(f"  ↯ SAM: downloading {ckpt_choice} checkpoint (~375MB-2.4GB)…")
            urllib.request.urlretrieve(_CKPT_URLS[ckpt_choice], ckpt_path)
            print(f"  ✓ SAM: cached {ckpt_path}")

        dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        sam = sam_model_registry[ckpt_choice](checkpoint=ckpt_path)
        sam.to(device=dev)
        sam.eval()

        if mode == "automatic":
            generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=points_per_side,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                crop_n_layers=1,
                min_mask_region_area=100,
            )
            masks = generator.generate(rgb)
            if masks:
                # Prefer a foreground mask. SAM's automatic mode tends to emit a
                # near-full-frame mask that represents the background; naive
                # "largest by area" selection returns that and makes the node
                # useless (it segments "everything"). Drop masks that cover more
                # than half the canvas, then pick the highest-quality remaining
                # mask by predicted IoU. Fall back to the best-overall mask only
                # if every mask is background-sized.
                total_px = float(rgb.shape[0] * rgb.shape[1])
                fg = [m for m in masks if m["segmentation"].sum() / total_px < 0.5]
                pool = fg if fg else masks
                best = max(pool, key=lambda m: m.get("predicted_iou", 0.0))
                out_mask = best["segmentation"].astype(_np.float32)
                # predicted_iou is SAM's sigmoid output; clamp to [0,1] so the
                # advertised SCALAR contract (score in [0,1]) holds even when the
                # model emits a value numerically >1 (e.g. 1.007).
                out_score = float(min(1.0, max(0.0, best.get("predicted_iou", out_mask.mean()))))
        else:
            predictor = SamPredictor(sam)
            predictor.set_image(rgb)
            if mode == "box" and box_raw and box_raw != "none":
                parts = [float(v) for v in str(box_raw).replace(",", " ").split()]
                if len(parts) == 4:
                    bw, bh = float(W), float(H)
                    input_box = _np.array([
                        parts[0] * bw, parts[1] * bh, parts[2] * bw, parts[3] * bh
                    ], dtype=_np.float32)
                    masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
                    idx = int(_np.argmax(scores))
                    out_mask = masks[idx].astype(_np.float32)
                    # predictor.predict() scores are raw mask-decoder logits and
                    # can exceed 1.0; clamp to the advertised SCALAR [0,1] contract.
                    out_score = float(min(1.0, max(0.0, float(scores[idx]))))
            else:  # point mode (or fallback to point when coordinates valid)
                if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0:
                    input_point = _np.array([[px * float(W), py * float(H)]], dtype=_np.float32)
                    input_label = _np.array([1], dtype=_np.int64)
                    masks, scores, _ = predictor.predict(
                        point_coords=input_point, point_labels=input_label, multimask_output=True)
                    idx = int(_np.argmax(scores))
                    out_mask = masks[idx].astype(_np.float32)
                    # predictor.predict() scores are raw logits; clamp to [0,1].
                    out_score = float(min(1.0, max(0.0, float(scores[idx]))))

        sam_ran = True  # reached the end of the model path with no exception
        print(f"  ✓ SAM Segment: mode={mode} score={out_score:.3f} "
              f"coverage={out_mask.mean():.3f} sam_ran={sam_ran}")
    except Exception as e:
        print(f"  ✗ SAM Segment: {e}")

    # ── Write scalar + mask outputs ──
    write_scalars(out_dir, score=float(out_score), coverage=float(out_mask.mean()),
                 sam_ran=1.0 if sam_ran else 0.0)
    write_mask(out_dir, out_mask)

    # ── Build the annotated visualization image ──
    vis = base_u8.copy()
    if out_mask.mean() > 0:
        ys, xs = _np.where(out_mask > 0.5)
        if len(xs):
            from PIL import ImageDraw
            accent = _np.array([255, 90, 170], dtype=_np.uint8)
            # Tint interior lightly
            interior = out_mask > 0.5
            vis[interior] = (vis[interior].astype(_np.float32) * 0.6 + accent * 0.4).astype(_np.uint8)
            # Outline the mask in a vivid accent color (dilate to find boundary)
            k = max(1, int(min(int(W), int(H)) * 0.004))
            dilated = _np.zeros_like(out_mask)
            dilated[max(0, ys.min() - k):ys.max() + k + 1,
                    max(0, xs.min() - k):xs.max() + k + 1] = 1.0
            edge = (dilated - out_mask) > 0.5
            vis[edge] = accent
            pil_vis = _PILImage.fromarray(vis)
            draw = ImageDraw.Draw(pil_vis)
            # Bounding-box badge of the segmentation
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            draw.rectangle([x0, y0, x1, y1], outline=tuple(accent.tolist()), width=2)
            vis = _np.asarray(pil_vis)

    save(vis, mn("__sam_segment__", "SAM Segment"), out_dir)
    print(f"  ✓ __sam_segment__ score={float(out_score):.3f} coverage={float(out_mask.mean()):.3f}")
