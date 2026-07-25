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
    id="__clip_sam__",
    name="CLIP-guided SAM",
    category="ml_models",
    tags=["ml", "clip", "sam", "segmentation", "vision-language", "utility"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "mask": "MASK", "score": "SCALAR"},
    params={
        "prompt": {
            "description": "text prompt CLIP uses to rank SAM's candidate masks (e.g. 'a red circle')",
            "default": "a red circle",
        },
        "prompt_prefix": {
            "description": "text prepended to the prompt (CLIP prompt template)",
            "default": "a photo of",
        },
        "checkpoint": {
            "description": "SAM ViT checkpoint (vit_b cached; vit_l/vit_h download on first use)",
            "default": "vit_b",
            "choices": ["vit_b", "vit_l", "vit_h"],
        },
        "points_per_side": {
            "description": "SAM automatic mask density (more = more candidates, slower)",
            "min": 8, "max": 64, "default": 32,
        },
        "max_masks": {
            "description": "cap on candidate masks scored by CLIP",
            "min": 1, "max": 100, "default": 40,
        },
        "device": {
            "description": "torch device for SAM + CLIP (cpu/mps/cuda)",
            "default": "cpu",
        },
        "model_name": {
            "description": "CLIP model id (RN50, ViT-B/32, ViT-B/16, ViT-L/14)",
            "default": "ViT-B/32",
        },
    },
    is_time_varying=False,
    timeout=300,
)
def method_clip_sam(out_dir: Path, seed: int, params=None):
    """Segment the object described by a text prompt using CLIP + SAM.

    SAM produces a pool of candidate masks; CLIP scores each mask's cropped
    region against a text prompt and the best-matching mask is selected. This
    turns a *language* query into a *segmentation mask* with no extra prompting
    — wire an image in, type what you want, get a MASK out.

    Outputs:
      - ``mask``   (MASK):   the CLIP-best candidate mask in [0, 1]
      - ``image``  (IMAGE):  the source image annotated with the chosen mask
      - ``score``  (SCALAR): CLIP probability of the chosen mask given the prompt

    Both models self-bootstrap: SAM's ViT checkpoint is cached in
    ``~/.cache/sam_segment/`` and CLIP's weights download on first use. If either
    model fails to load, a clean fallback (gray mask, score 0) is written so the
    graph keeps flowing.

    Params:
        prompt:        text query CLIP ranks masks against
        prompt_prefix: template prepended to the prompt
        checkpoint:    SAM ViT size (vit_b cached by default)
        points_per_side: SAM candidate density
        max_masks:     cap on candidates handed to CLIP
        device:        cpu/mps/cuda
        model_name:    CLIP architecture id
    """
    if params is None:
        params = {}
    seed_all(seed)

    import numpy as _np

    prompt = params.get("prompt", "a red circle")
    prompt_prefix = params.get("prompt_prefix", "a photo of")
    ckpt_choice = params.get("checkpoint", "vit_b")
    points_per_side = int(params.get("points_per_side", 32))
    max_masks = int(params.get("max_masks", 40))
    device = params.get("device", "cpu")
    model_name = params.get("model_name", "ViT-B/32")

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
    rgb = base_u8[:, :, ::-1].copy()  # BGR for OpenCV-style SAM image

    # Default outputs (overwritten on success)
    out_mask = _np.zeros((int(H), int(W)), dtype=_np.float32)
    out_score = 0.0
    # Honesty flag: True only when CLIP+SAM genuinely produced a segmentation
    # (mirrors __sam_segment__'s sam_ran and __clip_score__'s clip_ran).
    sam_ran = False

    try:
        import os
        import urllib.request
        import torch
        import clip
        from PIL import Image as _PILImage
        from segment_anything import (
            sam_model_registry,
            SamAutomaticMaskGenerator,
        )

        # ── CLIP: encode the text prompt once ──
        dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        clip_model, preprocess = clip.load(model_name, device=dev)
        text_tokens = clip.tokenize([f"{prompt_prefix} {prompt}".strip()]).to(dev)
        with torch.no_grad():
            text_feat = clip_model.encode_text(text_tokens)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)

        # ── SAM: generate candidate masks ──
        cache_dir = os.path.expanduser("~/.cache/sam_segment")
        os.makedirs(cache_dir, exist_ok=True)
        ckpt_path = os.path.join(cache_dir, _CKPT_FILES[ckpt_choice])
        if not os.path.exists(ckpt_path):
            print(f"  ↯ CLIP-SAM: downloading SAM {ckpt_choice} checkpoint (~375MB-2.4GB)…")
            urllib.request.urlretrieve(_CKPT_URLS[ckpt_choice], ckpt_path)
            print(f"  ✓ CLIP-SAM: cached {ckpt_path}")

        sam = sam_model_registry[ckpt_choice](checkpoint=ckpt_path)
        sam.to(device=dev)
        sam.eval()

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
            # Drop near-full-frame background masks (same heuristic as SAM node).
            total_px = float(rgb.shape[0] * rgb.shape[1])
            fg = [m for m in masks if m["segmentation"].sum() / total_px < 0.5]
            pool = fg if fg else masks
            pool = pool[:max_masks]

            # Score each candidate crop with CLIP.
            feats = []
            valid = []
            for m in pool:
                seg = m["segmentation"]
                ys, xs = _np.where(seg)
                if len(xs) < 10:
                    continue
                x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                pad = int(0.1 * (x1 - x0 + 1))
                x0 = max(0, x0 - pad)
                y0 = max(0, y0 - pad)
                x1 = min(int(W) - 1, x1 + pad)
                y1 = min(int(H) - 1, y1 + pad)
                crop = base_u8[y0:y1 + 1, x0:x1 + 1, :]
                crop_pil = _PILImage.fromarray(crop).resize((224, 224), _PILImage.BILINEAR)
                t = preprocess(crop_pil).unsqueeze(0).to(dev)
                with torch.no_grad():
                    f = clip_model.encode_image(t)
                feats.append(f)
                valid.append(m)

            if feats:
                F = torch.cat(feats, 0)
                F /= F.norm(dim=-1, keepdim=True)
                logits = (F @ text_feat.T).squeeze(1)  # (n,)
                probs = torch.softmax(logits, dim=0)
                best_i = int(probs.argmax())
                best = valid[best_i]
                out_mask = best["segmentation"].astype(_np.float32)
                sam_ran = True
                # CLIP softmax prob is already in [0,1]; clamp defensively so the
                # advertised SCALAR [0,1] contract holds for any downstream consumer.
                out_score = float(min(1.0, max(0.0, float(probs[best_i]))))
                print(f"  ✓ CLIP-SAM: mask {best_i}/{len(valid)} "
                      f"clip_p={out_score:.3f} sam_iou={float(best.get('predicted_iou', 0)):.3f} "
                      f"cov={out_mask.mean():.3f}")
    except Exception as e:
        print(f"  ✗ CLIP-SAM: {e}")

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
            accent = _np.array([90, 220, 255], dtype=_np.uint8)
            interior = out_mask > 0.5
            vis[interior] = (vis[interior].astype(_np.float32) * 0.6 + accent * 0.4).astype(_np.uint8)
            k = max(1, int(min(int(W), int(H)) * 0.004))
            dilated = _np.zeros_like(out_mask)
            dilated[max(0, ys.min() - k):ys.max() + k + 1,
                    max(0, xs.min() - k):xs.max() + k + 1] = 1.0
            edge = (dilated - out_mask) > 0.5
            vis[edge] = accent
            pil_vis = _PILImage.fromarray(vis)
            draw = ImageDraw.Draw(pil_vis)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(xs.max())
            draw.rectangle([x0, y0, x1, y1], outline=tuple(accent.tolist()), width=2)
            try:
                from PIL import ImageFont
                font = ImageFont.load_default()
                draw.text((x0, max(0, y0 - 14)), f'"{prompt}" {out_score:.2f}',
                          fill=(255, 255, 255), font=font)
            except Exception:
                pass
            vis = _np.asarray(pil_vis)

    save(vis, mn("__clip_sam__", "CLIP-guided SAM"), out_dir)
    print(f"  ✓ __clip_sam__ prompt='{prompt}' score={float(out_score):.3f} "
          f"coverage={float(out_mask.mean()):.3f}")
