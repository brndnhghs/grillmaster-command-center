"""
ML model methods — Stable Diffusion 1.5 (diffusers) and ComfyUI.
These require GPU or running services and may be slow/skip if unavailable.
"""
from __future__ import annotations
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from ..core.animation import capture_frame
from ..core.registry import method
from ..core.utils import save, mn, seed_all, W, H, load_input, write_scalars, write_field, write_mask


@method(id="21", name="SD1.5 (diffusers)", category="ml_models", tags=["ml", "slow", "gpu", "expanded"], timeout=300,
        params={
            "model_id": {"description": "HuggingFace model ID", "default": "runwayml/stable-diffusion-v1-5"},
            "device": {"description": "torch device (mps/cuda/cpu)", "default": "mps"},
            "prompt": {"description": "positive prompt text", "default": "oil painting of a computer workstation with a command-line terminal on screen showing fractal patterns, dramatic chiaroscuro lighting, neon blue and amber tones, keyboard with glowing keys, a complex generative algorithm visualization in progress, cyberpunk atmospheric aesthetic, detailed textures on desk surface, hyperrealistic render, cinematic composition"},
            "neg": {"description": "negative prompt text", "default": "text, watermark, signature, frame, border, cartoon, illustration, oversaturated, low quality, blurry, distorted, ugly, deformed, happy, peaceful, safe, warm, welcoming, bright daylight"},
            "width": {"description": "output width", "min": 64, "max": 1024, "default": 768},
            "height": {"description": "output height", "min": 64, "max": 1024, "default": 512},
            "guidance_scale": {"description": "CFG scale", "min": 1.0, "max": 20.0, "default": 8.0},
            "num_inference_steps": {"description": "denoising steps", "min": 5, "max": 100, "default": 30},
        })
def method_sd15(out_dir: Path, seed: int, params=None):
    """Generate an image using Stable Diffusion 1.5 via HuggingFace diffusers.

    Downloads the model on first run (cached afterward), runs inference on
    the specified device (mps/cuda/cpu), and saves the result. Requires
    torch, diffusers, and ~2GB disk for model weights.

    Params:
        model_id: HuggingFace model ID (default: runwayml/stable-diffusion-v1-5)
        device: torch device (mps/cuda/cpu)
        prompt: positive prompt text
        neg: negative prompt text
        width: output width (64-1024)
        height: output height (64-1024)
        guidance_scale: CFG scale (1.0-20.0)
        num_inference_steps: denoising steps (5-100)
    """
    if params is None:
        params = {}
    seed_all(seed)
    import torch
    from diffusers import StableDiffusionPipeline

    model_id = params.get("model_id", "runwayml/stable-diffusion-v1-5")
    device = params.get("device", "mps")
    prompt = params.get("prompt", (
        "oil painting of a computer workstation with a command-line terminal on screen "
        "showing fractal patterns, dramatic chiaroscuro lighting, neon blue and amber tones, "
        "keyboard with glowing keys, a complex generative algorithm visualization in progress, "
        "cyberpunk atmospheric aesthetic, detailed textures on desk surface, "
        "hyperrealistic render, cinematic composition"
    ))
    neg = params.get("neg", "text, watermark, signature, frame, border, cartoon, illustration, oversaturated, low quality, blurry, distorted, ugly, deformed, happy, peaceful, safe, warm, welcoming, bright daylight")
    img_width = params.get("width", 768)
    img_height = params.get("height", 512)
    guidance_scale = params.get("guidance_scale", 8.0)
    num_inference_steps = params.get("num_inference_steps", 30)

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
    except Exception as e:
        print(f"  ✗ SD1.5: failed to load model: {e}")
        return
    pipe.enable_attention_slicing()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    try:
        with torch.no_grad():
            img = pipe(
                prompt, negative_prompt=neg,
                width=img_width, height=img_height, guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps, generator=gen,
            ).images[0]
    except Exception as e:
        print(f"  ✗ SD1.5: inference failed: {e}")
        return
    save(img, mn(21, "SD1.5 (diffusers)"), out_dir)


@method(id="28", name="ComfyUI", category="ml_models", tags=["ml", "slow", "gpu", "expanded"], timeout=120,
        params={
            "comfy_dir": {"description": "ComfyUI base directory", "default": "~/Documents/ComfyUI"},
            "download_timeout": {"description": "seconds to wait for model download", "min": 60, "max": 600, "default": 300},
            "sampler_steps": {"description": "KSampler steps", "min": 1, "max": 150, "default": 30},
            "sampler_cfg": {"description": "CFG scale", "min": 1.0, "max": 20.0, "default": 8.0},
            "sampler_name": {"description": "sampler (euler, dpmpp_2m, etc.)", "default": "euler"},
            "scheduler": {"description": "scheduler (normal, karras, etc.)", "default": "normal"},
            "denoise": {"description": "denoise strength", "min": 0.0, "max": 1.0, "default": 1.0},
            "width": {"description": "generated image width", "min": 64, "max": 2048, "default": 768},
            "height": {"description": "generated image height", "min": 64, "max": 2048, "default": 512},
            "batch_size": {"description": "batch size", "min": 1, "max": 8, "default": 1},
            "prompt_positive": {"description": "positive prompt text", "default": "oil painting of a computer workstation with a command-line terminal on screen showing fractal patterns, dramatic chiaroscuro lighting, neon blue and amber tones, cyberpunk atmospheric aesthetic, hyperrealistic render, cinematic composition"},
            "prompt_negative": {"description": "negative prompt text", "default": "text, watermark, signature, frame, border, cartoon, illustration, oversaturated, low quality, blurry, distorted, ugly, deformed, happy, peaceful, safe, warm, welcoming, bright daylight"},
            "filename_prefix": {"description": "prefix for saved images", "default": "comfyui_v2"},
            "ports": {"description": "port(s) to check, comma-separated", "default": "8000,8188"},
            "api_timeout": {"description": "seconds to wait for API response", "min": 5, "max": 120, "default": 30},
            "queue_timeout": {"description": "seconds to wait for queue status", "min": 1, "max": 30, "default": 5},
            "poll_retries": {"description": "max queue poll attempts", "min": 1, "max": 120, "default": 30},
            "poll_interval": {"description": "seconds between queue polls", "min": 0.5, "max": 30, "default": 2},
        })
def method_comfyui(out_dir: Path, seed: int, params=None):
    """Generate an image using a running ComfyUI instance via its API.

    Connects to a local ComfyUI server on the specified port(s), submits a
    prompt workflow, and polls the queue until the image is generated. Falls
    back to downloading the SD1.5 checkpoint if none is found.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            comfy_dir: ComfyUI base directory (default: ~/Documents/ComfyUI)
            download_timeout: seconds to wait for model download (60-600)
            sampler_steps: KSampler steps (1-150)
            sampler_cfg: CFG scale (1.0-20.0)
            sampler_name: sampler (euler, dpmpp_2m, etc.)
            scheduler: scheduler (normal, karras, etc.)
            denoise: denoise strength (0.0-1.0)
            width: generated image width (64-2048)
            height: generated image height (64-2048)
            batch_size: batch size (1-8)
            prompt_positive: positive prompt text
            prompt_negative: negative prompt text
            filename_prefix: prefix for saved images
            ports: port(s) to check, comma-separated
            api_timeout: seconds to wait for API response (5-120)
            queue_timeout: seconds to wait for queue status (1-30)
            poll_retries: max queue poll attempts (1-120)
            poll_interval: seconds between queue polls (0.5-30)
    """
    if params is None:
        params = {}
    seed_all(seed)
    import urllib.request
    comfy_dir_raw = params.get("comfy_dir", "~/Documents/ComfyUI")
    comfy_dir = Path(comfy_dir_raw).expanduser()
    download_timeout = int(params.get("download_timeout", 300))
    sampler_steps = int(params.get("sampler_steps", 30))
    sampler_cfg = float(params.get("sampler_cfg", 8.0))
    sampler_name = params.get("sampler_name", "euler")
    scheduler = params.get("scheduler", "normal")
    denoise = float(params.get("denoise", 1.0))
    img_width = int(params.get("width", 768))
    img_height = int(params.get("height", 512))
    batch_size = int(params.get("batch_size", 1))
    prompt_positive = params.get("prompt_positive", "oil painting of a computer workstation with a command-line terminal on screen showing fractal patterns, dramatic chiaroscuro lighting, neon blue and amber tones, cyberpunk atmospheric aesthetic, hyperrealistic render, cinematic composition")
    prompt_negative = params.get("prompt_negative", "text, watermark, signature, frame, border, cartoon, illustration, oversaturated, low quality, blurry, distorted, ugly, deformed, happy, peaceful, safe, warm, welcoming, bright daylight")
    filename_prefix = params.get("filename_prefix", "comfyui_v2")
    ports_str = params.get("ports", "8000,8188")
    ports = [int(p.strip()) for p in ports_str.split(",")]
    api_timeout = int(params.get("api_timeout", 30))
    queue_timeout = int(params.get("queue_timeout", 5))
    poll_retries = int(params.get("poll_retries", 30))
    poll_interval = float(params.get("poll_interval", 2))

    ckpts = list((comfy_dir / "models" / "checkpoints").glob("*"))
    if not ckpts:
        print("  … downloading SD1.5 for ComfyUI (~1.7GB)")
        try:
            subprocess.run(
                [str(comfy_dir / ".venv" / "bin" / "python3"), "-m", "huggingface_hub",
                 "download", "--local-dir", str(comfy_dir / "models" / "checkpoints"),
                 "runwayml/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors"],
                capture_output=True, timeout=download_timeout,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  ✗ ComfyUI: download failed: {e}")
            return
        ckpts = list((comfy_dir / "models" / "checkpoints").glob("*"))
    if not ckpts:
        print("  ✗ ComfyUI: no checkpoint")
        return
    wf = {
        "3": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": sampler_steps, "cfg": sampler_cfg, "sampler_name": sampler_name, "scheduler": scheduler, "denoise": denoise, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpts[0].name}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": img_width, "height": img_height, "batch_size": batch_size}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {
            "text": prompt_positive,
            "clip": ["4", 1],
        }},
        "7": {"class_type": "CLIPTextEncode", "inputs": {
            "text": prompt_negative,
            "clip": ["4", 1],
        }},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]}},
    }
    for port in ports:
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/prompt",
                data=json.dumps({"prompt": wf}).encode(),
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=api_timeout)
            pid = json.loads(resp.read()).get("prompt_id", "")
            print(f"  … ComfyUI queued (port {port}): {pid}")
            for _ in range(poll_retries):
                time.sleep(poll_interval)
                try:
                    q = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/queue", timeout=queue_timeout).read())
                except Exception:
                    q = {"queue_running": True, "queue_pending": True}
                if not q.get("queue_running") and not q.get("queue_pending"):
                    break
            out = list(comfy_dir.glob(f"**/{filename_prefix}_*.png"))
            if out:
                shutil.copy(str(out[-1]), str(out_dir / mn(28, "ComfyUI")))
                # capture_frame() needs a numpy array in [0,1] (it calls
                # arr.copy()) — load the just-copied file via load_input.
                _cf_arr = load_input(str(out_dir / mn(28, "ComfyUI")), int(W), int(H))
                capture_frame("28", _cf_arr)
                print(f"  ✓ {mn(28, 'ComfyUI')}  ({out[-1].stat().st_size // 1024} KB)")
                return
            print("  ✗ ComfyUI: no output found")
        except Exception as e:
            print(f"  … ComfyUI port {port}: {e}")
    print("  ✗ ComfyUI: no running instance found — writing fallback")
    save(np.zeros((H, W, 3), dtype=np.float32), mn(28, "ComfyUI"), out_dir)


@method(
    id="__clip_score__",
    name="CLIP Score",
    category="ml_models",
    tags=["ml", "clip", "vision-language", "scoring", "utility"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "score": "SCALAR", "weights": "FIELD"},
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

        print(f"  ✓ SAM Segment: mode={mode} score={out_score:.3f} "
              f"coverage={out_mask.mean():.3f}")
    except Exception as e:
        print(f"  ✗ SAM Segment: {e}")

    # ── Write scalar + mask outputs ──
    write_scalars(out_dir, score=float(out_score), coverage=float(out_mask.mean()))
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
                # CLIP softmax prob is already in [0,1]; clamp defensively so the
                # advertised SCALAR [0,1] contract holds for any downstream consumer.
                out_score = float(min(1.0, max(0.0, float(probs[best_i]))))
                print(f"  ✓ CLIP-SAM: mask {best_i}/{len(valid)} "
                      f"clip_p={out_score:.3f} sam_iou={float(best.get('predicted_iou', 0)):.3f} "
                      f"cov={out_mask.mean():.3f}")
    except Exception as e:
        print(f"  ✗ CLIP-SAM: {e}")

    # ── Write scalar + mask outputs ──
    write_scalars(out_dir, score=float(out_score), coverage=float(out_mask.mean()))
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


# ── CLIP Semantic Palette (appended 2026-07-16) ──
@method(
    id="__clip_palette__",
    name="CLIP Palette",
    category="ml_models",
    tags=["ml", "clip", "vision-language", "recolor", "palette", "utility"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "palette": "SCALAR", "score": "SCALAR",
             "mask": "MASK", "weights": "FIELD"},
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
                  n_palettes=float(len(names)))
    col = probs.reshape(1, 1, -1)  # (1, 1, N)
    weights_field = _np.repeat(_np.repeat(col, int(W), axis=1), int(H), axis=0)
    write_field(out_dir, weights_field.astype(_np.float32))
    write_mask(out_dir, lum.astype(_np.float32))

    save(vis, mn("__clip_palette__", "CLIP Palette"), out_dir)
    print(f"  ✓ __clip_palette__ palette='{top_name}' ({top_score:.3f}) "
          f"mode={recolor_mode} n={len(names)}")