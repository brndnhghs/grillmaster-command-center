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

from ..core.registry import method
from ..core.utils import save, mn, seed_all, W, H, load_input, write_scalars, write_field


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
                capture_frame("28", out_dir / mn(28, "ComfyUI"))
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
    except Exception as e:
        print(f"  ✗ CLIP Score: {e}")
        top_label = label_list[int(probs.argmax())]

    # ── Write scalar + field outputs ──
    write_scalars(out_dir, score=top_score, n_labels=float(len(label_list)))
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