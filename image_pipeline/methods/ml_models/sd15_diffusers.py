"""
ML model methods — Stable Diffusion 1.5 (diffusers) and ComfyUI.
These require GPU or running services and may be slow/skip if unavailable.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, seed_all


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
