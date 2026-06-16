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
from ..core.utils import save, mn, seed_all, W, H
from ..core.animation import capture_frame


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
    seed_all(seed)
    if params is None:
        params = {}
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

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.enable_attention_slicing()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        img = pipe(
            prompt, negative_prompt=neg,
            width=img_width, height=img_height, guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps, generator=gen,
        ).images[0]
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
    seed_all(seed)
    if params is None:
        params = {}
    import urllib.request
    comfy_dir_raw = params.get("comfy_dir", "~/Documents/ComfyUI")
    comfy_dir = Path(comfy_dir_raw).expanduser()
    download_timeout = params.get("download_timeout", 300)
    sampler_steps = params.get("sampler_steps", 30)
    sampler_cfg = params.get("sampler_cfg", 8.0)
    sampler_name = params.get("sampler_name", "euler")
    scheduler = params.get("scheduler", "normal")
    denoise = params.get("denoise", 1.0)
    img_width = params.get("width", 768)
    img_height = params.get("height", 512)
    batch_size = params.get("batch_size", 1)
    prompt_positive = params.get("prompt_positive", "oil painting of a computer workstation with a command-line terminal on screen showing fractal patterns, dramatic chiaroscuro lighting, neon blue and amber tones, cyberpunk atmospheric aesthetic, hyperrealistic render, cinematic composition")
    prompt_negative = params.get("prompt_negative", "text, watermark, signature, frame, border, cartoon, illustration, oversaturated, low quality, blurry, distorted, ugly, deformed, happy, peaceful, safe, warm, welcoming, bright daylight")
    filename_prefix = params.get("filename_prefix", "comfyui_v2")
    ports_str = params.get("ports", "8000,8188")
    ports = [int(p.strip()) for p in ports_str.split(",")]
    api_timeout = params.get("api_timeout", 30)
    queue_timeout = params.get("queue_timeout", 5)
    poll_retries = params.get("poll_retries", 30)
    poll_interval = params.get("poll_interval", 2)

    ckpts = list((comfy_dir / "models" / "checkpoints").glob("*"))
    if not ckpts:
        print("  … downloading SD1.5 for ComfyUI (~1.7GB)")
        subprocess.run(
            [str(comfy_dir / ".venv" / "bin" / "python3"), "-m", "huggingface_hub",
             "download", "--local-dir", str(comfy_dir / "models" / "checkpoints"),
             "runwayml/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors"],
            capture_output=True, timeout=download_timeout,
        )
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
                q = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/queue", timeout=queue_timeout).read())
                if not q.get("queue_running") and not q.get("queue_pending"):
                    break
            out = list(comfy_dir.glob(f"**/{filename_prefix}_*.png"))
            if out:
                shutil.copy(str(out[-1]), str(out_dir / mn(28, "ComfyUI")))
                print(f"  ✓ {mn(28, 'ComfyUI')}  ({out[-1].stat().st_size // 1024} KB)")
            else:
                print("  ✗ ComfyUI: no output found")
            return
        except Exception as e:
            print(f"  … ComfyUI port {port}: {e}")
    print("  ✗ ComfyUI: no running instance found")