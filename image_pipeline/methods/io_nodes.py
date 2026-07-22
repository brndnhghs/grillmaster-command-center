"""Import source nodes — Image and Video.

Both are graph *source* nodes: they own their own ``inputs={}`` (no
``image_in`` port) and emit a single IMAGE + FIELD so they slot in anywhere a
generator node would.  They are pure processors — no fallback generation, no
dead params — reading bytes straight from disk into the canvas-sized ndarray
that the rest of the pipeline expects.

Image Import reads one still and emits it every frame (``is_time_varying=False``
— the output is fully determined by the file path, so the executor cooks it
once and reuses it).  Video Import pulls frame ``N`` from a video file, where
``N`` is the injected timeline frame (``is_time_varying=True`` — the output
advances per frame).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.registry import method
from ..core.utils import save, mn, W, H


# ═══════════════════════════════════════════════════════════════════════════
# 1. Image Import — load a still image from disk
# ═══════════════════════════════════════════════════════════════════════════

@method(
    id="__image_import__",
    name="Image Import",
    category="io",
    tags=["io", "import", "source", "image", "file"],
    new_image_contract=True,
    is_time_varying=False,
    inputs={},  # source node — no image_in port
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "file_path": {"content": True, 
            "description": "path to the source image (png/jpg/webp/bmp/tiff/gif)",
            "default": "",
        },
    },
)
def method_image_import(out_dir: Path, seed: int, params=None):
    """Load a still image from disk and emit it as the node's image output.

    The image is resized to the active canvas (W×H), exactly like
    ``load_input`` does for wired upstreams, so downstream nodes receive the
    same float32 [0,1] (H,W,3) array they expect.  No time read — the same
    file yields the same image on every frame.

    Outputs:
        image (IMAGE): the imported image, canvas-sized
        field (FIELD): the same array, for FIELD-input nodes
    """
    if params is None:
        params = {}
    path = (params.get("file_path") or "").strip()
    if not path:
        raise ValueError("Image Import: 'file_path' is empty")

    from PIL import Image as _PIL

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image Import: file not found: {p}")

    img = _PIL.open(str(p)).convert("RGB").resize((int(W), int(H)), _PIL.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    save(arr, mn(0, "Image Import"), out_dir)
    return {"image": arr, "field": arr}


# ═══════════════════════════════════════════════════════════════════════════
# 2. Video Import — pull frame N from a video file
# ═══════════════════════════════════════════════════════════════════════════

@method(
    id="__video_import__",
    name="Video Import",
    category="io",
    tags=["io", "import", "source", "video", "file", "frame"],
    new_image_contract=True,
    is_time_varying=True,
    inputs={},  # source node — no image_in port
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "file_path": {
            "description": "path to the source video (mp4/mov/webm/avi/mkv)",
            "default": "",
        },
        "loop": {
            "description": "wrap frame index at end of video (else hold last frame)",
            "choices": ["true", "false"],
            "default": "true",
        },
    },
)
def method_video_import(out_dir: Path, seed: int, params=None):
    """Pull frame N from a video file and emit it as the node's image output.

    ``N`` is the injected timeline frame (the executor sets ``params['frame']``
    for every node each frame), so the imported clip plays in sync with the
    rest of the graph.  When the timeline frame exceeds the video length, the
    index wraps (``loop=true``, default) or holds the final frame.

    Outputs:
        image (IMAGE): the extracted frame, canvas-sized
        field (FIELD): the same array, for FIELD-input nodes
    """
    if params is None:
        params = {}
    path = (params.get("file_path") or "").strip()
    if not path:
        raise ValueError("Video Import: 'file_path' is empty")

    import cv2

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video Import: file not found: {p}")

    frame_idx = int(params.get("frame", 0))
    loop = str(params.get("loop", "true")).lower() in ("true", "1", "yes")

    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise IOError(f"Video Import: cannot open video: {p}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total > 0:
            if loop:
                frame_idx = frame_idx % total
            else:
                frame_idx = min(frame_idx, total - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            # Seek failed (e.g. sparse keyframes) — retry from 0 and step.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(frame_idx):
                if not cap.grab():
                    break
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise IOError(f"Video Import: failed to read frame {frame_idx}")
    finally:
        cap.release()

    # BGR → RGB, resize to canvas.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image as _PIL
    arr = np.array(
        _PIL.fromarray(rgb).resize((int(W), int(H)), _PIL.LANCZOS),
        dtype=np.float32,
    ) / 255.0
    save(arr, mn(0, "Video Import"), out_dir)
    return {"image": arr, "field": arr}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Text Source — emit a string onto a TEXT wire
# ═══════════════════════════════════════════════════════════════════════════

@method(
    id="__text_source__",
    name="Text Source",
    category="io",
    tags=["io", "source", "text", "code", "content"],
    is_time_varying=False,
    inputs={},                       # source node — nothing upstream
    outputs={"image": "IMAGE", "text": "TEXT"},
    params={
        "text": {
            "content": True,
            "description": "text emitted on the TEXT port — prose, source code, or a path",
            "default": "hello",
        },
        "file_path": {
            "content": True,
            "description": "read the text from this file instead (overrides `text` when set); "
                           "relative paths resolve under output/assets/",
            "default": "",
        },
    },
    description="Emits a string on a TEXT port: drives QR payloads, typography "
                "copy, GLSL source, font and model paths.",
)
def method_text_source(out_dir: Path, seed: int, params=None):
    """Emit a string downstream, from a param or a file.

    The image output is a small legible preview so the node is visible on the
    canvas; the payload consumers care about is the TEXT port.
    """
    params = params or {}
    text = str(params.get("text", "hello"))

    file_path = str(params.get("file_path", "") or "").strip()
    if file_path:
        p = Path(file_path)
        if not p.is_absolute():
            # Resolve against the asset store the upload endpoint writes to, so
            # a wired path from an upload lands without the caller knowing where
            # that lives on disk.
            p = Path(__file__).resolve().parent.parent / "output" / "assets" / file_path
        try:
            text = p.read_text(errors="replace")
        except OSError as e:
            text = f"[Text Source: cannot read {file_path}: {e}]"

    # Preview: render the first few lines so the node reads at a glance.
    from PIL import Image as _PIL, ImageDraw as _Draw
    from ..core.utils import get_font
    img = _PIL.new("RGB", (int(W), int(H)), (10, 10, 18))
    d = _Draw.Draw(img)
    f = get_font(max(8, int(H) // 12))
    for i, line in enumerate(text.splitlines()[:8]):
        d.text((6, 6 + i * (int(H) // 10)), line[:48], fill=(200, 200, 180), font=f)
    arr = np.array(img, dtype=np.float32) / 255.0

    save(arr, mn(0, "Text Source"), out_dir)
    return {"image": arr, "text": text}
