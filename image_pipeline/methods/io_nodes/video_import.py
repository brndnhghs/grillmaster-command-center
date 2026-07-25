"""Video Import — pull frame N from a video file as a graph source node."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, W, H


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
