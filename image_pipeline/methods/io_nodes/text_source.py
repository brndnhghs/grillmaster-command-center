"""Text Source — emit a string onto a TEXT wire as a graph source node."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, W, H


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
            p = Path(__file__).resolve().parent.parent.parent / "output" / "assets" / file_path
        try:
            text = p.read_text(errors="replace")
        except OSError as e:
            text = f"[Text Source: cannot read {file_path}: {e}]"

    # Preview: render the first few lines so the node reads at a glance.
    from PIL import Image as _PIL, ImageDraw as _Draw
    from ...core.utils import get_font
    img = _PIL.new("RGB", (int(W), int(H)), (10, 10, 18))
    d = _Draw.Draw(img)
    f = get_font(max(8, int(H) // 12))
    for i, line in enumerate(text.splitlines()[:8]):
        d.text((6, 6 + i * (int(H) // 10)), line[:48], fill=(200, 200, 180), font=f)
    arr = np.array(img, dtype=np.float32) / 255.0

    save(arr, mn(0, "Text Source"), out_dir)
    return {"image": arr, "text": text}
