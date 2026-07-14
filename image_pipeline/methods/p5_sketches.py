"""p5.js sketches — headless browser rendering via Playwright.

Each sketch is an HTML template with %CONFIG% injected as JSON.
The method launches headless Chromium, captures canvas frames,
and returns them as pipeline-compatible dicts.
"""
from __future__ import annotations
import io
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.registry import method
from ..core.utils import seed_all, W, H
from ..core.animation import capture_frame

# ── Sketch templates ──────────────────────────────────────────

SKETCHES: dict[str, str] = {}

SKETCHES["particle_swarm"] = r"""<html><body><script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"></script>
<script>
const C = %CONFIG%;
let particles = [];
function setup() {
  createCanvas(C.w, C.h);
  pixelDensity(1);
  randomSeed(C.seed);
  noiseSeed(C.seed);
  for (let i = 0; i < C.count; i++) {
    particles.push({x: random(C.w), y: random(C.h), vx: random(-1,1), vy: random(-1,1)});
  }
}
function draw() {
  background(10, 10, 18);
  for (let p of particles) {
    let angle = noise(p.x*0.01, p.y*0.01, frameCount*0.005) * TWO_PI * 2;
    p.vx += cos(angle) * 0.1;
    p.vy += sin(angle) * 0.1;
    p.vx *= 0.99; p.vy *= 0.99;
    p.x += p.vx; p.y += p.vy;
    if (p.x < 0) p.x = C.w;
    if (p.x > C.w) p.x = 0;
    if (p.y < 0) p.y = C.h;
    if (p.y > C.h) p.y = 0;
    stroke(200, 180, 100, 50);
    strokeWeight(1.5);
    point(p.x, p.y);
  }
}
</script></body></html>"""

SKETCHES["flow_field"] = r"""<html><body><script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"></script>
<script>
const C = %CONFIG%;
let particles = [];
function setup() {
  createCanvas(C.w, C.h);
  randomSeed(C.seed); noiseSeed(C.seed);
  for (let i = 0; i < C.count; i++) particles.push({x: random(C.w), y: random(C.h)});
}
function draw() {
  background(10, 10, 18);
  stroke(200, 180, 100, 30);
  strokeWeight(1);
  for (let p of particles) {
    let angle = noise(p.x*0.005, p.y*0.005, frameCount*0.002) * TAU * 2;
    p.x += cos(angle); p.y += sin(angle);
    if (p.x < 0 || p.x > C.w || p.y < 0 || p.y > C.h) { p.x = random(C.w); p.y = random(C.h); }
    point(p.x, p.y);
  }
}
</script></body></html>"""

SKETCHES["metaballs"] = r"""<html><body><script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"></script>
<script>
const C = %CONFIG%;
let balls = [];
function setup() {
  createCanvas(C.w, C.h);
  pixelDensity(1);
  randomSeed(C.seed);
  for (let i = 0; i < 8; i++) {
    balls.push({x: random(C.w), y: random(C.h), vx: random(-1,1), vy: random(-1,1), r: random(40, 100)});
  }
}
function draw() {
  loadPixels();
  for (let y = 0; y < C.h; y++) {
    for (let x = 0; x < C.w; x++) {
      let sum = 0;
      for (let b of balls) {
        let dx = x - b.x, dy = y - b.y;
        sum += b.r * b.r / (dx*dx + dy*dy);
      }
      let v = constrain(sum / 8, 0, 1);
      let idx = (y * C.w + x) * 4;
      pixels[idx] = v * 200;
      pixels[idx+1] = v * 150;
      pixels[idx+2] = v * 255;
      pixels[idx+3] = 255;
    }
  }
  updatePixels();
  for (let b of balls) {
    b.x += b.vx; b.y += b.vy;
    if (b.x < 0 || b.x > C.w) b.vx *= -1;
    if (b.y < 0 || b.y > C.h) b.vy *= -1;
  }
}
</script></body></html>"""

SKETCHES["typography"] = r"""<html><body><script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"></script>
<script>
const C = %CONFIG%;
function setup() {
  createCanvas(C.w, C.h);
  textFont('monospace');
  textSize(14);
  textAlign(CENTER, CENTER);
}
function draw() {
  background(10, 10, 18);
  fill(200, 180, 100);
  let chars = '@%#*+=-:. ';
  for (let y = 0; y < C.h; y += 16) {
    for (let x = 0; x < C.w; x += 10) {
      let n = noise(x*0.01, y*0.01, frameCount*0.01);
      let c = chars[floor(n * chars.length)];
      let bri = 100 + 155 * n;
      fill(bri, bri * 0.9, bri * 0.5);
      text(c, x + 5, y + 8);
    }
  }
}
</script></body></html>"""


@method(id="171", name="p5.js Sketch", category="p5_sketches",
        tags=["p5js", "webgl", "fast", "expanded"],
        inputs={},
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
            "sketch": {
                "description": "p5.js sketch name",
                "choices": list(SKETCHES.keys()),
                "default": "particle_swarm",
            },
            "count": {
                "description": "number of particles / elements",
                "min": 50, "max": 5000, "default": 500,
            },
            "n_frames": {
                "description": "frames to render (1 = single, >1 = animation)",
                "min": 1, "max": 300, "default": 1,
            },
        })
def method_p5(out_dir: Path, seed: int, params=None):
    """Run a p5.js sketch headlessly via Playwright.

    Writes the sketch HTML with CONFIG injected, opens it in a
    headless Chromium browser, captures canvas frames, returns them.

    Returns a single dict for n_frames=1, or a list of dicts for animation.
    """
    if params is None:
        params = {}
    seed_all(seed)

    # W/H arrive as _DynDim canvas-proxy objects. NumPy resolves them via
    # __index__, but Playwright JSON-serializes the new_page() viewport dict,
    # which raises "Object of type _DynDim is not JSON serializable". Coerce to
    # plain ints for all downstream use (config dict, np.zeros, viewport).
    # Read via globals() (not a bare `W = int(W)`) so we don't shadow the
    # module-level _DynDim import — the local name becomes an int while the
    # module global stays a _DynDim that still resolves the live canvas size.
    W = int(globals()["W"])
    H = int(globals()["H"])

    sketch_name = params.get("sketch", "particle_swarm")
    template = SKETCHES.get(sketch_name)
    if template is None:
        template = SKETCHES["particle_swarm"]

    n_frames = int(params.get("n_frames", 1))
    count = int(params.get("count", 500))

    # Build config
    config = {"w": W, "h": H, "count": count, "seed": seed}
    html = template.replace("%CONFIG%", str(config).replace("'", '"'))

    # Write HTML
    html_path = out_dir / f"sketch_{sketch_name}.html"
    html_path.write_text(html)

    # Launch headless browser and capture frames
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        img = np.zeros((H, W, 3), dtype=np.float32)
        return {"image": img, "luminance": 0.0}

    frames = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": W, "height": H})
        page.goto(f"file://{html_path}")

        for frame_idx in range(n_frames):
            page.wait_for_timeout(1000 // 30)

            # Fast pixel capture: canvas.toDataURL → base64 JPEG → numpy
            # Browser-optimized JPEG encoding is faster than raw pixel transfer
            data_url = page.evaluate("""() => {
                const c = document.querySelector('canvas');
                if (!c) return null;
                return c.toDataURL('image/jpeg', 0.85);
            }""")

            if data_url:
                import base64
                jpeg_bytes = base64.b64decode(data_url.split(',', 1)[1])
                img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
                arr = np.array(img, dtype=np.float32) / 255.0
            else:
                arr = np.zeros((H, W, 3), dtype=np.float32)

            capture_frame("83", arr)
            frames.append({"image": arr, "luminance": float(np.mean(arr))})

        browser.close()

    if n_frames == 1:
        return frames[0]
    return frames
