"""Batch regenerate all 42 param grids using smiley.png as input."""
import shutil, sys, glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path('.')
IM = "smiley.png"
font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 10)
tile_w, tile_h = 192, 160

# Method ID → (file_module, function_name)
METHOD_LOOKUP = {}
import importlib
for f in sorted(OUT.glob("image_pipeline/methods/*.py")):
    mod_name = f"image_pipeline.methods.{f.stem}"
    try:
        mod = importlib.import_module(mod_name)
        for name in dir(mod):
            if name.startswith("method_") and callable(getattr(mod, name)):
                METHOD_LOOKUP[name] = mod
    except Exception as e:
        print(f"  ! {mod_name}: {e}")

def get_fn(mid: str):
    """Resolve method_NN function from any module."""
    for name in [f"method_{int(mid)}", f"method_0{int(mid)}"]:
        # Try padded/unpadded variations
        pass
    # Try all method_* names and match any that contain this id
    for fname, mod in METHOD_LOOKUP.items():
        if fname == f"method_{mid}":
            return getattr(mod, fname)
    # Some methods have names not matching their id — just search registry
    from image_pipeline.core.registry import _registry
    meta = _registry.get(mid)
    if meta and meta.fn:
        return meta.fn
    raise ValueError(f"Method {mid} not found")

# Each entry: (method_id, param_key, values, label_template, extra_params)
grids = [
    ("01", "charset", ["@%#*+=-:. ","█▓▒░ ","●◆▲■○","MNHQ$OC?7>!:-;.","0123456789",
                        "ABCDEF","▏▎▍▌▋▊▉█","░▒▓█","▁▂▃▄▅▆▇█","◢◣◤◥",
                        "⣀⤶⣿","┌┐└┘│─","╱╲╳","✦✧","○◔◐◕●",
                        "♠♣♥♦","☰☱☲☳☴☵☶☷","◇◆◈◉","◉◎◌⋅","∴∷⋮",
                        "⋰⋱","▲△▴▵▼▽","◀◁◂◃","🬀🬁🬂🬃"], "{v}", {"sw": 12, "sh": 16}),
    ("02", "pw", [8,16,32,64,128], "{v}", {}),
    ("03", "curves", [10,30,60,100,200], "{v}", {}),
    ("04", "blur", [5,10,20,40,80], "σ={v}", {}),
    ("05", "octaves", [1,2,3,4,5], "{v}", {}),
    ("06", "elements", [10,30,60,100,200], "{v}", {}),
    ("08", "speckles", [10,30,60,100,200], "{v}", {}),
    ("09", "box_size", [2,4,6,8,12], "{v}", {}),
    ("10", "palette", [4,6,8,10,15], "{v}", {}),
    ("11", "stops", [2,3,4,5,6], "{v}", {}),
    ("12", "segments", [4,6,8,12,16], "{v}", {}),
    ("13", "freq_x", [0.1,0.2,0.3,0.5,1.0], "fx={v}", {}),
    ("14", "rectangles", [5,15,30,60,100], "{v}", {}),
    ("15", "title_size", [18,24,36,48,72], "{v}", {}),
    ("16", "hours", [6,12,24,48,72], "{v}h", {}),
    ("17", "shift_count", [5,15,30,50,80], "{v}", {}),
    ("18", "rule_set", [[1,2,3,4],[1,3,5,7],[2,4,6,8],[1,4,7],[3,5,6]], "rule", {}),
    ("19", "iterations", [2,3,4,5,6], "{v}", {}),
    ("20", "particles", [100,200,500,1000,2000], "{v}", {"steps": 200}),
    ("22", "font_size", [8,12,18,24,36], "{v}", {}),
    ("23", "spread", [1,3,5,10,20], "{v}", {}),
    ("29", "points", [20,40,80,150,300], "{v}", {}),
    ("34", "boids", [10,20,50,100,200], "{v}", {"frames": 60}),
    ("35", "particles", [500,2000,5000], "{v}", {"frames": 10}),
    ("37", "levels", [4,8,16,24,32], "{v}", {}),
    ("38", "max_offset", [20,40,60,80,100], "{v}", {}),
    ("39", "colors", [4,6,8,12,16], "{v}", {}),
    ("40", "threshold", [30,60,100,150,200], "{v}", {}),
    ("41", "radius", [3,5,7,11,15], "{v}", {}),
    ("42", "gamma", [0.5,1.0,2.2,3.0,4.0], "γ={v}", {}),
    ("43", "points", [1000,5000,20000], "{v}", {}),
    ("44", "circle_count", [10,25,50,100,200], "{v}", {}),
    ("47", "shape_count", [10,30,60,100,200], "{v}", {}),
    ("48", "ring1_center", [30,60,90,120,150], "r={v}", {}),
    ("57", "amplitude", [10,40,100], "{v}", {}),
    ("59", "corruption", [50,200,1000], "1/{v}", {}),
    ("63", "thread_step", [4,6,8,12,16], "{v}", {}),
    ("64", "dot_size", [2,3,4,6,8], "{v}", {}),
    ("65", "freq1", [0.5,1.0,2.0,3.0,5.0], "f={v}", {}),
    ("74", "strength", [0.002,0.005,0.01,0.02,0.05], "{v}", {}),
    ("76", "bits", [4,6,8,10,12], "{v}", {}),
    ("77", "sigma", [5,10,20,40,80], "σ={v}", {}),
    ("79", "walkers", [5,10,20,30,50], "{v}", {"steps": 600}),
    ("80", "tile_size", [4,8,16,32,64], "{v}", {}),
]

for mid, pk, vals, lblfmt, extra in grids:
    try:
        fn = get_fn(mid)
    except ValueError as e:
        print(f"  ✗ #{mid}: {e}")
        continue

    n = len(vals)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    w = cols * tile_w
    h = rows * (tile_h + 24)
    grid_img = Image.new("RGB", (w, h), (12, 12, 22))
    draw = ImageDraw.Draw(grid_img)

    for i, v in enumerate(vals):
        params = {"input_image": IM, pk: v, **extra}
        try:
            fn(OUT, 42, params=params)
        except Exception as e:
            print(f"  [{mid}] tile {i} {pk}={v}: {e}")
            continue

        matches = sorted(OUT.glob(f"{mid}-*.png"))
        src = matches[-1] if matches else None
        dst = OUT / f"_tmp_{mid}_{i}.png"
        if src and src.exists():
            shutil.copy2(str(src), str(dst))

    # Assemble grid
    for i, v in enumerate(vals):
        dst = OUT / f"_tmp_{mid}_{i}.png"
        x0 = (i % cols) * tile_w
        y0 = (i // cols) * (tile_h + 24)
        if dst.exists():
            tile = Image.open(str(dst)).convert("RGB").resize((tile_w, tile_h), Image.LANCZOS)
            grid_img.paste(tile, (x0, y0))
            lbl = str(v)[:20] if lblfmt == "{v}" else lblfmt.replace("{v}", str(v))[:20]
            draw.text((x0 + 6, y0 + tile_h + 4), lbl, fill=(180, 180, 200), font=font)
        else:
            draw.rectangle([x0, y0, x0+tile_w, y0+tile_h], fill=(30, 20, 20))

    grid_img.save(f"grid_{mid}_{pk}.png")
    print(f"  ✓ grid_{mid}_{pk}.png ({n} tiles)")

    for i in range(n):
        (OUT / f"_tmp_{mid}_{i}.png").unlink(missing_ok=True)

print(f"\nDone: {len(grids)} grids regenerated with smiley.png")