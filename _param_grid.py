"""Generate a 5×5 parameter grid for a single method + input image."""
import sys, json, math, shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(".")

def gen_grid(method_id: str, param_key: str, values: list, input_img: str,
             label_fn=None, cols=5):
    """Generate tiles and composite into a labeled grid."""
    from image_pipeline.core.registry import get_meta, get_all
    # Ensure registry loaded
    import image_pipeline.methods.fractals
    import image_pipeline.methods.filters
    import image_pipeline.methods.simulations
    import image_pipeline.methods.patterns
    import image_pipeline.methods.codegen
    import image_pipeline.methods.math_art
    import image_pipeline.methods.cli_tools
    import image_pipeline.methods.ml_models

    meta = get_meta(method_id)
    if not meta:
        print(f"Unknown method: {method_id}")
        return
    label_fn = label_fn or str

    rows = math.ceil(len(values) / cols)
    tile_w, tile_h = 300, 300
    label_h = 40
    cell_h = tile_h + label_h

    grid_img = Image.new("RGB", (cols * tile_w, rows * cell_h), (15, 15, 25))
    draw_grid = ImageDraw.Draw(grid_img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
        small = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11)
    except OSError:
        font = ImageFont.load_default()
        small = font

    for idx, val in enumerate(values):
        row = idx // cols
        col = idx % cols
        x0, y0 = col * tile_w, row * cell_h

        # Generate with this param
        params = {"input_image": input_img, param_key: val}
        try:
            # Cap steps for performance on walker/simulation methods
            if param_key in ("walkers", "particles"):
                params["steps"] = 800
                params["max_steps"] = 800
            if param_key == "boids":
                params["frames"] = 120
            meta.fn(OUT, 420691337, params=params)
            out_fn = meta.filename()
            src = OUT / out_fn
            dst = OUT / f"_tmp_{method_id}_{idx}.png"
            if src.exists():
                shutil.copy2(str(src), str(dst))
                tile = Image.open(str(dst)).convert("RGB").resize((tile_w, tile_h), Image.LANCZOS)
            else:
                tile = Image.new("RGB", (tile_w, tile_h), (30, 20, 20))
                d = ImageDraw.Draw(tile)
                d.text((10, tile_h//2), "no output", fill=(200, 80, 80), font=small)
        except Exception as e:
            tile = Image.new("RGB", (tile_w, tile_h), (30, 20, 20))
            d = ImageDraw.Draw(tile)
            d.text((10, tile_h//2), str(e)[:60], fill=(200, 80, 80), font=small)

        grid_img.paste(tile, (x0, y0))

        # Label below tile
        label = label_fn(val)
        draw_grid.text((x0 + 6, y0 + tile_h + 6), label, fill=(180, 180, 200), font=font)

    # Header banner
    banner_h = 36
    final = Image.new("RGB", (grid_img.width, grid_img.height + banner_h), (15, 15, 25))
    final.paste(grid_img, (0, banner_h))
    d = ImageDraw.Draw(final)
    d.text((8, 6), f"#{meta.id}  {meta.name}  —  param: {param_key}", fill=(200, 180, 140), font=font)
    d.text((8, 22), f"input: 30-strange-attractor.png  |  {len(values)} tiles ({cols}×{rows})", fill=(120, 120, 140), font=small)

    out_path = OUT / f"grid_{method_id}_{param_key}.png"
    final.save(str(out_path))
    print(f"  ✓ {out_path.name}")
    return out_path


if __name__ == "__main__":
    method_id = sys.argv[1]
    param_key = sys.argv[2]
    values = json.loads(sys.argv[3])
    label = sys.argv[4] if len(sys.argv) > 4 else None
    gen_grid(method_id, param_key, values, "30-strange-attractor.png",
             label_fn=lambda v: str(v) if label is None else label.format(v=v))