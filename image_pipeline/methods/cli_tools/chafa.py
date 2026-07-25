from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ...core.registry import method
from ...core.utils import seed_all, get_font, W, H
from ...core.animation import capture_frame


@method(
    id="45",
    name="Chafa",
    category="cli_tools",
    tags=["text", "caca", "expanded", "interactive"],
    inputs={
        "image_in": "IMAGE",
        "char_scale": "SCALAR",
    },
    outputs={
        "image": "IMAGE",
        "luminance": "FIELD",
    },
    params={
        "bg_color": {
            "description": "Background RGB tuple or hex color",
            "default": "0,0,0",
        },
        "text_color": {
            "description": "Text RGB tuple or hex color",
            "default": "255,255,255",
        },
        "chafa_symbols": {
            "description": "Chafa --symbols argument",
            "default": "all",
        },
        "char_scale": {
            "description": "Character density multiplier. Higher = more characters and finer detail.",
            "default": 1.0,
        },
        "invert": {
            "description": "Invert the rendered ASCII mask",
            "default": False,
        },
        "contrast": {
            "description": "Contrast applied to the ASCII mask",
            "default": 1.0,
        },
        "font_scale": {
            "description": "Scale of the rendered ASCII characters",
            "default": 1.0,
        },
    },
)
def method_chafa(
    out_dir: Path,
    seed: int,
    params=None,
):
    """
    Interactive Chafa ASCII renderer.

    Inputs:
        image_in:
            Source image.

        char_scale:
            Controls the density of the Chafa character output.

    Parameters:
        bg_color:
            Background color as RGB string or hex.

        text_color:
            ASCII character color as RGB string or hex.

        chafa_symbols:
            Character set passed to Chafa.

        char_scale:
            Character density multiplier.

        invert:
            Invert the ASCII mask.

        contrast:
            Increase or decrease ASCII mask contrast.

        font_scale:
            Scale the final rendered characters.

    Outputs:
        image:
            Rendered RGB image.

        luminance:
            Grayscale luminance field.
    """

    if params is None:
        params = {}

    # ---------------------------------------------------------
    # Seed
    # ---------------------------------------------------------

    seed = seed & 0xFFFF0000
    seed_all(seed)

    # ---------------------------------------------------------
    # Read upstream image
    # ---------------------------------------------------------

    input_img = params.get("_input_image")

    if input_img is None:
        print(
            "  ✗ chafa: no input image — "
            "requires image_in to be wired"
        )

        empty_image = np.zeros(
            (H, W, 3),
            dtype=np.float32,
        )

        empty_luminance = np.zeros(
            (H, W),
            dtype=np.float32,
        )

        return {
            "image": empty_image,
            "luminance": empty_luminance,
        }

    img = Image.fromarray(
        (
            np.clip(
                input_img,
                0,
                1,
            )
            * 255
        ).astype(np.uint8)
    )

    # ---------------------------------------------------------
    # Read interactive parameters
    # ---------------------------------------------------------

    char_scale = params.get(
        "char_scale",
        1.0,
    )

    try:
        char_scale = float(char_scale)
    except (
        ValueError,
        TypeError,
    ):
        char_scale = 1.0

    char_scale = float(
        np.clip(
            char_scale,
            0.1,
            4.0,
        )
    )

    chafa_symbols = params.get(
        "chafa_symbols",
        "all",
    )

    if not chafa_symbols:
        chafa_symbols = "all"

    invert = params.get(
        "invert",
        False,
    )

    if isinstance(invert, str):
        invert = invert.lower() in (
            "true",
            "1",
            "yes",
            "on",
        )
    else:
        invert = bool(invert)

    contrast = params.get(
        "contrast",
        1.0,
    )

    try:
        contrast = float(contrast)
    except (
        ValueError,
        TypeError,
    ):
        contrast = 1.0

    contrast = float(
        np.clip(
            contrast,
            0.25,
            4.0,
        )
    )

    font_scale = params.get(
        "font_scale",
        1.0,
    )

    try:
        font_scale = float(font_scale)
    except (
        ValueError,
        TypeError,
    ):
        font_scale = 1.0

    font_scale = float(
        np.clip(
            font_scale,
            0.5,
            2.0,
        )
    )

    # ---------------------------------------------------------
    # Parse colors
    # ---------------------------------------------------------

    bg_color = _parse_color(
        params.get(
            "bg_color",
            "0,0,0",
        ),
        fallback=(0, 0, 0),
    )

    text_color = _parse_color(
        params.get(
            "text_color",
            "255,255,255",
        ),
        fallback=(255, 255, 255),
    )

    # ---------------------------------------------------------
    # Run Chafa
    # ---------------------------------------------------------

    # Base width is 80 characters at scale 1.0.
    # Increasing char_scale produces finer detail.
    chafa_width = max(
        10,
        int(
            80
            * char_scale
        ),
    )

    chafa_out = _run_chafa(
        img=img,
        out_dir=out_dir,
        width=chafa_width,
        symbols=chafa_symbols,
    )

    # ---------------------------------------------------------
    # Fallback if Chafa is unavailable
    # ---------------------------------------------------------

    if not chafa_out:

        chafa_out = (
            "CHAFA UNAVAILABLE\n"
            "\n"
            "Install chafa"
        )

    # ---------------------------------------------------------
    # Render ASCII output
    # ---------------------------------------------------------

    result_arr = _render_ascii(
        text=chafa_out,
        bg_color=bg_color,
        text_color=text_color,
        contrast=contrast,
        invert=invert,
        font_scale=font_scale,
    )

    # ---------------------------------------------------------
    # Calculate luminance
    # ---------------------------------------------------------

    luminance = (
        0.2126 * result_arr[:, :, 0]
        + 0.7152 * result_arr[:, :, 1]
        + 0.0722 * result_arr[:, :, 2]
    ).astype(
        np.float32
    )

    # ---------------------------------------------------------
    # Capture animation frame
    # ---------------------------------------------------------

    capture_frame(
        "47",
        result_arr,
    )

    # ---------------------------------------------------------
    # Return outputs
    # ---------------------------------------------------------

    return {
        "image": result_arr,
        "luminance": luminance,
    }


def _parse_color(
    value,
    fallback,
):
    """
    Parse a color value.

    Supported formats:

        "255,255,255"

        "#FFFFFF"

        "#FFF"

        (255, 255, 255)

        [255, 255, 255]
    """

    # ---------------------------------------------------------
    # Tuple / list
    # ---------------------------------------------------------

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):

        if len(value) >= 3:

            try:
                return tuple(
                    int(
                        np.clip(
                            x,
                            0,
                            255,
                        )
                    )
                    for x in value[:3]
                )

            except (
                ValueError,
                TypeError,
            ):
                return fallback

    # ---------------------------------------------------------
    # String
    # ---------------------------------------------------------

    if isinstance(
        value,
        str,
    ):

        value = value.strip()

        # -----------------------------------------------------
        # Hex color
        # -----------------------------------------------------

        if value.startswith("#"):

            hex_value = value.lstrip("#")

            try:

                # Convert #FFF -> #FFFFFF
                if len(hex_value) == 3:

                    hex_value = "".join(
                        char * 2
                        for char in hex_value
                    )

                if len(hex_value) == 6:

                    return tuple(
                        int(
                            hex_value[i:i + 2],
                            16,
                        )
                        for i in (
                            0,
                            2,
                            4,
                        )
                    )

            except ValueError:
                return fallback

        # -----------------------------------------------------
        # RGB string
        # -----------------------------------------------------

        try:

            values = [
                int(
                    x.strip()
                )
                for x in value.split(",")
            ]

            if len(values) >= 3:

                return tuple(
                    int(
                        np.clip(
                            x,
                            0,
                            255,
                        )
                    )
                    for x in values[:3]
                )

        except (
            ValueError,
            TypeError,
        ):
            return fallback

    return fallback


def _run_chafa(
    img: Image.Image,
    out_dir: Path,
    width: int,
    symbols: str,
):
    """
    Run Chafa CLI and return its terminal output.
    """

    src = out_dir / "_chafa_src.png"

    try:

        # -----------------------------------------------------
        # Save temporary source image
        # -----------------------------------------------------

        img.save(
            str(src)
        )

        # -----------------------------------------------------
        # Execute Chafa
        # -----------------------------------------------------

        result = subprocess.run(
            [
                "chafa",
                str(src),
                "--symbols",
                symbols,
                "--size",
                str(width),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

        # -----------------------------------------------------
        # Successful output
        # -----------------------------------------------------

        if result.returncode == 0:

            return result.stdout

        print(
            "  ✗ chafa: process returned "
            f"exit code {result.returncode}"
        )

    except FileNotFoundError:

        print(
            "  ✗ chafa: executable not found"
        )

    except subprocess.TimeoutExpired:

        print(
            "  ✗ chafa: process timed out"
        )

    except OSError as e:

        print(
            f"  ✗ chafa: {e}"
        )

    finally:

        # -----------------------------------------------------
        # Clean up temporary file
        # -----------------------------------------------------

        src.unlink(
            missing_ok=True
        )

    return ""


def _render_ascii(
    text: str,
    bg_color,
    text_color,
    contrast: float,
    invert: bool,
    font_scale: float,
):
    """
    Render Chafa terminal text into a full-frame image.
    """

    # ---------------------------------------------------------
    # Parse lines
    # ---------------------------------------------------------

    lines = text.splitlines()

    if (
        not lines
        or all(
            not line.strip()
            for line in lines
        )
    ):

        lines = [
            "CHAFA",
            "",
            "No output",
        ]

    # ---------------------------------------------------------
    # Calculate character dimensions
    # ---------------------------------------------------------

    n_cols = max(
        len(line)
        for line in lines
    )

    n_rows = len(lines)

    # ---------------------------------------------------------
    # Calculate font size
    # ---------------------------------------------------------

    base_font_size = min(
        W / max(
            n_cols,
            1,
        ),
        H / max(
            n_rows,
            1,
        ),
    )

    font_size = int(
        np.clip(
            base_font_size
            * font_scale,
            6,
            48,
        )
    )

    # ---------------------------------------------------------
    # Load font
    # ---------------------------------------------------------

    font = get_font(
        font_size
    )

    bbox = font.getbbox(
        "A"
    )

    fw = max(
        4,
        bbox[2] - bbox[0],
    )

    fh = max(
        8,
        bbox[3] - bbox[1],
    )

    # ---------------------------------------------------------
    # Create monochrome mask
    # ---------------------------------------------------------

    mask = Image.new(
        "L",
        (
            W,
            H,
        ),
        0,
    )

    draw = ImageDraw.Draw(
        mask
    )

    # ---------------------------------------------------------
    # Draw ASCII characters
    # ---------------------------------------------------------

    for y, line in enumerate(lines):

        draw.text(
            (
                0,
                y * fh,
            ),
            line,
            fill=255,
            font=font,
        )

    # ---------------------------------------------------------
    # Apply contrast
    # ---------------------------------------------------------

    if contrast != 1.0:

        def adjust_contrast(pixel):

            normalized = (
                pixel / 255.0
            )

            adjusted = (
                (
                    normalized
                    - 0.5
                )
                * contrast
                + 0.5
            )

            return int(
                np.clip(
                    adjusted
                    * 255,
                    0,
                    255,
                )
            )

        mask = mask.point(
            adjust_contrast
        )

    # ---------------------------------------------------------
    # Apply inversion
    # ---------------------------------------------------------

    if invert:

        mask = ImageOps.invert(
            mask
        )

    # ---------------------------------------------------------
    # Apply foreground/background colors
    # ---------------------------------------------------------

    colored = ImageOps.colorize(
        mask,
        black=bg_color,
        white=text_color,
    )

    # ---------------------------------------------------------
    # Convert to normalized NumPy array
    # ---------------------------------------------------------

    result_arr = np.asarray(
        colored,
        dtype=np.float32,
    ) / 255.0

    return result_arr