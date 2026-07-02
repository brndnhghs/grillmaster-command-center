"""Transform — scale, rotate, translate, shear, flip, and corner-pin warp.

Accepts an IMAGE wire (image_in) or a FIELD wire (field_a — normalised to
greyscale before transform). All affine operations compose into a single matrix
so there is no quality loss from chaining scale + rotate + translate.

Corner-pin mode uses a full perspective homography
(cv2.getPerspectiveTransform) which is not limited to affine transforms —
each of the four corners can be repositioned freely.
"""
from __future__ import annotations
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.animation import capture_frame
from ...core.utils import save, W, H


@method(
    id="__transform__",
    name="Transform",
    category="filters",
    tags=["transform", "scale", "rotate", "translate", "warp", "perspective", "flip", "shear"],
    inputs={
        "image_in": "IMAGE",
        "field_a":  "FIELD",
    },
    outputs={
        "image":     "IMAGE",
        "luminance": "SCALAR",
        "field":     "FIELD",
    },
    params={
        # ── Scale ────────────────────────────────────────────────────
        "scale": {
            "description": "uniform scale (multiplied with scale_x / scale_y)",
            "min": 0.01, "max": 10.0, "default": 1.0,
        },
        "scale_x": {
            "description": "horizontal scale factor",
            "min": 0.01, "max": 10.0, "default": 1.0,
        },
        "scale_y": {
            "description": "vertical scale factor",
            "min": 0.01, "max": 10.0, "default": 1.0,
        },
        # ── Translate ────────────────────────────────────────────────
        "translate_x": {
            "description": "horizontal translation in pixels (positive = right)",
            "min": -2048.0, "max": 2048.0, "default": 0.0,
        },
        "translate_y": {
            "description": "vertical translation in pixels (positive = down)",
            "min": -2048.0, "max": 2048.0, "default": 0.0,
        },
        # ── Rotation ─────────────────────────────────────────────────
        "rotate": {
            "description": "rotation angle in degrees (positive = counter-clockwise)",
            "min": -360.0, "max": 360.0, "default": 0.0,
        },
        # ── Pivot ────────────────────────────────────────────────────
        "pivot_x": {
            "description": "rotation / scale pivot X (0 = left edge, 0.5 = centre, 1 = right edge)",
            "min": 0.0, "max": 1.0, "default": 0.5,
        },
        "pivot_y": {
            "description": "rotation / scale pivot Y (0 = top edge, 0.5 = centre, 1 = bottom edge)",
            "min": 0.0, "max": 1.0, "default": 0.5,
        },
        # ── Shear / skew ─────────────────────────────────────────────
        "shear_x": {
            "description": "horizontal shear (skews along X)",
            "min": -2.0, "max": 2.0, "default": 0.0,
        },
        "shear_y": {
            "description": "vertical shear (skews along Y)",
            "min": -2.0, "max": 2.0, "default": 0.0,
        },
        # ── Flip ─────────────────────────────────────────────────────
        "flip_h": {
            "description": "flip horizontally (mirror left ↔ right)",
            "default": False,
        },
        "flip_v": {
            "description": "flip vertically (mirror top ↔ bottom)",
            "default": False,
        },
        # ── Corner pin ────────────────────────────────────────────────
        "corner_pin": {
            "description": "enable four-corner perspective warp (overrides affine)",
            "default": False,
        },
        "pin_tl_x": {
            "description": "top-left corner X destination (normalised 0–1)",
            "min": -1.0, "max": 2.0, "default": 0.0,
        },
        "pin_tl_y": {
            "description": "top-left corner Y destination (normalised 0–1)",
            "min": -1.0, "max": 2.0, "default": 0.0,
        },
        "pin_tr_x": {
            "description": "top-right corner X destination (normalised 0–1)",
            "min": -1.0, "max": 2.0, "default": 1.0,
        },
        "pin_tr_y": {
            "description": "top-right corner Y destination (normalised 0–1)",
            "min": -1.0, "max": 2.0, "default": 0.0,
        },
        "pin_bl_x": {
            "description": "bottom-left corner X destination (normalised 0–1)",
            "min": -1.0, "max": 2.0, "default": 0.0,
        },
        "pin_bl_y": {
            "description": "bottom-left corner Y destination (normalised 0–1)",
            "min": -1.0, "max": 2.0, "default": 1.0,
        },
        "pin_br_x": {
            "description": "bottom-right corner X destination (normalised 0–1)",
            "min": -1.0, "max": 2.0, "default": 1.0,
        },
        "pin_br_y": {
            "description": "bottom-right corner Y destination (normalised 0–1)",
            "min": -1.0, "max": 2.0, "default": 1.0,
        },
        # ── Edge handling ─────────────────────────────────────────────
        "edge_mode": {
            "description": "fill mode for pixels pulled from outside the source boundary",
            "default": "clamp",
            "choices": ["clamp", "wrap", "reflect", "transparent"],
        },
        # ── Interpolation ─────────────────────────────────────────────
        "interpolation": {
            "description": "pixel interpolation quality",
            "default": "bilinear",
            "choices": ["nearest", "bilinear", "bicubic", "lanczos"],
        },
    },
)
def method_transform(out_dir: Path, seed: int, params=None) -> dict:
    if params is None:
        params = {}

    # ── Load input ─────────────────────────────────────────────────────
    arr = params.get("_input_image")  # ndarray injected by graph.py for image_in

    if arr is None:
        image_path = params.get("input_image", "")
        if image_path:
            pil = Image.open(image_path).convert("RGB")
            if pil.size != (W, H):
                pil = pil.resize((W, H), Image.LANCZOS)
            arr = np.array(pil, dtype=np.float32) / 255.0

    if arr is None:
        field_path = params.get("field_a_path", "")
        if field_path:
            field = np.load(field_path).astype(np.float32)
            fmin, fmax = field.min(), field.max()
            norm = (field - fmin) / (fmax - fmin + 1e-8)
            arr = np.stack([norm, norm, norm], axis=-1)

    if arr is None:
        arr = np.zeros((H, W, 3), dtype=np.float32)

    # Ensure (H, W, 3) float32 in [0, 1]
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[:2] != (H, W):
        pil = Image.fromarray((arr.clip(0, 1) * 255).astype(np.uint8))
        pil = pil.resize((W, H), Image.LANCZOS)
        arr = np.array(pil, dtype=np.float32) / 255.0

    # ── Read params ────────────────────────────────────────────────────
    scale   = float(params.get("scale",   1.0))
    scale_x = float(params.get("scale_x", 1.0)) * scale
    scale_y = float(params.get("scale_y", 1.0)) * scale

    tx = float(params.get("translate_x", 0.0))
    ty = float(params.get("translate_y", 0.0))

    angle = float(params.get("rotate", 0.0))

    pivot_x = float(params.get("pivot_x", 0.5)) * W
    pivot_y = float(params.get("pivot_y", 0.5)) * H

    shear_x = float(params.get("shear_x", 0.0))
    shear_y = float(params.get("shear_y", 0.0))

    def _bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)

    flip_h = _bool(params.get("flip_h", False))
    flip_v = _bool(params.get("flip_v", False))

    corner_pin = _bool(params.get("corner_pin", False))
    pin_tl_x = float(params.get("pin_tl_x", 0.0)) * W
    pin_tl_y = float(params.get("pin_tl_y", 0.0)) * H
    pin_tr_x = float(params.get("pin_tr_x", 1.0)) * W
    pin_tr_y = float(params.get("pin_tr_y", 0.0)) * H
    pin_bl_x = float(params.get("pin_bl_x", 0.0)) * W
    pin_bl_y = float(params.get("pin_bl_y", 1.0)) * H
    pin_br_x = float(params.get("pin_br_x", 1.0)) * W
    pin_br_y = float(params.get("pin_br_y", 1.0)) * H

    edge_mode    = params.get("edge_mode",    "clamp")
    interp_name  = params.get("interpolation", "bilinear")

    # ── Map to OpenCV constants ─────────────────────────────────────────
    interp = {
        "nearest":  cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic":  cv2.INTER_CUBIC,
        "lanczos":  cv2.INTER_LANCZOS4,
    }.get(interp_name, cv2.INTER_LINEAR)

    border = {
        "clamp":       cv2.BORDER_REPLICATE,
        "wrap":        cv2.BORDER_WRAP,
        "reflect":     cv2.BORDER_REFLECT_101,
        "transparent": cv2.BORDER_CONSTANT,
    }.get(edge_mode, cv2.BORDER_REPLICATE)

    border_val = (0.0, 0.0, 0.0) if edge_mode == "transparent" else 0

    # ── Apply transform ─────────────────────────────────────────────────
    src = arr  # float32 (H, W, 3)

    if corner_pin:
        # Perspective homography: specify where each source corner lands in the output
        src_pts = np.float32([
            [0,     0    ],
            [W - 1, 0    ],
            [W - 1, H - 1],
            [0,     H - 1],
        ])
        dst_pts = np.float32([
            [pin_tl_x, pin_tl_y],
            [pin_tr_x, pin_tr_y],
            [pin_br_x, pin_br_y],
            [pin_bl_x, pin_bl_y],
        ])
        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(
            src, M_persp, (W, H),
            flags=interp,
            borderMode=border,
            borderValue=border_val,
        )
    else:
        # Compose affine in homogeneous coords:
        # T_translate ∘ T_from_pivot ∘ S ∘ R ∘ Sh ∘ T_to_pivot
        # Flip is absorbed into the scale signs so it naturally respects the pivot.
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))

        sx = scale_x * (-1.0 if flip_h else 1.0)
        sy = scale_y * (-1.0 if flip_v else 1.0)

        T_to_pivot = np.float64([
            [1, 0, -pivot_x],
            [0, 1, -pivot_y],
            [0, 0,  1      ],
        ])
        M_shear = np.float64([
            [1,       shear_x, 0],
            [shear_y, 1,       0],
            [0,       0,       1],
        ])
        M_rotate = np.float64([
            [ cos_a, -sin_a, 0],
            [ sin_a,  cos_a, 0],
            [0,       0,     1],
        ])
        M_scale = np.float64([
            [sx, 0,  0],
            [0,  sy, 0],
            [0,  0,  1],
        ])
        T_from_pivot = np.float64([
            [1, 0, pivot_x],
            [0, 1, pivot_y],
            [0, 0, 1      ],
        ])
        T_translate = np.float64([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1 ],
        ])

        M_full = T_translate @ T_from_pivot @ M_scale @ M_rotate @ M_shear @ T_to_pivot
        M_affine = M_full[:2, :].astype(np.float32)

        result = cv2.warpAffine(
            src, M_affine, (W, H),
            flags=interp,
            borderMode=border,
            borderValue=border_val,
        )

    result = np.clip(result.astype(np.float32), 0.0, 1.0)

    # ── Build outputs ──────────────────────────────────────────────────
    lum_field  = result.mean(axis=2).astype(np.float32)
    lum_scalar = float(lum_field.mean())

    capture_frame("__transform__", result)
    save(result, "transform.png", out_dir)

    return {
        "image":     result,
        "luminance": lum_scalar,
        "field":     lum_field,
    }
