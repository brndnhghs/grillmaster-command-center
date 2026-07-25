from __future__ import annotations
import subprocess
from pathlib import Path

from ...core.registry import method
from ...core.utils import save, mn, seed_all
from ...core.animation import capture_frame
from ..codegen.qr_code import method_09_qr_code


@method(id="25", name="qrencode", category="cli_tools", tags=["code", "expanded"],
        params={
            "qr_data": {"description": "QR code payload text", "default": "ImagePipeline v2: method 27 (QR Code)"},
            "module_size": {"description": "QR module size in pixels", "min": 1, "max": 20, "default": 8},
            "ecc_level": {"description": "QR error correction level (L/M/Q/H)", "default": "H"},
        })
def method_qrencode(out_dir: Path, seed: int, params=None):
    """Generate a QR code using the qrencode CLI tool, with pure-Python fallback.

    Uses the system `qrencode` binary for fast QR generation. Falls back to
    the pure-Python QR code method (#09) if the CLI tool is unavailable.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            qr_data: QR code payload text (default: "ImagePipeline v2: method 27 (QR Code)")
            module_size: QR module size in pixels, 1-20 (default: 8)
            ecc_level: Error correction level, L/M/Q/H (default: "H")
    """
    if params is None:
        params = {}
    seed_all(seed)
    qr_data = params.get("qr_data", "ImagePipeline v2: method 27 (QR Code)")
    module_size = int(params.get("module_size", 8))
    ecc_level = params.get("ecc_level", "H")
    try:
        subprocess.run(
            ["qrencode", "-o", str(out_dir / mn(27, "qrencode")), "-s", str(module_size), "-l", ecc_level, qr_data],
            capture_output=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    if (out_dir / mn(27, "qrencode")).exists():
        capture_frame("27", out_dir / mn(27, "qrencode"))
        print(f"  ✓ {mn(27, 'qrencode')}  ({(out_dir / mn(27, 'qrencode')).stat().st_size // 1024} KB)")
    else:
        # Fall back to pure-Python QR
        method_09_qr_code(out_dir, seed)
        import shutil
        shutil.copy(str(out_dir / mn(9, "QR Code")), str(out_dir / mn(27, "qrencode")))
        capture_frame("27", out_dir / mn(27, "qrencode"))
        print(f"  ✓ {mn(27, 'qrencode')} (fallback)")
