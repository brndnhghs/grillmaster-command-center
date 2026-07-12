"""Shared utilities for all methods — saving, normalization, naming."""
from __future__ import annotations
import contextvars as _cv
import math
import operator as _operator
import random
from io import BytesIO
from pathlib import Path

# cv2 imported lazily inside functions that need it
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps


# ── Per-job canvas context ─────────────────────────────────────────────
# All ~160 node files do ``from ...core.utils import W, H``.  These imports
# run once at module-load time and bind the local names to the _DynDim
# objects below.  _DynDim deliberately does NOT subclass int: CPython
# fast-paths PyLong_Check for int subclasses, bypassing __index__ and
# __int__ overrides and reading the stored C-level integer directly.  As a
# plain non-int class, every C extension that needs an integer calls the
# Python-level nb_index / nb_int slot bridges, which honour our overrides
# and resolve from the active ContextVar.  This makes the following work:
#
#   np.zeros((H, W, 3))           → __index__ → dynamic ✓
#   range(W)                       → __index__ → dynamic ✓
#   cv2.warpAffine(…, (W, H))     → __int__   → dynamic ✓
#   pil.resize((W, H))            → __int__   → dynamic ✓
#   W // 2,  W - 1,  W * 0.5 …   → arithmetic overrides → plain int ✓
#
# Perf: ContextVar.get() is O(1) ~100 ns; only called inside node bodies.

CANVAS_DEFAULT: tuple[int, int] = (768, 512)
_CANVAS: _cv.ContextVar[tuple[int, int]] = _cv.ContextVar(
    "_canvas", default=CANVAS_DEFAULT
)


class _DynDim:
    """Canvas dimension proxy that resolves from the active canvas ContextVar."""
    __slots__ = ("_idx",)

    def __init__(self, idx: int):
        self._idx = idx

    def _val(self) -> int:
        return _CANVAS.get()[self._idx]

    # ── C-extension protocol hooks ───────────────────────────────────
    def __index__(self):         return self._val()
    def __int__(self):           return self._val()
    def __float__(self):         return float(self._val())
    def __bool__(self):          return bool(self._val())

    # ── Arithmetic — always return plain int so callers stay unaware ─
    def _o(self, o): return o._val() if isinstance(o, _DynDim) else o

    def __add__(self, o):         return self._val() + self._o(o)
    def __radd__(self, o):        return self._o(o) + self._val()
    def __sub__(self, o):         return self._val() - self._o(o)
    def __rsub__(self, o):        return self._o(o) - self._val()
    def __mul__(self, o):         return self._val() * self._o(o)
    def __rmul__(self, o):        return self._o(o) * self._val()
    def __floordiv__(self, o):    return self._val() // self._o(o)
    def __rfloordiv__(self, o):   return self._o(o) // self._val()
    def __truediv__(self, o):     return self._val() / self._o(o)
    def __rtruediv__(self, o):    return self._o(o) / self._val()
    def __mod__(self, o):         return self._val() % self._o(o)
    def __rmod__(self, o):        return self._o(o) % self._val()
    def __pow__(self, o, m=None): return pow(self._val(), self._o(o), m)
    def __neg__(self):            return -self._val()
    def __pos__(self):            return +self._val()
    def __abs__(self):            return abs(self._val())
    def __lshift__(self, o):      return self._val() << self._o(o)
    def __rshift__(self, o):      return self._val() >> self._o(o)
    def __and__(self, o):         return self._val() & self._o(o)
    def __or__(self, o):          return self._val() | self._o(o)
    def __xor__(self, o):         return self._val() ^ self._o(o)

    # ── Comparisons — Python prefers subclass reflected op, so
    #    ``plain_int < W`` calls W.__gt__(plain_int) → correct ──────
    def __lt__(self, o):  return self._val() <  self._o(o)
    def __le__(self, o):  return self._val() <= self._o(o)
    def __gt__(self, o):  return self._val() >  self._o(o)
    def __ge__(self, o):  return self._val() >= self._o(o)
    def __eq__(self, o):  return self._val() == self._o(o)
    def __ne__(self, o):  return self._val() != self._o(o)
    def __hash__(self):   return hash(self._val())

    # ── Formatting / math protocol ───────────────────────────────────
    def __repr__(self):           return repr(self._val())
    def __str__(self):            return str(self._val())
    def __format__(self, spec):   return format(self._val(), spec)
    def __round__(self, n=None):  return round(self._val(), n)
    def __trunc__(self):          return int(self._val())
    def __floor__(self):          return math.floor(self._val())
    def __ceil__(self):           return math.ceil(self._val())

    # ── NumPy interop — makes np.result_type(), np.mgrid[:H,:W], etc. work ──
    # numpy's dtype-detection code (nd_grid, result_type, …) calls np.asarray()
    # on unknown objects.  Returning a 0-d int64 array makes numpy treat us as
    # an integer scalar without affecting shape or broadcasting semantics.
    # NumPy 2.0 changed __array__ to be called as __array__(dtype, copy=False);
    # older numpy omits `copy`.  We ignore `copy` (a plain Python int always
    # yields a fresh scalar array, so there is nothing to share) and route
    # through np.asarray, which accepts `copy` in 2.x and is copy-tolerant —
    # unlike np.array(..., copy=False), which hard-raises when it cannot avoid
    # a copy.  This keeps the method working on both numpy generations without
    # a DeprecationWarning.
    def __array__(self, dtype=None, copy=None):
        v = self._val()
        if dtype is not None:
            return np.asarray(v, dtype=dtype)
        return np.asarray(v, dtype=np.intp)


def set_canvas(w: int, h: int) -> "_cv.Token":
    """Activate canvas dimensions for the current thread. Returns a reset token."""
    return _CANVAS.set((int(w), int(h)))


def reset_canvas(token: "_cv.Token") -> None:
    """Restore the canvas context to what it was before set_canvas()."""
    _CANVAS.reset(token)


def get_canvas() -> tuple[int, int]:
    """Return (width, height) of the currently active canvas context."""
    return _CANVAS.get()


# Module-level canvas dimension proxies.  All node files that do
# ``from ...core.utils import W, H`` get these objects once at import
# time; every subsequent use inside a node function resolves dynamically.
W = _DynDim(0)
H = _DynDim(1)


# ── PIL patch — apply operator.index() before the C extension sees sizes ──
# PIL's C layer calls PyLong_AsLong which reads the stored int of a _DynDim
# subclass rather than calling __index__.  Wrapping new() and resize() on
# the Python side fixes this with zero impact on non-_DynDim callers.
def _install_pil_canvas_patch() -> None:
    try:
        import PIL.Image as _PI
        _orig_new = _PI.new

        def _new_patched(mode, size, color=0):
            if hasattr(size, "__len__") and len(size) == 2:
                size = (_operator.index(size[0]), _operator.index(size[1]))
            return _orig_new(mode, size, color)

        _PI.new = _new_patched

        _orig_resize = _PI.Image.resize

        def _resize_patched(self, size, *args, **kwargs):
            if hasattr(size, "__len__") and len(size) == 2:
                size = (_operator.index(size[0]), _operator.index(size[1]))
            return _orig_resize(self, size, *args, **kwargs)

        _PI.Image.resize = _resize_patched
    except Exception:
        pass  # PIL unavailable or API changed; degrade gracefully


_install_pil_canvas_patch()


# ── cv2 patch — convert (W, H) size tuples to actual ints ─────────────
# cv2's C bindings use PyLong_Check for dsize elements, which rejects our
# non-int _DynDim proxy.  Wrapping the specific functions that accept a
# canvas-sized dsize tuple resolves the dimensions on the Python side first.
def _install_cv2_canvas_patch() -> None:
    try:
        import cv2 as _cv

        def _int_tuple(t):
            return tuple(int(x) for x in t)

        _orig_resize = _cv.resize
        def _resize(src, dsize, *a, **kw):
            return _orig_resize(src, _int_tuple(dsize), *a, **kw)
        _cv.resize = _resize

        _orig_warp = _cv.warpAffine
        def _warpAffine(src, M, dsize, *a, **kw):
            return _orig_warp(src, M, _int_tuple(dsize), *a, **kw)
        _cv.warpAffine = _warpAffine

        _orig_persp = _cv.warpPerspective
        def _warpPerspective(src, M, dsize, *a, **kw):
            return _orig_persp(src, M, _int_tuple(dsize), *a, **kw)
        _cv.warpPerspective = _warpPerspective

    except Exception:
        pass  # cv2 unavailable; degrade gracefully


_install_cv2_canvas_patch()


# ── numpy mgrid/ogrid patch ────────────────────────────────────────────
# np.mgrid[:H, :W] builds a slice list and passes the bounds to
# np.result_type().  numpy's result_type C code doesn't honour __array__
# for unknown objects — it calls str(H) → '512' → np.dtype('512') which
# raises TypeError.  Pre-converting _DynDim bounds to plain int fixes all
# ~70 call sites without touching the node files.
def _install_numpy_canvas_patch() -> None:
    try:
        def _fix_key(key):
            def _fix(s):
                if isinstance(s, slice):
                    return slice(
                        int(s.start) if isinstance(s.start, _DynDim) else s.start,
                        int(s.stop)  if isinstance(s.stop,  _DynDim) else s.stop,
                        int(s.step)  if isinstance(s.step,  _DynDim) else s.step,
                    )
                return int(s) if isinstance(s, _DynDim) else s

            if isinstance(key, tuple):
                return tuple(_fix(s) for s in key)
            return _fix(key)

        for grid_obj in (np.mgrid, np.ogrid):
            cls = type(grid_obj)
            _orig = cls.__getitem__

            def _patched(self, key, _orig=_orig):
                return _orig(self, _fix_key(key))

            cls.__getitem__ = _patched
    except Exception:
        pass


_install_numpy_canvas_patch()


# ── Sidecar capture context ──────────────────────────────────────────
# Like capture_frame() in animation.py, the write_* helpers below route their
# payloads through a per-thread sink when one is installed. This lets the live
# loop (in_memory=True) collect sidecars in memory instead of hitting the disk
# every frame — otherwise a FIELD/MASK/PARTICLES-emitting sim calls np.save()
# on every live frame, which both wastes I/O and (on a restricted output/ dir)
# can raise PermissionError and hard-fail the frame.
import threading as _threading

_sidecar_local = _threading.local()


def set_sidecar_context(sink: dict | None) -> None:
    """Install a per-thread sidecar sink used by the write_* helpers in the
    server/live path.

    `sink` maps key -> ndarray (field/particles/mask) or key -> float (named
    scalars). Pass None to restore unconditional disk writes (CLI path).
    """
    _sidecar_local.sink = sink


def _sidecar_sink() -> dict | None:
    return getattr(_sidecar_local, "sink", None)


# ── Save capture context ─────────────────────────────────────────────
# The executor used to intercept node images by monkeypatching the module
# attribute `utils.save`. That never worked for the ~150 method files that do
# `from ...core.utils import save` at import time — they hold a direct
# reference to the original function, so every live frame paid a full PNG
# encode + disk write + PNG decode read-back (~100ms+ per node per frame).
# This per-thread sink is checked inside save() itself, so it intercepts
# every call regardless of import style, and it is thread-safe: concurrent
# jobs (live loop + render job) each install their own sink.
_save_capture_local = _threading.local()


def set_save_capture(sink: dict | None, *, skip_disk: bool = False) -> None:
    """Install a per-thread sink that captures save() images in memory.

    `sink["image"]` receives the saved image as float32 [0,1] (H,W,3) —
    exactly what the PNG round-trip would have produced. With skip_disk=True
    (live mode) the disk write is skipped entirely; with skip_disk=False
    (render mode) the PNG is still written for the on-disk audit trail.
    Pass None to uninstall.
    """
    _save_capture_local.sink = sink
    _save_capture_local.skip_disk = bool(skip_disk) and sink is not None


def _capture_as_float01(arr: "np.ndarray | Image.Image") -> np.ndarray:
    """Normalize a save() payload to float32 [0,1], mirroring the PNG round-trip."""
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr.convert("RGB") if hasattr(arr, "convert") else arr)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype.kind == "f":
        a = arr.astype(np.float32, copy=True)  # single copy; clip in place
        if a.max() > 1.0:
            np.clip(a, 0, 255, out=a)
            a /= 255.0
        else:
            np.clip(a, 0.0, 1.0, out=a)
        return a
    return arr.astype(np.float32)


# ── Current-method context ───────────────────────────────────────────
# Nodes call capture_frame("NN", arr) and save(arr, mn(NN, "Name"), out_dir)
# with a HARDCODED id literal. When a node is renumbered, those literals go
# stale (e.g. Gray-Scott was renumbered 134 -> 155 but its body still wrote
# "134-*.png" and capture_frame("134")). To make renumbers safe, the executor
# installs the real method id as a per-thread context; capture_frame() and mn()
# then use it instead of the (possibly stale) literal. This removes the entire
# class of id-drift bugs.
_method_id_local = _threading.local()


def set_method_id(method_id: str | None) -> None:
    """Install the currently-executing method id (executor sets this before
    each meta.fn() call). Pass None to clear (CLI path reverts to literals)."""
    _method_id_local.id = method_id


def get_method_id() -> str | None:
    return getattr(_method_id_local, "id", None)


def write_scalars(node_dir: Path, **kwargs: float) -> None:
    """Write named scalar outputs to the node graph sidecar (scalars.json).

    Called by methods to expose per-frame scalars (e.g. sync order r, wave amplitude).
    Values are merged into flat_outputs[node_id] by GraphExecutor and become
    wirable SCALAR output ports when the method declares them in outputs=.

    When a sidecar sink is active (live/in_memory mode), the scalars are
    collected in memory and no disk file is written.
    """
    import json
    sink = _sidecar_sink()
    if sink is not None:
        sink.update({k: float(v) for k, v in kwargs.items()})
        return
    (node_dir / "scalars.json").write_text(json.dumps({k: float(v) for k, v in kwargs.items()}))


def write_field(node_dir: Path, arr: np.ndarray) -> None:
    """Write a 2D float32 field array to the node graph sidecar (field.npy).

    GraphExecutor reads this into flat_outputs[node_id]["field"], making it
    available to downstream FIELD wires without the image-fallback.

    When a sidecar sink is active (live/in_memory mode), the array is collected
    in memory instead of being written to disk.
    """
    sink = _sidecar_sink()
    if sink is not None:
        sink["field"] = arr.astype(np.float32)
        return
    np.save(str(node_dir / "field.npy"), arr.astype(np.float32))


def write_particles(node_dir: Path, arr: np.ndarray) -> None:
    """Write an (N, 4) float32 particles array [x, y, vx, vy] to the sidecar (particles.npy).

    GraphExecutor reads this into flat_outputs[node_id]["particles"].
    Shape convention: rows are particles, columns are [x, y, vx, vy].

    When a sidecar sink is active (live/in_memory mode), the array is collected
    in memory instead of being written to disk.
    """
    sink = _sidecar_sink()
    if sink is not None:
        sink["particles"] = arr.astype(np.float32)
        return
    np.save(str(node_dir / "particles.npy"), arr.astype(np.float32))


def write_mask(node_dir: Path, arr: np.ndarray) -> None:
    """Write a H×W float32 mask [0,1] for the node graph sidecar protocol.

    GraphExecutor reads this into flat_outputs[node_id]["mask"], making it
    available to downstream MASK wires.

    When a sidecar sink is active (live/in_memory mode), the array is collected
    in memory instead of being written to disk.
    """
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    sink = _sidecar_sink()
    if sink is not None:
        sink["mask"] = arr
        return
    np.save(str(node_dir / "mask.npy"), arr)


def save(arr: np.ndarray | Image.Image, name: str, out_dir: Path):
    """Save array (float32 [0,1] or uint8) or PIL Image to out_dir/name.

    When a save-capture sink is installed (executor in-memory mode) the image
    is also captured as float32 [0,1]; with skip_disk the PNG write is
    skipped entirely — the hot-path transport is the in-memory payload bus.
    """
    _sink = getattr(_save_capture_local, "sink", None)
    if _sink is not None:
        _sink["image"] = _capture_as_float01(arr)
        if getattr(_save_capture_local, "skip_disk", False):
            return
    if isinstance(arr, np.ndarray):
        if arr.max() <= 1 and arr.dtype.kind == "f":
            arr = (arr.clip(0, 1) * 255).astype(np.uint8)
        elif arr.dtype.kind == "f":
            arr = arr.clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        img = arr
    path = out_dir / name
    img.save(str(path))
    print(f"  ✓ {name}  ({path.stat().st_size // 1024} KB)")


def norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0,1]."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def mn(i: int | str, label: str) -> str:
    """Generate filename from method number and label.

    When a method-id context is active (set by the executor via
    set_method_id), the context id is used instead of the passed `i` so that
    a renumbered node never writes a stale filename. The passed `i` is ignored
    in that case; nodes may keep their historical literal for readability.
    """
    effective = get_method_id()
    if effective is not None:
        i = effective
    try:
        i = int(i)
    except (TypeError, ValueError):
        pass
    slug = (
        label.lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
    )
    if isinstance(i, int):
        return f"{i:02d}-{slug}.png"
    return f"{i}-{slug}.png"


FONT_SMALL = "/System/Library/Fonts/Menlo.ttc"
FONT_LARGE = "/System/Library/Fonts/Helvetica.ttc"

_FONT_SEARCH_PATHS = [
    # macOS
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    # Linux (Debian/Ubuntu/Arch)
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    # Windows
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/cour.ttf",
]


def get_font(size: int = 10, font_path: str = FONT_SMALL):
    for path in [font_path, *_FONT_SEARCH_PATHS]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def seed_all(s: int):
    """Seed random, numpy, and built-in random."""
    random.seed(s)
    np.random.seed(s)


BG_DEFAULT = (128, 128, 128)

# ── Palettes for pixel art & posterize ────────────────────────────────

PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "none": [],
    "bw": [(10, 10, 18), (220, 220, 200)],
    "grayscale": [(15, 15, 15), (45, 45, 45), (75, 75, 75), (105, 105, 105),
                  (135, 135, 135), (165, 165, 165), (195, 195, 195), (225, 225, 225)],
    "amber": [(10, 5, 0), (30, 20, 0), (60, 40, 0), (90, 65, 5),
              (120, 90, 10), (160, 125, 15), (200, 160, 20), (255, 200, 30)],
    "green": [(5, 15, 5), (5, 40, 10), (5, 70, 15), (10, 100, 25),
              (15, 140, 35), (20, 180, 50), (30, 220, 70), (60, 255, 100)],
    "gameboy": [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)],
    "cga": [(0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170),
            (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
            (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
            (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255)],
    "pico8": [(0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81),
              (171, 82, 54), (95, 87, 79), (194, 195, 199), (255, 241, 232),
              (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
              (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170)],
    "nes": [(0, 0, 0), (254, 254, 254), (124, 124, 124), (0, 0, 252),
            (0, 0, 188), (68, 40, 188), (148, 0, 132), (168, 0, 32),
            (168, 16, 0), (136, 20, 0), (80, 48, 0), (0, 120, 0),
            (0, 104, 0), (0, 88, 0), (0, 64, 88), (0, 0, 0),
            (188, 188, 0), (0, 120, 248), (0, 88, 248), (104, 68, 252),
            (216, 0, 204), (228, 0, 88), (248, 56, 0), (228, 92, 16),
            (172, 124, 0), (0, 184, 0), (0, 168, 0), (0, 168, 68),
            (0, 136, 136), (248, 248, 248), (60, 188, 252), (104, 136, 252),
            (152, 120, 248), (248, 120, 248), (248, 88, 152), (248, 120, 88),
            (252, 160, 68), (248, 184, 0), (184, 248, 24), (88, 216, 84),
            (88, 248, 152), (0, 232, 216), (120, 120, 120), (252, 252, 252),
            (164, 228, 252), (184, 184, 248), (216, 184, 248), (248, 184, 248),
            (248, 164, 192), (240, 208, 176), (252, 224, 168), (248, 216, 120),
            (216, 248, 120), (184, 248, 184), (184, 248, 216), (0, 252, 252)],
    "apple2": [(0, 0, 0), (140, 40, 60), (80, 80, 255), (140, 140, 200),
               (200, 60, 40), (220, 220, 255), (60, 200, 80), (255, 255, 255)],
    "zxspectrum": [(0, 0, 0), (0, 0, 215), (215, 0, 0), (215, 0, 215),
                   (0, 215, 0), (0, 215, 215), (215, 215, 0), (215, 215, 215)],
    "c64": [(0, 0, 0), (255, 255, 255), (136, 57, 50), (100, 180, 175),
            (73, 65, 55), (144, 170, 155), (84, 100, 170), (190, 190, 150),
            (115, 85, 65), (100, 120, 55), (160, 130, 70), (115, 165, 140),
            (75, 75, 80), (90, 145, 130), (185, 140, 100), (170, 190, 200)],
    "megadrive": [(0, 0, 0), (32, 32, 32), (64, 64, 64), (96, 96, 96),
                  (128, 128, 128), (160, 160, 160), (192, 192, 192), (224, 224, 224),
                  (0, 0, 128), (0, 0, 255), (64, 64, 255), (128, 128, 255),
                  (0, 128, 0), (0, 255, 0), (64, 255, 64), (128, 255, 128),
                  (128, 0, 0), (255, 0, 0), (255, 64, 64), (255, 128, 128),
                  (128, 128, 0), (255, 255, 0), (255, 255, 64), (192, 192, 255),
                  (128, 0, 128), (255, 0, 255), (64, 255, 255), (0, 255, 255),
                  (0, 128, 128), (128, 64, 0), (255, 128, 0), (192, 128, 64)],
    "sms": [(0, 0, 0), (85, 255, 0), (0, 220, 0), (0, 170, 0),
            (255, 255, 85), (220, 220, 0), (170, 170, 0), (255, 85, 85),
            (220, 0, 0), (170, 0, 0), (85, 85, 255), (0, 0, 220),
            (0, 0, 170), (255, 255, 255), (200, 200, 200), (140, 140, 140)],
    "atari2600": [(0, 0, 0), (132, 0, 0), (0, 132, 0), (132, 132, 0),
                  (38, 38, 132), (132, 38, 132), (0, 132, 132), (132, 132, 132),
                  (64, 64, 64), (255, 64, 64), (64, 255, 64), (255, 255, 64),
                  (96, 96, 255), (255, 64, 255), (64, 255, 255), (255, 255, 255)],
    "amiga": [(0, 0, 0), (17, 17, 17), (34, 34, 34), (51, 51, 51),
              (68, 68, 68), (85, 85, 85), (102, 102, 102), (119, 119, 119),
              (136, 136, 136), (153, 153, 153), (170, 170, 170), (187, 187, 187),
              (204, 204, 204), (221, 221, 221), (238, 238, 238), (255, 255, 255),
              (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)],
    "warm": [(20, 10, 8), (50, 30, 20), (80, 50, 30), (110, 70, 40),
             (140, 90, 50), (170, 110, 60), (200, 140, 80), (230, 180, 120)],
    "cool": [(10, 10, 25), (15, 30, 55), (20, 50, 85), (30, 70, 115),
             (50, 100, 150), (80, 140, 190), (130, 180, 220), (190, 220, 245)],
    "vapor": [(20, 10, 30), (80, 20, 60), (140, 30, 100), (200, 40, 140),
              (240, 60, 120), (255, 100, 80), (255, 180, 60), (220, 240, 255)],
    "sepia": [(40, 25, 15), (70, 45, 25), (100, 65, 35), (130, 85, 45),
              (160, 105, 55), (190, 130, 70), (210, 160, 100), (240, 200, 150)],
}


def quantize_to_palette(arr: np.ndarray, palette_name: str) -> np.ndarray:
    """Quantize float32 [0,1] (H,W,3) array to named palette colors.
    Uses nearest-neighbor in RGB space. Returns same shape float32.
    If palette_name is "none" or empty, returns arr unchanged.
    """
    if not palette_name or palette_name == "none":
        return arr
    pal = PALETTES.get(palette_name)
    if not pal:
        return arr
    pal_arr = np.array(pal, dtype=np.float32) / 255.0  # (N, 3)
    h, w = arr.shape[:2]
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]  # drop alpha
    flat = arr.reshape(-1, 3)
    # Process in chunks to cap peak memory. Full-image broadcast over a large
    # palette (e.g. NES 54-color) allocates ~(H*W * N * 3 * 4) bytes at once —
    # ~250 MB for 768×512. Chunks of 8 192 pixels keep it under ~15 MB.
    CHUNK = 8192
    nearest = np.empty(len(flat), dtype=np.intp)
    for i in range(0, len(flat), CHUNK):
        chunk = flat[i : i + CHUNK]
        diffs = chunk[:, None, :] - pal_arr[None, :, :]
        nearest[i : i + CHUNK] = np.argmin(np.sum(diffs ** 2, axis=2), axis=1)
    return pal_arr[nearest].reshape(h, w, 3)


def apply_palette(arr: np.ndarray, palette_name: str) -> np.ndarray:
    """Map image luminance through named palette via linear interpolation.
    Returns arr unchanged when palette_name is 'none', empty, or not found.
    """
    if not palette_name or palette_name == "none":
        return arr
    pal = PALETTES.get(palette_name)
    if not pal or len(pal) < 2:
        return arr
    pal_f = np.array(pal, dtype=np.float32) / 255.0
    lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).clip(0, 1)
    N = len(pal_f)
    idx_f = lum * (N - 1)
    idx0 = np.floor(idx_f).astype(np.int32).clip(0, N - 2)
    idx1 = idx0 + 1
    frac = (idx_f - idx0)[:, :, None]
    return (pal_f[idx0] + frac * (pal_f[idx1] - pal_f[idx0])).clip(0, 1)


# Bayer 4x4 ordered dither matrix
BAYER_4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
]) / 16.0


def ordered_dither(arr: np.ndarray, levels: int = 2, bayer: np.ndarray = BAYER_4) -> np.ndarray:
    """Apply Bayer ordered dither to float32 [0,1] array (H,W) or (H,W,3).
    levels = number of quantization levels per channel.
    """
    h, w = arr.shape[:2]
    tile_h, tile_w = bayer.shape
    bayer_tiled = np.tile(bayer, (h // tile_h + 1, w // tile_w + 1))[:h, :w]
    if arr.ndim == 3:
        bayer_tiled = bayer_tiled[:, :, None]
    quantized = np.floor(arr * (levels - 1) + bayer_tiled) / (levels - 1)
    return quantized.clip(0, 1)


def floyd_steinberg_dither(arr: np.ndarray, levels: int = 2) -> np.ndarray:
    """Apply Floyd-Steinberg error diffusion dithering.
    arr: float32 [0,1] (H,W) grayscale or (H,W,3) color.
    levels: number of quantization levels per channel.
    Returns quantized float32 same shape.
    """
    h, w = arr.shape[:2]
    out = arr.copy()
    step = 1.0 / (levels - 1)

    for y in range(h):
        for x in range(w):
            old = out[y, x].copy() if out.ndim == 3 else out[y, x]
            new = np.round(old / step) * step
            new = new.clip(0, 1)
            out[y, x] = new
            err = old - new

            if x + 1 < w:
                out[y, x + 1] = out[y, x + 1] + err * (7 / 16)
            if y + 1 < h:
                if x > 0:
                    out[y + 1, x - 1] = out[y + 1, x - 1] + err * (3 / 16)
                out[y + 1, x] = out[y + 1, x] + err * (5 / 16)
                if x + 1 < w:
                    out[y + 1, x + 1] = out[y + 1, x + 1] + err * (1 / 16)

    return out.clip(0, 1)


def load_input(
    path: str | Path,
    target_w: int | None = None,
    target_h: int | None = None,
) -> np.ndarray:
    """Load an external image, resize to canvas size, return float32 [0,1] (H,W,3).

    target_w / target_h default to the active canvas context so that callers
    with hardcoded ``load_input(path, W, H)`` still receive explicit values
    (W and H are _DynDim objects whose __index__ resolves correctly), and
    callers with no explicit size get the job's canvas automatically.
    """
    from PIL import Image as _PILI
    cw, ch = get_canvas()
    tw = target_w if target_w is not None else cw
    th = target_h if target_h is not None else ch
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input image not found: {p}")
    img = _PILI.open(str(p)).convert("RGB")
    img = img.resize((int(tw), int(th)), _PILI.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def wired_source_rgb(params: dict, w: int, h: int) -> np.ndarray | None:
    """Return the wired upstream image as float32 [0,1] (H,W,3), or None.

    Generators that can take an image "as a source" call this once near their
    field/seed initialization. Returns None when nothing is wired (so the
    method falls back to its procedural generation). Mirrors the Rule-#12
    contract used by filter nodes (a wired image overrides internal gen).

    Handles both executor contracts: live in-memory mode injects ``_input_image``
    (an ndarray); render/audit mode injects ``input_image`` (a disk path).
    """
    # Live in-memory: ndarray already in memory
    arr = params.get("_input_image", None)
    if isinstance(arr, np.ndarray) and arr.size > 0:
        from PIL import Image as _PILI
        # resize to canvas if needed
        if arr.shape[0] != int(h) or arr.shape[1] != int(w):
            img = _PILI.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
            img = img.resize((int(w), int(h)), _PILI.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[..., None].repeat(3, axis=-1)
        return arr.astype(np.float32)[..., :3]
    # Render/audit mode: disk path
    p = params.get("input_image", "")
    if not p:
        return None
    try:
        return load_input(p, w, h)
    except (FileNotFoundError, OSError, ValueError):
        return None


def wired_source_lum(params: dict, w: int, h: int) -> np.ndarray | None:
    """Return the wired upstream image's luminance as float32 [0,1] (H,W), or None.

    Convenience for simulations / fields / fractals that seed from a scalar
    field: the image's brightness becomes the seed/initial-condition field.
    """
    rgb = wired_source_rgb(params, w, h)
    if rgb is None:
        return None
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)