"""Regression tests for the webcam node's camera-backend selection.

These lock in the fix for "webcam never shows an image / camera LED lights up
~30s after toggling live on Windows". The root cause was that webcam.py (and
the /api/webcam/configure TCC probe) only ever tried cv2 backends CAP_ANY and
CAP_AVFOUNDATION.  CAP_AVFOUNDATION is a macOS-only backend; on Windows/Linux
calling cv2.VideoCapture(idx, CAP_AVFOUNDATION) can hang for tens of seconds
(LED on, then off) before failing, so the camera never delivered a frame.

The node must now pick an OS-valid backend via _camera_backends().
"""
import platform

import pytest

import image_pipeline.methods.io_nodes.webcam as webcam_mod
from image_pipeline.methods.io_nodes.webcam import _camera_backends, release_cameras


def test_camera_backends_never_uses_avfoundation_off_macos():
    """On Windows/Linux the backend list must NOT contain CAP_AVFOUNDATION."""
    sysname = platform.system().lower()
    if sysname == "darwin":
        pytest.skip("AVFoundation is valid on macOS")
    backends = _camera_backends()
    names = [name for name, _ in backends]
    assert "CAP_AVFOUNDATION" not in names, (
        f"{sysname} must not try CAP_AVFOUNDATION: {names}"
    )


def test_camera_backends_use_os_correct_backends():
    """Each OS selects its real capture backend(s)."""
    sysname = platform.system().lower()
    names = [name for name, _ in _camera_backends()]
    if sysname == "windows":
        # DSHOW must lead on Windows: CAP_ANY resolves to MSMF, which is the
        # backend that hangs for ~30s on many live USB webcams — the exact
        # regression this suite guards against (LED on ~30s, no image).
        assert "CAP_DSHOW" in names, f"Windows needs DSHOW: {names}"
    elif sysname == "linux":
        assert "CAP_V4L2" in names, f"Linux needs V4L2: {names}"
    elif sysname == "darwin":
        assert "CAP_AVFOUNDATION" in names, f"macOS needs AVFoundation: {names}"


def test_release_cameras_is_safe_noop_when_empty():
    """release_cameras must not raise when no camera is open (live-stop path)."""
    assert release_cameras() == 0


def test_read_frame_bounded_returns_frame():
    """A healthy cap returns its frame through the watchdog helper."""
    class _FakeCap:
        def read(self):
            import numpy as np
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

    ok, frame = webcam_mod._read_frame_bounded(_FakeCap(), timeout=2.0)
    assert ok is True
    assert frame is not None


def test_read_frame_bounded_times_out_on_hang():
    """A cap whose read() blocks must not freeze the caller — returns fail."""
    import threading as _thr

    class _HangingCap:
        def read(self):
            _thr.Event().wait(10.0)  # block far past the timeout
            return True, None

    ok, frame = webcam_mod._read_frame_bounded(_HangingCap(), timeout=0.2)
    assert ok is False
    assert frame is None


def test_webcam_method_registered():
    from image_pipeline.core.registry import get_meta
    meta = get_meta("__webcam__")
    assert meta is not None
    assert meta.is_time_varying is True
