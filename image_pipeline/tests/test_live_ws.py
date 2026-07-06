"""Integration test for Phase 4 WebSocket live stream.

Tests that /api/live/ws delivers per-frame JSON messages containing image data
and per-frame metadata when the live graph loop is active.

Requires a running server at localhost:7860 (skipped otherwise).
"""
import asyncio
import base64
import json
import socket
import time

import pytest
import requests


SERVER = "http://localhost:7860"
WS_URL = "ws://localhost:7860/api/live/ws"


def _server_available() -> bool:
    try:
        s = socket.create_connection(("localhost", 7860), timeout=1)
        s.close()
        return True
    except OSError:
        return False


def _start_live(nodes, edges, width=64, height=64, seed=1):
    requests.post(f"{SERVER}/api/graph/live", json={
        "nodes": nodes, "edges": edges,
        "seed": seed, "frames": 1,
        "width": width, "height": height,
    }, timeout=5)


def _stop_live():
    requests.post(f"{SERVER}/api/graph/live", json={
        "nodes": [], "edges": [], "seed": 0, "frames": 0,
        "width": 64, "height": 64,
    }, timeout=5)


pytestmark = pytest.mark.skipif(
    not _server_available(),
    reason="live server not running on localhost:7860",
)


class TestLiveWebSocket:
    """WebSocket integration tests — require the server to be running."""

    def test_ws_delivers_json_frame(self):
        """Connecting to /api/live/ws and starting a graph delivers a JSON frame."""
        import websocket  # websocket-client

        _start_live(
            nodes=[{"id": "n", "method_id": "05",
                    "params": {"noise_type": "perlin"}, "dirty": True}],
            edges=[],
        )
        try:
            messages = []
            ws = websocket.create_connection(WS_URL, timeout=8)
            # Collect up to 3 frames or until deadline
            deadline = time.time() + 6.0
            while time.time() < deadline and len(messages) < 3:
                try:
                    ws.settimeout(2.0)
                    raw = ws.recv()
                    messages.append(raw)
                except websocket.WebSocketTimeoutException:
                    continue  # no frame yet; retry
                except Exception:
                    break  # real disconnect
            ws.close()

            assert messages, "No messages received from WebSocket"
            msg = json.loads(messages[0])

            # Required top-level keys
            assert "frame"        in msg, f"Missing 'frame': {list(msg)}"
            assert "cook_ms"      in msg, f"Missing 'cook_ms'"
            assert "fps"          in msg, f"Missing 'fps'"
            assert "node_timings" in msg, f"Missing 'node_timings'"
            assert "node_names"   in msg, f"Missing 'node_names'"
            assert "node_errors"  in msg, f"Missing 'node_errors'"
            assert "canvas_w"     in msg, f"Missing 'canvas_w'"
            assert "canvas_h"     in msg, f"Missing 'canvas_h'"
            assert "img"          in msg, f"Missing 'img'"

            # 'img' must be valid base64-encoded JPEG
            jpeg = base64.b64decode(msg["img"])
            assert jpeg[:2] == b'\xff\xd8', \
                f"img is not a JPEG (first bytes: {jpeg[:4].hex()})"
            assert jpeg[-2:] == b'\xff\xd9', \
                "JPEG missing EOI marker"

            # Numeric fields are sane
            assert isinstance(msg["frame"],   int)   and msg["frame"]   >= 0
            assert isinstance(msg["cook_ms"], float) and msg["cook_ms"] >= 0
            assert isinstance(msg["fps"],     float) and msg["fps"]     >= 0
            assert isinstance(msg["canvas_w"], int)  and msg["canvas_w"] == 64
            assert isinstance(msg["canvas_h"], int)  and msg["canvas_h"] == 64

        finally:
            _stop_live()

    def test_ws_node_timings_populated(self):
        """node_timings contains an entry for each executed node."""
        import websocket

        _start_live(
            nodes=[
                {"id": "src",   "method_id": "05",
                 "params": {"noise_type": "perlin"}, "dirty": True},
                {"id": "glitch","method_id": "17",
                 "params": {"intensity": 0.3},       "dirty": True},
            ],
            edges=[{"src_node": "src", "src_port": "image",
                    "dst_node": "glitch", "dst_port": "image_in"}],
        )
        try:
            ws = websocket.create_connection(WS_URL, timeout=8)
            msg = None
            for _ in range(15):
                try:
                    ws.settimeout(2.0)
                    raw = ws.recv()
                    candidate = json.loads(raw)
                    if candidate.get("node_timings"):
                        msg = candidate
                        break
                except websocket.WebSocketTimeoutException:
                    continue  # no frame yet; retry
                except Exception:
                    break  # real disconnect
            ws.close()

            assert msg is not None, "No frame with node_timings received"
            timings = msg["node_timings"]
            assert "src"   in timings, f"'src' missing from timings: {timings}"
            assert "glitch" in timings, f"'glitch' missing from timings: {timings}"
            assert timings["src"]   > 0, "src timing is zero"
            assert timings["glitch"] > 0, "glitch timing is zero"

            # Names map should parallel timings
            names = msg["node_names"]
            assert "src"   in names
            assert "glitch" in names

        finally:
            _stop_live()

    def test_ws_gpu_nodes_flagged(self):
        """gpu_nodes counter is 2 for a GPU-only graph."""
        import websocket

        _start_live(
            nodes=[
                {"id": "p", "method_id": "175",
                 "params": {"p1": 0.5}, "dirty": True},
                {"id": "b", "method_id": "198",
                 "params": {"strength": 0.5}, "dirty": True},
            ],
            edges=[{"src_node": "p", "src_port": "image",
                    "dst_node": "b", "dst_port": "image_in"}],
            width=64, height=64,
        )
        try:
            ws = websocket.create_connection(WS_URL, timeout=8)
            msg = None
            for _ in range(15):
                try:
                    ws.settimeout(2.0)
                    raw = ws.recv()
                    candidate = json.loads(raw)
                    if candidate.get("gpu_nodes", 0) > 0:
                        msg = candidate
                        break
                except websocket.WebSocketTimeoutException:
                    continue  # no frame yet; retry
                except Exception:
                    break  # real disconnect
            ws.close()

            assert msg is not None, "No frame with gpu_nodes > 0 received"
            assert msg["gpu_nodes"] == 2, f"Expected gpu_nodes=2, got {msg['gpu_nodes']}"
            assert msg["cpu_nodes"] == 0, f"Expected cpu_nodes=0, got {msg['cpu_nodes']}"
            assert msg["mem_edges"] == 1, f"Expected mem_edges=1, got {msg['mem_edges']}"

        finally:
            _stop_live()

    def test_mjpeg_fallback_still_works(self):
        """MJPEG stream at /api/live/stream returns multipart data while live."""
        _start_live(
            nodes=[{"id": "n", "method_id": "05",
                    "params": {"noise_type": "perlin"}, "dirty": True}],
            edges=[],
        )
        try:
            # Stream a few bytes to confirm the multipart response starts
            resp = requests.get(f"{SERVER}/api/live/stream", stream=True, timeout=5)
            assert resp.status_code == 200
            content_type = resp.headers.get("content-type", "")
            assert "multipart" in content_type, \
                f"Expected multipart content-type, got: {content_type}"

            # Read just enough to see the first JPEG boundary
            chunk = b""
            for data in resp.iter_content(chunk_size=4096):
                chunk += data
                if len(chunk) > 512:
                    break
            resp.close()

            assert b"--frame" in chunk, \
                f"MJPEG boundary '--frame' not found in first {len(chunk)} bytes"
            assert b"image/jpeg" in chunk, \
                "MJPEG Content-Type header not found"

        finally:
            _stop_live()
