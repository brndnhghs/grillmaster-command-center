"""Tests for Phase 5 keyframe editor — Python evaluation + UI round-trip.

Python tests: unit-test _evaluate_param_track and apply_easing directly.
UI tests:     Playwright headless — verify serialize/load round-trip and
              that _renderNodeKfLanes produces the expected DOM.
"""
import json
import math
import socket

import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _server_available() -> bool:
    try:
        s = socket.create_connection(("localhost", 7860), timeout=1)
        s.close()
        return True
    except OSError:
        return False


SERVER = "http://localhost:7860"


# ── Python unit tests ─────────────────────────────────────────────────────────

class TestParamTrackEvaluation:
    """Unit tests for _evaluate_param_track (no server required)."""

    def _eval(self, kfs, frame):
        from image_pipeline.core.graph import _evaluate_param_track
        return _evaluate_param_track(kfs, frame)

    def test_empty_returns_none(self):
        assert self._eval([], 5) is None

    def test_before_first_holds(self):
        kfs = [{"frame": 10, "value": 0.5}]
        assert self._eval(kfs, 0)  == 0.5
        assert self._eval(kfs, 10) == 0.5

    def test_after_last_holds(self):
        kfs = [{"frame": 0, "value": 0.1}, {"frame": 10, "value": 0.9}]
        assert self._eval(kfs, 10) == pytest.approx(0.9)
        assert self._eval(kfs, 99) == pytest.approx(0.9)

    def test_linear_midpoint(self):
        kfs = [{"frame": 0, "value": 0.0, "easing": "linear"},
               {"frame": 10, "value": 1.0, "easing": "linear"}]
        assert self._eval(kfs, 5) == pytest.approx(0.5, abs=1e-6)

    def test_linear_quarter(self):
        kfs = [{"frame": 0, "value": 0.0, "easing": "linear"},
               {"frame": 20, "value": 1.0, "easing": "linear"}]
        assert self._eval(kfs, 5)  == pytest.approx(0.25, abs=1e-6)
        assert self._eval(kfs, 15) == pytest.approx(0.75, abs=1e-6)

    def test_step_holds_until_end(self):
        # Python step easing: 0.0 if t < 1.0 else 1.0 — holds src value until
        # the destination frame is exactly reached, then jumps
        kfs = [{"frame": 0,  "value": 0.0, "easing": "linear"},
               {"frame": 10, "value": 1.0, "easing": "step"}]
        assert self._eval(kfs, 3)  == pytest.approx(0.0, abs=1e-6)
        assert self._eval(kfs, 7)  == pytest.approx(0.0, abs=1e-6)  # still holding
        assert self._eval(kfs, 10) == pytest.approx(1.0, abs=1e-6)  # at/past dst → jump

    def test_three_keyframe_segment_selection(self):
        kfs = [{"frame": 0,  "value": 0.0, "easing": "linear"},
               {"frame": 10, "value": 0.5, "easing": "linear"},
               {"frame": 20, "value": 1.0, "easing": "linear"}]
        assert self._eval(kfs, 5)  == pytest.approx(0.25, abs=1e-6)
        assert self._eval(kfs, 10) == pytest.approx(0.5,  abs=1e-6)
        assert self._eval(kfs, 15) == pytest.approx(0.75, abs=1e-6)

    def test_ease_in_out_is_slower_at_ends(self):
        """ease-in-out should be closer to 0 at t=0.25 than linear."""
        from image_pipeline.core.easing import apply_easing
        t_linear = 0.25
        t_eio    = apply_easing(0.25, "ease-in-out")
        assert t_eio < t_linear, "ease-in-out should lag linear near t=0"

    def test_bounce_stays_in_range(self):
        from image_pipeline.core.easing import apply_easing
        for i in range(20):
            t = i / 19
            result = apply_easing(t, "bounce")
            assert -0.1 <= result <= 1.1, f"bounce({t}) = {result} out of range"

    def test_elastic_overshoots_then_returns(self):
        from image_pipeline.core.easing import apply_easing
        results = [apply_easing(i / 30, "elastic") for i in range(31)]
        assert results[0]  == pytest.approx(0.0, abs=1e-6)
        assert results[-1] == pytest.approx(1.0, abs=1e-3)
        # elastic should overshoot (go negative before 1)
        assert any(r < -0.01 for r in results), "elastic should undershoot"


# ── UI round-trip tests (require running server) ──────────────────────────────

_UI_SKIP = pytest.mark.skipif(
    not _server_available(),
    reason="live server not running on localhost:7860",
)


@_UI_SKIP
class TestKeyframeUIRoundtrip:
    """Playwright headless tests — serialization, restore, and lane render."""

    @pytest.fixture(scope="class")
    def page(self):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
            ctx = browser.new_context(viewport={"width": 1440, "height": 900})
            pg  = ctx.new_page()
            pg.goto(SERVER, wait_until="domcontentloaded")
            pg.wait_for_timeout(1200)
            yield pg
            ctx.close()
            browser.close()

    def _load_graph_with_kfs(self, page):
        graph = {
            "nodes": [{
                "id": "p", "method_id": "33",
                "params": {"p1": 0.5, "p2": 0.6, "p3": 0.3, "p4": 0.4},
                "paramKeyframes": {
                    "p1": [
                        {"frame": 1,  "value": 0.1, "easing": "linear",      "handle_in": None, "handle_out": None},
                        {"frame": 12, "value": 0.9, "easing": "ease-in-out", "handle_in": None, "handle_out": None},
                        {"frame": 24, "value": 0.3, "easing": "ease-out",    "handle_in": None, "handle_out": None},
                    ],
                    "p2": [
                        {"frame": 1,  "value": 0.2, "easing": "ease-in", "handle_in": None, "handle_out": None},
                        {"frame": 24, "value": 0.8, "easing": "bounce",  "handle_in": None, "handle_out": None},
                    ],
                },
                "dirty": True,
            }],
            "edges": [],
            "version": 1,
        }
        page.evaluate(f"""async () => {{
          const g = {json.dumps(graph)};
          await gLoadGraph(g);
          gSelectNode('p');
          renderTimelineRuler();
        }}""")
        page.wait_for_timeout(500)

    def test_serialize_preserves_param_keyframes(self, page):
        self._load_graph_with_kfs(page)
        serialized = page.evaluate("""() => {
          const g = gSerializeGraph();
          const node = g.nodes.find(n => n.id === 'p');
          return node ? node.paramKeyframes : null;
        }""")
        assert serialized is not None, "paramKeyframes missing from serialized graph"
        assert "p1" in serialized, "p1 track missing"
        assert "p2" in serialized, "p2 track missing"
        assert len(serialized["p1"]) == 3, f"Expected 3 p1 KFs, got {len(serialized['p1'])}"
        assert len(serialized["p2"]) == 2, f"Expected 2 p2 KFs, got {len(serialized['p2'])}"

    def test_serialize_preserves_easing(self, page):
        self._load_graph_with_kfs(page)
        serialized = page.evaluate("""() => {
          const g = gSerializeGraph();
          const node = g.nodes.find(n => n.id === 'p');
          return node ? node.paramKeyframes : null;
        }""")
        p1 = serialized["p1"]
        # KFs are stored in insertion order; index 1 = frame 12 (ease-in-out)
        assert p1[1]["easing"] == "ease-in-out", f"KF[1] easing: {p1[1]['easing']}"
        assert p1[2]["easing"] == "ease-out",    f"KF[2] easing: {p1[2]['easing']}"

    def test_load_restores_param_keyframes(self, page):
        self._load_graph_with_kfs(page)
        # Serialize, clear, reload
        result = page.evaluate("""async () => {
          const g = gSerializeGraph();
          await gLoadGraph({ nodes: [], edges: [], version: 1 });
          await gLoadGraph(g);
          const node = gNodes.find(n => n.id === 'p');
          if (!node) return { err: 'node not found' };
          const pkf = node.paramKeyframes || {};
          return {
            hasP1: 'p1' in pkf,
            hasP2: 'p2' in pkf,
            p1Count: (pkf.p1 || []).length,
            p2Count: (pkf.p2 || []).length,
            p1Frame12Easing: (pkf.p1 || [])[1]?.easing,
            p1Frame1Value:   (pkf.p1 || [])[0]?.value,
          };
        }""")
        assert result.get("hasP1"),                          "p1 track not restored"
        assert result.get("hasP2"),                          "p2 track not restored"
        assert result["p1Count"] == 3,                       f"p1 count: {result['p1Count']}"
        assert result["p2Count"] == 2,                       f"p2 count: {result['p2Count']}"
        assert result["p1Frame12Easing"] == "ease-in-out",   f"easing: {result['p1Frame12Easing']}"
        assert abs(result["p1Frame1Value"] - 0.1) < 1e-9,    f"value: {result['p1Frame1Value']}"

    def test_kf_lanes_render_for_selected_node(self, page):
        self._load_graph_with_kfs(page)
        state = page.evaluate("""() => ({
          laneCount:    document.querySelectorAll('#tl-lanes .tl-lane').length,
          diamondCount: document.querySelectorAll('#tl-lanes .tl-kf-diamond').length,
          svgPaths:     document.querySelectorAll('#tl-lanes svg path').length,
          hasHeader:    !!document.querySelector('#tl-lanes [class*="node-kf-header"]'),
        })""")
        assert state["laneCount"]    >= 2, f"Expected >=2 lanes, got {state['laneCount']}"
        assert state["diamondCount"] >= 5, f"Expected >=5 diamonds, got {state['diamondCount']}"
        assert state["svgPaths"]     >= 2, f"Expected >=2 SVG easing paths, got {state['svgPaths']}"
        assert state["hasHeader"],          "Node KF header not found in timeline"

    def test_js_easing_evaluation_correct(self, page):
        result = page.evaluate("""() => {
          const kfs = [
            { frame: 0,  value: 0.1, easing: 'linear' },
            { frame: 12, value: 0.9, easing: 'ease-in-out' },
            { frame: 24, value: 0.3, easing: 'ease-out' },
          ];
          return {
            before: _kfEvaluateTrack(kfs, -5),  // before first → hold
            at0:    _kfEvaluateTrack(kfs, 0),
            at6:    _kfEvaluateTrack(kfs, 6),    // linear midpoint
            at12:   _kfEvaluateTrack(kfs, 12),
            at18:   _kfEvaluateTrack(kfs, 18),   // ease-out segment
            at24:   _kfEvaluateTrack(kfs, 24),
            after:  _kfEvaluateTrack(kfs, 99),   // after last → hold
          };
        }""")
        assert result["before"] == pytest.approx(0.1, abs=1e-6), f"before: {result['before']}"
        assert result["at0"]    == pytest.approx(0.1, abs=1e-6), f"at0:    {result['at0']}"
        assert result["at6"]    == pytest.approx(0.5, abs=0.02), f"at6:    {result['at6']}"  # linear mid
        assert result["at12"]   == pytest.approx(0.9, abs=1e-6), f"at12:   {result['at12']}"
        assert result["at18"]   != pytest.approx(0.6, abs=0.05), f"at18 should differ from linear mid"
        assert result["at24"]   == pytest.approx(0.3, abs=1e-6), f"at24:   {result['at24']}"
        assert result["after"]  == pytest.approx(0.3, abs=1e-6), f"after:  {result['after']}"

    def test_no_js_errors_after_kf_setup(self, page):
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        self._load_graph_with_kfs(page)
        page.wait_for_timeout(300)
        assert errors == [], f"JS errors after KF setup: {errors}"
