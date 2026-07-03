# Audit Remediation Plan — 2026-07-02

Phased execution of the roadmap in `CODE_AUDIT_2026-07-02.md`. One commit per phase on `claude/node-editor-code-audit-p4suyp`. Status is updated in place as phases land.

**Constraint (owner decision):** Hermes agent is the sole LLM backend for all LLM calls. No other providers.

## Phase 1 — Registry integrity & executor correctness — STATUS: done

Landed beyond the original scope: Langton's Ant #83 was itself broken in all committed history (`_render_langton_frame` never existed) — renderer written and verified across all color/render modes; sidecar files are now read for every method return type (ndarray-returning methods silently dropped `write_particles`/`write_field`); `.nd-bak` deletion pulled forward from Phase 3 to green the gate; `tools/audit_methods.py` PNG-fallback heuristic updated for the return-dict contract (re-raise/return handlers and swallow-then-fallback are safe; only genuinely silent broad catches are violations); `tools/next_id.py` no longer crashes on named ids; `brusselator.py` renamed `moire_animation.py` (it contains Moiré Patterns #164); #166/#167 `capture_frame` id typos fixed.

- Registry raises on duplicate method-ID registration (audit #2).
- Restore Langton's Ant as #83; re-home p5.js Sketch to a fresh ID.
- Wire the unregistered method files into their packages: chua_lattice (#150), brusselator (#164), turing_morphogenesis (#169), pfc (#170); re-home sand_dune_migration off colliding #146; register the Timeline node (audit #3, #12).
- Fix `params_hash` NameError on the Arch-B list path (audit #4).
- Stable node seeds via hashlib, not randomized `hash()` (audit #5).
- Declare the confirmed missing sidecar scalars on #166/#167 so `tools/audit_methods.py --fail-on-violations` is green (audit #21).

## Phase 2 — Real-time path — STATUS: done

Verified over HTTP: MJPEG stream pushes ~23 fps for a cheap node graph; re-POSTing hot-swaps the running loop; stop works; `/api/graph/live/status` added. Single-frame `/api/graph/execute` now honors client dirty flags (clean nodes log "skipped (clean)") and invalidates on seed/frame/wiring changes; multi-frame renders still force-dirty.

- Wire the 📺 Live buttons to `/api/graph/live` and the MJPEG `/api/live/stream` (audit #13).
- Live loop: double-start guard, error logging instead of silent `pass` (audit #14).
- Hoist `get_all_node_defs()` out of the per-edge loop (audit #7).
- Honor client dirty flags for single-frame runs; force-dirty only for multi-frame animation renders (audit #6, conservative first step).

## Phase 3 — Hermes backend & security — STATUS: done

Verified: `HERMES_AGENT_DIR`/`HERMES_PYTHON` override the resolved interpreter in both `server.py` and `nd_runner.py`; startup logs found/not-found; Node Doctor chat fails fast with a clear message instead of a generic subprocess error. With `GRILLMASTER_API_TOKEN` set: mutating endpoints return 401 without/with-wrong token and 200 with it; unset = open (localhost default). Backups now write to `output/nd-backups/`; `.nd-bak` deletion had already landed in Phase 1; `.gitignore` narrowed (no more global `*.html`/`*.txt`).

- Hermes install path resolved from `HERMES_AGENT_DIR` env var (default `~/.hermes/hermes-agent`), shared by `server.py` and `nd_runner.py`; startup log states whether Hermes was found (audit #18). Hermes remains the sole LLM backend.
- Node Doctor backups written to `output/nd-backups/` instead of `methods/`; committed `.nd-bak-*.py` files deleted; pattern gitignored and excluded from the audit tool (audit #19).
- Optional `GRILLMASTER_API_TOKEN`: when set, mutating endpoints (`/admin/restart`, `/api/node-doctor/apply|undo`, `/api/graph/live`) require the token header; startup warning when tunneling without it (audit #22).

## Phase 4 — Philosophy docs & hygiene — STATUS: done

`DESIGN.md` rewritten around the three-pillar vision (Houdini payloads × TouchDesigner live × Hermes-driven evolution) with the port registry/colors, MASK/COLORMAP, per-pixel luminance, dirty-flag and live-mode semantics, stable seeds, duplicate-id policy, and system/channel nodes all matching the code. `AGENT_GUIDE.md`: COLORMAP row, return-dict image contract (§2 rewritten), duplicate-id-raises rule, `description=` requirement, audit-gate step in the checklist, Hermes named as sole LLM backend. `README.md` rewritten for the node editor (old text described the Streamlit app removed 2026-06-20). `chord_bot.egg-info` untracked; `.gitignore` cleaned up in Phase 3.

- `DESIGN.md`: vision statement (Houdini × TouchDesigner × Hermes-driven evolution, real-time goal); port table corrected against `core/port_types.py` (+ MASK, COLORMAP); dirty-flag section updated to describe actual behavior; outputs table corrected; per-pixel luminance documented.
- `AGENT_GUIDE.md`: COLORMAP row; duplicate-ID-now-raises rule; `description=` required in the checklist; Hermes-based Node Doctor noted.
- `README.md`: rewritten to describe the node editor (the Streamlit-era text predates the 2026-06-20 removal).
- `.gitignore`: add `*.nd-bak-*.py`; narrow the global `*.html` / `*.txt` ignores.
- Remove committed `chord_bot/chord_bot.egg-info/`.

## Phase 5 — UI design pass & viewer capabilities (user-directed, 2026-07-02) — STATUS: done

Design system: refined dark palette + spacing/radius/shadow/focus tokens in `:root`, mono accent for node/port/param labels, consistent ghost/primary button families (including the previously unstyled desktop Run/Clear), "on air" pulsing Live toggle, category color chips on node headers, per-payload-type wire colors from the server port-type registry, canvas grid + vignette, app title/favicon. Capabilities: ⧉ pop-out viewer window (live MJPEG stream, images — blob frames snapshotted to data URLs — and videos) + Picture-in-Picture for videos; drag-resizable main preview (persisted); keyboard cheat-sheet overlay on `?` / toolbar button. Verified with Playwright: zero console/page errors desktop + mobile, picker spawn, help toggle, run → preview, pop-out (image and live-stream variants), live start/stop, resize drag with persistence.

## Deferred (needs owner decision or larger design)

- Converging the three animation systems on `paramKeyframes` (audit #16) — medium refactor, separate branch recommended.
- chord_bot: shared node-graph kernel vs. independent app (audit #24).
- Replacing `_DynDim` global monkeypatches with explicit canvas params (audit #17) — large, gradual.
- Persistent per-session executor with sim-cache reuse (audit #6 full form) — after Phase 2's conservative step proves out.
- Backfilling method `description=` across the library (audit #20) — good batch job for the Node Doctor once Phase 3 lands.
