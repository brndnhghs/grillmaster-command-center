# Repository Map — Grillmaster Command Center

> Generated: 2026-07-13 · Commit: `5d0eb0e` · Phase 1: Repository Inventory

---

## Purpose

Grillmaster Command Center is a **node-based generative image & video editor**. It replicates Houdini's named-attribute payload model and TouchDesigner's live continuous-cook loop, infused with LLM-agent extensibility (Hermes). 180+ generative methods ship in the library spanning physics simulations, fractals, patterns, filters, math art, compositing, and CL-tool wrappers.

---

## Repository Statistics

| Metric | Value |
|--------|-------|
| Git-tracked files | 560 |
| Total source lines (Python) | ~142,738 |
| Python files | ~230+ |
| Core pipeline | ~19,500 lines |
| Methods (all categories) | ~89,000 lines |
| Tests | ~9,600 lines |
| Frontend (HTML/JS) | ~11,631 lines |
| Chord Bot | ~6,869 lines |
| Instruments (tools) | ~1,763 lines |
| Shell scripts | ~330 lines |
| Contributors | 1 primary (brndnhghs) |

---

## Directory Tree

```
grillmaster-command-center/
├── .claude/                        # Hermes agent launch configuration
│   ├── launch.json
│   └── settings.local.json
├── .obsidian/                      # Obsidian vault config (notes/diaries)
│   ├── app.json, appearance.json, core-plugins.json, graph.json, workspace.json
├── .pre-commit-config.yaml         # Pre-commit hook: method audit
├── .gitignore
├── AGENT_GUIDE.md                  # Method file contract — READ BEFORE TOUCHING METHODS
├── CODEBASE_AUDIT.md               # Historical cleanup audit (2026-06-20)
├── CODE_AUDIT_2026-07-02.md        # Known gaps + remediation roadmap
├── DESIGN.md                       # Authoritative architecture document
├── PHASE1_PLAN.md                  # Infrastructure plan for 7 backlog tasks
├── README.md                       # Project overview
├── SKILL_UPDATE_PROMPT.md          # Meta: instructions for updating agent skills
│
├── image_pipeline/                 # ★ PRIMARY APPLICATION — FastAPI server + node graph
│   ├── __init__.py                 # Package marker
│   ├── server.py                   # ★ FastAPI app (3,015 lines) — routes, live loop, SSE, Node Doctor
│   ├── pipeline.py                 # CLI entry point (legacy, imports some core)
│   ├── nd_runner.py                # Hermes Node Doctor runner (subprocess to Hermes agent)
│   ├── core/                       # ★ CORE ENGINE — executor, registry, utilities
│   │   ├── graph.py                #   ★ GraphExecutor (1,685 lines): topological sort, dirty flags, payload propagation
│   │   ├── registry.py             #   ★ @method decorator + MethodMeta (281 lines)
│   │   ├── port_types.py           #   Open port-type registry (45 lines)
│   │   ├── arch.py                 #   Architecture A/B detection (52 lines)
│   │   ├── timeline.py             #   Animation clock (270 lines)
│   │   ├── easing.py               #   Keyframe easing functions (165 lines)
│   │   ├── utils.py                #   Shared utilities: save_image, write_scalars, write_field, etc. (730 lines)
│   │   ├── animation.py            #   capture_frame helper (296 lines)
│   │   ├── compositing.py          #   53 blend modes + layout compositing (385 lines)
│   │   ├── expr.py                 #   Safe per-frame expression evaluator (123 lines)
│   │   ├── cache.py                #   Content-addressed output cache (69 lines)
│   │   ├── node_tester.py          #   Batch method testing harness (311 lines)
│   │   ├── shaders.py              #   ModernGL GPU shader pipeline (9,454 lines — standalone GLSL)
│   │   ├── postprocess.py          #   OpenCV-based CLI post-processor (1,922 lines)
│   │   ├── annotator.py            #   Output annotation overlay (95 lines)
│   │   ├── quality.py              #   Auto-quality detection (92 lines)
│   │   └── runner.py               #   Parallel/sequential runner (127 lines, CLI-only)
│   ├── methods/                    # ★ NODE LIBRARY — 180+ registered generative methods
│   │   ├── __init__.py             #   Auto-imports all method group packages
│   │   ├── simulations/            #   80+ files, ~35,621 lines — physics/biology sims
│   │   │   ├── __init__.py
│   │   │   ├── gray_scott.py, reaction_diffusion.py, turing_morphogenesis.py
│   │   │   ├── boids.py, particle_life.py, physarum.py, nbody_gravity.py
│   │   │   ├── fitzhugh_nagumo.py, burridge_knopoff.py, faraday_waves.py
│   │   │   ├── cellular_potts.py, ising.py, kuramoto.py, lenia.py
│   │   │   ├── lattice_boltzmann.py, shallow_water.py, sph.py
│   │   │   ├── stable_fluids.py, wave_equation.py, dielectric_breakdown.py
│   │   │   ├── viscous_fingering.py, sandpile.py, dla.py, forest_fire.py
│   │   │   ├── chladni.py, magnetic_pendulum.py, swarmalators.py
│   │   │   ├── ... (80+ files total)
│   │   ├── patterns/              #   53 files, ~13,161 lines — geometric/texture patterns
│   │   │   ├── __init__.py
│   │   │   ├── noise.py, worley_noise.py, phasor_noise.py, gabor_noise.py
│   │   │   ├── truchet.py, smooth_truchet.py, penrose.py, moire.py
│   │   │   ├── metaballs.py, quasicrystal.py, verlet_cloth.py
│   │   │   ├── sdf_scene.py, caustics.py, water_caustics.py
│   │   │   ├── ... (53 files total)
│   │   ├── filters/               #   46 files, ~18,789 lines — image filters/effects
│   │   │   ├── __init__.py
│   │   │   ├── glitch.py, dither.py, pixelsort.py, bloom.py, bokeh.py
│   │   │   ├── chromatic_aberration.py, lens_flare.py, tilt_shift.py, god_rays.py
│   │   │   ├── oil_paint.py, kuwahara.py, bilateral_grid.py
│   │   │   ├── screen_fluid.py, slitscan.py, rolling_shutter.py
│   │   │   ├── tone_mapping.py, hdr.py, color_grade.py, clahe.py
│   │   │   ├── transform.py, data_bending.py, seam_carving.py
│   │   │   ├── ... (46 files total)
│   │   ├── fractals/              #   17 files, ~5,796 lines — escape-time + IFS fractals
│   │   │   ├── __init__.py
│   │   │   ├── fractal.py, julia_set.py, buddhabrot.py, burning_ship.py
│   │   │   ├── mandelbulb.py, newton_fractal.py, chaos_game.py
│   │   │   ├── fractal_flame.py, lsystem.py, sierpinski.py
│   │   │   ├── ... (17 files total)
│   │   ├── math_art/              #   24 files, ~8,280 lines — math visualizations
│   │   │   ├── __init__.py
│   │   │   ├── maze.py, spirograph.py, spherical_harmonics.py
│   │   │   ├── nishita_sky.py, fourier_circles.py, strange_attractors.py
│   │   │   ├── flow_field.py, space_colonization.py, circle_packing.py
│   │   │   ├── marching_squares_contours.py, poincare_tessellation.py
│   │   │   ├── ulam_spiral.py, polytope_4d.py, domain_coloring.py
│   │   │   ├── ... (24 files total)
│   │   ├── codegen/               #   13 files, ~7,202 lines — programmatic/algorithmic generators
│   │   │   ├── __init__.py
│   │   │   ├── simulations.py     #   ★ Method #18 Cellular Automata (major node)
│   │   │   ├── flow_field.py, voronoi_tiles.py, geometric_abstraction.py
│   │   │   ├── collage.py, typography.py, qr_code.py, svg_vector.py
│   │   │   ├── gradient.py, posterize.py, color_palette.py, kaleidoscope.py
│   │   │   ├── ascii_art.py, false_color_ir.py
│   │   ├── compositing/           #   9 files, ~1,605 lines — blend/merge nodes
│   │   │   ├── __init__.py
│   │   │   ├── blend.py, math_merge.py, field_combine.py
│   │   │   ├── particle_merge.py, apply_mask.py, image_to_mask.py
│   │   │   ├── noise_node.py, poisson_edit.py, test_node.py
│   │   ├── system/                #   2 files, ~132 lines — system nodes
│   │   │   ├── __init__.py
│   │   │   └── timeline_node.py   #   Global animation clock node
│   │   ├── channels.py            #   CHOP-style data nodes (Counter, LFO, Beats, etc.)
│   │   ├── blender_render.py      #   Blender 3D render sidecar (796 lines)
│   │   ├── cli_tools.py           #   CLI tool wrappers (ffmpeg, ImageMagick, etc.)
│   │   ├── custom_shader.py       #   Custom GLSL shader node
│   │   ├── gpu_shaders.py         #   Method #82 — GPU shader (1,275 lines)
│   │   ├── io_nodes.py            #   Image input/output nodes (161 lines)
│   │   ├── ml_models.py           #   ML model wrappers (SD 1.5, etc.)
│   │   ├── p5_sketches.py         #   p5.js sketch runner (226 lines)
│   │   └── simulations_cellular.py #   Method #58 — duplicate CA variant (367 lines)
│   ├── tests/                     #   ★ TEST SUITE
│   │   ├── test_method_registration.py   # Core registration integrity
│   │   ├── test_method_id_uniqueness.py  # No duplicate IDs
│   │   ├── test_live_regression.py       # Live mode non-regression (critical)
│   │   ├── test_incremental_recook.py    # Phase 6 incremental cook tests
│   │   ├── test_gpu_shaders.py, test_gpu_parity.py, test_gpu_coverage_audit.py
│   │   ├── test_gpu_node_typed_ports.py, test_gpu_twin_invariant.py
│   │   ├── test_typed_uniforms.py
│   │   ├── test_fidelity.py, test_driver_e2e_fast.py
│   │   ├── test_driver_animation_reaches_pixels.py
│   │   ├── test_chop_drivers_advance.py
│   │   ├── test_keyframe_editor.py
│   │   ├── test_live_server_swap.py, test_live_ws.py, test_live_transport.py
│   │   ├── test_sim_render_health.py, test_generator_render_health.py
│   │   ├── test_ml_nodes_e2e.py, test_3d_sidecar_render.py
│   │   ├── test_blender_render_node.py, test_client3d.py
│   │   ├── test_marching_squares.py, test_utils_dyndim.py
│   │   ├── gpu_parity.py, profile_live.py
│   ├── config/
│   │   └── groups.yaml            # Method grouping for UI
│   ├── 3d/                        # (empty or forthcoming — 3D extension)
│   └── output/                    # Runtime: generated images, sequences, backups
│
├── chord_bot/                     # INDEPENDENT APPLICATION — music chord progression node system
│   ├── __init__.py, server.py, executor.py, registry.py, cli.py
│   ├── chord_types.py, port_types.py, keyframes.py
│   ├── nodes/                     # Chord Bot node library
│   │   ├── __init__.py, tonic.py, function.py, cadence.py, bass.py
│   │   ├── modulation.py, neapolitan.py, passing_chord.py, pedal.py
│   │   ├── phrase.py, planing.py, secondary_dominant.py
│   │   ├── sequence.py, substitution.py, suspension.py
│   │   ├── arpeggiator.py, color.py, repeat.py, rest.py, rhythm.py
│   │   ├── tension_shaper.py, voice_leader.py
│   ├── export/                    # Chord export: MIDI, text
│   │   ├── midi.py, text.py
│   ├── ui/                        # Chord Bot frontend (SPA)
│   │   ├── index.html, wiki.html, app.js, api.js, audio.js
│   │   ├── config.js, drawer.js, preview.js, rail.js, state.js
│   ├── tests/                     # Chord Bot tests
│   │   ├── test_executor.py, test_function.py, test_neapolitan.py
│   │   ├── test_nodes.py, test_planing.py, test_secondary_dominant.py
│   ├── demo/                      # Demo scripts + graph files
│   │   ├── planing_demo_graph.json, render_neapolitan_demo.py, render_planing_demo.py
│   ├── pyproject.toml
│
├── dashboard/                     # UNIFIED CONTROL PANEL (port 7870)
│   ├── __init__.py, __main__.py
│   └── ui/index.html
│
├── ui/                            # Shared frontend assets
│   ├── __init__.py
│   ├── index.html                 # ★ Main editor SPA (9,697 lines, single file)
│   ├── js/
│   │   ├── client3d.js            # 3D viewer client
│   │   └── editor3d.js            # 3D scene editor
│   └── vendor/                    # Third-party JS libs
│       ├── three.module.js, OrbitControls.js, TransformControls.js
│       ├── GLTFLoader.js, USDZLoader.js, BufferGeometryUtils.js
│       ├── p5.min.js, fflate.module.js
│
├── tools/                         # DEVELOPMENT UTILITIES
│   ├── audit_methods.py           # Method contract enforcement (831 lines) — pre-commit hook
│   ├── next_id.py                 # Get next available method ID (38 lines)
│   ├── cron_image_input.py        # Cron-based image input verification (526 lines)
│   ├── validate_image_wiring.py   # Validate image wiring in saved graphs (330 lines)
│   ├── image_wiring_cron.sh, validate_image_wiring.sh  # Shell wrappers
│   └── audit_report.json, audit_report.md, image_wiring_report.json, image_wiring_report.md
│
├── scripts/                       # LAUNCHERS & UTILITY SCRIPTS
│   ├── grillmaster-launcher.sh    # Main server launcher (FastAPI)
│   ├── chord-bot-launcher.sh      # Chord Bot server launcher
│   ├── dashboard.sh               # Unified dashboard launcher
│   ├── tunnel.sh, localhostrun-tunnel.sh, tunnel-watchdog.sh  # Tunneling
│   ├── generate_ca_demos.py       # Generate cellular automata demo graphs
│   ├── ml_node_probe.py           # ML model availability probe
│   └── sim_perf_probe.py          # Simulation performance probe
│
├── tests/                         # Root-level tests (mostly empty)
│   └── __init__.py
│
├── data/                          # Runtime data
│   ├── cache/                     # Server cache files
│   ├── logs/                      # Server logs
│   ├── saved-graphs/              # ★ Persistent graph saves (.json, ~30 files)
│   └── tunnel-info.json           # Tunnel status
│
├── _grid_params/                  # Grid parameter schemes for simulation sweeps
│   └── NN_*.json (40+ files)
│
├── references/                    # Method reference docs
│   ├── method-*.md                # Per-method research references
│   ├── new-method-id-tracker.md
│   └── render-health-contracts.md
│
├── noise_output/                  # Noise method output samples
│   └── (cell, cloud, debug, marble, perlin, plasma, terrain, value, wood)/
│
├── requirements.txt               # Python dependencies
├── package.json / package-lock.json  # Node.js deps (Puppeteer, Three.js, gl)
├── pytest.ini                     # pytest config (slow marker excluded by default)
├── pyproject.toml                 # (project config, if any — note chord_bot has its own)
│
├── *.mp4, *.png, *.jpg            # Generated output files (at repo root)
├── _*.py, _*.sh                   # Scratch/diagnostic scripts
└── .venv/                         # Python virtual environment
```

---

## Build Pipeline

```
                    ┌─────────────────────────────────┐
                    │      uv venv + requirements.txt    │
                    │    fastapi, uvicorn, numpy, scipy  │
                    │    opencv-python, Pillow, pydantic  │
                    │    pyngrok, watchdog, moderngl      │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │    uv run python -m image_pipeline.server  │
                    │          FastAPI (port 7860)               │
                    │    - /api/node-defs  - /api/graph/run      │
                    │    - /api/graph/live - SSE streaming       │
                    │    - Node Doctor / Node Tester             │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
    ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐
    │   ui/index.html  │  │ chord_bot/   │  │  dashboard/     │
    │   Editor SPA     │  │ Music nodes  │  │ Control panel   │
    │   (port 7860)    │  │ (port 7861)  │  │ (port 7870)     │
    └─────────────────┘  └──────────────┘  └────────────────┘
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies for image_pipeline |
| `chord_bot/pyproject.toml` | Chord Bot Python package config |
| `package.json` | Node.js dependencies (Puppeteer, Three.js) |
| `pytest.ini` | pytest markers, excludes slow tests by default |
| `.pre-commit-config.yaml` | Pre-commit hook: runs `audit_methods.py --fail-on-violations` |
| `.claude/launch.json` | Hermes agent launch config |
| `.claude/settings.local.json` | Hermes agent local settings |
| `image_pipeline/config/groups.yaml` | Method grouping for Tab menu |
| `image_pipeline/references/*.md` | Per-method research/design reference docs |

---

## External Services

| Service | Purpose | Required |
|---------|---------|----------|
| Hermes Agent | LLM backend for Node Doctor, Node Tester | Optional (no-LLM fallback) |
| Blender | 3D rendering sidecar (method) | Optional |
| ModernGL | GPU shader execution | Optional |
| Pyngrok | Localhost tunneling | Optional |
| Stable Diffusion (Torch) | ML model method | Optional |

---

## Key Entry Points

| File | Role | Port |
|------|------|------|
| `image_pipeline/server.py` | FastAPI server — main application | 7860 |
| `chord_bot/server.py` | Chord Bot FastAPI server | 7861 |
| `dashboard/__main__.py` | Unified dashboard server | 7870 |
| `scripts/grillmaster-launcher.sh` | Production launcher for server.py | 7860 |
| `scripts/dashboard.sh` | Dashboard launcher | 7870 |

---

## Testing Infrastructure

| Test area | Location | Marker | Purpose |
|-----------|----------|--------|---------|
| Core tests | `image_pipeline/tests/` | ~40 files | Registration, live regression, GPU, simulations, transports |
| Slow tests | same | `-m slow` | Long-running render/perf guards — excluded from default run |
| Chord Bot tests | `chord_bot/tests/` | 6 files | Node execution, harmony correctness |
| Pre-commit gate | `tools/audit_methods.py` | CI | Method contract enforcement |

---

## Key Relationships Between Directories

```
image_pipeline/core/  ←── image_pipeline/methods/  (registry → @method decorator)
       │                         │
       │                   executor calls meta.fn()
       ▼                         │
image_pipeline/server.py ◄───────┘
       │
       ├── ui/index.html     (editor served as static files)
       ├── ui/vendor/        (Three.js, p5.js, etc.)
       │
       └── chord_bot/        (mounted at /chordbot, separate app)
             └── chord_bot/ui/  (separate but similar frontend)

dashboard/
       └── spawns: image_pipeline.server + chord_bot.server
```

---

## Deployment Pipeline

```
git clone → uv venv + uv pip install → run server.py → (optional) tunnel
```

No Docker containers. No CI/CD pipeline (beyond pre-commit). No database dependency. The data layer is the filesystem: output PNGs, NPY sidecar files, JSON graph saves, JSON scalar sidecars.

---

## Design Patterns

1. **Named-Attribute Payload Model** — Nodes produce typed dicts (not blobs); downstream nodes consume by name.
2. **Open Registry** — Port types (`port_types.py`), methods (`registry.py`), and palettes are all pluggable without core changes.
3. **Dual Animation Pattern** — Architecture A (cook-a-window: simulation caches frames) / Architecture B (re-cook per-frame: stateless, time-driven).
4. **Sidecar Protocol** — Non-image outputs (fields, particles, masks, scalars) written as `.npy`/`.json` files alongside the PNG.
5. **Dirty-Flag Selective Recooking** — Graph nodes skip re-execution when params haven't changed (single-frame mode only; live mode force-dirties).
6. **SSE Live Streaming** — Real-time frame preview via MJPEG over multipart/x-mixed-replace.
7. **Hermes LLM Integration** — Node Doctor and Node Tester shell out to the Hermes agent for LLM-powered code repair.
