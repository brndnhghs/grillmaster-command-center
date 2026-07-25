# Module: chord_bot

## Purpose

**Chord Bot** is a generative music theory node graph — a standalone FastAPI application where the graph itself _is_ the timeline. Users compose chord progressions by placing and connecting nodes in a left-to-right directed graph, and the system executes it once to produce a full harmonic sequence.

Unlike the image pipeline (which renders frames over time), Chord Bot treats graph execution as a single pass: horizontal nodes advance a beat clock, vertical nodes augment state without consuming time, and the output is an ordered list of `SequenceEntry` events that can be exported as MIDI, text notation, or JSON.

It lives at `/chordbot` in the main Grillmaster server (reverse-proxied) and is also fully usable standalone on port `7861`.

---

## Architecture

Chord Bot mirrors the architecture of `image_pipeline/core/graph.py` but adapted for music:

```
┌──────────────────────────────────────────────────────────────┐
│                    chord_bot/ package                        │
│                                                              │
│  ┌──────────┐   ┌───────────┐   ┌──────────────────────┐    │
│  │ server.py │──▶│ executor  │──▶│ chord_types.py       │    │
│  │ (FastAPI) │   │ .ChordExe │   │ (HarmonicState,      │    │
│  │  :7861    │   │  cutor    │   │  SequenceEntry, etc)│    │
│  └────┬─────┘   └─────┬─────┘   └──────────────────────┘    │
│       │                │                                     │
│       │         ┌──────┴──────┐                              │
│       │         │  registry   │                              │
│       │         │  .py        │                              │
│       │         │  (@chord)   │                              │
│       │         └──────┬──────┘                              │
│       │                │                                     │
│       │         ┌──────┴──────────────────────┐              │
│       │         │  nodes/ (22 modules)        │              │
│       │         │  tonic, function, phrase,   │              │
│       │         │  cadence, modulation, ...    │              │
│       │         └─────────────────────────────┘              │
│       │                                                      │
│  ┌────┴──────┐    ┌──────────────┐    ┌───────────────┐     │
│  │ export/   │    │ keyframes.py │    │ port_types.py │     │
│  │ midi.py   │    │ (easing &    │    │ (HARMONIC,    │     │
│  │ text.py   │    │  interpolation)│   │  BEAT types)  │     │
│  └───────────┘    └──────────────┘    └───────────────┘     │
│                                                              │
│  ┌────────────┐                                              │
│  │ ui/        │  index.html + 7 JS modules (SPA)             │
│  └────────────┘                                              │
│                                                              │
│  ┌────────────┐                                              │
│  │ cli.py     │  Command-line graph → MIDI                   │
│  └────────────┘                                              │
└──────────────────────────────────────────────────────────────┘
```

### Key architectural principles

| Concept | Chord Bot | Image Pipeline |
|---------|-----------|----------------|
| **Execution model** | Single pass — full progression at once | Per-frame loop |
| **Time** | Beat clock (float), horizontal nodes advance | Frame counter (int), horizontal nodes advance |
| **State** | `HarmonicState` — key, mode, chord, tension, voices, etc. | Image tensor + metadata |
| **Vertical nodes** | Augment state without advancing beat | Modify texture without advancing frame |
| **Topo sort** | Kahn's algorithm, x-position tiebreaker | Same algorithm |
| **Registry** | `@chord` decorator → `ChordMeta` | `@method` decorator → `MethodMeta` |
| **Keyframes** | Per-param tracks evaluated at start-beat | Per-param tracks evaluated at frame |
| **Port types** | HARMONIC, BEAT (open registry) | Various (IMAGE, FIELD, etc.) |
| **Output** | List of `SequenceEntry` → MIDI / text / JSON | Image frames → video / GIF / PNG |

---

## Server Structure

**File:** `server.py` (257 lines)  
**Server:** FastAPI on port 7861, CORS enabled for all origins.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the single-page UI (`index.html`) |
| `GET` | `/wiki` | In-app documentation page |
| `GET` | `/health` | Health check (`{"ok": true}`) |
| `GET` | `/api/node-defs` | All registered node definitions with params, inputs, outputs |
| `GET` | `/api/port-types` | All registered port types (name, color, accepts_from) |
| `GET` | `/api/nodes` | Alias for `/api/node-defs` |
| `GET` | `/api/tunnel-url` | Tunnel info for public URL access |
| `POST` | `/api/graph/execute` | Execute a graph → JSON sequence |
| `POST` | `/api/graph/export-midi` | Execute + download MIDI file |
| `POST` | `/api/execute` | Flat alias for `/api/graph/execute` |
| `POST` | `/api/export/midi` | Flat alias for `/api/graph/export-midi` |
| `POST` | `/api/export/text` | Execute + return plain-text chord chart |
| `GET` | `/{js_file}.js` | Serve JS modules (path-traversal guarded) |

### Request/Response Models

- **`NodeModel`**: `id`, `type`, `x`, `y`, `params`, `paramKeyframes`, `dirty`
- **`EdgeModel`**: `src_node`, `dst_node`, `src_port`, `dst_port`
- **`GraphRequest`**: `nodes[]`, `edges[]`, `tempo` (20–400 BPM, default 120)
- **Response**: `list[SequenceEntry]` — each with `node_id`, `start_beat`, `end_beat`, `duration`, `state`

### Launch

```bash
python -m chord_bot.server                  # http://127.0.0.1:7861
python -m chord_bot.server --port 8000
python -m chord_bot.server --host 0.0.0.0
```

---

## Node System

### Registry (`registry.py`)

The `@chord` decorator (mirroring `@method` in the image pipeline) registers node functions into a global dictionary. Each node is a `ChordMeta` object:

```python
@chord(
    id="tonic",
    name="Tonic",
    category="horizontal",
    axis="horizontal",
    inputs={},                           # source nodes have no harmonic_in
    outputs={"harmonic_out": "HARMONIC"},
    params={"key": {"default": "C"}, ...},
    description="Seeds the key and mode.",
)
def node_tonic(state: HarmonicState, params: dict) -> HarmonicState: ...
```

**`ChordMeta` fields:**
- `id` — unique string key (e.g. `"tonic"`, `"function"`, `"cadence"`)
- `name` — human-readable display name
- `category` — `"horizontal"` or `"vertical"`
- `axis` — `"horizontal"` (advances beat) or `"vertical"` (augments only)
- `params` — dict of `{name: {default, min, max, description}}`
- `inputs` / `outputs` — dict of `{port_name: "HARMONIC"}` (default: `{"harmonic_in": "HARMONIC"}`)
- `fn` — the callable `(HarmonicState, dict) → HarmonicState | list[HarmonicState]`
- `tags` — list of string tags
- `version` — int
- `module` — automatically set to `fn.__module__`

Lookup functions: `get_meta()`, `get_all()`, `get_ids()`, `get_category()`, `get_categories()`, `get_node_defs()`.

### Executor (`executor.py`)

`ChordExecutor` runs the graph:

1. **Parse** nodes/edges into `ChordNode`/`ChordEdge` dataclasses.
2. **Topological sort** (Kahn's algorithm) — x-position as tiebreaker so left-to-right layout matches execution order.
3. **Classify** each node as horizontal or vertical via `ChordMeta.axis`.
4. **Build augmenter map** — BFS from each horizontal node through vertical chains (V→V edges supported).
5. **Execute** each horizontal node in order:
   - Evaluate per-param keyframe tracks at the node's start-beat position.
   - Inject `_beat` (current beat) and `_sequence` (accumulated sequence for Repeat node).
   - Call the horizontal node function → `HarmonicState` or `list[HarmonicState]`.
   - Apply all vertical augmenters (in topological order) to each sub-state.
   - Record `SequenceEntry(state, start_beat, end_beat, node_id)`.
   - Advance beat clock by `state.duration`.
6. **Error handling** — individual node failures print tracebacks but don't crash the graph; downstream nodes continue with the unmodified state.

### Port Types (`port_types.py`)

Open registry for port type definitions:

| Type | Color | Description |
|------|-------|-------------|
| `HARMONIC` | `#9b59b6` (purple) | `HarmonicState` — key, mode, chord, voices, tension, duration |
| `BEAT` | `#e67e22` (orange) | Python float — beat position |

Every node gets `harmonic_in` (input) and `harmonic_out` (output) ports by default, unless it overrides `inputs={}` (source-only nodes like Tonic).

### Keyframes (`keyframes.py`)

Self-contained keyframe interpolation system, mirroring `image_pipeline/core/timeline.py`. Each numeric param can have an independent keyframe track evaluated at the node's start-beat position.

**Easing functions:** linear, ease, ease-in, ease-out, ease-in-out, step, bounce, elastic, cubic-bezier.

**Data model:** `paramKeyframes = {paramName: [{frame: beat, value, easing, handle_in, handle_out}]}`. The `frame` field stores beat position (named `frame` to match the image pipeline's data model).

---

## Music Theory Model

### HarmonicState (`chord_types.py`)

The core data type passed along every wire — a living harmonic context:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `key` | `str` | `"C"` | Root key (C, D, Eb, F#, ...) |
| `mode` | `str` | `"major"` | Scale mode: major, minor, dorian, mixolydian, phrygian, lydian, locrian |
| `function` | `str` | `"tonic"` | Harmonic function: tonic, subdominant, dominant, pre-dominant |
| `chord` | `str` | `"Cmaj7"` | Full chord symbol |
| `root` | `str` | `"C"` | Chord root note |
| `quality` | `str` | `"maj7"` | maj, min, dim, aug, maj7, min7, dom7, dim7, m7b5, sus2, sus4 |
| `inversion` | `int` | `0` | Voicing inversion (0 = root position) |
| `tensions` | `list[int]` | `[]` | Added tensions as semitone offsets |
| `voices` | `list[int]` | `[60,64,67,71]` | MIDI note numbers for chord voices |
| `tension` | `float` | `0.3` | Tension level 0–1 |
| `cadence_count` | `int` | `2` | Number of cadences so far |
| `duration` | `float` | `4.0` | Beats until next chord change |
| `velocity` | `int` | `80` | MIDI velocity (1–127) |
| `bass_note` | `int` | `48` | MIDI note for bass |
| `arp_pattern` | `str` | `None` | Arpeggiation pattern string |
| `numeral` | `str` | `""` | Roman numeral (e.g. "ii7", "V7", "IM7") |
| `degree` | `int` | `0` | Scale degree (0=I, 1=II, ..., 6=VII) |

### Scale Intervals

Predefined for 7 modes: major, minor, dorian, mixolydian, phrygian, lydian, locrian — each with 7 scale degrees.

### Chord Quality Intervals

11 chord qualities mapped to semitone intervals from root: maj, min, dim, aug, maj7, min7, dom7, dim7, m7b5, sus2, sus4.

### Utility Functions

- `note_to_pc(name)` — note name → pitch class 0–11
- `pc_to_note(pc)` — pitch class → note name (sharps or flats)
- `build_chord_name(root, quality)` — e.g. `"Cmaj7"`, `"Dm7"`
- `compute_voices(root_pc, quality, inversion, octave)` → MIDI note list
- `compute_bass(root_pc, inversion, quality, octave)` → MIDI note
- `degree_to_numeral(degree, quality)` → Roman numeral string

### SequenceEntry

Output of the executor — a single event in the rendered chord sequence:

| Field | Type | Description |
|-------|------|-------------|
| `state` | `HarmonicState` | The harmonic state at this event |
| `start_beat` | `float` | Start beat position |
| `end_beat` | `float` | End beat position |
| `node_id` | `str` | Source node ID |

---

## Node Library

### Horizontal Nodes (advance the beat clock)

| Node ID | File | Lines | Description |
|---------|------|-------|-------------|
| `tonic` | `nodes/tonic.py` | 82 | Seeds the key and mode. Sets tension=0, function=tonic. Determines tonic quality from mode (major→maj7, minor→min7, etc.). Source-only node (no `harmonic_in`). |
| `function` | `nodes/function.py` | 566 | Weighted Markov model over harmonic functions. Supports 6 genre-specific Markov tables (classical, jazz, pop, blues, modal, film), 7 pop-chord schemas (singer-songwriter, doo-wop, puff, aeolian, blues, ii-V-I, circle), voice-leading distance scoring, tritone substitution, and style-extended tensions (b9, 9). The most feature-rich node. |
| `phrase` | `nodes/phrase.py` | 301 | Generates a complete body phrase at once, returning `list[HarmonicState]`. Uses style-aware diatonic patterns (classical, jazz, pop, modal) with up to 5 patterns per style. Normalises pattern weights to a beat budget, applies a tension arc (sinusoidal peak), and smooths voice leading within the phrase. |
| `cadence` | `nodes/cadence.py` | 112 | Forces a harmonic resolution. Supports 4 types: authentic (V→I), plagal (IV→I), deceptive (V→vi), half (I→V). Tension resolves proportionally to strength. |
| `modulation` | `nodes/modulation.py` | 95 | Shifts to a new key. Three types: pivot (V7 of new key), direct (tonic jump), chromatic-mediant (major third away). Resets cadence count. |
| `repeat` | `nodes/repeat.py` | 96 | Replays a section of the accumulated sequence. Supports offset, transpose, and velocity scaling. Enables AABA forms and vamps without copying nodes. |
| `rest` | `nodes/rest.py` | 26 | Silent chord — sets velocity=0. Used for pauses and rests. |
| `pedal` | `nodes/pedal.py` | 50 | Holds a bass pedal note while the harmony above changes. |
| `passing_chord` | `nodes/passing_chord.py` | 476 | Inserts diminished passing chords, chromatic approach chords, and secondary leading-tone chords between target harmonies. Includes scale-degree resolution, voice-leading aware placement, and configurable density. |
| `sequence` | `nodes/sequence.py` | 448 | Diatonic sequence generator — repeats a melodic/harmonic pattern at successive scale degrees (e.g. descending thirds sequence). Supports various interval patterns and step sizes. |
| `secondary_dominant` | `nodes/secondary_dominant.py` | 108 | Inserts a secondary dominant (V7/V, V7/ii, etc.) before a target chord. Tonicises non-tonic degrees. |
| `suspension` | `nodes/suspension.py` | 368 | Suspension node — creates 4-3, 7-6, 9-8, and other suspension types. Handles preparation, suspension, and resolution phases with voice-leading constraints. |
| `neapolitan` | `nodes/neapolitan.py` | 120 | Neapolitan sixth chord (♭II in first inversion). A classical chromatic harmony with strong subdominant function. |
| `planing` | `nodes/planing.py` | 161 | Parallel chord motion (planing/parallelism) — moves a chord shape up or down by a fixed interval. Common in impressionist and modern music. |
| `color` | `nodes/color.py` | 59 | Adds chord extensions and alterations (9ths, 11ths, 13ths, ♯5, ♭5, etc.) to enrich the harmonic palette. |
| `bass` | `nodes/bass.py` | 86 | Modifies the bass note independently of the chord. Supports root, 3rd, 5th, 7th, and passing bass patterns. |
| `rhythm` | `nodes/rhythm.py` | 46 | Controls rhythmic feel — swing, syncopation, and rhythmic density for the progression. |
| `tension_shaper` | `nodes/tension_shaper.py` | 71 | Vertical augmenter that adjusts the tension level (0–1) without changing the chord. |
| `voice_leader` | `nodes/voice_leader.py` | 87 | Vertical augmenter that optimises voice leading between the previous and current chord using nearest-note matching. |
| `arpeggiator` | `nodes/arpeggiator.py` | 50 | Vertical augmenter that sets the `arp_pattern` on the state, enabling arpeggiated playback in MIDI export. |
| `substitution` | `nodes/substitution.py` | 195 | Vertical augmenter that applies chord substitutions — tritone substitution, relative minor/major, and modal interchange. |

### Vertical Nodes (augment without advancing beat)

| Node ID | File | Lines | Description |
|---------|------|-------|-------------|
| `tension_shaper` | `nodes/tension_shaper.py` | 71 | Adjusts tension level |
| `voice_leader` | `nodes/voice_leader.py` | 87 | Optimises voice leading |
| `arpeggiator` | `nodes/arpeggiator.py` | 50 | Sets arpeggiation pattern |
| `substitution` | `nodes/substitution.py` | 195 | Applies chord substitutions |

---

## Export Formats

### MIDI (`export/midi.py`)

Pure-Python MIDI writer (no external dependencies). Produces Standard MIDI File Format 1 (multi-track):

- **Track 0:** Tempo map (setpoint event)
- **Track 1:** Chord block — all chord voices on channel 0
- **Track 2:** Bass line — `bass_note` on channel 1 (optional)
- **Track 3:** Arpeggio — when `arp_pattern` is set on channel 2 (optional)

**Arpeggiation patterns** (parsed from `arp_pattern` string):
- `up` — ascending notes
- `down` — descending notes
- `up-down` / `pendulum` — ascending then descending
- `random` — shuffled order
- Pattern syntax: `pattern:rate:gate:span` (e.g. `up:4:0.8:2`)

**Parameters:** `tempo_bpm`, `ticks_per_beat` (default 480), `include_bass`, `include_arp`.

### Text (`export/text.py`)

Three export functions:

- `progression_to_text(sequence)` — returns a human-readable ASCII chord chart with columns: Beat range, Chord, Function, Tension, Key, Mode, Duration, Voices.
- `write_text(sequence, path)` — writes the chart to a file.
- `write_json(sequence, path)` — writes the full sequence as JSON (all `SequenceEntry.to_dict()`).

---

## UI

**File:** `ui/index.html` (308 lines) + 7 JavaScript modules (total ~2,400 lines)

A single-page application (SPA) with a dark theme, built as a modular ES module system.

### Architecture

```
index.html
├── app.js        (525 lines) — Main controller, event wiring, init
├── state.js      (47 lines)  — Central state: nodes, edges, selection
├── rail.js       (212 lines) — Timeline rail: renders node blocks, handles drag
├── api.js        (42 lines)  — REST API calls (execute, export, node-defs)
├── drawer.js     (136 lines) — Parameter drawer UI for selected node
├── preview.js    (268 lines) — Piano roll + tension graph canvases
├── audio.js      (126 lines) — Web Audio API playback (MIDI synthesis)
├── config.js     (29 lines)  — UI configuration constants
└── wiki.html     (1021 lines)— In-app documentation/help page
```

### Features

1. **Timeline rail** — horizontal scrollable strip showing node blocks with chord name, Roman numeral, function badge, tension bar, and augmenter dots.
2. **Node adding** — via "+ Add" button or insert buttons between nodes, with a popup picker showing all registered node types (colour-coded by function).
3. **Parameter drawer** — opens when a node is selected; shows all configurable params with appropriate controls (text, number, range, select, checkbox). Includes augmenter management.
4. **Piano roll preview** — canvas-based piano roll showing all MIDI voices with colour-coded chords (green=tonic, blue=subdominant, red=dominant, yellow=pre-dominant).
5. **Tension graph** — canvas-based tension curve over the progression.
6. **Playback** — Web Audio API synthesis with real-time playback, loop mode, and BPM control.
7. **Export** — MIDI download, JSON save/load (local file), text chart.
8. **Mobile responsive** — bottom tab bar switching between Timeline, Params, and Preview panels.
9. **Tunnel integration** — shows public tunnel URL when available, links to the image pipeline.

### UI Config (`config.js`)

```javascript
const CONFIG = {
    API_BASE: window.location.origin,
    COLORS: {
        tonic: "var(--tonic)", subdominant: "var(--sub)",
        dominant: "var(--dom)", pre_dominant: "var(--pre)",
    },
    DEFAULT_BPM: 120,
    PIANO_ROLL_OCTAVES: 5,
};
```

---

## CLI

**File:** `cli.py` (140 lines)

Command-line tool for headless graph execution:

```bash
python -m chord_bot.cli graph.json --output output.mid --tempo 120
python -m chord_bot.cli graph.json --text               # print chart only
python -m chord_bot.cli graph.json --json out.json       # also export JSON
```

Options: `--output`, `--tempo`, `--ticks`, `--no-bass`, `--no-arp`, `--text`, `--json`.

---

## Demos

| File | Lines | Description |
|------|-------|-------------|
| `demo/render_neapolitan_demo.py` | 198 | Generates a Neapolitan sixth chord progression via the graph API |
| `demo/render_planing_demo.py` | 224 | Generates a parallel chord motion (planing) progression |

---

## Test Coverage

| File | Lines | Description |
|------|-------|-------------|
| `tests/test_executor.py` | 190 | Executor tests: graph execution, topo sort, augmenter chains, error handling |
| `tests/test_function.py` | 308 | Function node tests: Markov transitions, schema progressions, voice leading, quality tables |
| `tests/test_nodes.py` | 714 | Comprehensive node tests: all node types, parameter combinations, edge cases |
| `tests/test_neapolitan.py` | 142 | Neapolitan node tests |
| `tests/test_planing.py` | 171 | Planing node tests |
| `tests/test_secondary_dominant.py` | 64 | Secondary dominant node tests |
| `tests/__init__.py` | 0 | Empty package init |

**Total: 1,589 lines of tests.**

---

## Key Differences from the Image Pipeline

| Aspect | Chord Bot | Image Pipeline |
|--------|-----------|----------------|
| **Domain** | Music theory — chord progressions | Visual — image/video generation |
| **State type** | `HarmonicState` (key, mode, chord, voices, tension) | `numpy.ndarray` (image tensor) |
| **Execution** | Single pass, once per graph | Frame loop, once per frame |
| **Output** | `SequenceEntry` list → MIDI / text / JSON | Image frames → video / GIF |
| **Port types** | HARMONIC, BEAT | IMAGE, FIELD, MASK, etc. |
| **Node count** | 22 node modules | 100+ method modules |
| **Server port** | 7861 | 7860 |
| **UI** | SPA with piano roll + tension graph | SPA with canvas-based graph editor |
| **Export** | MIDI (pure Python), text, JSON | MP4, GIF, PNG, WebP |
| **Dependencies** | Minimal: FastAPI, uvicorn, pydantic | Heavy: OpenCV, numpy, torch, CUDA, etc. |
| **CLI** | `python -m chord_bot.cli` | `python -m image_pipeline.cli` |
| **Keyframes** | Beat-based `paramKeyframes` | Frame-based `paramKeyframes` |
| **Augmenters** | Vertical nodes in chord graph | Pre/post-processor nodes in image graph |

---

## Dependencies

**Runtime:**
- `fastapi` — web framework
- `uvicorn` — ASGI server
- `pydantic` — request/response validation

**No external music dependencies** — MIDI is written in pure Python (no `midiutil`, no `music21`, no `pretty_midi`). The entire system is self-contained.

---

## File Listing

### Core package (7 files, 1,093 lines)

| File | Lines | Description |
|------|-------|-------------|
| `__init__.py` | 2 | Package init, triggers all node registrations |
| `server.py` | 257 | FastAPI server, API endpoints, CORS, static file serving |
| `executor.py` | 285 | Graph executor — topo sort, horizontal/vertical execution, augmenter chains |
| `registry.py` | 152 | `@chord` decorator, `ChordMeta`, global registry, lookup functions |
| `chord_types.py` | 228 | `HarmonicState`, `SequenceEntry`, note/scale/quality utilities |
| `port_types.py` | 51 | Port type registry — HARMONIC, BEAT |
| `keyframes.py` | 230 | Per-param keyframe tracks, easing functions, interpolation |
| `cli.py` | 140 | CLI tool — graph JSON → MIDI |

### Nodes (22 files, 3,666 lines)

| File | Lines | Description |
|------|-------|-------------|
| `nodes/__init__.py` | 23 | Auto-imports all 22 node modules |
| `nodes/tonic.py` | 82 | Seeds key and mode, zero tension |
| `nodes/function.py` | 566 | Weighted Markov model with genre tables, schemas, voice leading |
| `nodes/phrase.py` | 301 | Generates complete body phrase from style-aware patterns |
| `nodes/cadence.py` | 112 | Authentic/plagal/deceptive/half cadence |
| `nodes/modulation.py` | 95 | Pivot/direct/chromatic-mediant key change |
| `nodes/repeat.py` | 96 | Replays section of accumulated sequence |
| `nodes/rest.py` | 26 | Silent chord (velocity=0) |
| `nodes/pedal.py` | 50 | Holds bass pedal note |
| `nodes/passing_chord.py` | 476 | Diminished/chromatic approach chords |
| `nodes/sequence.py` | 448 | Diatonic sequence generator |
| `nodes/secondary_dominant.py` | 108 | V7/V, V7/ii, etc. |
| `nodes/suspension.py` | 368 | 4-3, 7-6, 9-8 suspensions |
| `nodes/neapolitan.py` | 120 | ♭II Neapolitan sixth |
| `nodes/planing.py` | 161 | Parallel chord motion |
| `nodes/substitution.py` | 195 | Tritone, relative, modal interchange substitutions |
| `nodes/color.py` | 59 | Chord extensions and alterations |
| `nodes/bass.py` | 86 | Independent bass note control |
| `nodes/rhythm.py` | 46 | Swing, syncopation, rhythmic density |
| `nodes/tension_shaper.py` | 71 | Adjusts tension level (vertical augmenter) |
| `nodes/voice_leader.py` | 87 | Optimises voice leading (vertical augmenter) |
| `nodes/arpeggiator.py` | 50 | Sets arpeggiation pattern (vertical augmenter) |

### Export (3 files, 309 lines)

| File | Lines | Description |
|------|-------|-------------|
| `export/__init__.py` | 1 | Package init |
| `export/midi.py` | 237 | Pure-Python MIDI format 1 writer — chord, bass, arp tracks |
| `export/text.py` | 71 | Text chart, text file, and JSON export |

### UI (9 files, 3,697 lines)

| File | Lines | Description |
|------|-------|-------------|
| `ui/index.html` | 308 | Main SPA with dark theme, 4-panel layout |
| `ui/wiki.html` | 1,021 | In-app documentation page |
| `ui/app.js` | 525 | Main controller, event wiring, initialization |
| `ui/state.js` | 47 | Central state management (nodes, edges, selection) |
| `ui/rail.js` | 212 | Timeline rail rendering and drag interaction |
| `ui/api.js` | 42 | REST API client (execute, export, node-defs) |
| `ui/drawer.js` | 136 | Parameter drawer UI for selected node |
| `ui/preview.js` | 268 | Piano roll canvas + tension graph canvas |
| `ui/audio.js` | 126 | Web Audio API playback |
| `ui/config.js` | 29 | UI configuration constants |

### Tests (7 files, 1,589 lines)

| File | Lines | Description |
|------|-------|-------------|
| `tests/__init__.py` | 0 | Empty package init |
| `tests/test_executor.py` | 190 | Executor tests |
| `tests/test_function.py` | 308 | Function node tests |
| `tests/test_nodes.py` | 714 | Comprehensive node tests |
| `tests/test_neapolitan.py` | 142 | Neapolitan node tests |
| `tests/test_planing.py` | 171 | Planing node tests |
| `tests/test_secondary_dominant.py` | 64 | Secondary dominant tests |

### Demos (2 files, 422 lines)

| File | Lines | Description |
|------|-------|-------------|
| `demo/render_neapolitan_demo.py` | 198 | Neapolitan progression demo |
| `demo/render_planing_demo.py` | 224 | Planing progression demo |

---

**Totals:**
- Core: 7 files, 1,093 lines
- Nodes: 22 files, 3,666 lines
- Export: 3 files, 309 lines
- UI: 10 files, 3,697 lines
- Tests: 7 files, 1,589 lines
- Demos: 2 files, 422 lines
- **Grand total: 51 files, 10,776 lines**