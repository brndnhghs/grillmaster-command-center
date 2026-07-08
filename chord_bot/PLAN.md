# Chord Bot × Music Theory: Integration Plan

## Current State

- **21 existing nodes** — 11 horizontal, 10 vertical
- **20 of 43 skills** have at least one node touching their domain
- **23 skills** have zero node representation
- Even "covered" skills are **shallow** — each contains 10–50 sub-concepts, parameters, and data tables that aren't exposed in the node graph

## Guiding Principles

1. **Every phase delivers a working, testable thing** — never commit to "figure out X" without producing a concrete node, param set, or data table
2. **Prefer deepening existing nodes over creating new ones** — the existing 19 nodes already handle the most common harmonic patterns. Most skill content translates to **new params, styles, and lookup tables** for existing nodes, not new surfaces
3. **One skill session = one node spec or one deep-param session** — each is 45–90 minutes of focused work
4. **Wiki updates follow each phase** — when a node or param changes, the wiki's live `/api/node-defs` endpoint auto-updates the Node Reference and Augmenter sections

---

## Phase 0: Wiki Completion

**Goal:** Make the interactive wiki fully functional before adding new content.

| Item | Est. | Skill Source |
|------|------|-------------|
| Wire node-grid sections 6 & 7 to `/api/node-defs` | 1 session | — |
| Live p5.js sketch for Graph Model section (H×V topology) | 1 session | — |
| Tension Arc interactive (section 4 already has sketch but verify) | 0.5 session | — |

---

## Phase 1: Deepen Existing Nodes (highest impact per effort)

### ✅ Session 1a — Function Node Deepen [DONE 2026-07-07]

| What | From Skill | Deliverable |
|------|-----------|-------------|
| Add genre-specific Markov tables (blues, classical, pop, film) | `popular-music-analysis` | 4 new transition matrices in `MARKOV_BY_STYLE` dict |
| Add `progression_mode` param (markov/classical) | `diatonic-harmony` | Classical T-Int-D-T mode |
| Add `cadence_chance` param (% chance to auto-resolve from dominant → tonic) | `diatonic-harmony` | `cadence_chance` param (0–1) |
| Add blues + film style quality tables | `popular-music-analysis` | `_QUALITY_TABLE` extended with blues/film rows for all 7 modes |
| Pop chord schema catalog | `popular-music-analysis` | `SCHEMA_CATALOG` with 7 schemas (rotation-capable) |
| 12 new tests passing (132 total, up from 120) | — | `test_function.py` extended |
| UI verified: all 11 params render, style dropdown shows blues/film | — | Live at port 7861 |

Each of these sessions audits one existing node against its mapped skill and adds missing params, styles, or data.

### Session 1a — Function Node Deepen

| What | From Skill | Deliverable |
|------|-----------|-------------|
| Add genre-specific Markov tables (blues, classical, pop, film) | `popular-music-analysis` | 4 new transition matrices in lookup dict |
| Add weighted function target (not just single target) | `diatonic-harmony` | `seed` param + weighted random target |
| Add cadence tendency param (% chance to auto-insert cadence) | `diatonic-harmony` | `cadence_chance` param (0–1) |

### Session 1b — Phrase Node Deepen

| What | From Skill | Deliverable |
|------|-----------|-------------|
| Add mode-specific phrase patterns (dorian, phrygian, lydian) | `modal-harmony` | 7 lookup tables (one per mode) |
| Add folk-style phrase patterns (asymmetric, pentatonic) | `folk-modalities` | Celtic/Balkan patterns |
| Add phrase-type param (sentence, period, repeated, sequential) | `classical-period-theory`, `form-analysis` | 4 types with different internal structures |

### Session 1c — Substitution Node Deepen

| What | From Skill | Deliverable |
|------|-----------|-------------|
| Add Neapolitan substitution type | `chromatic-harmony` | New sub-type in existing node |
| Add augmented-sixth family to substitution (already has node but can also work as V) | `chromatic-harmony` | Cross-reference with existing `augmented_sixth` node |
| Add style-dependent substitution preferences | `jazz-harmony` | Style weight tables |

### Session 1d — Sequence Node Deepen

| What | From Skill | Deliverable |
|------|-----------|-------------|
| Add modal sequence patterns (dorian, phrygian etc.) | `modal-harmony` | 7 mode-specific patterns |
| Add chromatic sequence types (descending chromatic, etc.) | `chromatic-harmony` | New sequence types |
| Add mediant-cycle sequence (desc/asc thirds) | `romantic-harmony` | `mediant-cycle` type |

### Session 1e — Voice Leader Deepen

| What | From Skill | Deliverable |
|------|-----------|-------------|
| Add species-based constraints (no parallel 5ths/8ves, contrary motion bias) | `counterpoint-fugue`, `renaissance-counterpoint` | `species` param + constraint tables |
| Add chromatic voice-leading rules (augmented 2nd avoidance) | `romantic-harmony` | Chromatic constraint set |
| Add voice range limits per instrument family | `orchestration-instrumentation` | Range tables |

### Session 1f — Tension Shaper Deepen

| What | From Skill | Deliverable |
|------|-----------|-------------|
| Add tension profile presets (film arc, classical, pop, minimal) | `film-scoring-media-composition`, `music-cognition-perception` | Profile curves |
| Add tension-per-function override table | `diatonic-harmony` | Per-function tension levels |
| Add microtiming/groove tension effect | `music-cognition-perception` | `timing_offset` param |

### Session 1g — Rhythm Node Deepen

| What | From Skill | Deliverable |
|------|-----------|-------------|
| Add asymmetric meter patterns (5/4, 7/8, 11/8) | `rhythm-meter`, `folk-modalities` | Meter pattern lookup |
| Add swing ratio param (not just binary) | `rhythm-meter`, `jazz-harmony` | `swing_ratio` (float, 1.0–3.0) |
| Add accent pattern library (clave, backbeat, etc.) | `music-dance`, `world-harmony` | Pattern dictionary |

---

## Phase 2: New Nodes — Blues, Gospel, Pop

| # | Node | Axis | Skill Source | Key Parameters |
|---|------|------|-------------|----------------|
| 2a | **Blues Progression** | H | `blues-gospel-harmony` | type (12-bar, 8-bar, 16-bar, minor-blues), style, turnarounds, extensions |
| 2b | **Gospel Turnaround** | V | `blues-gospel-harmony` | type (amen, walkup, walkdown, extended), strength |
| 2c | **Pop Schema** | H | `popular-music-analysis` | schema (axis-progression, doo-wop, pachelbel, singer-songwriter), key, duration |

---

## Phase 3: New Nodes — Rhythm & Movement

| # | Node | Axis | Skill Source | Key Parameters |
|---|------|------|-------------|----------------|
| 3a | **Ostinato** | H | `melody-motif-phrase`, `minimalism-process-music` | pattern (select from presets or custom MIDI notes), octave, length, accent |
| 3b | **Polyrhythm** | V | `rhythm-meter`, `world-harmony` | ratio (3:2, 4:3, 5:4, etc.), emphasis, phase |
| 3c | **Metric Modulator** | V | `rhythm-meter` | from/time, to/time, transition (direct/gradual) |

---

## Phase 4: New Nodes — Chromatic & Extended Harmony

| # | Node | Axis | Skill Source | Key Parameters |
|---|------|------|-------------|----------------|
| 4a | **Neapolitan** | V | `chromatic-harmony` | ✅ DONE — `neapolitan` node (variant, inversion, strength, octave, velocity) |
| 4b | **Planing Chord** | V | `impressionism-early-modern` | ✅ DONE — `planing` node (direction, interval, stack, invert, octave, velocity) |
| 4c | **Quartal Chord** | V | `jazz-harmony`, `impressionism-early-modern` | stack (4ths/5ths/2nds), size (3–5 notes), root |

### ✅ Session 4a — Neapolitan Node [DONE 2026-07-07]

| What | From Skill | Deliverable |
|------|-----------|-------------|
| New vertical `neapolitan` node | `chromatic-harmony` | `nodes/neapolitan.py` + registered in `nodes/__init__.py` |
| ♭II major triad (N), ♭II7, ♭IIM7 | Neapolitan family | `variant` param (neapolitan/neapolitan7/neapolitan_maj7) |
| N⁶ first-inversion voicing | "almost always first inversion (bass on ♭6)" | `inversion` param (0=N, 1=N⁶); numeral `N`/`N⁶` |
| Pre-dominant function, tension bump | ♭2 pulls to V | `function="pre-dominant"`, `+0.25*strength` tension |
| Flat-spelled root (Db/Bb/Eb) | ♭II notation | `FLAT_NAMES[root_pc]` instead of default sharp spelling |
| 11 new tests passing (154 total) | — | `tests/test_neapolitan.py` |
| UI verified: served by `/api/node-defs` with 5 params | — | Live at port 7861 |

**Note — overlaps with Substitution node:** `substitution` already has a one-line `neapolitan` *type* (plain ♭II triad, numeral "N"). This is the fuller, dedicated node (variant + inversion + strength + maj7) per PLAN Phase 4a. Both coexist; the dedicated node exposes the fuller control surface.

| 4c | **Quartal Chord** | V | `jazz-harmony`, `impressionism-early-modern` | stack (4ths/5ths/2nds), size (3–5 notes), root |

### ✅ Session 4b — Planing Chord Node [DONE 2026-07-07]

| What | From Skill | Deliverable |
|------|-----------|-------------|
| New vertical `planing` node | `impressionism-early-modern` | `nodes/planing.py` + registered in `nodes/__init__.py` |
| Parallel motion by fixed interval (whole-tone planing default) | planing technique | `direction` + `interval` params; `keep` stack preserves shape |
| Impressionist sonority rebuild | parallel 7th/9th/quartal | `stack` param (triad/7th/9th/quartal); quartal uses stacked 4ths |
| Parallel 6/3 voicing | *Clair de Lune* style | `invert` param (first-inversion voicing) |
| Non-functional clears numeral | color-over-function | `numeral` set to `""` after planing |
| 13 new tests passing (143 total) | — | `tests/test_planing.py` |
| UI verified: served by `/api/node-defs` with 6 params | — | Live at port 7861 |

**Design notes:** vertical augmenter — produces no sequence entry of its own, modifies the parent horizontal node's state (per `executor.py` augmentation pipeline). `stack="keep"` shifts every voice by the interval; other stacks rebuild the sonority at the planed root, preserving minor/major polarity of the incoming chord. `interval<=0` or empty `voices` → passthrough.

---

## Phase 5: New Nodes — Form & Structure

| # | Node | Axis | Skill Source | Key Parameters |
|---|------|------|-------------|----------------|
| 5a | **Period** | H | `classical-period-theory`, `form-analysis` | type (parallel/contrasting/modulating), length, cadence_type |
| 5b | **Song Section** | H | `songwriting-composition`, `form-analysis` | section (verse/chorus/bridge/breakdown/prechorus), repetitions, dynamic_level |

---

## Phase 6: New Nodes — World & Experimental

| # | Node | Axis | Skill Source | Key Parameters |
|---|------|------|-------------|----------------|
| 6a | **Scale Source** | H | `world-harmony`, `scales-modes` | tradition (Western/Indian/Arabic/Gamelan/Chinese), scale_name, tonic |
| 6b | **PC-Set Transformer** | V | `post-tonal-theory`, `20th-century-atonality` | operation (transpose/invert/multiply), interval, apply_to (voices/root) |
| 6c | **Minimal Process** | H | `minimalism-process-music` | process (additive/phase/multiply/eliminate), cycle_beats, max_voices |

---

## Phase 7: Data & Wiki Polish

| Item | Source Skills | Deliverable |
|------|-------------|-------------|
| World scale lookup tables for Phase 6a | `world-harmony`, `scales-modes` | `data/world_scales.json` |
| Interval tension rankings | `intervals-ear-training`, `music-cognition-perception` | `data/tension_rankings.py` |
| Historical tuning data | `acoustics-tuning-systems` | `data/tuning_systems.py` (just intonation ratios, meantone tables, etc.) |
| Chord quality → emotion mapping | `music-cognition-perception`, `film-scoring-media-composition` | `data/quality_affect.py` |
| Wiki content sync — verify all interactive sketches render live data from `/api/node-defs` | all | Verified wiki sections 1–7 |

---

## Execution Model

### Session Structure
Each session is one focused item from the plan above:

```
1. Load the target music-theory skill (skill_view)
2. Read the target node file (or create stub if new)
3. Extract 2–5 concepts from the skill that are actionable as params/data
4. Implement in the node file
5. Write or update tests (test_nodes.py or a new test file)
6. Run pytest to verify
7. Update skill reference notes if new patterns were discovered
```

### Prioritization

| Tier | Rule | Examples |
|------|------|----------|
| **P0** | Ships this week | Phase 0 (wiki), Phase 1a–1b (function, phrase deepen) |
| **P1** | High impact, independent | Phase 1c–1f (substitution, sequence, voice_leader, tension_shaper) |
| **P2** | Medium effort, clear spec | Phase 2 (blues/gospel/pop), Phase 4 (Neapolitan, planing) |
| **P3** | Requires research | Phase 3 (rhythm), Phase 5 (form), Phase 6 (world/experimental) |
| **P4** | Data/reference | Phase 7 (lookup tables, wiki polish) |

### Signal to Stop or Pivot
Each session ends by asking: *"Which of these should I tackle next?"* — if the answer is one of the items you just shipped doesn't feel right, the user can redirect before the next session starts.

---

## Appendix: Skill → Node Mapping (Full)

| Skill | Existing Node(s) | Gap | Phase |
|-------|-----------------|-----|-------|
| diatonic-harmony | tonic, function, cadence, sequence | Deeper style tables | 1a, 1d |
| scales-modes | tonic (mode param) | Mode-specific patterns for phrase | 1b, 6a |
| chromatic-harmony | modulation, substitution, secondary_dominant, augmented_sixth | Neapolitan, complete augmented 6th family | 1c, 4a |
| jazz-harmony | substitution, color, secondary_dominant | Quartal, drop voicings | 4c |
| counterpoint-fugue | voice_leader | Species constraints | 1e |
| rhythm-meter | rest, rhythm | Polyrhythm, asymmetric meters | 1g, 3b |
| non-chord-tones | pedal, suspension, passing_chord | All types complete (appoggiatura, escape, etc.) | 1d (via sequence deepen) |
| form-analysis | phrase, repeat | Period, sentence, binary | 1b (param), 5a, 5b |
| music-cognition-perception | tension_shaper | Profile curves, affect data | 1f, 7 |
| popular-music-analysis | function, sequence | Pop schemas, genre transitions | 2c |
| film-scoring-media-composition | tension_shaper | Leitmotif, cue structure | 1f |
| improvisation-theory | arpeggiator | More pattern types | (future) |
| classical-period-theory | cadence | Period form, sentence form | 5a |
| songwriting-composition | phrase | Verse/chorus structure | 5b |
| orchestration-instrumentation | bass | Range per instrument family | 1e |
| blues-gospel-harmony | — | Blues progression, gospel turnaround | 2a, 2b |
| modal-harmony | — | Characteristic chord system | 1b, 1d |
| folk-modalities | — | Asymmetric patterns, drone | 1b, 1g |
| romantic-harmony | modulation | Mediant cycles, chromatic voice-leading | 1d, 1e |
| impressionism-early-modern | — | Planing, whole-tone, parallel chords | 4b, 4c |
| world-harmony | — | Scale/chord tables | 6a, 7 |
| melody-motif-phrase | — | Motif builder | 3a |
| minimalism-process-music | — | Phase, additive process | 3a, 6c |
| post-tonal-theory | — | PC-set operations | 6b |
| 20th-century-atonality | — | Serial rows, combinatoriality | 6b |
| renaissance-counterpoint | — | Species rules for voice_leader | 1e |
| medieval-music-theory | — | Organum rules, modal theory | 1b (mode patterns) |
| electronic-music-theory | — | Synth parameter mapping | (future phase) |
| spectral-music | — | Harmonic series data | 7 |
| acoustics-tuning-systems | — | Just intonation tables | 7 |
| aleatoric-graphic-scores | — | Chance operations | (future phase) |
| contemporary-classical | — | Extended techniques | (future phase) |
| extended-techniques-microtonality | — | Microtonal voice data | (future phase) |
| hyper-experimental | — | Generative frameworks | (future phase) |
| music-dance | rhythm | Tempo-range tables | 1g |
| music-language | — | Prosody mapping | (future phase) |
| music-history-period-overview | — | Style period presets | 1a (style tables) |
| music-mathematics | — | Transformation groups | 6b |
| notation-fundamentals | — | Notation export | (future phase) |
| schenkerian-analysis | — | Structural levels | (future phase) |
| choral-vocal-theory | voice_leader | SATB range rules | 1e |
| intervals-ear-training | color | Tension ranking | 1f, 7 |
