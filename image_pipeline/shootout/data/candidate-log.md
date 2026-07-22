# Shootout Candidate Log

Append-only audit trail for the evolution loop. Each entry is real data
from `diagnose_health.py`, never invented. Rotate oldest when >400 lines.

---

## 2026-07-22 — Schema repair + timeout discovery

**Run context:** Route 0 (Leverage Tier) — PHASE 1B/1C artifacts.

### Verified numbers (from `diagnose_health.py` on 649 genomes)

| Metric | Value |
|--------|-------|
| Total genomes | 649 |
| Alive | 357 |
| Dead / rejected | 292 (45%) |
| Render-null genomes | 56 |
| wall_s-null (among render dicts) | 12 |
| Render samples with wall_s | 581 |
| >150s cap (timeout culls) | 165 (28%) |
| >100s slow | 194 |
| Max render wall_s | 669s |
| Ratings in ratings.jsonl | 19 |
| Rated genomes | 19 (STARVED — <20 threshold) |
| Cheap-alive recombine seeds | 180 |

### Top-3 rated seeds (promotion candidates)

| genome_id | rating | origin | motifs | n_drivers |
|-----------|--------|--------|--------|----------|
| g-328f0d37 | 5 | random | sim_backbone, post_fx, post_fx | 5 |
| g-e181c881 | 5 | explorer | masked_composite, post_fx | 9 |
| g-97f1158a | 5 | random | sim_backbone, post_fx, post_fx | 7 |

### Surviving-motif coverage

post_fx (380), sim_backbone (148), masked_composite (48), pattern_blend (47), feedback_loop (21)

### Dead-genome driver hotspots (raw counts)

__lfo__ (800), __counter__ (233), __noise1d__ (118), __ramp__ (93), __envelope__ (39), __image_to_mask__ (38), __strobe__ (37)

### Per-driver dead-rate (THE KEY FINDING — Route 8 hypothesis rejected)

| Driver | Total uses | Dead uses | Dead-rate |
|--------|-----------|-----------|----------|
| __envelope__ | 69 | 39 | 57% |
| __counter__ | 427 | 233 | 55% |
| __image_to_mask__ | 70 | 38 | 54% |
| __lfo__ | 1586 | 800 | 50% |
| __ramp__ | 189 | 93 | 49% |
| __noise1d__ | 252 | 118 | 47% |
| __strobe__ | 91 | 37 | 41% |

**Baseline dead-rate (all genomes): 45%**

**Verdict:** No driver exceeds baseline by more than 12 percentage points.
The Route 8 claim that "driver modulation is NOT reaching the rendered output"
is NOT supported by the data. Drivers are not the primary cull mechanism.

### Real killer: render timeouts

165/581 renders (28%) exceed the 150s cap. Max observed: 669s.
The `avoid_methods` list in `config.json` already names heavy sim IDs
(49, 137, 98, 141, 92, 36, 52, 174, 12) — someone anticipated this,
but no code in the repo consumes that list.

### Action taken this run

- Wrote `diagnose_health.py` (schema-correct, null-safe replacement for
  the stale cron-prompt probes).
- Documented the orphaned `config.json` hooks as a gap.
- Logged the timeout problem as the priority evolution-engineering task.

### Next action

Implement render-timeout pre-check in the shootout engine (once engine
code is present in the repo), OR raise the cap selectively for known-heavy
sims. See `evolution-research.md` entry #3.

---

_End of candidate-log.md. Rotate when >400 lines._
