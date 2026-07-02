# Cellular Automata (#18) — Animation Mode Wiring Reference

Each of the 20 removed `anim_mode` branches is recreated by wiring channel
nodes to the CA node's SCALAR inputs. The CA node has these inputs:

| Input | Type | What it controls |
|-------|------|------------------|
| `density` | SCALAR | Initial live cell density (0-1) |
| `speed` | SCALAR | Generations per frame multiplier |
| `hue_shift` | SCALAR | Color shift for rainbow mode (0-1) |
| `rule_select` | SCALAR | Rule index (0-1 → 16 rules) |
| `init_select` | SCALAR | Init pattern index (0-1 → 15 patterns) |
| `cell_size` | SCALAR | Cell size (0-1 → 1-16px) |
| `inject_rate` | SCALAR | Random life injection rate (0-1) |
| `wave_phase` | SCALAR | Wave propagation phase (0-1) |
| `age_input` | SCALAR | Age value for heatmap coloring (0-1) |

## Wiring Recipes

### 1. simulate (default)
No wiring needed. Set `seed_pattern` and `rule` in params. The CA runs
cumulative generations based on time.

### 2. f2l (Frames-to-Live / Age Heatmap)
```
Counter.value → CA.age_input
```
Set Counter `mode=loop`, `end=100`. The Counter's phase drives the age
heatmap coloring (newer cells brighter, older cells dimmer). The CA node
uses the wired age value to color live cells with a heat gradient.

### 3. rule_cycle
```
Counter.value → Math.a (map_range 0→15 to 0→1)
Math.value → CA.rule_select
```
Set Counter `mode=loop`, `end=15`. Cycles through all 16 rules.

### 4. density_sweep
```
LFO.value → CA.density
```
Set LFO `waveform=sine`, `min=0.05`, `max=0.7`. Sweeps density low→high→low.

### 5. size_morph
```
LFO.value → CA.cell_size
```
Set LFO `waveform=sine`, `min=0.0`, `max=1.0`. Morphs cell size 1→16px.

### 6. color_cycle
```
LFO.value → CA.hue_shift
```
Set LFO `waveform=sine`, `min=0.0`, `max=1.0`. Cycles through hues.
Set CA `color` to `rainbow` for full effect.

### 7. pulse
```
Strobe.value → CA.inject_rate
```
Set Strobe `rate=0.5`, `duty_cycle=0.2`. Periodically injects life.

### 8. wave
```
LFO.value → CA.wave_phase
```
Set LFO `waveform=sine`, `min=0.0`, `max=1.0`. Propagates a wave across
the grid.

### 9. glider_stream
```
Burst.value → CA.inject_rate
```
Set Burst `n_pulses=5`, `pulse_interval=6`, `loop=true`. Periodically
injects bursts of life that form glider-like patterns.

### 10. life_music
```
LFO.value → CA.rule_select
```
Set LFO `waveform=sine`, `min=0.0`, `max=1.0`, `rate=1.0`. Rapidly
alternates between rule sets like a musical phrase.

### 11. explosion
```
Ramp.value → CA.density
```
Set Ramp `mode=once`, `start=0.3`, `end=0.6`, `duration_frames=30`.
A single high-density burst that dies out.

### 12. freeze_frame
```
Strobe.value → CA.speed
```
Set Strobe `rate=4.0`, `duty_cycle=0.3`. Alternates between running
simulation and freezing.

### 13. rain
```
Noise1D.value → CA.inject_rate
```
Set Noise1D `min=0.0`, `max=0.3`, `rate=2.0`. Random cells rain down.

### 14. sandpile
```
LFO.value → CA.inject_rate
```
Set LFO `waveform=sine`, `min=0.0`, `max=0.3`. Periodic sand additions.

### 15. edge_growth
Set CA `seed_pattern=edge_fill`. No wiring needed — the init pattern
fills the edges and the CA grows inward.

### 16. spark
Set CA `seed_pattern=spark_center`. No wiring needed — a single spark
at center spreads outward.

### 17. breed
```
LFO.value → CA.rule_select
```
Set LFO `waveform=sine`, `min=0.0`, `max=1.0`, `rate=0.2`. Slowly
oscillates between two rule sets.

### 18. invasion
Set CA `seed_pattern=two_species`. No wiring needed — two species
clusters compete.

### 19. domination
```
Counter.value → Math.a (map_range 0→15 to 0→1)
Math.value → CA.rule_select
```
Set Counter `mode=loop`, `end=15`, `step_size=2`. One rule dominates,
then another sweeps in.

### 20. maze_generator
Set CA `rule=maze`, `seed_pattern=maze_seeds`. No wiring needed —
maze-like growth from scattered seeds.

## Combined Examples

### Glider Swarm (glider_stream + rule_cycle)
```
Burst.value → CA.inject_rate
Counter.value → Math.a (map_range 0→15 to 0→1)
Math.value → CA.rule_select
```

### Color Pulse (pulse + color_cycle)
```
Strobe.value → CA.inject_rate
LFO.value → CA.hue_shift
```

### Wave Explosion (wave + explosion)
```
LFO.value → CA.wave_phase
Ramp.value → CA.density
```

### Freeze-Frame Rule Cycle (freeze_frame + rule_cycle)
```
Strobe.value → CA.speed
Counter.value → Math.a (map_range 0→15 to 0→1)
Math.value → CA.rule_select
```

### Age Heatmap with Density Sweep (f2l + density_sweep)
```
Counter.value → CA.age_input
LFO.value → CA.density
```

## New Channel Nodes

### Strobe (`__strobe__`)
Periodic on/off gate with adjustable duty cycle. Replaces freeze_frame,
spark, and pulse modes.

| Input | Type | Description |
|-------|------|-------------|
| `rate` | SCALAR | Strobe rate in Hz |
| `duty_cycle` | SCALAR | Fraction of cycle that is on (0-1) |

| Output | Type | Description |
|--------|------|-------------|
| `value` | SCALAR | on_value when gate open, off_value when closed |
| `trigger` | SCALAR | 1.0 on rising edge, 0 otherwise |

### Burst (`__burst__`)
Generates a burst of pulses. Replaces glider_stream mode.

| Input | Type | Description |
|-------|------|-------------|
| `trigger` | SCALAR | Rising edge starts a burst |
| `rate` | SCALAR | Pulse rate multiplier |

| Output | Type | Description |
|--------|------|-------------|
| `value` | SCALAR | Pulse amplitude when active, 0 otherwise |
| `active` | SCALAR | 1.0 during burst, 0 otherwise |

### AgeHeat (`__age_heat__`)
Maps a scalar age value to a color output. Can be wired into CA.hue_shift
for rainbow-based age coloring, or used standalone.

| Input | Type | Description |
|-------|------|-------------|
| `age` | SCALAR | Age value to map |
| `max_age` | SCALAR | Max age for normalization |

| Output | Type | Description |
|--------|------|-------------|
| `value` | SCALAR | Normalized age 0-1 |
| `r` | SCALAR | Red channel 0-1 |
| `g` | SCALAR | Green channel 0-1 |
| `b` | SCALAR | Blue channel 0-1 |

## New Init Patterns

| Pattern | Description | Replaces |
|---------|-------------|----------|
| `edge_fill` | Fills all 4 edges with live cells | edge_growth mode |
| `spark_center` | Single spark cluster at center | spark mode |
| `two_species` | Two competing clusters (top-left + bottom-right) | invasion mode |
| `maze_seeds` | Scattered seeds for maze growth | maze_generator mode |
