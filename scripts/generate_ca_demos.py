#!/usr/bin/env python3
"""Generate demo graph files for all 20 Cellular Automata animation modes.

Usage:
    python generate_ca_demos.py

Output: data/saved-graphs/ca-*.json (loadable via pipeline UI → Save/Load)
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "image_pipeline" / "output" / "saved-graphs"
OUT.mkdir(parents=True, exist_ok=True)

# ── Helpers ──

_nc = 0
def nid(): global _nc; _nc += 1; return f"n{_nc}"

_ec = 0
def eid(): global _ec; _ec += 1; return f"e{_ec}"

def node(method_id, params=None, x=0, y=0, render=False):
    return {
        "id": nid(),
        "method_id": method_id,
        "params": params or {},
        "x": x, "y": y,
        "render": render,
    }

def edge(src, src_port, dst, dst_port):
    return {
        "src_node": src["id"],
        "src_port": src_port,
        "dst_node": dst["id"],
        "dst_port": dst_port,
    }

def graph(name, nodes, edges):
    return {
        "version": 1,
        "name": name,
        "nodes": nodes,
        "edges": edges,
    }

def write(name, g):
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(g, indent=2))
    print(f"  ✓ {name}.json  ({len(g['nodes'])} nodes, {len(g['edges'])} edges)")

# ── CA node helper ──
CA_PARAMS = {"rule":"conway","seed_pattern":"random","density":0.3,"color":"mono","size":4}

# ══════════════════════════════════════════════════════════════════════
# 1. simulate
# ══════════════════════════════════════════════════════════════════════
ca = node("18", {**CA_PARAMS}, x=300, y=200, render=True)
write("01-simulate", graph("simulate", [ca], []))

# ══════════════════════════════════════════════════════════════════════
# 2. f2l (age heatmap)
# ══════════════════════════════════════════════════════════════════════
cnt = node("__counter__", {"mode":"loop","end":100}, x=100, y=200)
ca  = node("18", {**CA_PARAMS}, x=400, y=200, render=True)
write("02-f2l", graph("f2l", [cnt, ca], [edge(cnt, "value", ca, "age_input")]))

# ══════════════════════════════════════════════════════════════════════
# 3. rule_cycle
# ══════════════════════════════════════════════════════════════════════
cnt = node("__counter__", {"mode":"loop","end":15}, x=100, y=200)
mth = node("__math__", {"operation":"map_range","map_src_min":0,"map_src_max":15,"map_dst_min":0,"map_dst_max":1}, x=250, y=200)
ca  = node("18", {**CA_PARAMS}, x=500, y=200, render=True)
write("03-rule-cycle", graph("rule_cycle", [cnt, mth, ca], [
    edge(cnt, "value", mth, "a"),
    edge(mth, "value", ca, "rule_select"),
]))

# ══════════════════════════════════════════════════════════════════════
# 4. density_sweep
# ══════════════════════════════════════════════════════════════════════
lfo = node("__lfo__", {"waveform":"sine","min":0.05,"max":0.7}, x=100, y=200)
ca  = node("18", {**CA_PARAMS}, x=400, y=200, render=True)
write("04-density-sweep", graph("density_sweep", [lfo, ca], [edge(lfo, "value", ca, "density")]))

# ══════════════════════════════════════════════════════════════════════
# 5. size_morph
# ══════════════════════════════════════════════════════════════════════
lfo = node("__lfo__", {"waveform":"sine","min":0.0,"max":1.0}, x=100, y=200)
ca  = node("18", {**CA_PARAMS}, x=400, y=200, render=True)
write("05-size-morph", graph("size_morph", [lfo, ca], [edge(lfo, "value", ca, "cell_size")]))

# ══════════════════════════════════════════════════════════════════════
# 6. color_cycle
# ══════════════════════════════════════════════════════════════════════
lfo = node("__lfo__", {"waveform":"sine","min":0.0,"max":1.0}, x=100, y=200)
ca  = node("18", {**CA_PARAMS, "color":"rainbow"}, x=400, y=200, render=True)
write("06-color-cycle", graph("color_cycle", [lfo, ca], [edge(lfo, "value", ca, "hue_shift")]))

# ══════════════════════════════════════════════════════════════════════
# 7. pulse
# ══════════════════════════════════════════════════════════════════════
stb = node("__strobe__", {"rate":0.5,"duty_cycle":0.2}, x=100, y=200)
ca  = node("18", {**CA_PARAMS}, x=400, y=200, render=True)
write("07-pulse", graph("pulse", [stb, ca], [edge(stb, "value", ca, "inject_rate")]))

# ══════════════════════════════════════════════════════════════════════
# 8. wave
# ══════════════════════════════════════════════════════════════════════
lfo = node("__lfo__", {"waveform":"sine","min":0.0,"max":1.0}, x=100, y=200)
ca  = node("18", {**CA_PARAMS}, x=400, y=200, render=True)
write("08-wave", graph("wave", [lfo, ca], [edge(lfo, "value", ca, "wave_phase")]))

# ══════════════════════════════════════════════════════════════════════
# 9. glider_stream
# ══════════════════════════════════════════════════════════════════════
brs = node("__burst__", {"n_pulses":5,"pulse_interval":6,"loop":True}, x=100, y=200)
ca  = node("18", {**CA_PARAMS}, x=400, y=200, render=True)
write("09-glider-stream", graph("glider_stream", [brs, ca], [edge(brs, "value", ca, "inject_rate")]))

# ══════════════════════════════════════════════════════════════════════
# 10. life_music
# ══════════════════════════════════════════════════════════════════════
lfo = node("__lfo__", {"waveform":"sine","min":0.0,"max":1.0,"rate":1.0}, x=100, y=200)
ca  = node("18", {**CA_PARAMS}, x=400, y=200, render=True)
write("10-life-music", graph("life_music", [lfo, ca], [edge(lfo, "value", ca, "rule_select")]))

# ══════════════════════════════════════════════════════════════════════
# 11. explosion
# ══════════════════════════════════════════════════════════════════════
rmp = node("__ramp__", {"mode":"once","start":0.3,"end":0.6,"duration_frames":30}, x=100, y=200)
ca  = node("18", {**CA_PARAMS}, x=400, y=200, render=True)
write("11-explosion", graph("explosion", [rmp, ca], [edge(rmp, "value", ca, "density")]))

# ══════════════════════════════════════════════════════════════════════
# 12. freeze_frame
# ══════════════════════════════════════════════════════════════════════
stb = node("__strobe__", {"rate":4.0,"duty_cycle":0.3}, x=100, y=200)
ca  = node("18", {**CA_PARAMS}, x=400, y=200, render=True)
write("12-freeze-frame", graph("freeze_frame", [stb, ca], [edge(stb, "value", ca, "speed")]))

# ══════════════════════════════════════════════════════════════════════
# 13. rain
# ══════════════════════════════════════════════════════════════════════
n1d = node("__noise1d__", {"min":0.0,"max":0.3,"rate":2.0}, x=100, y=200)
ca  = node("18", {**CA_PARAMS}, x=400, y=200, render=True)
write("13-rain", graph("rain", [n1d, ca], [edge(n1d, "value", ca, "inject_rate")]))

# ══════════════════════════════════════════════════════════════════════
# 14. sandpile
# ══════════════════════════════════════════════════════════════════════
lfo = node("__lfo__", {"waveform":"sine","min":0.0,"max":0.3}, x=100, y=200)
ca  = node("18", {**CA_PARAMS}, x=400, y=200, render=True)
write("14-sandpile", graph("sandpile", [lfo, ca], [edge(lfo, "value", ca, "inject_rate")]))

# ══════════════════════════════════════════════════════════════════════
# 15. edge_growth
# ══════════════════════════════════════════════════════════════════════
ca = node("18", {**CA_PARAMS, "seed_pattern":"edge_fill"}, x=300, y=200, render=True)
write("15-edge-growth", graph("edge_growth", [ca], []))

# ══════════════════════════════════════════════════════════════════════
# 16. spark
# ══════════════════════════════════════════════════════════════════════
ca = node("18", {**CA_PARAMS, "seed_pattern":"spark_center"}, x=300, y=200, render=True)
write("16-spark", graph("spark", [ca], []))

# ══════════════════════════════════════════════════════════════════════
# 17. breed
# ══════════════════════════════════════════════════════════════════════
lfo = node("__lfo__", {"waveform":"sine","min":0.0,"max":1.0,"rate":0.2}, x=100, y=200)
ca  = node("18", {**CA_PARAMS}, x=400, y=200, render=True)
write("17-breed", graph("breed", [lfo, ca], [edge(lfo, "value", ca, "rule_select")]))

# ══════════════════════════════════════════════════════════════════════
# 18. invasion
# ══════════════════════════════════════════════════════════════════════
ca = node("18", {**CA_PARAMS, "seed_pattern":"two_species"}, x=300, y=200, render=True)
write("18-invasion", graph("invasion", [ca], []))

# ══════════════════════════════════════════════════════════════════════
# 19. domination
# ══════════════════════════════════════════════════════════════════════
cnt = node("__counter__", {"mode":"loop","end":15,"step_size":2}, x=100, y=200)
mth = node("__math__", {"operation":"map_range","map_src_min":0,"map_src_max":15,"map_dst_min":0,"map_dst_max":1}, x=250, y=200)
ca  = node("18", {**CA_PARAMS}, x=500, y=200, render=True)
write("19-domination", graph("domination", [cnt, mth, ca], [
    edge(cnt, "value", mth, "a"),
    edge(mth, "value", ca, "rule_select"),
]))

# ══════════════════════════════════════════════════════════════════════
# 20. maze_generator
# ══════════════════════════════════════════════════════════════════════
ca = node("18", {**CA_PARAMS, "rule":"maze", "seed_pattern":"maze_seeds"}, x=300, y=200, render=True)
write("20-maze-generator", graph("maze_generator", [ca], []))

# ══════════════════════════════════════════════════════════════════════
# Combined demo: Glider Swarm
# ══════════════════════════════════════════════════════════════════════
brs = node("__burst__", {"n_pulses":5,"pulse_interval":6,"loop":True}, x=100, y=150)
cnt = node("__counter__", {"mode":"loop","end":15}, x=100, y=300)
mth = node("__math__", {"operation":"map_range","map_src_min":0,"map_src_max":15,"map_dst_min":0,"map_dst_max":1}, x=300, y=300)
ca  = node("18", {**CA_PARAMS}, x=550, y=200, render=True)
write("combined-glider-swarm", graph("glider_swarm", [brs, cnt, mth, ca], [
    edge(brs, "value", ca, "inject_rate"),
    edge(cnt, "value", mth, "a"),
    edge(mth, "value", ca, "rule_select"),
]))

# ══════════════════════════════════════════════════════════════════════
# Combined demo: Color Pulse
# ══════════════════════════════════════════════════════════════════════
stb = node("__strobe__", {"rate":0.5,"duty_cycle":0.2}, x=100, y=150)
lfo = node("__lfo__", {"waveform":"sine","min":0.0,"max":1.0}, x=100, y=300)
ca  = node("18", {**CA_PARAMS, "color":"rainbow"}, x=400, y=200, render=True)
write("combined-color-pulse", graph("color_pulse", [stb, lfo, ca], [
    edge(stb, "value", ca, "inject_rate"),
    edge(lfo, "value", ca, "hue_shift"),
]))

# ══════════════════════════════════════════════════════════════════════
# Combined demo: Wave Explosion
# ══════════════════════════════════════════════════════════════════════
lfo = node("__lfo__", {"waveform":"sine","min":0.0,"max":1.0}, x=100, y=150)
rmp = node("__ramp__", {"mode":"once","start":0.3,"end":0.6,"duration_frames":30}, x=100, y=300)
ca  = node("18", {**CA_PARAMS}, x=400, y=200, render=True)
write("combined-wave-explosion", graph("wave_explosion", [lfo, rmp, ca], [
    edge(lfo, "value", ca, "wave_phase"),
    edge(rmp, "value", ca, "density"),
]))

# ══════════════════════════════════════════════════════════════════════
# Combined demo: Freeze-Frame Rule Cycle
# ══════════════════════════════════════════════════════════════════════
stb = node("__strobe__", {"rate":4.0,"duty_cycle":0.3}, x=100, y=150)
cnt = node("__counter__", {"mode":"loop","end":15}, x=100, y=350)
mth = node("__math__", {"operation":"map_range","map_src_min":0,"map_src_max":15,"map_dst_min":0,"map_dst_max":1}, x=300, y=350)
ca  = node("18", {**CA_PARAMS}, x=550, y=250, render=True)
write("combined-freeze-rule-cycle", graph("freeze_rule_cycle", [stb, cnt, mth, ca], [
    edge(stb, "value", ca, "speed"),
    edge(cnt, "value", mth, "a"),
    edge(mth, "value", ca, "rule_select"),
]))

# ══════════════════════════════════════════════════════════════════════
# Combined demo: Age Heatmap with Density Sweep
# ══════════════════════════════════════════════════════════════════════
cnt = node("__counter__", {"mode":"loop","end":100}, x=100, y=150)
lfo = node("__lfo__", {"waveform":"sine","min":0.05,"max":0.7}, x=100, y=300)
ca  = node("18", {**CA_PARAMS}, x=400, y=200, render=True)
write("combined-age-density", graph("age_density", [cnt, lfo, ca], [
    edge(cnt, "value", ca, "age_input"),
    edge(lfo, "value", ca, "density"),
]))

print(f"\n✅ Generated {len(list(OUT.glob('ca-*.json')))} demo graphs in {OUT}")
