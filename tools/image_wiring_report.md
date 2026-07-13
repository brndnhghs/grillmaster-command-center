# Image-Input Wiring Report

- generated: 2026-07-13T15:14:42.683103+00:00
- graphs scanned: 45
- errors: 9
- warnings: 32

```
========================================================================
 IMAGE-INPUT WIRING VALIDATION
========================================================================
 graphs scanned : 45
 ERROR findings : 9
 WARN  findings : 32

── graph: active   (0 ERROR / 1 WARN) ──
  [WARN] W_DEAD_PORT @ n1 (91)
        image_in unwired but method never reads the wired image (dead/optional port — safe but worth pruning)

── graph: age_density   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n54 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: breed   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n32 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: color_cycle   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n12 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: color_pulse   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n44 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: density_sweep   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n8 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: domination   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n36 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: edge_growth   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n29 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: explosion   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n22 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: f2l   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n3 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: freeze_frame   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n24 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: freeze_rule_cycle   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n51 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: glider_stream   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n18 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: glider_swarm   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n41 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: invasion   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n33 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: life_music   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n20 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: maze_generator   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n37 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: pulse   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n14 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: rain   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n26 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: roundtrip-test   (2 ERROR / 0 WARN) ──
  [ERROR] E_DST_MISSING @ -
        edge src=sn1->sn2:image_in references missing dst node
  [ERROR] E_ORPHAN_NODE @ sn1 (gradient)
        node references unknown method_id 'gradient' (no NodeDef)

── graph: rt-server-test   (1 ERROR / 0 WARN) ──
  [ERROR] E_DANGLING_REQUIRED @ x1 (10)
        image_in has no inbound edge and the method consumes the wired image — will run without an upstream image (broken)

── graph: rule_cycle   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n6 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: sandpile   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n28 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: shootout-g-06f260d7   (0 ERROR / 1 WARN) ──
  [WARN] W_DEAD_PORT @ n1 (170)
        image_in unwired but method never reads the wired image (dead/optional port — safe but worth pruning)

── graph: shootout-g-18612554   (1 ERROR / 0 WARN) ──
  [ERROR] E_ORPHAN_NODE @ n1 (156)
        node references unknown method_id '156' (no NodeDef)

── graph: shootout-g-2661455a   (1 ERROR / 3 WARN) ──
  [ERROR] E_DANGLING_REQUIRED @ n1 (42)
        image_in has no inbound edge and the method consumes the wired image — will run without an upstream image (broken)
  [WARN] W_OPTIONAL_UNWIRED @ n5 (37)
        optional image port 'image_2' is unwired (method may fall back to an internal/default source)
  [WARN] W_OPTIONAL_UNWIRED @ n5 (37)
        optional image port 'image_3' is unwired (method may fall back to an internal/default source)
  [WARN] W_OPTIONAL_UNWIRED @ n5 (37)
        optional image port 'image_4' is unwired (method may fall back to an internal/default source)

── graph: shootout-g-47679b1a   (0 ERROR / 1 WARN) ──
  [WARN] W_DEAD_PORT @ n4 (159)
        image_in unwired but method never reads the wired image (dead/optional port — safe but worth pruning)

── graph: shootout-g-a4d26afb   (1 ERROR / 1 WARN) ──
  [WARN] W_DEAD_PORT @ n1 (114)
        image_in unwired but method never reads the wired image (dead/optional port — safe but worth pruning)
  [ERROR] E_DANGLING_REQUIRED @ n2 (43)
        image_in has no inbound edge and the method consumes the wired image — will run without an upstream image (broken)

── graph: shootout-g-daf71c35   (1 ERROR / 0 WARN) ──
  [ERROR] E_DANGLING_REQUIRED @ n2 (119)
        image_in has no inbound edge and the method consumes the wired image — will run without an upstream image (broken)

── graph: shootout-g-f5d79c35   (1 ERROR / 0 WARN) ──
  [ERROR] E_DANGLING_REQUIRED @ n1 (434)
        image_in has no inbound edge and the method consumes the wired image — will run without an upstream image (broken)

── graph: simulate   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n1 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: size_morph   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n10 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: spark   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n30 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: test-edge-rt   (1 ERROR / 0 WARN) ──
  [ERROR] E_DANGLING_REQUIRED @ nmr18e379 (10)
        image_in has no inbound edge and the method consumes the wired image — will run without an upstream image (broken)

── graph: tune-smoke   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n1 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

── graph: wave_explosion   (0 ERROR / 1 WARN) ──
  [WARN] W_OPTIONAL_UNWIRED @ n47 (18)
        optional image port 'seed_image' is unwired (method may fall back to an internal/default source)

```
