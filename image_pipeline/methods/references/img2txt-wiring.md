# Method 44 — img2txt Animation Mode Wiring Reference

Each removed `anim_mode` is recreated by wiring channel nodes to the node's SCALAR/FIELD inputs.

## Removed Modes

| Old Mode | What It Did | Replacement |
|----------|-------------|-------------|
| `circle_morph` | Modulated circle radius with sine wave | Wire `LFO → img2txt.circle_radius` (SCALAR input) |
| `char_cycle` | Cycled through character sets | Wire `Counter → Math(map_range) → img2txt.charset` (via a custom node or manual charset switching) |

## Wiring Recipes

### circle_morph → LFO → circle_radius

```
[LFO] ──value──→ [img2txt.circle_radius]
```

- LFO mode: `sine`
- Rate: 0.3 Hz (matches old `t * 0.3` frequency)
- The FIELD input on `circle_radius` accepts the LFO's SCALAR output (executor cross-wires SCALAR→FIELD as uniform array)
- Circle radius oscillates between ~2 and ~50 pixels

### char_cycle → Counter → charset (manual)

The old `char_cycle` mode cycled through 4 hardcoded charsets. Since `charset` is a string param (not wireable), this requires either:

1. **A custom node** that outputs a charset string based on a SCALAR input
2. **Manual switching** — wire a Counter to a display node and manually set the charset per frame

For now, the recommended approach is to wire an LFO into `circle_radius` (circle_morph replacement) which provides the most visually interesting animation.

## Combined Examples

### Circle morph + speed modulation

```
[LFO(sine, 0.3Hz)] ──value──→ [img2txt.circle_radius]
[LFO(sine, 0.1Hz)] ──value──→ [img2txt.anim_speed]
```

Circle radius pulses while animation speed slowly oscillates.

### Input image + circle radius modulation

```
[Perlin Noise] ──image──→ [img2txt.image_in]
[LFO(sine, 0.3Hz)] ──value──→ [img2txt.circle_radius]
```

When `image_in` is wired, the node uses the upstream image instead of generating circles. The `circle_radius` FIELD input still modulates but has no visible effect (circles aren't generated when an input image is present). Wire FIELD inputs to `ascii_width` or `font_size` instead for visible modulation on input images:

```
[Perlin Noise] ──image──→ [img2txt.image_in]
[LFO(sine, 0.2Hz)] ──value──→ [img2txt.font_size]
[LFO(triangle, 0.1Hz)] ──value──→ [img2txt.ascii_width]
```

Font size pulses and ASCII width sweeps, creating a dynamic text rendering of the input image.
