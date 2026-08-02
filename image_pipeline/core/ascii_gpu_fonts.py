#!/usr/bin/env python3
"""Generate the multi-font GLSL arrays for the ASCII shader.

Reads ascii_gpu_fonts.json, writes GLSL const int[] arrays
for embedding in ascii_gpu_shader.py.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent / '..' / 'image_pipeline' / 'core'
data = json.loads((HERE / 'ascii_gpu_fonts.json').read_text())

names = data['FONT_NAMES']
lo = data['GLYPH_LO']
hi = data['GLYPH_HI']
nf = data['NUM_FONTS']
nc = data['CHARS_PER_FONT']
gw = data['GLYPH_W']
gh = data['GLYPH_H']

# Emit GLSL code
print(f'// {nf} font glyph bitmap sets ({gw}x{gh}, dual-int)')
print(f'// Access: GLYPH_LO[font_idx * {nc} + char_idx]')
print(f'//         GLYPH_HI[font_idx * {nc} + char_idx]')
print(f'#define NUM_FONTS {nf}')
print(f'#define CHARS_PER_FONT {nc}')
print()

# Font name → index map (embedded as int array via choice indices)
print(f'const char* FONT_NAMES[{nf}] = char[{nf}](')
for i, n in enumerate(names):
    comma = ',' if i < nf-1 else ''
    print(f'    \"{n}\"{comma}')
print(');')
print()

offset = 0
for fi in range(nf):
    print(f'// Font {fi}: {names[fi]}')
    lo_start = fi * nc
    hi_start = fi * nc
    nlo = lo[lo_start:lo_start+nc]
    nhi = hi[hi_start:hi_start+nc]
    
    # Verify max values fit
    max_lo = max(abs(v) for v in nlo) if nlo else 0
    max_hi = max(nhi) if nhi else 0
    
    # Print first 10 values as a sample
    sample_lo = ', '.join(str(v) for v in nlo[:10])
    sample_hi = ', '.join(str(v) for v in nhi[:10])
    print(f'//   max_lo={max_lo} max_hi={max_hi}')
    print(f'//   LO sample: {sample_lo}...')
    print(f'//   HI sample: {sample_hi}...')
    print()

# Print flat arrays suitable for ShaderGen
print(f'// Flat GLYPH_LO array ({len(lo)} entries)')
for fi in range(nf):
    for ci in range(0, nc, 16):
        row = lo[fi*nc + ci:fi*nc + min(ci+16, nc)]
        print(f'// f{fi}c{ci}: {" ".join(str(v).rjust(10) for v in row)}')
