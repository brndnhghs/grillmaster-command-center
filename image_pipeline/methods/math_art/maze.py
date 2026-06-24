from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, get_font, BG_DEFAULT, W, H, write_field
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

@method(id="56", name="Maze", category="math_art", tags=["recursive", "fast", "expanded"],
        outputs={"image": "IMAGE", "field": "FIELD"},
         params={
             "cell_size": {"description": "cells size (px)", "min": 4, "max": 40, "default": 10},
             "algorithm": {"description": "maze generation algorithm", "choices": ["recursive_backtracker", "ellers", "prims", "kruskals", "hunt_and_kill", "sidewinder", "growing_tree"], "default": "recursive_backtracker"},
             "geometry": {"description": "grid geometry", "choices": ["rect", "hex", "polar", "circular"], "default": "rect"},
             "style": {"description": "render style", "choices": ["standard", "gradient", "heatmap", "color_regions", "solvetrace", "markers", "corridor_radius"], "default": "standard"},
             "palette": {"description": "PALETTES name for walls", "default": ""},
             "bg_palette": {"description": "PALETTES name for paths/bg (or blank for auto)", "default": ""},
             "wall_thickness": {"description": "wall thickness fraction (0-1)", "min": 0.1, "max": 1.0, "default": 0.5, "step": 0.05},
             "braid": {"description": "braid probability (0=none, 1=max)", "min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05},
             "loops": {"description": "extra loop wall removals per cell", "min": 0, "max": 5, "default": 0},
             "multi_seed": {"description": "number of starting seeds (0=auto)", "min": 0, "max": 20, "default": 1},
             "show_solution": {"description": "highlight solution path", "choices": ["no", "yes"], "default": "no"},
             "entrance_marks": {"description": "draw entrance/exit markers", "choices": ["no", "yes"], "default": "no"},
             "growing_bias": {"description": "growing_tree bias: 0=random, 1=newest, -1=oldest", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.1},
             "color_saturation": {"description": "color intensity", "min": 0.3, "max": 1.5, "default": 0.9, "step": 0.1},
             "rings": {"description": "polar/circular ring count (0=auto)", "min": 0, "max": 60, "default": 0},"anim_mode": {"description": "animation mode", "choices": ["none", "color_cycle"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_maze(out_dir: Path, seed: int, params=None):
    """Render Maze — procedurally generated maze with multiple algorithms.

    Generates a rectangular maze using one of 7 algorithms, then renders
    it with various visual styles. Supports braiding, loops, solution path
    highlighting, and entrance/exit markers. Animation is color-based
    (color_cycle) since the maze geometry is static after generation.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            cell_size: cells size (px)
            algorithm: maze generation algorithm
            geometry: grid geometry (only rect supported)
            style: render style
            palette: PALETTES name for walls
            bg_palette: PALETTES name for paths/bg
            wall_thickness: wall thickness fraction (0-1)
            braid: braid probability (0=none, 1=max)
            loops: extra loop wall removals per cell
            multi_seed: number of starting seeds (0=auto)
            show_solution: highlight solution path
            entrance_marks: draw entrance/exit markers
            growing_bias: growing_tree bias
            color_saturation: color intensity
            rings: polar/circular ring count (0=auto)
            time: animation time in radians
            anim_mode: animation mode (none/color_cycle)
            anim_speed: animation speed multiplier
    """
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = random.Random(seed)

    # ── Optional imports ──
    try:
        import cv2
        _has_cv2 = True
    except ImportError:
        _has_cv2 = False
    from ...core.utils import PALETTES, quantize_to_palette, norm as norm_fn

    # ── Animation ──
    t = anim_time * anim_speed
    if anim_mode == "none":
        t = 0.0

    # ── Params ──
    cs = int(params.get("cell_size", 10))
    algo = params.get("algorithm", "recursive_backtracker")
    geo = params.get("geometry", "rect")
    style = params.get("style", "standard")
    pal_name = params.get("palette", "")
    bg_pal_name = params.get("bg_palette", "")
    wall_thick = float(params.get("wall_thickness", 0.5))
    braid_p = float(params.get("braid", 0.0))
    loop_n = int(params.get("loops", 0))
    n_seeds = int(params.get("multi_seed", 1))
    show_sol = params.get("show_solution", "no")
    ent_marks = params.get("entrance_marks", "no")
    grow_bias = float(params.get("growing_bias", 0.0))
    color_sat = float(params.get("color_saturation", 0.9))
    rings_c = int(params.get("rings", 0))
    pal = PALETTES.get(pal_name, [])
    bg_pal = PALETTES.get(bg_pal_name, [])
    def pc(idx, pl=None):
        p = pl or pal
        return p[idx % len(p)] if p else None

    if geo != "rect":
        img = np.ones((H, W, 3), dtype=np.float32) * 0.15
        write_field(out_dir, np.zeros((H, W), dtype=np.float32))
        capture_frame('56', img)
        save(img.clip(0, 1), mn(56, "Maze"), out_dir)
        return
    cols = max(4, W//cs); rows = max(4, H//cs)
    hw = np.ones((rows+1, cols), dtype=bool); vw = np.ones((rows, cols+1), dtype=bool)
    def _rb():
        v = np.zeros((rows,cols),dtype=bool); s = [(0,0)]; v[0,0]=True
        while s:
            r,c = s[-1]; nb = []
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = r+dr,c+dc
                if 0<=nr<rows and 0<=nc<cols and not v[nr,nc]: nb.append((nr,nc,dr,dc))
            if nb:
                nr,nc,dr,dc = rng.choice(nb); v[nr,nc]=True
                if dr==-1: hw[r,c]=False
                elif dr==1: hw[r+1,c]=False
                elif dc==-1: vw[r,c]=False
                elif dc==1: vw[r,c+1]=False
                s.append((nr,nc))
            else: s.pop()
    def _el():
        sets = list(range(cols))
        def f(x):
            while sets[x]!=x: sets[x]=sets[sets[x]]; x=sets[x]; return x
        def u(a,b):
            ra, rb = f(a), f(b)
            if ra!=rb: sets[rb]=ra
        for r in range(rows-1):
            for c in range(cols-1):
                if rng.random()<0.5 and f(c)!=f(c+1): vw[r,c+1]=False; u(c,c+1)
            for c in range(cols):
                if rng.random()<0.4: hw[r+1,c]=False; sets[c]=c
        for c in range(cols-1):
            if f(c)!=f(c+1): vw[rows-1,c+1]=False; u(c,c+1)
    def _pr():
        v = np.zeros((rows,cols),dtype=bool); v[0,0]=True; wl = []
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = 0+dr,0+dc
            if 0<=nr<rows and 0<=nc<cols: wl.append((0,0,nr,nc))
        rng.shuffle(wl)
        while wl:
            r1,c1,r2,c2 = wl.pop()
            if v[r2,c2]: continue
            v[r2,c2]=True
            if r2==r1-1: hw[r1,c1]=False
            elif r2==r1+1: hw[r2,c1]=False
            elif c2==c1-1: vw[r1,c1]=False
            elif c2==c1+1: vw[r1,c2]=False
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = r2+dr,c2+dc
                if 0<=nr<rows and 0<=nc<cols and not v[nr,nc]: wl.insert(0,(r2,c2,nr,nc))
    def _kr():
        p = {}
        def f(x):
            while p[x]!=x: p[x]=p[p[x]]; x=p[x]; return x
        def u(a,b):
            ra, rb = f(a), f(b)
            if ra!=rb: p[rb]=ra
        ed = []
        for r in range(rows):
            for c in range(cols):
                p[(r,c)]=(r,c)
                if c<cols-1: ed.append(((r,c),(r,c+1),'v'))
                if r<rows-1: ed.append(((r,c),(r+1,c),'h'))
        rng.shuffle(ed)
        for (r1,c1),(r2,c2),ty in ed:
            if f((r1,c1))!=f((r2,c2)):
                u((r1,c1),(r2,c2))
                if ty=='h': hw[r2,c1]=False
                else: vw[r1,c2]=False
    def _hk():
        v = np.zeros((rows,cols),dtype=bool); r=c=0; v[r,c]=True
        while True:
            nb = []
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = r+dr,c+dc
                if 0<=nr<rows and 0<=nc<cols and not v[nr,nc]: nb.append((nr,nc,dr,dc))
            if nb:
                nr,nc,dr,dc = rng.choice(nb); v[nr,nc]=True
                if dr==-1: hw[r,c]=False
                elif dr==1: hw[r+1,c]=False
                elif dc==-1: vw[r,c]=False
                elif dc==1: vw[r,c+1]=False
                r,c = nr,nc
            else:
                fd = False
                for hr in range(rows):
                    for hc in range(cols):
                        if not v[hr,hc]:
                            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr,nc = hr+dr,hc+dc
                                if 0<=nr<rows and 0<=nc<cols and v[nr,nc]:
                                    r,c = hr,hc; v[r,c]=True
                                    if dr==-1: hw[nr,nc]=False
                                    elif dr==1: hw[r,c]=False
                                    elif dc==-1: vw[nr,nc]=False
                                    elif dc==1: vw[r,c]=False
                                    fd=True; break
                            if fd: break
                    if fd: break
                if not fd: return
    def _sw():
        for r in range(rows):
            rs = 0
            for c in range(cols):
                if r>0 and (c==cols-1 or rng.random()<0.5):
                    cl = rng.randint(rs,c); hw[r,cl]=False; rs=c+1
                elif r>0: vw[r,c]=False
    def _gt():
        v = np.zeros((rows,cols),dtype=bool); cl = [(0,0)]; v[0,0]=True
        while cl:
            if grow_bias>=0: idx = -1 if rng.random()<grow_bias else rng.randint(0,len(cl)-1)
            else: idx = 0 if rng.random()<-grow_bias else rng.randint(0,len(cl)-1)
            r,c = cl[idx]; nb = []
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = r+dr,c+dc
                if 0<=nr<rows and 0<=nc<cols and not v[nr,nc]: nb.append((nr,nc,dr,dc))
            if nb:
                nr,nc,dr,dc = rng.choice(nb); v[nr,nc]=True
                if dr==-1: hw[r,c]=False
                elif dr==1: hw[r+1,c]=False
                elif dc==-1: vw[r,c]=False
                elif dc==1: vw[r,c+1]=False
                cl.append((nr,nc))
            else: cl.pop(idx)
    # Generate
    if algo=="recursive_backtracker": _rb()
    elif algo=="ellers": _el()
    elif algo=="prims": _pr()
    elif algo=="kruskals": _kr()
    elif algo=="hunt_and_kill": _hk()
    elif algo=="sidewinder": _sw()
    elif algo=="growing_tree": _gt()
    # Braid
    if braid_p>0:
        de = []
        for r in range(rows):
            for c in range(cols):
                wc = (r>0 and hw[r,c])+(r<rows-1 and hw[r+1,c])+(c>0 and vw[r,c])+(c<cols-1 and vw[r,c+1])
                if wc==3: de.append((r,c))
        rng.shuffle(de)
        for r,c in de[:int(len(de)*braid_p)]:
            op = []
            if r>0 and hw[r,c]: op.append((r,c,'h'))
            if r<rows-1 and hw[r+1,c]: op.append((r+1,c,'h'))
            if c>0 and vw[r,c]: op.append((r,c,'v'))
            if c<cols-1 and vw[r,c+1]: op.append((r,c+1,'v'))
            if op:
                rr,cc,ty = rng.choice(op)
                if ty=='h': hw[rr,cc]=False
                else: vw[rr,cc]=False
    # Loops
    for _ in range(loop_n*rows*cols):
        r,c = rng.randint(0,rows-1),rng.randint(0,cols-1)
        op = []
        if r>0 and hw[r,c]: op.append((r,c,'h'))
        if r<rows-1 and hw[r+1,c]: op.append((r+1,c,'h'))
        if c>0 and vw[r,c]: op.append((r,c,'v'))
        if c<cols-1 and vw[r,c+1]: op.append((r,c+1,'v'))
        if op:
            rr,cc,ty = rng.choice(op)
            if ty=='h': hw[rr,cc]=False
            else: vw[rr,cc]=False
    # Render
    img = np.ones((H,W,3),dtype=np.float32)*0.15
    wc = np.array([0.35,0.25,0.15],dtype=np.float32)
    pc = np.array([0.12,0.10,0.18],dtype=np.float32)
    if anim_mode == "color_cycle":
        hue = t * 0.3
        wc = np.array([
            0.5 + 0.5 * math.sin(hue * 2 * math.pi),
            0.5 + 0.5 * math.sin((hue + 0.33) * 2 * math.pi),
            0.5 + 0.5 * math.sin((hue + 0.67) * 2 * math.pi),
        ], dtype=np.float32) * 0.5 + 0.1
        pc = np.array([
            0.5 + 0.5 * math.sin((hue + 0.5) * 2 * math.pi),
            0.5 + 0.5 * math.sin((hue + 0.83) * 2 * math.pi),
            0.5 + 0.5 * math.sin((hue + 1.17) * 2 * math.pi),
        ], dtype=np.float32) * 0.15 + 0.05
    if pal: wc = np.array(pal[0],dtype=np.float32)/255.0
    if bg_pal: pc = np.array(bg_pal[0],dtype=np.float32)/255.0
    for r in range(rows):
        for c in range(cols):
            img[r*cs:(r+1)*cs, c*cs:(c+1)*cs] = pc
    wt = max(1,int(cs*wall_thick))
    for r in range(rows+1):
        for c in range(cols):
            if hw[r,c]: img[max(0,r*cs-wt//2):min(H,r*cs+wt//2+1), c*cs:(c+1)*cs] = wc
    for r in range(rows):
        for c in range(cols+1):
            if vw[r,c]: img[r*cs:(r+1)*cs, max(0,c*cs-wt//2):min(W,c*cs+wt//2+1)] = wc
    _maze_field = np.ones((H, W), dtype=np.float32)
    for _r in range(rows + 1):
        for _c in range(cols):
            if hw[_r, _c]:
                _maze_field[max(0, _r*cs - wt//2):min(H, _r*cs + wt//2 + 1), _c*cs:(_c+1)*cs] = 0.0
    for _r in range(rows):
        for _c in range(cols + 1):
            if vw[_r, _c]:
                _maze_field[_r*cs:(_r+1)*cs, max(0, _c*cs - wt//2):min(W, _c*cs + wt//2 + 1)] = 0.0
    write_field(out_dir, _maze_field)
    if style=="gradient":
        yy,xx = np.ogrid[:H,:W]; d = np.sqrt(xx**2+yy**2); d = np.clip(1-d/d.max(),0,1)
        for c in range(3): img[:,:,c] *= (0.5+d*0.5)
    if style=="heatmap":
        dm = np.full((rows,cols),1e9,dtype=np.float32); dm[0,0]=0; ch=True
        while ch:
            ch=False
            for r in range(rows):
                for c in range(cols):
                    d = dm[r,c]
                    if r>0 and not hw[r,c] and dm[r-1,c]>d+1: dm[r-1,c]=d+1; ch=True
                    if r<rows-1 and not hw[r+1,c] and dm[r+1,c]>d+1: dm[r+1,c]=d+1; ch=True
                    if c>0 and not vw[r,c] and dm[r,c-1]>d+1: dm[r,c-1]=d+1; ch=True
                    if c<cols-1 and not vw[r,c+1] and dm[r,c+1]>d+1: dm[r,c+1]=d+1; ch=True
        md = dm.max()
        if md>0:
            for r in range(rows):
                for c in range(cols):
                    v = dm[r,c]/md; img[r*cs:(r+1)*cs, c*cs:(c+1)*cs] = np.array([v,0.2,1.0-v],dtype=np.float32)*color_sat
    if style=="color_regions":
        for r in range(rows):
            for c in range(cols):
                col = pc((r*7+c*13)%100) if pal else None
                if col: img[r*cs:(r+1)*cs, c*cs:(c+1)*cs] = np.array(col,dtype=np.float32)/255.0
    if show_sol=="yes":
        par = {}; q = [(0,0)]; vs = {(0,0)}
        while q:
            r,c = q.pop(0)
            if r==rows-1 and c==cols-1: break
            for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in vs:
                    if nr==r-1 and not hw[r,c]: vs.add((nr,nc)); par[(nr,nc)]=(r,c); q.append((nr,nc))
                    if nr==r+1 and not hw[r+1,c]: vs.add((nr,nc)); par[(nr,nc)]=(r,c); q.append((nr,nc))
                    if nc==c-1 and not vw[r,c]: vs.add((nr,nc)); par[(nr,nc)]=(r,c); q.append((nr,nc))
                    if nc==c+1 and not vw[r,c+1]: vs.add((nr,nc)); par[(nr,nc)]=(r,c); q.append((nr,nc))
        cur = (rows-1,cols-1)
        while cur in par:
            r,c = cur; img[r*cs+cs//4:(r+1)*cs-cs//4, c*cs+cs//4:(c+1)*cs-cs//4] = np.array([0.9,0.5,0.1],dtype=np.float32)
            cur = par[cur]
    if ent_marks=="yes":
        img[1:cs-1,1:cs-1] = np.array([0.1,0.8,0.1],dtype=np.float32)
        img[H-cs+1:H-1, W-cs+1:W-1] = np.array([0.8,0.1,0.1],dtype=np.float32)
    capture_frame('56', img); save(img.clip(0,1), mn(56,"Maze"), out_dir)

