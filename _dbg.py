"""Debug KIFS."""
import numpy as np
from pathlib import Path

# replicate core math
def _box_fold(z, s):
    zr = z.real; zi = z.imag
    zr = np.where(zr > s, 2*s-zr, zr); zr = np.where(zr < -s, -2*s-zr, zr)
    zi = np.where(zi > s, 2*s-zi, zi); zi = np.where(zi < -s, -2*s-zi, zi)
    return zr + 1j*zi

def _rot_fold(z, n, extra):
    a = np.angle(z)+extra; r=np.abs(z); period=2*np.pi/max(2,int(n))
    a = np.mod(a, period); a = np.where(a>period*0.5, period-a, a)
    return r*np.exp(1j*a)

W,H=200,150
view=1.5; aspect=H/float(W)
xs=np.linspace(-view,view,W); ys=np.linspace(-view*aspect,view*aspect,H)
xg,yg=np.meshgrid(xs,ys); z=xg+1j*yg
scale=2.5; box=1.0; folds=6; fro=0.4; c=complex(-1.1,0.5); er=10.0; iters=18
escaped=np.zeros(z.shape,dtype=bool)
for i in range(iters):
    z=_box_fold(z,box); z=_rot_fold(z,folds,fro); z=z*scale-c
    mag=np.abs(z); newly=(~escaped)&(mag>er)
    if np.any(newly): escaped|=newly
    if np.all(escaped): break
print("frac escaped:", np.mean(escaped), "iters used:", i+1)
