"""Generate big white + black spot animations using blob-seeded SH."""
from pathlib import Path
from image_pipeline.core.animation import (
    enable_frame_capture, get_frames, disable_frame_capture,
    frames_to_mp4, capture_frame,
)
import numpy as np, math
from PIL import Image

sh, sw = 512, 768
kx = np.fft.fftfreq(sw) * 2 * math.pi
ky = np.fft.fftfreq(sh) * 2 * math.pi
k2 = kx[np.newaxis, :]**2 + ky[:, np.newaxis]**2
k2[0, 0] = 1.0

r, q0, dt = 2.5, 0.08, 0.08
denom = 1.0 / (1.0 - dt * (-(k2 - q0**2)**2))
dealias = np.ones((sh, sw))
dealias[np.abs(ky) > math.pi * 2 / 3] = 0.0
dealias[:, np.abs(kx) > math.pi * 2 / 3] = 0.0
yy, xx = np.ogrid[:sh, :sw]

for tag, sv in [('42', 42), ('77', 77), ('144', 144), ('999', 999),
                ('big1', 2001), ('big2', 3007)]:
    rng = np.random.default_rng(sv)
    u = np.zeros((sh, sw), np.float64)

    # Add big white blobs
    for _ in range(7):
        cx = rng.uniform(40, sw - 40)
        cy = rng.uniform(40, sh - 40)
        sigma = rng.uniform(25, 50)
        u += 2.5 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))

    # Add big black blobs
    for _ in range(7):
        cx = rng.uniform(40, sw - 40)
        cy = rng.uniform(40, sh - 40)
        sigma = rng.uniform(25, 50)
        u -= 2.5 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))

    print(f'{tag}: init u∈[{u.min():.2f},{u.max():.2f}]')

    enable_frame_capture('138')
    for fr in range(150):
        N = r * u - u**3
        uh = np.fft.fft2(u)
        Nh = np.fft.fft2(N.astype(np.float64))
        uh = (uh + dt * Nh) * denom * dealias
        u = np.fft.ifft2(uh).real

        if fr % 2 == 0:
            u += rng.normal(0, 0.004, (sh, sw))
        u = np.clip(u, -3, 3)

        c = u - np.mean(u)
        scale = max(float(np.percentile(np.abs(c), 95)), 1e-10)
        g = ((np.tanh(c / scale * 3.0) + 1.0) * 127.5).astype(np.uint8)
        capture_frame('138', np.array(Image.fromarray(np.stack([g] * 3, -1), 'RGB')))

    frames = get_frames('138')
    disable_frame_capture()
    if len(frames) >= 2:
        fp = Path(f'blobs-{tag}.mp4')
        frames_to_mp4(iter(frames), fp, fps=24, quality=23, max_frames=len(frames))
        print(f'  {fp.name}: {fp.stat().st_size // 1024}KB | uσ={np.std(u):.2f}')
