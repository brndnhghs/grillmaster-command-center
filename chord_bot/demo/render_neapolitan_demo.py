"""Neapolitan Chord — visual + audio demo renderer.

Renders the classic chromatic progression I → N⁶ (Neapolitan in first
inversion) → V → I, with the characteristic ♭6→5 bass half-step resolution
(F → G). Produces a piano-roll MP4 with synthesized audio.
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chord_bot.chord_types import HarmonicState, compute_voices, compute_bass, note_to_pc
from chord_bot.nodes.neapolitan import node_neapolitan

BPM = 72
PX_PER_BEAT = 110
TOP_PC = note_to_pc("C") + 12 * 5   # ~C6
BOT_PC = note_to_pc("C") + 12 * 2   # ~C3
PITCH_SPAN = TOP_PC - BOT_PC
ROLL_H = PITCH_SPAN * 9 + 60
FPS = 30
SR = 44100
OUTDIR = "/Users/admin/Documents/GitHub/grillmaster-command-center/chord_bot/demo"


def midi_to_freq(m: int) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def make_state(root: str, quality: str, duration: float, num: str = "") -> HarmonicState:
    rpc = note_to_pc(root)
    return HarmonicState(
        key=root, mode="major", chord=f"{root}{quality}", root=root, quality=quality,
        voices=compute_voices(rpc, quality), bass_note=compute_bass(rpc, 0, quality),
        tension=0.2, duration=duration, velocity=85, numeral=num, degree=0,
    )


def build_progression() -> list[tuple[str, HarmonicState]]:
    """I (Cmaj7) → N⁶ (Db, first inversion) → V7 (G7) → I (Cmaj7)."""
    seq: list[tuple[str, HarmonicState]] = []
    i = make_state("C", "maj7", 4.0, "IM7")
    seq.append((i.chord, i))
    n6 = node_neapolitan(i.copy(), {"variant": "neapolitan", "inversion": 1,
                                    "octave": 4, "velocity": 85, "strength": 1.0})
    n6 = n6.copy(); n6.duration = 4.0
    seq.append((n6.chord, n6))
    v = make_state("G", "dom7", 4.0, "V7")
    v = v.copy(); v.function = "dominant"
    seq.append((v.chord, v))
    i2 = make_state("C", "maj7", 4.0, "IM7")
    seq.append((i2.chord, i2))
    return seq


def synth_audio(seq) -> np.ndarray:
    beats = [s.duration for _, s in seq]
    total_beats = sum(beats)
    total_samples = int(total_beats * 60.0 / BPM * SR)
    out = np.zeros(total_samples, dtype=np.float64)
    t0 = 0
    for (label, st), dur in zip(seq, beats):
        seg = int(dur * 60.0 / BPM * SR)
        if seg <= 0:
            continue
        t = np.linspace(0.0, dur * 60.0 / BPM, seg, endpoint=False)
        sig = np.zeros(seg)
        for v in st.voices:
            f = midi_to_freq(v)
            sig += (np.sin(2 * math.pi * f * t)
                    + 0.35 * np.sin(2 * math.pi * 2 * f * t)
                    + 0.18 * np.sin(2 * math.pi * 3 * f * t))
        sig /= max(1.0, len(st.voices))
        bf = midi_to_freq(st.bass_note)
        sig += 0.5 * (np.sin(2 * math.pi * bf * t)
                      + 0.3 * np.sin(2 * math.pi * 2 * bf * t))
        a = int(0.02 * SR); d = int(0.06 * SR); r = int(0.25 * SR)
        env = np.ones(seg)
        env[:a] = np.linspace(0, 1, a) if a else 1
        env[a:a + d] = np.linspace(1, 0.75, max(1, d))
        env[max(0, seg - r):] = np.linspace(0.75, 0.0, max(1, r))
        sig *= env
        out[t0:t0 + seg] += sig * 0.22
        t0 += seg
    out = np.clip(out, -1.0, 1.0)
    fade = int(0.05 * SR)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    return (out * 32767).astype(np.int16)


def _font(size: int):
    for path in ["/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_roll_frame(seq, play_beat: float) -> Image.Image:
    beat_w = PX_PER_BEAT
    total_beats = sum(s.duration for _, s in seq)
    W = int(total_beats * beat_w) + 220
    H = ROLL_H + 70
    img = Image.new("RGB", (W, H), (16, 18, 26))
    d = ImageDraw.Draw(img)

    for pc in range(BOT_PC, TOP_PC + 1):
        y = H - 50 - (pc - BOT_PC) * 9
        is_c = (pc % 12) == 0
        d.line([(150, y), (W, y)], fill=(54, 58, 70) if is_c else (30, 33, 44), width=1)
    f = _font(13)
    for pc in range(BOT_PC, TOP_PC + 1):
        if pc % 12 == 0:
            y = H - 50 - (pc - BOT_PC) * 9
            name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][pc % 12]
            d.text((8, y - 7), f"{name}{pc // 12 - 1}", fill=(150, 160, 180), font=f)

    x = 160.0
    bf = _font(16)
    # distinct colors per function: I blue, N teal, V amber
    func_col = {"tonic": (90, 170, 240), "pre-dominant": (80, 220, 210),
                "dominant": (245, 190, 90)}
    for i, (label, st) in enumerate(seq):
        w = st.duration * beat_w
        col = func_col.get(st.function, (200, 200, 220))
        h0 = x
        # bass voice highlighted (shows the F → G half-step resolution)
        for j, v in enumerate(st.voices):
            y = H - 50 - (v - BOT_PC) * 9
            d.rectangle([h0, y - 4, h0 + w - 4, y + 4], fill=col, outline=(20, 22, 30))
            d.ellipse([h0 - 6, y - 6, h0 + 6, y + 6], fill=(240, 245, 255), outline=col)
        # mark the bass note separately in a warm color
        by = H - 50 - (st.bass_note - BOT_PC) * 9
        d.rectangle([h0, by - 5, h0 + w - 4, by + 5], fill=(255, 110, 130), outline=(20, 22, 30))
        d.text((h0 + 2, H - 50 - (TOP_PC - BOT_PC) * 9 - 20),
               f"{label}  (bass {PITCHNAME(st.bass_note)})", fill=(235, 240, 255), font=bf)
        x += w

    px = 160.0 + play_beat * beat_w
    d.line([(px, 10), (px, H - 45)], fill=(255, 90, 120), width=2)

    tf = _font(19)
    d.text((160, 12), "Neapolitan (♭II) — I → N⁶ → V → I   (watch the red bass resolve F → G)",
           fill=(255, 255, 255), font=tf)
    rf = _font(11)
    for b in range(0, int(total_beats) + 1):
        bx = 160 + b * beat_w
        d.line([(bx, H - 45), (bx, H - 40)], fill=(120, 130, 150), width=1)
        d.text((bx + 2, H - 38), str(b), fill=(120, 130, 150), font=rf)
    return img


def PITCHNAME(m: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[m % 12]}{m // 12 - 1}"


def render_video(seq, audio, frames_dir) -> str:
    import os
    os.makedirs(frames_dir, exist_ok=True)
    total_beats = sum(s.duration for _, s in seq)
    dur_sec = total_beats * 60.0 / BPM
    n_frames = int(dur_sec * FPS)
    wav_path = f"{OUTDIR}/neapolitan_demo.wav"
    import wave
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(audio.tobytes())
    for fi in range(n_frames):
        play_beat = (fi / n_frames) * total_beats
        draw_roll_frame(seq, play_beat).save(f"{frames_dir}/frame_{fi:05d}.png")
    mp4 = f"{OUTDIR}/neapolitan_demo.mp4"
    os.system(
        f"ffmpeg -y -r {FPS} -i {frames_dir}/frame_%05d.png -i {wav_path} "
        f"-c:v libx264 -pix_fmt yuv420p -c:a aac -shortest -movflags +faststart {mp4}"
    )
    return mp4


def main():
    import os, shutil
    seq = build_progression()
    print("Progression:")
    for label, st in seq:
        print(f"  {label:7} func={st.function:11} bass={PITCHNAME(st.bass_note)} voices={st.voices}")
    audio = synth_audio(seq)
    mp4 = render_video(seq, audio, f"{OUTDIR}/frames_n")
    print(f"Wrote: {OUTDIR}/neapolitan_demo.wav")
    print(f"Wrote: {mp4}")
    shutil.rmtree(f"{OUTDIR}/frames_n", ignore_errors=True)


if __name__ == "__main__":
    main()
