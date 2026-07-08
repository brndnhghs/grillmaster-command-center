"""Planing Chord — visual + audio demo renderer.

Builds a whole-tone planing passage by repeatedly applying the new `planing`
node, then renders:
  - demo/planing_demo.wav  (synthesized audio of the planed voices)
  - demo/planing_demo.mp4  (scrolling piano-roll with audio muxed)

The scrolling roll makes the *parallel* (planing) motion visible: every voice
shifts by the same interval, so the chord blocks climb as a rigid diagonal.
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chord_bot.chord_types import HarmonicState, compute_voices, compute_bass, note_to_pc
from chord_bot.nodes.planing import node_planing

# ── Config ────────────────────────────────────────────────────────────────────
BPM = 88
PX_PER_BEAT = 96
TOP_PC = note_to_pc("C") + 12 * 5   # top of roll ≈ C6 (climb peaks at MIDI 83)
BOT_PC = note_to_pc("C") + 12 * 2   # bottom ≈ C3 (MIDI 48)
PITCH_SPAN = TOP_PC - BOT_PC
ROLL_H = PITCH_SPAN * 9 + 60         # 9 px per semitone
FPS = 30
SR = 44100

OUTDIR = "/Users/admin/Documents/GitHub/grillmaster-command-center/chord_bot/demo"


def midi_to_freq(m: int) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def make_state(root: str, quality: str, num: str = "") -> HarmonicState:
    rpc = note_to_pc(root)
    return HarmonicState(
        key=root, mode="major", chord=f"{root}{quality}", root=root,
        quality=quality, voices=compute_voices(rpc, quality),
        bass_note=compute_bass(rpc, 0, quality), tension=0.2,
        duration=2.0, velocity=85, numeral=num, degree=0,
    )


def build_progression() -> list[tuple[str, HarmonicState]]:
    """Pure whole-tone planing: Cmaj7, then climb by whole steps (interval=2)."""
    seq: list[tuple[str, HarmonicState]] = []
    st = make_state("C", "maj7", "IM7")
    seq.append((st.chord, st))
    for _ in range(6):
        st = node_planing(
            st.copy(),
            {"direction": "up", "interval": 2, "stack": "keep",
             "octave": 4, "velocity": 85},
        )
        seq.append((st.chord, st))
    return seq


def synth_audio(seq: list[tuple[str, HarmonicState]]) -> np.ndarray:
    """Synthesize planed voices (triangle-ish) + bass, with a soft ADSR."""
    beats = [s.duration for _, s in seq]
    total_beats = sum(beats)
    total_samples = int(total_beats * 60.0 / BPM * SR)
    out = np.zeros(total_samples, dtype=np.float64)
    t0 = 0  # integer sample index
    for (label, st), dur in zip(seq, beats):
        seg = int(dur * 60.0 / BPM * SR)
        if seg <= 0:
            continue
        t = np.linspace(0.0, dur * 60.0 / BPM, seg, endpoint=False)
        # voices
        sig = np.zeros(seg)
        for v in st.voices:
            f = midi_to_freq(v)
            # fundamental + 2nd/3rd partials for a soft electric-piano timbre
            osc = (np.sin(2 * math.pi * f * t)
                   + 0.35 * np.sin(2 * math.pi * 2 * f * t)
                   + 0.18 * np.sin(2 * math.pi * 3 * f * t))
            sig += osc
        sig /= max(1.0, len(st.voices))
        # bass an octave below root
        bf = midi_to_freq(st.bass_note)
        sig += 0.5 * (np.sin(2 * math.pi * bf * t)
                      + 0.3 * np.sin(2 * math.pi * 2 * bf * t))
        # ADSR
        a = int(0.02 * SR); d = int(0.06 * SR); r = int(0.18 * SR)
        env = np.ones(seg)
        env[:a] = np.linspace(0, 1, a) if a else 1
        decay = slice(a, a + d)
        env[decay] = np.linspace(1, 0.75, max(1, d))
        env[max(0, seg - r):] = np.linspace(0.75, 0.0, max(1, r))
        sig *= env
        out[t0:t0 + seg] += sig * 0.22
        t0 += seg
    out = np.clip(out, -1.0, 1.0)
    # fade in/out
    fade = int(0.05 * SR)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    return (out * 32767).astype(np.int16)


def _font(size: int):
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]:
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

    # ── background pitch grid (every semitone) ──
    for pc in range(BOT_PC, TOP_PC + 1):
        y = H - 50 - (pc - BOT_PC) * 9
        is_c = (pc % 12) == 0
        line_col = (54, 58, 70) if is_c else (30, 33, 44)
        d.line([(150, y), (W, y)], fill=line_col, width=1)
    # octave labels
    f = _font(13)
    for pc in range(BOT_PC, TOP_PC + 1):
        if pc % 12 == 0:
            y = H - 50 - (pc - BOT_PC) * 9
            name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][pc % 12]
            d.text((8, y - 7), f"{name}{pc // 12 - 1}", fill=(150, 160, 180), font=f)

    # ── chord blocks (parallel planing) ──
    x = 160.0
    bf = _font(15)
    for i, (label, st) in enumerate(seq):
        w = st.duration * beat_w
        # color gradient along the climb (cool → warm) to show progression
        hue = (i / max(1, len(seq) - 1))
        col = (int(80 + 150 * hue), int(180 - 80 * hue), int(230 - 120 * hue))
        h0 = x
        for v in st.voices:
            y = H - 50 - (v - BOT_PC) * 9
            # thick note bar + round note head so each voice is clearly visible
            d.rectangle([h0, y - 4, h0 + w - 4, y + 4], fill=col, outline=(20, 22, 30))
            d.ellipse([h0 - 6, y - 6, h0 + 6, y + 6], fill=(240, 245, 255), outline=col)
        # chord label above first chord
        d.text((h0 + 2, H - 50 - (TOP_PC - BOT_PC) * 9 - 18),
               label, fill=(235, 240, 255), font=bf)
        x += w

    # ── playhead ──
    px = 160.0 + play_beat * beat_w
    d.line([(px, 10), (px, H - 45)], fill=(255, 90, 120), width=2)

    # ── title + beat ruler ──
    tf = _font(20)
    d.text((160, 12), "Planing Chord — whole-tone parallel motion (Cmaj7 → climb by whole steps)",
           fill=(255, 255, 255), font=tf)
    rf = _font(11)
    for b in range(0, int(total_beats) + 1):
        bx = 160 + b * beat_w
        d.line([(bx, H - 45), (bx, H - 40)], fill=(120, 130, 150), width=1)
        d.text((bx + 2, H - 38), str(b), fill=(120, 130, 150), font=rf)

    return img


def render_video(seq, audio: np.ndarray, frames_dir: str) -> str:
    import os
    os.makedirs(frames_dir, exist_ok=True)
    total_beats = sum(s.duration for _, s in seq)
    dur_sec = total_beats * 60.0 / BPM
    n_frames = int(dur_sec * FPS)
    wav_path = f"{OUTDIR}/planing_demo.wav"
    # write wav
    import wave
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(audio.tobytes())

    for fi in range(n_frames):
        play_beat = (fi / n_frames) * total_beats
        img = draw_roll_frame(seq, play_beat)
        img.save(f"{frames_dir}/frame_{fi:05d}.png")

    mp4 = f"{OUTDIR}/planing_demo.mp4"
    cmd = (
        f"ffmpeg -y -r {FPS} -i {frames_dir}/frame_%05d.png "
        f"-i {wav_path} -c:v libx264 -pix_fmt yuv420p -c:a aac "
        f"-shortest -movflags +faststart {mp4}"
    )
    os.system(cmd)
    return mp4


def main():
    import os
    seq = build_progression()
    print("Progression (chord : root voices):")
    for label, st in seq:
        print(f"  {label:8} -> voices {st.voices}")
    audio = synth_audio(seq)
    frames_dir = f"{OUTDIR}/frames"
    mp4 = render_video(seq, audio, frames_dir)
    print(f"\nWrote: {OUTDIR}/planing_demo.wav")
    print(f"Wrote: {mp4}")
    # cleanup frames
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
