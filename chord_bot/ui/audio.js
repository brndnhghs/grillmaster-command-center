// Chord Bot — Web Audio API scheduling
'use strict';

import { S } from './state.js';

export function ensureAudioCtx() {
  if (!S.audioCtx) S.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (S.audioCtx.state === 'suspended') S.audioCtx.resume();
  return S.audioCtx;
}

export function stopAudio() {
  if (S.audioSched) { S.audioSched.stopTime = 0; S.audioSched = null; }
}

/**
 * Schedule all notes in `seq` starting at `startBeat` offset.
 * Polyphonic triangle-wave chords + sine bass.
 */
export function scheduleSequence(seq, bpm, startBeat = 0) {
  const ctx = ensureAudioCtx();
  const beatSec = 60 / bpm;
  const now = ctx.currentTime;
  const g = 0.12;

  stopAudio();
  S.audioSched = { stopTime: now + 999 };

  for (const en of seq) {
    if (en.end_beat <= startBeat) continue;
    const ls = Math.max(en.start_beat, startBeat);
    const st = now + (ls - startBeat) * beatSec;
    const et = now + (en.end_beat - startBeat) * beatSec;
    const d  = et - st;
    if (d <= 0) continue;

    const s   = en.state;
    const vel = Math.max(0.05, (s.velocity || 80) / 127);

    // Chord voices
    for (const midi of (s.voices || [])) {
      if (midi < 24 || midi > 108) continue;
      const freq = 440 * Math.pow(2, (midi - 69) / 12);
      const osc  = ctx.createOscillator();
      const gn   = ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.value = freq;
      gn.gain.setValueAtTime(0, st);
      gn.gain.linearRampToValueAtTime(g * vel * 0.6, st + 0.01);
      gn.gain.setValueAtTime(g * vel * 0.6, et - 0.02);
      gn.gain.linearRampToValueAtTime(0, et);
      osc.connect(gn); gn.connect(ctx.destination);
      osc.start(st); osc.stop(et + 0.05);
    }

    // Bass note
    if (s.bass_note > 0 && s.bass_note <= 127) {
      const freq = 440 * Math.pow(2, (s.bass_note - 69) / 12);
      const osc  = ctx.createOscillator();
      const gn   = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;
      gn.gain.setValueAtTime(0, st);
      gn.gain.linearRampToValueAtTime(g * vel * 1.2, st + 0.01);
      gn.gain.setValueAtTime(g * vel * 1.2, et - 0.02);
      gn.gain.linearRampToValueAtTime(0, et);
      osc.connect(gn); gn.connect(ctx.destination);
      osc.start(st); osc.stop(et + 0.05);
    }
  }
}
