// Chord Bot — Web Audio API scheduling
// Tracks all scheduled oscillators so stopAudio() can silence them immediately.
'use strict';

import { S } from './state.js';

let _nodes = [];         // { osc, gain } for active voices — cleared on stop
let _masterGain = null;  // master gain node for instant fade-out on stop
let _masterConnected = false;

export function ensureAudioCtx() {
  if (!S.audioCtx) {
    S.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    _masterGain = S.audioCtx.createGain();
    _masterGain.gain.value = 1;
    _masterGain.connect(S.audioCtx.destination);
    _masterConnected = true;
  }
  if (S.audioCtx.state === 'suspended') S.audioCtx.resume();
  return S.audioCtx;
}

/** Immediately silence all scheduled oscillators and discard references. */
export function stopAudio() {
  const now = S.audioCtx ? S.audioCtx.currentTime : 0;
  // Ramp master gain to 0 instantly then disconnect
  if (_masterGain) {
    try {
      _masterGain.gain.cancelScheduledValues(now);
      _masterGain.gain.setValueAtTime(0, now);
    } catch (_) { /* context may be closed */ }
  }
  // Stop all tracked oscillators
  for (const n of _nodes) {
    try {
      n.osc.cancelScheduledValues(now);
      // For already-started oscillators, we can't unschedule — but we can
      // rely on the master gain silence to mute everything.
    } catch (_) { /* ignore closed-context errors */ }
  }
  _nodes = [];
  if (S.audioSched) { S.audioSched.stopTime = 0; S.audioSched = null; }
}

/**
 * Schedule all notes in `seq` starting at `startBeat` offset.
 * Polyphonic triangle-wave chords + sine bass, tracked for stop support.
 */
export function scheduleSequence(seq, bpm, startBeat = 0) {
  const ctx = ensureAudioCtx();
  const beatSec = 60 / bpm;
  const now = ctx.currentTime;
  const g = 0.12;

  // Clear previous nodes
  _nodes = [];
  S.audioSched = { stopTime: now + 999 };

  // Ensure master gain is ramped back up in case a previous stop silenced it
  try {
    _masterGain.gain.cancelScheduledValues(now);
    _masterGain.gain.setValueAtTime(1, now);
  } catch (_) {}

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

      // Softer attack: 30ms ramp, gentle sustain, 30ms release
      const att = Math.max(0.005, Math.min(0.03, d * 0.04));
      const rel = Math.max(0.01, Math.min(0.03, d * 0.03));
      gn.gain.setValueAtTime(0, st);
      gn.gain.linearRampToValueAtTime(g * vel * 0.6, st + att);
      gn.gain.setValueAtTime(g * vel * 0.6, et - rel);
      gn.gain.linearRampToValueAtTime(0.0001, et);

      osc.connect(gn);
      gn.connect(_masterGain);

      osc.start(st);
      osc.stop(et + 0.05);

      _nodes.push({ osc, gain: gn });
    }

    // Bass note
    if (s.bass_note > 0 && s.bass_note <= 127) {
      const freq = 440 * Math.pow(2, (s.bass_note - 69) / 12);
      const osc  = ctx.createOscillator();
      const gn   = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;

      const att = Math.max(0.005, Math.min(0.03, d * 0.04));
      const rel = Math.max(0.01, Math.min(0.03, d * 0.03));
      gn.gain.setValueAtTime(0, st);
      gn.gain.linearRampToValueAtTime(g * vel * 1.2, st + att);
      gn.gain.setValueAtTime(g * vel * 1.2, et - rel);
      gn.gain.linearRampToValueAtTime(0.0001, et);

      osc.connect(gn);
      gn.connect(_masterGain);

      osc.start(st);
      osc.stop(et + 0.05);

      _nodes.push({ osc, gain: gn });
    }
  }
}
