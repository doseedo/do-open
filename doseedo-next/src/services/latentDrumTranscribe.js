/**
 * latent_drum_transcribe — drum stem latent → drum-roll MIDI.
 *
 * Pipeline:
 *   drum stem latent [64, T] @ 25 Hz  (channels-first, as produced by sem4Decoder)
 *     → transpose to [T, 64] (time-major, layout splitDrumLatent expects)
 *     → splitDrumLatent  (latent_drumsep ONNX)
 *     → 6 sub-stem latents [T, 64]  (kick, snare, toms, hh, ride, crash)
 *     → latent_visual ONNX per sub-stem → [2, T] peak envelope
 *     → range = max − min per frame (waveform-space amplitude envelope)
 *     → local-max peak-pick with threshold + refractory
 *     → GM drum note events, merged
 *
 * latent_visual is a trained 62 K-param 1D conv net that maps a VAE latent
 * straight to the audio-waveform peak envelope. Validated on synthesized
 * drum latents (tests/latent_visual_onset/run_test.py): isolated hits →
 * F1=1.00, 0-frame error; 120 BPM rock pattern → F1≈0.78; dense 16th-note
 * hats → F1≈0.92. Raw latent L2 energy, by contrast, scores F1≈0.2–0.3
 * because silence latents have a standing L2 norm of ~7 — every frame fires.
 *
 * Output matches the shape utils/midiParser.js produces, so the DAW's
 * existing MIDI window renders it as a drum roll with no extra work.
 */

import { splitDrumLatent, getDrumStemNames } from './latentDrumSep';
import { initLatentVisual, envelopeFromLatent } from './latentVisual';

const LATENT_CHANS = 64;
const VAE_HZ       = 25;
const DT           = 1.0 / VAE_HZ;  // 40 ms

// General MIDI drum note numbers (channel 10 convention).
const DRUM_NOTE = {
  kick:  36,
  snare: 38,
  hh:    42,   // closed hi-hat
  toms:  45,   // low tom
  ride:  51,
  crash: 49,
};

// Onset-detection params — tuned on the synthetic-drum test harness.
// thresh=0.3 on the range envelope hit F1≥0.78 on every scenario tested.
const THRESH_RATIO      = 0.30;  // peak ≥ this fraction of the sub-stem's max envelope range
const MIN_ABS_ENVELOPE  = 0.01;  // absolute floor — rejects near-silent sub-stems
const REFRACTORY_FRAMES = 2;     // 80 ms between hits on the same sub-stem
const NOTE_DURATION_S   = 0.10;  // visual duration of each drum note in the piano roll
const VEL_MIN           = 50;
const VEL_MAX           = 120;

/** Channels-first [64*T] → time-major [T*64]. */
function ctToTD(latentCT, T) {
  const out = new Float32Array(T * LATENT_CHANS);
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < LATENT_CHANS; d++) {
      out[t * LATENT_CHANS + d] = latentCT[d * T + t];
    }
  }
  return out;
}

/**
 * @param {Float32Array} drumLatentCT  [64*T] channels-first drum stem latent
 *                                     (sem4Decoder/distill_demucs output layout)
 * @param {number} T                   number of latent frames
 * @returns {Promise<{notes: Array<{note,time,duration,velocity}>, duration: number}>}
 */
export async function extractDrumMIDI(drumLatentCT, T) {
  if (!T || T < 2) return { notes: [], duration: 0 };

  await initLatentVisual();

  // splitDrumLatent wraps its input as [1, T, 64] → needs time-major order.
  const drumLatentTD = ctToTD(drumLatentCT, T);
  const substems = await splitDrumLatent(drumLatentTD, T);
  const stemNames = getDrumStemNames();

  const notes = [];
  for (const name of stemNames) {
    const subLatentTD = substems[name];   // time-major [T*64]
    if (!subLatentTD) continue;
    const midiNote = DRUM_NOTE[name];
    if (midiNote == null) continue;

    // latent_visual envelope: [2*T] = T mins followed by T maxes.
    const env = await envelopeFromLatent(subLatentTD, T);
    const mins = env.subarray(0, T);
    const maxs = env.subarray(T, 2 * T);

    // Peak range per frame = waveform amplitude. Transients light up,
    // silence collapses to ~0.
    const range = new Float32Array(T);
    let maxR = 0;
    for (let t = 0; t < T; t++) {
      const r = maxs[t] - mins[t];
      range[t] = r;
      if (r > maxR) maxR = r;
    }
    if (maxR < MIN_ABS_ENVELOPE) continue;

    const thresh = Math.max(MIN_ABS_ENVELOPE, maxR * THRESH_RATIO);

    // Peak-pick: strict local max above threshold, respecting refractory.
    let lastOnset = -Infinity;
    for (let t = 1; t < T - 1; t++) {
      const r = range[t];
      if (r < thresh) continue;
      if (r <= range[t - 1] || r <= range[t + 1]) continue;
      if (t - lastOnset < REFRACTORY_FRAMES) continue;
      lastOnset = t;

      const vNorm = Math.min(1, (r - thresh) / (maxR - thresh + 1e-9));
      const velocity = Math.round(VEL_MIN + vNorm * (VEL_MAX - VEL_MIN));
      notes.push({
        note: midiNote,
        time: t * DT,
        duration: NOTE_DURATION_S,
        velocity,
      });
    }
  }

  notes.sort((a, b) => a.time - b.time);
  return { notes, duration: T * DT };
}
