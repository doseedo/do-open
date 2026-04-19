/**
 * latent_drum_transcribe — drum stem latent → drum-roll MIDI.
 *
 * Pipeline:
 *   drum stem latent [64, T] @ 25 Hz
 *     → splitDrumLatent  (latentDrumSep ONNX)
 *     → 6 sub-stem latents [64, T]  (kick, snare, toms, hh, ride, crash)
 *     → per-frame L2 energy for each sub-stem
 *     → peak-pick with threshold + refractory
 *     → GM drum note events, merged
 *
 * L2 norm of the latent channels is already a clean amplitude envelope for
 * percussive transients — no separate envelope model needed. We only need
 * RELATIVE peaks per sub-stem to place onsets, so normalising by max per
 * sub-stem is enough.
 *
 * Output matches the shape utils/midiParser.js produces, so the DAW's
 * existing MIDI window renders it as a drum roll with no extra work.
 */

import { splitDrumLatent, getDrumStemNames } from './latentDrumSep';

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

// Onset-detection params — tuned so typical kit playing produces a
// roll the user recognises without a sea of ghost hits.
const THRESH_RATIO     = 0.18;  // peak must be >= this fraction of the sub-stem's max energy
const MIN_ABS_ENERGY   = 0.02;  // absolute floor — rejects near-silent sub-stems entirely
const REFRACTORY_FRAMES = 2;    // 80 ms between hits on the same sub-stem
const NOTE_DURATION_S  = 0.10;  // visual duration of each drum note in the piano roll
const VEL_MIN          = 50;
const VEL_MAX          = 120;

/**
 * @param {Float32Array} drumLatentCT  [64*T] channels-first drum stem latent
 * @param {number} T                   number of latent frames
 * @returns {Promise<{notes: Array<{note,time,duration,velocity}>, duration: number}>}
 */
export async function extractDrumMIDI(drumLatentCT, T) {
  if (!T || T < 2) return { notes: [], duration: 0 };

  const substems = await splitDrumLatent(drumLatentCT, T);
  const stemNames = getDrumStemNames();

  const notes = [];
  for (const name of stemNames) {
    const latent = substems[name];
    if (!latent) continue;
    const midiNote = DRUM_NOTE[name];
    if (midiNote == null) continue;

    // Per-frame L2 energy.
    const energy = new Float32Array(T);
    let maxE = 0;
    for (let t = 0; t < T; t++) {
      let s = 0;
      for (let d = 0; d < LATENT_CHANS; d++) {
        const v = latent[d * T + t];
        s += v * v;
      }
      const e = Math.sqrt(s);
      energy[t] = e;
      if (e > maxE) maxE = e;
    }
    if (maxE < MIN_ABS_ENERGY) continue;

    const thresh = Math.max(MIN_ABS_ENERGY, maxE * THRESH_RATIO);

    // Peak pick: frame is an onset when it's a strict local max AND above
    // threshold AND at least REFRACTORY_FRAMES past the previous onset on
    // this sub-stem.
    let lastOnset = -Infinity;
    for (let t = 1; t < T - 1; t++) {
      const e = energy[t];
      if (e < thresh) continue;
      if (e <= energy[t - 1] || e <= energy[t + 1]) continue;
      if (t - lastOnset < REFRACTORY_FRAMES) continue;
      lastOnset = t;

      const vNorm = Math.min(1, (e - thresh) / (maxE - thresh + 1e-9));
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
