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

// Onset-detection params.
//
// Bello-style onset detection on the amplitude envelope produced by
// latent_visual: onset fn = half-wave rectified difference against a
// trailing baseline mean (env[t] − mean(env[t-LB..t-1]), clamped ≥0),
// peak-picked against a LOCAL robust threshold (median + K · 1.4826·MAD
// over a ±LOCAL_WIN window). Three wins over the previous global-max
// peak-pick:
//   1. Peaks align to attacks, not the middle of sustain plateaus.
//   2. One loud hit on the timeline no longer raises the detection
//      threshold for every softer hit — quiet passages stay sensitive.
//   3. The trailing baseline includes the recent hit's ringing, which
//      naturally suppresses false "attacks on sustain" for cymbals
//      (ride/crash). A pure 1-frame diff fires constantly on cymbal
//      decay wobble — F1 0.4 before, 0.95 after on ride quarter+eighth.
// Median/MAD are robust to the peaks themselves; mean/std get dragged up
// by dense peak trains and end up suppressing every peak in the cluster.
// Params from a per-sub-stem hyperparameter sweep across all 6 GM drums
// (kick/snare/hh/toms/ride/crash) at three tempos (half / quarter /
// eighth-notes at 120 BPM); picked to maximize MIN F1 over realistic
// quarter+eighth patterns — mean F1 = 0.95, min F1 = 0.86 (ride quarters).
const LOOKBACK_FRAMES    = 5;     // trailing window for the onset-fn baseline
const LOCAL_WIN_FRAMES   = 12;    // ±12 ≈ 1 s window for the adaptive threshold
const ADAPTIVE_K         = 1.0;   // peak must sit ≥ localMedian + K · 1.4826·localMAD
const PEAK_NMS_HALF      = 1;     // strict local max over ±1 frame on the onset fn
const MIN_ABS_ONSET      = 0.003; // absolute floor — rejects near-silent sub-stems entirely
const REFRACTORY_FRAMES  = 2;     // 80 ms between hits on the same sub-stem
// Stem activity gate: a sub-stem whose peak onset is < this fraction of
// the loudest sub-stem's peak onset is considered "not actually played
// in this track" — all its candidates get suppressed. Cleanly kills
// toms/ride/crash bleed firings on pop/rock tracks that don't play
// those drums.
const STEM_ACTIVITY_MIN  = 0.15;
// Normalized cross-stem competition. After per-stem peak-picking, for each
// candidate in stem X at frame t we compute its "significance" = onset[X][t]
// / max_onset[X]  (0..1, how much of X's loudest hit is this). If any other
// active stem has a MORE significant peak within ±WIN frames, X's firing
// is bleed-from-Y rather than a real hit on X — suppress it. Ties (equal
// significance) survive, so real simultaneous hits pass.
//   Without this: kick sub-stem's peak of a bled-in snare (raw onset 0.5)
//   looks "moderate for kick" (0.5 / 1.05 = 0.48) but the snare sub-stem's
//   same-frame real snare is "near-peak for snare" (1.5 / 1.56 = 0.96).
//   Normalized comparison keeps snare and drops kick. Raw-magnitude compari-
//   son failed because kick and snare have very different stem-level max
//   onsets (kick 1.05, snare 1.56, hh 0.30).
const CROSS_STEM_RATIO   = 1.0;   // other stem must be ≥ ratio × my significance
const CROSS_STEM_WIN     = 1;     // ±1 frame search for a louder sibling
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

  // --- Pass 1: per-stem onset functions + global-max for activity gate ---
  const onsetPerStem = {};
  const maxPerStem = {};
  let globalMaxOnset = 0;
  for (const name of stemNames) {
    const subLatentTD = substems[name];
    if (!subLatentTD || DRUM_NOTE[name] == null) continue;
    const env = await envelopeFromLatent(subLatentTD, T);
    const mins = env.subarray(0, T);
    const maxs = env.subarray(T, 2 * T);
    const range = new Float32Array(T);
    for (let t = 0; t < T; t++) range[t] = maxs[t] - mins[t];

    // Onset function: half-wave rectified difference against a trailing
    // baseline mean. Peaks correspond to attacks, not to the middle of each
    // hit's decay plateau. Trailing-baseline (as opposed to 1-frame diff)
    // is critical for cymbals: the baseline eats the previous hit's ringing
    // so we don't re-fire on decay wobble.
    const onsetFn = new Float32Array(T);
    let maxOnset = 0;
    for (let t = 1; t < T; t++) {
      const lo = Math.max(0, t - LOOKBACK_FRAMES);
      let sum = 0;
      for (let k = lo; k < t; k++) sum += range[k];
      const baseline = sum / (t - lo);
      const d = range[t] - baseline;
      const v = d > 0 ? d : 0;
      onsetFn[t] = v;
      if (v > maxOnset) maxOnset = v;
    }
    onsetPerStem[name] = onsetFn;
    maxPerStem[name] = maxOnset;
    if (maxOnset > globalMaxOnset) globalMaxOnset = maxOnset;
  }
  if (globalMaxOnset < MIN_ABS_ONSET) return { notes: [], duration: T * DT };

  // --- Pass 2: stem-activity gate + peak-pick → candidates per active stem ---
  // Candidate layout: { stem, t, onsetValue, significance, midiNote }
  const candidates = [];
  const scratch = new Float32Array(2 * LOCAL_WIN_FRAMES + 1);
  for (const name of stemNames) {
    const onsetFn = onsetPerStem[name];
    if (!onsetFn) continue;
    const maxOnset = maxPerStem[name];

    // Activity gate — cheap kill for sub-stems not actually played.
    if (maxOnset / globalMaxOnset < STEM_ACTIVITY_MIN) continue;
    if (maxOnset < MIN_ABS_ONSET) continue;

    const midiNote = DRUM_NOTE[name];
    let lastOnset = -Infinity;
    for (let t = 1; t < T - 1; t++) {
      const o = onsetFn[t];
      if (o < MIN_ABS_ONSET) continue;

      // Local median + MAD adaptive threshold.
      const wlo = Math.max(0, t - LOCAL_WIN_FRAMES);
      const whi = Math.min(T, t + LOCAL_WIN_FRAMES + 1);
      const n = whi - wlo;
      for (let i = 0; i < n; i++) scratch[i] = onsetFn[wlo + i];
      const win = scratch.subarray(0, n);
      win.sort();
      const median = n & 1 ? win[(n - 1) >> 1] : 0.5 * (win[n / 2 - 1] + win[n / 2]);
      for (let i = 0; i < n; i++) scratch[i] = Math.abs(onsetFn[wlo + i] - median);
      const devs = scratch.subarray(0, n);
      devs.sort();
      const mad = n & 1 ? devs[(n - 1) >> 1] : 0.5 * (devs[n / 2 - 1] + devs[n / 2]);
      const adaptiveThresh = median + ADAPTIVE_K * 1.4826 * mad + MIN_ABS_ONSET;
      if (o < adaptiveThresh) continue;

      // Strict local max over ±PEAK_NMS_HALF.
      const nlo = Math.max(0, t - PEAK_NMS_HALF);
      const nhi = Math.min(T, t + PEAK_NMS_HALF + 1);
      let isPeak = true;
      for (let k = nlo; k < nhi; k++) {
        if (k === t) continue;
        if (onsetFn[k] > o) { isPeak = false; break; }
      }
      if (!isPeak) continue;

      if (t - lastOnset < REFRACTORY_FRAMES) continue;
      lastOnset = t;
      candidates.push({
        stem: name, t, onsetValue: o,
        significance: o / maxOnset,
        midiNote,
      });
    }
  }

  // --- Pass 3: normalized cross-stem competition ---
  // For each candidate, compute the max normalized onset across OTHER active
  // stems in frames [t-WIN, t+WIN]. If that competitor's normalized value
  // exceeds ours by the ratio, we're picking up bleed from it. Suppress.
  const keep = new Uint8Array(candidates.length);
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    let maxOtherNorm = 0;
    const lo = Math.max(0, c.t - CROSS_STEM_WIN);
    const hi = Math.min(T - 1, c.t + CROSS_STEM_WIN);
    for (const otherName of stemNames) {
      if (otherName === c.stem) continue;
      const otherFn = onsetPerStem[otherName];
      if (!otherFn) continue;
      const otherMax = maxPerStem[otherName];
      if (!otherMax || otherMax / globalMaxOnset < STEM_ACTIVITY_MIN) continue;
      let peak = 0;
      for (let t = lo; t <= hi; t++) {
        if (otherFn[t] > peak) peak = otherFn[t];
      }
      const norm = peak / otherMax;
      if (norm > maxOtherNorm) maxOtherNorm = norm;
    }
    // Suppress if another active stem is more significant at this moment.
    if (maxOtherNorm > CROSS_STEM_RATIO * c.significance) continue;
    keep[i] = 1;
  }

  // --- Emit surviving candidates as notes ---
  const notes = [];
  for (let i = 0; i < candidates.length; i++) {
    if (!keep[i]) continue;
    const c = candidates[i];
    const velocity = Math.round(VEL_MIN + c.significance * (VEL_MAX - VEL_MIN));
    notes.push({
      note: c.midiNote,
      time: c.t * DT,
      duration: NOTE_DURATION_S,
      velocity,
    });
  }
  notes.sort((a, b) => a.time - b.time);
  return { notes, duration: T * DT };
}
