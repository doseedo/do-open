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
// latent_visual: onset fn = half-wave rectified first difference
// (env[t] − env[t−1], clamped ≥0), peak-picked against a LOCAL robust
// threshold (median + K · 1.4826·MAD over a ±LOCAL_WIN window). Two wins
// over the previous global-max peak-pick:
//   1. Peaks align to attacks, not the middle of sustain plateaus.
//   2. One loud hit on the timeline no longer raises the detection
//      threshold for every softer hit — quiet passages stay sensitive.
// Median/MAD are robust to the peaks themselves; mean/std get dragged up
// by dense peak trains and end up suppressing every peak in the cluster.
// Params chosen by hyperparameter sweep on three synthetic scenarios
// (isolated kicks / rock kick+snare / 16th-note hats); min-F1 across all
// three peaks at w=12, k=0.8, nms=1.
const LOCAL_WIN_FRAMES  = 12;    // ±12 ≈ 1 s window for the adaptive threshold
const ADAPTIVE_K        = 0.8;   // peak must sit ≥ localMedian + K · 1.4826·localMAD
const PEAK_NMS_HALF     = 1;     // strict local max over ±1 frame on the onset fn
const MIN_ABS_ONSET     = 0.002; // absolute floor — rejects near-silent sub-stems entirely
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

    // Range = max − min per frame is the waveform peak-to-peak amplitude.
    // Transients light up, silence collapses to ~0.
    const range = new Float32Array(T);
    for (let t = 0; t < T; t++) range[t] = maxs[t] - mins[t];

    // Onset function: half-wave rectified first difference. Peaks
    // correspond to attacks, not to the middle of each hit's decay plateau.
    const onsetFn = new Float32Array(T);
    let maxOnset = 0;
    for (let t = 1; t < T; t++) {
      const d = range[t] - range[t - 1];
      const v = d > 0 ? d : 0;
      onsetFn[t] = v;
      if (v > maxOnset) maxOnset = v;
    }
    if (maxOnset < MIN_ABS_ONSET) continue;

    // Peak-pick the onset function with a LOCAL robust threshold.
    // For each candidate frame t:
    //   * compute localMedian and localMAD of onsetFn over ±LOCAL_WIN_FRAMES
    //   * require onsetFn[t] ≥ localMedian + K · 1.4826·localMAD
    //   * require onsetFn[t] to be a strict local max over ±PEAK_NMS_HALF
    // Threshold adapts to track dynamics; median+MAD stay stable even when
    // the window is dominated by peak clusters.
    let lastOnset = -Infinity;
    const scratch = new Float32Array(2 * LOCAL_WIN_FRAMES + 1);
    for (let t = 1; t < T - 1; t++) {
      const o = onsetFn[t];
      if (o < MIN_ABS_ONSET) continue;

      // Collect window → sort → median.
      const wlo = Math.max(0, t - LOCAL_WIN_FRAMES);
      const whi = Math.min(T, t + LOCAL_WIN_FRAMES + 1);
      const n = whi - wlo;
      for (let i = 0; i < n; i++) scratch[i] = onsetFn[wlo + i];
      const win = scratch.subarray(0, n);
      win.sort();
      const median = n & 1 ? win[(n - 1) >> 1] : 0.5 * (win[n / 2 - 1] + win[n / 2]);
      // MAD = median(|x - median|). Reuse scratch.
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

      const vNorm = Math.min(1, o / (maxOnset + 1e-9));
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
