/**
 * latent_pitch — stem latent → MIDI notes.
 *
 * Runs the LatentBasicPitchStudent ONNX (drop-in BasicPitch replacement)
 * on a [T, 64] VAE latent and produces a sorted array of
 * `{ note, time, duration, velocity }` events, compatible with the shape
 * produced by `utils/midiParser.js` so the DAW's existing MIDI window
 * renders it with no extra work.
 *
 * Model metadata (baked into pitch_054000.pt):
 *   • input:  `latent`  [B=1, T, 64]   time-major
 *   • outputs (all [B, T, 128]):
 *       - onset_logits    pre-sigmoid
 *       - frame_logits    pre-sigmoid
 *       - velocity        already sigmoid'd, [0,1]
 *       - onset_offset    already sigmoid'd, [0,1) sub-frame timing
 *   • max_len = 256 frames ≈ 10.24 s @ 25 Hz (positional encoding cap)
 *
 * We chunk into ≤CHUNK_FRAMES windows to stay inside that cap. The
 * transcription post-process mirrors `latent_pitch/infer.py`:
 *   onset NMS → extend while frame>thresh → min duration → velocity mean.
 */

import * as ort from 'onnxruntime-web';

// ?v=2 cache-busts the first-released ONNX, which was traced at T=200
// and so blew up the attention reshape at runtime. The bytes from R2
// are different but the length is identical, so the IndexedDB model
// cache wouldn't evict on size mismatch. Bumping the query string
// gives a fresh cache key.
const MODEL_URL     = '/static/models/latent_pitch.onnx?v=2';
const LATENT_CHANS  = 64;
const VAE_HZ        = 25;
const N_PITCH       = 128;
const CHUNK_FRAMES  = 256;  // ≤ max_len in the checkpoint's pos encoding
// Post-processing thresholds from a per-pitch bakeoff on single-note
// piano + bass soundfont latents (MIDI 36..96 / 28..60, tol ±160ms):
//
//                                       recall  precision  wrong-notes
//   0.7 baseline, no vel gate           94%        56%        73
//   0.5 flat vel 0.55                   89%        73%        31
//   0.3 split vel 0.60/0.40             95%        81%        21
//   0.7 vel 0.70/0.55 + oct-arb (ship)  70%       *96%*        3
//
// Precision-first tuning: "losing notes is better than wrong notes."
// The model outputs a lot of plausible-looking but false predictions
// (octave ghosts, harmonic fires) in the onset-prob 0.3-0.7 range.
// A high ONSET_THRESH + tight velocity floors + octave-arbitration
// pushes those out at the cost of quiet and extreme-range real notes.
// The user's MIDI window fills only with notes we're highly confident
// about; the DAW lets them add missing notes manually.
//
// Pitch-aware velocity floor is still in: the model's velocity output
// is systematically lower at pitch extremes (MIDI <40 / >85), so a
// single flat floor mis-filters real edge fires. Middle register gets
// the tight floor where ghosts live; edges relax so extreme real
// notes still pass.
//
// Octave arbitration applied AFTER velocity filtering, not before: the
// velocity gate already drops most ghost candidates, so arbitrating
// survivors is safe. Arbitrating RAW candidates was tested earlier and
// rejected — bass ghost onsets can outscore the fundamental and NMS
// deletes the true low note.
// Tuned on real audio against BasicPitch ground truth (the teacher the
// latent_pitch student was distilled from). Full grid sweep of
// (onset × frame_mean × vel_mid × vel_edge × octave_arb) across bass,
// other, and vocals stems of a real track. Previous 0.4/0.7/0.55/0.40
// was biased for precision on synthetic notes and missed most real
// notes in prod. New operating point prioritizes RECALL:
//
//   stem     pred   P    R    F1
//   bass     85    0.26 0.23 0.24   (model weak at low range regardless)
//   other    123   0.37 0.64 0.47
//   vocals   341   0.39 0.71 0.51
//
// Precision ~0.35-0.40 is the student's ceiling on VAE-real-audio
// latents (vs BasicPitch-on-audio ground truth at ±150 ms); retuning
// further strict loses real notes much faster than it gains precision.
const ONSET_THRESH        = 0.3;
const FRAME_THRESH        = 0.5;
const FRAME_MEAN_FLOOR    = 0.30;
const MIN_NOTE_FRAMES     = 2;
const NMS_RADIUS          = 2;
const VELOCITY_FLOOR_MID  = 0.35;  // pitches 40..85
const VELOCITY_FLOOR_EDGE = 0.25;  // pitches <40 or >85
const PITCH_EDGE_LO       = 40;
const PITCH_EDGE_HI       = 85;
const OCTAVE_ARB_RATIO    = 1.20;  // min onset-score ratio to suppress an
                                    // octave neighbor; <1.20 leaves both
                                    // (could be legit octave doubling)
const CHUNK_OVERLAP       = CHUNK_FRAMES / 2;  // 128-frame step → notes
                                                // straddling a 256-frame
                                                // chunk boundary fall into
                                                // two overlapping windows

let _session = null;
let _sessionPromise = null;

/** Load the model (IndexedDB-cached, then session). Safe to call many times. */
export async function initLatentPitch() {
  if (_session) return _session;
  if (_sessionPromise) return _sessionPromise;

  _sessionPromise = (async () => {
    if (ort.env?.wasm) {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
      ort.env.wasm.numThreads = Math.min(2, navigator.hardwareConcurrency || 1);
      ort.env.wasm.simd = true;
    }
    const t0 = performance.now();
    const { fetchModelWithCache } = await import('./modelCacheService');
    const bytes = await fetchModelWithCache(MODEL_URL);
    _session = await ort.InferenceSession.create(bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    const ms = (performance.now() - t0).toFixed(0);
    console.log(`[latentPitch] model loaded in ${ms}ms (${(bytes.byteLength / (1 << 20)).toFixed(1)} MB)`);
    return _session;
  })();

  return _sessionPromise;
}

/**
 * Batched pitch extraction: run all stems in a single ONNX call per chunk.
 *
 * Much faster than per-stem looping because on WASM the graph-launch and
 * wasm↔JS copy overhead dominates the actual kernel cost for a 6.5M-param
 * model. Empirically ~3× faster for 4 stems.
 *
 * @param {Float32Array[]} stemLatentsCT  array of length S; each is a
 *                                        length-LATENT_CHANS*T Float32Array
 *                                        in channels-first [64, T] layout
 * @param {number} T                      number of latent frames (shared)
 * @returns {Promise<Array<{notes, duration}>>}  one entry per stem, same order
 */
export async function extractPitchFromLatentsBatch(stemLatentsCT, T) {
  const S = stemLatentsCT.length;
  if (!S || !T || T < MIN_NOTE_FRAMES) {
    return Array.from({ length: S }, () => ({ notes: [], duration: 0 }));
  }
  const sess = await initLatentPitch();

  // Per-stem aggregation buffers.
  const agg = Array.from({ length: S }, () => ({
    onset:    new Float32Array(T * N_PITCH),
    frame:    new Float32Array(T * N_PITCH),
    velocity: new Float32Array(T * N_PITCH),
    offset:   new Float32Array(T * N_PITCH),
  }));

  // Overlapped chunking: step by CHUNK_OVERLAP so every frame is covered
  // by two graph windows (except the first/last OVERLAP/2 frames). When a
  // frame appears in two runs we take the MAX-onset prediction's full
  // tuple, so the winning onset's velocity and sub-frame offset stay
  // consistent with its own onset evidence. Frame-prob is max'd
  // independently — it measures sustain, not which run fired.
  for (let start = 0; start < T; start += CHUNK_OVERLAP) {
    const end = Math.min(T, start + CHUNK_FRAMES);
    const Tc = end - start;
    if (Tc <= 0) break;
    // Build [S, CHUNK_FRAMES, 64] — each stem's slice occupies its own
    // batch row. Tail chunks shorter than CHUNK_FRAMES are zero-padded
    // (the ONNX graph was traced at T=256; the attention reshape trips
    // on any other T). Output frames beyond Tc are discarded.
    const batched = new Float32Array(S * CHUNK_FRAMES * LATENT_CHANS);
    for (let s = 0; s < S; s++) {
      const latentCT = stemLatentsCT[s];
      const rowOff = s * CHUNK_FRAMES * LATENT_CHANS;
      for (let t = 0; t < Tc; t++) {
        const gt = start + t;
        const tOff = rowOff + t * LATENT_CHANS;
        for (let d = 0; d < LATENT_CHANS; d++) {
          batched[tOff + d] = latentCT[d * T + gt];
        }
      }
    }
    const input = new ort.Tensor('float32', batched, [S, CHUNK_FRAMES, LATENT_CHANS]);
    // Serialize against every other WebGPU ORT session — ORT's WebGPU EP
    // shares one GPUDevice and throws "Session mismatch" on overlapping runs.
    const { ortWebGPURun } = await import('./webgpuOrtQueue');
    const out = await ortWebGPURun(() => sess.run({ latent: input }));

    const onLog = out.onset_logits.data;     // [S, CHUNK_FRAMES, 128]
    const frLog = out.frame_logits.data;
    const vel   = out.velocity.data;
    const off   = out.onset_offset.data;
    const strideS = CHUNK_FRAMES * N_PITCH;
    for (let s = 0; s < S; s++) {
      const srcBase = s * strideS;
      const a = agg[s];
      for (let t = 0; t < Tc; t++) {
        const dst = (start + t) * N_PITCH;
        const src = srcBase + t * N_PITCH;
        for (let p = 0; p < N_PITCH; p++) {
          const newOnset = _sigmoid(onLog[src + p]);
          if (newOnset > a.onset[dst + p]) {
            a.onset[dst + p]    = newOnset;
            a.velocity[dst + p] = vel[src + p];
            a.offset[dst + p]   = off[src + p];
          }
          const newFrame = _sigmoid(frLog[src + p]);
          if (newFrame > a.frame[dst + p]) {
            a.frame[dst + p] = newFrame;
          }
        }
      }
    }
  }

  return agg.map((a) => _postprocess(a, T));
}

/**
 * Thin single-stem wrapper around the batched entrypoint. Kept for any
 * caller that has one stem at a time.
 */
export async function extractPitchFromLatent(latentCT, T) {
  const [out] = await extractPitchFromLatentsBatch([latentCT], T);
  return out;
}

/**
 * Posterior arrays → note events. Three-pass: (1) enumerate onset-peak
 * candidates, (2) velocity-filter with pitch-aware floor, (3) octave-
 * arbitrate surviving candidates, then emit the notes.
 */
function _postprocess({ onset, frame, velocity, offset }, T) {
  const dt = 1.0 / VAE_HZ;

  // Pass 1: candidate enumeration (onset NMS per pitch, build note object).
  const cands = [];   // { t, pitch, onsetScore, startT, endT, vMean, subFrame }
  for (let pitch = 0; pitch < N_PITCH; pitch++) {
    for (let t = 0; t < T; t++) {
      const on = onset[t * N_PITCH + pitch];
      if (on <= ONSET_THRESH) continue;
      let peak = true;
      const lo = Math.max(0, t - NMS_RADIUS);
      const hi = Math.min(T, t + NMS_RADIUS + 1);
      for (let k = lo; k < hi; k++) {
        if (k === t) continue;
        if (onset[k * N_PITCH + pitch] > on + 1e-9) { peak = false; break; }
      }
      if (!peak) continue;

      let endFrame = t + 1;
      while (endFrame < T && frame[endFrame * N_PITCH + pitch] > FRAME_THRESH) {
        endFrame++;
      }
      const nFrames = endFrame - t;
      if (nFrames < MIN_NOTE_FRAMES) continue;
      let vSum = 0;
      let fSum = 0;
      for (let k = t; k < endFrame; k++) {
        vSum += velocity[k * N_PITCH + pitch];
        fSum += frame[k * N_PITCH + pitch];
      }
      const vMean = vSum / nFrames;
      const fMean = fSum / nFrames;

      // Pass 2a: frame-mean sustain check. Real notes hold frame-prob near
      // 1.0 for their entire duration; ghost fires drop the frame-prob
      // immediately and never sustain. Cheapest ghost filter we have.
      if (fMean < FRAME_MEAN_FLOOR) continue;

      // Pass 2b: pitch-aware velocity floor.
      const floor = (pitch < PITCH_EDGE_LO || pitch > PITCH_EDGE_HI)
        ? VELOCITY_FLOOR_EDGE
        : VELOCITY_FLOOR_MID;
      if (vMean < floor) continue;

      cands.push({
        t, pitch, onsetScore: on,
        startT: t, endT: endFrame, vMean,
        subFrame: offset[t * N_PITCH + pitch],
      });
    }
  }

  // Pass 3: octave arbitration. When two candidates fire within ±1 frame
  // of each other at an exact octave apart, the one with meaningfully
  // lower onset score (< 1/OCTAVE_ARB_RATIO of the winner) is a ghost.
  // Ties within the ratio stay — could be legit octave doubling.
  const suppressed = new Uint8Array(cands.length);
  for (let i = 0; i < cands.length; i++) {
    if (suppressed[i]) continue;
    const a = cands[i];
    for (let j = 0; j < cands.length; j++) {
      if (i === j || suppressed[j]) continue;
      const b = cands[j];
      if (Math.abs(a.t - b.t) > 1) continue;
      const dp = b.pitch - a.pitch;
      if (dp === 0 || dp % 12 !== 0) continue;
      const hi = Math.max(a.onsetScore, b.onsetScore);
      const lo = Math.min(a.onsetScore, b.onsetScore);
      if (hi / Math.max(1e-9, lo) < OCTAVE_ARB_RATIO) continue;
      suppressed[a.onsetScore >= b.onsetScore ? j : i] = 1;
    }
  }

  const notes = [];
  for (let i = 0; i < cands.length; i++) {
    if (suppressed[i]) continue;
    const c = cands[i];
    notes.push({
      note: c.pitch,
      time: (c.t + c.subFrame) * dt,
      duration: (c.endT - c.startT) * dt,
      velocity: Math.max(1, Math.min(127, Math.round(c.vMean * 127))),
    });
  }
  notes.sort((a, b) => a.time - b.time);
  return { notes, duration: T * dt };
}

function _sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
