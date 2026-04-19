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
//              exact     ghosts   comment
//   0.7        54+30       73     baseline (no vel gate)
//   0.5 v.55   53+31       31     onset 0.5 + flat velocity floor 0.55
//   0.3 split  56+33       21     + pitch-aware vel floor (ship)
//
// The winning config uses a PITCH-AWARE velocity floor: the model's
// raw velocity output is systematically lower for extreme pitches
// (MIDI <40 and >85) — low-bass / high-piano exact fires sit at vel
// 0.40-0.55 while middle-range ghosts sit at 0.50-0.60. A single flat
// floor can't separate them. Two floors (edge=0.40, middle=0.60) plus
// a lower onset threshold (0.3) recover 5 real notes AND cut 10 ghosts
// versus the previous flat-0.55 config.
//
// Octave-NMS arbitration was tested and rejected — the model's onset-
// prob ranking is inverted for low bass (ghost > fundamental), so
// NMS would silently delete true low notes.
const ONSET_THRESH       = 0.3;
const FRAME_THRESH       = 0.5;
const MIN_NOTE_FRAMES    = 2;
const NMS_RADIUS         = 2;
const VELOCITY_FLOOR_MID = 0.60;  // pitches 40..85 — middle register, where
                                   // ghost and real-note velocity overlap most
const VELOCITY_FLOOR_EDGE = 0.40;  // pitches <40 or >85 — extreme ranges where
                                   // the model's velocity prediction is biased low
const PITCH_EDGE_LO      = 40;
const PITCH_EDGE_HI      = 85;
const CHUNK_OVERLAP      = CHUNK_FRAMES / 2;  // 128-frame step → boundary
                                              // notes fall into two windows

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
 * Posterior arrays → note events. Mirrors latent_pitch/infer.py:transcribe —
 * onset NMS → extend while frame>thresh → min duration → velocity mean.
 */
function _postprocess({ onset, frame, velocity, offset }, T) {
  const dt = 1.0 / VAE_HZ;
  const notes = [];
  for (let pitch = 0; pitch < N_PITCH; pitch++) {
    for (let t = 0; t < T; t++) {
      const on = onset[t * N_PITCH + pitch];
      if (on <= ONSET_THRESH) continue;
      // Local max within ±NMS_RADIUS frames for this pitch.
      let peak = true;
      const lo = Math.max(0, t - NMS_RADIUS);
      const hi = Math.min(T, t + NMS_RADIUS + 1);
      for (let k = lo; k < hi; k++) {
        if (k === t) continue;
        if (onset[k * N_PITCH + pitch] > on + 1e-9) { peak = false; break; }
      }
      if (!peak) continue;
      // Extend while frame probability stays above threshold.
      let endFrame = t + 1;
      while (endFrame < T && frame[endFrame * N_PITCH + pitch] > FRAME_THRESH) {
        endFrame++;
      }
      const nFrames = endFrame - t;
      if (nFrames < MIN_NOTE_FRAMES) continue;
      let vSum = 0;
      for (let k = t; k < endFrame; k++) vSum += velocity[k * N_PITCH + pitch];
      const vMean = vSum / nFrames;
      // Pitch-aware velocity gate — see constants block. Extreme pitches
      // have a lower floor because the model underpredicts velocity there.
      const floor = (pitch < PITCH_EDGE_LO || pitch > PITCH_EDGE_HI)
        ? VELOCITY_FLOOR_EDGE
        : VELOCITY_FLOOR_MID;
      if (vMean < floor) continue;
      const sub = offset[t * N_PITCH + pitch];
      notes.push({
        note: pitch,
        time: (t + sub) * dt,
        duration: nFrames * dt,
        velocity: Math.max(1, Math.min(127, Math.round(vMean * 127))),
      });
    }
  }
  notes.sort((a, b) => a.time - b.time);
  return { notes, duration: T * dt };
}

function _sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
