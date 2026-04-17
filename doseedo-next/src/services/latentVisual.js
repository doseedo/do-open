/**
 * Tiny latent → peak-envelope runtime.
 *
 * 62K-param 4-layer 1D conv. Sub-millisecond forward pass even for
 * 20+ minutes of audio. Used to render waveform strips INSTANTLY from
 * a latent, before the heavyweight VAE decoder has finished loading /
 * decoding.
 *
 * Input:  [T, 64] latent (time-major, same as what /api/latent/<id> returns)
 * Output: Float32Array length 2*T with [min0, max0, min1, max1, ...]
 *         (or Float32Array[T*2] interpreted as [2, T] channels-first)
 */

const MODEL_URL = '/static/models/latent_visual.onnx';
const MODEL_DATA_URL = '/static/models/latent_visual.onnx.data';
const LATENT_CHANNELS = 64;

let _ort = null;
let _session = null;
let _sessionPromise = null;

export async function initLatentVisual() {
  if (_session) return _session;
  if (_sessionPromise) return _sessionPromise;

  _sessionPromise = (async () => {
    const ort = await import('onnxruntime-web');
    _ort = ort;
    if (ort.env?.wasm) {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
      ort.env.wasm.numThreads = Math.min(2, navigator.hardwareConcurrency || 1);
      ort.env.wasm.simd = true;
    }
    // Model is tiny (244 KB) — single fetch, no progress needed.
    // Fetch both graph + external data explicitly for ORT-Web compat.
    const graphResp = await fetch(MODEL_URL, { cache: 'force-cache' });
    if (!graphResp.ok) throw new Error(`latent_visual graph HTTP ${graphResp.status}`);
    const graphBytes = new Uint8Array(await graphResp.arrayBuffer());
    const dataResp = await fetch(MODEL_DATA_URL, { cache: 'force-cache' });
    if (!dataResp.ok) throw new Error(`latent_visual data HTTP ${dataResp.status}`);
    const dataBytes = new Uint8Array(await dataResp.arrayBuffer());
    const externalData = [{ path: 'latent_visual.onnx.data', data: dataBytes.buffer }];
    // WASM is plenty — WebGPU setup overhead isn't worth it for a tiny
    // conv net where compute is already kernel-launch-bound.
    const sess = await ort.InferenceSession.create(graphBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      externalData,
    });
    _session = sess;
    console.log('[latentVisual] ready');
    return sess;
  })();

  return _sessionPromise;
}

/**
 * flatTD: Float32Array of length T*64 in time-major [T, 64] order.
 * Returns Float32Array of length 2*T as [T] mins followed by [T] maxes
 * (channels-first, matches the model's [B, 2, T] output layout).
 */
export async function envelopeFromLatent(flatTD, T) {
  const sess = await initLatentVisual();
  const ort = _ort;
  // Model expects [B, 64, T] channels-first. Transpose [T, 64] → [64, T].
  const dt = new Float32Array(T * LATENT_CHANNELS);
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < LATENT_CHANNELS; d++) {
      dt[d * T + t] = flatTD[t * LATENT_CHANNELS + d];
    }
  }
  const input = new ort.Tensor('float32', dt, [1, LATENT_CHANNELS, T]);
  const out = await sess.run({ latent: input });
  // Model output name is "envelope" per latent_visual/train.py
  const envT = out.envelope || out[Object.keys(out)[0]];
  // envT.data is Float32Array length 1*2*T in [B, 2, T] layout
  return envT.data;
}

/**
 * Build a fake AudioBuffer from an envelope [2*T] (first T mins, then T maxes)
 * that produces the RIGHT visual when useWaveform draws RMS bars from it.
 *
 * We fill each of the T × 1920 samples per frame with ±amp, alternating
 * so the RMS at that frame ≈ (max - min) / 2 = peak amplitude.
 *
 * Fast-path: only ~T*1920 writes per channel (~1.5M per 30s stem @ 48kHz),
 * takes <50ms even on a slow machine. Much faster than waiting for the
 * 322MB VAE decoder to run.
 */
export function buildFakeBufferFromEnvelope(envFlat, T, audioCtx) {
  const SR = 48000;
  const SAMPLES_PER_FRAME = 1920;
  const numFrames = T * SAMPLES_PER_FRAME;
  const buf = audioCtx.createBuffer(2, numFrames, SR);
  const ch0 = buf.getChannelData(0);
  const ch1 = buf.getChannelData(1);
  const mins = envFlat.subarray(0, T);
  const maxs = envFlat.subarray(T, 2 * T);
  // Use the center and half-range of the envelope so RMS matches peak.
  for (let t = 0; t < T; t++) {
    const mn = mins[t];
    const mx = maxs[t];
    const center = (mn + mx) * 0.5;
    const amp = Math.max(1e-6, (mx - mn) * 0.5);
    const base = t * SAMPLES_PER_FRAME;
    for (let s = 0; s < SAMPLES_PER_FRAME; s++) {
      // Alternating square so RMS ≈ amp
      const v = (s & 1) ? center + amp : center - amp;
      ch0[base + s] = v;
      ch1[base + s] = v;
    }
  }
  return buf;
}

/**
 * Build an array of normalized peak pairs for canvas drawing.
 * Returns { peaks: Float32Array[2 * width], min, max }
 * where peaks[2*i]=min and peaks[2*i+1]=max at pixel column i.
 */
export function envelopeToPeaks(envFlat, T, width) {
  // envFlat is [2*T]: first T mins, then T maxes
  const mins = envFlat.subarray(0, T);
  const maxs = envFlat.subarray(T, 2 * T);

  const peaks = new Float32Array(2 * width);
  let minAll = Infinity, maxAll = -Infinity;
  for (let i = 0; i < width; i++) {
    const start = Math.floor((i * T) / width);
    const end = Math.max(start + 1, Math.floor(((i + 1) * T) / width));
    let mn = Infinity, mx = -Infinity;
    for (let j = start; j < end; j++) {
      if (mins[j] < mn) mn = mins[j];
      if (maxs[j] > mx) mx = maxs[j];
    }
    peaks[2 * i]     = mn;
    peaks[2 * i + 1] = mx;
    if (mn < minAll) minAll = mn;
    if (mx > maxAll) maxAll = mx;
  }
  return { peaks, min: minAll, max: maxAll };
}
