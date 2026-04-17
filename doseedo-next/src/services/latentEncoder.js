/**
 * Client-side VAE encoder — stereo 48 kHz waveform → latents.
 *
 * This is the browser-first replacement for the server-side
 * /api/encode-audio-latent and /api/encode-latents-bulk endpoints that
 * currently POST WAV bytes to Modal. The oobleck_encoder ONNX is the
 * same VAE encoder stemphonic_server.py runs on the backend; once this
 * service is wired into those code paths, raw audio never leaves the
 * browser for latent extraction.
 *
 * I/O (matches the backend exactly):
 *   audio   [1, 2, N_samples]          float32, stereo 48 kHz
 *   latent  [1, 64, N_samples / 1920]  float32, 25 Hz frames
 *
 * This module is currently LOAD-ONLY — initLatentEncoder() warms the
 * model in the background so the first real call isn't cold. No
 * encode path is wired into the DAW yet; that's a follow-up commit
 * that will replace encodeAudioLatent + encodeLatentsBulk in
 * trackAnalysisAPI.js / DAWOptimized.js.
 */

const MODEL_URL = '/static/models/oobleck_encoder.onnx';
const MODEL_DATA_URL = '/static/models/oobleck_encoder.onnx.data';
const TARGET_SR = 48000;
const FRAME_SAMPLES = 1920;     // one latent frame per 1920 audio samples
const LATENT_CHANNELS = 64;

let _ort = null;
let _session = null;
let _sessionPromise = null;
let _loadProgress = null;
let _runQueue = Promise.resolve();

/**
 * Idempotently load the oobleck_encoder ONNX. Resolves to the
 * InferenceSession once ready. Safe to call multiple times — subsequent
 * calls share the first call's promise.
 *
 * @param {(p: {bytesLoaded: number, bytesTotal: number}) => void} onProgress
 */
export async function initLatentEncoder(onProgress = null) {
  if (_session) return _session;
  if (_sessionPromise) return _sessionPromise;

  _sessionPromise = (async () => {
    const ort = await import('onnxruntime-web');
    _ort = ort;
    if (ort.env?.wasm) {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
      // Multi-threaded WASM needs SharedArrayBuffer, which needs the page to
      // be cross-origin-isolated (COOP: same-origin + COEP: require-corp).
      // Without that, requesting numThreads > 1 hard-fails in onnxruntime-web
      // ("pthread_create failed") instead of gracefully falling back — so we
      // explicitly clamp to 1 thread when isolation is off.
      const coi = typeof crossOriginIsolated === 'boolean' ? crossOriginIsolated : false;
      ort.env.wasm.numThreads = coi ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
      ort.env.wasm.simd = true;
    }
    // Silence harmless "Could not find a CPU kernel" / "Some nodes were not
    // assigned to the preferred EP" warnings during WebGPU compilation. These
    // fire once per ReduceL2 op on every load and spam dozens of errors in
    // the console. 'error' hides warnings but keeps real errors visible.
    if (ort.env) ort.env.logLevel = 'error';

    const { fetchModelWithCache } = await import('./modelCacheService');
    const graphBuf = await fetchModelWithCache(MODEL_URL);
    const graphBytes = new Uint8Array(graphBuf);

    const dataBuf = await fetchModelWithCache(MODEL_DATA_URL, (p) => {
      _loadProgress = p;
      if (onProgress) onProgress(p);
    });
    const dataBytes = new Uint8Array(dataBuf);

    // Register the weight blob under BOTH possible filenames so ORT resolves
    // the graph's internal external_data_file reference regardless of whether
    // we're serving the fp32 or fp16 variant (the fp16 .onnx internally
    // references `oobleck_encoder_fp16.onnx.data`; fp32 references the plain
    // `oobleck_encoder.onnx.data`). The URL we fetch is stable either way.
    const externalData = [
      { path: 'oobleck_encoder.onnx.data', data: dataBytes.buffer },
      { path: 'oobleck_encoder_fp16.onnx.data', data: dataBytes.buffer },
    ];

    // Prefer WebGPU; fall back to WASM on systems without GPU support.
    const backends = [];
    if (ort.env?.webgpu) backends.push('webgpu');
    backends.push('wasm');
    let lastErr = null;
    for (const ep of backends) {
      try {
        const sess = await ort.InferenceSession.create(graphBytes, {
          executionProviders: [ep],
          graphOptimizationLevel: 'all',
          externalData,
          // logSeverityLevel: 0=verbose, 1=info, 2=warning, 3=error, 4=fatal.
          // We want the session itself quiet (the CPU-kernel / EP-assignment
          // warnings are expected for WebGPU compilation and just spam).
          logSeverityLevel: 3,
        });
        _session = sess;
        console.log(`[latentEncoder] ready on ${ep}`);
        return sess;
      } catch (err) {
        const msg = err?.message || err?.toString?.() || JSON.stringify(err) || 'unknown';
        console.warn(`[latentEncoder] ${ep} init failed:`, msg);
        if (ep === 'webgpu') {
          try {
            const { trackEvent, PRODUCT_EVENTS, platformString } = await import('../lib/telemetry');
            trackEvent(PRODUCT_EVENTS.WEBGPU_INIT_FAILED, {
              model: 'latentEncoder',
              reason: msg.slice(0, 300),
              platform: platformString(),
            });
          } catch (_) { /* best-effort */ }
        }
        lastErr = err;
      }
    }
    // If we get here, BOTH backends failed — encoder is unusable for this session.
    try {
      const { trackEvent, PRODUCT_EVENTS, platformString } = await import('../lib/telemetry');
      trackEvent(PRODUCT_EVENTS.ENCODER_UNAVAILABLE, {
        reason: lastErr?.message?.slice(0, 300) || 'unknown',
        platform: platformString(),
      });
    } catch (_) { /* best-effort */ }
    throw lastErr || new Error('no ORT backend available');
  })();

  return _sessionPromise;
}

export function getLoadProgress() { return _loadProgress; }
export function isLatentEncoderReady() { return _session != null; }

/** Decode an arbitrary audio file in the browser → stereo 48 kHz Float32. */
export async function audioFileToStereo48k(file) {
  const ACtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new ACtx({ sampleRate: TARGET_SR });
  const arrayBuf = await file.arrayBuffer();
  const audioBuf = await ctx.decodeAudioData(arrayBuf.slice(0));
  const numCh = audioBuf.numberOfChannels;
  const numFrames = audioBuf.length;
  const left = audioBuf.getChannelData(0);
  const right = numCh >= 2 ? audioBuf.getChannelData(1) : left;
  const flat = new Float32Array(2 * numFrames);
  flat.set(left, 0);
  flat.set(right, numFrames);
  return { flat, numFrames, sampleRate: audioBuf.sampleRate };
}

/**
 * Run the VAE encoder on a stereo waveform.
 * Returns a Float32Array of length (T * 64) in time-major [T, 64] order,
 * matching the DOAE binary serialization format used by uploadLatent().
 *
 * chunkSamples: if the input is longer than this, run in non-overlapping
 *   chunks and concatenate. Keeps peak WebGPU memory bounded.
 *
 * NOTE: not wired into any code path yet. Exported for future use by
 *   trackAnalysisAPI.js (replacing encodeAudioLatent).
 */
export async function encodeToLatent(flat, numFrames, chunkSamples = 48000 * 30) {
  const sess = await initLatentEncoder();
  const ort = _ort;
  const work = _runQueue.then(() => _encode(sess, ort, flat, numFrames, chunkSamples));
  _runQueue = work.catch(() => {}); // don't let one failure poison the queue
  return work;
}

async function _encode(sess, ort, flat, numFrames, chunkSamples) {
  const outFrames = Math.floor(numFrames / FRAME_SAMPLES);
  if (outFrames <= 0) return new Float32Array(0);

  // Chunked inference: keep each call bounded in size.
  const stepSamples = Math.floor(chunkSamples / FRAME_SAMPLES) * FRAME_SAMPLES;
  const out = new Float32Array(outFrames * LATENT_CHANNELS);

  let consumed = 0;
  let tWritten = 0;
  while (consumed < numFrames) {
    const thisN = Math.min(stepSamples, numFrames - consumed);
    // Skip any trailing remainder shorter than one latent frame.
    if (thisN < FRAME_SAMPLES) break;
    const useN = Math.floor(thisN / FRAME_SAMPLES) * FRAME_SAMPLES;
    const chunkFlat = new Float32Array(2 * useN);
    // Rebuild channels-first layout for this slice: [L; R]
    chunkFlat.set(flat.subarray(consumed, consumed + useN), 0);
    chunkFlat.set(flat.subarray(numFrames + consumed, numFrames + consumed + useN), useN);

    const input = new ort.Tensor('float32', chunkFlat, [1, 2, useN]);
    const res = await sess.run({ audio: input });
    const lat = res.latent || res[Object.keys(res)[0]];
    // lat.data is Float32Array length 1*64*T_chunk in [1, 64, T] layout.
    // Convert to time-major [T, 64] and append.
    const Tchunk = useN / FRAME_SAMPLES;
    const src = lat.data;
    for (let t = 0; t < Tchunk; t++) {
      for (let c = 0; c < LATENT_CHANNELS; c++) {
        out[(tWritten + t) * LATENT_CHANNELS + c] = src[c * Tchunk + t];
      }
    }
    tWritten += Tchunk;
    consumed += useN;
  }
  return out.subarray(0, tWritten * LATENT_CHANNELS);
}
