/**
 * Client-side 4-stem LATENT extractor.
 *
 * Runs distill_demucs_fp16 to produce per-stem oobleck latents
 * (stem_latents [1, 4, 64, T]) for a stereo 48 kHz input. Latents get
 * cached on each stem track so downstream features (drum sub-sep,
 * regeneration, etc.) can send latent to the backend instead of audio.
 *
 * Previously this module also ran sem_demucs + sem_decoder to paint an
 * intermediate decoded-audio envelope/preview between rmsDemucs and the
 * backend /separate-stems WAVs. That path was removed: the sem_decoder
 * output was low-amplitude on WebGPU and visually looked nothing like the
 * real htdemucs waveform. The new UX is just:
 *
 *   rmsDemucs (instant)  →  backend WAV envelope (final)
 *
 * with no intermediate audio decode. `streamPreviewSeparation` now only
 * emits `onAllLatentsReady({ stemLatents })` once distill_demucs has run
 * on every chunk.
 *
 * I/O (matches the ONNX graph exactly):
 *   distill_demucs_fp16 :  audio[1,2,N] -> stem_latents[1,4,64,N/1920]
 */

const MODELS = {
  demucs:     '/static/models/distill_demucs_fp16.onnx',
  demucsData: '/static/models/distill_demucs_fp16.onnx.data',
};

const SR = 48000;
const FRAME_SAMPLES = 1920;          // oobleck latent frame
const LATENT_CHANS = 64;
const LATENT_FPS = SR / FRAME_SAMPLES;  // 25 fps
const N_STEMS = 4;
const STEM_NAMES = ['drums', 'bass', 'vocals', 'other'];

// Default chunk: 8 s. Latent T = 200. Keeps GPU memory bounded on the
// 170 MB fp16 weights even on lower-end machines.
const DEFAULT_CHUNK_SAMPLES = 8 * SR;

let _ort = null;
let _session = null;
let _initPromise = null;

/** Idempotent load. Returns the distill_demucs session when ready. */
export async function initSem4Decoder(onProgress = null) {
  if (_session) return _session;
  if (_initPromise) return _initPromise;

  _initPromise = (async () => {
    const ort = await import('onnxruntime-web');
    _ort = ort;

    if (ort.env?.wasm) {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
      const coi = typeof crossOriginIsolated === 'boolean' ? crossOriginIsolated : false;
      ort.env.wasm.numThreads = coi ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
      ort.env.wasm.simd = true;
    }
    if (ort.env) ort.env.logLevel = 'error';

    const { fetchModelWithCache } = await import('./modelCacheService');

    const subProg = (tag, base) => (p) => {
      if (onProgress) onProgress({ tag, ...p, base });
    };
    console.log('[sem4Decoder] loading distill_demucs_fp16 (170 MB cold / IndexedDB warm)');
    const [demucsGraph, demucsWeights] = await Promise.all([
      fetchModelWithCache(MODELS.demucs,      subProg('demucs.onnx', 'demucs')),
      fetchModelWithCache(MODELS.demucsData,  subProg('demucs.data', 'demucs')),
    ]);

    const backends = [];
    if (ort.env?.webgpu) backends.push('webgpu');
    backends.push('wasm');
    const demucsBytes = new Uint8Array(demucsWeights);
    let lastErr = null;
    for (const ep of backends) {
      try {
        _session = await ort.InferenceSession.create(new Uint8Array(demucsGraph), {
          executionProviders: [ep],
          graphOptimizationLevel: 'all',
          logSeverityLevel: 3,
          externalData: [{
            path: 'distill_demucs_fp16.onnx.data',
            data: demucsBytes.buffer || demucsBytes,
          }],
        });
        console.log(`[sem4Decoder] ready on ${ep} (distill_demucs only)`);
        return _session;
      } catch (err) {
        console.warn(`[sem4Decoder] ${ep} init failed:`, err?.message || err);
        lastErr = err;
      }
    }
    throw lastErr || new Error('no ORT backend available');
  })();

  return _initPromise;
}

export function isSem4DecoderReady() { return _session != null; }

// ─── helpers ────────────────────────────────────────────────────────────

/** Remove per-channel DC offset. Avoids NaN from quiet/DC-heavy intros. */
function dcRemove(flat, numFrames) {
  let meanL = 0, meanR = 0;
  for (let i = 0; i < numFrames; i++) {
    meanL += flat[i];
    meanR += flat[numFrames + i];
  }
  meanL /= numFrames; meanR /= numFrames;
  const out = new Float32Array(flat.length);
  for (let i = 0; i < numFrames; i++) {
    out[i] = flat[i] - meanL;
    out[numFrames + i] = flat[numFrames + i] - meanR;
  }
  return out;
}

/** Extract stereo chunk [1, 2, N] from channels-first [L0..L_{M-1}, R0..R_{M-1}]. */
function buildChunkTensor(ort, flat, numFrames, startSample, chunkSamples) {
  const N = chunkSamples;
  const buf = new Float32Array(2 * N);
  buf.set(flat.subarray(startSample, startSample + N), 0);
  buf.set(flat.subarray(numFrames + startSample, numFrames + startSample + N), N);
  return new ort.Tensor('float32', buf, [1, 2, N]);
}

// ─── main entry point ───────────────────────────────────────────────────

/**
 * Run distill_demucs over the full track in chunks, accumulating per-stem
 * latents. Emits a single `onAllLatentsReady({ stemLatents })` when done.
 *
 * @param {Float32Array} flat        channels-first stereo [L..L_{N-1}, R..R_{N-1}]
 * @param {number}       numFrames
 * @param {object}       opts
 * @param {(e: {stemLatents: Float32Array[], stemNames: string[], fps: number}) => void} [opts.onAllLatentsReady]
 * @param {AbortSignal}  [opts.abortSignal]       cancels the pipeline
 * @param {number}       [opts.chunkSamples=8*SR] per-chunk size (multiple of 1920)
 */
export async function streamPreviewSeparation(flat, numFrames, opts = {}) {
  const {
    onAllLatentsReady,
    abortSignal,
    chunkSamples: _chunkSamples = DEFAULT_CHUNK_SAMPLES,
  } = opts;

  const demucs = await initSem4Decoder();
  const ort = _ort;
  const { ortWebGPURun } = await import('./webgpuOrtQueue');

  const chunkSamples = Math.floor(_chunkSamples / FRAME_SAMPLES) * FRAME_SAMPLES;
  const nChunks = Math.ceil(numFrames / chunkSamples);

  const cleanFlat = dcRemove(flat, numFrames);

  const totalLatentT = Math.floor(numFrames / FRAME_SAMPLES);
  const latentBufs = Array.from({ length: N_STEMS },
    () => new Float32Array(LATENT_CHANS * totalLatentT));

  const isAborted = () => abortSignal?.aborted;
  console.log(`[sem4Decoder] latent extract: ${nChunks} chunks × 4 stems on ${numFrames / SR | 0}s input`);

  for (let c = 0; c < nChunks; c++) {
    if (isAborted()) return;
    const startSample = c * chunkSamples;
    const thisN = Math.min(chunkSamples, numFrames - startSample);
    const useN = Math.floor(thisN / FRAME_SAMPLES) * FRAME_SAMPLES;
    if (useN < FRAME_SAMPLES) continue;

    const chunkTensor = buildChunkTensor(ort, cleanFlat, numFrames, startSample, useN);
    const demucsOut = await ortWebGPURun(() => demucs.run({ audio: chunkTensor }));
    const stemLatents = demucsOut['stem_latents']?.data;  // Float32 [1, 4, 64, Tchunk]
    if (!stemLatents) throw new Error('distill_demucs returned no stem_latents');
    const Tchunk = useN / FRAME_SAMPLES;
    const latentOffFrames = Math.floor(startSample / FRAME_SAMPLES);
    for (let s = 0; s < N_STEMS; s++) {
      const srcStride = LATENT_CHANS * Tchunk;
      const srcOff = s * srcStride;
      for (let ct = 0; ct < LATENT_CHANS; ct++) {
        const dstRowOff = ct * totalLatentT + latentOffFrames;
        for (let t = 0; t < Tchunk; t++) {
          latentBufs[s][dstRowOff + t] = stemLatents[srcOff + ct * Tchunk + t];
        }
      }
    }
  }

  if (!isAborted()) {
    console.log('[sem4Decoder] latent extract: complete');
    onAllLatentsReady && onAllLatentsReady({
      stemLatents: latentBufs,
      stemNames: STEM_NAMES,
      fps: LATENT_FPS,
    });
  }
}

export { STEM_NAMES as SEM4_STEM_NAMES, LATENT_FPS as SEM4_LATENT_FPS, LATENT_CHANS as SEM4_LATENT_CHANS };
