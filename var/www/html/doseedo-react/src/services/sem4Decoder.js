/**
 * Client-side 4-stem preview separator + decoder.
 *
 * Fills the gap between upload and when /separate-stems comes back with
 * authoritative WAVs: runs distill_demucs_fp16 to pull per-stem oobleck
 * latents, sem_demucs_packed once to get per-stem semantic embeddings,
 * then the 10M-param sem_decoder_fp16 to reconstruct placeholder audio
 * per stem. Emits chunk-by-chunk so the envelope animation can progress
 * smoothly without ever reloading the waveform.
 *
 * If the backend finishes before we do, the caller signals
 * `decodeAbortSignal` to stop further sem_decoder calls; distill_demucs
 * keeps running to completion so `onAllLatentsReady()` still fires (we
 * want the oobleck stem latents regardless — no need to run the oobleck
 * encoder on the mix afterwards). Full cancel via `abortSignal` stops
 * everything.
 *
 * I/O (matches the ONNX graphs exactly):
 *   distill_demucs_fp16 :  audio[1,2,N]           -> stem_latents[1,4,64,N/1920]
 *   sem_demucs_packed   :  waveform[1,2,N]        -> embedding[1,4,128] (+ rms)
 *   sem_decoder_fp16    :  latent[1,64,T], sem_emb[1,128] -> audio[1,2,1920*T]
 *
 * Envelope format matches rmsDemucs output (93.75 fps, [2*T] min/max
 * pairs: first T are -|amp|, last T are +|amp|), so the splice is a
 * direct replacement of the rms-painted values with actual decoded-audio
 * envelope for the chunk's time range. The waveform hook sees only a
 * Float32Array swap and doesn't re-render from scratch.
 */

const MODELS = {
  demucs:  '/static/models/distill_demucs_fp16.onnx',
  demucsData: '/static/models/distill_demucs_fp16.onnx.data',
  sem:     '/static/models/sem_demucs_packed.onnx',
  decoder: '/static/models/sem_decoder_fp16.onnx',
  decoderData: '/static/models/sem_decoder_fp16.onnx.data',
};

const SR = 48000;
const FRAME_SAMPLES = 1920;          // oobleck latent frame
const LATENT_CHANS = 64;
const ENV_HOP = 512;                 // 93.75 fps — matches rmsDemucs
const ENV_FPS = SR / ENV_HOP;
const N_STEMS = 4;
const STEM_NAMES = ['drums', 'bass', 'vocals', 'other'];

// Default chunk: 8 s. Latent T = 400. Keeps WebGPU mem bounded, one pass is
// quick, and stems stream in ~2 s increments at typical preview rates.
const DEFAULT_CHUNK_SAMPLES = 8 * SR;

let _ort = null;
let _sessions = null;             // { demucs, sem, decoder }
let _initPromise = null;

/** Idempotent load. Returns { demucs, sem, decoder } when ready. */
export async function initSem4Decoder(onProgress = null) {
  if (_sessions) return _sessions;
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

    // Fetch graphs + external data in parallel. The graph bytes are small; the
    // .data blobs (170 MB + 21 MB) dominate the budget and are what we cache
    // in IndexedDB so second loads are instant.
    const subProg = (tag, base) => (p) => {
      if (onProgress) onProgress({ tag, ...p, base });
    };
    const [
      demucsGraph, demucsWeights,
      semGraph,
      decoderGraph, decoderWeights,
    ] = await Promise.all([
      fetchModelWithCache(MODELS.demucs,      subProg('demucs.onnx', 'demucs')),
      fetchModelWithCache(MODELS.demucsData,  subProg('demucs.data', 'demucs')),
      fetchModelWithCache(MODELS.sem,         subProg('sem.onnx',    'sem')),
      fetchModelWithCache(MODELS.decoder,     subProg('decoder.onnx', 'decoder')),
      fetchModelWithCache(MODELS.decoderData, subProg('decoder.data', 'decoder')),
    ]);

    async function makeSession(graphBuf, extPairs /* [{path, bytes}, ...] */) {
      const backends = [];
      if (ort.env?.webgpu) backends.push('webgpu');
      backends.push('wasm');
      let lastErr = null;
      for (const ep of backends) {
        try {
          const options = {
            executionProviders: [ep],
            graphOptimizationLevel: 'all',
            logSeverityLevel: 3,
          };
          if (extPairs && extPairs.length) {
            options.externalData = extPairs.map(({ path, bytes }) => ({
              path,
              data: bytes.buffer || bytes,
            }));
          }
          const sess = await ort.InferenceSession.create(new Uint8Array(graphBuf), options);
          console.log(`[sem4Decoder] ${extPairs ? 'ext' : 'inline'} ready on ${ep}`);
          return sess;
        } catch (err) {
          const msg = err?.message || String(err);
          console.warn(`[sem4Decoder] ${ep} init failed:`, msg);
          lastErr = err;
        }
      }
      throw lastErr || new Error('no ORT backend available');
    }

    // Load all three in series — WebGPU sessions contend on the same device,
    // and the graphs are small. Avoids a GPU OOM race on lower-end machines.
    const demucsBytes = new Uint8Array(demucsWeights);
    const decoderBytes = new Uint8Array(decoderWeights);
    const demucsSess = await makeSession(demucsGraph, [
      { path: 'distill_demucs_fp16.onnx.data', bytes: demucsBytes },
    ]);
    const semSess = await makeSession(semGraph, null);
    const decoderSess = await makeSession(decoderGraph, [
      { path: 'sem_decoder_fp16.onnx.data', bytes: decoderBytes },
    ]);

    _sessions = { demucs: demucsSess, sem: semSess, decoder: decoderSess };
    return _sessions;
  })();

  return _initPromise;
}

export function isSem4DecoderReady() { return _sessions != null; }

// ─── helpers ────────────────────────────────────────────────────────────

/** Remove per-channel DC offset in place-ish; returns a fresh Float32Array.
 * Same fix as the 6-stem model uses — avoids NaN from quiet/DC-heavy intros. */
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

/** Extract stereo chunk [1, 2, N] in channels-first layout from a flat
 * [L0..L_{M-1}, R0..R_{M-1}] layout at sample offset + N. */
function buildChunkTensor(ort, flat, numFrames, startSample, chunkSamples) {
  const N = chunkSamples;
  const buf = new Float32Array(2 * N);
  buf.set(flat.subarray(startSample, startSample + N), 0);
  buf.set(flat.subarray(numFrames + startSample, numFrames + startSample + N), N);
  return new ort.Tensor('float32', buf, [1, 2, N]);
}

/** Envelope at 93.75 fps (hop 512) of a stereo decoded chunk [1,2,N].
 * Returns Float32Array[2*T] min/max pairs like rmsDemucs. */
function envelopeOfStereoChunk(audio, numSamples) {
  // audio: Float32Array length 2*N, channels-first [L...; R...]
  const T = Math.floor(numSamples / ENV_HOP);
  const env = new Float32Array(2 * T);
  for (let t = 0; t < T; t++) {
    const off = t * ENV_HOP;
    let peak = 0;
    const endL = off + ENV_HOP;
    for (let i = off; i < endL; i++) {
      const l = Math.abs(audio[i]);
      const r = Math.abs(audio[numSamples + i]);
      const v = l > r ? l : r;
      if (v > peak) peak = v;
    }
    env[t]     = -peak;
    env[T + t] =  peak;
  }
  return env;
}

/** Splice a per-chunk envelope into the full-track envelope at a time offset.
 * Format is [min0..min_{T-1}, max0..max_{T-1}] so mins/maxes live in two halves
 * that we have to patch independently. */
function spliceEnvelope(fullEnv, fullT, chunkEnv, chunkT, startFrame) {
  // mins slice
  const mEnd = Math.min(chunkT, fullT - startFrame);
  if (mEnd <= 0) return;
  for (let t = 0; t < mEnd; t++) {
    fullEnv[startFrame + t] = chunkEnv[t];                   // mins
    fullEnv[fullT + startFrame + t] = chunkEnv[chunkT + t];  // maxes
  }
}

/** Encode a stereo Float32Array [2*N] (channels-first) to a 16-bit PCM WAV
 * blob URL. Simple and fast; good enough for preview playback. */
function stereoBufferToWavUrl(buf, numSamples, sampleRate = SR) {
  const N = numSamples;
  const byteLen = 44 + N * 2 * 2;  // stereo 16-bit
  const ab = new ArrayBuffer(byteLen);
  const view = new DataView(ab);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, 'RIFF');
  view.setUint32(4, byteLen - 8, true);
  writeStr(8, 'WAVE'); writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);              // fmt size
  view.setUint16(20, 1, true);               // PCM
  view.setUint16(22, 2, true);               // stereo
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2 * 2, true);
  view.setUint16(32, 4, true);               // block align
  view.setUint16(34, 16, true);              // bits
  writeStr(36, 'data');
  view.setUint32(40, N * 2 * 2, true);
  let off = 44;
  for (let i = 0; i < N; i++) {
    let l = Math.max(-1, Math.min(1, buf[i]));
    let r = Math.max(-1, Math.min(1, buf[numSamples + i]));
    view.setInt16(off, (l * 0x7fff) | 0, true); off += 2;
    view.setInt16(off, (r * 0x7fff) | 0, true); off += 2;
  }
  const blob = new Blob([ab], { type: 'audio/wav' });
  return URL.createObjectURL(blob);
}

// ─── main entry point ───────────────────────────────────────────────────

/**
 * Stream a 4-stem preview separation + decode over a stereo 48k input.
 *
 * @param {Float32Array} flat        channels-first stereo [L..L_{N-1}, R..R_{N-1}]
 * @param {number}       numFrames
 * @param {object}       opts
 * @param {(e: {stemIdx: number, envelope: Float32Array, endSample: number}) => void} [opts.onStemEnvelopeExtended]
 *        Fires after each chunk × stem decode. `envelope` is the full per-stem
 *        envelope so the caller can replace track.metadata.envelopeData wholesale.
 * @param {(e: {stemIdx: number, wavUrl: string}) => void} [opts.onStemDecoded]
 *        Fires once per stem when all decode chunks land.
 * @param {(e: {stemLatents: Float32Array[]}) => void} [opts.onAllLatentsReady]
 *        Fires when distill_demucs has finished all chunks for all 4 stems.
 *        Array of 4 latent buffers, each Float32Array[LATENT_CHANS*T_total].
 * @param {AbortSignal} [opts.abortSignal]       cancels the whole pipeline
 * @param {AbortSignal} [opts.decodeAbortSignal] cancels sem_decoder only
 * @param {number}      [opts.chunkSamples=8*SR] per-chunk size (multiple of 1920)
 */
export async function streamPreviewSeparation(flat, numFrames, opts = {}) {
  const {
    onStemEnvelopeExtended, onStemDecoded, onAllLatentsReady,
    abortSignal, decodeAbortSignal,
    chunkSamples: _chunkSamples = DEFAULT_CHUNK_SAMPLES,
  } = opts;

  const { demucs, sem, decoder } = await initSem4Decoder();
  const ort = _ort;

  // Round chunk size to a multiple of FRAME_SAMPLES so latent T stays integer.
  const chunkSamples = Math.floor(_chunkSamples / FRAME_SAMPLES) * FRAME_SAMPLES;
  const nChunks = Math.ceil(numFrames / chunkSamples);
  const totalT = Math.floor(numFrames / ENV_HOP);

  // DC-remove once up front — see debug_6stem_pos.py / rmsDemucs notes.
  const cleanFlat = dcRemove(flat, numFrames);

  // Full-song embedding [1,4,128]. Running sem_demucs on the whole thing gives
  // us a semantically stable per-stem "what this stem should be" vector — much
  // simpler than recomputing per chunk and, in practice, no worse for preview.
  const semInput = new ort.Tensor('float32', (() => {
    const buf = new Float32Array(2 * numFrames);
    buf.set(cleanFlat.subarray(0, numFrames), 0);
    buf.set(cleanFlat.subarray(numFrames, 2 * numFrames), numFrames);
    return buf;
  })(), [1, 2, numFrames]);
  const semOut = await sem.run({ waveform: semInput });
  const embedding = semOut['embedding']?.data;
  if (!embedding || embedding.length !== N_STEMS * 128) {
    throw new Error('sem_demucs_packed did not return expected [1,4,128] embedding');
  }

  // Per-stem accumulators
  const envelopes = Array.from({ length: N_STEMS }, () => new Float32Array(2 * totalT));
  const audioBufs = Array.from({ length: N_STEMS }, () => new Float32Array(2 * numFrames));
  const latentBufs = Array.from({ length: N_STEMS },
    () => new Float32Array(LATENT_CHANS * Math.floor(numFrames / FRAME_SAMPLES)));

  // Track per-stem completion (all chunks decoded)
  const stemChunksDone = new Array(N_STEMS).fill(0);
  const stemsEmitted = new Array(N_STEMS).fill(false);

  const isAborted = () => abortSignal?.aborted;

  for (let c = 0; c < nChunks; c++) {
    if (isAborted()) return;
    const startSample = c * chunkSamples;
    const thisN = Math.min(chunkSamples, numFrames - startSample);
    const useN = Math.floor(thisN / FRAME_SAMPLES) * FRAME_SAMPLES;
    if (useN < FRAME_SAMPLES) continue;
    const startFrame = Math.floor(startSample / ENV_HOP);

    // 1) distill_demucs on this chunk → stem_latents [1,4,64,T_chunk]
    const chunkTensor = buildChunkTensor(ort, cleanFlat, numFrames, startSample, useN);
    const demucsOut = await demucs.run({ audio: chunkTensor });
    const stemLatents = demucsOut['stem_latents']?.data;     // Float32 length 1*4*64*T
    if (!stemLatents) throw new Error('distill_demucs returned no stem_latents');
    const Tchunk = useN / FRAME_SAMPLES;
    // Copy into per-stem latent accumulators.
    const latentOffFrames = Math.floor(startSample / FRAME_SAMPLES);
    for (let s = 0; s < N_STEMS; s++) {
      const srcStride = LATENT_CHANS * Tchunk;
      const srcOff = s * srcStride;
      for (let ct = 0; ct < LATENT_CHANS; ct++) {
        for (let t = 0; t < Tchunk; t++) {
          latentBufs[s][(ct * latentBufs[s].length / LATENT_CHANS) + latentOffFrames + t] =
            stemLatents[srcOff + ct * Tchunk + t];
        }
      }
    }

    // 2) sem_decoder per stem (if decode not aborted)
    if (!decodeAbortSignal?.aborted) {
      for (let s = 0; s < N_STEMS; s++) {
        if (isAborted()) return;
        if (decodeAbortSignal?.aborted) break;
        // Latent slice for this chunk × this stem: [1, 64, Tchunk]
        const lat = new Float32Array(LATENT_CHANS * Tchunk);
        const srcOff = s * LATENT_CHANS * Tchunk;
        lat.set(stemLatents.subarray(srcOff, srcOff + lat.length));
        const latTensor = new ort.Tensor('float32', lat, [1, LATENT_CHANS, Tchunk]);
        // sem_emb slice [1, 128]
        const emb = new Float32Array(128);
        emb.set(embedding.subarray(s * 128, (s + 1) * 128));
        const embTensor = new ort.Tensor('float32', emb, [1, 128]);

        const decOut = await decoder.run({ latent: latTensor, sem_emb: embTensor });
        const audioChunkFlat = decOut['audio']?.data;   // [1, 2, 1920*Tchunk]
        if (!audioChunkFlat) throw new Error('sem_decoder returned no audio');
        const chunkN = 1920 * Tchunk;

        // Copy into stem's full audio accumulator (channels-first [L; R])
        const ch0 = audioChunkFlat.subarray(0, chunkN);
        const ch1 = audioChunkFlat.subarray(chunkN, 2 * chunkN);
        audioBufs[s].set(ch0, startSample);
        audioBufs[s].set(ch1, numFrames + startSample);

        // Envelope for this chunk → splice into stem's envelope
        const chunkEnv = envelopeOfStereoChunk(audioChunkFlat, chunkN);
        const chunkEnvT = chunkEnv.length / 2;
        spliceEnvelope(envelopes[s], totalT, chunkEnv, chunkEnvT, startFrame);

        stemChunksDone[s] += 1;

        if (onStemEnvelopeExtended) {
          onStemEnvelopeExtended({
            stemIdx: s,
            envelope: envelopes[s],
            endSample: startSample + chunkN,
          });
        }

        // If this stem just completed all chunks, emit the WAV URL.
        if (stemChunksDone[s] === nChunks && !stemsEmitted[s] && onStemDecoded) {
          const url = stereoBufferToWavUrl(audioBufs[s], numFrames);
          stemsEmitted[s] = true;
          onStemDecoded({ stemIdx: s, stemName: STEM_NAMES[s], wavUrl: url });
        }
      }
    }
  }

  if (onAllLatentsReady && !isAborted()) {
    onAllLatentsReady({ stemLatents: latentBufs, stemNames: STEM_NAMES });
  }
}

export { STEM_NAMES as SEM4_STEM_NAMES, ENV_FPS as SEM4_ENVELOPE_FPS };
