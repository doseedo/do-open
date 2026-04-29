/**
 * Client-side latent_demucs student — waveform → 4 stem latents.
 *
 * This is the browser-first source-separation path. The distill_demucs
 * ONNX takes a stereo waveform [1, 2, N_samples] @ 48 kHz and produces
 * 4 stem latents [1, 4, 64, T] where T = N_samples / 1920.
 *
 * The stem latents are uploaded to the backend as DOAE binaries via
 * /api/upload-latent, which stores them and returns a latent_id that
 * downstream endpoints (/api/repaint-meter, /api/regen-stem-for-chord,
 * etc.) can consume.
 *
 * Backend never sees audio in this path — the waveform stays in the
 * browser. Only if this service fails (no WebGPU+WASM, load error,
 * etc.) does the caller fall back to backend /separate-stems.
 */

import { ortWebGPURun } from './webgpuOrtQueue';

const MODEL_URL = '/static/models/distill_demucs.onnx';
const MODEL_DATA_URL = '/static/models/distill_demucs.onnx.data';
const TARGET_SR = 48000;
const FRAME_SAMPLES = 1920;
const STEM_NAMES = ['drums', 'bass', 'other', 'vocals']; // order must match training
const LATENT_CHANNELS = 64;

let _ort = null;
let _session = null;
let _sessionPromise = null;
let _loadProgress = null;
let _vaeVersion = null;
let _runQueue = Promise.resolve();  // serialize InferenceSession.run() calls

export async function initLatentDemucs(onProgress = null) {
  if (_session) return _session;
  if (_sessionPromise) return _sessionPromise;

  _sessionPromise = (async () => {
    const ort = await import('onnxruntime-web');
    _ort = ort;
    if (ort.env?.wasm) {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
      ort.env.wasm.simd = true;
    }

    // Fetch both the graph AND the external data file. ORT-Web needs
    // them explicitly via the `externalData` option when loading from
    // bytes. Streaming the .data file (325 MB) gives us progress.
    const graphResp = await fetch(MODEL_URL, { cache: 'force-cache' });
    if (!graphResp.ok) throw new Error(`distill_demucs.onnx HTTP ${graphResp.status}`);
    const graphBytes = new Uint8Array(await graphResp.arrayBuffer());

    const dataResp = await fetch(MODEL_DATA_URL, { cache: 'force-cache' });
    if (!dataResp.ok) throw new Error(`distill_demucs.onnx.data HTTP ${dataResp.status}`);
    const total = parseInt(dataResp.headers.get('content-length') || '0', 10);
    const reader = dataResp.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      _loadProgress = { bytesLoaded: loaded, bytesTotal: total };
      if (onProgress) onProgress(_loadProgress);
    }
    const dataBytes = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) { dataBytes.set(c, off); off += c.byteLength; }

    // The external-data filename embedded in the .onnx is the basename
    // of the .onnx.data file (as the exporter wrote it alongside).
    const externalData = [
      { path: 'distill_demucs.onnx.data', data: dataBytes.buffer },
    ];

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
        });
        _session = sess;
        console.log(`[latentDemucs] ready on ${ep}`);
        return sess;
      } catch (err) {
        const msg = err?.message || err?.toString?.() || JSON.stringify(err) || 'unknown';
        console.warn(`[latentDemucs] ${ep} init failed:`, msg);
        // Record a typed product event for WebGPU-specific failures so
        // we can separate "user has no WebGPU" from "ONNX graph broke".
        if (ep === 'webgpu') {
          try {
            const { trackEvent, PRODUCT_EVENTS, platformString } = await import('../lib/telemetry');
            trackEvent(PRODUCT_EVENTS.WEBGPU_INIT_FAILED, {
              model: 'latentDemucs',
              reason: msg.slice(0, 300),
              platform: platformString(),
            });
          } catch (_) { /* telemetry is best-effort */ }
        }
        lastErr = err;
      }
    }
    throw lastErr || new Error('no ORT backend available');
  })();

  return _sessionPromise;
}

export function getLoadProgress() { return _loadProgress; }
export const LATENT_DEMUCS_STEM_NAMES = STEM_NAMES;

/** Decode an arbitrary audio file in the browser → stereo 48kHz Float32. */
export async function audioFileToStereo48k(file) {
  const ACtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new ACtx({ sampleRate: TARGET_SR });
  const arrayBuf = await file.arrayBuffer();
  // decodeAudioData uses the AudioContext's rate when it needs to resample.
  const audioBuf = await ctx.decodeAudioData(arrayBuf.slice(0));
  const numCh = audioBuf.numberOfChannels;
  const numFrames = audioBuf.length;
  // Force stereo. If mono, duplicate. If >2 channels, take first 2.
  const left = audioBuf.getChannelData(0);
  const right = numCh >= 2 ? audioBuf.getChannelData(1) : left;
  // ONNX wants [2, N] channels-first as one flat Float32Array
  const flat = new Float32Array(2 * numFrames);
  flat.set(left, 0);
  flat.set(right, numFrames);
  // If AudioContext's native SR differs from TARGET_SR, we've already
  // been resampled by the decoder's resampling pipeline.
  return { flat, numFrames, sampleRate: audioBuf.sampleRate };
}

/**
 * Run the distill_demucs ONNX on a stereo waveform.
 * Returns an array of 4 Float32Arrays (one per stem), each in [T, 64]
 * time-major format ready to be DOAE-serialized.
 *
 * chunkSamples: if the input is longer than this many samples, run in
 *   non-overlapping chunks. Default = 48000 * 30 (30 seconds) which
 *   keeps peak WASM memory bounded.
 */
export async function separateToLatentStems(flat, numFrames, chunkSamples = 48000 * 30) {
  const sess = await initLatentDemucs();
  const ort = _ort;
  // Serialize across any overlapping callers.
  const work = _runQueue.then(() => _separate(sess, ort, flat, numFrames, chunkSamples));
  _runQueue = work.catch(() => undefined);
  return work;
}

async function _separate(sess, ort, flat, numFrames, chunkSamples) {

  // Ensure numFrames is a multiple of FRAME_SAMPLES; pad with zeros if not.
  const padded = numFrames % FRAME_SAMPLES === 0
    ? numFrames
    : Math.ceil(numFrames / FRAME_SAMPLES) * FRAME_SAMPLES;
  let src;
  if (padded === numFrames) {
    src = flat;
  } else {
    src = new Float32Array(2 * padded);
    src.set(flat.subarray(0, numFrames), 0);
    src.set(flat.subarray(numFrames, 2 * numFrames), padded);
    // zeros already in the gaps by default
  }

  // chunk along time
  const totalFramesLat = padded / FRAME_SAMPLES; // latent frames
  // Prepare per-stem output buffers [T_lat, 64]
  const outs = [0,1,2,3].map(() => new Float32Array(totalFramesLat * LATENT_CHANNELS));

  // Round chunkSamples to a multiple of FRAME_SAMPLES so outputs line up.
  const chunkAligned = Math.floor(chunkSamples / FRAME_SAMPLES) * FRAME_SAMPLES;
  let latOff = 0;
  for (let s = 0; s < padded; s += chunkAligned) {
    const e = Math.min(s + chunkAligned, padded);
    const N = e - s;                        // samples in this chunk
    const latN = N / FRAME_SAMPLES;         // latent frames in this chunk
    // Slice stereo channels-first
    const chunkFlat = new Float32Array(2 * N);
    chunkFlat.set(src.subarray(s, s + N), 0);
    chunkFlat.set(src.subarray(padded + s, padded + s + N), N);
    const input = new ort.Tensor('float32', chunkFlat, [1, 2, N]);
    const out = await ortWebGPURun(() => sess.run({ waveform: input }));
    // output name is "stem_latents", shape [1, 4, 64, latN]
    const t = out.stem_latents || out[Object.keys(out)[0]];
    const raw = t.data; // Float32Array length 1*4*64*latN in [B, S, C, T] order
    // Convert each stem to [T, 64] time-major and copy into outs
    for (let si = 0; si < 4; si++) {
      const baseIn = si * LATENT_CHANNELS * latN;   // start of stem si in ONNX output
      const buf = outs[si];
      for (let t2 = 0; t2 < latN; t2++) {
        const outBase = (latOff + t2) * LATENT_CHANNELS;
        for (let c = 0; c < LATENT_CHANNELS; c++) {
          buf[outBase + c] = raw[baseIn + c * latN + t2];
        }
      }
    }
    latOff += latN;
  }

  // Build {name: flatTD} dict
  const stems = {};
  for (let i = 0; i < 4; i++) {
    stems[STEM_NAMES[i]] = { flatTD: outs[i], T: totalFramesLat, D: LATENT_CHANNELS };
  }
  return stems;
}

/** Get the backend's VAE version hash (needed for DOAE header). Cached. */
export async function getVaeVersion() {
  if (_vaeVersion) return _vaeVersion;
  const r = await fetch('/api/vae-version');
  if (!r.ok) throw new Error(`/api/vae-version HTTP ${r.status}`);
  const d = await r.json();
  _vaeVersion = d.vae_version;
  return _vaeVersion;
}

/**
 * Serialize a [T, 64] latent as DOAE binary (matches the backend format
 * parsed by _parse_doae in stemphonic_server.py).
 *
 * Header (28 bytes LE):
 *   0..3   magic "DOAE"
 *   4..5   uint16 version = 1
 *   6..17  vae_hash (12 ASCII chars, null-padded)
 *   18..19 uint16 fps
 *   20..23 uint32 T
 *   24..27 uint32 D
 * Body: T*D float32 LE (time-major)
 */
export function serializeDoae(flatTD, T, D, fps, vaeHash) {
  const header = 28;
  const bodyBytes = T * D * 4;
  const out = new ArrayBuffer(header + bodyBytes);
  const view = new DataView(out);
  view.setUint8(0, 68); view.setUint8(1, 79); view.setUint8(2, 65); view.setUint8(3, 69); // "DOAE"
  view.setUint16(4, 1, true);
  // 12 bytes of hash, ASCII, zero-padded
  const hashBytes = new TextEncoder().encode(vaeHash || '');
  for (let i = 0; i < 12; i++) {
    view.setUint8(6 + i, i < hashBytes.length ? hashBytes[i] : 0);
  }
  view.setUint16(18, fps, true);
  view.setUint32(20, T, true);
  view.setUint32(24, D, true);
  // body (float32 LE)
  new Float32Array(out, header).set(flatTD);
  return out;
}

/**
 * POST a latent to the backend, get back a latent_id the backend can
 * use for downstream operations. Backend never sees audio.
 */
export async function uploadLatent(flatTD, T, D = LATENT_CHANNELS, fps = 25) {
  const vaeHash = await getVaeVersion();
  const body = serializeDoae(flatTD, T, D, fps, vaeHash);
  const r = await fetch('/api/upload-latent', {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body,
  });
  if (!r.ok) {
    const errText = await r.text();
    throw new Error(`/api/upload-latent HTTP ${r.status}: ${errText}`);
  }
  return r.json(); // { latent_id, ... }
}
