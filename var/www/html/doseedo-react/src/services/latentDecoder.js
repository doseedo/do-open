/**
 * Client-side VAE decoder for stemphonic latents.
 *
 * Architecture:
 *   /api/latent/<id>  →  DOAE binary  →  parseDoae  →  Float32Array [T*64]
 *   Float32Array      →  onnxruntime-web  →  Float32Array [T*1920*2]  (stereo @ 48kHz)
 *   Float32Array      →  wavFromPCM  →  Blob  →  blob:URL (playable by wavesurfer/<audio>)
 *
 * Lazy-loaded on first use. Model (~323 MB) is downloaded once and
 * cached by the browser for the session. Subsequent decodes reuse the
 * loaded ORT session.
 */

const SAMPLE_RATE = 48000;
const FRAME_SAMPLES = 1920;        // 48000 / 25 fps
const LATENT_CHANNELS = 64;

// ── Web Worker for ONNX inference (keeps main thread responsive) ──
let _worker = null;
let _workerReady = false;
let _workerInitPromise = null;
let _pendingDecodes = new Map();   // id → { resolve, reject }
let _nextId = 0;
let _loadProgress = null;

/**
 * Start the decoder Web Worker. Safe to call multiple times — returns
 * the same promise. The worker downloads the ONNX model, creates the
 * ORT session, and handles all sess.run() calls off the main thread.
 */
export async function initLatentDecoder(onProgress = null) {
  if (_workerReady) return;
  if (_workerInitPromise) return _workerInitPromise;

  _workerInitPromise = new Promise((resolve, reject) => {
    _worker = new Worker(
      new URL('./decoderWorker.js', import.meta.url),
      { type: 'module' },
    );

    _worker.onmessage = (e) => {
      const msg = e.data;

      if (msg.type === 'progress' && onProgress) {
        _loadProgress = {
          bytesLoaded: parseInt(msg.mb, 10) * 1e6,
          bytesTotal: 63e6, // sem_encoder (22MB) + sem_decoder (41MB)
        };
        onProgress(_loadProgress);
        return;
      }

      if (msg.type === 'initDone') {
        _workerReady = true;
        console.log(`[latentDecoder] worker ready on ${msg.backend}`);
        resolve();
        return;
      }

      if (msg.type === 'initFailed') {
        reject(new Error(msg.error));
        return;
      }

      if (msg.type === 'decoded') {
        const pending = _pendingDecodes.get(msg.id);
        if (pending) {
          _pendingDecodes.delete(msg.id);
          pending.resolve(new Float32Array(msg.audio));
        }
        return;
      }

      if (msg.type === 'decodeFailed') {
        const pending = _pendingDecodes.get(msg.id);
        if (pending) {
          _pendingDecodes.delete(msg.id);
          pending.reject(new Error(msg.error));
        }
        return;
      }

      if (msg.type === 'envelopeDone') {
        const pending = _pendingDecodes.get(msg.id);
        if (pending) {
          _pendingDecodes.delete(msg.id);
          pending.resolve(new Float32Array(msg.envelope));
        }
        return;
      }

      if (msg.type === 'envelopeFailed') {
        const pending = _pendingDecodes.get(msg.id);
        if (pending) {
          _pendingDecodes.delete(msg.id);
          pending.reject(new Error(msg.error));
        }
        return;
      }
    };

    _worker.onerror = (err) => {
      reject(new Error(err.message || 'worker error'));
    };

    _worker.postMessage({ type: 'init' });
  });

  return _workerInitPromise;
}

export function getLoadProgress() { return _loadProgress; }

/**
 * Send a decode request to the worker and return a promise for the result.
 * The input Float32Array is TRANSFERRED (zero-copy) to the worker.
 */
/**
 * Run latent_visual in the worker: [T, 64] → [2, T] envelope.
 * Returns Float32Array of length 2*T (first T mins, then T maxes).
 */
export function workerEnvelope(flatTD, T) {
  return new Promise((resolve, reject) => {
    const id = _nextId++;
    _pendingDecodes.set(id, { resolve, reject });
    const buf = flatTD.buffer.slice(
      flatTD.byteOffset,
      flatTD.byteOffset + flatTD.byteLength,
    );
    _worker.postMessage({ type: 'envelope', id, input: buf, T }, [buf]);
  });
}

function _workerDecode(inputFloat32, shape) {
  return new Promise((resolve, reject) => {
    const id = _nextId++;
    _pendingDecodes.set(id, { resolve, reject });
    const buf = inputFloat32.buffer.slice(
      inputFloat32.byteOffset,
      inputFloat32.byteOffset + inputFloat32.byteLength,
    );
    _worker.postMessage(
      { type: 'decode', id, input: buf, shape },
      [buf],
    );
  });
}

/** Build a .doae binary from a Float32Array [T*D] time-major latent.
 *  Header (28 bytes LE): magic="DOAE"(4) version=u16 vaeHash(12) fps=u16 T=u32 D=u32
 */
export function buildDoae(flatTD, T, D = 64, fps = 25, vaeHash = '') {
  const body = flatTD instanceof ArrayBuffer ? flatTD : flatTD.buffer.slice(flatTD.byteOffset, flatTD.byteOffset + flatTD.byteLength);
  const header = new ArrayBuffer(28);
  const hv = new DataView(header);
  // magic
  hv.setUint8(0, 68); hv.setUint8(1, 79); hv.setUint8(2, 65); hv.setUint8(3, 69); // "DOAE"
  hv.setUint16(4, 1, true); // version
  // vaeHash (12 bytes, zero-padded)
  for (let i = 0; i < 12; i++) {
    hv.setUint8(6 + i, i < vaeHash.length ? vaeHash.charCodeAt(i) : 0);
  }
  hv.setUint16(18, fps, true);
  hv.setUint32(20, T, true);
  hv.setUint32(24, D, true);
  // Combine header + body
  const out = new Uint8Array(28 + T * D * 4);
  out.set(new Uint8Array(header), 0);
  out.set(new Uint8Array(body), 28);
  return out.buffer;
}

/**
 * Parse a .doae binary blob produced by stemphonic_server._serialize_latent_doae.
 * Header (28 bytes LE): magic="DOAE"(4) version=u16 vaeHash(12) fps=u16 T=u32 D=u32
 * Body: T*D float32.
 */
export function parseDoae(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  if (view.byteLength < 28) throw new Error('DOAE blob too small');
  const magic = String.fromCharCode(
    view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3),
  );
  if (magic !== 'DOAE') throw new Error(`DOAE magic mismatch: ${magic}`);
  const version = view.getUint16(4, true);
  let vaeHash = '';
  for (let i = 6; i < 18; i++) {
    const c = view.getUint8(i);
    if (c === 0) break;
    vaeHash += String.fromCharCode(c);
  }
  const fps = view.getUint16(18, true);
  const T = view.getUint32(20, true);
  const D = view.getUint32(24, true);
  const expected = 28 + T * D * 4;
  if (view.byteLength !== expected) {
    throw new Error(`DOAE length mismatch: header says ${expected}, got ${view.byteLength}`);
  }
  if (D !== LATENT_CHANNELS) {
    throw new Error(`DOAE channel count unexpected: got D=${D}, expected ${LATENT_CHANNELS}`);
  }
  // Copy body to a plain Float32Array. The .pt pipeline stores [T, D]
  // (time-major); the ONNX decoder expects [1, D, T] (channels-first).
  // Transpose below.
  const flatTD = new Float32Array(arrayBuffer, 28, T * D);
  return { version, vaeHash, fps, T, D, flatTD };
}

/** Transpose [T,D] (row-major) into [D,T] (row-major) for ONNX input. */
function transposeTDToDT(flatTD, T, D) {
  const out = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < D; d++) {
      out[d * T + t] = flatTD[t * D + d];
    }
  }
  return out;
}

/**
 * Run the ONNX decoder on a latent. Returns a Float32Array of interleaved
 * audio [L*2] at 48kHz, where L = T_frames * 1920.
 *
 * chunkFrames: if > 0, splits long latents into chunks of this many frames
 *   to bound memory usage during decode. Default 256 frames (~10s audio).
 */
export async function decodeLatent(flatTD, T, chunkFrames = 256) {
  await initLatentDecoder();

  if (chunkFrames <= 0 || T <= chunkFrames) {
    return await decodeLatentFrameRange(flatTD, T, 0, T);
  }
  // Chunked: decode non-overlapping windows, concat.
  // Each chunk goes to the worker sequentially (worker handles one
  // at a time, but main thread is free between chunks).
  const chunks = [];
  for (let s = 0; s < T; s += chunkFrames) {
    const e = Math.min(s + chunkFrames, T);
    chunks.push(await decodeLatentFrameRange(flatTD, T, s, e));
  }
  const total = chunks.reduce((a, c) => a + c.length, 0);
  const out = new Float32Array(total);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

/**
 * ONNX output is [1, 2, L] channels-first (non-interleaved).
 * Web Audio wants interleaved [L0 R0 L1 R1 ...] OR separate channel buffers.
 * Use AudioBuffer.copyToChannel with two Float32Array slices.
 */
function audioBufferFromDecoderOutput(out, audioCtx) {
  const lenPerCh = out.length / 2;
  const buf = audioCtx.createBuffer(2, lenPerCh, SAMPLE_RATE);
  buf.copyToChannel(out.subarray(0, lenPerCh), 0);
  buf.copyToChannel(out.subarray(lenPerCh, 2 * lenPerCh), 1);
  return buf;
}

/**
 * Decode a specific frame range of a [T, 64] latent via the shared
 * ORT session (serialized through _runQueue).
 *
 * flatTD: Float32Array [T*64] time-major
 * totalT: total frames in flatTD
 * startFrame, endFrame: the range to decode (non-overlapping chunk)
 *
 * Returns Float32Array of length 2 * (endFrame - startFrame) * 1920
 * in channels-first layout: [ch0_0..ch0_N, ch1_0..ch1_N].
 */
export async function decodeLatentFrameRange(flatTD, totalT, startFrame, endFrame) {
  await initLatentDecoder();
  const subT = endFrame - startFrame;
  const subSlice = new Float32Array(subT * LATENT_CHANNELS);
  subSlice.set(flatTD.subarray(startFrame * LATENT_CHANNELS, endFrame * LATENT_CHANNELS));
  const dt = transposeTDToDT(subSlice, subT, LATENT_CHANNELS);
  return _workerDecode(dt, [1, LATENT_CHANNELS, subT]);
}

/**
 * Batched frame-range decode.
 *
 * Decodes the same frame range across multiple stems in ONE ONNX pass
 * by stacking them into a [B, 64, subT] tensor. Avoids per-call kernel
 * launch overhead and keeps the weights hot in VRAM — typically 1.5-3×
 * faster than 4 sequential batch=1 calls on WebGPU.
 *
 * stems: array of { flatTD: Float32Array, totalT: number }
 * startFrame, endFrame: range to decode (same for all stems)
 *
 * Returns: array of Float32Arrays, one per stem, each of length
 * 2 * (endFrame - startFrame) * 1920 in channels-first layout.
 * Order matches the input `stems` array.
 */
export async function decodeLatentFrameRangeBatched(stems, startFrame, endFrame) {
  await initLatentDecoder();
  const B = stems.length;
  if (B === 0) return [];
  if (B === 1) {
    const only = await decodeLatentFrameRange(stems[0].flatTD, stems[0].totalT, startFrame, endFrame);
    return [only];
  }
  const subT = endFrame - startFrame;
  const perBatch = subT * LATENT_CHANNELS;
  const batched = new Float32Array(B * perBatch);
  for (let b = 0; b < B; b++) {
    const slice = new Float32Array(perBatch);
    slice.set(stems[b].flatTD.subarray(startFrame * LATENT_CHANNELS, endFrame * LATENT_CHANNELS));
    for (let t = 0; t < subT; t++) {
      for (let d = 0; d < LATENT_CHANNELS; d++) {
        batched[b * perBatch + d * subT + t] = slice[t * LATENT_CHANNELS + d];
      }
    }
  }
  // Send to worker — returns the full [B, 2, subT*1920] flat array
  const all = await _workerDecode(batched, [B, LATENT_CHANNELS, subT]);
  const samplesPerCh = subT * 1920;
  const perStemFlat = 2 * samplesPerCh;
  const result = [];
  for (let b = 0; b < B; b++) {
    const copy = new Float32Array(perStemFlat);
    copy.set(all.subarray(b * perStemFlat, (b + 1) * perStemFlat));
    result.push(copy);
  }
  return result;
}

/** Concat two channels-first stereo Float32Arrays (layout: [L0..Ln, R0..Rn]). */
export function concatChannelsFirstStereo(a, aLen, b, bLen) {
  // aLen, bLen are per-channel sample counts
  const merged = new Float32Array(2 * (aLen + bLen));
  merged.set(a.subarray(0, aLen),                      0);                   // L: a
  merged.set(b.subarray(0, bLen),                      aLen);                // L: b
  merged.set(a.subarray(aLen, 2 * aLen),               aLen + bLen);         // R: a
  merged.set(b.subarray(bLen, 2 * bLen),               2 * aLen + bLen);     // R: b
  return merged;
}

/** Convert channels-first stereo Float32Array into a playable WAV blob URL. */
export function audioDataToBlobUrl(audio, numSamplesPerCh, audioCtx) {
  const buf = audioCtx.createBuffer(2, numSamplesPerCh, SAMPLE_RATE);
  buf.copyToChannel(audio.subarray(0, numSamplesPerCh), 0);
  buf.copyToChannel(audio.subarray(numSamplesPerCh, 2 * numSamplesPerCh), 1);
  const wavBytes = wavFromAudioBuffer(buf);
  const blob = new Blob([wavBytes], { type: 'audio/wav' });
  return { blobUrl: URL.createObjectURL(blob), audioBuffer: buf };
}

/** End-to-end: latent_id → playable blob URL. */
export async function latentIdToBlobUrl(latentId, audioCtx) {
  const resp = await fetch(`/api/latent/${latentId}`);
  if (!resp.ok) throw new Error(`fetch /api/latent/${latentId} HTTP ${resp.status}`);
  const buf = await resp.arrayBuffer();
  const { flatTD, T } = parseDoae(buf);
  const audio = await decodeLatent(flatTD, T);
  const audioBuffer = audioBufferFromDecoderOutput(audio, audioCtx);
  const wavBytes = wavFromAudioBuffer(audioBuffer);
  const blob = new Blob([wavBytes], { type: 'audio/wav' });
  return URL.createObjectURL(blob);
}

/** Minimal 32-bit-float WAV encoder — no compression, no resampling. */
function wavFromAudioBuffer(audioBuffer) {
  const numCh = audioBuffer.numberOfChannels;
  const numFrames = audioBuffer.length;
  const sr = audioBuffer.sampleRate;
  const bytesPerSample = 4; // float32
  const blockAlign = numCh * bytesPerSample;
  const dataBytes = numFrames * blockAlign;
  const header = 44;
  const out = new ArrayBuffer(header + dataBytes);
  const view = new DataView(out);

  function writeStr(off, s) { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); }

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);            // fmt chunk size
  view.setUint16(20, 3, true);             // format = IEEE float
  view.setUint16(22, numCh, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bytesPerSample * 8, true);
  writeStr(36, 'data');
  view.setUint32(40, dataBytes, true);

  // Interleave channels
  const chans = [];
  for (let c = 0; c < numCh; c++) chans.push(audioBuffer.getChannelData(c));
  let off = header;
  for (let i = 0; i < numFrames; i++) {
    for (let c = 0; c < numCh; c++) {
      view.setFloat32(off, chans[c][i], true);
      off += 4;
    }
  }
  return new Uint8Array(out);
}
