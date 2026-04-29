/**
 * audioEncode — client-side Opus 128 encoder.
 *
 * Two backends, picked at runtime:
 *
 *   1. MediaRecorder ({ mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: 128_000 })
 *      Available everywhere except old Safari. Fast (real-time-ish), no extra
 *      bytes shipped. Output is Opus-in-WebM rather than Opus-in-OGG, but
 *      decodeAudioData handles both and the server doesn't care — same
 *      payload, smaller container header.
 *
 *   2. WASM libopus fallback. We stream the AudioBuffer through libopusjs in
 *      a Worker. Used for: Safari without MediaRecorder + Opus, and the
 *      "encode an already-rendered AudioBuffer" path (the export-mix flow).
 *
 * Public API is a single async function that takes either a File or an
 * AudioBuffer and returns a Blob ready to ship.
 */

const OPUS_BITRATE = 128_000;

// ---------------------------------------------------------------------------
// Capability detection
// ---------------------------------------------------------------------------

function _hasMediaRecorderOpus() {
  if (typeof MediaRecorder === 'undefined') return false;
  try {
    return (
      MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ||
      MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')
    );
  } catch {
    return false;
  }
}

function _pickRecorderMime() {
  if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
    return 'audio/ogg;codecs=opus';
  }
  return 'audio/webm;codecs=opus';
}

// ---------------------------------------------------------------------------
// Backend A: stream a File / AudioBuffer through MediaRecorder via an
// OfflineAudioContext-fed MediaStreamDestination.
// ---------------------------------------------------------------------------

async function _encodeViaMediaRecorder(audioBuffer) {
  const mimeType = _pickRecorderMime();
  const ctx = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: audioBuffer.sampleRate,
  });
  const dest = ctx.createMediaStreamDestination();
  const src = ctx.createBufferSource();
  src.buffer = audioBuffer;
  src.connect(dest);

  const chunks = [];
  const recorder = new MediaRecorder(dest.stream, {
    mimeType,
    audioBitsPerSecond: OPUS_BITRATE,
  });
  recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };

  return new Promise((resolve, reject) => {
    recorder.onstop = () => {
      try {
        const blob = new Blob(chunks, { type: mimeType.split(';')[0] });
        ctx.close().catch(() => {});
        resolve(blob);
      } catch (e) { reject(e); }
    };
    recorder.onerror = (e) => reject(new Error(`MediaRecorder error: ${e?.error?.message || e}`));
    recorder.start();
    src.onended = () => {
      // give the encoder one last frame to flush
      setTimeout(() => recorder.state !== 'inactive' && recorder.stop(), 50);
    };
    src.start();
    // Hard cap: don't run longer than buffer + 200ms
    setTimeout(() => recorder.state !== 'inactive' && recorder.stop(),
      (audioBuffer.duration * 1000) + 250);
  });
}

// ---------------------------------------------------------------------------
// Backend B: WASM libopus fallback (lazy-loaded only if Backend A is
// unavailable). The encoder library itself is ~150 KB gz.
// ---------------------------------------------------------------------------

let _libopusPromise = null;
async function _loadLibopus() {
  if (_libopusPromise) return _libopusPromise;
  _libopusPromise = (async () => {
    // Lazy dynamic import — bundler tree-shakes this out for browsers that
    // never hit the fallback path.
    const mod = await import(/* webpackChunkName: "libopus" */ 'opus-recorder/dist/encoderWorker.min.js');
    return mod;
  })().catch((e) => {
    _libopusPromise = null;
    throw e;
  });
  return _libopusPromise;
}

async function _encodeViaWasm(audioBuffer) {
  await _loadLibopus();
  // Minimal opus-recorder shim: write interleaved float32 frames into the
  // worker, get back .opus bytes. The library is wide-spread; the API:
  //
  //   const enc = new Encoder({ encoderSampleRate: 48000, originalSampleRate: ..., numberOfChannels });
  //   enc.encode([Float32Array(left), Float32Array(right)]) per frame
  //   enc.done() -> ArrayBuffer
  //
  // We keep this implementation small — it's the cold path.
  const { default: Encoder } = await import('opus-recorder');
  const enc = new Encoder({
    encoderApplication: 2049,           // VOIP=2048, AUDIO=2049, LOWDELAY=2051
    encoderBitRate: OPUS_BITRATE,
    encoderSampleRate: 48_000,
    originalSampleRate: audioBuffer.sampleRate,
    numberOfChannels: Math.min(2, audioBuffer.numberOfChannels),
    streamPages: false,
    monitorGain: 0,
    rawOpus: false,
  });

  const channels = [];
  const ch = Math.min(2, audioBuffer.numberOfChannels);
  for (let c = 0; c < ch; c++) channels.push(audioBuffer.getChannelData(c));
  enc.encode(channels);
  const blob = await enc.done();
  return new Blob([blob], { type: 'audio/ogg' });
}

// ---------------------------------------------------------------------------
// Decode helper — turn a File into an AudioBuffer.
// ---------------------------------------------------------------------------

async function _fileToAudioBuffer(file) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const ab = await file.arrayBuffer();
  const buf = await ctx.decodeAudioData(ab.slice(0));
  ctx.close().catch(() => {});
  return buf;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Encode any File or AudioBuffer to Opus 128 kbps.
 * @param {File|Blob|AudioBuffer} input
 * @returns {Promise<{ blob: Blob, mime: string, bitrate: number }>}
 */
export async function encodeOpus128(input) {
  let buf;
  if (input instanceof AudioBuffer) {
    buf = input;
  } else if (input instanceof Blob || input instanceof File) {
    buf = await _fileToAudioBuffer(input);
  } else {
    throw new TypeError('encodeOpus128: expected File, Blob, or AudioBuffer');
  }

  if (_hasMediaRecorderOpus()) {
    try {
      const blob = await _encodeViaMediaRecorder(buf);
      return { blob, mime: blob.type || 'audio/ogg', bitrate: OPUS_BITRATE };
    } catch (e) {
      console.warn('[audioEncode] MediaRecorder failed, falling back to WASM:', e?.message);
    }
  }
  const blob = await _encodeViaWasm(buf);
  return { blob, mime: 'audio/ogg', bitrate: OPUS_BITRATE };
}

/**
 * Sniff a File's MIME type. Browsers leave .wav/.flac with the right type
 * already; this just normalizes a few aliases the server expects.
 */
export function inferAudioMime(file) {
  const t = (file.type || '').toLowerCase();
  if (t === 'audio/x-wav' || t === 'audio/wave') return 'audio/wav';
  if (t === 'audio/x-aiff') return 'audio/aiff';
  if (t === 'audio/x-flac') return 'audio/flac';
  return t || 'application/octet-stream';
}

/**
 * Decide whether to encode-to-Opus before upload, given a user tier and a
 * "preserve lossless" UI toggle. Free + Pro: always encode. Pro+: respect
 * the toggle (default: encode unless user explicitly chose lossless).
 */
export function shouldEncodeForTier(tier, preserveLossless = false) {
  if (tier === 'pro_plus' || tier === 'b2b' || tier === 'admin') {
    return !preserveLossless;
  }
  return true;
}
