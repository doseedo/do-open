/**
 * latentPlayer.js — fetch latents from the backend and decode them in
 * the browser via WebGPU instead of pulling wav files.
 *
 * Backend pairs:
 *   GET /api/vae-version           → { vae_version: "ae1c12f2d5a5" }
 *   GET /api/latent/<id>           → application/x-doae binary blob
 *   GET /api/onnx/<file>           → packed Oobleck VAE ONNX bundle
 *   POST /api/upload-latent        → body = .doae bytes, returns { latent_id }
 *
 * Wire format (.doae, little-endian):
 *   bytes  0..3   magic = "DOAE"
 *   bytes  4..5   uint16 version (currently 1)
 *   bytes  6..17  vae_version_hash (12 ascii chars)
 *   bytes 18..19  uint16 fps (25 for Oobleck)
 *   bytes 20..23  uint32 T  (number of latent frames)
 *   bytes 24..27  uint32 D  (channels per frame, 64 for Oobleck)
 *   bytes 28..    T*D float32 little-endian
 *
 * Browser flow:
 *   const player = new LatentPlayer({ audioContext });
 *   await player.init();                                  // loads decoder ONNX
 *   const buf = await player.fetchAndDecode(latent_id);   // → AudioBuffer
 *   const src = audioContext.createBufferSource();
 *   src.buffer = buf;
 *   src.connect(audioContext.destination);
 *   src.start();
 *
 * Falls back to legacy wav fetch when WebGPU isn't available so the
 * studio still works on Safari/Firefox/older browsers.
 */

const SAMPLES_PER_FRAME = 1920;
const SAMPLE_RATE = 48000;
const LATENT_CHANNELS = 64;
const DOAE_HEADER_BYTES = 28;

let _ortPromise = null;
async function _loadOrt() {
  if (_ortPromise) return _ortPromise;
  _ortPromise = (async () => {
    // ONNX Runtime Web — assumed available as `ort` global, or via
    // dynamic import. Adjust to your bundler setup.
    if (typeof window !== "undefined" && window.ort) return window.ort;
    return await import(/* webpackIgnore: true */ "onnxruntime-web");
  })();
  return _ortPromise;
}

export function isWebGPUSupported() {
  return typeof navigator !== "undefined" && !!navigator.gpu;
}

/**
 * Parse a .doae binary blob.
 * @param {ArrayBuffer} buf
 * @returns {{ vaeVersion: string, fps: number, T: number, D: number, latents: Float32Array }}
 */
export function parseDoae(buf) {
  const view = new DataView(buf);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== "DOAE") throw new Error(`bad magic: ${magic}`);
  const version = view.getUint16(4, true);
  if (version !== 1) throw new Error(`unsupported .doae version ${version}`);
  let vaeVersion = "";
  for (let i = 6; i < 18; i++) {
    const c = view.getUint8(i);
    if (c === 0) break;
    vaeVersion += String.fromCharCode(c);
  }
  const fps = view.getUint16(18, true);
  const T = view.getUint32(20, true);
  const D = view.getUint32(24, true);
  if (D !== LATENT_CHANNELS) throw new Error(`expected ${LATENT_CHANNELS} channels, got ${D}`);
  const expected = DOAE_HEADER_BYTES + T * D * 4;
  if (buf.byteLength !== expected)
    throw new Error(`length mismatch: header says ${expected}, body is ${buf.byteLength}`);
  const latents = new Float32Array(buf, DOAE_HEADER_BYTES, T * D);
  return { vaeVersion, fps, T, D, latents };
}

export class LatentPlayer {
  constructor({
    audioContext,
    decoderUrl       = "/api/onnx/oobleck_decoder_packed.onnx",
    encoderUrl       = "/api/onnx/oobleck_encoder_packed.onnx",
    latentDemucsUrl  = "/api/onnx/latent_demucs_student_packed.onnx",
    chunkFrames = 64,           // ~2.5s of audio per ORT call
    fallbackToWav = true,
  } = {}) {
    this.audioContext = audioContext || (typeof AudioContext !== "undefined" ? new AudioContext({ sampleRate: SAMPLE_RATE }) : null);
    this.decoderUrl = decoderUrl;
    this.encoderUrl = encoderUrl;
    this.latentDemucsUrl = latentDemucsUrl;
    this.chunkFrames = chunkFrames;
    this.fallbackToWav = fallbackToWav;
    this.decoderSession = null;
    this.encoderSession = null;
    this.latentDemucsSession = null;
    this.expectedVaeVersion = null;
    this.useWebGPU = isWebGPUSupported();
  }

  async init() {
    // Pin the VAE version the backend is currently serving so we can
    // reject any latent whose .doae header doesn't match.
    try {
      const r = await fetch("/api/vae-version");
      if (r.ok) {
        const j = await r.json();
        this.expectedVaeVersion = j.vae_version;
        console.log("[latentPlayer] backend vae_version =", this.expectedVaeVersion);
      }
    } catch (e) {
      console.warn("[latentPlayer] could not fetch /api/vae-version:", e);
    }

    if (!this.useWebGPU) {
      console.warn("[latentPlayer] WebGPU not supported, will fall back to wav");
      return;
    }
    const ort = await _loadOrt();
    const opts = {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
    };
    try {
      const buf = await fetch(this.decoderUrl).then(r => r.arrayBuffer());
      this.decoderSession = await ort.InferenceSession.create(buf, opts);
      console.log("[latentPlayer] WebGPU decoder ready");
    } catch (e) {
      console.error("[latentPlayer] WebGPU decoder load failed:", e);
      this.useWebGPU = false;
    }
  }

  async _ensureEncoder() {
    if (this.encoderSession) return this.encoderSession;
    if (!this.useWebGPU) throw new Error("WebGPU not available for encoder");
    const ort = await _loadOrt();
    const buf = await fetch(this.encoderUrl).then(r => r.arrayBuffer());
    this.encoderSession = await ort.InferenceSession.create(buf, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
    });
    return this.encoderSession;
  }

  async _ensureLatentDemucs() {
    if (this.latentDemucsSession) return this.latentDemucsSession;
    if (!this.useWebGPU) throw new Error("WebGPU not available for latent_demucs");
    const ort = await _loadOrt();
    const buf = await fetch(this.latentDemucsUrl).then(r => r.arrayBuffer());
    this.latentDemucsSession = await ort.InferenceSession.create(buf, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
    });
    console.log("[latentPlayer] WebGPU latent_demucs student ready");
    return this.latentDemucsSession;
  }

  /**
   * Run the latent_demucs student LOCALLY in the browser. Takes an
   * AudioBuffer (any sample rate, mono or stereo) and returns a dict
   * { drums, bass, vocals, other } where each value is the parsed
   * latent ready to feed into `_decodeLatentTensor` for playback.
   *
   * No backend round-trip — separation happens entirely on the user's
   * GPU. The browser never uploads the source audio.
   *
   * @param {AudioBuffer} audioBuffer
   * @returns {Promise<Record<string, { latents: Float32Array, T: number, D: number }>>}
   */
  async separateLocally(audioBuffer) {
    if (!this.useWebGPU) {
      throw new Error("separateLocally requires WebGPU. Fall back to /separate-stems.");
    }
    const sess = await this._ensureLatentDemucs();
    const ort = await _loadOrt();

    // Convert to 48k stereo, pad to a multiple of 1920
    const stereo = await this._toStereo48k(audioBuffer);
    let n = stereo[0].length;
    const padded = Math.ceil(n / SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME;
    const interleaved = new Float32Array(2 * padded);
    interleaved.set(stereo[0]);
    interleaved.set(stereo[1], padded);
    const inputTensor = new ort.Tensor("float32", interleaved, [1, 2, padded]);

    const t0 = performance.now();
    const out = await sess.run({ audio: inputTensor });
    const elapsed = performance.now() - t0;
    const stems = out.stem_latents || out[Object.keys(out)[0]];
    // Output: [1, 4, 64, T]
    const dims = stems.dims;     // [1, 4, 64, T]
    const data = stems.data;     // Float32Array length = 4 * 64 * T
    const N = dims[1];           // 4 stems
    const D = dims[2];           // 64 channels
    const T = dims[3];           // frames
    console.log(`[latentPlayer] latent_demucs ${(padded/SAMPLE_RATE).toFixed(1)}s in ${elapsed.toFixed(0)}ms → ${T} frames`);

    // Slice each stem out as a row-major [T, D] Float32Array so it
    // matches the .doae body layout that _decodeLatentTensor expects.
    const result = {};
    for (let s = 0; s < N; s++) {
      const stemName = LATENT_DEMUCS_STEMS[s];
      const rowMajor = new Float32Array(T * D);
      // source layout is [N, D, T] (channels-first per stem). Reshape to row-major [T, D].
      for (let t = 0; t < T; t++) {
        for (let d = 0; d < D; d++) {
          rowMajor[t * D + d] = data[((s * D) + d) * T + t];
        }
      }
      result[stemName] = { latents: rowMajor, T, D };
    }
    return result;
  }

  /**
   * Convenience: run separateLocally + immediately decode each stem to
   * an AudioBuffer. Returns { drums: AudioBuffer, bass: ..., vocals: ..., other: ... }.
   * The full chain is local — no network calls beyond the one-time
   * model fetch.
   */
  async separateAndDecodeLocally(audioBuffer) {
    const stems = await this.separateLocally(audioBuffer);
    const out = {};
    for (const [name, { latents, T, D }] of Object.entries(stems)) {
      out[name] = await this._decodeLatentTensor(latents, T, D);
    }
    return out;
  }

  /**
   * Fetch a latent_id from the backend and decode to an AudioBuffer.
   * Falls back to /api/generate-stemphonic/download/<id> wav if WebGPU
   * is unavailable.
   */
  async fetchAndDecode(latentId) {
    if (!this.useWebGPU || !this.decoderSession) {
      if (!this.fallbackToWav) throw new Error("WebGPU unavailable and fallback disabled");
      return this._fetchAndDecodeWav(latentId);
    }
    const r = await fetch(`/api/latent/${latentId}`);
    if (!r.ok) throw new Error(`/api/latent/${latentId} → ${r.status}`);
    const buf = await r.arrayBuffer();
    const { vaeVersion, fps, T, D, latents } = parseDoae(buf);
    if (this.expectedVaeVersion && vaeVersion !== this.expectedVaeVersion) {
      throw new Error(
        `vae_version mismatch — backend has ${this.expectedVaeVersion}, latent says ${vaeVersion}. ` +
        `The browser decoder is pinned to a stale VAE; reload to refresh.`
      );
    }
    return this._decodeLatentTensor(latents, T, D);
  }

  /**
   * Decode a [T, D] Float32Array (row-major) → AudioBuffer.
   * Chunked through ORT so a single forward pass fits typical GPU
   * memory budgets even on integrated chips.
   */
  async _decodeLatentTensor(latents, T, D) {
    const ort = await _loadOrt();
    const totalSamples = T * SAMPLES_PER_FRAME;
    const ab = this.audioContext.createBuffer(2, totalSamples, SAMPLE_RATE);
    let cursor = 0;
    for (let i = 0; i < T; i += this.chunkFrames) {
      const end = Math.min(T, i + this.chunkFrames);
      const chunkT = end - i;
      // Decoder expects [B=1, D=64, T_chunk]. Source is row-major [T, D].
      const chunkData = new Float32Array(D * chunkT);
      for (let t = 0; t < chunkT; t++) {
        for (let d = 0; d < D; d++) {
          chunkData[d * chunkT + t] = latents[(i + t) * D + d];
        }
      }
      const tensor = new ort.Tensor("float32", chunkData, [1, D, chunkT]);
      const out = await this.decoderSession.run({ latent: tensor });
      // Output: [1, 2, S_chunk]
      const audio = out.audio || out[Object.keys(out)[0]];
      const audioData = audio.data;          // Float32Array length = 2*S
      const sChunk = audio.dims[2];
      // De-interleave channels from [2, S] layout
      const left = ab.getChannelData(0);
      const right = ab.getChannelData(1);
      for (let s = 0; s < sChunk && cursor + s < totalSamples; s++) {
        left[cursor + s]  = audioData[0 * sChunk + s];
        right[cursor + s] = audioData[1 * sChunk + s];
      }
      cursor += sChunk;
    }
    return ab;
  }

  /**
   * Encode a local AudioBuffer in the browser → upload as .doae.
   * Returns the new latent_id.
   */
  async encodeAndUpload(audioBuffer) {
    if (!this.useWebGPU) {
      throw new Error("encodeAndUpload requires WebGPU. Use the legacy wav-upload path.");
    }
    const enc = await this._ensureEncoder();
    const ort = await _loadOrt();
    // Resample / re-channel to 48k stereo if needed
    const stereo = await this._toStereo48k(audioBuffer);
    const totalSamples = stereo[0].length;
    // Pad to a multiple of SAMPLES_PER_FRAME so the encoder doesn't drop frames
    const padded = Math.ceil(totalSamples / SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME;
    const interleaved = new Float32Array(2 * padded);
    interleaved.set(stereo[0]);
    interleaved.set(stereo[1], padded);

    const tensor = new ort.Tensor("float32", interleaved, [1, 2, padded]);
    const out = await enc.run({ audio: tensor });
    const latent = out.latent || out[Object.keys(out)[0]];
    // Output: [1, 64, T]
    const D = latent.dims[1];
    const T = latent.dims[2];
    const data = latent.data;            // Float32Array length 64*T
    // Build .doae body: row-major [T, D]
    const rowMajor = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      for (let d = 0; d < D; d++) {
        rowMajor[t * D + d] = data[d * T + t];
      }
    }
    const blob = this._writeDoae(rowMajor, T, D, 25);
    const r = await fetch("/api/upload-latent", {
      method: "POST",
      headers: { "Content-Type": "application/x-doae" },
      body: blob,
    });
    if (!r.ok) throw new Error(`upload-latent → ${r.status}`);
    return await r.json();
  }

  _writeDoae(rowMajor, T, D, fps) {
    const total = DOAE_HEADER_BYTES + T * D * 4;
    const body = new ArrayBuffer(total);
    const view = new DataView(body);
    // magic
    "DOAE".split("").forEach((c, i) => view.setUint8(i, c.charCodeAt(0)));
    view.setUint16(4, 1, true);            // version
    const hash = (this.expectedVaeVersion || "").padEnd(12, "\0").slice(0, 12);
    for (let i = 0; i < 12; i++) view.setUint8(6 + i, hash.charCodeAt(i) || 0);
    view.setUint16(18, fps, true);
    view.setUint32(20, T, true);
    view.setUint32(24, D, true);
    // body
    new Float32Array(body, DOAE_HEADER_BYTES, T * D).set(rowMajor);
    return body;
  }

  async _toStereo48k(audioBuffer) {
    // Best-effort: if already 48k stereo, return as-is. Otherwise use
    // OfflineAudioContext to resample.
    if (audioBuffer.sampleRate === SAMPLE_RATE && audioBuffer.numberOfChannels === 2) {
      return [audioBuffer.getChannelData(0), audioBuffer.getChannelData(1)];
    }
    const targetSamples = Math.ceil(audioBuffer.duration * SAMPLE_RATE);
    const oac = new OfflineAudioContext(2, targetSamples, SAMPLE_RATE);
    const src = oac.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(oac.destination);
    src.start();
    const rendered = await oac.startRendering();
    return [rendered.getChannelData(0), rendered.getChannelData(1)];
  }

  /**
   * Legacy wav fallback — fetch /api/generate-stemphonic/download/<id>
   * (or whatever path the backend exposes for raw stem audio).
   */
  async _fetchAndDecodeWav(latentId) {
    // First try the wav download endpoint
    const url = `/api/latent-as-wav/${latentId}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`wav fallback failed: ${url} → ${r.status}`);
    const buf = await r.arrayBuffer();
    return await this.audioContext.decodeAudioData(buf);
  }
}

// Singleton convenience — most studio call sites just want one player.
let _singleton = null;
export async function getLatentPlayer(audioContext) {
  if (_singleton) return _singleton;
  _singleton = new LatentPlayer({ audioContext });
  await _singleton.init();
  return _singleton;
}
