/**
 * Client-side latent-mask playback source.
 *
 * Takes the 4-stem oobleck latents that distill_demucs_fp16 emits
 * (shape [1, 4, 64, T]), concatenates the 4 stems' 64-channel latents
 * into a single [1, 256, T] input, runs latent_mask_e2e.onnx to get a
 * 4-stem STFT mask (`mask_logits [1, 4, 1025, T]`, n_fft=2048, hop=1920),
 * computes the STFT of the master stereo waveform once, then applies
 * sigmoid(mask[s]) × master_STFT and iSTFTs back to audio for each
 * stem. The result is four stereo PCM buffers that get encoded as WAV
 * blob URLs.
 *
 * This audio becomes the `audioUrl` playback source for each stem
 * track UNTIL the backend /separate-stems WAV arrives. Because the
 * signal is master-through-STFT-iSTFT (no VAE decode), it sounds
 * clean — the model is just shaping which bins belong to which stem.
 *
 * The UI waveform canvas is still driven exclusively by
 * `metadata.envelopeData`. This service's job is playback audio only.
 */

const MODEL_URL = '/static/models/latent_mask_e2e.onnx';
const SR         = 48000;
const N_FFT      = 2048;
const HOP        = 1920;          // matches latent frame rate — 25 fps
const N_FREQ     = N_FFT / 2 + 1; // 1025
const N_STEMS    = 4;
const LATENT_CHANS = 64;

export const LATENT_MASK_STEM_NAMES = ['drums', 'bass', 'vocals', 'other'];

let _ort = null;
let _session = null;
let _initPromise = null;

export function isLatentMaskReady() { return _session != null; }

export async function initLatentMask() {
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
    const graphBuf = await fetchModelWithCache(MODEL_URL);
    const graphBytes = new Uint8Array(graphBuf);

    const backends = [];
    if (ort.env?.webgpu) backends.push('webgpu');
    backends.push('wasm');
    let lastErr = null;
    for (const ep of backends) {
      try {
        _session = await ort.InferenceSession.create(graphBytes, {
          executionProviders: [ep],
          graphOptimizationLevel: 'all',
          logSeverityLevel: 3,
        });
        console.log(`[latentMask] ready on ${ep}`);
        return _session;
      } catch (err) {
        console.warn(`[latentMask] ${ep} init failed:`, err?.message || err);
        lastErr = err;
      }
    }
    throw lastErr || new Error('no ORT backend for latentMask');
  })();

  return _initPromise;
}

// ─── FFT (radix-2 Cooley-Tukey, complex in/out, in-place) ───────────────────
function _fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k], uIm = im[i + k];
        const vRe = re[i + k + len/2] * curRe - im[i + k + len/2] * curIm;
        const vIm = re[i + k + len/2] * curIm + im[i + k + len/2] * curRe;
        re[i + k]         = uRe + vRe; im[i + k]         = uIm + vIm;
        re[i + k + len/2] = uRe - vRe; im[i + k + len/2] = uIm - vIm;
        const nr = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nr;
      }
    }
  }
}

function _ifft(re, im) {
  // IFFT via conjugate trick: swap re/im, FFT, swap again, divide by N
  _fft(im, re);
  const n = re.length;
  const inv = 1 / n;
  for (let i = 0; i < n; i++) { re[i] *= inv; im[i] *= inv; }
}

/** Hann window of length N. */
function _hann(N) {
  const w = new Float32Array(N);
  for (let i = 0; i < N; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (N - 1)));
  return w;
}

/** STFT of a mono Float32Array → { re, im } each [nFreq * T] row-major [t, f].
 * Uses reflect-padded n_fft/2 boundary to match librosa/scipy `boundary='zeros'`
 * approximately. Our model was trained at hop=1920, n_fft=2048, Hann. */
function _stft(mono, nFft = N_FFT, hop = HOP) {
  const N = mono.length;
  const pad = nFft / 2;
  const total = N + 2 * pad;
  const padded = new Float32Array(total);
  // zero-pad each side (boundary='zeros' in scipy)
  padded.set(mono, pad);

  const T = Math.floor((padded.length - nFft) / hop) + 1;
  const nFreq = nFft / 2 + 1;
  const reOut = new Float32Array(nFreq * T);
  const imOut = new Float32Array(nFreq * T);

  const win = _hann(nFft);
  const reBuf = new Float32Array(nFft);
  const imBuf = new Float32Array(nFft);

  for (let t = 0; t < T; t++) {
    const off = t * hop;
    for (let k = 0; k < nFft; k++) { reBuf[k] = padded[off + k] * win[k]; imBuf[k] = 0; }
    _fft(reBuf, imBuf);
    for (let f = 0; f < nFreq; f++) {
      reOut[t * nFreq + f] = reBuf[f];
      imOut[t * nFreq + f] = imBuf[f];
    }
  }
  return { re: reOut, im: imOut, T, nFreq };
}

/** iSTFT using overlap-add with Hann² normalization. Matches scipy istft with
 * `window='hann', boundary=True, nperseg=n_fft, noverlap=n_fft-hop`. */
function _istft({ re, im, T, nFreq }, targetLen, nFft = N_FFT, hop = HOP) {
  const pad = nFft / 2;
  const total = (T - 1) * hop + nFft;
  const out = new Float32Array(total);
  const norm = new Float32Array(total);
  const win = _hann(nFft);

  const reBuf = new Float32Array(nFft);
  const imBuf = new Float32Array(nFft);

  for (let t = 0; t < T; t++) {
    // Rebuild full spectrum (Hermitian symmetry for real iFFT input)
    reBuf.fill(0); imBuf.fill(0);
    for (let f = 0; f < nFreq; f++) {
      reBuf[f] = re[t * nFreq + f];
      imBuf[f] = im[t * nFreq + f];
    }
    for (let f = 1; f < nFreq - 1; f++) {
      reBuf[nFft - f] =  reBuf[f];
      imBuf[nFft - f] = -imBuf[f];
    }
    _ifft(reBuf, imBuf);
    const off = t * hop;
    for (let k = 0; k < nFft; k++) {
      out[off + k]  += reBuf[k] * win[k];
      norm[off + k] += win[k] * win[k];
    }
  }
  // normalize overlap-add
  for (let i = 0; i < total; i++) {
    if (norm[i] > 1e-8) out[i] /= norm[i];
  }
  // strip the boundary pad and crop to target length
  return out.subarray(pad, pad + targetLen);
}

function _sigmoidInPlace(a) {
  for (let i = 0; i < a.length; i++) a[i] = 1 / (1 + Math.exp(-a[i]));
  return a;
}

/** Encode a channels-first stereo Float32Array [L..L_{N-1}, R..R_{N-1}] to a
 * 16-bit PCM WAV blob URL. */
function _stereoToWavUrl(flat, numSamples) {
  const N = numSamples;
  const byteLen = 44 + N * 2 * 2;
  const ab = new ArrayBuffer(byteLen);
  const view = new DataView(ab);
  const s = (o, str) => { for (let i = 0; i < str.length; i++) view.setUint8(o + i, str.charCodeAt(i)); };
  s(0, 'RIFF'); view.setUint32(4, byteLen - 8, true);
  s(8, 'WAVE'); s(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 2, true);
  view.setUint32(24, SR, true);
  view.setUint32(28, SR * 2 * 2, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 16, true);
  s(36, 'data'); view.setUint32(40, N * 2 * 2, true);
  let off = 44;
  for (let i = 0; i < N; i++) {
    let l = Math.max(-1, Math.min(1, flat[i]));
    let r = Math.max(-1, Math.min(1, flat[numSamples + i]));
    view.setInt16(off, (l * 0x7fff) | 0, true); off += 2;
    view.setInt16(off, (r * 0x7fff) | 0, true); off += 2;
  }
  return URL.createObjectURL(new Blob([ab], { type: 'audio/wav' }));
}

/**
 * Run the latent_mask_e2e pipeline and produce 4 per-stem WAV blob URLs.
 *
 * @param {Float32Array[]} stemLatents length-4 array; each Float32Array is
 *   [64 * T] for one stem (channel-major, same layout sem4Decoder accumulates)
 * @param {Float32Array}   masterFlat  channels-first stereo [L..L_{N-1}, R..R_{N-1}]
 * @param {number}         numFrames   sample count in master per channel
 * @returns {Promise<{stemName:string, wavUrl:string}[]>}
 */
export async function stemAudiosFromMaskOverMaster(stemLatents, masterFlat, numFrames) {
  const sess = await initLatentMask();
  const ort = _ort;
  const { ortWebGPURun } = await import('./webgpuOrtQueue');

  // --- 1. concat 4 stems × 64 channels into [1, 256, T] in channel-major order.
  const T = stemLatents[0].length / LATENT_CHANS;
  if (!Number.isInteger(T)) throw new Error(`stemLatents[0] length ${stemLatents[0].length} not multiple of ${LATENT_CHANS}`);
  const latFlat = new Float32Array(256 * T);
  for (let s = 0; s < N_STEMS; s++) {
    latFlat.set(stemLatents[s], s * LATENT_CHANS * T);
  }
  const latTensor = new ort.Tensor('float32', latFlat, [1, 256, T]);

  // --- 2. run mask model on shared WebGPU queue
  const runOut = await ortWebGPURun(() => sess.run({ latents: latTensor }));
  const maskLogits = runOut['mask_logits']?.data;
  if (!maskLogits) throw new Error('latent_mask_e2e produced no mask_logits');
  const maskT = maskLogits.length / (N_STEMS * N_FREQ);
  _sigmoidInPlace(maskLogits);   // logits → [0, 1] probabilities, in place

  // --- 3. STFT of master (left + right channels)
  const masterL = masterFlat.subarray(0, numFrames);
  const masterR = masterFlat.subarray(numFrames, 2 * numFrames);
  const XL = _stft(masterL);
  const XR = _stft(masterR);
  const stftT = Math.min(XL.T, maskT);

  // --- 4. per-stem: apply mask to each channel, iSTFT, encode WAV
  const stems = [];
  for (let s = 0; s < N_STEMS; s++) {
    // mask slice for this stem: stride = N_FREQ * maskT
    const maskBase = s * N_FREQ * maskT;

    const reL = new Float32Array(N_FREQ * stftT);
    const imL = new Float32Array(N_FREQ * stftT);
    const reR = new Float32Array(N_FREQ * stftT);
    const imR = new Float32Array(N_FREQ * stftT);
    // Broadcast mask (freq-major within a frame) × STFT (frame-major within
    // a freq). Mask layout: [1, 4, nFreq, T] — so mask[s][f][t] lives at
    // maskBase + f * maskT + t.  STFT layout: row-major [t, f].
    for (let t = 0; t < stftT; t++) {
      for (let f = 0; f < N_FREQ; f++) {
        const m = maskLogits[maskBase + f * maskT + t];
        const idxX = t * N_FREQ + f;
        reL[idxX] = XL.re[idxX] * m;
        imL[idxX] = XL.im[idxX] * m;
        reR[idxX] = XR.re[idxX] * m;
        imR[idxX] = XR.im[idxX] * m;
      }
    }

    const yL = _istft({ re: reL, im: imL, T: stftT, nFreq: N_FREQ }, numFrames);
    const yR = _istft({ re: reR, im: imR, T: stftT, nFreq: N_FREQ }, numFrames);
    const flat = new Float32Array(2 * numFrames);
    flat.set(yL, 0);
    flat.set(yR, numFrames);
    const url = _stereoToWavUrl(flat, numFrames);
    stems.push({ stemName: LATENT_MASK_STEM_NAMES[s], wavUrl: url });
  }
  return stems;
}
