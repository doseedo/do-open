/**
 * Mask-based playback via AudioWorklet.
 *
 * All audio sources from the master WAV. Stem "playback" is real-time
 * spectral masking applied to the master in an AudioWorklet processor.
 *
 * Flow:
 *   1. User uploads audio → master buffer cached
 *   2. Latent demucs → stem latents → E2E mask model → per-stem masks
 *   3. Masks + master sent to AudioWorklet
 *   4. Worklet applies masks in real-time via FFT → mask → IFFT
 *   5. Solo/mute/volume = change active stem / gains (instant, no re-render)
 *
 * No decoder needed. No pre-computation. Real-time streaming masking.
 */

import * as ort from 'onnxruntime-web';

const MASK_MODEL_URL = '/models/latent_mask_e2e.onnx';
const WORKLET_URL = '/workers/maskWorklet.js';
const N_BANDS = 32;
const STEMS = ['drums', 'bass', 'vocals', 'other'];

let maskSession = null;
let workletNode = null;
let audioCtx = null;
let isInitialized = false;
let masterSent = false;
let masksSent = false;

// Band/bin lookup constants — must match maskWorklet.js exactly so the
// worklet's bilinear interpolation reads the values we send at the
// intended frequencies.
const N_FFT = 2048;
const HOP = 512;
const SR = 48000;
const F_BINS = N_FFT / 2 + 1; // 1025
const LATENT_HOP = 1920;
const STFT_PER_LATENT = LATENT_HOP / HOP; // 3.75 STFT frames per latent frame

// Precompute band-edges and per-bin band membership once. Same log-
// spacing formula as maskWorklet (20 Hz → SR/2).
const bandEdges = new Float32Array(N_BANDS + 1);
for (let i = 0; i <= N_BANDS; i++) {
  bandEdges[i] = 20 * Math.pow((SR / 2) / 20, i / N_BANDS);
}
// binToBand[f] = the band index that FFT bin f falls into, or -1 if
// it's below 20 Hz / above Nyquist. Each bin contributes to exactly
// one band (unlike the worklet's two-band bilinear blend — that's
// only needed for smooth per-sample playback, not for the mask values
// themselves, which are already band-rate).
const binToBand = new Int16Array(F_BINS);
for (let f = 0; f < F_BINS; f++) {
  const freq = f * (SR / 2) / (F_BINS - 1);
  binToBand[f] = -1;
  for (let b = 0; b < N_BANDS; b++) {
    if (freq >= bandEdges[b] && freq < bandEdges[b + 1]) {
      binToBand[f] = b;
      break;
    }
  }
  // Edge cases: freq at Nyquist → last band.
  if (binToBand[f] < 0 && freq >= bandEdges[N_BANDS - 1]) binToBand[f] = N_BANDS - 1;
}

/**
 * Initialize the mask playback system.
 * MUST receive the existing AudioContext from useAudioPlayback.
 */
export async function initMaskPlayback(ctx) {
  if (!ctx) throw new Error('maskPlayback requires an existing AudioContext');
  audioCtx = ctx;

  // Load ONNX mask model
  if (!maskSession) {
    try {
      maskSession = await ort.InferenceSession.create(MASK_MODEL_URL, {
        executionProviders: ['wasm'],
      });
      console.log('[maskPlayback] ONNX model loaded (9KB)');
    } catch (err) {
      console.error('[maskPlayback] model load failed:', err);
      return false;
    }
  }

  // Register AudioWorklet
  try {
    await audioCtx.audioWorklet.addModule(WORKLET_URL);
    console.log('[maskPlayback] worklet registered');
  } catch (err) {
    console.error('[maskPlayback] worklet registration failed:', err);
    return false;
  }

  isInitialized = true;
  return true;
}

/**
 * Create a mask playback node connected to the audio output.
 * Returns the worklet node for routing into the signal chain.
 */
export function createMaskPlaybackNode(destination) {
  if (!isInitialized) throw new Error('maskPlayback not initialized');

  // Disconnect any existing node to avoid leaks
  if (workletNode) {
    try { workletNode.disconnect(); } catch (e) {}
  }
  masterSent = false;
  masksSent = false;

  workletNode = new AudioWorkletNode(audioCtx, 'mask-playback-processor', {
    outputChannelCount: [1],
  });
  workletNode.connect(destination || audioCtx.destination);
  console.log('[maskPlayback] worklet node created');
  return workletNode;
}

/**
 * Send the master audio to the worklet.
 * @param {AudioBuffer} masterBuffer — the original uploaded audio
 */
export function setMaster(masterBuffer) {
  if (!workletNode) return;
  const mono = masterBuffer.getChannelData(0);
  // Transfer a copy to the worklet
  const copy = new Float32Array(mono);
  workletNode.port.postMessage({ type: 'setMaster', data: copy }, [copy.buffer]);
  masterSent = true;
  console.log(`[maskPlayback] master set: ${mono.length} samples (${(mono.length/48000).toFixed(1)}s)`);
}

/**
 * Compute stem masks from latents and send to worklet.
 *
 * @param {Object} stemLatents - { drums: Float32Array[T*64], ... } time-major
 * @param {number} T - latent frames
 */
export async function computeAndSetMasks(stemLatents, T) {
  if (!maskSession) throw new Error('mask model not loaded');

  // Concatenate stem latents: [4*64, T] channels-first for the model
  const concat = new Float32Array(4 * 64 * T);
  STEMS.forEach((name, si) => {
    const lat = stemLatents[name];
    if (!lat) return;
    for (let t = 0; t < T; t++) {
      for (let d = 0; d < 64; d++) {
        // Input layout: [1, 256, T] where 256 = 4 stems × 64 dims
        concat[(si * 64 + d) * T + t] = lat[t * 64 + d];
      }
    }
  });

  const t0 = performance.now();
  const input = new ort.Tensor('float32', concat, [1, 256, T]);
  const result = await maskSession.run({ latents: input });
  const logits = result.band_logits.data; // [1, 4, N_BANDS, T]

  // Compute softmax masks per stem
  const masks = {};
  const T_mask = T;
  STEMS.forEach((name, si) => {
    masks[name] = new Float32Array(N_BANDS * T_mask);
  });

  for (let t = 0; t < T_mask; t++) {
    for (let b = 0; b < N_BANDS; b++) {
      // Softmax across stems
      let maxL = -Infinity;
      for (let si = 0; si < 4; si++) {
        const v = logits[si * N_BANDS * T_mask + b * T_mask + t];
        if (v > maxL) maxL = v;
      }
      let sumExp = 0;
      const exps = [];
      for (let si = 0; si < 4; si++) {
        const e = Math.exp(logits[si * N_BANDS * T_mask + b * T_mask + t] - maxL);
        exps.push(e);
        sumExp += e;
      }
      for (let si = 0; si < 4; si++) {
        masks[STEMS[si]][b * T_mask + t] = exps[si] / sumExp;
      }
    }
  }

  const ms = performance.now() - t0;
  console.log(`[maskPlayback] masks computed in ${ms.toFixed(0)}ms (${T} frames)`);

  // Send to worklet
  if (workletNode) {
    // Transfer mask data to worklet (structured clone, not transfer — masks are reused)
    workletNode.port.postMessage({ type: 'setMasks', masks });
    masksSent = true;
  }

  return masks;
}

/**
 * Solo a specific stem (or null for full mix).
 */
export function setActiveStem(stemName) {
  if (!workletNode) return;
  workletNode.port.postMessage({ type: 'setActiveStem', stem: stemName });
  console.log(`[maskPlayback] active stem: ${stemName || 'mix'}`);
}

/**
 * Set per-stem gains for mixing.
 * @param {Object} gains - { drums: 1.0, bass: 0.5, vocals: 0, other: 1.0 }
 */
export function setGains(gains) {
  if (!workletNode) return;
  workletNode.port.postMessage({ type: 'setGains', gains });
}

/**
 * Seek to a position in seconds.
 */
export function seek(timeSec) {
  if (!workletNode) return;
  workletNode.port.postMessage({ type: 'seek', frame: Math.floor(timeSec * 48000) });
}

/**
 * Play / pause.
 */
export function play() { workletNode?.port.postMessage({ type: 'play' }); }
export function stop() { workletNode?.port.postMessage({ type: 'stop' }); }

// ── Pre-computed stem audio buffers (fallback for worklet issues) ────
let precomputedBuffers = null; // { drums: AudioBuffer, bass: ..., vocals: ..., other: ... }

/**
 * Pre-compute all stem audio buffers from masks applied to master STFT.
 * This is the reliable fallback — no real-time worklet needed.
 * Uses the Web Audio OfflineAudioContext for fast processing.
 *
 * @param {Object} masks - per-stem masks from computeAndSetMasks
 * @param {AudioBuffer} masterBuffer - the original uploaded audio
 * @returns {Object} - { drums: Blob URL, bass: Blob URL, ... }
 */
export async function precomputeStemAudio(masks, masterBuffer) {
  const mono = masterBuffer.getChannelData(0);
  const N = mono.length;
  const sr = masterBuffer.sampleRate;
  const nfft = 2048, hop = 512;
  const F = nfft / 2 + 1;
  const T_stft = Math.floor((N - nfft) / hop) + 1;

  // Hann window
  const win = new Float32Array(nfft);
  for (let i = 0; i < nfft; i++) win[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / nfft));

  // Inline radix-2 FFT
  function fft(re, im, inverse) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    const dir = inverse ? -1 : 1;
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1, angle = dir * 2 * Math.PI / len;
      const wRe = Math.cos(angle), wIm = Math.sin(angle);
      for (let i = 0; i < n; i += len) {
        let cRe = 1, cIm = 0;
        for (let j = 0; j < half; j++) {
          const a = i + j, b = a + half;
          const tRe = re[b] * cRe - im[b] * cIm, tIm = re[b] * cIm + im[b] * cRe;
          re[b] = re[a] - tRe; im[b] = im[a] - tIm;
          re[a] += tRe; im[a] += tIm;
          const nRe = cRe * wRe - cIm * wIm; cIm = cRe * wIm + cIm * wRe; cRe = nRe;
        }
      }
    }
    if (inverse) { for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; } }
  }

  // Band edges (log-spaced)
  const nBands = 32;
  const bandEdges = [];
  for (let i = 0; i <= nBands; i++) bandEdges.push(20 * Math.pow(sr / 2 / 20, i / nBands));
  const binToBand = new Int32Array(F);
  const freqStep = (sr / 2) / (F - 1);
  for (let f = 0; f < F; f++) {
    binToBand[f] = -1;
    for (let b = 0; b < nBands; b++) {
      if (f * freqStep >= bandEdges[b] && f * freqStep < bandEdges[b + 1]) { binToBand[f] = b; break; }
    }
  }

  const t0 = performance.now();
  const stemAudio = {};

  for (const stemName of STEMS) {
    const mask = masks[stemName];
    if (!mask) continue;
    const T_mask = mask.length / nBands;

    const out = new Float32Array(N);
    const winSum = new Float32Array(N);

    for (let t = 0; t < T_stft; t++) {
      const offset = t * hop;
      const latFrame = Math.floor(offset / 1920);

      // Extract windowed frame
      const re = new Float32Array(nfft);
      const im = new Float32Array(nfft);
      for (let n2 = 0; n2 < nfft; n2++) {
        re[n2] = (offset + n2 < N) ? mono[offset + n2] * win[n2] : 0;
      }

      // FFT
      fft(re, im, false);

      // Apply mask per frequency bin
      for (let f = 0; f < F; f++) {
        const b = binToBand[f];
        let gain = 1.0;
        if (b >= 0) {
          const tIdx = Math.min(latFrame, T_mask - 1);
          gain = mask[b * T_mask + tIdx];
        }
        re[f] *= gain; im[f] *= gain;
        if (f > 0 && f < F - 1) { re[nfft - f] = re[f]; im[nfft - f] = -im[f]; }
      }

      // IFFT
      fft(re, im, true);

      // Overlap-add
      for (let n2 = 0; n2 < nfft && offset + n2 < N; n2++) {
        out[offset + n2] += re[n2] * win[n2];
        winSum[offset + n2] += win[n2] * win[n2];
      }
    }

    // Normalize
    for (let i = 0; i < N; i++) { if (winSum[i] > 1e-8) out[i] /= winSum[i]; }

    // Create blob URL
    const buf = audioCtx.createBuffer(1, N, sr);
    buf.getChannelData(0).set(out);
    stemAudio[stemName] = buf;
  }

  const ms = performance.now() - t0;
  console.log(`[maskPlayback] pre-computed ${Object.keys(stemAudio).length} stem buffers in ${ms.toFixed(0)}ms`);
  precomputedBuffers = stemAudio;
  return stemAudio;
}

/**
 * Get a pre-computed stem AudioBuffer (for playback via standard BufferSource).
 */
export function getPrecomputedStemBuffer(stemName) {
  return precomputedBuffers?.[stemName] || null;
}

/**
 * Check if mask playback is available.
 */
export function isMaskPlaybackReady() {
  return isInitialized && workletNode !== null && masterSent && masksSent;
}

export function getMaskStemNames() {
  return [...STEMS];
}

/**
 * Feed refined STFT masks from the latent_mask_refiner pipeline
 * straight into the worklet, bypassing computeAndSetMasks (which runs
 * a small ONNX on the latents). Input is in STFT space (1025 freq
 * bins × T_stft frames); we reduce to the worklet's [32 bands × T_lat]
 * grid before posting.
 *
 * The refiner's per-stem sigmoids don't sum to 1 across stems.
 * Normalize per (band, frame) so the worklet sees probabilities.
 *
 * @param {Object} refinedByStem  { drums: Float32Array(F*T_stft), ... }
 * @param {number} F              1025
 * @param {number} T_stft         time frames at STFT resolution
 * @param {number} T_lat          time frames at latent resolution
 */
export function setRefinedMasks(refinedByStem, F, T_stft, T_lat) {
  if (!workletNode) throw new Error('maskPlayback worklet not initialized');
  if (F !== F_BINS) throw new Error(`setRefinedMasks: expected F=${F_BINS} got ${F}`);

  const stemNames = Object.keys(refinedByStem).filter(n => refinedByStem[n]);
  if (stemNames.length === 0) return;

  // Count how many FFT bins fall into each band (for averaging).
  const binsPerBand = new Int32Array(N_BANDS);
  for (let f = 0; f < F; f++) {
    const b = binToBand[f];
    if (b >= 0) binsPerBand[b] += 1;
  }

  // Intermediate: [N_BANDS × T_stft] per stem (space-reduced only).
  const bandStft = {};
  for (const name of stemNames) {
    const src = refinedByStem[name];
    const dst = new Float32Array(N_BANDS * T_stft);
    for (let f = 0; f < F; f++) {
      const b = binToBand[f];
      if (b < 0) continue;
      for (let t = 0; t < T_stft; t++) {
        dst[b * T_stft + t] += src[f * T_stft + t];
      }
    }
    // Normalize by bin count per band so each band's value is the mean
    // mask magnitude across its covered bins.
    for (let b = 0; b < N_BANDS; b++) {
      const n = binsPerBand[b] || 1;
      for (let t = 0; t < T_stft; t++) dst[b * T_stft + t] /= n;
    }
    bandStft[name] = dst;
  }

  // Time-reduce T_stft → T_lat. Each latent frame covers ~3.75 STFT
  // frames; average the STFT slice that belongs to each latent frame.
  const bandLat = {};
  for (const name of stemNames) {
    const src = bandStft[name];
    const dst = new Float32Array(N_BANDS * T_lat);
    for (let t = 0; t < T_lat; t++) {
      const s0 = Math.round(t * STFT_PER_LATENT);
      const s1 = Math.min(Math.round((t + 1) * STFT_PER_LATENT), T_stft);
      const span = Math.max(1, s1 - s0);
      for (let b = 0; b < N_BANDS; b++) {
        let sum = 0;
        for (let s = s0; s < s1; s++) sum += src[b * T_stft + s];
        dst[b * T_lat + t] = sum / span;
      }
    }
    bandLat[name] = dst;
  }

  // Cross-stem normalization so masks sum to 1 per (band, latent-frame).
  // Protects the AudioWorklet's STFT reconstruction from double-counting
  // when two stems both claim the same frequency cell.
  for (let t = 0; t < T_lat; t++) {
    for (let b = 0; b < N_BANDS; b++) {
      let sum = 0;
      for (const name of stemNames) sum += bandLat[name][b * T_lat + t];
      if (sum > 1e-6) {
        for (const name of stemNames) bandLat[name][b * T_lat + t] /= sum;
      }
    }
  }

  workletNode.port.postMessage({ type: 'setMasks', masks: bandLat });
  masksSent = true;
  console.log(`[maskPlayback] refined masks posted — ${stemNames.length} stems, ${N_BANDS} bands × ${T_lat} frames`);
}
