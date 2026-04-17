/**
 * AudioWorklet processor for real-time spectral masking.
 *
 * Reads from the master audio buffer and applies per-stem frequency masks
 * in real-time during playback. No pre-computation needed.
 *
 * The mask is a lookup table: [stem_index][freq_band][time_frame] -> gain [0,1]
 * The worklet reads the master, applies the active mask, outputs masked audio.
 *
 * Masks are 32 log-spaced bands. To avoid ringing from step-function band
 * boundaries, each FFT bin's gain is log-frequency interpolated between
 * its two nearest band centers, and temporally crossfaded between adjacent
 * latent frames.
 *
 * Messages from main thread:
 *   { type: 'setMaster', data: Float32Array }     - master audio mono
 *   { type: 'setMasks', masks: { drums: Float32Array, ... } } - per-stem masks [N_BANDS * T_mask]
 *   { type: 'setActiveStem', stem: 'drums' | null } - which stem to solo (null = mix all)
 *   { type: 'setGains', gains: { drums: 0.8, ... } } - per-stem volume
 *   { type: 'seek', frame: number }                - seek to sample position
 */

const N_FFT = 2048;
const HOP = 512;
const N_BANDS = 32;
const SR = 48000;
const F = N_FFT / 2 + 1; // 1025
const LATENT_HOP = 1920; // samples per latent frame

class MaskPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.master = null;        // Float32Array - full master audio mono
    this.masks = {};           // { stemName: Float32Array[N_BANDS * T_mask] }
    this.gains = {};           // { stemName: number }
    this.activeStem = null;    // null = weighted mix, 'drums' = solo drums
    this.playhead = 0;         // current sample position
    this.playing = false;
    this.needsInitFrame = true; // trigger first STFT frame on start/seek

    // Pre-compute band edges (log-spaced) and band centers
    this.bandEdges = new Float32Array(N_BANDS + 1);
    for (let i = 0; i <= N_BANDS; i++) {
      this.bandEdges[i] = 20 * Math.pow(SR / 2 / 20, i / N_BANDS);
    }
    const bandCenters = new Float32Array(N_BANDS);
    for (let b = 0; b < N_BANDS; b++) {
      bandCenters[b] = Math.sqrt(this.bandEdges[b] * this.bandEdges[b + 1]);
    }

    // Pre-compute per-bin interpolation weights for smooth frequency masking.
    // Each FFT bin maps to two adjacent band centers with a log-frequency
    // interpolation weight, eliminating the harsh step-function boundaries.
    const freqStep = (SR / 2) / (F - 1);
    this.binLowBand = new Int32Array(F);
    this.binHighBand = new Int32Array(F);
    this.binAlpha = new Float32Array(F);

    for (let f = 0; f < F; f++) {
      const freq = f * freqStep;
      if (freq < this.bandEdges[0] || freq >= this.bandEdges[N_BANDS]) {
        // Below 20 Hz or at/above Nyquist — pass through
        this.binLowBand[f] = -1;
        this.binHighBand[f] = -1;
        this.binAlpha[f] = 0;
        continue;
      }

      if (freq <= bandCenters[0]) {
        this.binLowBand[f] = 0;
        this.binHighBand[f] = 0;
        this.binAlpha[f] = 0;
      } else if (freq >= bandCenters[N_BANDS - 1]) {
        this.binLowBand[f] = N_BANDS - 1;
        this.binHighBand[f] = N_BANDS - 1;
        this.binAlpha[f] = 0;
      } else {
        for (let b = 0; b < N_BANDS - 1; b++) {
          if (freq >= bandCenters[b] && freq < bandCenters[b + 1]) {
            this.binLowBand[f] = b;
            this.binHighBand[f] = b + 1;
            const logFreq = Math.log(freq);
            const logLow = Math.log(bandCenters[b]);
            const logHigh = Math.log(bandCenters[b + 1]);
            this.binAlpha[f] = (logFreq - logLow) / (logHigh - logLow);
            break;
          }
        }
      }
    }

    // Hann window
    this.window = new Float32Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) {
      this.window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / N_FFT));
    }

    // Overlap-add buffers
    this.outputAccum = new Float32Array(N_FFT * 2);
    this.windowAccum = new Float32Array(N_FFT * 2);
    this.lastHopSample = 0;
    this.accumOffset = 0;

    // Pre-allocate FFT scratch buffers (avoid per-frame GC pressure)
    this._fftRe = new Float32Array(N_FFT);
    this._fftIm = new Float32Array(N_FFT);

    this.port.onmessage = (e) => this._handleMessage(e.data);
  }

  _handleMessage(msg) {
    switch (msg.type) {
      case 'setMaster':
        this.master = msg.data;
        // Don't auto-play; wait for explicit 'play' message
        this.needsInitFrame = true;
        break;
      case 'setMasks':
        this.masks = msg.masks;
        break;
      case 'setActiveStem':
        this.activeStem = msg.stem;
        break;
      case 'setGains':
        this.gains = msg.gains;
        break;
      case 'seek':
        this.playhead = msg.frame;
        this.outputAccum.fill(0);
        this.windowAccum.fill(0);
        this.accumOffset = 0;
        this.needsInitFrame = true;
        break;
      case 'stop':
        this.playing = false;
        break;
      case 'play':
        this.playing = true;
        this.needsInitFrame = true;
        break;
    }
  }

  /**
   * Get the interpolated mask gain for a given frequency bin and time.
   * Uses log-frequency interpolation between band centers and linear
   * temporal interpolation between latent frames.
   */
  _getMaskGain(freqBin, exactLatentTime) {
    const bLow = this.binLowBand[freqBin];
    if (bLow < 0) return 1.0; // unmapped bin, pass through

    const bHigh = this.binHighBand[freqBin];
    const freqAlpha = this.binAlpha[freqBin];

    // Temporal interpolation indices
    const tLow = Math.floor(exactLatentTime);
    const tAlpha = exactLatentTime - tLow;

    if (this.activeStem) {
      // Solo mode: only active stem's mask
      const mask = this.masks[this.activeStem];
      if (!mask) return 0.0;
      const T_mask = mask.length / N_BANDS;
      const t0 = Math.min(tLow, T_mask - 1);
      const t1 = Math.min(tLow + 1, T_mask - 1);
      const gain = this.gains[this.activeStem] ?? 1.0;

      // Bilinear: interpolate freq then time
      const vLL = mask[bLow * T_mask + t0];
      const vHL = mask[bHigh * T_mask + t0];
      const vLH = mask[bLow * T_mask + t1];
      const vHH = mask[bHigh * T_mask + t1];
      const v0 = vLL + freqAlpha * (vHL - vLL);
      const v1 = vLH + freqAlpha * (vHH - vLH);
      return (v0 + tAlpha * (v1 - v0)) * gain;
    }

    // Mix mode: weighted sum of all stem masks (bilinear-interpolated)
    let total = 0;
    const stemNames = Object.keys(this.masks);
    for (const name of stemNames) {
      const mask = this.masks[name];
      const T_mask = mask.length / N_BANDS;
      const t0 = Math.min(tLow, T_mask - 1);
      const t1 = Math.min(tLow + 1, T_mask - 1);
      const gain = this.gains[name] ?? 1.0;

      const vLL = mask[bLow * T_mask + t0];
      const vHL = mask[bHigh * T_mask + t0];
      const vLH = mask[bLow * T_mask + t1];
      const vHH = mask[bHigh * T_mask + t1];
      const v0 = vLL + freqAlpha * (vHL - vLL);
      const v1 = vLH + freqAlpha * (vHH - vLH);
      total += (v0 + tAlpha * (v1 - v0)) * gain;
    }
    return Math.min(total, 1.0);
  }

  process(inputs, outputs, parameters) {
    if (!this.playing || !this.master) {
      return true; // keep alive
    }

    const output = outputs[0][0]; // mono output
    const blockSize = output.length; // typically 128

    for (let i = 0; i < blockSize; i++) {
      const sampleIdx = this.playhead + i;
      if (sampleIdx >= this.master.length) {
        output[i] = 0;
        continue;
      }

      // Check if we need to process a new STFT frame
      if (sampleIdx >= this.lastHopSample + HOP || this.needsInitFrame) {
        this._processFrame(sampleIdx);
        this.lastHopSample = sampleIdx - (sampleIdx % HOP);
        this.needsInitFrame = false;
      }

      // Read from overlap-add accumulator
      const accumIdx = (sampleIdx - this.lastHopSample + this.accumOffset) % this.outputAccum.length;
      const winVal = this.windowAccum[accumIdx];
      output[i] = winVal > 1e-8 ? this.outputAccum[accumIdx] / winVal : 0;

      // Clear consumed position so the circular buffer doesn't accumulate stale data
      this.outputAccum[accumIdx] = 0;
      this.windowAccum[accumIdx] = 0;
    }

    this.playhead += blockSize;
    return true;
  }

  // -- Radix-2 FFT (in-place, Cooley-Tukey) --
  _fft(re, im, inverse = false) {
    const n = re.length;
    // Bit-reversal permutation
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    // Butterfly
    const dir = inverse ? -1 : 1;
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1;
      const angle = dir * 2 * Math.PI / len;
      const wRe = Math.cos(angle), wIm = Math.sin(angle);
      for (let i = 0; i < n; i += len) {
        let curRe = 1, curIm = 0;
        for (let j = 0; j < half; j++) {
          const a = i + j, b = a + half;
          const tRe = re[b] * curRe - im[b] * curIm;
          const tIm = re[b] * curIm + im[b] * curRe;
          re[b] = re[a] - tRe;
          im[b] = im[a] - tIm;
          re[a] += tRe;
          im[a] += tIm;
          const nextRe = curRe * wRe - curIm * wIm;
          curIm = curRe * wIm + curIm * wRe;
          curRe = nextRe;
        }
      }
    }
    if (inverse) {
      for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    }
  }

  _processFrame(centerSample) {
    const frameStart = centerSample - (centerSample % HOP);
    if (frameStart < 0) return;

    // Exact fractional latent frame for temporal interpolation
    const exactLatentTime = frameStart / LATENT_HOP;

    // Reuse pre-allocated scratch buffers (zero-fill instead of allocating)
    const re = this._fftRe;
    const im = this._fftIm;
    for (let n = 0; n < N_FFT; n++) {
      const idx = frameStart + n;
      re[n] = (idx >= 0 && idx < this.master.length)
        ? this.master[idx] * this.window[n] : 0;
      im[n] = 0;
    }

    // Forward FFT
    this._fft(re, im, false);

    // Apply interpolated mask to frequency bins
    for (let f = 0; f < F; f++) {
      const gain = this._getMaskGain(f, exactLatentTime);
      re[f] *= gain;
      im[f] *= gain;
      // Mirror for negative frequencies (conjugate symmetric for real input)
      if (f > 0 && f < F - 1) {
        re[N_FFT - f] = re[f];
        im[N_FFT - f] = -im[f];
      }
    }

    // Inverse FFT
    this._fft(re, im, true);

    // Overlap-add with window
    const accumBase = this.accumOffset;
    for (let n = 0; n < N_FFT; n++) {
      const idx = (accumBase + n) % this.outputAccum.length;
      this.outputAccum[idx] += re[n] * this.window[n];
      this.windowAccum[idx] += this.window[n] * this.window[n];
    }
    this.accumOffset = (accumBase + HOP) % this.outputAccum.length;
  }
}

registerProcessor('mask-playback-processor', MaskPlaybackProcessor);
