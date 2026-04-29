/**
 * r1-pitch-shift-processor — time-domain SOLA pitch shifter with mix
 *
 * Implements granular SOLA (Synchronized Overlap-Add) pitch-shifting:
 *   1. Input is written into a circular buffer at the normal rate.
 *   2. Read with a fractional pointer that moves at `shiftRatio` (= 2^(semitones/12)).
 *   3. To prevent the read pointer from drifting away from the write pointer,
 *      two overlapping read-grains are crossfaded — at any time one grain is
 *      "fading in" while the other is "fading out". When a grain reaches its
 *      end-of-window, it gets relocated near the write pointer.
 *
 * This is the standard "windowed lookup" PSOLA-style approach used by Soundtouch.
 * Quality: acceptable for ±12 semitones; some smearing on transients.
 * Latency: one grain length (~46 ms by default).
 *
 * AudioWorkletParams:
 *   - semitones [-24..24]   integer-ish; fractional OK
 *   - mix       [0..1]      wet amount
 *
 * @author Doseedo R1
 */

const GRAIN_MS = 46;        // grain length in ms
const CROSSFADE_RATIO = 0.5; // half of the grain is crossfade
const MAX_BUF_SECONDS = 0.5; // half-second circular buffer is plenty

class PitchShiftProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'semitones', defaultValue: 0, minValue: -24, maxValue: 24, automationRate: 'k-rate' },
      { name: 'mix',       defaultValue: 1, minValue:   0, maxValue:  1, automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    const sr = sampleRate;
    this._grainSize = Math.floor((GRAIN_MS / 1000) * sr);
    this._bufLen = Math.max(this._grainSize * 4, Math.ceil(MAX_BUF_SECONDS * sr));

    // Stereo buffers — process L/R independently so stereo image is preserved
    this._buf = [new Float32Array(this._bufLen), new Float32Array(this._bufLen)];
    this._writeIdx = 0;

    // Two grain read positions (fractional sample indices)
    this._readA = 0;
    this._readB = this._grainSize / 2;
    // Position within grain (0..grainSize) — used for window
    this._posA = 0;
    this._posB = this._grainSize / 2;

    this._lastShift = 1;
  }

  // Hann-like window over [0..grainSize) producing fade-in/out shape.
  _window(pos, grainSize) {
    const t = pos / grainSize; // 0..1
    return 0.5 * (1 - Math.cos(2 * Math.PI * t));
  }

  // Read a sample from circular buffer at fractional index with linear interp
  _readBuf(channel, fracIdx) {
    const bufLen = this._bufLen;
    let idx = fracIdx;
    while (idx < 0) idx += bufLen;
    while (idx >= bufLen) idx -= bufLen;
    const i0 = Math.floor(idx);
    const i1 = (i0 + 1) % bufLen;
    const f = idx - i0;
    const buf = this._buf[channel];
    return buf[i0] * (1 - f) + buf[i1] * f;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!output || !output.length) return true;

    const semitones = parameters.semitones[0];
    const mix = parameters.mix[0];
    const shift = Math.pow(2, semitones / 12);

    const grainSize = this._grainSize;
    const bufLen = this._bufLen;
    const blockSize = output[0].length;

    const inL = input && input[0] ? input[0] : null;
    const inR = input && input[1] ? input[1] : inL;
    const outL = output[0];
    const outR = output[1] || output[0];

    let writeIdx = this._writeIdx;
    let readA = this._readA;
    let readB = this._readB;
    let posA = this._posA;
    let posB = this._posB;

    const dryGain = 1 - mix;
    const wetGain = mix;

    for (let i = 0; i < blockSize; i++) {
      const l = inL ? inL[i] : 0;
      const r = inR ? inR[i] : l;

      // Write to circular buffer (stereo)
      this._buf[0][writeIdx] = l;
      this._buf[1][writeIdx] = r;

      // Read grains
      const wA = this._window(posA, grainSize);
      const wB = this._window(posB, grainSize);

      const aL = this._readBuf(0, readA);
      const aR = this._readBuf(1, readA);
      const bL = this._readBuf(0, readB);
      const bR = this._readBuf(1, readB);

      // Normalize so windowed sum has unity gain at center crossover
      const wetL = aL * wA + bL * wB;
      const wetR = aR * wA + bR * wB;

      outL[i] = l * dryGain + wetL * wetGain;
      outR[i] = r * dryGain + wetR * wetGain;

      // Advance read positions by `shift` (fractional sample step)
      readA += shift;
      readB += shift;
      posA += 1; // grain time advances at audio rate
      posB += 1;

      // Wrap reads in buffer
      while (readA >= bufLen) readA -= bufLen;
      while (readA < 0) readA += bufLen;
      while (readB >= bufLen) readB -= bufLen;
      while (readB < 0) readB += bufLen;

      // When a grain reaches end-of-window, relocate near write pointer.
      // Stagger so the two grains are 50% out of phase.
      if (posA >= grainSize) {
        posA = 0;
        // Reset to "behind" the write pointer by ~quarter grain
        readA = (writeIdx - grainSize * 0.25 + bufLen) % bufLen;
      }
      if (posB >= grainSize) {
        posB = 0;
        readB = (writeIdx - grainSize * 0.25 + bufLen) % bufLen;
      }

      writeIdx += 1;
      if (writeIdx >= bufLen) writeIdx = 0;
    }

    this._writeIdx = writeIdx;
    this._readA = readA;
    this._readB = readB;
    this._posA = posA;
    this._posB = posB;
    return true;
  }
}

registerProcessor('r1-pitch-shift-processor', PitchShiftProcessor);
