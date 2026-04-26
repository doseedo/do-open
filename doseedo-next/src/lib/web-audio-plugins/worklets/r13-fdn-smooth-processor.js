/**
 * r13-fdn-smooth-processor — long, gentle "Vocal Hall / Dark Hall" FDN reverb.
 *
 * Character: high diffusion (4 input allpasses + Hadamard FB), gentle
 * damping curve biased toward LF transparency, long delay set (60–160 ms)
 * to give a wide, slow build-up. Stereo decorrelation via even/odd tap
 * differencing (same trick as R9). Fixed FDN order 8 (Hadamard).
 *
 * Differs from R9 hall: longer base delays, stronger input diffusion,
 * darker default HF target — feels closer to a long plate / vocal hall
 * rather than R9's mid-sized concert hall.
 */

const FDN_ORDER = 8;
const MAX_PRE_DELAY_MS = 500;
const MAX_DELAY_LINE_MS = 250;

// Long, mutually-prime base delays (ms). The wider spread vs. R9 hall gives
// a slower modal density build-up — perceived as "smoother".
const BASE_DELAYS_MS = [ 61.3, 73.7, 89.1, 97.3, 113.7, 127.1, 139.7, 157.3 ];

// HF target at damping=0; very high since the variant is always somewhat dark
const HF_TARGET_HZ = 5500;

// Schroeder allpass diffusers — 4 stages, longer than R9's set for denser early diffusion
const INPUT_AP_DELAYS_MS = [ 6.13, 9.71, 14.27, 21.97 ];
const INPUT_AP_COEF = 0.6;

function pow2Ceil(n) { let p = 1; while (p < n) p <<= 1; return p; }

class FdnSmoothProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'decay_time', defaultValue: 3.5, minValue: 0.1, maxValue: 20.0, automationRate: 'k-rate' },
      { name: 'pre_delay',  defaultValue: 0,   minValue: 0,   maxValue: MAX_PRE_DELAY_MS, automationRate: 'k-rate' },
      { name: 'damping',    defaultValue: 0.5, minValue: 0,   maxValue: 1.0,  automationRate: 'k-rate' },
      { name: 'diffusion',  defaultValue: 0.9, minValue: 0,   maxValue: 1.0,  automationRate: 'k-rate' },
      { name: 'width',      defaultValue: 0.8, minValue: 0,   maxValue: 1.0,  automationRate: 'k-rate' },
      { name: 'mix',        defaultValue: 0.3, minValue: 0,   maxValue: 1.0,  automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    const sr = sampleRate;

    this._preLen = pow2Ceil(Math.ceil((MAX_PRE_DELAY_MS / 1000) * sr) + 4);
    this._preMask = this._preLen - 1;
    this._preL = new Float32Array(this._preLen);
    this._preR = new Float32Array(this._preLen);
    this._preIdx = 0;

    const lineLen = pow2Ceil(Math.ceil((MAX_DELAY_LINE_MS / 1000) * sr) + 4);
    this._lineLen = lineLen;
    this._lineMask = lineLen - 1;
    this._lines = [];
    for (let i = 0; i < FDN_ORDER; i++) this._lines.push(new Float32Array(lineLen));
    this._writeIdx = 0;

    this._delaySamps = new Float32Array(FDN_ORDER);
    this._fbGain     = new Float32Array(FDN_ORDER);
    this._lpZ        = new Float32Array(FDN_ORDER);
    this._lpA        = 0.5;
    this._tapOut     = new Float32Array(FDN_ORDER);
    this._tapIn      = new Float32Array(FDN_ORDER);

    this._apLines = [];
    this._apMasks = [];
    this._apDelays = [];
    this._apIdx = new Int32Array(INPUT_AP_DELAYS_MS.length);
    for (let i = 0; i < INPUT_AP_DELAYS_MS.length; i++) {
      const apLen = pow2Ceil(Math.ceil((INPUT_AP_DELAYS_MS[i] / 1000) * sr) + 4);
      this._apLines.push(new Float32Array(apLen));
      this._apMasks.push(apLen - 1);
      this._apDelays.push(Math.round((INPUT_AP_DELAYS_MS[i] / 1000) * sr));
    }

    this._curDecay  = -1;
    this._curDamp   = -1;
    this._hadScale8 = 1.0 / Math.sqrt(8);

    this._configureDelays(3.5);
    this._setDamping(0.5);
  }

  _configureDelays(decay) {
    const sr = sampleRate;
    for (let i = 0; i < FDN_ORDER; i++) {
      const dms = BASE_DELAYS_MS[i];
      const dsamps = Math.max(1, Math.round((dms / 1000) * sr));
      this._delaySamps[i] = Math.min(dsamps, this._lineLen - 4);
      const dt = dms / 1000;
      const g = Math.pow(10, (-3 * dt) / Math.max(0.05, decay));
      this._fbGain[i] = Math.min(0.999, g);
    }
    this._curDecay = decay;
  }

  _setDamping(damp) {
    // Skewed: even at damp=0 we already roll off above ~5.5 kHz (the
    // "smooth/dark hall" character). damp=1 → ~250 Hz.
    const fc = Math.max(250, HF_TARGET_HZ * Math.pow(0.045, damp));
    this._lpA = Math.exp(-2 * Math.PI * fc / sampleRate);
    this._curDamp = damp;
  }

  _hadamard8(v) {
    let a, b;
    for (let i = 0; i < 8; i += 2) { a = v[i]; b = v[i + 1]; v[i] = a + b; v[i + 1] = a - b; }
    for (let i = 0; i < 8; i += 4) {
      a = v[i];     b = v[i + 2]; v[i]     = a + b; v[i + 2] = a - b;
      a = v[i + 1]; b = v[i + 3]; v[i + 1] = a + b; v[i + 3] = a - b;
    }
    for (let i = 0; i < 4; i++) { a = v[i]; b = v[i + 4]; v[i] = a + b; v[i + 4] = a - b; }
    const s = this._hadScale8;
    for (let i = 0; i < 8; i++) v[i] *= s;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!output || output.length < 1) return true;

    const decay = parameters.decay_time[0];
    const preDelayMs = parameters.pre_delay[0];
    const damping = parameters.damping[0];
    const diffusion = Math.max(0, Math.min(1, parameters.diffusion[0]));
    const width = parameters.width[0];
    const mix = parameters.mix[0];

    if (Math.abs(decay - this._curDecay) > 1e-4) this._configureDelays(decay);
    if (Math.abs(damping - this._curDamp) > 1e-4) this._setDamping(damping);

    const blockSize = output[0].length;
    const inL = (input && input[0]) ? input[0] : null;
    const inR = (input && input[1]) ? input[1] : inL;
    const outL = output[0];
    const outR = output[1] || output[0];

    const wet = mix;
    const dry = 1 - mix;

    const preDelaySamps = Math.min(this._preLen - 4, Math.round((preDelayMs / 1000) * sampleRate));
    const lpA = this._lpA;
    const oneMinusA = 1 - lpA;
    const oneMinusDiff = 1 - diffusion;
    const invSqrtN = 1 / Math.sqrt(FDN_ORDER);

    const lines = this._lines;
    const lineMask = this._lineMask;
    const delaySamps = this._delaySamps;
    const fbGain = this._fbGain;
    const lpZ = this._lpZ;
    const tapOut = this._tapOut;
    const tapIn = this._tapIn;
    let writeIdx = this._writeIdx;
    let preIdx = this._preIdx;

    for (let n = 0; n < blockSize; n++) {
      const drySL = inL ? inL[n] : 0;
      const drySR = inR ? inR[n] : drySL;

      // Pre-delay
      this._preL[preIdx] = drySL;
      this._preR[preIdx] = drySR;
      const readPre = (preIdx - preDelaySamps) & this._preMask;
      const pdMono = (this._preL[readPre] + this._preR[readPre]) * 0.5;
      preIdx = (preIdx + 1) & this._preMask;

      // 4-stage Schroeder diffuser
      let injection = pdMono;
      for (let a = 0; a < this._apLines.length; a++) {
        const apBuf = this._apLines[a];
        const apMask = this._apMasks[a];
        const apDel = this._apDelays[a];
        let apIdx = this._apIdx[a];
        const readIdx = (apIdx - apDel) & apMask;
        const delayed = apBuf[readIdx];
        const out = -INPUT_AP_COEF * injection + delayed;
        apBuf[apIdx] = injection + INPUT_AP_COEF * delayed;
        this._apIdx[a] = (apIdx + 1) & apMask;
        injection = out;
      }

      // FDN tap read + per-line LPF
      for (let i = 0; i < FDN_ORDER; i++) {
        const ridx = (writeIdx - delaySamps[i]) & lineMask;
        const raw = lines[i][ridx];
        const z = oneMinusA * raw + lpA * lpZ[i];
        lpZ[i] = z;
        tapOut[i] = z;
      }

      // Stereo output: mono + decorrelated side
      let sumAll = 0, sumEven = 0, sumOdd = 0;
      for (let i = 0; i < FDN_ORDER; i++) {
        const v = tapOut[i];
        sumAll += v;
        if ((i & 1) === 0) sumEven += v; else sumOdd += v;
      }
      const monoOut = sumAll * invSqrtN;
      const sideOut = (sumEven - sumOdd) * invSqrtN;
      const wetL = monoOut + width * sideOut;
      const wetR = monoOut - width * sideOut;

      // Hadamard mix + diffusion crossfade
      for (let i = 0; i < FDN_ORDER; i++) tapIn[i] = tapOut[i];
      this._hadamard8(tapIn);

      const injNorm = injection * invSqrtN;
      for (let i = 0; i < FDN_ORDER; i++) {
        const fb = oneMinusDiff * tapOut[i] + diffusion * tapIn[i];
        lines[i][writeIdx] = injNorm + fb * fbGain[i];
      }

      writeIdx = (writeIdx + 1) & lineMask;

      outL[n] = drySL * dry + wetL * wet;
      outR[n] = drySR * dry + wetR * wet;
    }

    this._writeIdx = writeIdx;
    this._preIdx = preIdx;
    return true;
  }
}

registerProcessor('r13-fdn-smooth-processor', FdnSmoothProcessor);
