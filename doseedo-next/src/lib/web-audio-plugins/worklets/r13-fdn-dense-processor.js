/**
 * r13-fdn-dense-processor — high-density "Bright Room / Drum Plate" FDN reverb.
 *
 * Character: maximum density (FDN order 16 — double R9's largest), tightly
 * packed short delays (3–22 ms), aggressive 6-stage Schroeder pre-diffuser,
 * bright HF target. Decays fast but the modal density is so high that
 * transients smear immediately into a thick wash — ideal for drum bus /
 * percussive content.
 *
 * Order-16 Hadamard mixer is built via fast Walsh–Hadamard (4 butterfly
 * stages, all in-place). Per-sample cost is roughly 2× R9 hall but still
 * comfortably real-time at 48 kHz.
 */

const FDN_ORDER = 16;
const MAX_PRE_DELAY_MS = 500;
const MAX_DELAY_LINE_MS = 60;

// 16 short, mutually-prime base delays (ms). Picked to span a narrow
// 3–22 ms band — that's what gives the drum-plate density.
const BASE_DELAYS_MS = [
   3.1,  4.7,  5.3,  6.7,  7.9,  9.1, 10.7, 12.3,
  13.7, 14.9, 16.3, 17.7, 18.9, 19.7, 20.9, 22.1,
];

const HF_TARGET_HZ = 13000;  // very bright

// 6-stage Schroeder diffuser cascade — more aggressive than R9 plate's 4
const INPUT_AP_DELAYS_MS = [ 2.3, 3.7, 5.1, 7.3, 9.7, 13.1 ];
const INPUT_AP_COEF = 0.65;

function pow2Ceil(n) { let p = 1; while (p < n) p <<= 1; return p; }

class FdnDenseProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'decay_time', defaultValue: 1.5, minValue: 0.1, maxValue: 20.0, automationRate: 'k-rate' },
      { name: 'pre_delay',  defaultValue: 0,   minValue: 0,   maxValue: MAX_PRE_DELAY_MS, automationRate: 'k-rate' },
      { name: 'damping',    defaultValue: 0.2, minValue: 0,   maxValue: 1.0,  automationRate: 'k-rate' },
      { name: 'diffusion',  defaultValue: 1.0, minValue: 0,   maxValue: 1.0,  automationRate: 'k-rate' },
      { name: 'width',      defaultValue: 0.7, minValue: 0,   maxValue: 1.0,  automationRate: 'k-rate' },
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

    this._curDecay = -1;
    this._curDamp  = -1;
    this._hadScale16 = 1.0 / Math.sqrt(16);
    this._configureDelays(1.5);
    this._setDamping(0.2);
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
    const fc = Math.max(300, HF_TARGET_HZ * Math.pow(0.04, damp));
    this._lpA = Math.exp(-2 * Math.PI * fc / sampleRate);
    this._curDamp = damp;
  }

  // Walsh-Hadamard on 16 elements: 4 butterfly stages with strides 1,2,4,8.
  _hadamard16(v) {
    let a, b;
    // stage 1, stride 1
    for (let i = 0; i < 16; i += 2) { a = v[i]; b = v[i + 1]; v[i] = a + b; v[i + 1] = a - b; }
    // stage 2, stride 2
    for (let i = 0; i < 16; i += 4) {
      a = v[i];     b = v[i + 2]; v[i]     = a + b; v[i + 2] = a - b;
      a = v[i + 1]; b = v[i + 3]; v[i + 1] = a + b; v[i + 3] = a - b;
    }
    // stage 3, stride 4
    for (let i = 0; i < 16; i += 8) {
      for (let j = 0; j < 4; j++) {
        a = v[i + j]; b = v[i + j + 4];
        v[i + j] = a + b; v[i + j + 4] = a - b;
      }
    }
    // stage 4, stride 8
    for (let j = 0; j < 8; j++) {
      a = v[j]; b = v[j + 8];
      v[j] = a + b; v[j + 8] = a - b;
    }
    const s = this._hadScale16;
    for (let i = 0; i < 16; i++) v[i] *= s;
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

      this._preL[preIdx] = drySL;
      this._preR[preIdx] = drySR;
      const readPre = (preIdx - preDelaySamps) & this._preMask;
      const pdMono = (this._preL[readPre] + this._preR[readPre]) * 0.5;
      preIdx = (preIdx + 1) & this._preMask;

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

      for (let i = 0; i < FDN_ORDER; i++) {
        const ridx = (writeIdx - delaySamps[i]) & lineMask;
        const raw = lines[i][ridx];
        const z = oneMinusA * raw + lpA * lpZ[i];
        lpZ[i] = z;
        tapOut[i] = z;
      }

      // Stereo decorrelation: split the 16 taps in halves rather than
      // interleaved even/odd — gives a wider perceived field for dense FDN.
      let sumAll = 0, sumLeft = 0, sumRight = 0;
      for (let i = 0; i < FDN_ORDER; i++) {
        const v = tapOut[i];
        sumAll += v;
        if (i < FDN_ORDER / 2) sumLeft += v; else sumRight += v;
      }
      const monoOut = sumAll * invSqrtN;
      const sideOut = (sumLeft - sumRight) * invSqrtN;
      const wetL = monoOut + width * sideOut;
      const wetR = monoOut - width * sideOut;

      for (let i = 0; i < FDN_ORDER; i++) tapIn[i] = tapOut[i];
      this._hadamard16(tapIn);

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

registerProcessor('r13-fdn-dense-processor', FdnDenseProcessor);
