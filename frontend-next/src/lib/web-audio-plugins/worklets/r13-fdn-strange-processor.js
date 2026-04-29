/**
 * r13-fdn-strange-processor — modulated, non-stationary "Strange Room /
 * Synth Hall" FDN reverb.
 *
 * Character: each delay line has its own slow LFO modulating the read tap
 * (fractional-delay interpolation) which gives a chorused, drifting tail.
 * Modest input diffusion (1 AP) preserves transient identity. FDN order 6
 * via Householder mixing — non-power-of-2 size further reduces metallic
 * coloration vs. the standard Hadamard FDNs.
 *
 * The LFO depths/rates are mutually irrational so the modulation never
 * phase-locks — that's what makes the tail feel "alive" rather than rigid.
 */

const FDN_ORDER = 6;
const MAX_PRE_DELAY_MS = 500;
const MAX_DELAY_LINE_MS = 200;
const MOD_DEPTH_MAX_MS = 6;   // peak read-tap excursion per line

// Mid-length, mutually-prime base delays
const BASE_DELAYS_MS = [ 19.7, 27.3, 33.1, 41.7, 53.3, 67.1 ];

// Per-line LFO rates (Hz). Irrational ratios — no GCD with any other rate.
const LFO_RATES_HZ = [ 0.31, 0.47, 0.61, 0.79, 1.03, 1.27 ];

// Per-line LFO depths (ms). Smaller for short delays so we never read into
// the future relative to the write head.
const LFO_DEPTHS_MS = [ 1.7, 2.3, 2.9, 3.5, 4.1, 4.7 ];

const HF_TARGET_HZ = 7000;
const INPUT_AP_DELAYS_MS = [ 8.51 ];
const INPUT_AP_COEF = 0.5;

function pow2Ceil(n) { let p = 1; while (p < n) p <<= 1; return p; }

class FdnStrangeProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'decay_time', defaultValue: 2.5, minValue: 0.1, maxValue: 20.0, automationRate: 'k-rate' },
      { name: 'pre_delay',  defaultValue: 0,   minValue: 0,   maxValue: MAX_PRE_DELAY_MS, automationRate: 'k-rate' },
      { name: 'damping',    defaultValue: 0.3, minValue: 0,   maxValue: 1.0,  automationRate: 'k-rate' },
      { name: 'diffusion',  defaultValue: 0.6, minValue: 0,   maxValue: 1.0,  automationRate: 'k-rate' },
      { name: 'width',      defaultValue: 0.9, minValue: 0,   maxValue: 1.0,  automationRate: 'k-rate' },
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

    // Add headroom so LFO modulation cannot push the read tap past the buffer
    const lineLen = pow2Ceil(Math.ceil(((MAX_DELAY_LINE_MS + MOD_DEPTH_MAX_MS) / 1000) * sr) + 4);
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

    // LFO state (per line). Stored as phase 0..1
    this._lfoPhase = new Float32Array(FDN_ORDER);
    this._lfoInc   = new Float32Array(FDN_ORDER);
    this._lfoDepthSamps = new Float32Array(FDN_ORDER);
    for (let i = 0; i < FDN_ORDER; i++) {
      this._lfoInc[i] = LFO_RATES_HZ[i] / sr;
      this._lfoDepthSamps[i] = (LFO_DEPTHS_MS[i] / 1000) * sr;
      // Stagger initial phases so the lines don't all start aligned
      this._lfoPhase[i] = i / FDN_ORDER;
    }

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
    this._configureDelays(2.5);
    this._setDamping(0.3);
  }

  _configureDelays(decay) {
    const sr = sampleRate;
    for (let i = 0; i < FDN_ORDER; i++) {
      const dms = BASE_DELAYS_MS[i];
      const dsamps = Math.max(1, Math.round((dms / 1000) * sr));
      // Reserve headroom for LFO excursion
      this._delaySamps[i] = Math.min(dsamps, this._lineLen - 4 - Math.ceil(this._lfoDepthSamps[i]));
      const dt = dms / 1000;
      const g = Math.pow(10, (-3 * dt) / Math.max(0.05, decay));
      this._fbGain[i] = Math.min(0.999, g);
    }
    this._curDecay = decay;
  }

  _setDamping(damp) {
    const fc = Math.max(200, HF_TARGET_HZ * Math.pow(0.04, damp));
    this._lpA = Math.exp(-2 * Math.PI * fc / sampleRate);
    this._curDamp = damp;
  }

  // Householder reflection mixer (real symmetric orthogonal). Cheaper than
  // Hadamard for non-power-of-2 N and energy-preserving.
  _householder6(v) {
    let sum = 0;
    for (let i = 0; i < 6; i++) sum += v[i];
    const k = (2 / 6) * sum;
    for (let i = 0; i < 6; i++) v[i] = v[i] - k;
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
    const lfoPhase = this._lfoPhase;
    const lfoInc = this._lfoInc;
    const lfoDepth = this._lfoDepthSamps;
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

      // FDN tap read with fractional-delay LFO modulation per line
      for (let i = 0; i < FDN_ORDER; i++) {
        // Sine LFO from phase
        const lfo = Math.sin(2 * Math.PI * lfoPhase[i]);
        lfoPhase[i] += lfoInc[i];
        if (lfoPhase[i] >= 1) lfoPhase[i] -= 1;

        const modOffset = lfo * lfoDepth[i];
        const totalDelay = delaySamps[i] + modOffset;
        const dInt = Math.floor(totalDelay);
        const dFrac = totalDelay - dInt;
        const r0 = (writeIdx - dInt) & lineMask;
        const r1 = (writeIdx - dInt - 1) & lineMask;
        const raw = lines[i][r0] * (1 - dFrac) + lines[i][r1] * dFrac;
        const z = oneMinusA * raw + lpA * lpZ[i];
        lpZ[i] = z;
        tapOut[i] = z;
      }

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

      for (let i = 0; i < FDN_ORDER; i++) tapIn[i] = tapOut[i];
      this._householder6(tapIn);

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

registerProcessor('r13-fdn-strange-processor', FdnStrangeProcessor);
