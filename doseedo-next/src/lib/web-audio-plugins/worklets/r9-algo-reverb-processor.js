/**
 * r9-algo-reverb-processor — multi-algorithm Feedback Delay Network reverb
 *
 * A ChromaVerb-style algorithmic reverb supporting 4 selectable topologies via
 * the `algorithm` parameter:
 *
 *   0 = room    → 4-line FDN, short delays (~7-30 ms),  modest feedback, 0 input APs
 *   1 = hall    → 8-line FDN, long  delays (~30-100ms), high feedback,   2 input APs
 *   2 = chamber → 6-line FDN, mid   delays (~15-50 ms), mid feedback,    1 input AP
 *   3 = plate   → 8-line FDN, short delays (~5-25 ms),  high feedback,   4 input APs
 *
 * The feedback matrix is a Hadamard mix (orthogonal => energy-preserving)
 * scaled by per-line decay gains derived from RT60 = `decay_time`. A
 * `diffusion` knob crossfades between identity (no mixing) and full Hadamard.
 *
 *   H4 = (1/2) [ 1  1  1  1 ;  1 -1  1 -1 ;  1  1 -1 -1 ;  1 -1 -1  1 ]
 *   H8 = block-recursive 2x scaling of H4 (Sylvester construction).
 *
 * Per-line damping is a 1-pole low-pass placed inside each feedback tap.
 * Pre-delay is a circular buffer applied before the network.
 * Plate algorithm prepends a 4-allpass diffusion cascade for early density
 * and uses a brighter HF target. Stereo decorrelation comes from picking
 * different delay-line subsets for L/R and from the `width` knob crossfading
 * between mono and stereo-decorrelated taps.
 *
 * Per-sample cost (8 lines, plate worst case):
 *   8 delay reads + 8 LPFs + 64 mults for Hadamard + 8 writes + 4 input APs
 *   ≈ 100 mults/sample = ~5 MCps at 48 kHz. Comfortably within budget.
 *
 * No allocations inside process(). Delay lines are power-of-2 ring buffers
 * indexed via bit-mask. Worklet is self-contained (no importScripts).
 *
 * Author: Doseedo R9
 */

const ALGO_ROOM    = 0;
const ALGO_HALL    = 1;
const ALGO_CHAMBER = 2;
const ALGO_PLATE   = 3;

// Per-algorithm base delay lengths in milliseconds. Mutually-prime-ish to
// avoid metallic resonances. Length of each array dictates FDN order.
const ALGO_DELAYS = {
  [ALGO_ROOM]:    [ 7.3, 11.7, 17.9, 23.1 ],
  [ALGO_HALL]:    [ 29.7, 37.1, 41.1, 43.7, 53.1, 67.3, 79.1, 97.7 ],
  [ALGO_CHAMBER]: [ 13.7, 19.1, 23.3, 31.7, 41.3, 49.7 ],
  [ALGO_PLATE]:   [  4.7,  6.1,  7.9,  9.7, 12.3, 15.1, 19.3, 23.7 ],
};

// Default damping target frequency (used as upper bound when damping=0)
const ALGO_HF_TARGET = {
  [ALGO_ROOM]:    8000,
  [ALGO_HALL]:    6000,
  [ALGO_CHAMBER]: 7000,
  [ALGO_PLATE]:  12000,  // brighter
};

// Number of input allpass diffusers per algorithm
const ALGO_INPUT_APS = {
  [ALGO_ROOM]:    0,
  [ALGO_HALL]:    2,
  [ALGO_CHAMBER]: 1,
  [ALGO_PLATE]:   4,
};

// Allpass diffuser delay times (ms) — Schroeder-style, prime-ratio'd
const INPUT_AP_DELAYS = [ 4.77, 3.59, 12.73, 9.31 ];
const INPUT_AP_COEF   = 0.5;

const MAX_FDN_ORDER     = 8;
const MAX_PRE_DELAY_MS  = 500;
const MAX_DELAY_LINE_MS = 200;   // longest per-line delay we'll ever need

// Round up to next power of 2 (for bit-mask circular indexing)
function pow2Ceil(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

class AlgoReverbProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'algorithm',  defaultValue: ALGO_HALL, minValue: 0, maxValue: 3, automationRate: 'k-rate' },
      { name: 'decay_time', defaultValue: 2.5, minValue: 0.1, maxValue: 20.0, automationRate: 'k-rate' },
      { name: 'pre_delay',  defaultValue: 0,   minValue: 0,   maxValue: MAX_PRE_DELAY_MS, automationRate: 'k-rate' },
      { name: 'damping',    defaultValue: 0.4, minValue: 0,   maxValue: 1.0,  automationRate: 'k-rate' },
      { name: 'diffusion',  defaultValue: 0.8, minValue: 0,   maxValue: 1.0,  automationRate: 'k-rate' },
      { name: 'width',      defaultValue: 0.7, minValue: 0,   maxValue: 1.0,  automationRate: 'k-rate' },
      { name: 'mix',        defaultValue: 0.3, minValue: 0,   maxValue: 1.0,  automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    const sr = sampleRate;

    // ── Pre-delay (stereo, single circular buffer per channel) ────────────
    this._preLen = pow2Ceil(Math.ceil((MAX_PRE_DELAY_MS / 1000) * sr) + 4);
    this._preMask = this._preLen - 1;
    this._preL = new Float32Array(this._preLen);
    this._preR = new Float32Array(this._preLen);
    this._preIdx = 0;

    // ── FDN delay lines ───────────────────────────────────────────────────
    // One ring buffer per line, shared across stereo (mono FDN core; stereo
    // decorrelation comes from output tap selection + width knob). This is
    // standard practice for FDN reverbs and saves ~50% of memory + cost.
    const lineLen = pow2Ceil(Math.ceil((MAX_DELAY_LINE_MS / 1000) * sr) + 4);
    this._lineLen = lineLen;
    this._lineMask = lineLen - 1;
    this._lines = [];
    for (let i = 0; i < MAX_FDN_ORDER; i++) {
      this._lines.push(new Float32Array(lineLen));
    }
    this._writeIdx = 0;

    // Per-line state: delay (samples), feedback gain, lowpass z^-1
    this._delaySamps = new Float32Array(MAX_FDN_ORDER);
    this._fbGain     = new Float32Array(MAX_FDN_ORDER);
    this._lpZ        = new Float32Array(MAX_FDN_ORDER);
    this._lpA        = 0.5; // recomputed from damping
    // Scratch buffers for FDN tap I/O (avoid allocation in process)
    this._tapOut     = new Float32Array(MAX_FDN_ORDER);
    this._tapIn      = new Float32Array(MAX_FDN_ORDER);

    // ── Input allpass diffusers (up to 4, plate uses all) ─────────────────
    this._apLines = [];
    this._apMasks = [];
    this._apDelays = [];
    this._apIdx = new Int32Array(4);
    for (let i = 0; i < 4; i++) {
      const apLen = pow2Ceil(Math.ceil((INPUT_AP_DELAYS[i] / 1000) * sr) + 4);
      this._apLines.push(new Float32Array(apLen));
      this._apMasks.push(apLen - 1);
      this._apDelays.push(Math.round((INPUT_AP_DELAYS[i] / 1000) * sr));
    }

    // ── Cached parameter state (so we only recompute when changed) ────────
    this._curAlgo   = -1;
    this._curDecay  = -1;
    this._curDamp   = -1;
    this._curDiff   = -1;
    this._fdnOrder  = 4;
    this._diffMix   = 0.0;   // 0=identity, 1=full Hadamard

    // Hadamard scaling factor: H_N has rows of magnitude 1, so 1/sqrt(N) keeps
    // unit-norm => energy-preserving.
    this._hadScale4 = 1.0 / Math.sqrt(4);
    this._hadScale8 = 1.0 / Math.sqrt(8);
    this._hadScale6 = 1.0 / Math.sqrt(6); // approx — we use a hybrid 6x6
  }

  // ── Algorithm reconfig: delays, feedback gains ───────────────────────────
  _configureAlgorithm(algo, decay) {
    const sr = sampleRate;
    const delays = ALGO_DELAYS[algo] || ALGO_DELAYS[ALGO_HALL];
    const order = delays.length;
    this._fdnOrder = order;

    for (let i = 0; i < order; i++) {
      const dms = delays[i];
      const dsamps = Math.max(1, Math.round((dms / 1000) * sr));
      // Clamp to ring-buffer length minus a guard
      this._delaySamps[i] = Math.min(dsamps, this._lineLen - 4);
      // Per-line feedback gain: g = 10^(-3 * delay / RT60)  (Jot/Sabine)
      const dt = dms / 1000;
      const g = Math.pow(10, (-3 * dt) / Math.max(0.05, decay));
      this._fbGain[i] = Math.min(0.999, g);
    }
    // Zero unused taps so they don't leak from a previous algo
    for (let i = order; i < MAX_FDN_ORDER; i++) {
      this._delaySamps[i] = 0;
      this._fbGain[i] = 0;
      this._lpZ[i] = 0;
    }
    this._curAlgo = algo;
    this._curDecay = decay;
  }

  // Convert damping [0..1] to 1-pole LPF coefficient (one-pole IIR:
  //   y[n] = (1-a) * x[n] + a * y[n-1]
  // a = exp(-2π * fc / sr)
  _setDamping(damp, algo) {
    const hfTarget = ALGO_HF_TARGET[algo] || 6000;
    // damping=0 → fc = hfTarget (transparent), damping=1 → fc ~= 200 Hz (very dark)
    const fc = Math.max(200, hfTarget * Math.pow(0.04, damp));
    this._lpA = Math.exp(-2 * Math.PI * fc / sampleRate);
    this._curDamp = damp;
  }

  // ── Hadamard mixers (in-place using _tapOut → _tapIn) ─────────────────────
  // H4 = (1/2) [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
  // We compute via fast Walsh-Hadamard (in-place butterflies) for clarity.
  _hadamard4(v) {
    // Stage 1: pairs (0,1) (2,3)
    let a = v[0], b = v[1];
    v[0] = a + b; v[1] = a - b;
    a = v[2]; b = v[3];
    v[2] = a + b; v[3] = a - b;
    // Stage 2: pairs (0,2) (1,3)
    a = v[0]; b = v[2];
    v[0] = a + b; v[2] = a - b;
    a = v[1]; b = v[3];
    v[1] = a + b; v[3] = a - b;
    // Normalize 1/sqrt(4) = 1/2
    const s = this._hadScale4;
    v[0] *= s; v[1] *= s; v[2] *= s; v[3] *= s;
  }

  _hadamard8(v) {
    // Walsh-Hadamard (Sylvester) on 8 elements: 3 butterfly stages
    let a, b;
    // stage 1, stride 1
    for (let i = 0; i < 8; i += 2) {
      a = v[i]; b = v[i + 1];
      v[i] = a + b; v[i + 1] = a - b;
    }
    // stage 2, stride 2
    for (let i = 0; i < 8; i += 4) {
      a = v[i];     b = v[i + 2]; v[i]     = a + b; v[i + 2] = a - b;
      a = v[i + 1]; b = v[i + 3]; v[i + 1] = a + b; v[i + 3] = a - b;
    }
    // stage 3, stride 4
    for (let i = 0; i < 4; i++) {
      a = v[i]; b = v[i + 4];
      v[i] = a + b; v[i + 4] = a - b;
    }
    const s = this._hadScale8;
    for (let i = 0; i < 8; i++) v[i] *= s;
  }

  // 6x6 mixer: Householder reflection  H = I − (2/N) · 1·1ᵀ
  // This is a real symmetric ORTHOGONAL matrix (H·Hᵀ = I) so the mixer is
  // exactly energy-preserving — the gold-standard choice for FDN orders that
  // aren't powers of 2. Stautner-Puckette 1982 + Jot 1991 both use Householder
  // for non-power-of-2 FDNs. Cost: 1 sum + N MACs (cheaper than full Hadamard).
  _hadamard6(v) {
    let sum = 0;
    for (let i = 0; i < 6; i++) sum += v[i];
    const k = (2 / 6) * sum;
    for (let i = 0; i < 6; i++) v[i] = v[i] - k;
  }

  // ── Process loop ──────────────────────────────────────────────────────────
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!output || output.length < 1) return true;

    // ── Param read (k-rate) ───────────────────────────────────────────────
    const algo = Math.max(0, Math.min(3, Math.round(parameters.algorithm[0])));
    const decay = parameters.decay_time[0];
    const preDelayMs = parameters.pre_delay[0];
    const damping = parameters.damping[0];
    const diffusion = parameters.diffusion[0];
    const width = parameters.width[0];
    const mix = parameters.mix[0];

    // Reconfigure on changes
    if (algo !== this._curAlgo || Math.abs(decay - this._curDecay) > 1e-4) {
      this._configureAlgorithm(algo, decay);
    }
    if (Math.abs(damping - this._curDamp) > 1e-4 || algo !== this._curAlgo) {
      this._setDamping(damping, algo);
    }
    this._diffMix = Math.max(0, Math.min(1, diffusion));

    const order = this._fdnOrder;
    const numAPs = ALGO_INPUT_APS[algo] || 0;
    const blockSize = output[0].length;

    const inL = (input && input[0]) ? input[0] : null;
    const inR = (input && input[1]) ? input[1] : inL;
    const outL = output[0];
    const outR = output[1] || output[0];

    const wet = mix;
    const dry = 1 - mix;
    const widthAmt = width;

    const preDelaySamps = Math.min(this._preLen - 4,
      Math.round((preDelayMs / 1000) * sampleRate));
    const lpA = this._lpA;
    const oneMinusA = 1 - lpA;
    const diffMix = this._diffMix;
    const oneMinusDiff = 1 - diffMix;
    const invSqrtOrder = 1 / Math.sqrt(order);

    // Output tap selection: alternating sign + L/R interleave for stereo decorr
    // L = sum over even taps (positive), R = sum over odd taps (mixed sign)
    // We compute mono = sum/N, side = (Σ_even − Σ_odd)/N. width crossfades.

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
      const monoIn = (drySL + drySR) * 0.5;

      // ── 1. Pre-delay ────────────────────────────────────────────────────
      this._preL[preIdx] = drySL;
      this._preR[preIdx] = drySR;
      const readPre = (preIdx - preDelaySamps) & this._preMask;
      const pdL = this._preL[readPre];
      const pdR = this._preR[readPre];
      const pdMono = (pdL + pdR) * 0.5;
      preIdx = (preIdx + 1) & this._preMask;

      // ── 2. Input diffuser cascade (Schroeder allpasses) ─────────────────
      let injection = pdMono;
      for (let a = 0; a < numAPs; a++) {
        const apBuf = this._apLines[a];
        const apMask = this._apMasks[a];
        const apDel = this._apDelays[a];
        let apIdx = this._apIdx[a];
        const readIdx = (apIdx - apDel) & apMask;
        const delayed = apBuf[readIdx];
        // Allpass: y = -g*x + delayed;  store: x + g*delayed
        const out = -INPUT_AP_COEF * injection + delayed;
        apBuf[apIdx] = injection + INPUT_AP_COEF * delayed;
        apIdx = (apIdx + 1) & apMask;
        this._apIdx[a] = apIdx;
        injection = out;
      }

      // ── 3. Read FDN delay-line outputs and apply per-line damping LPF ───
      for (let i = 0; i < order; i++) {
        const d = delaySamps[i];
        const ridx = (writeIdx - d) & lineMask;
        const raw = lines[i][ridx];
        // 1-pole LPF (in feedback path): z = (1-a)*raw + a*z
        const z = oneMinusA * raw + lpA * lpZ[i];
        lpZ[i] = z;
        tapOut[i] = z;
      }

      // ── 4. Compute stereo output from FDN taps (before feedback step) ───
      // Mono sum = average of all taps (orthogonal mixing => unit-RMS scaling)
      let sumAll = 0;
      let sumEven = 0;
      let sumOdd = 0;
      for (let i = 0; i < order; i++) {
        const v = tapOut[i];
        sumAll += v;
        if ((i & 1) === 0) sumEven += v; else sumOdd += v;
      }
      const monoOut = sumAll * invSqrtOrder;
      // Stereo decorrelated component: (even − odd) × normalisation
      const sideOut = (sumEven - sumOdd) * invSqrtOrder;
      // width=0 → fully mono, width=1 → max stereo separation
      const wetL = monoOut + widthAmt * sideOut;
      const wetR = monoOut - widthAmt * sideOut;

      // ── 5. Hadamard feedback mixing (with diffusion crossfade) ──────────
      // tapIn_mixed = H * tapOut, then tapIn = (1-diff)*tapOut + diff*mixed
      // Copy tapOut into tapIn, mix in place, then crossfade against orig
      // We need the original tapOut for the identity branch — keep it.
      // Strategy: load tapIn = tapOut, run Hadamard on tapIn, then blend.
      for (let i = 0; i < order; i++) tapIn[i] = tapOut[i];
      if (order === 8)      this._hadamard8(tapIn);
      else if (order === 6) this._hadamard6(tapIn);
      else if (order === 4) this._hadamard4(tapIn);
      // else: no mixing for unknown sizes (shouldn't happen)

      // diffusion crossfade: 0 = identity (no mixing), 1 = full Hadamard
      // ── 6. Write injection + (mixed feedback × per-line decay gain) ─────
      // Drive injection equally into all lines (1/√N for unit-norm input)
      const injNorm = injection * invSqrtOrder;
      for (let i = 0; i < order; i++) {
        const fb = oneMinusDiff * tapOut[i] + diffMix * tapIn[i];
        lines[i][writeIdx] = injNorm + fb * fbGain[i];
      }

      writeIdx = (writeIdx + 1) & lineMask;

      // ── 7. Mix dry + wet ────────────────────────────────────────────────
      outL[n] = drySL * dry + wetL * wet;
      outR[n] = drySR * dry + wetR * wet;
    }

    this._writeIdx = writeIdx;
    this._preIdx = preIdx;
    return true;
  }
}

registerProcessor('r9-algo-reverb-processor', AlgoReverbProcessor);
