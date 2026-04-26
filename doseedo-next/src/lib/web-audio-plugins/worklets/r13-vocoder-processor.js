/**
 * r13-vocoder-processor — N-band classic phase vocoder
 *
 * Implements a Logic-style Vocoder in a single AudioWorkletProcessor:
 *
 *   modulator (input[0]) ─┬─ analysis bandpass[i] ─ envFollow[i] ──┐
 *                                                                  ▼
 *   carrier   (input[1]) ── synthesis bandpass[i] ─── × env[i] ─ Σ ─ output
 *
 * The carrier may be supplied externally (input[1] non-empty) OR generated
 * internally from the `carrier_type` parameter (saw / square / noise) at
 * `carrier_freq` Hz. Internal generation is selected when the carrier input
 * is silent (RMS below epsilon) over the rendered block.
 *
 * Topology details:
 *   - N (= bands) state-variable BP filters per side (analysis + synthesis).
 *     N is one of {8, 16, 24, 32}; default 16. Center freqs are spaced
 *     logarithmically from 100 Hz to 8000 Hz (Logic Pro Vocoder default).
 *   - Each analysis envelope is a 1-pole asymmetric peak/RMS follower with
 *     attack_ms / release_ms time constants.
 *   - The synthesis bands' centers are shifted by `formant_shift` semitones
 *     relative to analysis ⇒ the timbre of the modulator can be remapped
 *     up / down a male/female vocal range without retuning the carrier.
 *   - `unvoiced_mix` blends a band-limited high-frequency noise into the
 *     synthesis sum, scaled by the upper-band envelopes — this is what
 *     restores sibilants ("s", "sh", "f") that pure-tonal carriers can't
 *     reproduce.
 *   - `mix` is dry/wet between the modulator (dry) and the vocoded sum (wet).
 *
 * Design notes / why one worklet rather than a graph of N×3 nodes:
 *   - Up to 32×3 = 96 BiquadFilterNode instances per slot is too many for the
 *     audio-graph scheduler to handle cleanly when several vocoder slots are
 *     instantiated in a session. Self-contained worklet is O(N) per sample.
 *   - All state lives on Float32Arrays sized to MAX_BANDS, so reconfiguring
 *     `bands` at runtime is allocation-free.
 *
 * Per-sample cost (N=16, voiced):
 *   16 SVF (analysis, 4 mults each) +
 *   16 SVF (synthesis) +
 *   16 env-followers (~3 ops) +
 *   16 multiplies + sum + mix → ~150 mults / sample = ~7 MCps @ 48 kHz.
 *   Comfortably realtime even for 32 bands.
 *
 * Author: Doseedo R13
 */

const MAX_BANDS = 32;

const CARRIER_SAW      = 0;
const CARRIER_SQUARE   = 1;
const CARRIER_NOISE    = 2;
const CARRIER_EXTERNAL = 3;

// Map carrier-type symbol → integer for k-rate parameterDescriptor passing.
// (The builder converts string → integer before assigning.)
function carrierTypeFromValue(v) {
  if (typeof v === 'number') return Math.max(0, Math.min(3, Math.round(v)));
  return CARRIER_SAW;
}

// Logarithmic band-center frequency layout: lo .. hi distributed log-spaced.
// Returns Float32Array of length n.
function logBandCenters(n, lo, hi) {
  const out = new Float32Array(n);
  if (n <= 1) { out[0] = Math.sqrt(lo * hi); return out; }
  const ratio = Math.log(hi / lo) / (n - 1);
  for (let i = 0; i < n; i++) {
    out[i] = lo * Math.exp(i * ratio);
  }
  return out;
}

class VocoderProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      // bands index: 0=8, 1=16, 2=24, 3=32 (k-rate enum). Stored as int.
      { name: 'bands_idx',    defaultValue: 1,   minValue: 0,    maxValue: 3,    automationRate: 'k-rate' },
      { name: 'attack_ms',    defaultValue: 5,   minValue: 0.1,  maxValue: 100,  automationRate: 'k-rate' },
      { name: 'release_ms',   defaultValue: 50,  minValue: 1,    maxValue: 500,  automationRate: 'k-rate' },
      { name: 'formant_shift',defaultValue: 0,   minValue: -12,  maxValue: 12,   automationRate: 'k-rate' },
      // 0=saw,1=square,2=noise,3=external (k-rate enum)
      { name: 'carrier_type', defaultValue: CARRIER_SAW, minValue: 0, maxValue: 3, automationRate: 'k-rate' },
      { name: 'carrier_freq', defaultValue: 110, minValue: 20,   maxValue: 4000, automationRate: 'k-rate' },
      { name: 'mix',          defaultValue: 1.0, minValue: 0,    maxValue: 1,    automationRate: 'k-rate' },
      { name: 'unvoiced_mix', defaultValue: 0.2, minValue: 0,    maxValue: 1,    automationRate: 'k-rate' },
      // Q for the bandpass filters (shared across bands). 4..30 typical.
      { name: 'q',            defaultValue: 12,  minValue: 1,    maxValue: 50,   automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    const sr = sampleRate;
    this._sr = sr;

    // Per-band SVF state (analysis + synthesis). State-variable filter:
    //   bp = q1 ; lp = q2 ; with f = 2 sin(π fc / sr), q = 1/Q
    // Two states per filter.
    this._anaQ1 = new Float32Array(MAX_BANDS);
    this._anaQ2 = new Float32Array(MAX_BANDS);
    this._synQ1 = new Float32Array(MAX_BANDS);
    this._synQ2 = new Float32Array(MAX_BANDS);
    // Per-band SVF coefficient caches
    this._anaF = new Float32Array(MAX_BANDS);
    this._synF = new Float32Array(MAX_BANDS);
    this._anaQc = 1 / 12;
    this._synQc = 1 / 12;

    // Per-band envelope state
    this._env = new Float32Array(MAX_BANDS);
    this._envAttack = 0.01;
    this._envRelease = 0.001;

    // Internal carrier oscillator phase (saw/square)
    this._oscPhase = 0;
    // Pink-ish noise state for the unvoiced (sibilant) blend
    this._noiseLpZ = 0;

    // Cached config
    this._bandCount = 16;
    this._curBandsIdx = -1;
    this._curFormant = -999;
    this._curQ = -1;
    this._anaCenters = logBandCenters(this._bandCount, 100, 8000);
    this._synCenters = new Float32Array(this._anaCenters);
    this._configureBands(1, 0, 12);
    this._setEnvTimes(5, 50);
  }

  _bandsFromIdx(idx) {
    return [8, 16, 24, 32][Math.max(0, Math.min(3, idx | 0))];
  }

  _configureBands(idx, formantSemis, q) {
    const n = this._bandsFromIdx(idx);
    if (idx !== this._curBandsIdx) {
      this._bandCount = n;
      this._anaCenters = logBandCenters(n, 100, 8000);
      // reset state on band-count change to avoid blowup
      this._anaQ1.fill(0); this._anaQ2.fill(0);
      this._synQ1.fill(0); this._synQ2.fill(0);
      this._env.fill(0);
      this._curBandsIdx = idx;
    }
    // Synthesis centers = analysis × 2^(formant/12)
    const shift = Math.pow(2, formantSemis / 12);
    this._synCenters = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      this._synCenters[i] = this._anaCenters[i] * shift;
    }
    // SVF f coefficients per band (f = 2 sin(π fc / sr))
    for (let i = 0; i < n; i++) {
      const fa = 2 * Math.sin(Math.PI * Math.min(this._anaCenters[i], this._sr * 0.45) / this._sr);
      const fs = 2 * Math.sin(Math.PI * Math.min(this._synCenters[i], this._sr * 0.45) / this._sr);
      this._anaF[i] = fa;
      this._synF[i] = fs;
    }
    const qc = 1 / Math.max(0.5, q);
    this._anaQc = qc;
    this._synQc = qc;
    this._curFormant = formantSemis;
    this._curQ = q;
  }

  _setEnvTimes(attackMs, releaseMs) {
    // exp(-1 / (t * sr)) one-pole coefficient
    const sr = this._sr;
    this._envAttack  = Math.exp(-1 / Math.max(1, (attackMs  / 1000) * sr));
    this._envRelease = Math.exp(-1 / Math.max(1, (releaseMs / 1000) * sr));
  }

  // SVF step (Chamberlin / Hal Chamberlin trapezoidal-ish version):
  //   lp += f * bp ; hp = x - lp - q*bp ; bp += f * hp
  //   returns bp
  _svfStep(idx, x, fArr, qc, q1Arr, q2Arr) {
    let bp = q1Arr[idx];
    let lp = q2Arr[idx];
    const f = fArr[idx];
    lp = lp + f * bp;
    const hp = x - lp - qc * bp;
    bp = bp + f * hp;
    q1Arr[idx] = bp;
    q2Arr[idx] = lp;
    return bp;
  }

  process(inputs, outputs, parameters) {
    const out = outputs[0];
    if (!out || out.length === 0) return true;
    const outL = out[0];
    const outR = out.length > 1 ? out[1] : null;
    const blockSize = outL.length;

    // Pull k-rate params (0th frame is the value for the whole block)
    const bandsIdx     = parameters.bands_idx[0]      | 0;
    const attackMs     = parameters.attack_ms[0];
    const releaseMs    = parameters.release_ms[0];
    const formantSemis = parameters.formant_shift[0];
    const carrierType  = parameters.carrier_type[0]   | 0;
    const carrierFreq  = parameters.carrier_freq[0];
    const mixWet       = parameters.mix[0];
    const unvoicedMix  = parameters.unvoiced_mix[0];
    const q            = parameters.q[0];

    // Reconfig if anything changed
    if (bandsIdx !== this._curBandsIdx ||
        formantSemis !== this._curFormant ||
        q !== this._curQ) {
      this._configureBands(bandsIdx, formantSemis, q);
    }
    this._setEnvTimes(attackMs, releaseMs);

    // Modulator input (input[0]) — required
    const modIn = inputs[0];
    const modL = (modIn && modIn[0]) ? modIn[0] : null;
    if (!modL) {
      // No modulator → output silence (don't try to vocode noise).
      for (let i = 0; i < blockSize; i++) {
        outL[i] = 0;
        if (outR) outR[i] = 0;
      }
      return true;
    }
    const modR = (modIn[1]) ? modIn[1] : modL;

    // Carrier input (input[1]) — only used when carrier_type === EXTERNAL.
    const carIn = inputs[1];
    const useExternal = (carrierType === CARRIER_EXTERNAL) && carIn && carIn[0];
    const carL = useExternal ? carIn[0] : null;
    const carR = useExternal ? (carIn[1] || carIn[0]) : null;

    // Internal carrier osc phase increment
    const phaseInc = carrierFreq / this._sr;

    const n = this._bandCount;
    const sr = this._sr;
    // Index from which bands are considered "high" for sibilant blend
    const highStart = Math.max(1, Math.floor(n * 0.66));

    for (let i = 0; i < blockSize; i++) {
      // ── 1. Modulator → mono mix → analysis filter bank → envelopes ──────
      const modSample = 0.5 * (modL[i] + modR[i]);
      // ── 2. Carrier sample ──────────────────────────────────────────────
      let carSample;
      if (useExternal) {
        carSample = 0.5 * (carL[i] + carR[i]);
      } else if (carrierType === CARRIER_NOISE) {
        carSample = Math.random() * 2 - 1;
      } else if (carrierType === CARRIER_SQUARE) {
        // PolyBLEP-free square (acceptable for vocoder use — bands kill aliasing)
        this._oscPhase += phaseInc;
        if (this._oscPhase >= 1) this._oscPhase -= 1;
        carSample = this._oscPhase < 0.5 ? 1 : -1;
      } else { // SAW (default)
        this._oscPhase += phaseInc;
        if (this._oscPhase >= 1) this._oscPhase -= 1;
        carSample = 2 * this._oscPhase - 1;
      }

      // ── 3. Per-band: analysis BP → env follower → synthesis BP × env ───
      let synSum = 0;
      // Pre-generate a fresh sample of HF-emphasis noise once per output sample
      const hfNoise = Math.random() * 2 - 1;

      for (let b = 0; b < n; b++) {
        const anaBP = this._svfStep(b, modSample,
          this._anaF, this._anaQc, this._anaQ1, this._anaQ2);
        // Asymmetric envelope follower (peak-style on |x|)
        const aMag = Math.abs(anaBP);
        const prev = this._env[b];
        const coef = (aMag > prev) ? this._envAttack : this._envRelease;
        const newEnv = aMag + coef * (prev - aMag);
        this._env[b] = newEnv;

        // Synthesis side: BP filter the carrier, then scale by analysis env.
        const synBP = this._svfStep(b, carSample,
          this._synF, this._synQc, this._synQ1, this._synQ2);
        let bandOut = synBP * newEnv;

        // Add HF-noise blend to the upper bands, scaled by their envelopes.
        if (b >= highStart) {
          bandOut += hfNoise * newEnv * unvoicedMix;
        }
        synSum += bandOut;
      }

      // Normalize: dividing by sqrt(N) is good enough; bands are mostly
      // disjoint so summing tends to ~RMS-add not peak-add.
      const wet = synSum * (1.0 / Math.sqrt(n));
      const dry = modSample;
      const y = dry * (1 - mixWet) + wet * mixWet;
      outL[i] = y;
      if (outR) outR[i] = y;
    }

    return true;
  }
}

registerProcessor('r13-vocoder-processor', VocoderProcessor);
