/**
 * r13-modulation-delay-processor — Logic Pro Modulation Delay parity.
 *
 * Tape-style chorus/flanger combo: a longer-range modulated delay line per
 * channel with feedback through tape saturation and band-limit (HPF + LPF).
 *
 * Topology (per channel):
 *
 *   x ──┬──────────────────────────── dry ───────────┐
 *       │                                            │
 *       └─→ (+) ──→ delayLine(read = base+lfo*depth) ─┬──→ wet ─┴─→ y
 *           ▲                                         │
 *           │                                         │
 *           └─── HPF → LPF → tape_sat → fb_gain ←─────┘
 *
 * The feedback path runs through:
 *   1. high_cut LPF  (1-pole)
 *   2. low_cut  HPF  (1-pole)
 *   3. tape_saturation (asymmetric soft-clip, dry-wet by `tape_saturation`)
 *   4. feedback gain (signed: negative = polarity-inverted feedback)
 *
 * The two channels run independent delay lines + LFOs whose phase differs by
 * `stereo_phase` degrees so the modulation produces stereo width.
 *
 * Param ranges match the builder schema exactly. All k-rate to keep CPU low —
 * mod-delay is rarely automated at audio rate, and the inner per-sample loop
 * already does enough work.
 *
 * @author Doseedo R13
 */

const SHAPE_SINE     = 0;
const SHAPE_TRIANGLE = 1;
const SHAPE_RANDOM   = 2;
const SHAPE_SQUARE   = 3;

const MAX_DELAY_MS   = 100;   // headroom over the 80 ms top of range
const MAX_DEPTH_MS   = 40;    // depth=100% sweeps ±40 ms relative to base
const TWO_PI         = Math.PI * 2;

class LFO {
  constructor(sampleRate) {
    this.sr = sampleRate;
    this.phase = 0;
    this.freq = 1.0;
    this.shape = SHAPE_SINE;
    // S&H state for `random` shape — interpolate between two random targets.
    this._rndA = (Math.random() * 2 - 1);
    this._rndB = (Math.random() * 2 - 1);
    this._rndPhase = 0;
  }

  setPhaseDeg(deg) {
    this.phase = ((deg % 360) + 360) % 360 / 360;
  }

  process() {
    let out = 0;
    switch (this.shape) {
      case SHAPE_SINE:
        out = Math.sin(this.phase * TWO_PI);
        break;
      case SHAPE_TRIANGLE:
        out = this.phase < 0.5 ? (4 * this.phase - 1) : (3 - 4 * this.phase);
        break;
      case SHAPE_SQUARE:
        out = this.phase < 0.5 ? 1 : -1;
        break;
      case SHAPE_RANDOM: {
        // Smoothed sample-and-hold: interpolate from prevTarget to nextTarget
        // across one cycle; pick a new nextTarget on wrap.
        const t = this.phase;
        out = this._rndA * (1 - t) + this._rndB * t;
        break;
      }
      default:
        out = 0;
    }
    this.phase += this.freq / this.sr;
    if (this.phase >= 1) {
      this.phase -= 1;
      // Advance S&H targets at each cycle wrap.
      this._rndA = this._rndB;
      this._rndB = (Math.random() * 2 - 1);
    }
    return out;
  }
}

class DelayLine {
  constructor(maxSamples) {
    this.size = Math.max(2, maxSamples | 0);
    this.buf = new Float32Array(this.size);
    this.w = 0;
  }

  write(x) {
    this.buf[this.w] = x;
    this.w = (this.w + 1) % this.size;
  }

  // Linear-interpolated read. `delaySamples` is how many samples ago to read.
  read(delaySamples) {
    const ds = Math.max(1, Math.min(this.size - 2, delaySamples));
    const rIdx = this.w - ds;
    const i = ((rIdx % this.size) + this.size) % this.size;
    const i0 = i | 0;
    const frac = i - i0;
    const i1 = (i0 + 1) % this.size;
    return this.buf[i0] * (1 - frac) + this.buf[i1] * frac;
  }
}

// Asymmetric soft saturator — odd-dominant with a small DC-corrected bias term
// to add 2nd harmonic. dryWet ∈ 0..1: 0 = bypass, 1 = fully saturated.
function tapeSaturate(x, dryWet) {
  if (dryWet <= 0) return x;
  // Drive scales with saturation amount so 0..1 sweeps a useful range.
  const drive = 1 + 3 * dryWet;
  const xb = x * drive + 0.18 * dryWet;
  const ax = Math.abs(xb);
  const y = xb / (1 + ax + 0.28 * xb * xb);
  // Remove DC introduced by the bias term so we don't fight the HPF.
  const off = (0.18 * dryWet);
  const aoff = Math.abs(off);
  const yOff = off / (1 + aoff + 0.28 * off * off);
  const sat = (y - yOff) / drive;
  return x * (1 - dryWet) + sat * dryWet;
}

class ModulationDelayProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'delay_ms',        defaultValue: 8.0,  minValue: 0.1,  maxValue: 80,     automationRate: 'k-rate' },
      { name: 'rate_hz',         defaultValue: 0.5,  minValue: 0.05, maxValue: 10,     automationRate: 'k-rate' },
      { name: 'depth',           defaultValue: 30,   minValue: 0,    maxValue: 100,    automationRate: 'k-rate' },
      { name: 'feedback',        defaultValue: 0,    minValue: -100, maxValue: 100,    automationRate: 'k-rate' },
      { name: 'tape_saturation', defaultValue: 0.0,  minValue: 0,    maxValue: 1,      automationRate: 'k-rate' },
      { name: 'lfo_shape',       defaultValue: 0,    minValue: 0,    maxValue: 3,      automationRate: 'k-rate' },
      { name: 'stereo_phase',    defaultValue: 90,   minValue: 0,    maxValue: 360,    automationRate: 'k-rate' },
      { name: 'low_cut',         defaultValue: 50,   minValue: 20,   maxValue: 2000,   automationRate: 'k-rate' },
      { name: 'high_cut',        defaultValue: 12000,minValue: 1000, maxValue: 20000,  automationRate: 'k-rate' },
      { name: 'mix',             defaultValue: 0.5,  minValue: 0,    maxValue: 1,      automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    const sr = sampleRate; // global in worklet scope
    const maxSamples = Math.ceil(((MAX_DELAY_MS + MAX_DEPTH_MS) / 1000) * sr) + 4;

    this._delayL = new DelayLine(maxSamples);
    this._delayR = new DelayLine(maxSamples);
    this._lfoL   = new LFO(sr);
    this._lfoR   = new LFO(sr);

    // Feedback-path filter state — one-pole LPF/HPF per channel.
    this._lpStateL = 0;
    this._lpStateR = 0;
    this._hpPrevInL = 0;
    this._hpPrevInR = 0;
    this._hpStateL = 0;
    this._hpStateR = 0;

    // Cached coefficients
    this._lpAlpha = this._onePoleLP(12000);
    this._hpAlpha = this._onePoleHP(50);

    // Cached state-tracking
    this._lastShape = -1;
    this._lastStereoPhase = -1;
  }

  _onePoleLP(fc) {
    // y = a*x + (1-a)*y;  a = dt/(rc+dt)
    const dt = 1 / sampleRate;
    const rc = 1 / (TWO_PI * Math.max(20, fc));
    return dt / (rc + dt);
  }

  _onePoleHP(fc) {
    // y = a*(y_prev + x - x_prev);  a = rc/(rc+dt)
    const dt = 1 / sampleRate;
    const rc = 1 / (TWO_PI * Math.max(10, fc));
    return rc / (rc + dt);
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!output || !output.length) return true;

    // Pull k-rate params (length 1)
    const delayMs   = parameters.delay_ms[0];
    const rateHz    = parameters.rate_hz[0];
    const depthPct  = parameters.depth[0];
    const fbPct     = parameters.feedback[0];
    const satAmount = parameters.tape_saturation[0];
    const shape     = Math.round(parameters.lfo_shape[0]) | 0;
    const stereoDeg = parameters.stereo_phase[0];
    const lowCutHz  = parameters.low_cut[0];
    const highCutHz = parameters.high_cut[0];
    const mix       = parameters.mix[0];

    // Update LFO config
    this._lfoL.freq = rateHz;
    this._lfoR.freq = rateHz;
    if (shape !== this._lastShape) {
      this._lfoL.shape = shape;
      this._lfoR.shape = shape;
      this._lastShape = shape;
    }
    if (stereoDeg !== this._lastStereoPhase) {
      // Set the right channel's phase OFFSET from left. We don't reset L —
      // we anchor R relative to L's running phase.
      const off = ((stereoDeg % 360) + 360) % 360 / 360;
      this._lfoR.phase = (this._lfoL.phase + off) % 1;
      this._lastStereoPhase = stereoDeg;
    }

    // Update filter coeffs
    this._lpAlpha = this._onePoleLP(highCutHz);
    this._hpAlpha = this._onePoleHP(lowCutHz);

    // Convert ranges
    const baseDelaySamples  = (delayMs / 1000) * sampleRate;
    const depthSamples      = (depthPct / 100) * (MAX_DEPTH_MS / 1000) * sampleRate;
    const fbGain            = (fbPct / 100) * 0.95; // hard cap to keep stable
    const wetGain           = Math.sin(mix * Math.PI / 2);
    const dryGain           = Math.cos(mix * Math.PI / 2);

    const blockSize  = output[0].length;
    const inL = (input && input[0]) ? input[0] : null;
    const inR = (input && input[1]) ? input[1] : (input && input[0]) ? input[0] : null;
    const outL = output[0];
    const outR = output[1] || null;

    for (let i = 0; i < blockSize; i++) {
      const xL = inL ? inL[i] : 0;
      const xR = inR ? inR[i] : xL;

      // ── LFO → delay time (samples) ─────────────────────────────────────
      const lfoL = this._lfoL.process();
      const lfoR = this._lfoR.process();
      const dLs = baseDelaySamples + lfoL * depthSamples;
      const dRs = baseDelaySamples + lfoR * depthSamples;

      // ── Read delay outputs (these become the wet signal AND feedback) ──
      const yLraw = this._delayL.read(dLs);
      const yRraw = this._delayR.read(dRs);

      // ── Feedback path: HPF → LPF → tape_sat → gain ─────────────────────
      // Order chosen: HPF first to remove DC build-up before saturation,
      // then LPF (dark tape repeat character), then sat which adds nonlinear
      // harmonics that the next iteration's HPF cleans up.
      const hpA = this._hpAlpha;
      const hpL = hpA * (this._hpStateL + yLraw - this._hpPrevInL);
      const hpR = hpA * (this._hpStateR + yRraw - this._hpPrevInR);
      this._hpPrevInL = yLraw; this._hpStateL = hpL;
      this._hpPrevInR = yRraw; this._hpStateR = hpR;

      const lpA = this._lpAlpha;
      this._lpStateL = lpA * hpL + (1 - lpA) * this._lpStateL;
      this._lpStateR = lpA * hpR + (1 - lpA) * this._lpStateR;
      const lpL = this._lpStateL;
      const lpR = this._lpStateR;

      const satL = tapeSaturate(lpL, satAmount);
      const satR = tapeSaturate(lpR, satAmount);

      // Write back into the delay line: input + filtered/saturated feedback
      this._delayL.write(xL + satL * fbGain);
      this._delayR.write(xR + satR * fbGain);

      // ── Output: dry + wet ──────────────────────────────────────────────
      // Wet is the raw delay output (pre-feedback shaping) so the user hears
      // the unfiltered modulated signal; the filter only shapes regeneration.
      outL[i] = xL * dryGain + yLraw * wetGain;
      if (outR) outR[i] = xR * dryGain + yRraw * wetGain;
    }

    return true;
  }
}

registerProcessor('r13-modulation-delay-processor', ModulationDelayProcessor);
