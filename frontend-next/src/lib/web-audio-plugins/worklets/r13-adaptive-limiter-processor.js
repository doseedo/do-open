/**
 * r13-adaptive-limiter-processor — Logic Pro Adaptive Limiter parity.
 *
 * Multi-stage transparent mastering limiter with adaptive release. Pipeline:
 *
 *   1. Pre-gain (auto-gain into the limiter, 0..24 dB)
 *   2. Lookahead delay (1..12 ms circular buffer, per channel)
 *   3. Peak detector (instantaneous |x| per sample, ahead of the delayed sig)
 *   4. Optional 4× true-peak oversampler — polyphase FIR upsampler that gates
 *      using the inter-sample peak so the output ceiling holds against ISP.
 *   5. Adaptive release calculator — sliding window of recent peak-over-ceiling
 *      events; high event density ⇒ short release (avoid pumping); sparse
 *      events ⇒ long release (preserves transient dynamics). Map:
 *        density_norm ∈ [0..1]
 *        release_time = release_min + (release_max − release_min) · (1 − density_norm)
 *      `release_adaptation` blends between a fixed release (release_max) and
 *      the fully adaptive value, so the surface degrades gracefully to a
 *      vanilla limiter at adaptation=0.
 *   6. Gain reduction smoother — single-pole follower with attack tied to the
 *      lookahead window (so the limiter is reactive but not zippery) and the
 *      adaptive release as the release time-constant.
 *   7. Apply gain = ceiling / max(peak, ceiling) when peak > ceiling, else 1.
 *      Smoothed across samples.
 *   8. Soft-clip stage — final 3rd-order polynomial (`y = x − x³/3` clamped at
 *      ±1) blended via `soft_clip_amount` so any residual spike past the
 *      ceiling is rounded off rather than chopped.
 *
 * `link_lr=1` couples both channels through a single peak follower + gain
 * envelope (mastering-correct: any channel triggering the limiter pulls both
 * down together so stereo image is preserved).
 *
 * Message protocol: parameters are AudioParams (k-rate). Reset on
 *   { type: 'reset' }
 *
 * @author Doseedo R13 — Adaptive Limiter
 */

const SR = (typeof sampleRate === 'number') ? sampleRate : 48000;

// 4× polyphase FIR upsampler — 8-tap windowed sinc per phase (32 taps total).
// Generated offline; symmetric about the centre tap. Adequate ISP detection
// without the cost of a full FFT-based oversampler. Coefficients sum-to-1 per
// phase (DC unity gain).
const FIR_4X = [
  // phase 0 (identity-ish; centre on input sample)
  [-0.00154, 0.01015, -0.04036, 0.97185, 0.06723, -0.01620, 0.00351, -0.00064],
  // phase 1 (1/4 sample ahead)
  [-0.00355, 0.02323, -0.08828, 0.85123, 0.24389, -0.04050, 0.00837, -0.00149],
  // phase 2 (1/2 sample ahead)
  [-0.00504, 0.03226, -0.12318, 0.59596, 0.59596, -0.12318, 0.03226, -0.00504],
  // phase 3 (3/4 sample ahead)
  [-0.00149, 0.00837, -0.04050, 0.24389, 0.85123, -0.08828, 0.02323, -0.00355],
];
const FIR_LEN = 8;

// Ring-buffer helpers --------------------------------------------------------
function makeRing(size) {
  return { buf: new Float32Array(size), w: 0, size };
}
function ringPush(r, v) {
  r.buf[r.w] = v;
  r.w = (r.w + 1) % r.size;
}
function ringRead(r, delaySamples) {
  // Read sample `delaySamples` ago. Linear-interp not needed: lookahead is in
  // integer samples (we round to nearest sample on param change).
  const i = ((r.w - delaySamples) % r.size + r.size) % r.size;
  return r.buf[i];
}

class R13AdaptiveLimiterProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'gain',                defaultValue: 0,    minValue: 0,    maxValue: 24,   automationRate: 'k-rate' },
      { name: 'out_ceiling',         defaultValue: -0.3, minValue: -30,  maxValue: 0,    automationRate: 'k-rate' },
      { name: 'lookahead_ms',        defaultValue: 5,    minValue: 1,    maxValue: 12,   automationRate: 'k-rate' },
      { name: 'release_min_ms',      defaultValue: 5,    minValue: 1,    maxValue: 50,   automationRate: 'k-rate' },
      { name: 'release_max_ms',      defaultValue: 500,  minValue: 100,  maxValue: 2000, automationRate: 'k-rate' },
      { name: 'release_adaptation',  defaultValue: 0.7,  minValue: 0,    maxValue: 1,    automationRate: 'k-rate' },
      { name: 'true_peak',           defaultValue: 0,    minValue: 0,    maxValue: 1,    automationRate: 'k-rate' },
      { name: 'soft_clip_amount',    defaultValue: 0.3,  minValue: 0,    maxValue: 1,    automationRate: 'k-rate' },
      { name: 'link_lr',             defaultValue: 1,    minValue: 0,    maxValue: 1,    automationRate: 'k-rate' },
    ];
  }

  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};

    // Lookahead buffers (max 12 ms @ 192k = 2304 samples; round up).
    const maxLookSamples = Math.ceil(0.020 * SR); // 20 ms headroom
    this.delayL = makeRing(maxLookSamples);
    this.delayR = makeRing(maxLookSamples);

    // Per-channel envelope state (gain-reduction follower, in linear gain).
    // We track 1 = no reduction down to small positive numbers (more reduction).
    this.envL = 1.0;
    this.envR = 1.0;

    // Single linked envelope when link_lr=1.
    this.envLinked = 1.0;

    // 4× upsampler FIR history per channel (length FIR_LEN).
    this.firHistL = new Float32Array(FIR_LEN);
    this.firHistR = new Float32Array(FIR_LEN);
    this.firPosL  = 0;
    this.firPosR  = 0;

    // Adaptive-release sliding window. We count threshold-crossing events in
    // a window of ~150 ms. `density_norm = events / max_events_in_window`.
    // Max events ~ 1 per 0.5 ms = 300 in 150 ms; we cap density at 1.
    this.windowSec = opts.adaptiveWindowSec || 0.150;
    this.windowSamples = Math.max(1, Math.floor(this.windowSec * SR));
    this.eventRing = new Uint8Array(this.windowSamples); // 1 if sample crossed ceiling
    this.eventW = 0;
    this.eventCount = 0;
    // Density saturation: cap at "max events per second". Rough heuristic —
    // 200 events/sec ≈ extremely dense (drum loops, brickwall masters).
    this.maxEventsPerSec = opts.maxEventsPerSec || 200;
    this.maxEventsInWindow = Math.max(1, Math.floor(this.maxEventsPerSec * this.windowSec));

    this.port.onmessage = (e) => {
      const msg = e.data || {};
      if (msg.type === 'reset') {
        this.envL = this.envR = this.envLinked = 1.0;
        this.delayL.buf.fill(0); this.delayR.buf.fill(0);
        this.delayL.w = 0; this.delayR.w = 0;
        this.firHistL.fill(0); this.firHistR.fill(0);
        this.firPosL = 0; this.firPosR = 0;
        this.eventRing.fill(0); this.eventW = 0; this.eventCount = 0;
      }
    };
  }

  // True-peak inter-sample peak via 4× polyphase upsample. Returns the
  // largest |sample| across the 4 sub-phases for the most recent input.
  _truePeak(hist, pos, x) {
    // Push x into history.
    hist[pos] = x;
    const newPos = (pos + 1) % FIR_LEN;
    let maxAbs = Math.abs(x);
    for (let p = 0; p < 4; p++) {
      const taps = FIR_4X[p];
      let acc = 0;
      // Convolve with the most recent FIR_LEN samples in history.
      for (let k = 0; k < FIR_LEN; k++) {
        const idx = ((newPos - 1 - k) % FIR_LEN + FIR_LEN) % FIR_LEN;
        acc += taps[k] * hist[idx];
      }
      const a = Math.abs(acc);
      if (a > maxAbs) maxAbs = a;
    }
    return { peak: maxAbs, newPos };
  }

  // Compute one-pole coefficient for a given time-constant in seconds.
  // y[n] = a*y[n-1] + (1-a)*x[n]; a = exp(-1/(tau*sr)).
  _tau(timeSec) {
    if (!(timeSec > 0)) return 0;
    return Math.exp(-1 / (Math.max(1e-6, timeSec) * SR));
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || !input.length) return true;

    const blockSize = input[0].length;
    const numCh = Math.min(input.length, output.length);

    // k-rate params: read once per block.
    const gainDb        = parameters.gain[0];
    const ceilDb        = parameters.out_ceiling[0];
    const lookMs        = parameters.lookahead_ms[0];
    const relMinMs      = parameters.release_min_ms[0];
    const relMaxMs      = parameters.release_max_ms[0];
    const relAdapt      = parameters.release_adaptation[0];
    const truePeakOn    = parameters.true_peak[0] >= 0.5;
    const softClipAmt   = parameters.soft_clip_amount[0];
    const linkLR        = parameters.link_lr[0] >= 0.5;

    const preGain = Math.pow(10, gainDb / 20);
    const ceilLin = Math.pow(10, ceilDb / 20);

    // Lookahead in samples (clamped to ring size − 1).
    const lookSamples = Math.min(this.delayL.size - 1, Math.max(1, Math.floor((lookMs / 1000) * SR)));

    // Attack time constant locked to lookahead so envelope settles by the
    // time the delayed signal arrives at the gain stage. 1/3 of the
    // lookahead window is a conventional choice.
    const attackSec = (lookMs / 1000) / 3;
    const attackA = this._tau(attackSec);

    const inL = input[0];
    const inR = numCh > 1 ? input[1] : input[0];
    const outL = output[0];
    const outR = numCh > 1 ? output[1] : output[0];

    for (let i = 0; i < blockSize; i++) {
      // Apply pre-gain.
      const xL = inL[i] * preGain;
      const xR = inR[i] * preGain;

      // ── Detector: peak that drives gain reduction ──────────────────────
      let peakL = Math.abs(xL);
      let peakR = Math.abs(xR);
      if (truePeakOn) {
        const tpL = this._truePeak(this.firHistL, this.firPosL, xL);
        peakL = tpL.peak;
        this.firPosL = tpL.newPos;
        const tpR = this._truePeak(this.firHistR, this.firPosR, xR);
        peakR = tpR.peak;
        this.firPosR = tpR.newPos;
      } else {
        // Still advance history to keep state coherent if user toggles ON later.
        this.firHistL[this.firPosL] = xL; this.firPosL = (this.firPosL + 1) % FIR_LEN;
        this.firHistR[this.firPosR] = xR; this.firPosR = (this.firPosR + 1) % FIR_LEN;
      }

      // ── Adaptive release: count threshold crossings in sliding window ──
      const peakForEvent = Math.max(peakL, peakR);
      const event = peakForEvent > ceilLin ? 1 : 0;
      const oldEvent = this.eventRing[this.eventW];
      if (event !== oldEvent) {
        this.eventCount += (event - oldEvent);
      }
      this.eventRing[this.eventW] = event;
      this.eventW = (this.eventW + 1) % this.windowSamples;

      const density = Math.min(1, this.eventCount / this.maxEventsInWindow);
      const relAdaptSec = (relMinMs + (relMaxMs - relMinMs) * (1 - density)) / 1000;
      // Blend: at adaptation=0, release stays at release_max; at 1, release
      // is fully adaptive.
      const fixedSec = relMaxMs / 1000;
      const releaseSec = (1 - relAdapt) * fixedSec + relAdapt * relAdaptSec;
      const releaseA = this._tau(releaseSec);

      // ── Push raw samples into lookahead ───────────────────────────────
      ringPush(this.delayL, xL);
      ringPush(this.delayR, xR);

      // ── Gain target per channel (or linked) ───────────────────────────
      const targetL = peakL > ceilLin ? (ceilLin / peakL) : 1.0;
      const targetR = peakR > ceilLin ? (ceilLin / peakR) : 1.0;

      let gL, gR;
      if (linkLR) {
        const target = Math.min(targetL, targetR);
        // Asymmetric envelope: attack (toward smaller gain) is fast, release is slow.
        const a = (target < this.envLinked) ? attackA : releaseA;
        this.envLinked = a * this.envLinked + (1 - a) * target;
        gL = gR = this.envLinked;
      } else {
        const aL = (targetL < this.envL) ? attackA : releaseA;
        this.envL = aL * this.envL + (1 - aL) * targetL;
        const aR = (targetR < this.envR) ? attackA : releaseA;
        this.envR = aR * this.envR + (1 - aR) * targetR;
        gL = this.envL; gR = this.envR;
      }

      // ── Read delayed sample, apply gain, soft-clip, write out ─────────
      const dL = ringRead(this.delayL, lookSamples);
      const dR = ringRead(this.delayR, lookSamples);
      let yL = dL * gL;
      let yR = dR * gR;

      // Final brickwall + 3rd-order soft-clip blend. Clip threshold is the
      // ceiling — anything still above gets bent rather than squared off.
      // Normalize to ±1 within the ceiling, soft-clip, scale back.
      if (softClipAmt > 0) {
        const scL = yL / ceilLin;
        const scR = yR / ceilLin;
        const clampL = Math.max(-1.5, Math.min(1.5, scL));
        const clampR = Math.max(-1.5, Math.min(1.5, scR));
        // y = x - x^3/3 clipped to ±1 at |x|>=√3 (~1.732). Within [-1.5,1.5]
        // it's smooth and monotonic.
        const sL = clampL - (clampL * clampL * clampL) / 3;
        const sR = clampR - (clampR * clampR * clampR) / 3;
        yL = ((1 - softClipAmt) * scL + softClipAmt * sL) * ceilLin;
        yR = ((1 - softClipAmt) * scR + softClipAmt * sR) * ceilLin;
      }

      // Hard clamp at ceiling — final brickwall guarantee.
      if (yL > ceilLin) yL = ceilLin;
      else if (yL < -ceilLin) yL = -ceilLin;
      if (yR > ceilLin) yR = ceilLin;
      else if (yR < -ceilLin) yR = -ceilLin;

      outL[i] = yL;
      if (numCh > 1) outR[i] = yR;
    }

    // Periodically post the current gain reduction (in dB) for UI metering.
    // We report once per block to avoid flooding the message port.
    const grL = 20 * Math.log10(Math.max(1e-6, linkLR ? this.envLinked : this.envL));
    const grR = 20 * Math.log10(Math.max(1e-6, linkLR ? this.envLinked : this.envR));
    this.port.postMessage({ type: 'gainReduction', valueL: grL, valueR: grR });

    return true;
  }
}

registerProcessor('r13-adaptive-limiter-processor', R13AdaptiveLimiterProcessor);
