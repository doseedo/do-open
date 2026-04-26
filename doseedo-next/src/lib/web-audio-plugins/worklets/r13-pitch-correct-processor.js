/**
 * r13-pitch-correct-processor — Logic Pro Pitch Correction parity (Auto-Tune-class)
 *
 * Single-worklet pipeline:
 *   1. YIN pitch detection on a sliding analysis window (default 2048 samples)
 *   2. Scale-snap quantizer: given key (0-11) + scale mask (uint12), map the
 *      detected f0 onto the nearest in-scale tone. `correction_amount`
 *      blends raw and quantized targets; `response_ms` low-passes the
 *      target shift to keep transitions natural (T-Pain at ~0, vocal-fix
 *      at ~50-150 ms).
 *   3. PSOLA pitch shift driven by the smoothed semitone delta. Epoch
 *      length is locked to the detected period (fallback ~5 ms when
 *      unvoiced), so the time-domain windowing preserves transient timing
 *      better than a phase-vocoder for typical vocal material.
 *
 *   formant_preserve is exposed in the worklet param surface but is a
 *   TODO: the current PSOLA does not separate the spectral envelope from
 *   the excitation. See INTEGRATION_R13_PITCH_CORRECTION.md § "Formant
 *   preservation status".
 *
 * Message protocol (port):
 *   { type: 'key',                value: 0..11 }
 *   { type: 'scale_mask',         value: uint12 (bitmask, bit0=root) }
 *   { type: 'response_ms',        value: 0..500 }
 *   { type: 'correction_amount',  value: 0..1 }
 *   { type: 'formant_preserve',   value: 0|1 }
 *   { type: 'mix',                value: 0..1 }
 *   { type: 'reset' }
 *
 * Equivalent AudioParam controls are exposed for k-rate automation.
 *
 * Latency: ~ analysisWindow / 2 + 1 PSOLA epoch (≈ 25–30 ms at 48 kHz).
 *
 * @author Doseedo R13
 */

// ──────────────────────────────────────────────────────────────────────
// Scale masks (bit i = semitone i is in scale; bit 0 = root)
// ──────────────────────────────────────────────────────────────────────
const SCALE_MASKS = {
  major:     0b101010110101, // C D E F G A B  → 0,2,4,5,7,9,11
  minor:     0b010110101101, // C D Eb F G Ab Bb (natural minor) → 0,2,3,5,7,8,10
  chromatic: 0b111111111111,
};

// Map { name → mask } accessible to consumers in workletGlobalScope.
// Exporting from a worklet isn't possible, but the masks above mirror the
// builder side r13_pitch_correct.js exactly — keep the two in sync.

class R13PitchCorrectProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'key',                defaultValue: 0,    minValue: 0, maxValue: 11,  automationRate: 'k-rate' },
      { name: 'scale_mask',         defaultValue: SCALE_MASKS.chromatic, minValue: 0, maxValue: 4095, automationRate: 'k-rate' },
      { name: 'response_ms',        defaultValue: 50,   minValue: 0, maxValue: 500, automationRate: 'k-rate' },
      { name: 'correction_amount',  defaultValue: 1,    minValue: 0, maxValue: 1,   automationRate: 'k-rate' },
      { name: 'formant_preserve',   defaultValue: 0,    minValue: 0, maxValue: 1,   automationRate: 'k-rate' },
      { name: 'mix',                defaultValue: 1,    minValue: 0, maxValue: 1,   automationRate: 'k-rate' },
    ];
  }

  constructor(options) {
    super();

    const opts = (options && options.processorOptions) || {};
    this.analysisWindow = opts.analysisWindow || 2048;   // ≈ 43 ms @ 48 kHz
    this.yinThreshold   = opts.yinThreshold   || 0.15;   // canonical YIN
    this.minF0Hz        = opts.minF0Hz        || 70;
    this.maxF0Hz        = opts.maxF0Hz        || 1100;

    // ── Analysis state ────────────────────────────────────────────────
    this.inputRing     = new Float32Array(this.analysisWindow);
    this.inputWritePos = 0;
    this.inputFilled   = 0;

    // YIN difference + cumulative-mean buffer
    const halfW = this.analysisWindow >> 1;
    this.yinDiff = new Float32Array(halfW);
    this.yinCMND = new Float32Array(halfW);

    this.detectedF0Hz   = 0;        // 0 → unvoiced
    this.lastVoicedF0Hz = 220;
    this.detectIntervalSamples = 256; // run YIN every 256 samples
    this.samplesUntilDetect    = 0;

    // ── PSOLA state ───────────────────────────────────────────────────
    // Larger ring so we can pull epochs from "around" the write head.
    this.psolaRingSize = Math.max(8192, this.analysisWindow * 4);
    this.psolaRing     = new Float32Array(this.psolaRingSize);
    this.psolaWritePos = 0;
    this.psolaSamplesIn = 0;

    // Read pointer (fractional). Output epoch placement is driven by
    // the synthesis period (target).
    this.psolaReadPos = 0;
    this.synthSamplesUntilNextEpoch = 0;

    // Smoothed log-ratio (semitones) target between detected and snap.
    // Smoothing per-sample one-pole: y += (target - y) / tau_samples
    this.smoothedSemitoneDelta = 0;

    // ── Port overrides (used by host to bypass AudioParam where needed) ─
    this._override = {};
    this.port.onmessage = (e) => {
      const m = e.data;
      if (!m || typeof m !== 'object') return;
      switch (m.type) {
        case 'key':
        case 'scale_mask':
        case 'response_ms':
        case 'correction_amount':
        case 'formant_preserve':
        case 'mix':
          this._override[m.type] = +m.value;
          break;
        case 'reset':
          this.inputRing.fill(0);
          this.psolaRing.fill(0);
          this.inputWritePos = 0;
          this.inputFilled   = 0;
          this.psolaWritePos = 0;
          this.psolaSamplesIn = 0;
          this.psolaReadPos  = 0;
          this.synthSamplesUntilNextEpoch = 0;
          this.smoothedSemitoneDelta = 0;
          break;
      }
    };
  }

  // ──────────────────────────────────────────────────────────────────
  // YIN pitch detection (Cheveigné & Kawahara 2002)
  // Operates on this.inputRing (analysisWindow length, current write pos).
  // Returns f0 in Hz, or 0 if unvoiced.
  // ──────────────────────────────────────────────────────────────────
  _runYIN() {
    const W = this.analysisWindow;
    const halfW = W >> 1;
    const ring = this.inputRing;
    const wp   = this.inputWritePos;

    // Step 1: difference function d(τ) = Σ (x[i] - x[i+τ])^2 over i in [0, W/2)
    const d = this.yinDiff;
    for (let tau = 0; tau < halfW; tau++) {
      let sum = 0;
      for (let i = 0; i < halfW; i++) {
        const a = ring[(wp + i)      % W];
        const b = ring[(wp + i + tau) % W];
        const diff = a - b;
        sum += diff * diff;
      }
      d[tau] = sum;
    }

    // Step 2: cumulative-mean normalised difference d'(τ) = d(τ) / ((1/τ) Σ d(j))
    const cmnd = this.yinCMND;
    cmnd[0] = 1;
    let running = 0;
    for (let tau = 1; tau < halfW; tau++) {
      running += d[tau];
      cmnd[tau] = d[tau] * tau / Math.max(running, 1e-12);
    }

    // Step 3: absolute threshold — first dip below threshold.
    const thr = this.yinThreshold;
    const sr  = sampleRate;
    const minTau = Math.floor(sr / this.maxF0Hz);
    const maxTau = Math.min(halfW - 1, Math.floor(sr / this.minF0Hz));

    let tauEst = -1;
    for (let tau = Math.max(2, minTau); tau <= maxTau; tau++) {
      if (cmnd[tau] < thr) {
        // Walk to the local minimum
        while (tau + 1 <= maxTau && cmnd[tau + 1] < cmnd[tau]) tau++;
        tauEst = tau;
        break;
      }
    }
    if (tauEst < 0) return 0;

    // Step 4: parabolic interpolation around tauEst for sub-sample accuracy
    let betterTau = tauEst;
    if (tauEst > 0 && tauEst < halfW - 1) {
      const s0 = cmnd[tauEst - 1];
      const s1 = cmnd[tauEst];
      const s2 = cmnd[tauEst + 1];
      const denom = (s0 + s2 - 2 * s1);
      if (Math.abs(denom) > 1e-12) {
        betterTau = tauEst + 0.5 * (s0 - s2) / denom;
      }
    }

    if (betterTau <= 0) return 0;
    return sr / betterTau;
  }

  // ──────────────────────────────────────────────────────────────────
  // Scale-snap quantizer.
  // f0Hz → nearest in-scale frequency. mask is a 12-bit mask, bit i = semitone i
  // relative to the key (0=C ... 11=B). Returns target Hz (0 if unvoiced).
  // ──────────────────────────────────────────────────────────────────
  _snapToScale(f0Hz, key, mask) {
    if (!f0Hz || f0Hz <= 0) return 0;
    if (mask === 0) return f0Hz; // empty mask → no snap
    // semitones above C0 (Hz convention not critical — we only need the
    // fractional residue mod 12 for snap, and the absolute octave for
    // reconstruction).
    const REF = 16.3516; // C0 ≈ 16.35 Hz
    const semitones = 12 * Math.log2(f0Hz / REF);
    const semInt    = Math.floor(semitones);
    const semFrac   = semitones - semInt;

    // pitch class relative to KEY = (semInt - key) mod 12
    const _mod12 = (n) => ((n % 12) + 12) % 12;

    // Find the in-scale pitch-class closest to (semInt - key) + semFrac
    const target = (semInt - key) + semFrac; // continuous
    let bestSem = null;
    let bestDist = 1e9;
    // Check current PC and neighbors ±12 to handle wrap-around at edges.
    for (let octShift = -1; octShift <= 1; octShift++) {
      for (let pc = 0; pc < 12; pc++) {
        if (((mask >> pc) & 1) === 0) continue;
        const candidate = pc + octShift * 12; // relative-to-key semitone, allowing octave shift
        const dist = Math.abs(candidate - target);
        if (dist < bestDist) {
          bestDist = dist;
          bestSem  = candidate;
        }
      }
    }
    if (bestSem == null) return f0Hz;

    const snappedSem = bestSem + key;
    return REF * Math.pow(2, snappedSem / 12);
  }

  // ──────────────────────────────────────────────────────────────────
  // Main process loop.
  // ──────────────────────────────────────────────────────────────────
  process(inputs, outputs, parameters) {
    const inp = inputs[0];
    const out = outputs[0];
    if (!out || out.length < 1) return true;

    const _readK = (name) => {
      if (this._override[name] != null) return this._override[name];
      return parameters[name][0];
    };

    const key       = Math.max(0, Math.min(11, Math.round(_readK('key'))));
    const scaleMask = Math.max(0, Math.min(4095, Math.round(_readK('scale_mask'))));
    const respMs    = Math.max(0, Math.min(500, _readK('response_ms')));
    const corrAmt   = Math.max(0, Math.min(1, _readK('correction_amount')));
    const mix       = Math.max(0, Math.min(1, _readK('mix')));
    // formant_preserve is read but not yet acted on — TODO.
    void _readK('formant_preserve');

    const sr = sampleRate;
    // Smoothing time-constant in samples. response_ms = 0 → instant
    // (alpha = 1); response_ms = 200 → tau_samples ≈ 0.2 * sr → soft.
    const tauSamples = Math.max(1, (respMs * sr) / 1000);
    const alpha      = (respMs <= 0.5) ? 1.0 : (1 - Math.exp(-1 / tauSamples));

    const inL  = (inp && inp[0]) ? inp[0] : null;
    const outL = out[0];
    const outR = out[1] || null;
    const block = outL.length;

    for (let i = 0; i < block; i++) {
      const dry = inL ? inL[i] : 0;

      // 1) Push input into both rings
      this.inputRing[this.inputWritePos] = dry;
      this.inputWritePos = (this.inputWritePos + 1) % this.analysisWindow;
      this.inputFilled = Math.min(this.analysisWindow, this.inputFilled + 1);

      this.psolaRing[this.psolaWritePos] = dry;
      this.psolaWritePos = (this.psolaWritePos + 1) % this.psolaRingSize;
      this.psolaSamplesIn = Math.min(this.psolaRingSize, this.psolaSamplesIn + 1);

      // 2) Periodic YIN (cheap relative to full-block cost since we throttle)
      this.samplesUntilDetect--;
      if (this.samplesUntilDetect <= 0 && this.inputFilled >= this.analysisWindow) {
        const f0 = this._runYIN();
        this.detectedF0Hz = f0;
        if (f0 > 0) this.lastVoicedF0Hz = f0;
        this.samplesUntilDetect = this.detectIntervalSamples;
      }

      // 3) Compute target shift in semitones.
      let targetSemDelta = 0;
      if (this.detectedF0Hz > 0) {
        const snapped = this._snapToScale(this.detectedF0Hz, key, scaleMask);
        if (snapped > 0) {
          // Δ = corrAmt * log2(snapped / detected) * 12
          const fullDelta = 12 * Math.log2(snapped / this.detectedF0Hz);
          targetSemDelta  = corrAmt * fullDelta;
        }
      }
      // 4) Smooth toward the target.
      this.smoothedSemitoneDelta += (targetSemDelta - this.smoothedSemitoneDelta) * alpha;

      // 5) Synthesise PSOLA epoch when due.
      // Synthesis epoch length T_s = T_a / ratio  where ratio = 2^(Δsem/12).
      // We simply drive the read pointer at `1/ratio` while writing at 1.
      // This produces sample-by-sample resampled output through grain
      // crossfades, which matches the standard PSOLA "vary the read rate"
      // construction for small shifts (< ±5 semitones) typical of pitch
      // correction.
      const ratio = Math.pow(2, this.smoothedSemitoneDelta / 12);
      const period = (this.detectedF0Hz > 0) ? (sr / this.detectedF0Hz)
                                             : (sr / this.lastVoicedF0Hz);
      // Two-grain crossfade — read from two positions period/2 apart, fade.
      const wet = this._readPsolaTwoGrain(this.psolaReadPos, period);

      // Advance read pos by 1/ratio so we cover (1 sample of input) per (1/ratio of output).
      // ratio > 1 → shifting up → consume input faster → readPos advances faster
      this.psolaReadPos += 1 / Math.max(1e-6, ratio);
      // Bound psolaReadPos to lag a bit behind the write head (otherwise we
      // fall off the back of the ring).
      const lag = (this.psolaWritePos - this.psolaReadPos + this.psolaRingSize) % this.psolaRingSize;
      // Reset if drift exceeds ring half — an audible glitch but avoids
      // catastrophic divergence on long shifts.
      if (lag < period * 2 || lag > this.psolaRingSize - period * 4) {
        // Re-anchor read pointer ~ a couple periods behind write
        this.psolaReadPos = (this.psolaWritePos - 2 * period + this.psolaRingSize) % this.psolaRingSize;
      }

      const y = dry * (1 - mix) + wet * mix;
      outL[i] = y;
      if (outR) outR[i] = y;
    }

    return true;
  }

  // Two-grain PSOLA read: samples from the ring at `pos` and `pos + period/2`,
  // crossfaded with a Hann shape. period in samples.
  _readPsolaTwoGrain(pos, period) {
    const N = this.psolaRingSize;
    const half = period * 0.5;
    const a = this._readRingFractional(pos, N);
    const b = this._readRingFractional(pos + half, N);
    // Compute Hann weights based on phase within half-period
    const phaseA = (pos / period) - Math.floor(pos / period); // 0..1
    const wA = 0.5 * (1 + Math.cos(2 * Math.PI * phaseA));
    const wB = 1 - wA;
    return a * wA + b * wB;
  }

  _readRingFractional(idx, N) {
    let i = idx;
    while (i < 0) i += N;
    while (i >= N) i -= N;
    const i0 = Math.floor(i);
    const i1 = (i0 + 1) % N;
    const f  = i - i0;
    return this.psolaRing[i0] * (1 - f) + this.psolaRing[i1] * f;
  }
}

registerProcessor('r13-pitch-correct-processor', R13PitchCorrectProcessor);
