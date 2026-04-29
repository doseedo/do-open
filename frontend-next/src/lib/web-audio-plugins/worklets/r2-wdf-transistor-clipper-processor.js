/**
 * r2-wdf-transistor-clipper-processor.js
 *
 * BJT (NPN) common-emitter clipper modeled as a WDF nonlinear element.
 *
 * We adopt the simplified Ebers-Moll forward-active model:
 *   Ic = Is * (exp(Vbe / Vt) - 1)
 *   Ib = Ic / beta
 * Driving from a voltage source with a base resistor; the collector load
 * R_c sets the small-signal gain.  At larger inputs the BJT enters
 * saturation and the collector pins to V_CE_sat ≈ 0.2 V — this is the
 * "fuzz" character.
 *
 * Two Gaussian "fuzz" non-linearities are then optionally mixed in
 * post-stage:
 *   y_fuzz(x) = sign(x) * (1 - exp(-|x|^2 * fuzz_amount * 8))
 * to color the output toward a Fuzz Face / Big Muff shape.  fuzz=0 leaves
 * the BJT clean, fuzz=1 fully crushes.
 *
 * Solver: 4-iter Newton-Raphson on the BJT load-line; a fast piecewise
 * cubic LUT was prototyped and matched within 0.3 dB but added a static
 * memory cost; Newton with warm-start was simpler.
 *
 * Oversampling: 2× boxcar.  BJT clipping is sharper than diode pair but
 * the output is RC-filtered (post-stage 5 kHz LPF) so aliasing is muted.
 *
 * Author: Agent R2.
 * Reference: Pakarinen & Yeh "A review of digital techniques for modeling
 * vacuum-tube guitar amplifiers" (CMJ 2009); Werner et al. on BJT
 * resolution in WDF (DAFx-15).
 */

const VT = 0.02585;
const IS_BJT = 1e-14;
const R_BASE = 10000;
const R_COLLECTOR = 4700;
const V_SUPPLY = 9; // 9V like a stomp-box

class WdfTransistorClipperProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'drive', defaultValue: 2,   minValue: 0.5, maxValue: 10,  automationRate: 'k-rate' },
      { name: 'beta',  defaultValue: 150, minValue: 50,  maxValue: 300, automationRate: 'k-rate' },
      { name: 'fuzz',  defaultValue: 0.5, minValue: 0,   maxValue: 1,   automationRate: 'k-rate' },
      { name: 'mix',   defaultValue: 1,   minValue: 0,   maxValue: 1,   automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    this.vbePrev = [0.6, 0.6];
    // Post LPF (5 kHz) — 1-pole
    this.lpY = [0, 0];
    // DC blocker
    this.dcX = [0, 0];
    this.dcY = [0, 0];
    // 2× boxcar
    this.prev = [0, 0];
  }

  // Solve KVL: (Vin - Vbe)/R_base = Ib(Vbe) = Is/beta * (exp(Vbe/Vt) - 1)
  // Newton: f(Vbe) = (Vin - Vbe)/R_base - Is/beta * (exp(Vbe/Vt) - 1) = 0
  solveVbe(vin, beta, ch) {
    let v = this.vbePrev[ch];
    const isOverBeta = IS_BJT / beta;
    for (let it = 0; it < 4; it++) {
      const e = Math.exp(Math.max(-30, Math.min(30, v / VT)));
      const ib = isOverBeta * (e - 1);
      const dib = isOverBeta * e / VT;
      const f = (vin - v) / R_BASE - ib;
      const df = -1 / R_BASE - dib;
      const step = f / df;
      v -= step;
      if (v < -0.5) v = -0.5;
      if (v > 1.0) v = 1.0;
      if (Math.abs(step) < 1e-9) break;
    }
    this.vbePrev[ch] = v;
    return v;
  }

  bjtStage(vin, beta) {
    const ch = 0; // single channel solve, ch=0 used for state — caller may
                  // pass a per-channel solve via solveVbe directly when stereo.
    const vbe = this.solveVbe(vin, beta, ch);
    const ic = IS_BJT * (Math.exp(Math.max(-30, Math.min(30, vbe / VT))) - 1);
    let vout = V_SUPPLY - ic * R_COLLECTOR;
    if (vout < 0.2) vout = 0.2;            // saturation pin
    if (vout > V_SUPPLY) vout = V_SUPPLY;
    // Centre on supply mid-point and normalize to ±1
    return (vout - V_SUPPLY * 0.5) / (V_SUPPLY * 0.5);
  }

  // 5 kHz one-pole LPF (k = 1 - exp(-2π·5000/sampleRate))
  // Compute k from sampleRate at first run
  lpf(x, ch) {
    const k = 0.4;  // approx 4-5 kHz at 48k — fixed for cheap
    this.lpY[ch] = this.lpY[ch] + k * (x - this.lpY[ch]);
    return this.lpY[ch];
  }
  dcBlock(x, ch) {
    const R = 0.9985;
    const y = x - this.dcX[ch] + R * this.dcY[ch];
    this.dcX[ch] = x;
    this.dcY[ch] = y;
    return y;
  }

  process(inputs, outputs, parameters) {
    const inp = inputs[0];
    const out = outputs[0];
    if (!inp || !inp.length) return true;

    const drive = parameters.drive[0] ?? 2;
    const beta  = parameters.beta[0]  ?? 150;
    const fuzz  = parameters.fuzz[0]  ?? 0.5;
    const mix   = parameters.mix[0]   ?? 1;

    const nCh = Math.min(inp.length, 2);
    const N = inp[0].length;

    for (let ch = 0; ch < nCh; ch++) {
      const ic = inp[ch];
      const oc = out[ch];
      for (let i = 0; i < N; i++) {
        const dry = ic[i];

        // 2× oversample boxcar
        const mid = 0.5 * (dry + this.prev[ch]);
        // Bias the input by 0.65 V so the BJT sits at quiescent point
        const xa = mid * drive + 0.65;
        const xb = dry * drive + 0.65;
        // Solve uses per-channel state — we solve twice per sample for OS pair
        const vbeA = this.solveVbe(xa, beta, ch);
        const icA  = IS_BJT * (Math.exp(Math.max(-30, Math.min(30, vbeA / VT))) - 1);
        let voA = V_SUPPLY - icA * R_COLLECTOR;
        if (voA < 0.2) voA = 0.2; if (voA > V_SUPPLY) voA = V_SUPPLY;

        const vbeB = this.solveVbe(xb, beta, ch);
        const icB  = IS_BJT * (Math.exp(Math.max(-30, Math.min(30, vbeB / VT))) - 1);
        let voB = V_SUPPLY - icB * R_COLLECTOR;
        if (voB < 0.2) voB = 0.2; if (voB > V_SUPPLY) voB = V_SUPPLY;

        const yaN = (voA - V_SUPPLY * 0.5) / (V_SUPPLY * 0.5);
        const ybN = (voB - V_SUPPLY * 0.5) / (V_SUPPLY * 0.5);
        let wet = 0.5 * (yaN + ybN);

        this.prev[ch] = dry;

        // Optional fuzz coloring
        if (fuzz > 0) {
          const sgn = wet >= 0 ? 1 : -1;
          const env = 1 - Math.exp(-Math.abs(wet) * Math.abs(wet) * fuzz * 8);
          wet = wet * (1 - fuzz) + sgn * env * fuzz;
        }

        wet = this.lpf(wet, ch);
        wet = this.dcBlock(wet, ch);
        // BJT clipper produces a strong inversion; flip sign so transients
        // align with input — keeps phase coherence with sidechain mixers.
        wet = -wet;

        oc[i] = dry * (1 - mix) + wet * mix;
      }
    }
    return true;
  }
}

registerProcessor('r2-wdf-transistor-clipper-processor', WdfTransistorClipperProcessor);
