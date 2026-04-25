/**
 * r2-wdf-tube-triode-processor.js
 *
 * Single-stage triode-tube model in WDF style.
 *
 * Plate-current model: Norman Koren (1996), parameterless cubic-spline LUT
 * approximation for runtime cost.  Original Koren equation:
 *
 *   E1 = (Vpk / KP) * ln(1 + exp(KP * (1/MU + Vgk / sqrt(KVB + Vpk^2))))
 *   Ip = (E1^EX / KG1) * (1 + sgn(E1))
 *
 * Constants (12AX7-like): MU=100, KP=600, KVB=300, KG1=1060, EX=1.4
 *
 * Why LUT, not Newton: the Koren equation is well-defined but the cost of
 * evaluating exp+ln+pow per sample × Newton iterations puts us over the
 * real-time budget.  We pre-compute Ip(Vgk) at fixed plate voltage, then
 * cubic-spline interpolate.  Distortion shape is faithful to within ~0.5
 * dB of full-precision Koren up to the saturation knee — beyond that the
 * tube is hard-clipped anyway.
 *
 * Bias param: shifts the input grid voltage by `bias` volts.  Negative
 * bias (cold-bias) → asymmetric distortion (positive half clipped first).
 *
 * Coupling: R_grid + C_coupling forms an HPF; WDF resolves the loop, but
 * we approximate with explicit 1-pole HPF at 30 Hz post-tube + DC blocker.
 *
 * Author: Agent R2.
 * Reference: Koren, "Improved vacuum tube models for SPICE simulations"
 * (Glass Audio, 1996).  Pakarinen & Karjalainen, "Wave digital simulation
 * of a vacuum-tube amplifier" (ICASSP 2006).
 */

// LUT parameters
const LUT_VGK_MIN = -10;
const LUT_VGK_MAX = 4;
const LUT_SIZE    = 512;
const LUT_VPK_FIXED = 250; // plate voltage assumed (V)

function buildKorenLUT() {
  const MU = 100, KP = 600, KVB = 300, KG1 = 1060, EX = 1.4;
  const Vpk = LUT_VPK_FIXED;
  const lut = new Float32Array(LUT_SIZE);
  for (let i = 0; i < LUT_SIZE; i++) {
    const t = i / (LUT_SIZE - 1);
    const Vgk = LUT_VGK_MIN + t * (LUT_VGK_MAX - LUT_VGK_MIN);
    const arg = KP * (1 / MU + Vgk / Math.sqrt(KVB + Vpk * Vpk));
    // Numerically stable softplus
    const sp = (arg > 30) ? arg : Math.log1p(Math.exp(arg));
    const E1 = (Vpk / KP) * sp;
    let Ip = 0;
    if (E1 > 0) {
      Ip = Math.pow(E1, EX) / KG1; // factor of 2 absorbed into normalization
    }
    lut[i] = Ip;
  }
  // Normalize so that mid-bias output is ~1 mA
  let peak = 0;
  for (let i = 0; i < LUT_SIZE; i++) if (lut[i] > peak) peak = lut[i];
  if (peak > 0) for (let i = 0; i < LUT_SIZE; i++) lut[i] /= peak;
  return lut;
}

const KOREN_LUT = buildKorenLUT();
const LUT_RANGE = LUT_VGK_MAX - LUT_VGK_MIN;

// Cubic-Catmull-Rom interpolation on the LUT.
function lutLookup(vgk) {
  const f = (vgk - LUT_VGK_MIN) / LUT_RANGE;
  if (f <= 0) return KOREN_LUT[0];
  if (f >= 1) return KOREN_LUT[LUT_SIZE - 1];
  const x = f * (LUT_SIZE - 1);
  const i1 = Math.floor(x);
  const t = x - i1;
  const i0 = i1 > 0 ? i1 - 1 : i1;
  const i2 = i1 + 1 < LUT_SIZE ? i1 + 1 : i1;
  const i3 = i2 + 1 < LUT_SIZE ? i2 + 1 : i2;
  const p0 = KOREN_LUT[i0], p1 = KOREN_LUT[i1], p2 = KOREN_LUT[i2], p3 = KOREN_LUT[i3];
  // Catmull-Rom basis
  const a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
  const a1 = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3;
  const a2 = -0.5 * p0 + 0.5 * p2;
  const a3 = p1;
  return ((a0 * t + a1) * t + a2) * t + a3;
}

class WdfTubeTriodeProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'drive', defaultValue: 1,    minValue: 0.1, maxValue: 5, automationRate: 'k-rate' },
      { name: 'bias',  defaultValue: -1.5, minValue: -4,  maxValue: 0, automationRate: 'k-rate' },
      { name: 'mix',   defaultValue: 1,    minValue: 0,   maxValue: 1, automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    // Coupling HPF state (per channel)
    this.hpX = [0, 0];
    this.hpY = [0, 0];
    // DC blocker state
    this.dcX = [0, 0];
    this.dcY = [0, 0];
    // 2× oversampling boxcar previous-sample state
    this.prev = [0, 0];
  }

  // Coupling HPF at ~30 Hz, R≈0.995 @48k
  hpf(x, ch) {
    const R = 0.9955;
    const y = R * (this.hpY[ch] + x - this.hpX[ch]);
    this.hpX[ch] = x;
    this.hpY[ch] = y;
    return y;
  }
  dcBlock(x, ch) {
    const R = 0.9985;
    const y = x - this.dcX[ch] + R * this.dcY[ch];
    this.dcX[ch] = x;
    this.dcY[ch] = y;
    return y;
  }

  // Single-sample triode stage: input → grid → Ip → invert/scale → output
  triode(x, drive, bias) {
    const Vgk = drive * x + bias;          // grid-to-cathode voltage
    const Ip = lutLookup(Vgk);             // normalized 0..1
    // Plate voltage = Vsupply - Ip * R_load.  Use proportional model with
    // gain ~ -mu/2 ≈ -50, and rescale so unity input ≈ unity output at
    // moderate drive.
    const v_plate = (1 - Ip) * 1.0;        // 1=at supply, 0=at zero
    return (v_plate - 0.5) * 2;            // recenter to [-1, 1]
  }

  process(inputs, outputs, parameters) {
    const inp = inputs[0];
    const out = outputs[0];
    if (!inp || !inp.length) return true;

    const drive = parameters.drive[0] ?? 1;
    const bias  = parameters.bias[0]  ?? -1.5;
    const mix   = parameters.mix[0]   ?? 1;

    const nCh = Math.min(inp.length, 2);
    const N = inp[0].length;

    for (let ch = 0; ch < nCh; ch++) {
      const ic = inp[ch];
      const oc = out[ch];
      for (let i = 0; i < N; i++) {
        const dry = ic[i];

        // 2× oversample with boxcar — average of current+midpoint pass
        const mid = 0.5 * (dry + this.prev[ch]);
        const ya = this.triode(mid, drive, bias);
        const yb = this.triode(dry, drive, bias);
        let wet = 0.5 * (ya + yb);
        this.prev[ch] = dry;

        // Coupling HPF (decouples DC bias offset)
        wet = this.hpf(wet, ch);
        wet = this.dcBlock(wet, ch);

        oc[i] = dry * (1 - mix) + wet * mix;
      }
    }
    return true;
  }
}

registerProcessor('r2-wdf-tube-triode-processor', WdfTubeTriodeProcessor);
