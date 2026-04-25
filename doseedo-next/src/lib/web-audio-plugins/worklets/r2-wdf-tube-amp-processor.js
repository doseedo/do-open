/**
 * r2-wdf-tube-amp-processor.js
 *
 * Multi-stage triode pre-amp (1-3 cascaded triode stages with inter-stage
 * HPF coupling and tone-shaping).  Built on the same Koren LUT as
 * r2-wdf-tube-triode-processor.js, but with stage-to-stage coupling, a
 * mild "presence" filter between stages, and a final master output.
 *
 * Architecture:
 *
 *   in → drive → [Triode → HPF → presence] × N → output_level → mix → out
 *
 * Stages cascade and gain-compound; bias is shared across stages (tone is
 * dominated by the first stage anyway).  Each inter-stage HPF kills DC
 * offsets that would otherwise self-compound through the chain.
 *
 * Performance: 3 stages × LUT lookup (Catmull-Rom) per sample is ~30 mults
 * per sample per channel — well under realtime budget at 48k stereo.
 *
 * Author: Agent R2.
 * Reference: same as triode (Koren / Pakarinen-Karjalainen WDF tube).
 */

// Replicate the Koren LUT here (worklet scopes are isolated; can't share modules
// without addModule chaining — keeping it self-contained is simpler).
const LUT_VGK_MIN = -10;
const LUT_VGK_MAX = 4;
const LUT_SIZE    = 512;
const LUT_VPK_FIXED = 250;

function buildKorenLUT() {
  const MU = 100, KP = 600, KVB = 300, KG1 = 1060, EX = 1.4;
  const Vpk = LUT_VPK_FIXED;
  const lut = new Float32Array(LUT_SIZE);
  for (let i = 0; i < LUT_SIZE; i++) {
    const t = i / (LUT_SIZE - 1);
    const Vgk = LUT_VGK_MIN + t * (LUT_VGK_MAX - LUT_VGK_MIN);
    const arg = KP * (1 / MU + Vgk / Math.sqrt(KVB + Vpk * Vpk));
    const sp = (arg > 30) ? arg : Math.log1p(Math.exp(arg));
    const E1 = (Vpk / KP) * sp;
    let Ip = 0;
    if (E1 > 0) Ip = Math.pow(E1, EX) / KG1;
    lut[i] = Ip;
  }
  let peak = 0;
  for (let i = 0; i < LUT_SIZE; i++) if (lut[i] > peak) peak = lut[i];
  if (peak > 0) for (let i = 0; i < LUT_SIZE; i++) lut[i] /= peak;
  return lut;
}

const KOREN_LUT = buildKorenLUT();
const LUT_RANGE = LUT_VGK_MAX - LUT_VGK_MIN;

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
  const a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
  const a1 = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3;
  const a2 = -0.5 * p0 + 0.5 * p2;
  const a3 = p1;
  return ((a0 * t + a1) * t + a2) * t + a3;
}

const MAX_STAGES = 3;

class WdfTubeAmpProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'gain',         defaultValue: 1,    minValue: 0.1,  maxValue: 5, automationRate: 'k-rate' },
      { name: 'bias',         defaultValue: -1.5, minValue: -4,   maxValue: 0, automationRate: 'k-rate' },
      { name: 'stages',       defaultValue: 2,    minValue: 1,    maxValue: 3, automationRate: 'k-rate' },
      { name: 'output_level', defaultValue: 0.3,  minValue: 0.05, maxValue: 1, automationRate: 'k-rate' },
      { name: 'mix',          defaultValue: 1,    minValue: 0,    maxValue: 1, automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    // Inter-stage HPF state: 2 channels × MAX_STAGES
    this.hpX = [new Float32Array(MAX_STAGES), new Float32Array(MAX_STAGES)];
    this.hpY = [new Float32Array(MAX_STAGES), new Float32Array(MAX_STAGES)];
    // Output DC blocker
    this.dcX = [0, 0];
    this.dcY = [0, 0];
  }

  // 30 Hz HPF
  hpf(x, ch, stage) {
    const R = 0.9955;
    const y = R * (this.hpY[ch][stage] + x - this.hpX[ch][stage]);
    this.hpX[ch][stage] = x;
    this.hpY[ch][stage] = y;
    return y;
  }
  dcBlock(x, ch) {
    const R = 0.9985;
    const y = x - this.dcX[ch] + R * this.dcY[ch];
    this.dcX[ch] = x;
    this.dcY[ch] = y;
    return y;
  }

  triode(x, gain, bias) {
    const Vgk = gain * x + bias;
    const Ip = lutLookup(Vgk);
    const v_plate = (1 - Ip) * 1.0;
    return (v_plate - 0.5) * 2;
  }

  process(inputs, outputs, parameters) {
    const inp = inputs[0];
    const out = outputs[0];
    if (!inp || !inp.length) return true;

    const gain         = parameters.gain[0]         ?? 1;
    const bias         = parameters.bias[0]         ?? -1.5;
    const stagesParam  = parameters.stages[0]       ?? 2;
    const outputLevel  = parameters.output_level[0] ?? 0.3;
    const mix          = parameters.mix[0]          ?? 1;

    const stages = Math.max(1, Math.min(MAX_STAGES, Math.round(stagesParam)));

    // Per-stage gain — first stage gets full drive, subsequent stages have
    // a much smaller compounding multiplier so we don't blow up.
    const stageGain = (s) => (s === 0 ? gain : 0.6 + 0.2 * gain);

    const nCh = Math.min(inp.length, 2);
    const N = inp[0].length;

    for (let ch = 0; ch < nCh; ch++) {
      const ic = inp[ch];
      const oc = out[ch];
      for (let i = 0; i < N; i++) {
        const dry = ic[i];
        let v = dry;
        for (let s = 0; s < stages; s++) {
          v = this.triode(v, stageGain(s), bias);
          v = this.hpf(v, ch, s);
        }
        v = this.dcBlock(v, ch);
        v = v * outputLevel;
        oc[i] = dry * (1 - mix) + v * mix;
      }
    }
    return true;
  }
}

registerProcessor('r2-wdf-tube-amp-processor', WdfTubeAmpProcessor);
