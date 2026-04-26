/**
 * R13 — Vintage Amp Modeling composite builder
 *
 * Registers a NEW DSP node type **`vintage_amp_modeling`** that composes
 * existing R2/R3 primitives (tube preamp, tone stack, transformer) into a
 * Logic-Pro-Vintage-Amp-Modeling-style guitar amp signal chain. Differs from
 * the more parametric `circuit_fender_bassman` (R4) in that:
 *
 *   1. It is keyed off an `amp_model` preset table — eight era-correct heads,
 *      each baking in a specific tube_type, harmonic_signature (h2..h5
 *      relative weights), transformer_color (output-stage LP cutoff +
 *      saturation amount), bias drift offset and NFB amount.
 *   2. The signal chain has explicit "phase inverter" + "push-pull power
 *      amp" + "output transformer LUT" + "PSU sag with drift" stages, instead
 *      of collapsing everything into a single tube_amp worklet pass.
 *   3. The `amp_model` switch RETUNES the underlying R2/R3 nodes
 *      (waveshaper curves, biquad cutoffs, fallback gain stages) rather than
 *      changing topology. This keeps the AudioGraph stable across model
 *      switches (no rebuilds).
 *
 * Topology:
 *   input → input_pad → tube_preamp_v12ax7 (R2 wdf_tube_amp, stages=2,
 *                                            curve baked from harmonic_sig)
 *         → vintage_tone_stack (R2 wdf_tone_stack, fallback: bass/mid/treble
 *                               biquads + presence in NFB position)
 *         → phase_inverter (gentle drive + fixed lowshelf — emulates the
 *                           long-tail-pair stage feeding the power tubes)
 *         → push_pull_power_amp_eltype (R2 wdf_tube_amp, curve baked from
 *                                       tube_type EL84/EL34/6V6/6L6 plus
 *                                       per-model `bias` asymmetry)
 *         → output_transformer (R3 wdf_transformer, drive + saturation
 *                                from transformer_color preset)
 *         → output_transformer_LP (post-xfmr LPF — the LUT cutoff)
 *         → psu_sag_with_drift (R3 wdf_power_supply_sag — sag amount +
 *                               recovery from amp model)
 *         → vintage_cab_ir (ConvolverNode — synthetic per-cab IR)
 *         → mic_position_pre_eq (mic_type-keyed peak/shelf curve + position
 *                                LP/HP shift)
 *         → output_pad → output
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildVintageAmpModeling(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: { audioParam?, paramDef,
 *                                                      customSetter?, scale? } } }
 *
 * R2/R3 worklets may not be registered yet at composite-build time; every
 * `new AudioWorkletNode(ctx, name, ...)` is wrapped in `_safeWorklet()` which
 * falls back to a primitive substitute on registration miss. This lets the
 * graph build succeed today; once R2/R3 ship (audioWorklet.addModule run),
 * the same builder picks up the real WDF processors with no code change.
 *
 * Author: Agent R13-VintageAmp
 */

import { buildCabinetIR } from '../cabinet-ir.js';

// ── Worklet processor names owned by other agents (R2 / R3) ───────────────
const R2_TUBE_AMP    = 'r2-wdf-tube-amp-processor';
const R2_TONE_STACK  = 'r2-wdf-tone-stack-processor';
const R3_TRANSFORMER = 'r3-wdf-transformer-processor';
const R3_PSU_SAG     = 'r3-wdf-power-supply-sag-processor';

// ── Internal helpers ──────────────────────────────────────────────────────

function _safeWorklet(ctx, name, options = {}) {
  try {
    return new AudioWorkletNode(ctx, name, options);
  } catch (e) {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[R13-VAM] worklet ${name} unavailable, using fallback:`,
        e && e.message);
    }
    return null;
  }
}

function _workletParam(workletNode, paramName) {
  if (!workletNode || !workletNode.parameters) return null;
  return workletNode.parameters.get(paramName) || null;
}

// ── Amp-model preset table ────────────────────────────────────────────────
//
// Each preset bakes in:
//   tube_type           — preamp/power-tube class (informational + steers curve)
//   power_tube_type     — power amp tube (EL84 / EL34 / 6V6 / 6L6)
//   harmonic_signature  — relative weights for h2/h3/h4/h5 used to sculpt the
//                         preamp waveshaper curve. Even-order = warm/musical,
//                         odd-order = aggressive/squared. Values 0..1.
//   transformer_color   — { lp_cutoff_hz, sat_amount } applied to output
//                         transformer. Lower cutoff = darker (vintage iron),
//                         higher sat = more compression at master.
//   bias_drift          — −1..+1, asymmetric clipping offset for the power
//                         amp curve. Negative = colder bias (push-pull cleaner
//                         until breakup), positive = hotter (class-A flavour).
//   nfb_default         — 0..1, default amount of negative feedback
//                         (compression character).
//   tone_voicing        — { bass, mid, treble } default tone-stack centers
//                         (in Hz) used by the fallback biquads when the
//                         tone-stack worklet isn't loaded.
//   presence_freq       — corner of the post-NFB high shelf (the "presence" pot
//                         on a vintage head sits inside the NFB loop).
//   gain_makeup         — output-level multiplier so each model lands at
//                         roughly the same perceived loudness at unity master.
//
const AMP_MODEL_PRESETS = {
  tweed_5e3: {
    // Fender Tweed Deluxe 5E3 — cathode-biased class-A, warm 6V6s, low-NFB.
    tube_type: '12AX7',
    power_tube_type: '6V6',
    harmonic_signature: [0.85, 0.45, 0.25, 0.10],
    transformer_color: { lp_cutoff_hz: 4500, sat_amount: 0.55 },
    bias_drift: 0.30,
    nfb_default: 0.15,
    tone_voicing: { bass: 90, mid: 700, treble: 4500 },
    presence_freq: 3500,
    gain_makeup: 1.10,
  },
  tweed_5f6: {
    // Fender Bassman 5F6-A — fixed-bias 5881s, mid scoop-able.
    tube_type: '12AX7',
    power_tube_type: '6L6',
    harmonic_signature: [0.75, 0.55, 0.30, 0.15],
    transformer_color: { lp_cutoff_hz: 5500, sat_amount: 0.45 },
    bias_drift: 0.05,
    nfb_default: 0.30,
    tone_voicing: { bass: 80, mid: 500, treble: 5000 },
    presence_freq: 3000,
    gain_makeup: 1.00,
  },
  vox_ac30: {
    // VOX AC30TB — cathode-biased EL84, no NFB, top-boost circuit.
    tube_type: '12AX7',
    power_tube_type: 'EL84',
    harmonic_signature: [0.70, 0.65, 0.40, 0.25],
    transformer_color: { lp_cutoff_hz: 6500, sat_amount: 0.60 },
    bias_drift: 0.45,
    nfb_default: 0.05,
    tone_voicing: { bass: 100, mid: 1000, treble: 6500 },
    presence_freq: 4500,
    gain_makeup: 1.20,
  },
  marshall_plexi: {
    // Marshall 1959 Super Lead 'Plexi' — fixed-bias EL34, classic NFB loop.
    tube_type: '12AX7',
    power_tube_type: 'EL34',
    harmonic_signature: [0.55, 0.75, 0.45, 0.30],
    transformer_color: { lp_cutoff_hz: 6000, sat_amount: 0.50 },
    bias_drift: 0.10,
    nfb_default: 0.55,
    tone_voicing: { bass: 110, mid: 650, treble: 5500 },
    presence_freq: 3000,
    gain_makeup: 0.95,
  },
  marshall_jcm800: {
    // Marshall JCM800 2203 — cascaded-gain Plexi descendant, hotter preamp.
    tube_type: '12AX7',
    power_tube_type: 'EL34',
    harmonic_signature: [0.45, 0.85, 0.55, 0.35],
    transformer_color: { lp_cutoff_hz: 5500, sat_amount: 0.55 },
    bias_drift: 0.15,
    nfb_default: 0.65,
    tone_voicing: { bass: 110, mid: 600, treble: 6000 },
    presence_freq: 3200,
    gain_makeup: 0.90,
  },
  hiwatt: {
    // Hiwatt DR-103 — clean, articulate EL34, very high headroom.
    tube_type: '12AX7',
    power_tube_type: 'EL34',
    harmonic_signature: [0.65, 0.50, 0.20, 0.10],
    transformer_color: { lp_cutoff_hz: 8000, sat_amount: 0.30 },
    bias_drift: 0.00,
    nfb_default: 0.75,
    tone_voicing: { bass: 90, mid: 800, treble: 5500 },
    presence_freq: 4000,
    gain_makeup: 0.95,
  },
  orange_or120: {
    // Orange OR120 — EL34, mid-forward voicing, "graphic" tone control.
    tube_type: '12AX7',
    power_tube_type: 'EL34',
    harmonic_signature: [0.55, 0.70, 0.40, 0.20],
    transformer_color: { lp_cutoff_hz: 4500, sat_amount: 0.55 },
    bias_drift: 0.20,
    nfb_default: 0.45,
    tone_voicing: { bass: 100, mid: 900, treble: 4800 },
    presence_freq: 3200,
    gain_makeup: 1.00,
  },
  silvertone: {
    // Silvertone 1484 — 6L6 combo, lo-fi early-Beatles flavour.
    tube_type: '12AX7',
    power_tube_type: '6L6',
    harmonic_signature: [0.80, 0.50, 0.25, 0.15],
    transformer_color: { lp_cutoff_hz: 3800, sat_amount: 0.65 },
    bias_drift: 0.35,
    nfb_default: 0.20,
    tone_voicing: { bass: 95, mid: 750, treble: 4000 },
    presence_freq: 2800,
    gain_makeup: 1.15,
  },
};

const DEFAULT_MODEL = 'tweed_5e3';

// ── Cabinet preset table ──────────────────────────────────────────────────
// Each preset overrides the cabinet IR generator's frequency targets so we
// get a distinct synthetic IR per cab without shipping any wave files.
const CAB_PRESETS = {
  '1x12_alnico':    { lowCutHz: 95,  highCutHz: 5000, presencePeakHz: 3000,  durationSec: 0.040 },
  '2x12_celestion': { lowCutHz: 85,  highCutHz: 4500, presencePeakHz: 2700,  durationSec: 0.045 },
  '4x12_greenback': { lowCutHz: 80,  highCutHz: 4000, presencePeakHz: 2500,  durationSec: 0.050 },
  '4x10_jensen':    { lowCutHz: 100, highCutHz: 5500, presencePeakHz: 3500,  durationSec: 0.040 },
  '1x15_jbl':       { lowCutHz: 60,  highCutHz: 3500, presencePeakHz: 2000,  durationSec: 0.055 },
  '2x10_oxford':    { lowCutHz: 110, highCutHz: 5200, presencePeakHz: 3400,  durationSec: 0.038 },
};

// ── Mic-type EQ presets ───────────────────────────────────────────────────
// Each mic gets a 2-biquad colour curve emulating the on-axis response of a
// real mic at a typical guitar-cab placement. Logic models these in the same
// place. Values: { peak_freq, peak_gain_db, peak_q, hp_freq, hp_q }.
const MIC_PRESETS = {
  sm57: { peak_freq: 5500, peak_gain_db:  6, peak_q: 1.4, hp_freq: 80,  hp_q: 0.7 },
  sm7:  { peak_freq: 4500, peak_gain_db:  3, peak_q: 1.0, hp_freq: 60,  hp_q: 0.7 },
  r121: { peak_freq: 1200, peak_gain_db:  2, peak_q: 0.7, hp_freq: 50,  hp_q: 0.5 },
  u87:  { peak_freq: 8500, peak_gain_db:  4, peak_q: 1.2, hp_freq: 40,  hp_q: 0.5 },
};

// ── Curve generators ──────────────────────────────────────────────────────
//
// Build a soft-clipping waveshaper curve weighted by the harmonic_signature
// table. We construct y(x) by summing Chebyshev-T basis polynomials of order
// 2..5 (h2..h5), then blend with a base tanh for the fundamental shape and
// an `asym` offset for bias drift.
function _makeHarmonicCurve(harmonicSig, asym = 0, drive = 1, N = 4096) {
  const [h2, h3, h4, h5] = harmonicSig;
  const curve = new Float32Array(N);
  // y_offset so the curve is DC-centred at x=0 once the asym shift is applied
  const xOff = asym;
  const baseAt = (x) => {
    // Chebyshev T_n: T2 = 2x^2 - 1, T3 = 4x^3 - 3x, T4 = 8x^4 - 8x^2 + 1,
    // T5 = 16x^5 - 20x^3 + 5x. Even-order terms produce h2/h4 (warmth),
    // odd-order produces h3/h5 (squared aggression).
    const t1 = x;
    const t2 = 2 * x * x - 1;
    const t3 = 4 * x * x * x - 3 * x;
    const t4 = 8 * x * x * x * x - 8 * x * x + 1;
    const t5 = 16 * x * x * x * x * x - 20 * x * x * x + 5 * x;
    // Base shape: tanh(x*drive) for the fundamental, then sprinkled with
    // weighted harmonics. We add a small fraction (~0.15) of each so the
    // curve is dominated by tanh — the harmonics colour, they don't replace.
    const base = Math.tanh(x * drive);
    return base
         + 0.15 * h2 * t2
         + 0.20 * h3 * t3
         + 0.10 * h4 * t4
         + 0.12 * h5 * t5
         - 0.15 * h2; // T2(0)=−1; subtract so curve passes near origin
  };
  // DC offset at x=−xOff so output is zero-mean
  const yOff = baseAt(-xOff);
  for (let i = 0; i < N; i++) {
    const x = (i / (N - 1)) * 2 - 1;
    let y = baseAt(x + xOff) - yOff;
    if (y >  1.0) y =  1.0;
    if (y < -1.0) y = -1.0;
    curve[i] = y;
  }
  return curve;
}

// Power-tube curve: same idea but with a different base. EL84 = looser/
// crunchier knee, EL34 = mid focus, 6V6 = soft warm, 6L6 = stiff-then-clip.
function _makePowerTubeCurve(tubeType, harmonicSig, biasDrift = 0, N = 4096) {
  const [h2, h3, h4, h5] = harmonicSig;
  const k = ({ EL84: 1.6, EL34: 1.4, '6V6': 1.2, '6L6': 1.1 })[tubeType] || 1.3;
  const curve = new Float32Array(N);
  const xOff = biasDrift * 0.4;
  const baseAt = (x) => {
    const xs = x * k;
    const t2 = 2 * xs * xs - 1;
    const t3 = 4 * xs * xs * xs - 3 * xs;
    const fund = Math.tanh(xs);
    return fund + 0.08 * h2 * t2 + 0.15 * h3 * t3 + 0.05 * (h4 + h5) * Math.sin(xs * 2);
  };
  const yOff = baseAt(-xOff);
  for (let i = 0; i < N; i++) {
    const x = (i / (N - 1)) * 2 - 1;
    let y = baseAt(x + xOff) - yOff;
    if (y >  1.0) y =  1.0;
    if (y < -1.0) y = -1.0;
    curve[i] = y;
  }
  return curve;
}

// ── Builder ───────────────────────────────────────────────────────────────

/**
 * Build the vintage_amp_modeling composite.
 *
 * Schema:
 *   {
 *     type: 'vintage_amp_modeling',
 *     params: {
 *       amp_model:    one of AMP_MODEL_PRESETS keys,
 *       gain:         0..10  (preamp drive; '@<id>' to modulate),
 *       bass:         0..1,
 *       mid:          0..1,
 *       treble:       0..1,
 *       presence:     0..1,
 *       master:       0..1   (push power amp),
 *       bias:         0..1   (override of preset bias_drift),
 *       nfb:          0..1   (override of preset nfb_default),
 *       cab_model:    one of CAB_PRESETS keys,
 *       mic_type:     one of MIC_PRESETS keys,
 *       mic_position: 0..1   (0=on-cap bright, 1=off-axis darker),
 *       output_level: 0..1
 *     }
 *   }
 */
export function buildVintageAmpModeling(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  // ── Resolve initial preset ────────────────────────────────────────────
  const initialModelKey = (typeof params.amp_model === 'string'
    && !params.amp_model.startsWith('@')
    && AMP_MODEL_PRESETS[params.amp_model])
    ? params.amp_model
    : DEFAULT_MODEL;
  let preset = AMP_MODEL_PRESETS[initialModelKey];

  const initialCabKey = (typeof params.cab_model === 'string'
    && !params.cab_model.startsWith('@')
    && CAB_PRESETS[params.cab_model])
    ? params.cab_model
    : '4x12_greenback';
  let cabPreset = CAB_PRESETS[initialCabKey];

  const initialMicKey = (typeof params.mic_type === 'string'
    && !params.mic_type.startsWith('@')
    && MIC_PRESETS[params.mic_type])
    ? params.mic_type
    : 'sm57';
  let micPreset = MIC_PRESETS[initialMicKey];

  // ── Graph nodes ───────────────────────────────────────────────────────
  const input = ctx.createGain();
  const output = ctx.createGain();

  // input padding — keeps level sane regardless of upstream gain
  const inputPad = ctx.createGain();
  inputPad.gain.value = 0.9;

  // -- Preamp stage (R2 wdf_tube_amp w/ harmonic-signature-baked fallback) --
  const preampWorklet = _safeWorklet(ctx, R2_TUBE_AMP, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    parameterData: { stages: 2, gain: 1.0, bias: -1.5, output_level: 0.55 },
  });
  let preampShaper = null;
  let preampDrive = null;
  if (!preampWorklet) {
    preampDrive = ctx.createGain();
    preampDrive.gain.value = 1.0;
    preampShaper = ctx.createWaveShaper();
    preampShaper.oversample = '4x';
    preampShaper.curve = _makeHarmonicCurve(preset.harmonic_signature, 0, 2.0);
  }

  // -- Vintage tone stack (R2 wdf_tone_stack OR fallback FMV-3-band) --
  const toneStack = _safeWorklet(ctx, R2_TONE_STACK, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    parameterData: { bass: 0.5, mid: 0.5, treble: 0.5 },
  });
  const bassFb = ctx.createBiquadFilter();
  bassFb.type = 'lowshelf';
  bassFb.frequency.value = preset.tone_voicing.bass;
  bassFb.gain.value = 0;
  const midFb = ctx.createBiquadFilter();
  midFb.type = 'peaking';
  midFb.frequency.value = preset.tone_voicing.mid;
  midFb.Q.value = 0.7;
  midFb.gain.value = 0;
  const trebFb = ctx.createBiquadFilter();
  trebFb.type = 'highshelf';
  trebFb.frequency.value = preset.tone_voicing.treble;
  trebFb.gain.value = 0;

  // -- Phase inverter — small fixed-drive waveshaper + gentle low-shelf --
  // Real long-tail-pair stages add a slight tilt + asymmetric headroom into
  // the power amp; we approximate with a constant-curve waveshaper.
  const phaseInverter = ctx.createWaveShaper();
  phaseInverter.oversample = '2x';
  {
    const N = 1024;
    const c = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const x = (i / (N - 1)) * 2 - 1;
      c[i] = Math.tanh(x * 1.2);
    }
    phaseInverter.curve = c;
  }
  const piShelf = ctx.createBiquadFilter();
  piShelf.type = 'lowshelf';
  piShelf.frequency.value = 200;
  piShelf.gain.value = -1.5; // small tilt to mimic LTP coupling

  // -- Push-pull power amp (R2 worklet OR per-tube-type fallback shaper) --
  const powerAmpWorklet = _safeWorklet(ctx, R2_TUBE_AMP, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    parameterData: { stages: 1, gain: 1.0, bias: -1.0, output_level: 0.7 },
  });
  let powerAmpDrive = null;
  let powerAmpShaper = null;
  if (!powerAmpWorklet) {
    powerAmpDrive = ctx.createGain();
    powerAmpDrive.gain.value = 1.0;
    powerAmpShaper = ctx.createWaveShaper();
    powerAmpShaper.oversample = '4x';
    powerAmpShaper.curve = _makePowerTubeCurve(
      preset.power_tube_type,
      preset.harmonic_signature,
      preset.bias_drift
    );
  }

  // -- Output transformer (R3) — drive + saturation come from preset.transformer_color --
  const xfmrWorklet = _safeWorklet(ctx, R3_TRANSFORMER, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    parameterData: {
      drive: 1,
      saturation: preset.transformer_color.sat_amount,
      mix: 1,
    },
  });
  let xfmrFallbackShaper = null;
  if (!xfmrWorklet) {
    xfmrFallbackShaper = ctx.createWaveShaper();
    xfmrFallbackShaper.oversample = '4x';
    {
      const N = 2048;
      const c = new Float32Array(N);
      const sat = preset.transformer_color.sat_amount;
      for (let i = 0; i < N; i++) {
        const x = (i / (N - 1)) * 2 - 1;
        c[i] = (2 / Math.PI) * Math.atan(x * (1 + sat * 1.5)) * 0.95;
      }
      xfmrFallbackShaper.curve = c;
    }
  }

  // -- Output transformer LP — the LUT cutoff (always live) --
  const xfmrLP = ctx.createBiquadFilter();
  xfmrLP.type = 'lowpass';
  xfmrLP.frequency.value = preset.transformer_color.lp_cutoff_hz;
  xfmrLP.Q.value = 0.7071;

  // -- PSU sag (R3 wdf_power_supply_sag, ScriptProcessor fallback) --
  // Worklet may not be loaded — instead of using the R3 builder (which
  // wraps it in dry/wet routing), we instantiate a simple sag-controlled
  // gain stage as fallback.
  const psuWorklet = _safeWorklet(ctx, R3_PSU_SAG, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    parameterData: { sag: 0.3, recovery: 0.05 },
  });
  // For the fallback we use a side-chained gain envelope. We don't bother
  // with ScriptProcessor here — sag is subtle on vintage amps and a static
  // unity gain is acceptable when the worklet isn't present (consistent
  // with how R4's circuit_fender_bassman handles missing R3 nodes).
  const psuFallbackGain = psuWorklet ? null : ctx.createGain();
  if (psuFallbackGain) {
    // Slightly reduce headroom to emulate average sag compression
    psuFallbackGain.gain.value = 1.0 - 0.05 * preset.bias_drift;
  }

  // -- Cabinet IR --
  const cab = ctx.createConvolver();
  try {
    cab.buffer = buildCabinetIR(ctx, cabPreset);
  } catch (e) { /* ctx closed */ }

  // -- Mic position pre-EQ --
  // Two biquads: high-pass (per-mic) + peaking (per-mic), plus an extra
  // tilt biquad whose corner is shifted by mic_position.
  const micHP = ctx.createBiquadFilter();
  micHP.type = 'highpass';
  micHP.frequency.value = micPreset.hp_freq;
  micHP.Q.value = micPreset.hp_q;
  const micPeak = ctx.createBiquadFilter();
  micPeak.type = 'peaking';
  micPeak.frequency.value = micPreset.peak_freq;
  micPeak.Q.value = micPreset.peak_q;
  micPeak.gain.value = micPreset.peak_gain_db;
  const micPositionTilt = ctx.createBiquadFilter();
  micPositionTilt.type = 'highshelf';
  micPositionTilt.frequency.value = 4000;
  micPositionTilt.gain.value = 0; // updated by mic_position param

  // -- Presence (post-cab high shelf — vintage amps wire this in NFB) --
  const presence = ctx.createBiquadFilter();
  presence.type = 'highshelf';
  presence.frequency.value = preset.presence_freq;
  presence.gain.value = 0;

  // -- Output padding + master --
  const outputPad = ctx.createGain();
  outputPad.gain.value = preset.gain_makeup;

  // ── Wire the chain ────────────────────────────────────────────────────
  input.connect(inputPad);

  let head = inputPad;
  if (preampWorklet) {
    head.connect(preampWorklet); head = preampWorklet;
  } else {
    head.connect(preampDrive);
    preampDrive.connect(preampShaper);
    head = preampShaper;
  }

  if (toneStack) {
    head.connect(toneStack); head = toneStack;
  } else {
    head.connect(bassFb);
    bassFb.connect(midFb);
    midFb.connect(trebFb);
    head = trebFb;
  }

  head.connect(phaseInverter);
  phaseInverter.connect(piShelf);
  head = piShelf;

  if (powerAmpWorklet) {
    head.connect(powerAmpWorklet); head = powerAmpWorklet;
  } else {
    head.connect(powerAmpDrive);
    powerAmpDrive.connect(powerAmpShaper);
    head = powerAmpShaper;
  }

  if (xfmrWorklet) {
    head.connect(xfmrWorklet); head = xfmrWorklet;
  } else {
    head.connect(xfmrFallbackShaper); head = xfmrFallbackShaper;
  }

  head.connect(xfmrLP);
  head = xfmrLP;

  if (psuWorklet) {
    head.connect(psuWorklet); head = psuWorklet;
  } else {
    head.connect(psuFallbackGain); head = psuFallbackGain;
  }

  head.connect(cab);
  cab.connect(micHP);
  micHP.connect(micPeak);
  micPeak.connect(micPositionTilt);
  micPositionTilt.connect(presence);
  presence.connect(outputPad);
  outputPad.connect(output);

  // ── Helper: rebuild model-dependent state after `amp_model` change ────
  function applyAmpModel(modelKey) {
    const p = AMP_MODEL_PRESETS[modelKey];
    if (!p) return;
    preset = p;

    // Update fallback preamp curve
    if (preampShaper) {
      preampShaper.curve = _makeHarmonicCurve(p.harmonic_signature, 0, 2.0);
    }
    // Update fallback power-tube curve (incl. bias drift)
    if (powerAmpShaper) {
      powerAmpShaper.curve = _makePowerTubeCurve(
        p.power_tube_type,
        p.harmonic_signature,
        p.bias_drift
      );
    }
    // Update fallback transformer waveshaper
    if (xfmrFallbackShaper) {
      const N = 2048;
      const c = new Float32Array(N);
      const sat = p.transformer_color.sat_amount;
      for (let i = 0; i < N; i++) {
        const x = (i / (N - 1)) * 2 - 1;
        c[i] = (2 / Math.PI) * Math.atan(x * (1 + sat * 1.5)) * 0.95;
      }
      xfmrFallbackShaper.curve = c;
    }
    // Update transformer LP cutoff
    xfmrLP.frequency.setTargetAtTime(
      p.transformer_color.lp_cutoff_hz, ctx.currentTime, 0.02
    );
    // Update tone-stack centres for fallback biquads (works whether or not
    // the worklet is live — fallback band gains are still driven by params).
    bassFb.frequency.value = p.tone_voicing.bass;
    midFb.frequency.value  = p.tone_voicing.mid;
    trebFb.frequency.value = p.tone_voicing.treble;
    presence.frequency.value = p.presence_freq;
    outputPad.gain.setTargetAtTime(p.gain_makeup, ctx.currentTime, 0.02);

    // Update PSU sag amount via the worklet's k-rate param (best effort).
    if (psuWorklet) {
      const sagP = _workletParam(psuWorklet, 'sag');
      if (sagP) sagP.value = 0.3 + 0.2 * p.bias_drift;
    } else if (psuFallbackGain) {
      psuFallbackGain.gain.setTargetAtTime(
        1.0 - 0.05 * p.bias_drift, ctx.currentTime, 0.02
      );
    }
    // Update transformer worklet saturation
    if (xfmrWorklet) {
      const satP = _workletParam(xfmrWorklet, 'saturation');
      if (satP) satP.value = p.transformer_color.sat_amount;
    }
  }

  function applyCab(cabKey) {
    const cp = CAB_PRESETS[cabKey];
    if (!cp) return;
    cabPreset = cp;
    try { cab.buffer = buildCabinetIR(ctx, cp); } catch (e) { /* ignore */ }
  }

  function applyMic(micKey) {
    const mp = MIC_PRESETS[micKey];
    if (!mp) return;
    micPreset = mp;
    micHP.frequency.setTargetAtTime(mp.hp_freq, ctx.currentTime, 0.01);
    micHP.Q.setTargetAtTime(mp.hp_q, ctx.currentTime, 0.01);
    micPeak.frequency.setTargetAtTime(mp.peak_freq, ctx.currentTime, 0.01);
    micPeak.Q.setTargetAtTime(mp.peak_q, ctx.currentTime, 0.01);
    micPeak.gain.setTargetAtTime(mp.peak_gain_db, ctx.currentTime, 0.01);
  }

  // ── Param wiring ──────────────────────────────────────────────────────
  for (const [key, val] of Object.entries(params)) {
    const isModulated = typeof val === 'string' && val.startsWith('@');
    const paramId = isModulated ? val.slice(1) : null;

    switch (key) {
      case 'amp_model': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              // Accept either a string key or a numeric 0..N index
              const keys = Object.keys(AMP_MODEL_PRESETS);
              let modelKey;
              if (typeof v === 'string') modelKey = v;
              else if (typeof v === 'number') {
                const idx = Math.max(0, Math.min(keys.length - 1,
                  v <= 1 ? Math.floor(v * keys.length) : Math.round(v)));
                modelKey = keys[idx];
              }
              if (modelKey && AMP_MODEL_PRESETS[modelKey]) applyAmpModel(modelKey);
            },
          };
        }
        break;
      }
      case 'gain': {
        const ap = _workletParam(preampWorklet, 'gain');
        if (isModulated) {
          if (ap) {
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => { ap.value = Math.max(0, v); },
            };
          } else {
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => {
                if (preampDrive) preampDrive.gain.value = Math.max(0, v);
              },
            };
          }
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          else if (preampDrive) preampDrive.gain.value = val;
        }
        break;
      }
      case 'bass': {
        const ap = _workletParam(toneStack, 'bass');
        if (isModulated) {
          if (ap) targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          else targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => { bassFb.gain.value = (v - 0.5) * 24; },
          };
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          else bassFb.gain.value = (val - 0.5) * 24;
        }
        break;
      }
      case 'mid': {
        const ap = _workletParam(toneStack, 'mid');
        if (isModulated) {
          if (ap) targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          else targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => { midFb.gain.value = (v - 0.5) * 18; },
          };
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          else midFb.gain.value = (val - 0.5) * 18;
        }
        break;
      }
      case 'treble': {
        const ap = _workletParam(toneStack, 'treble');
        if (isModulated) {
          if (ap) targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          else targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => { trebFb.gain.value = (v - 0.5) * 24; },
          };
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          else trebFb.gain.value = (val - 0.5) * 24;
        }
        break;
      }
      case 'presence': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              presence.gain.setTargetAtTime(v * 12, ctx.currentTime, 0.01);
            },
          };
        } else if (typeof val === 'number') {
          presence.gain.value = val * 12;
        }
        break;
      }
      case 'master': {
        const ap = _workletParam(powerAmpWorklet, 'gain');
        if (isModulated) {
          if (ap) {
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => { ap.value = Math.max(0, v); },
            };
          } else {
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => {
                if (powerAmpDrive) powerAmpDrive.gain.value = Math.max(0, v);
              },
            };
          }
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          else if (powerAmpDrive) powerAmpDrive.gain.value = val;
        }
        break;
      }
      case 'bias': {
        // Override of preset.bias_drift. For the worklet path we forward to
        // the power-amp worklet's bias param if it has one. For the fallback
        // we re-bake the power-tube curve on the fly.
        const ap = _workletParam(powerAmpWorklet, 'bias');
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              if (ap) ap.value = (v - 0.5) * 3;     // map 0..1 → -1.5..+1.5
              else if (powerAmpShaper) {
                powerAmpShaper.curve = _makePowerTubeCurve(
                  preset.power_tube_type,
                  preset.harmonic_signature,
                  (v - 0.5) * 2
                );
              }
            },
          };
        } else if (typeof val === 'number') {
          if (ap) ap.value = (val - 0.5) * 3;
          else if (powerAmpShaper) {
            powerAmpShaper.curve = _makePowerTubeCurve(
              preset.power_tube_type,
              preset.harmonic_signature,
              (val - 0.5) * 2
            );
          }
        }
        break;
      }
      case 'nfb': {
        // NFB increases damping in the presence/output stage and slightly
        // tightens low end. We simulate by gently pulling presence shelf gain
        // back toward zero and lifting the LP slightly (more NFB = wider).
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              const k = Math.max(0, Math.min(1, v));
              // More NFB = LP cutoff opens up by up to +1 octave
              const f = preset.transformer_color.lp_cutoff_hz * (1 + 0.5 * k);
              xfmrLP.frequency.setTargetAtTime(f, ctx.currentTime, 0.02);
              // More NFB = gentler input pad to compensate
              inputPad.gain.setTargetAtTime(0.9 - 0.1 * k, ctx.currentTime, 0.02);
            },
          };
        } else if (typeof val === 'number') {
          const k = Math.max(0, Math.min(1, val));
          const f = preset.transformer_color.lp_cutoff_hz * (1 + 0.5 * k);
          xfmrLP.frequency.value = f;
          inputPad.gain.value = 0.9 - 0.1 * k;
        }
        break;
      }
      case 'cab_model': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              const keys = Object.keys(CAB_PRESETS);
              let cabKey;
              if (typeof v === 'string') cabKey = v;
              else if (typeof v === 'number') {
                const idx = Math.max(0, Math.min(keys.length - 1,
                  v <= 1 ? Math.floor(v * keys.length) : Math.round(v)));
                cabKey = keys[idx];
              }
              if (cabKey && CAB_PRESETS[cabKey]) applyCab(cabKey);
            },
          };
        }
        break;
      }
      case 'mic_type': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              const keys = Object.keys(MIC_PRESETS);
              let micKey;
              if (typeof v === 'string') micKey = v;
              else if (typeof v === 'number') {
                const idx = Math.max(0, Math.min(keys.length - 1,
                  v <= 1 ? Math.floor(v * keys.length) : Math.round(v)));
                micKey = keys[idx];
              }
              if (micKey && MIC_PRESETS[micKey]) applyMic(micKey);
            },
          };
        }
        break;
      }
      case 'mic_position': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              // 0 = on-cap (bright, +HF), 1 = off-axis (darker, -HF).
              const tilt = (0.5 - Math.max(0, Math.min(1, v))) * 6; // ±3 dB
              micPositionTilt.gain.setTargetAtTime(tilt, ctx.currentTime, 0.02);
            },
          };
        } else if (typeof val === 'number') {
          const tilt = (0.5 - Math.max(0, Math.min(1, val))) * 6;
          micPositionTilt.gain.value = tilt;
        }
        break;
      }
      case 'output_level': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            audioParam: output.gain,
          };
        } else if (typeof val === 'number') {
          output.gain.value = val;
        }
        break;
      }
      default:
        break;
    }
  }

  return { input, output, paramTargets: targets };
}

// ── Default export: NODE_BUILDERS map ─────────────────────────────────────
const R13_VINTAGE_AMP_BUILDERS = {
  vintage_amp_modeling: buildVintageAmpModeling,
};

// Export presets for tests + UI introspection
export { AMP_MODEL_PRESETS, CAB_PRESETS, MIC_PRESETS };

export default R13_VINTAGE_AMP_BUILDERS;
