/**
 * R13 — Amp Designer (composite guitar amp simulator).
 *
 * Registers a NEW node type `amp_designer`. Internally it composes existing
 * R2/R3 worklets + a ConvolverNode-backed cabinet IR loader to mimic Logic
 * Pro's Amp Designer chain:
 *
 *     input → pre_gain
 *           → R2 wdf_tube_triode  (preamp drive / character)
 *           → R2 wdf_tone_stack   (bass / mid / treble)
 *           → R2 wdf_tube_amp     (power-amp drive / bias / stages)
 *           → R3 wdf_transformer  (output transformer saturation)
 *           → R3 wdf_power_supply_sag (envelope-following gain droop)
 *           → ConvolverNode       (cabinet IR — procedurally generated per cab_model)
 *           → presence shelf      (high-shelf after power amp)
 *           → post_gain (output_level)
 *           → output
 *
 * R13 deliberately does NOT touch the R2/R3 builder files — it constructs the
 * worklets directly via `_safeWorklet()` and falls back to primitive
 * (BiquadFilter / WaveShaper) substitutes when a worklet isn't registered yet.
 * The same builder picks up the real worklets once they load — no code change.
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildAmpDesigner(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: {...} } }
 *
 * Sister builders `bass_amp_designer` and `vintage_amp_modeling` (other
 * agents) share the R2/R3 substrate but live in their own files.
 *
 * Author: Agent R13 (Amp Designer)
 */

// ── Worklet processor names (owned by R2 / R3) ─────────────────────────────
const R2_TUBE_TRIODE = 'r2-wdf-tube-triode-processor';
const R2_TUBE_AMP    = 'r2-wdf-tube-amp-processor';
const R2_TONE_STACK  = 'r2-wdf-tone-stack-processor';
const R3_TRANSFORMER = 'r3-wdf-transformer-processor';

// ── Helpers ─────────────────────────────────────────────────────────────────

function _safeWorklet(ctx, name, options = {}) {
  try {
    return new AudioWorkletNode(ctx, name, options);
  } catch (e) {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[R13:AmpDesigner] worklet ${name} unavailable, using fallback:`, e && e.message);
    }
    return null;
  }
}

function _workletParam(workletNode, paramName) {
  if (!workletNode || !workletNode.parameters) return null;
  return workletNode.parameters.get(paramName) || null;
}

function _clip01(v) { return Math.max(0, Math.min(1, v)); }
function _clip(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ── Amp model preset table ──────────────────────────────────────────────────
// Each preset retunes the existing R2/R3 worklets' internal params. Numbers
// are deliberately approximate ("sounds like"), not calibrated to a real Mesa.
//
// preamp_drive:  R2 wdf_tube_triode `drive` (0..2)
// preamp_bias:   R2 wdf_tube_triode `bias` (-2..0; more negative = colder)
// power_drive:   R2 wdf_tube_amp `gain` (0..3)
// power_bias:    R2 wdf_tube_amp `bias` (-2..0)
// power_stages:  R2 wdf_tube_amp `stages` (1..4)
// power_out:     R2 wdf_tube_amp `output_level` (0..1)
// xfmr_drive:    R3 wdf_transformer `drive` (0..3)
// xfmr_sat:      R3 wdf_transformer `saturation` (0..1)
// sag_amount:    R3 wdf_power_supply_sag `sag` (0..1)
// sag_recovery:  R3 wdf_power_supply_sag `recovery` (0.005..0.5 sec)
// presence_db:   high-shelf nominal gain at presence=0.5 (dB)
// tone_curve:    {bass_offset, mid_offset, treble_offset} — additive bias on
//                the user's tone-stack params so e.g. tweed sits darker.
// cab_default:   default cab_model when this amp is selected
const AMP_MODELS = {
  tweed: {
    preamp_drive: 1.4, preamp_bias: -1.4,
    power_drive:  1.6, power_bias:  -1.6, power_stages: 2, power_out: 0.55,
    xfmr_drive:   1.2, xfmr_sat:   0.55,
    sag_amount:   0.55, sag_recovery: 0.05,
    presence_db:  3,
    tone_curve:   { bass: 0.05, mid:  0.10, treble: -0.05 },
    cab_default:  '1x12',
  },
  british_clean: {
    preamp_drive: 0.7, preamp_bias: -1.6,
    power_drive:  0.9, power_bias:  -1.8, power_stages: 2, power_out: 0.65,
    xfmr_drive:   1.0, xfmr_sat:   0.30,
    sag_amount:   0.20, sag_recovery: 0.08,
    presence_db:  5,
    tone_curve:   { bass: 0.00, mid: -0.05, treble: 0.05 },
    cab_default:  '4x12_open',
  },
  british_crunch: {
    preamp_drive: 1.5, preamp_bias: -1.5,
    power_drive:  1.8, power_bias:  -1.5, power_stages: 3, power_out: 0.6,
    xfmr_drive:   1.4, xfmr_sat:   0.55,
    sag_amount:   0.40, sag_recovery: 0.06,
    presence_db:  6,
    tone_curve:   { bass: 0.00, mid:  0.05, treble: 0.05 },
    cab_default:  '4x12_closed',
  },
  modern_clean: {
    preamp_drive: 0.6, preamp_bias: -1.7,
    power_drive:  0.7, power_bias:  -1.7, power_stages: 2, power_out: 0.7,
    xfmr_drive:   0.9, xfmr_sat:   0.20,
    sag_amount:   0.10, sag_recovery: 0.02,
    presence_db:  6,
    tone_curve:   { bass: 0.05, mid:  0.00, treble: 0.05 },
    cab_default:  '2x12',
  },
  modern_high_gain: {
    preamp_drive: 1.9, preamp_bias: -1.2,
    power_drive:  2.1, power_bias:  -1.4, power_stages: 4, power_out: 0.55,
    xfmr_drive:   1.6, xfmr_sat:   0.70,
    sag_amount:   0.50, sag_recovery: 0.04,
    presence_db:  7,
    tone_curve:   { bass: -0.05, mid: -0.10, treble: 0.10 },
    cab_default:  '4x12_closed',
  },
  class_a: {
    preamp_drive: 1.1, preamp_bias: -1.5,
    power_drive:  1.3, power_bias:  -1.6, power_stages: 2, power_out: 0.6,
    xfmr_drive:   1.1, xfmr_sat:   0.45,
    sag_amount:   0.35, sag_recovery: 0.07,
    presence_db:  4,
    tone_curve:   { bass: 0.00, mid:  0.05, treble: 0.00 },
    cab_default:  '2x12',
  },
  blackface: {
    preamp_drive: 0.9, preamp_bias: -1.6,
    power_drive:  1.1, power_bias:  -1.7, power_stages: 2, power_out: 0.65,
    xfmr_drive:   1.0, xfmr_sat:   0.35,
    sag_amount:   0.25, sag_recovery: 0.08,
    presence_db:  5,
    tone_curve:   { bass: 0.05, mid: -0.05, treble: 0.10 },
    cab_default:  '2x12',
  },
  metal: {
    preamp_drive: 2.0, preamp_bias: -1.0,
    power_drive:  2.4, power_bias:  -1.3, power_stages: 4, power_out: 0.5,
    xfmr_drive:   1.8, xfmr_sat:   0.80,
    sag_amount:   0.60, sag_recovery: 0.03,
    presence_db:  9,
    tone_curve:   { bass: -0.10, mid: -0.20, treble: 0.15 },
    cab_default:  '4x12_closed',
  },
  boutique: {
    preamp_drive: 1.2, preamp_bias: -1.5,
    power_drive:  1.4, power_bias:  -1.6, power_stages: 2, power_out: 0.6,
    xfmr_drive:   1.2, xfmr_sat:   0.50,
    sag_amount:   0.30, sag_recovery: 0.06,
    presence_db:  4,
    tone_curve:   { bass: 0.00, mid:  0.05, treble: 0.05 },
    cab_default:  '1x12',
  },
};

const DEFAULT_AMP_MODEL = 'british_clean';

// ── Cabinet IR profile table ────────────────────────────────────────────────
// Each profile parametrises a procedurally-generated IR (no .wav assets).
// IR shape is computed in `_buildCabIR` below. The character of each cab
// derives from:
//   length_ms     — total IR length
//   low_cut_hz    — 1-pole HP corner (cabinet resonance)
//   high_cut_hz   — 1-pole LP corner (cone breakup)
//   peak_hz       — biquad peaking presence bump frequency
//   peak_q        — Q of the presence bump
//   peak_gain_db  — gain of the presence bump
//   refl_count    — number of synthetic early reflections
//   on_axis_tilt  — adjustment to high_cut for on-axis (mic_position=0)
//                   vs. off-axis (mic_position=1). On-axis = brighter.
const CAB_PROFILES = {
  '1x12': {
    length_ms: 35, low_cut_hz: 90,  high_cut_hz: 4500,
    peak_hz: 2500, peak_q: 1.8, peak_gain_db: 4,
    refl_count: 2, on_axis_tilt: 1500,
  },
  '2x12': {
    length_ms: 40, low_cut_hz: 75,  high_cut_hz: 4200,
    peak_hz: 2300, peak_q: 1.6, peak_gain_db: 3.5,
    refl_count: 3, on_axis_tilt: 1300,
  },
  '4x10': {
    length_ms: 32, low_cut_hz: 100, high_cut_hz: 5500,
    peak_hz: 3000, peak_q: 2.2, peak_gain_db: 5,
    refl_count: 3, on_axis_tilt: 1800,
  },
  '4x12_open': {
    length_ms: 45, low_cut_hz: 80,  high_cut_hz: 4000,
    peak_hz: 2200, peak_q: 1.5, peak_gain_db: 3,
    refl_count: 4, on_axis_tilt: 1200,
  },
  '4x12_closed': {
    length_ms: 50, low_cut_hz: 70,  high_cut_hz: 3800,
    peak_hz: 2000, peak_q: 1.7, peak_gain_db: 4,
    refl_count: 5, on_axis_tilt: 1000,
  },
  vintage_1x12: {
    length_ms: 38, low_cut_hz: 95,  high_cut_hz: 4100,
    peak_hz: 2400, peak_q: 1.9, peak_gain_db: 4.5,
    refl_count: 2, on_axis_tilt: 1400,
  },
  '2x10_combo': {
    length_ms: 30, low_cut_hz: 110, high_cut_hz: 5200,
    peak_hz: 3200, peak_q: 2.0, peak_gain_db: 4,
    refl_count: 2, on_axis_tilt: 1600,
  },
};

const DEFAULT_CAB_MODEL = '4x12_open';

// Build a procedural cabinet IR. `mic_position` 0..1 morphs on-axis (brighter)
// → off-axis (darker) by shifting the HP/LP corners.
function _buildCabIR(ctx, profile, micPosition = 0) {
  const sr = ctx.sampleRate;
  const len = Math.max(64, Math.floor(sr * profile.length_ms / 1000));
  const buf = ctx.createBuffer(1, len, sr);
  const data = buf.getChannelData(0);

  const m = _clip01(micPosition);
  const high_cut = profile.high_cut_hz - profile.on_axis_tilt * (1 - m);
  const low_cut  = profile.low_cut_hz  + 30 * m;        // off-axis: more LF rolloff too

  // 1) shaped impulse + early reflections + decay envelope
  for (let i = 0; i < len; i++) {
    const t = i / sr;
    const env = Math.exp(-t * 90);
    let s = 0;
    if (i === 0) s += 1;
    for (let r = 1; r <= profile.refl_count; r++) {
      const idx = Math.floor(sr * (0.0006 * r + 0.0003 * Math.random()));
      if (i === idx) s += (r % 2 === 0 ? -0.3 : 0.4) / r;
    }
    s += (Math.random() * 2 - 1) * 0.18 * env;
    data[i] = s;
  }

  // 2) cheap 1-pole HP, 1-pole LP, biquad peaking presence
  const hpA = Math.exp(-2 * Math.PI * Math.max(20, low_cut) / sr);
  let prevIn = 0, prevOut = 0;
  for (let i = 0; i < len; i++) {
    const x = data[i];
    const y = hpA * (prevOut + x - prevIn);
    prevIn = x; prevOut = y;
    data[i] = y;
  }

  const lpA = Math.exp(-2 * Math.PI * Math.max(500, high_cut) / sr);
  let lpState = 0;
  for (let i = 0; i < len; i++) {
    lpState = lpA * lpState + (1 - lpA) * data[i];
    data[i] = lpState;
  }

  const wp = 2 * Math.PI * profile.peak_hz / sr;
  const Q = profile.peak_q;
  const A = Math.pow(10, profile.peak_gain_db / 40);
  const alpha = Math.sin(wp) / (2 * Q);
  const cosw = Math.cos(wp);
  const b0 = 1 + alpha * A;
  const b1 = -2 * cosw;
  const b2 = 1 - alpha * A;
  const a0 = 1 + alpha / A;
  const a1 = -2 * cosw;
  const a2 = 1 - alpha / A;
  let z1 = 0, z2 = 0, w1 = 0, w2 = 0;
  for (let i = 0; i < len; i++) {
    const x = data[i];
    const y = (b0 * x + b1 * z1 + b2 * z2 - a1 * w1 - a2 * w2) / a0;
    z2 = z1; z1 = x;
    w2 = w1; w1 = y;
    data[i] = data[i] * 0.6 + y * 0.4;
  }

  // 3) Normalise
  let peak = 0;
  for (let i = 0; i < len; i++) {
    const a = Math.abs(data[i]);
    if (a > peak) peak = a;
  }
  if (peak > 0) {
    const scale = 0.95 / peak;
    for (let i = 0; i < len; i++) data[i] *= scale;
  }
  return buf;
}

// Cheap fallback waveshaper curves (tanh-flavored) when worklets aren't loaded.
function _makeTanhCurve(amount, asymmetry = 0) {
  const N = 2048;
  const curve = new Float32Array(N);
  const a = Math.max(0.1, amount);
  const ab = Math.abs(asymmetry);
  const yOff = asymmetry / (1 + ab + 0.28 * asymmetry * asymmetry);
  for (let i = 0; i < N; i++) {
    const x = (i / (N - 1)) * 2 - 1;
    curve[i] = Math.tanh((x + asymmetry) * a) - Math.tanh(yOff * a);
  }
  return curve;
}

// ── The builder ────────────────────────────────────────────────────────────

export function buildAmpDesigner(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const input  = ctx.createGain();
  const output = ctx.createGain();

  // Internal gain stages
  const preGain  = ctx.createGain();   // user `gain` 0..10 → 0..3 mapped
  const postGain = ctx.createGain();   // user `output_level` 0..1
  preGain.gain.value  = 1;
  postGain.gain.value = 0.7;

  // ── Preamp (R2 wdf_tube_triode) ─────────────────────────────────────────
  const preamp = _safeWorklet(ctx, R2_TUBE_TRIODE, {
    numberOfInputs: 1, numberOfOutputs: 1,
    parameterData: { drive: 1.2, bias: -1.4, mix: 1.0 },
  });
  let preampFb = null;
  if (!preamp) {
    preampFb = ctx.createWaveShaper();
    preampFb.curve = _makeTanhCurve(2.2, -0.05);
    preampFb.oversample = '2x';
  }
  const preampStage = preamp || preampFb;

  // ── Tone stack (R2 wdf_tone_stack) — fallback: 3-biquad shelves ─────────
  const toneStack = _safeWorklet(ctx, R2_TONE_STACK, {
    numberOfInputs: 1, numberOfOutputs: 1,
    parameterData: { bass: 0.5, mid: 0.5, treble: 0.5, mix: 1.0 },
  });
  const bassFb = ctx.createBiquadFilter();
  bassFb.type = 'lowshelf'; bassFb.frequency.value = 110; bassFb.gain.value = 0;
  const midFb = ctx.createBiquadFilter();
  midFb.type = 'peaking'; midFb.frequency.value = 600; midFb.Q.value = 0.8; midFb.gain.value = 0;
  const trebFb = ctx.createBiquadFilter();
  trebFb.type = 'highshelf'; trebFb.frequency.value = 3500; trebFb.gain.value = 0;

  // ── Power amp (R2 wdf_tube_amp) ─────────────────────────────────────────
  const powerAmp = _safeWorklet(ctx, R2_TUBE_AMP, {
    numberOfInputs: 1, numberOfOutputs: 1,
    parameterData: { gain: 1.2, bias: -1.5, stages: 2, output_level: 0.6, mix: 1.0 },
  });
  let powerFb = null;
  if (!powerAmp) {
    powerFb = ctx.createWaveShaper();
    powerFb.curve = _makeTanhCurve(2.5, 0);
    powerFb.oversample = '4x';
  }
  const powerStage = powerAmp || powerFb;

  // ── Output transformer (R3 wdf_transformer) ─────────────────────────────
  const xfmr = _safeWorklet(ctx, R3_TRANSFORMER, {
    numberOfInputs: 1, numberOfOutputs: 1,
    parameterData: { drive: 1.0, saturation: 0.5, mix: 1.0 },
  });
  let xfmrFb = null;
  if (!xfmr) {
    xfmrFb = ctx.createWaveShaper();
    xfmrFb.curve = _makeTanhCurve(1.5, 0);
  }
  const xfmrStage = xfmr || xfmrFb;

  // ── Power-supply sag — implemented inline (no R3 worklet construction)
  // because R3's sag is a ScriptProcessorNode under the hood. Replicate it
  // here so this builder owns its full graph.
  const BUF = 256;
  const sagSp = ctx.createScriptProcessor(BUF, 2, 2);
  let sagEnv = 0;
  let sagAmt = 0.3;
  let sagRecovery = 0.05;
  let sagRelease = Math.exp(-1 / (sagRecovery * ctx.sampleRate));
  const sagAttack  = Math.exp(-1 / (0.005 * ctx.sampleRate));
  sagSp.onaudioprocess = (e) => {
    const inL  = e.inputBuffer.getChannelData(0);
    const inR  = e.inputBuffer.numberOfChannels > 1 ? e.inputBuffer.getChannelData(1) : inL;
    const outL = e.outputBuffer.getChannelData(0);
    const outR = e.outputBuffer.numberOfChannels > 1 ? e.outputBuffer.getChannelData(1) : null;
    const floor = 1 - sagAmt;
    for (let i = 0; i < BUF; i++) {
      const load = (Math.abs(inL[i]) + Math.abs(inR[i])) * 0.5;
      if (load > sagEnv) sagEnv = sagAttack  * sagEnv + (1 - sagAttack)  * load;
      else               sagEnv = sagRelease * sagEnv + (1 - sagRelease) * load;
      let gr = 1 - sagAmt * sagEnv;
      if (gr < floor) gr = floor;
      if (gr > 1)     gr = 1;
      outL[i] = inL[i] * gr;
      if (outR) outR[i] = inR[i] * gr;
    }
  };

  // ── Cab IR convolver ────────────────────────────────────────────────────
  const cab = ctx.createConvolver();
  let currentCabModel = (typeof params.cab_model === 'string' && CAB_PROFILES[params.cab_model])
    ? params.cab_model : DEFAULT_CAB_MODEL;
  let currentMicPos   = (typeof params.mic_position === 'number') ? _clip01(params.mic_position) : 0;
  const _refreshCabBuffer = () => {
    try {
      cab.buffer = _buildCabIR(ctx, CAB_PROFILES[currentCabModel] || CAB_PROFILES[DEFAULT_CAB_MODEL], currentMicPos);
    } catch (e) { /* ctx closed */ }
  };
  _refreshCabBuffer();

  // ── Presence (high shelf, post-cab) ─────────────────────────────────────
  const presence = ctx.createBiquadFilter();
  presence.type = 'highshelf';
  presence.frequency.value = 3000;
  presence.gain.value = 0;

  // ── Build chain ─────────────────────────────────────────────────────────
  input.connect(preGain);

  let head = preGain;
  if (preampStage) { head.connect(preampStage); head = preampStage; }

  if (toneStack) {
    head.connect(toneStack); head = toneStack;
  } else {
    head.connect(bassFb);
    bassFb.connect(midFb);
    midFb.connect(trebFb);
    head = trebFb;
  }

  if (powerStage)   { head.connect(powerStage);   head = powerStage; }
  if (xfmrStage)    { head.connect(xfmrStage);    head = xfmrStage; }
  head.connect(sagSp); head = sagSp;
  head.connect(cab);
  cab.connect(presence);
  presence.connect(postGain);
  postGain.connect(output);

  // ── Helpers for amp-model preset application ────────────────────────────
  let userBass = 0.5, userMid = 0.5, userTreble = 0.5;
  let activeTone = AMP_MODELS[DEFAULT_AMP_MODEL].tone_curve;

  const _applyToneStack = () => {
    const b = _clip01(userBass + (activeTone.bass   || 0));
    const m = _clip01(userMid  + (activeTone.mid    || 0));
    const t = _clip01(userTreble + (activeTone.treble || 0));
    const tsB = _workletParam(toneStack, 'bass');
    const tsM = _workletParam(toneStack, 'mid');
    const tsT = _workletParam(toneStack, 'treble');
    if (tsB) tsB.value = b; else bassFb.gain.value = (b - 0.5) * 24;
    if (tsM) tsM.value = m; else midFb.gain.value  = (m - 0.5) * 18;
    if (tsT) tsT.value = t; else trebFb.gain.value = (t - 0.5) * 24;
  };

  const _applyAmpModel = (modelName) => {
    const preset = AMP_MODELS[modelName] || AMP_MODELS[DEFAULT_AMP_MODEL];
    activeTone = preset.tone_curve;

    // Preamp
    const apPreDrive = _workletParam(preamp, 'drive');
    const apPreBias  = _workletParam(preamp, 'bias');
    if (apPreDrive) apPreDrive.value = preset.preamp_drive;
    if (apPreBias)  apPreBias.value  = preset.preamp_bias;
    if (preampFb) {
      preampFb.curve = _makeTanhCurve(1 + preset.preamp_drive, preset.preamp_bias * 0.05);
    }

    // Power amp
    const apPwrGain    = _workletParam(powerAmp, 'gain');
    const apPwrBias    = _workletParam(powerAmp, 'bias');
    const apPwrStages  = _workletParam(powerAmp, 'stages');
    const apPwrOut     = _workletParam(powerAmp, 'output_level');
    if (apPwrGain)   apPwrGain.value   = preset.power_drive;
    if (apPwrBias)   apPwrBias.value   = preset.power_bias;
    if (apPwrStages) apPwrStages.value = preset.power_stages;
    if (apPwrOut)    apPwrOut.value    = preset.power_out;
    if (powerFb) {
      powerFb.curve = _makeTanhCurve(1 + preset.power_drive, 0);
    }

    // Transformer
    const apXfmrDrive = _workletParam(xfmr, 'drive');
    const apXfmrSat   = _workletParam(xfmr, 'saturation');
    if (apXfmrDrive) apXfmrDrive.value = preset.xfmr_drive;
    if (apXfmrSat)   apXfmrSat.value   = preset.xfmr_sat;
    if (xfmrFb) {
      xfmrFb.curve = _makeTanhCurve(0.8 + preset.xfmr_sat, 0);
    }

    // Sag
    sagAmt = _clip01(preset.sag_amount);
    sagRecovery = _clip(preset.sag_recovery, 0.001, 2);
    sagRelease = Math.exp(-1 / (sagRecovery * ctx.sampleRate));

    // Presence baseline
    presence.gain.value = preset.presence_db * 0.5;  // user `presence` (0..1) sums on top in setter

    // Re-apply tone with new offsets
    _applyToneStack();
  };

  // Apply default amp model first so initial param wiring picks up sane values
  const initialModel = (typeof params.amp_model === 'string' && AMP_MODELS[params.amp_model])
    ? params.amp_model : DEFAULT_AMP_MODEL;
  _applyAmpModel(initialModel);

  // ── Param wiring ────────────────────────────────────────────────────────
  for (const [key, val] of Object.entries(params)) {
    const isModulated = typeof val === 'string' && val.startsWith('@');
    const paramId = isModulated ? val.slice(1) : null;

    switch (key) {
      case 'amp_model': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              // Accept string name OR numeric index 0..N-1 from a slider
              let name;
              if (typeof v === 'string') name = v;
              else if (typeof v === 'number') {
                const keys = Object.keys(AMP_MODELS);
                const idx = Math.max(0, Math.min(keys.length - 1, Math.round(v * (keys.length - 1))));
                name = keys[idx];
              }
              if (!AMP_MODELS[name]) name = DEFAULT_AMP_MODEL;
              _applyAmpModel(name);
            },
          };
        }
        // Non-modulated: already applied via initialModel
        break;
      }
      case 'gain': {
        // User `gain` 0..10 → preGain 0..3 (clean ~1.0)
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              const g = Math.max(0, v) * 0.3;
              preGain.gain.setTargetAtTime(g, ctx.currentTime, 0.01);
            },
          };
        } else if (typeof val === 'number') {
          preGain.gain.value = Math.max(0, val) * 0.3;
        }
        break;
      }
      case 'bass': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => { userBass = _clip01(v); _applyToneStack(); },
          };
        } else if (typeof val === 'number') {
          userBass = _clip01(val); _applyToneStack();
        }
        break;
      }
      case 'mid': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => { userMid = _clip01(v); _applyToneStack(); },
          };
        } else if (typeof val === 'number') {
          userMid = _clip01(val); _applyToneStack();
        }
        break;
      }
      case 'treble': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => { userTreble = _clip01(v); _applyToneStack(); },
          };
        } else if (typeof val === 'number') {
          userTreble = _clip01(val); _applyToneStack();
        }
        break;
      }
      case 'presence': {
        // user 0..1 → 0..+12 dB on the post-cab high shelf, summed with the
        // amp-model baseline (clamped to ±18 dB)
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              const baseline = (AMP_MODELS[initialModel] || AMP_MODELS[DEFAULT_AMP_MODEL]).presence_db * 0.5;
              presence.gain.value = _clip(baseline + _clip01(v) * 12, -18, 18);
            },
          };
        } else if (typeof val === 'number') {
          const baseline = (AMP_MODELS[initialModel] || AMP_MODELS[DEFAULT_AMP_MODEL]).presence_db * 0.5;
          presence.gain.value = _clip(baseline + _clip01(val) * 12, -18, 18);
        }
        break;
      }
      case 'master': {
        // master 0..10 → power-amp drive (R2 wdf_tube_amp `gain` 0..3)
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              const g = _clip(v * 0.3, 0, 3);
              const ap = _workletParam(powerAmp, 'gain');
              if (ap) ap.value = g;
              else if (powerFb) powerFb.curve = _makeTanhCurve(1 + g, 0);
            },
          };
        } else if (typeof val === 'number') {
          const g = _clip(val * 0.3, 0, 3);
          const ap = _workletParam(powerAmp, 'gain');
          if (ap) ap.value = g;
          else if (powerFb) powerFb.curve = _makeTanhCurve(1 + g, 0);
        }
        break;
      }
      case 'cab_model': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              let name;
              if (typeof v === 'string') name = v;
              else if (typeof v === 'number') {
                const keys = Object.keys(CAB_PROFILES);
                const idx = Math.max(0, Math.min(keys.length - 1, Math.round(v * (keys.length - 1))));
                name = keys[idx];
              }
              if (!CAB_PROFILES[name]) name = DEFAULT_CAB_MODEL;
              currentCabModel = name;
              _refreshCabBuffer();
            },
          };
        }
        // Non-modulated: handled via currentCabModel constructor + _refreshCabBuffer above
        break;
      }
      case 'mic_position': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              currentMicPos = _clip01(v);
              _refreshCabBuffer();
            },
          };
        } else if (typeof val === 'number') {
          currentMicPos = _clip01(val);
          _refreshCabBuffer();
        }
        break;
      }
      case 'output_level': {
        if (isModulated) {
          targets[paramId] = { audioParam: postGain.gain, paramDef: paramDefs[paramId] };
        } else if (typeof val === 'number') {
          postGain.gain.value = _clip(val, 0, 4);
        }
        break;
      }
      default: break;
    }
  }

  return { input, output, paramTargets: targets, scriptProcessor: sagSp };
}

// ── Default export: NODE_BUILDERS map ─────────────────────────────────────
const R13_AMP_DESIGNER_BUILDERS = {
  amp_designer: buildAmpDesigner,
};

// Side-channel exports for tests + downstream tools
export { AMP_MODELS, CAB_PROFILES, DEFAULT_AMP_MODEL, DEFAULT_CAB_MODEL };

export default R13_AMP_DESIGNER_BUILDERS;
