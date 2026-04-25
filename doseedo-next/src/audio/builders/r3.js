/**
 * Builders for the R3 WDF passive / saturation primitives.
 *
 * Each builder follows the WebAudioDSPEngine convention:
 *   buildXxx(ctx, nodeDef, paramDefs) → { input, output, paramTargets }
 *
 * paramTargets[paramId] is one of:
 *   { audioParam, paramDef, scale? }       // direct AudioParam binding
 *   { paramDef, customSetter }             // imperative setter (e.g. WaveShaper.curve)
 *
 * For efficiency, the RC/RLC filters are implemented with native
 * BiquadFilterNode and physical R, L, C → biquad coefficient mapping.
 * Saturation stages use WaveShaperNode with curves recomputed on parameter
 * change. Power-supply sag is the one that genuinely needs a worklet (it
 * needs an envelope follower with explicit attack / release coefficients);
 * we still implement a ScriptProcessor fallback so the engine works without
 * the worklet being registered.
 *
 * @author Doseedo R3
 */

// ── small helpers ───────────────────────────────────────────────────────────

function bindOrSet(params, key, paramDefs, targets, audioParam, opts = {}) {
  const val = params[key];
  if (val === undefined) return;
  if (typeof val === 'string' && val.startsWith('@')) {
    const paramId = val.slice(1);
    targets[paramId] = { audioParam, paramDef: paramDefs[paramId], ...opts };
  } else {
    audioParam.value = opts.scale ? opts.scale(val) : val;
  }
}

function bindCustom(params, key, paramDefs, targets, setter) {
  const val = params[key];
  if (val === undefined) return;
  if (typeof val === 'string' && val.startsWith('@')) {
    const paramId = val.slice(1);
    targets[paramId] = { paramDef: paramDefs[paramId], customSetter: setter };
  } else {
    setter(val);
  }
}

// ── 1) wdf_tape_sat ─────────────────────────────────────────────────────────
// Soft-knee saturator (WaveShaper) → low-shelf head bump → speed-mapped LP →
// wet/dry mix. Bias adds asymmetry by re-baking the curve.

function buildWdfTapeSat(ctx, node, paramDefs) {
  const input  = ctx.createGain();
  const output = ctx.createGain();
  const dry    = ctx.createGain();
  const wet    = ctx.createGain();

  const preGain   = ctx.createGain();        // input_level
  const shaper    = ctx.createWaveShaper();  // saturator (asymmetric, bias-aware)
  const headShelf = ctx.createBiquadFilter();
  const speedLP   = ctx.createBiquadFilter();
  const dcOut     = ctx.createBiquadFilter(); // ~5 Hz HP DC blocker

  shaper.oversample = 'none';
  // Higher-quality sat path uses 4× oversampling — but only when curve is
  // long enough; native WaveShaper handles oversampling internally.
  shaper.oversample = '4x';

  headShelf.type = 'lowshelf';
  headShelf.frequency.value = 110;
  headShelf.gain.value = 0;        // 0..+6 dB driven by `head_bump`

  speedLP.type = 'lowpass';
  speedLP.frequency.value = 12000; // updated by `speed`
  speedLP.Q.value = 0.7071;

  dcOut.type = 'highpass';
  dcOut.frequency.value = 5;
  dcOut.Q.value = 0.7071;

  dry.gain.value = 0;
  wet.gain.value = 1;
  preGain.gain.value = 1.5;

  // Build saturation curve. `bias` 0..1 → asymmetry offset -0.4..+0.4
  const N = 4096;
  let curveBias = 0;
  const updateCurve = () => {
    const curve = new Float32Array(N);
    const b = curveBias;
    // y(x) = (x+b)/(1+|x+b|+0.28*(x+b)^2) - bias_offset
    const ab = Math.abs(b);
    const yOff = b / (1 + ab + 0.28 * b * b);
    for (let i = 0; i < N; i++) {
      const x = (i / (N - 1)) * 2 - 1;
      const xb = x + b;
      const ax = Math.abs(xb);
      curve[i] = xb / (1 + ax + 0.28 * xb * xb) - yOff;
    }
    shaper.curve = curve;
  };
  updateCurve();

  // Routing
  input.connect(dry);
  input.connect(preGain);
  preGain.connect(shaper);
  shaper.connect(headShelf);
  headShelf.connect(speedLP);
  speedLP.connect(dcOut);
  dcOut.connect(wet);
  dry.connect(output);
  wet.connect(output);

  const targets = {};
  const params = node.params || {};

  bindOrSet(params, 'input_level', paramDefs, targets, preGain.gain);

  bindCustom(params, 'bias', paramDefs, targets, (v) => {
    curveBias = (v - 0.5) * 0.8;
    updateCurve();
  });

  // Speed: 0..1 → LP cutoff 2.5kHz..18kHz
  bindCustom(params, 'speed', paramDefs, targets, (v) => {
    const fc = 2500 + Math.max(0, Math.min(1, v)) * (18000 - 2500);
    speedLP.frequency.setTargetAtTime(fc, ctx.currentTime, 0.01);
  });

  // Head bump: 0..1 → 0..+8 dB low-shelf gain
  bindCustom(params, 'head_bump', paramDefs, targets, (v) => {
    headShelf.gain.setTargetAtTime(Math.max(0, Math.min(1, v)) * 8, ctx.currentTime, 0.01);
  });

  // Mix: 0..1 between dry and processed
  bindCustom(params, 'mix', paramDefs, targets, (v) => {
    const m = Math.max(0, Math.min(1, v));
    dry.gain.setTargetAtTime(1 - m, ctx.currentTime, 0.005);
    wet.gain.setTargetAtTime(m, ctx.currentTime, 0.005);
  });

  return { input, output, paramTargets: targets };
}

// ── 2) wdf_transformer ──────────────────────────────────────────────────────
// drive → atan-style WaveShaper (curve scaled by saturation) → DC blocker.

function buildWdfTransformer(ctx, node, paramDefs) {
  const input  = ctx.createGain();
  const output = ctx.createGain();
  const dry    = ctx.createGain();
  const wet    = ctx.createGain();

  const drive = ctx.createGain();
  const shaper = ctx.createWaveShaper();
  const dcOut = ctx.createBiquadFilter();

  shaper.oversample = '4x';
  dcOut.type = 'highpass';
  dcOut.frequency.value = 5;
  dcOut.Q.value = 0.7071;

  drive.gain.value = 1;
  dry.gain.value = 0;
  wet.gain.value = 1;

  // Shaper curve = (2/π) * atan(scale * x) blended with a fraction of the
  // dry signal (controlled by `saturation`).
  const N = 4096;
  let satAmount = 0.5;
  const updateCurve = () => {
    const curve = new Float32Array(N);
    const k = 2 / Math.PI;
    for (let i = 0; i < N; i++) {
      const x = (i / (N - 1)) * 2 - 1;
      const sat = k * Math.atan(x);
      curve[i] = satAmount * sat + (1 - satAmount) * x * 0.5;
    }
    shaper.curve = curve;
  };
  updateCurve();

  input.connect(dry);
  input.connect(drive);
  drive.connect(shaper);
  shaper.connect(dcOut);
  dcOut.connect(wet);
  dry.connect(output);
  wet.connect(output);

  const targets = {};
  const params = node.params || {};

  bindOrSet(params, 'drive', paramDefs, targets, drive.gain);

  bindCustom(params, 'saturation', paramDefs, targets, (v) => {
    satAmount = Math.max(0.05, Math.min(1, v));
    updateCurve();
  });

  bindCustom(params, 'mix', paramDefs, targets, (v) => {
    const m = Math.max(0, Math.min(1, v));
    dry.gain.setTargetAtTime(1 - m, ctx.currentTime, 0.005);
    wet.gain.setTargetAtTime(m, ctx.currentTime, 0.005);
  });

  return { input, output, paramTargets: targets };
}

// ── 3) wdf_rc_filter ────────────────────────────────────────────────────────
// First-order LP with cutoff = 1/(2π·R·C). Implemented as native lowpass
// BiquadFilter (one of its poles is effectively unused at Q=0.5; the audible
// result for a single-pole RC LP is what we want).

function buildWdfRCFilter(ctx, node, paramDefs) {
  const input  = ctx.createGain();
  const output = ctx.createGain();
  const dry    = ctx.createGain();
  const wet    = ctx.createGain();
  const filter = ctx.createBiquadFilter();

  filter.type = 'lowpass';
  filter.Q.value = 0.5;            // first-order-flavored response
  filter.frequency.value = 1591.55; // ≈ 1/(2π·10k·1e-8)

  dry.gain.value = 0;
  wet.gain.value = 1;

  input.connect(dry);
  input.connect(filter);
  filter.connect(wet);
  dry.connect(output);
  wet.connect(output);

  const params = node.params || {};
  const targets = {};

  // Cache R, C and recompute fc whenever either changes.
  let R = (typeof params.resistance  === 'number') ? params.resistance  : 10000;
  let C = (typeof params.capacitance === 'number') ? params.capacitance : 1e-8;

  const updateFc = () => {
    const fc = 1 / (2 * Math.PI * R * C);
    const fcSafe = Math.max(1, Math.min(fc, 0.49 * ctx.sampleRate));
    filter.frequency.setTargetAtTime(fcSafe, ctx.currentTime, 0.01);
  };
  updateFc();

  bindCustom(params, 'resistance', paramDefs, targets, (v) => {
    R = Math.max(1, v); updateFc();
  });
  bindCustom(params, 'capacitance', paramDefs, targets, (v) => {
    C = Math.max(1e-15, v); updateFc();
  });
  bindCustom(params, 'mix', paramDefs, targets, (v) => {
    const m = Math.max(0, Math.min(1, v));
    dry.gain.setTargetAtTime(1 - m, ctx.currentTime, 0.005);
    wet.gain.setTargetAtTime(m, ctx.currentTime, 0.005);
  });

  return { input, output, paramTargets: targets };
}

// ── 4) wdf_rlc_filter ───────────────────────────────────────────────────────
// Series-RLC band-pass. f0 = 1/(2π·√(LC)),  Q = (1/R)·√(L/C).
// Native BiquadFilter type 'bandpass' with computed frequency + Q.

function buildWdfRLCFilter(ctx, node, paramDefs) {
  const input  = ctx.createGain();
  const output = ctx.createGain();
  const dry    = ctx.createGain();
  const wet    = ctx.createGain();
  const filter = ctx.createBiquadFilter();

  filter.type = 'bandpass';
  filter.frequency.value = 5033;
  filter.Q.value = 1;

  dry.gain.value = 0;
  wet.gain.value = 1;

  input.connect(dry);
  input.connect(filter);
  filter.connect(wet);
  dry.connect(output);
  wet.connect(output);

  const params = node.params || {};
  const targets = {};

  let R = (typeof params.resistance  === 'number') ? params.resistance  : 1000;
  let L = (typeof params.inductance  === 'number') ? params.inductance  : 0.01;
  let C = (typeof params.capacitance === 'number') ? params.capacitance : 1e-7;

  const updateFilter = () => {
    const f0 = 1 / (2 * Math.PI * Math.sqrt(L * C));
    const f0Safe = Math.max(20, Math.min(f0, 0.45 * ctx.sampleRate));
    const Q = Math.max(0.1, Math.min((1 / R) * Math.sqrt(L / C), 100));
    filter.frequency.setTargetAtTime(f0Safe, ctx.currentTime, 0.01);
    filter.Q.setTargetAtTime(Q, ctx.currentTime, 0.01);
  };
  updateFilter();

  bindCustom(params, 'resistance', paramDefs, targets, (v) => {
    R = Math.max(0.01, v); updateFilter();
  });
  bindCustom(params, 'inductance', paramDefs, targets, (v) => {
    L = Math.max(1e-9, v); updateFilter();
  });
  bindCustom(params, 'capacitance', paramDefs, targets, (v) => {
    C = Math.max(1e-15, v); updateFilter();
  });
  bindCustom(params, 'mix', paramDefs, targets, (v) => {
    const m = Math.max(0, Math.min(1, v));
    dry.gain.setTargetAtTime(1 - m, ctx.currentTime, 0.005);
    wet.gain.setTargetAtTime(m, ctx.currentTime, 0.005);
  });

  return { input, output, paramTargets: targets };
}

// ── 5) wdf_power_supply_sag ─────────────────────────────────────────────────
// Envelope follower → 1-pole release LP → gain reduction. Implemented as a
// ScriptProcessorNode for portability (no worklet registration needed by the
// engine). The companion AudioWorklet processor exists for hosts that prefer
// to register it explicitly.

function buildWdfPowerSupplySag(ctx, node, paramDefs) {
  const input  = ctx.createGain();
  const output = ctx.createGain();
  const dry    = ctx.createGain();
  const wet    = ctx.createGain();
  const dcOut  = ctx.createBiquadFilter();
  dcOut.type = 'highpass';
  dcOut.frequency.value = 5;
  dcOut.Q.value = 0.7071;

  dry.gain.value = 0;
  wet.gain.value = 1;

  // ScriptProcessorNode is deprecated but widely available and fine for v1
  // until the AudioWorklet is wired in by the engine's loader.
  const BUF = 256;
  const sp = ctx.createScriptProcessor(BUF, 2, 2);

  // Sag state — shared across channels (single PSU)
  let env = 0;
  let sagAmt = 0.5;
  let recovery = 0.05;
  let releaseCoeff = Math.exp(-1 / (recovery * ctx.sampleRate));
  const attackCoeff = Math.exp(-1 / (0.005 * ctx.sampleRate));

  sp.onaudioprocess = (e) => {
    const inL = e.inputBuffer.getChannelData(0);
    const inR = e.inputBuffer.numberOfChannels > 1 ? e.inputBuffer.getChannelData(1) : inL;
    const outL = e.outputBuffer.getChannelData(0);
    const outR = e.outputBuffer.numberOfChannels > 1 ? e.outputBuffer.getChannelData(1) : null;

    const floor = 1 - sagAmt;

    for (let i = 0; i < BUF; i++) {
      const load = (Math.abs(inL[i]) + Math.abs(inR[i])) * 0.5;
      if (load > env) env = attackCoeff * env + (1 - attackCoeff) * load;
      else            env = releaseCoeff * env + (1 - releaseCoeff) * load;

      let gr = 1 - sagAmt * env;
      if (gr < floor) gr = floor;
      if (gr > 1)     gr = 1;

      outL[i] = inL[i] * gr;
      if (outR) outR[i] = inR[i] * gr;
    }
  };

  input.connect(dry);
  input.connect(sp);
  sp.connect(dcOut);
  dcOut.connect(wet);
  dry.connect(output);
  wet.connect(output);

  const params = node.params || {};
  const targets = {};

  bindCustom(params, 'sag', paramDefs, targets, (v) => {
    sagAmt = Math.max(0, Math.min(1, v));
  });
  bindCustom(params, 'recovery', paramDefs, targets, (v) => {
    recovery = Math.max(0.001, Math.min(2, v));
    releaseCoeff = Math.exp(-1 / (recovery * ctx.sampleRate));
  });
  bindCustom(params, 'mix', paramDefs, targets, (v) => {
    const m = Math.max(0, Math.min(1, v));
    dry.gain.setTargetAtTime(1 - m, ctx.currentTime, 0.005);
    wet.gain.setTargetAtTime(m, ctx.currentTime, 0.005);
  });

  return { input, output, paramTargets: targets, scriptProcessor: sp };
}

// ── Registry export ─────────────────────────────────────────────────────────

const R3_BUILDERS = {
  wdf_tape_sat:           buildWdfTapeSat,
  wdf_transformer:        buildWdfTransformer,
  wdf_rc_filter:          buildWdfRCFilter,
  wdf_rlc_filter:         buildWdfRLCFilter,
  wdf_power_supply_sag:   buildWdfPowerSupplySag,
};

export {
  buildWdfTapeSat,
  buildWdfTransformer,
  buildWdfRCFilter,
  buildWdfRLCFilter,
  buildWdfPowerSupplySag,
};

export default R3_BUILDERS;
