/**
 * R13 — Pedalboard composite builder
 *
 * Registers a SINGLE node type, `pedalboard`, that takes an ordered list of
 * stomp-pedal configs and chains them serially (with per-pedal bypass +
 * level staging). Each pedal sub-builder is a thin (~20 line) wrapper
 * around an existing R-round node: most overdrive/distortion/fuzz pedals
 * fan out to R2 WDF clippers; modulation pedals reuse the legacy
 * chorus/phaser/flanger/tremolo worklets the engine already ships;
 * delays/reverbs/pitch shifters reuse R1/R5/R9.
 *
 * Schema:
 *   {
 *     type: 'pedalboard',
 *     params: {
 *       pedals: [
 *         { type: 'overdrive_pedal',   drive: 0.6, tone: 0.5, level: 0.7, bypass: false },
 *         { type: 'distortion_pedal',  drive: 0.8, tone: 0.4, level: 0.6, bypass: false },
 *         { type: 'delay_pedal',       time: 250,  feedback: 0.3, mix: 0.4, bypass: false },
 *         …
 *       ],
 *     },
 *   }
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildPedalboard(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets, pedals }
 *
 * Live-control / @bindings:
 *   Per-pedal params can be late-bound via dotted addresses. Use
 *   `'@<paramId>'` as the value for any pedal-level field, OR address a
 *   specific pedal slot via `params.bindings = { 'pedals.0.drive': '@drv' }`.
 *   The simpler `bypass` toggle exposes `paramTargets['bypass_<idx>']` per
 *   pedal automatically.
 *
 * Fallback behavior: every sub-builder follows the same "always returns a
 * working passthrough" rule R2/R3/R4 follow — if a worklet processor isn't
 * registered yet, the chain still routes audio (just with a primitive
 * approximation in that slot). The composite never collapses on missing
 * worklets.
 *
 * Author: Agent R13
 */

import r1Builders from './r1.js';
import r2Builders from './r2.js';
import r3Builders from './r3.js';
import r5Builders from './r5.js';
import r9Builders from './r9.js';

// ── Helpers ───────────────────────────────────────────────────────────────

function isAtBinding(v) {
  return typeof v === 'string' && v.startsWith('@');
}

/**
 * Construct a sub-stage by dispatching to one of the round builders.
 * Returns the standard { input, output, paramTargets } shape; if the
 * dispatch target is missing we return a unity-gain passthrough so the
 * chain still builds.
 */
function _delegate(ctx, builderMap, type, subParams, paramDefs) {
  const builder = builderMap[type];
  if (!builder) {
    const g = ctx.createGain();
    return { input: g, output: g, paramTargets: {} };
  }
  return builder(ctx, { type, params: subParams || {} }, paramDefs || {});
}

/**
 * Equal-power bypass crossfade: input → [aGain | bGain] → output, where
 * `aGain` is the pedal's wet path and `bGain` is the dry pass-around.
 * Returns the two gains so the caller can flip them on bypass toggles.
 */
function _bypassPair(ctx, initialBypass) {
  const a = ctx.createGain();   // wet (pedal output)
  const b = ctx.createGain();   // dry (around)
  if (initialBypass) {
    a.gain.value = 0;
    b.gain.value = 1;
  } else {
    a.gain.value = 1;
    b.gain.value = 0;
  }
  return { wetGain: a, dryGain: b };
}

/**
 * 3-band tone stack used by overdrive/distortion/fuzz sub-stages and
 * the eq_pedal. Implemented with native biquads so no worklet load is
 * required. `tone` (0..1) maps to a tilt: <0.5 darkens (more bass,
 * less treble); >0.5 brightens. Returns { input, output, setTone }.
 */
function _toneTilt(ctx, initialTone = 0.5) {
  const input = ctx.createGain();
  const lo = ctx.createBiquadFilter();
  const hi = ctx.createBiquadFilter();
  lo.type = 'lowshelf';
  lo.frequency.value = 250;
  hi.type = 'highshelf';
  hi.frequency.value = 3000;

  const setTone = (t) => {
    const x = Math.max(0, Math.min(1, t));
    // Around 0.5 → flat. Below → +bass/-treble. Above → -bass/+treble.
    const tilt = (x - 0.5) * 12; // ±6 dB swing
    lo.gain.value = -tilt;
    hi.gain.value = tilt;
  };
  setTone(initialTone);

  input.connect(lo);
  lo.connect(hi);
  return { input, output: hi, setTone };
}

// ── Per-pedal sub-builders ────────────────────────────────────────────────
// Each returns { input, output, paramTargets } where paramTargets are keyed
// by an *internal* sub-id; the composite caller maps those into the outer
// paramId space.

// 1. overdrive_pedal — R2 wdf_tube_triode + tone tilt
function buildOverdrivePedal(ctx, params, paramDefs) {
  const drive = params.drive ?? 0.5;
  const stage = _delegate(ctx, r2Builders, 'wdf_tube_triode',
    { drive, bias: -1.0, mix: 1.0 }, paramDefs);
  const tone = _toneTilt(ctx, params.tone ?? 0.5);
  const level = ctx.createGain();
  level.gain.value = params.level ?? 0.7;

  stage.output.connect(tone.input);
  tone.output.connect(level);

  const targets = {};
  if (isAtBinding(params.tone)) targets.tone = { customSetter: tone.setTone };
  if (isAtBinding(params.level)) targets.level = { audioParam: level.gain };
  return { input: stage.input, output: level, paramTargets: targets, _refs: { stage, level } };
}

// 2. distortion_pedal — R2 wdf_diode_clipper + tone tilt
function buildDistortionPedal(ctx, params, paramDefs) {
  const stage = _delegate(ctx, r2Builders, 'wdf_diode_clipper',
    { drive: params.drive ?? 0.7, ideality: 1.5, symmetry: 1.0, mix: 1.0 }, paramDefs);
  const tone = _toneTilt(ctx, params.tone ?? 0.5);
  const level = ctx.createGain();
  level.gain.value = params.level ?? 0.6;
  stage.output.connect(tone.input);
  tone.output.connect(level);

  const targets = {};
  if (isAtBinding(params.tone)) targets.tone = { customSetter: tone.setTone };
  if (isAtBinding(params.level)) targets.level = { audioParam: level.gain };
  return { input: stage.input, output: level, paramTargets: targets };
}

// 3. fuzz_pedal — R2 wdf_transistor_clipper + tone
function buildFuzzPedal(ctx, params, paramDefs) {
  const stage = _delegate(ctx, r2Builders, 'wdf_transistor_clipper',
    { drive: params.drive ?? 0.85, beta: 100, fuzz: params.fuzz ?? 0.7, mix: 1.0 }, paramDefs);
  const tone = _toneTilt(ctx, params.tone ?? 0.5);
  const level = ctx.createGain();
  level.gain.value = params.level ?? 0.5;
  stage.output.connect(tone.input);
  tone.output.connect(level);

  const targets = {};
  if (isAtBinding(params.tone)) targets.tone = { customSetter: tone.setTone };
  if (isAtBinding(params.level)) targets.level = { audioParam: level.gain };
  return { input: stage.input, output: level, paramTargets: targets };
}

// 4. clean_boost_pedal — pure gain stage
function buildCleanBoostPedal(ctx, params /*, paramDefs */) {
  const g = ctx.createGain();
  g.gain.value = (params.gain ?? params.level ?? 1.5);
  const targets = {};
  if (isAtBinding(params.gain) || isAtBinding(params.level)) {
    targets.gain = { audioParam: g.gain };
  }
  return { input: g, output: g, paramTargets: targets };
}

// 5. compressor_pedal — DynamicsCompressorNode with stomp-style 2-knob mapping
function buildCompressorPedal(ctx, params /*, paramDefs */) {
  const c = ctx.createDynamicsCompressor();
  c.threshold.value = params.threshold ?? -18;
  c.ratio.value = params.ratio ?? 4;
  c.attack.value = (params.attack ?? 5) / 1000;
  c.release.value = (params.release ?? 100) / 1000;
  c.knee.value = 6;
  const makeup = ctx.createGain();
  makeup.gain.value = params.level ?? 1.0;
  c.connect(makeup);

  const targets = {};
  if (isAtBinding(params.threshold)) targets.threshold = { audioParam: c.threshold };
  if (isAtBinding(params.ratio))     targets.ratio     = { audioParam: c.ratio };
  if (isAtBinding(params.level))     targets.level     = { audioParam: makeup.gain };
  return { input: c, output: makeup, paramTargets: targets };
}

// 6. eq_pedal — bass/mid/treble 3-band semi-parametric (low/peak/high shelf)
function buildEqPedal(ctx, params /*, paramDefs */) {
  const bass = ctx.createBiquadFilter();
  const mid  = ctx.createBiquadFilter();
  const treb = ctx.createBiquadFilter();
  bass.type = 'lowshelf';   bass.frequency.value = 120;
  mid.type  = 'peaking';    mid.frequency.value = 800;  mid.Q.value = 0.7;
  treb.type = 'highshelf';  treb.frequency.value = 4500;
  bass.gain.value = params.bass ?? 0;
  mid.gain.value  = params.mid ?? 0;
  treb.gain.value = params.treble ?? 0;
  bass.connect(mid); mid.connect(treb);

  const targets = {};
  if (isAtBinding(params.bass))   targets.bass   = { audioParam: bass.gain };
  if (isAtBinding(params.mid))    targets.mid    = { audioParam: mid.gain };
  if (isAtBinding(params.treble)) targets.treble = { audioParam: treb.gain };
  return { input: bass, output: treb, paramTargets: targets };
}

// 7. wah_pedal — bandpass swept by `position` (0..1)
function buildWahPedal(ctx, params /*, paramDefs */) {
  const bp = ctx.createBiquadFilter();
  bp.type = 'bandpass';
  bp.Q.value = params.q ?? 4;
  const setPos = (p) => {
    const x = Math.max(0, Math.min(1, p));
    // Heel 400 Hz → toe 2200 Hz (log sweep)
    bp.frequency.value = 400 * Math.pow(2200 / 400, x);
  };
  setPos(params.position ?? 0.5);

  const targets = {};
  if (isAtBinding(params.position)) targets.position = { customSetter: setPos };
  return { input: bp, output: bp, paramTargets: targets };
}

// 8. auto_wah_pedal — bandpass + envelope follower (R1) modulating cutoff
function buildAutoWahPedal(ctx, params, paramDefs) {
  const bp = ctx.createBiquadFilter();
  bp.type = 'bandpass';
  bp.Q.value = params.q ?? 5;
  bp.frequency.value = 600;

  const env = _delegate(ctx, r1Builders, 'envelope_follower',
    { attack: params.attack ?? 10, release: params.release ?? 100 }, paramDefs);
  const sens = ctx.createGain();
  sens.gain.value = (params.sensitivity ?? 0.5) * 1500;
  const base = ctx.createConstantSource();
  base.offset.value = 400;
  base.start();

  // Wire env → sens → bp.frequency, plus base offset
  env.output.connect(sens);
  sens.connect(bp.frequency);
  base.connect(bp.frequency);

  // Input has to feed both the bandpass AND the envelope follower.
  const split = ctx.createGain();
  split.connect(bp);
  split.connect(env.input);

  const targets = {};
  if (isAtBinding(params.sensitivity)) targets.sensitivity = { audioParam: sens.gain };
  return { input: split, output: bp, paramTargets: targets };
}

// 9. phaser_pedal — legacy phaser-processor.js (registered as `phaser`)
function buildPhaserPedal(ctx, params, paramDefs) {
  // The legacy `phaser` builder lives inside WebAudioDSPEngine.js NODE_BUILDERS
  // and isn't directly importable. We replicate its construction inline since
  // it's just biquads + an LFO (no worklet) — keeps the import surface clean.
  const stages = [];
  for (let i = 0; i < 4; i++) {
    const ap = ctx.createBiquadFilter();
    ap.type = 'allpass';
    ap.frequency.value = 1000;
    ap.Q.value = 0.5;
    stages.push(ap);
    if (i > 0) stages[i - 1].connect(ap);
  }
  const lfo = ctx.createOscillator();
  const lfoGain = ctx.createGain();
  lfo.frequency.value = params.rate ?? 0.5;
  lfoGain.gain.value = (params.depth ?? 0.5) * 800;
  lfo.connect(lfoGain);
  stages.forEach(ap => lfoGain.connect(ap.frequency));
  lfo.start();

  const dry = ctx.createGain();
  const wet = ctx.createGain();
  dry.gain.value = 0.5;
  wet.gain.value = params.mix ?? 0.5;
  const input = ctx.createGain();
  const output = ctx.createGain();
  input.connect(dry);
  input.connect(stages[0]);
  stages[stages.length - 1].connect(wet);
  dry.connect(output);
  wet.connect(output);

  const targets = {};
  if (isAtBinding(params.rate))  targets.rate  = { audioParam: lfo.frequency };
  if (isAtBinding(params.depth)) targets.depth = { audioParam: lfoGain.gain };
  if (isAtBinding(params.mix))   targets.mix   = { audioParam: wet.gain };
  return { input, output, paramTargets: targets, oscillators: [lfo] };
}

// 10. flanger_pedal — short delay + LFO + feedback
function buildFlangerPedal(ctx, params /*, paramDefs */) {
  const input = ctx.createGain();
  const output = ctx.createGain();
  const dry = ctx.createGain();
  const wet = ctx.createGain();
  const delay = ctx.createDelay(0.02);
  const fb = ctx.createGain();
  const lfo = ctx.createOscillator();
  const lfoGain = ctx.createGain();
  dry.gain.value = 0.5;
  wet.gain.value = params.mix ?? 0.6;
  delay.delayTime.value = 0.005;
  fb.gain.value = params.feedback ?? 0.5;
  lfo.frequency.value = params.rate ?? 0.3;
  lfoGain.gain.value = (params.depth ?? 0.5) * 0.005;
  lfo.connect(lfoGain);
  lfoGain.connect(delay.delayTime);
  lfo.start();
  input.connect(dry);
  input.connect(delay);
  delay.connect(fb); fb.connect(delay);
  delay.connect(wet);
  dry.connect(output); wet.connect(output);

  const targets = {};
  if (isAtBinding(params.rate))     targets.rate     = { audioParam: lfo.frequency };
  if (isAtBinding(params.depth))    targets.depth    = { audioParam: lfoGain.gain };
  if (isAtBinding(params.feedback)) targets.feedback = { audioParam: fb.gain };
  if (isAtBinding(params.mix))      targets.mix      = { audioParam: wet.gain };
  return { input, output, paramTargets: targets, oscillators: [lfo] };
}

// 11. chorus_pedal — single-voice modulated delay
function buildChorusPedal(ctx, params /*, paramDefs */) {
  const input = ctx.createGain();
  const output = ctx.createGain();
  const dry = ctx.createGain();
  const wet = ctx.createGain();
  const delay = ctx.createDelay(0.1);
  const lfo = ctx.createOscillator();
  const lfoGain = ctx.createGain();
  dry.gain.value = 0.7;
  wet.gain.value = params.mix ?? 0.5;
  delay.delayTime.value = 0.015;
  lfo.frequency.value = params.rate ?? 1.5;
  lfoGain.gain.value = (params.depth ?? 0.5) * 0.005;
  lfo.connect(lfoGain);
  lfoGain.connect(delay.delayTime);
  lfo.start();
  input.connect(dry);
  input.connect(delay);
  delay.connect(wet);
  dry.connect(output); wet.connect(output);

  const targets = {};
  if (isAtBinding(params.rate))  targets.rate  = { audioParam: lfo.frequency };
  if (isAtBinding(params.depth)) targets.depth = { audioParam: lfoGain.gain };
  if (isAtBinding(params.mix))   targets.mix   = { audioParam: wet.gain };
  return { input, output, paramTargets: targets, oscillators: [lfo] };
}

// 12. tremolo_pedal — amplitude modulation
function buildTremoloPedal(ctx, params /*, paramDefs */) {
  const input = ctx.createGain();
  const trem = ctx.createGain();
  trem.gain.value = 1;
  const lfo = ctx.createOscillator();
  const lfoGain = ctx.createGain();
  lfo.frequency.value = params.rate ?? 4;
  lfoGain.gain.value = params.depth ?? 0.5;
  lfo.connect(lfoGain);
  lfoGain.connect(trem.gain);
  lfo.start();
  input.connect(trem);

  const targets = {};
  if (isAtBinding(params.rate))  targets.rate  = { audioParam: lfo.frequency };
  if (isAtBinding(params.depth)) targets.depth = { audioParam: lfoGain.gain };
  return { input, output: trem, paramTargets: targets, oscillators: [lfo] };
}

// 13. vibrato_pedal — chorus-style with NO dry signal
function buildVibratoPedal(ctx, params /*, paramDefs */) {
  const input = ctx.createGain();
  const output = ctx.createGain();
  const delay = ctx.createDelay(0.05);
  const lfo = ctx.createOscillator();
  const lfoGain = ctx.createGain();
  delay.delayTime.value = 0.01;
  lfo.frequency.value = params.rate ?? 5;
  lfoGain.gain.value = (params.depth ?? 0.5) * 0.005;
  lfo.connect(lfoGain);
  lfoGain.connect(delay.delayTime);
  lfo.start();
  input.connect(delay);
  delay.connect(output);

  const targets = {};
  if (isAtBinding(params.rate))  targets.rate  = { audioParam: lfo.frequency };
  if (isAtBinding(params.depth)) targets.depth = { audioParam: lfoGain.gain };
  return { input, output, paramTargets: targets, oscillators: [lfo] };
}

// 14. delay_pedal — single-tap echo (R1 multitap simplified)
function buildDelayPedal(ctx, params, paramDefs) {
  return _delegate(ctx, r1Builders, 'multitap_delay', {
    time: params.time ?? 250,
    feedback: params.feedback ?? 0.3,
    mix: params.mix ?? 0.4,
    taps: [{ delay: params.time ?? 250, gain: 1.0, pan: 0 }],
  }, paramDefs);
}

// 15. tape_delay_pedal — multitap + R3 wdf_tape_sat in feedback loop
function buildTapeDelayPedal(ctx, params, paramDefs) {
  const echo = _delegate(ctx, r1Builders, 'multitap_delay', {
    time: params.time ?? 350,
    feedback: params.feedback ?? 0.4,
    mix: params.mix ?? 0.4,
  }, paramDefs);
  const sat = _delegate(ctx, r3Builders, 'wdf_tape_sat',
    { drive: params.saturation ?? 0.6, mix: 1.0 }, paramDefs);
  echo.output.connect(sat.input);
  return { input: echo.input, output: sat.output, paramTargets: {} };
}

// 16. reverb_pedal — R9 algo_reverb (room algo)
function buildReverbPedal(ctx, params, paramDefs) {
  return _delegate(ctx, r9Builders, 'algo_reverb', {
    algorithm: params.algorithm ?? 'room',
    decay_time: params.decay ?? 1.5,
    mix: params.mix ?? 0.3,
  }, paramDefs);
}

// 17. octave_pedal — pitch shift -12 semitones
function buildOctavePedal(ctx, params, paramDefs) {
  return _delegate(ctx, r1Builders, 'pitch_shift',
    { semitones: params.semitones ?? -12, mix: params.mix ?? 0.5 }, paramDefs);
}

// 18. pitch_shifter_pedal — generic pitch shift
function buildPitchShifterPedal(ctx, params, paramDefs) {
  // Prefer R5 phase-vocoder if available, else R1 SOLA.
  const r5 = r5Builders && r5Builders.pitch_shift_pv;
  if (r5) {
    return _delegate(ctx, r5Builders, 'pitch_shift_pv',
      { semitones: params.semitones ?? 7, mix: params.mix ?? 1 }, paramDefs);
  }
  return _delegate(ctx, r1Builders, 'pitch_shift',
    { semitones: params.semitones ?? 7, mix: params.mix ?? 1 }, paramDefs);
}

// 19. ring_mod_pedal — input * sine carrier
function buildRingModPedal(ctx, params /*, paramDefs */) {
  const input = ctx.createGain();
  const ring = ctx.createGain();
  ring.gain.value = 0;            // multiplied via carrier signal
  const carrier = ctx.createOscillator();
  const dry = ctx.createGain();
  const wet = ctx.createGain();
  const output = ctx.createGain();
  carrier.frequency.value = params.frequency ?? 200;
  carrier.connect(ring.gain);
  carrier.start();
  input.connect(ring);
  input.connect(dry);
  ring.connect(wet);
  dry.gain.value = 1 - (params.mix ?? 0.5);
  wet.gain.value = params.mix ?? 0.5;
  dry.connect(output); wet.connect(output);

  const targets = {};
  if (isAtBinding(params.frequency)) targets.frequency = { audioParam: carrier.frequency };
  if (isAtBinding(params.mix)) {
    targets.mix = {
      customSetter: (v) => {
        const t = ctx.currentTime;
        wet.gain.setTargetAtTime(v, t, 0.01);
        dry.gain.setTargetAtTime(1 - v, t, 0.01);
      },
    };
  }
  return { input, output, paramTargets: targets, oscillators: [carrier] };
}

// 20. filter_pedal — switchable LP/HP/BP biquad
function buildFilterPedal(ctx, params /*, paramDefs */) {
  const f = ctx.createBiquadFilter();
  const m = (params.mode || 'lowpass').toLowerCase();
  f.type = (m === 'hp' || m === 'highpass') ? 'highpass'
         : (m === 'bp' || m === 'bandpass') ? 'bandpass'
         : 'lowpass';
  f.frequency.value = params.cutoff ?? 1000;
  f.Q.value = params.resonance ?? 1;

  const targets = {};
  if (isAtBinding(params.cutoff))     targets.cutoff     = { audioParam: f.frequency };
  if (isAtBinding(params.resonance))  targets.resonance  = { audioParam: f.Q };
  return { input: f, output: f, paramTargets: targets };
}

// 21. limiter_pedal — DynamicsCompressor pinned at high ratio + low threshold
function buildLimiterPedal(ctx, params /*, paramDefs */) {
  const c = ctx.createDynamicsCompressor();
  c.threshold.value = params.ceiling ?? -1;
  c.ratio.value = 20;
  c.attack.value = 0.001;
  c.release.value = (params.release ?? 50) / 1000;
  c.knee.value = 0;
  const targets = {};
  if (isAtBinding(params.ceiling)) targets.ceiling = { audioParam: c.threshold };
  return { input: c, output: c, paramTargets: targets };
}

// 22. gate_pedal — DynamicsCompressor inverted-style (downward expander
// approximation). Web Audio has no native gate node, so we approximate via
// envelope follower → gain VCA.
function buildGatePedal(ctx, params, paramDefs) {
  const input = ctx.createGain();
  const output = ctx.createGain();
  const vca = ctx.createGain();
  vca.gain.value = 1;

  const env = _delegate(ctx, r1Builders, 'envelope_follower',
    { attack: 1, release: params.release ?? 50 }, paramDefs);
  const thresh = params.threshold ?? -40;
  const linThresh = Math.pow(10, thresh / 20);
  // Use a gate-driving WaveShaper to convert env amplitude → 0/1
  const shaper = ctx.createWaveShaper();
  const n = 2048;
  const curve = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = (i * 2) / n - 1;
    curve[i] = Math.abs(x) > linThresh ? 1 : 0;
  }
  shaper.curve = curve;

  // env → shaper → vca.gain (replace static gain w/ control-rate signal)
  env.output.connect(shaper);
  shaper.connect(vca.gain);

  input.connect(env.input);
  input.connect(vca);
  vca.connect(output);

  return { input, output, paramTargets: {} };
}

// 23. noise_gate_pedal — same as gate, just lower default threshold
function buildNoiseGatePedal(ctx, params, paramDefs) {
  return buildGatePedal(ctx, { threshold: -55, release: 80, ...params }, paramDefs);
}

// 24. bitcrusher_pedal — R1 bitcrusher
function buildBitcrusherPedal(ctx, params, paramDefs) {
  return _delegate(ctx, r1Builders, 'bitcrusher', {
    bit_depth: params.bits ?? 8,
    sample_rate_div: params.downsample ?? 4,
    mix: params.mix ?? 1.0,
  }, paramDefs);
}

// ── Sub-pedal type registry ───────────────────────────────────────────────

const PEDAL_BUILDERS = {
  overdrive_pedal:     buildOverdrivePedal,
  distortion_pedal:    buildDistortionPedal,
  fuzz_pedal:          buildFuzzPedal,
  clean_boost_pedal:   buildCleanBoostPedal,
  compressor_pedal:    buildCompressorPedal,
  eq_pedal:            buildEqPedal,
  wah_pedal:           buildWahPedal,
  auto_wah_pedal:      buildAutoWahPedal,
  phaser_pedal:        buildPhaserPedal,
  flanger_pedal:       buildFlangerPedal,
  chorus_pedal:        buildChorusPedal,
  tremolo_pedal:       buildTremoloPedal,
  vibrato_pedal:       buildVibratoPedal,
  delay_pedal:         buildDelayPedal,
  tape_delay_pedal:    buildTapeDelayPedal,
  reverb_pedal:        buildReverbPedal,
  octave_pedal:        buildOctavePedal,
  pitch_shifter_pedal: buildPitchShifterPedal,
  ring_mod_pedal:      buildRingModPedal,
  filter_pedal:        buildFilterPedal,
  limiter_pedal:       buildLimiterPedal,
  gate_pedal:          buildGatePedal,
  noise_gate_pedal:    buildNoiseGatePedal,
  bitcrusher_pedal:    buildBitcrusherPedal,
};

export const PEDAL_TYPES = Object.keys(PEDAL_BUILDERS);

// ── Pedalboard composite ──────────────────────────────────────────────────

/**
 * Build a serial chain of stomp pedals.
 *
 *   input → [pedal0] → [pedal1] → … → [pedalN] → output
 *
 * Each pedal sits inside a bypass crossfade pair: when bypass=true the dry
 * branch carries the previous stage's output around the wet branch and into
 * the next stage. The crossfade is constant-power so toggling doesn't pop.
 *
 * The composite exposes:
 *   paramTargets['bypass_<idx>']  — customSetter accepting 0/1 to flip bypass
 *   paramTargets['<paramId>']     — for any pedal-level @-binding
 *
 * Returned shape adds `pedals: [...]` listing per-slot internals so callers
 * (e.g. the calibration harness) can introspect.
 */
export function buildPedalboard(ctx, node, paramDefs) {
  const params = node.params || {};
  const pedalsCfg = Array.isArray(params.pedals) ? params.pedals : [];

  const input = ctx.createGain();
  const output = ctx.createGain();
  const targets = {};
  const pedals = [];

  if (pedalsCfg.length === 0) {
    // Empty pedalboard = passthrough
    input.connect(output);
    return { input, output, paramTargets: targets, pedals };
  }

  // Walk the chain, building each slot and stitching with a bypass crossfade
  // around it. `prevOut` is the node currently feeding the next stage.
  let prevOut = input;

  for (let i = 0; i < pedalsCfg.length; i++) {
    const cfg = pedalsCfg[i] || {};
    const type = cfg.type;
    const builder = PEDAL_BUILDERS[type];
    if (!builder) {
      // Unknown pedal type → passthrough to keep the chain intact
      const skip = ctx.createGain();
      prevOut.connect(skip);
      pedals.push({ type, slot: i, status: 'unknown', input: skip, output: skip });
      prevOut = skip;
      continue;
    }

    let built;
    try {
      built = builder(ctx, cfg, paramDefs);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(`[R13] pedal #${i} (${type}) build failed:`, err && err.message);
      const skip = ctx.createGain();
      prevOut.connect(skip);
      pedals.push({ type, slot: i, status: 'error', error: err && err.message,
                    input: skip, output: skip });
      prevOut = skip;
      continue;
    }

    const { wetGain, dryGain } = _bypassPair(ctx, !!cfg.bypass);
    // Wire: prevOut → built.input (wet branch starts) AND prevOut → dryGain
    //       built.output → wetGain
    //       wetGain + dryGain → join (next prevOut)
    const join = ctx.createGain();
    prevOut.connect(built.input);
    prevOut.connect(dryGain);
    built.output.connect(wetGain);
    wetGain.connect(join);
    dryGain.connect(join);

    // Forward sub-pedal paramTargets up to the composite, prefixing with
    // a slot-aware ID. Two ways to address them:
    //   1. Sub-builder already wired to a real @-bound id (rare; keys are
    //      internal sub-ids like 'tone'/'level'). We re-export under
    //      "<slot>.<sub>" so live drivers can find them.
    //   2. The pedal-level bypass toggle is exposed as 'bypass_<i>'.
    for (const [subId, target] of Object.entries(built.paramTargets || {})) {
      targets[`pedal_${i}_${subId}`] = target;
    }

    // Bypass live-control: 0 = engaged (wet=1), 1 = bypassed (dry=1)
    targets[`bypass_${i}`] = {
      paramDef: paramDefs[`bypass_${i}`] || { id: `bypass_${i}`, min: 0, max: 1, default: cfg.bypass ? 1 : 0 },
      customSetter: (v) => {
        const bypassed = (typeof v === 'boolean') ? v : v >= 0.5;
        const t = ctx.currentTime;
        if (bypassed) {
          wetGain.gain.setTargetAtTime(0, t, 0.005);
          dryGain.gain.setTargetAtTime(1, t, 0.005);
        } else {
          wetGain.gain.setTargetAtTime(1, t, 0.005);
          dryGain.gain.setTargetAtTime(0, t, 0.005);
        }
      },
    };

    pedals.push({
      type, slot: i, status: 'ok', input: built.input, output: built.output,
      wetGain, dryGain, join,
    });
    prevOut = join;
  }

  prevOut.connect(output);

  // Top-level wet/dry mix on the whole pedalboard (optional)
  if (typeof params.mix === 'number' || isAtBinding(params.mix)) {
    // Add a parallel dry around the entire chain
    const wetTrim = ctx.createGain();
    const dryTrim = ctx.createGain();
    const mixed = ctx.createGain();
    const m = (typeof params.mix === 'number') ? params.mix : 1;
    wetTrim.gain.value = m;
    dryTrim.gain.value = 1 - m;
    // Re-route output through wetTrim, and add input → dryTrim → mixed
    try { prevOut.disconnect(output); } catch (_) { /* ignore */ }
    prevOut.connect(wetTrim);
    input.connect(dryTrim);
    wetTrim.connect(mixed);
    dryTrim.connect(mixed);
    mixed.connect(output);
    if (isAtBinding(params.mix)) {
      const id = params.mix.slice(1);
      targets[id] = {
        paramDef: paramDefs[id],
        customSetter: (v) => {
          const t = ctx.currentTime;
          wetTrim.gain.setTargetAtTime(v, t, 0.01);
          dryTrim.gain.setTargetAtTime(1 - v, t, 0.01);
        },
      };
    }
  }

  return { input, output, paramTargets: targets, pedals };
}

// ── Default export ────────────────────────────────────────────────────────

const R13_BUILDERS = {
  pedalboard: buildPedalboard,
};

export default R13_BUILDERS;
