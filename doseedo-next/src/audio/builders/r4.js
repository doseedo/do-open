/**
 * R4 — "Circuit Models" composite builders
 *
 * Each builder composes existing primitives (BiquadFilterNode, GainNode,
 * DelayNode, ConvolverNode) plus AudioWorkletNode references owned by R2/R3
 * (e.g. `r2-wdf-tube-amp-processor`, `r3-wdf-tape-sat-processor`,
 * `r3-wdf-transformer-processor`, `r3-wdf-rc-filter-processor`).
 *
 * R2/R3 are running in parallel — at composite-build time their worklets may
 * not yet be registered on the AudioContext. We therefore wrap every
 * `new AudioWorkletNode(ctx, name, ...)` call in `_safeWorklet()`, which
 * falls back to a passthrough GainNode if the processor is unknown. This
 * lets the schema render and the audio graph build today; once R2/R3 ship
 * and `audioWorklet.addModule(...)` has run, the same builders will pick up
 * the real WDF processors with no code change.
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildFoo(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: { audioParam?, paramDef, customSetter?, scale? } } }
 *
 * Author: Agent R4
 */

import { buildCabinetIR } from '../cabinet-ir.js';

// ── Names of worklets owned by other agents ───────────────────────────────
// Treat as black boxes. R4 only constructs them; R2/R3 register them.
const R2_TUBE_AMP    = 'r2-wdf-tube-amp-processor';
const R2_TONE_STACK  = 'r2-wdf-tone-stack-processor';
const R3_TRANSFORMER = 'r3-wdf-transformer-processor';
const R3_TAPE_SAT    = 'r3-wdf-tape-sat-processor';
const R3_RC_FILTER   = 'r3-wdf-rc-filter-processor';

// ── Internal helpers ──────────────────────────────────────────────────────

/**
 * Try to construct an AudioWorkletNode. If the named processor isn't
 * registered on the context yet (R2/R3 haven't loaded), return null so the
 * caller can fall back to a primitive substitute. We never want a missing
 * worklet to break the whole graph build.
 */
function _safeWorklet(ctx, name, options = {}) {
  try {
    return new AudioWorkletNode(ctx, name, options);
  } catch (e) {
    // Processor not registered — composite must fall back.
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[R4] worklet ${name} unavailable, using fallback:`, e.message);
    }
    return null;
  }
}

/**
 * Pull a static param value or set up an @param mapping target.
 * Mirrors the pattern used throughout WebAudioDSPEngine.js so engine code
 * can bind real-time control to the returned `paramTargets`.
 */
function _wireParam(node, paramKey, paramVal, paramDefs, targets, mapping) {
  if (typeof paramVal === 'string' && paramVal.startsWith('@')) {
    const paramId = paramVal.slice(1);
    targets[paramId] = {
      paramDef: paramDefs[paramId],
      ...mapping(paramId),
    };
  } else if (paramVal !== undefined && mapping.staticSet) {
    mapping.staticSet(paramVal);
  }
}

/**
 * Map a worklet AudioParam by name if the worklet exists; else no-op.
 */
function _workletParam(workletNode, paramName) {
  if (!workletNode || !workletNode.parameters) return null;
  return workletNode.parameters.get(paramName) || null;
}

/**
 * Connect a → b safely (b might be null if it was a worklet that didn't exist
 * — in that case we no-op, the graph will route around it).
 */
function _connectChain(...nodes) {
  const filtered = nodes.filter(Boolean);
  for (let i = 0; i < filtered.length - 1; i++) {
    filtered[i].connect(filtered[i + 1]);
  }
  return { head: filtered[0], tail: filtered[filtered.length - 1] };
}

// ── 1. circuit_fender_bassman ─────────────────────────────────────────────
// input → tube preamp (R2 wdf_tube_amp, stages=2)
//       → tone stack  (R2 wdf_tone_stack, fallback: 3-band biquad shelves)
//       → transformer (R3 wdf_transformer, fallback: soft-knee waveshaper via gain stage)
//       → speaker IR  (ConvolverNode w/ synthetic cabinet IR)
//       → master out
export function buildCircuitFenderBassman(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const input = ctx.createGain();
  const output = ctx.createGain();
  const dry = ctx.createGain();
  const wet = ctx.createGain();
  dry.gain.value = 0;       // Bassman is normally 100% wet; mix kept for parallel use
  wet.gain.value = 1;

  // -- Tube preamp stage (R2) --
  const tubeAmp = _safeWorklet(ctx, R2_TUBE_AMP, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    parameterData: { stages: 2, gain: 1, bias: -1.5, output_level: 0.6 },
  });
  // Fallback: WaveShaper-style soft clip via createWaveShaper
  let tubeFallback = null;
  if (!tubeAmp) {
    tubeFallback = ctx.createWaveShaper();
    const curve = new Float32Array(2048);
    for (let i = 0; i < curve.length; i++) {
      const x = (i / curve.length) * 2 - 1;
      curve[i] = Math.tanh(x * 2.5);
    }
    tubeFallback.curve = curve;
  }
  const tubeStage = tubeAmp || tubeFallback;

  // -- Tone stack (R2 worklet OR 3-biquad fallback shaped as a Fender FMV stack) --
  const toneStack = _safeWorklet(ctx, R2_TONE_STACK, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    parameterData: { bass: 0.5, mid: 0.5, treble: 0.5 },
  });

  // Fallback FMV-like topology: 3 cascaded biquads (low shelf, peaking mid, high shelf)
  const bassFb = ctx.createBiquadFilter();
  bassFb.type = 'lowshelf'; bassFb.frequency.value = 100; bassFb.gain.value = 0;
  const midFb = ctx.createBiquadFilter();
  midFb.type = 'peaking'; midFb.frequency.value = 500; midFb.Q.value = 0.7; midFb.gain.value = 0;
  const trebFb = ctx.createBiquadFilter();
  trebFb.type = 'highshelf'; trebFb.frequency.value = 3500; trebFb.gain.value = 0;

  // -- Output transformer (R3) --
  const xfmr = _safeWorklet(ctx, R3_TRANSFORMER, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    parameterData: { drive: 1, saturation: 0.5 },
  });
  // Fallback: gentle waveshaper to emulate transformer core saturation
  let xfmrFallback = null;
  if (!xfmr) {
    xfmrFallback = ctx.createWaveShaper();
    const curve = new Float32Array(2048);
    for (let i = 0; i < curve.length; i++) {
      const x = (i / curve.length) * 2 - 1;
      curve[i] = Math.tanh(x * 1.5) * 0.95;
    }
    xfmrFallback.curve = curve;
  }
  const xfmrStage = xfmr || xfmrFallback;

  // -- Speaker IR (cabinet emulation) --
  const cab = ctx.createConvolver();
  try { cab.buffer = buildCabinetIR(ctx); } catch (e) { /* ctx closed */ }

  // -- Presence (post-cab high shelf, since presence sits in the NFB loop on real Bassmans) --
  const presence = ctx.createBiquadFilter();
  presence.type = 'highshelf';
  presence.frequency.value = 3000;
  presence.gain.value = 0;

  // -- Master --
  const master = ctx.createGain();
  master.gain.value = 0.5;

  // -- Build chain --
  input.connect(dry);
  input.connect(wet);

  let chainHead = wet;
  if (tubeStage) { chainHead.connect(tubeStage); chainHead = tubeStage; }
  if (toneStack) {
    chainHead.connect(toneStack);
    chainHead = toneStack;
  } else {
    chainHead.connect(bassFb);
    bassFb.connect(midFb);
    midFb.connect(trebFb);
    chainHead = trebFb;
  }
  if (xfmrStage) { chainHead.connect(xfmrStage); chainHead = xfmrStage; }
  chainHead.connect(cab);
  cab.connect(presence);
  presence.connect(master);
  master.connect(output);
  dry.connect(output);

  // -- Param wiring --
  for (const [key, val] of Object.entries(params)) {
    const isModulated = typeof val === 'string' && val.startsWith('@');
    const paramId = isModulated ? val.slice(1) : null;

    switch (key) {
      case 'gain': {
        const ap = _workletParam(tubeAmp, 'gain');
        if (isModulated) {
          if (ap) targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          else if (tubeFallback) {
            // Drive a pre-shaper gain instead — wrap tubeFallback with a gain in front
            // For simplicity, expose mapping via customSetter on input gain
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => { input.gain.value = v; },
            };
          }
        } else if (ap) ap.value = val;
        break;
      }
      case 'bass': {
        const ap = _workletParam(toneStack, 'bass');
        if (isModulated) {
          if (ap) targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          else {
            // bass [0..1] → ±12 dB shelf
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => { bassFb.gain.value = (v - 0.5) * 24; },
            };
          }
        } else if (ap) ap.value = val;
        else bassFb.gain.value = (val - 0.5) * 24;
        break;
      }
      case 'mid': {
        const ap = _workletParam(toneStack, 'mid');
        if (isModulated) {
          if (ap) targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          else {
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => { midFb.gain.value = (v - 0.5) * 18; },
            };
          }
        } else if (ap) ap.value = val;
        else midFb.gain.value = (val - 0.5) * 18;
        break;
      }
      case 'treble': {
        const ap = _workletParam(toneStack, 'treble');
        if (isModulated) {
          if (ap) targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          else {
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => { trebFb.gain.value = (v - 0.5) * 24; },
            };
          }
        } else if (ap) ap.value = val;
        else trebFb.gain.value = (val - 0.5) * 24;
        break;
      }
      case 'presence': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => { presence.gain.value = v * 12; },
          };
        } else if (typeof val === 'number') {
          presence.gain.value = val * 12;
        }
        break;
      }
      // Both schema-name (`master`) and prompt-name (`output_level`) accepted
      case 'master':
      case 'output_level': {
        if (isModulated) {
          targets[paramId] = { audioParam: master.gain, paramDef: paramDefs[paramId] };
        } else if (typeof val === 'number') {
          master.gain.value = val;
        }
        break;
      }
      case 'mix': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              wet.gain.value = v;
              dry.gain.value = 1 - v;
            },
          };
        } else if (typeof val === 'number') {
          wet.gain.value = val;
          dry.gain.value = 1 - val;
        }
        break;
      }
      default: break;
    }
  }

  return { input, output, paramTargets: targets };
}

// ── 2. circuit_pultec_eq ──────────────────────────────────────────────────
// Pultec "boost+attenuate same freq" trick implemented with parallel biquads.
// Low band: lowshelf boost + lowshelf cut at slightly offset corners produces
// the iconic phase-shifted curve where boost+cut ≠ flat.
export function buildCircuitPultecEq(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const input = ctx.createGain();
  const output = ctx.createGain();
  const dry = ctx.createGain();
  const wet = ctx.createGain();
  dry.gain.value = 0;
  wet.gain.value = 1;

  // -- Low band: serial boost-shelf + cut-shelf at offset frequencies --
  const lowBoost = ctx.createBiquadFilter();
  lowBoost.type = 'lowshelf';
  lowBoost.frequency.value = 60;     // boost corner
  lowBoost.gain.value = 0;
  const lowCut = ctx.createBiquadFilter();
  lowCut.type = 'lowshelf';
  lowCut.frequency.value = 90;       // cut corner — slightly higher creates "scoop"
  lowCut.gain.value = 0;

  // -- High band: bell boost (with bandwidth control) + high shelf cut --
  const highBoost = ctx.createBiquadFilter();
  highBoost.type = 'peaking';
  highBoost.frequency.value = 10000;
  highBoost.Q.value = 1.0;
  highBoost.gain.value = 0;
  const highCut = ctx.createBiquadFilter();
  highCut.type = 'highshelf';
  highCut.frequency.value = 20000;   // separate cut frequency on a real EQP-1A
  highCut.gain.value = 0;

  // -- Output makeup --
  const outGain = ctx.createGain();
  outGain.gain.value = 1;

  input.connect(dry);
  input.connect(wet);
  wet.connect(lowBoost);
  lowBoost.connect(lowCut);
  lowCut.connect(highBoost);
  highBoost.connect(highCut);
  highCut.connect(outGain);
  outGain.connect(output);
  dry.connect(output);

  for (const [key, val] of Object.entries(params)) {
    const isModulated = typeof val === 'string' && val.startsWith('@');
    const paramId = isModulated ? val.slice(1) : null;

    switch (key) {
      case 'low_freq': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              lowBoost.frequency.value = v;
              lowCut.frequency.value = v * 1.5; // cut frequency tracks ~1.5× boost (Pultec quirk)
            },
          };
        } else if (typeof val === 'number') {
          lowBoost.frequency.value = val;
          lowCut.frequency.value = val * 1.5;
        }
        break;
      }
      case 'low_boost': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => { lowBoost.gain.value = v * 12; }, // 0..1 → 0..+12 dB
          };
        } else if (typeof val === 'number') {
          lowBoost.gain.value = val * 12;
        }
        break;
      }
      case 'low_atten': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => { lowCut.gain.value = -v * 12; }, // 0..1 → 0..-12 dB
          };
        } else if (typeof val === 'number') {
          lowCut.gain.value = -val * 12;
        }
        break;
      }
      case 'high_freq': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => { highBoost.frequency.value = v; },
          };
        } else if (typeof val === 'number') {
          highBoost.frequency.value = val;
        }
        break;
      }
      case 'high_boost': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => { highBoost.gain.value = v * 16; }, // 0..+16 dB
          };
        } else if (typeof val === 'number') {
          highBoost.gain.value = val * 16;
        }
        break;
      }
      case 'high_atten': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => { highCut.gain.value = -v * 16; },
          };
        } else if (typeof val === 'number') {
          highCut.gain.value = -val * 16;
        }
        break;
      }
      case 'high_bandwidth': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            // 0..1 → Q 0.3..3.0 (sharp → broad inverse)
            customSetter: (v) => { highBoost.Q.value = 0.3 + (1 - v) * 2.7; },
          };
        } else if (typeof val === 'number') {
          highBoost.Q.value = 0.3 + (1 - val) * 2.7;
        }
        break;
      }
      case 'output': {
        if (isModulated) {
          targets[paramId] = { audioParam: outGain.gain, paramDef: paramDefs[paramId] };
        } else if (typeof val === 'number') {
          outGain.gain.value = val;
        }
        break;
      }
      case 'mix': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              wet.gain.value = v;
              dry.gain.value = 1 - v;
            },
          };
        } else if (typeof val === 'number') {
          wet.gain.value = val;
          dry.gain.value = 1 - val;
        }
        break;
      }
      default: break;
    }
  }

  return { input, output, paramTargets: targets };
}

// ── 3. circuit_tape_machine ───────────────────────────────────────────────
// Full tape chain. Record-EQ pre-emphasis, R3 tape saturation, playback de-
// emphasis, wow/flutter via DelayNode whose delayTime is modulated by an LFO,
// then a head-bump low shelf and a wet/dry mix.
export function buildCircuitTapeMachine(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const input = ctx.createGain();
  const output = ctx.createGain();
  const dry = ctx.createGain();
  const wet = ctx.createGain();
  dry.gain.value = 0;
  wet.gain.value = 1;

  const inputLevel = ctx.createGain();
  inputLevel.gain.value = 1;

  // -- Record-EQ: high-shelf boost ~10 kHz to compensate for HF loss in tape --
  const recordEq = ctx.createBiquadFilter();
  recordEq.type = 'highshelf';
  recordEq.frequency.value = 10000;
  recordEq.gain.value = 6; // ~+6 dB pre-emphasis

  // -- Tape saturation (R3) — fallback to waveshaper --
  const tapeSat = _safeWorklet(ctx, R3_TAPE_SAT, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    parameterData: { input_level: 1.5, bias: 0.5, speed: 0.5, head_bump: 0.5 },
  });
  let tapeFallback = null;
  if (!tapeSat) {
    tapeFallback = ctx.createWaveShaper();
    const curve = new Float32Array(2048);
    for (let i = 0; i < curve.length; i++) {
      const x = (i / curve.length) * 2 - 1;
      // Asymmetric tape curve: sligher more compression on positive half
      curve[i] = Math.tanh(x * 1.6) * (x > 0 ? 0.9 : 1.0);
    }
    tapeFallback.curve = curve;
  }
  const satStage = tapeSat || tapeFallback;

  // -- Playback-EQ: counterpart to recordEq (high-shelf cut) --
  const playbackEq = ctx.createBiquadFilter();
  playbackEq.type = 'highshelf';
  playbackEq.frequency.value = 10000;
  playbackEq.gain.value = -6;

  // -- Wow/flutter: DelayNode whose time is wobbled by an LFO --
  // Combine slow wow (~0.5 Hz) + fast flutter (~6 Hz) into one modulating signal.
  const wfDelay = ctx.createDelay(0.1);
  wfDelay.delayTime.value = 0.025; // 25 ms baseline so we can wobble around it

  const wowLfo = ctx.createOscillator();
  wowLfo.frequency.value = 0.5;
  wowLfo.type = 'sine';
  const wowDepth = ctx.createGain();
  wowDepth.gain.value = 0.0015; // ±1.5 ms

  const flutterLfo = ctx.createOscillator();
  flutterLfo.frequency.value = 6.0;
  flutterLfo.type = 'sine';
  const flutterDepth = ctx.createGain();
  flutterDepth.gain.value = 0.0005; // ±0.5 ms

  wowLfo.connect(wowDepth);
  flutterLfo.connect(flutterDepth);
  wowDepth.connect(wfDelay.delayTime);
  flutterDepth.connect(wfDelay.delayTime);
  wowLfo.start();
  flutterLfo.start();

  // -- Head bump: low-frequency resonance from tape contact (~80 Hz) --
  const headBump = ctx.createBiquadFilter();
  headBump.type = 'peaking';
  headBump.frequency.value = 80;
  headBump.Q.value = 2;
  headBump.gain.value = 3;

  // -- Build chain --
  input.connect(dry);
  input.connect(wet);
  wet.connect(inputLevel);
  inputLevel.connect(recordEq);

  let chainHead = recordEq;
  if (satStage) { chainHead.connect(satStage); chainHead = satStage; }
  chainHead.connect(playbackEq);
  playbackEq.connect(wfDelay);
  wfDelay.connect(headBump);
  headBump.connect(output);
  dry.connect(output);

  for (const [key, val] of Object.entries(params)) {
    const isModulated = typeof val === 'string' && val.startsWith('@');
    const paramId = isModulated ? val.slice(1) : null;

    switch (key) {
      case 'input_level': {
        if (isModulated) {
          targets[paramId] = { audioParam: inputLevel.gain, paramDef: paramDefs[paramId] };
        } else if (typeof val === 'number') {
          inputLevel.gain.value = val;
        }
        // Also pipe to the worklet's input_level if available
        const ap = _workletParam(tapeSat, 'input_level');
        if (ap && !isModulated && typeof val === 'number') ap.value = val;
        break;
      }
      case 'bias': {
        const ap = _workletParam(tapeSat, 'bias');
        if (isModulated && ap) {
          targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
        } else if (isModulated) {
          // No worklet → bias has no fallback effect; expose a no-op customSetter so
          // the engine still considers the param "bound".
          targets[paramId] = { paramDef: paramDefs[paramId], customSetter: () => {} };
        } else if (ap && typeof val === 'number') {
          ap.value = val;
        }
        break;
      }
      case 'speed': {
        const ap = _workletParam(tapeSat, 'speed');
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              if (ap) ap.value = v;
              // Speed also affects EQ corner: faster tape → higher HF capture
              recordEq.frequency.value = 6000 + v * 8000;   // 6–14 kHz
              playbackEq.frequency.value = recordEq.frequency.value;
            },
          };
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          recordEq.frequency.value = 6000 + val * 8000;
          playbackEq.frequency.value = recordEq.frequency.value;
        }
        break;
      }
      case 'wow_flutter': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              wowDepth.gain.value = v * 0.003;       // ±0..3 ms
              flutterDepth.gain.value = v * 0.001;   // ±0..1 ms
            },
          };
        } else if (typeof val === 'number') {
          wowDepth.gain.value = val * 0.003;
          flutterDepth.gain.value = val * 0.001;
        }
        break;
      }
      case 'head_bump': {
        const ap = _workletParam(tapeSat, 'head_bump');
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              headBump.gain.value = v * 8;
              if (ap) ap.value = v;
            },
          };
        } else if (typeof val === 'number') {
          headBump.gain.value = val * 8;
          if (ap) ap.value = val;
        }
        break;
      }
      case 'mix': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              wet.gain.value = v;
              dry.gain.value = 1 - v;
            },
          };
        } else if (typeof val === 'number') {
          wet.gain.value = val;
          dry.gain.value = 1 - val;
        }
        break;
      }
      default: break;
    }
  }

  return {
    input,
    output,
    paramTargets: targets,
    oscillators: [wowLfo, flutterLfo],
  };
}

// ── 4. circuit_tube_preamp ────────────────────────────────────────────────
// Generic tube preamp: optional bright-cap high-shelf → R2 wdf_tube_amp →
// 1-band tilt EQ → output level.
export function buildCircuitTubePreamp(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const input = ctx.createGain();
  const output = ctx.createGain();
  const dry = ctx.createGain();
  const wet = ctx.createGain();
  dry.gain.value = 0;
  wet.gain.value = 1;

  // -- Bright switch: high-shelf only engages when bright > 0.5 --
  const bright = ctx.createBiquadFilter();
  bright.type = 'highshelf';
  bright.frequency.value = 4000;
  bright.gain.value = 0;

  // -- Tube amp (R2) --
  const tubeAmp = _safeWorklet(ctx, R2_TUBE_AMP, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    parameterData: { stages: 2, gain: 1, bias: -1.5, output_level: 0.5 },
  });
  // Fallback: tanh waveshaper, repeated per stage
  let tubeFallback = null;
  if (!tubeAmp) {
    tubeFallback = ctx.createWaveShaper();
    const curve = new Float32Array(2048);
    for (let i = 0; i < curve.length; i++) {
      const x = (i / curve.length) * 2 - 1;
      curve[i] = Math.tanh(x * 2.0);
    }
    tubeFallback.curve = curve;
  }
  const tubeStage = tubeAmp || tubeFallback;

  // -- Tilt EQ: low shelf + matching high shelf with opposite gain.
  // A single "tone" param controls both — implemented here as a peaking biquad
  // for simplicity (any tone-shaped primitive will do). --
  const toneTilt = ctx.createBiquadFilter();
  toneTilt.type = 'highshelf';
  toneTilt.frequency.value = 1500;
  toneTilt.gain.value = 0;

  const outLevel = ctx.createGain();
  outLevel.gain.value = 0.4;

  input.connect(dry);
  input.connect(wet);
  wet.connect(bright);
  let chainHead = bright;
  if (tubeStage) { chainHead.connect(tubeStage); chainHead = tubeStage; }
  chainHead.connect(toneTilt);
  toneTilt.connect(outLevel);
  outLevel.connect(output);
  dry.connect(output);

  for (const [key, val] of Object.entries(params)) {
    const isModulated = typeof val === 'string' && val.startsWith('@');
    const paramId = isModulated ? val.slice(1) : null;

    switch (key) {
      case 'gain': {
        const ap = _workletParam(tubeAmp, 'gain');
        if (isModulated) {
          if (ap) targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          else {
            // Fallback: drive input gain so the waveshaper saturates harder
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => { input.gain.value = v; },
            };
          }
        } else if (ap && typeof val === 'number') {
          ap.value = val;
        }
        break;
      }
      case 'stages': {
        const ap = _workletParam(tubeAmp, 'stages');
        // stages is typically not modulated at audio rate, but support it anyway.
        if (isModulated) {
          if (ap) targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          else targets[paramId] = { paramDef: paramDefs[paramId], customSetter: () => {} };
        } else if (ap && typeof val === 'number') {
          ap.value = val;
        }
        break;
      }
      case 'bright': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            // bright > 0.5 engages, scaled to +0..+12 dB above the switch threshold
            customSetter: (v) => {
              bright.gain.value = v > 0.5 ? (v - 0.5) * 24 : 0;
            },
          };
        } else if (typeof val === 'number') {
          bright.gain.value = val > 0.5 ? (val - 0.5) * 24 : 0;
        }
        break;
      }
      case 'output_level': {
        if (isModulated) {
          targets[paramId] = { audioParam: outLevel.gain, paramDef: paramDefs[paramId] };
        } else if (typeof val === 'number') {
          outLevel.gain.value = val;
        }
        break;
      }
      case 'tone': {
        // Optional extra param if user requests tilt EQ
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => { toneTilt.gain.value = (v - 0.5) * 18; },
          };
        } else if (typeof val === 'number') {
          toneTilt.gain.value = (val - 0.5) * 18;
        }
        break;
      }
      case 'mix': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              wet.gain.value = v;
              dry.gain.value = 1 - v;
            },
          };
        } else if (typeof val === 'number') {
          wet.gain.value = val;
          dry.gain.value = 1 - val;
        }
        break;
      }
      default: break;
    }
  }

  return { input, output, paramTargets: targets };
}

// ── Default export: NODE_BUILDERS map ─────────────────────────────────────
const R4_BUILDERS = {
  circuit_fender_bassman: buildCircuitFenderBassman,
  circuit_pultec_eq:      buildCircuitPultecEq,
  circuit_tape_machine:   buildCircuitTapeMachine,
  circuit_tube_preamp:    buildCircuitTubePreamp,
};

export default R4_BUILDERS;
