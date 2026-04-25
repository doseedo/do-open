/**
 * R9 — Multi-algorithm FDN reverb composite builder
 *
 * Registers `algo_reverb` as a NEW node type. Internally it constructs an
 * AudioWorkletNode running `r9-algo-reverb-processor` (a Feedback Delay
 * Network with 4 selectable algorithms: room/hall/chamber/plate).
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildAlgoReverb(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: {...} } }
 *
 * Following the convention established by R4: if the worklet processor isn't
 * registered yet on the AudioContext (e.g. the engine hasn't called
 * `audioWorklet.addModule(...)` for r9-algo-reverb-processor.js), we fall
 * back to a ConvolverNode-based reverb so the schema still renders + audio
 * still flows. Once the worklet is loaded the same builder picks up the real
 * FDN with no code change.
 *
 * Author: Agent R9
 */

const R9_ALGO_REVERB = 'r9-algo-reverb-processor';

// Map symbolic algorithm names → enum index understood by the worklet
const ALGO_NAME_TO_INDEX = {
  room:    0,
  hall:    1,
  chamber: 2,
  plate:   3,
};

function _algoIndex(value) {
  if (typeof value === 'number') return Math.max(0, Math.min(3, Math.round(value)));
  if (typeof value === 'string') {
    const idx = ALGO_NAME_TO_INDEX[value.toLowerCase()];
    return idx == null ? 1 : idx; // default hall
  }
  return 1;
}

/**
 * Try to construct an AudioWorkletNode. If the named processor isn't
 * registered (worklet module not yet loaded), return null so the caller can
 * fall back to a primitive substitute. We never want a missing worklet to
 * break the whole graph build.
 */
function _safeWorklet(ctx, name, options = {}) {
  try {
    return new AudioWorkletNode(ctx, name, options);
  } catch (e) {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[R9] worklet ${name} unavailable, using fallback:`, e && e.message);
    }
    return null;
  }
}

/**
 * Synthetic exponentially-decaying noise IR for the ConvolverNode fallback.
 * Decay factor of ~3 maps roughly to a 2 s RT60 at 2.5 s buffer length.
 */
function _makeFallbackIR(ctx, durationSec, decay) {
  const sr = ctx.sampleRate;
  const len = Math.max(1, Math.floor(sr * durationSec));
  const buf = ctx.createBuffer(2, len, sr);
  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch);
    for (let i = 0; i < len; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.exp((-decay * i) / len);
    }
  }
  return buf;
}

// ── algo_reverb builder ───────────────────────────────────────────────────
//
// Schema:
//   {
//     type: 'algo_reverb',
//     params: {
//       algorithm:  'hall' | 'room' | 'chamber' | 'plate' | '@<id>',
//       decay_time: number (s) | '@<id>',
//       pre_delay:  number (ms) | '@<id>',
//       damping:    number (0..1) | '@<id>',
//       diffusion:  number (0..1) | '@<id>',
//       width:      number (0..1) | '@<id>',
//       mix:        number (0..1) | '@<id>',
//     }
//   }
export function buildAlgoReverb(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const input = ctx.createGain();
  const output = ctx.createGain();

  // Build the worklet (or fall back to a Convolver chain)
  const initialAlgo  = _algoIndex(typeof params.algorithm === 'string' && params.algorithm.startsWith('@')
    ? 'hall'
    : (params.algorithm ?? 'hall'));
  const initialDecay = (typeof params.decay_time === 'number') ? params.decay_time : 2.5;

  const worklet = _safeWorklet(ctx, R9_ALGO_REVERB, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    parameterData: {
      algorithm:  initialAlgo,
      decay_time: initialDecay,
    },
  });

  // ── Fallback path ────────────────────────────────────────────────────────
  // ConvolverNode + dry/wet gains, exposing the same param surface so the
  // engine doesn't see a different shape when the worklet isn't loaded yet.
  let convolver = null, dryF = null, wetF = null, fallbackState = null;
  if (!worklet) {
    convolver = ctx.createConvolver();
    dryF = ctx.createGain();
    wetF = ctx.createGain();
    dryF.gain.value = 1 - 0.3;
    wetF.gain.value = 0.3;
    fallbackState = { duration: 2.5, decay: 3.0 };
    try { convolver.buffer = _makeFallbackIR(ctx, fallbackState.duration, fallbackState.decay); }
    catch (e) { /* ctx closed — leave buffer null */ }

    input.connect(dryF);
    input.connect(convolver);
    convolver.connect(wetF);
    dryF.connect(output);
    wetF.connect(output);
  } else {
    // Worklet path: input → worklet → output. The worklet itself handles the
    // dry/wet mix internally via the `mix` param.
    input.connect(worklet);
    worklet.connect(output);
  }

  // Helper to fetch a worklet AudioParam by name (null if no worklet)
  const wpar = (name) => (worklet && worklet.parameters)
    ? (worklet.parameters.get(name) || null)
    : null;

  // ── Param wiring ─────────────────────────────────────────────────────────
  for (const [key, val] of Object.entries(params)) {
    const isModulated = typeof val === 'string' && val.startsWith('@');
    const paramId = isModulated ? val.slice(1) : null;

    switch (key) {
      case 'algorithm': {
        const ap = wpar('algorithm');
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              // accept normalised 0..1 (mapped to enum 0..3) OR direct 0..3 OR string
              let idx;
              if (typeof v === 'string') idx = _algoIndex(v);
              else if (v >= 0 && v <= 1) idx = Math.min(3, Math.floor(v * 4));
              else idx = _algoIndex(v);
              if (ap) ap.value = idx;
              // Fallback: regenerate IR with brighter/darker decay per algo
              if (!worklet && convolver) {
                const decayMap = [4.0, 2.0, 3.0, 1.5]; // room/hall/chamber/plate
                const durMap   = [1.2, 4.5, 2.5, 2.0];
                fallbackState.decay = decayMap[idx] || 2.0;
                fallbackState.duration = durMap[idx] || 2.5;
                try { convolver.buffer = _makeFallbackIR(ctx, fallbackState.duration, fallbackState.decay); }
                catch (e) { /* ignore */ }
              }
            },
          };
        } else if (val !== undefined) {
          const idx = _algoIndex(val);
          if (ap) ap.value = idx;
          // fallback handled at construction via initialAlgo
        }
        break;
      }
      case 'decay_time':
      case 'decay': {
        const ap = wpar('decay_time');
        if (isModulated) {
          if (ap) {
            targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          } else {
            // Fallback: regenerate IR with new decay tail
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => {
                fallbackState.decay = Math.max(0.1, 6 / Math.max(0.1, v));
                fallbackState.duration = Math.max(0.5, Math.min(8, v * 1.2));
                try { convolver.buffer = _makeFallbackIR(ctx, fallbackState.duration, fallbackState.decay); }
                catch (e) { /* ignore */ }
              },
            };
          }
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          else if (convolver) {
            fallbackState.duration = Math.max(0.5, Math.min(8, val * 1.2));
            fallbackState.decay = Math.max(0.1, 6 / Math.max(0.1, val));
            try { convolver.buffer = _makeFallbackIR(ctx, fallbackState.duration, fallbackState.decay); }
            catch (e) { /* ignore */ }
          }
        }
        break;
      }
      case 'pre_delay': {
        const ap = wpar('pre_delay');
        if (isModulated && ap) {
          targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
        } else if (isModulated) {
          // Fallback: pre-delay isn't free with a Convolver — approximate via
          // a serial DelayNode injection. To keep this builder simple we just
          // accept the param and no-op (graph still binds).
          targets[paramId] = { paramDef: paramDefs[paramId], customSetter: () => {} };
        } else if (ap && typeof val === 'number') {
          ap.value = val;
        }
        break;
      }
      case 'damping': {
        const ap = wpar('damping');
        if (isModulated && ap) {
          targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
        } else if (isModulated) {
          targets[paramId] = { paramDef: paramDefs[paramId], customSetter: () => {} };
        } else if (ap && typeof val === 'number') {
          ap.value = val;
        }
        break;
      }
      case 'diffusion': {
        const ap = wpar('diffusion');
        if (isModulated && ap) {
          targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
        } else if (isModulated) {
          targets[paramId] = { paramDef: paramDefs[paramId], customSetter: () => {} };
        } else if (ap && typeof val === 'number') {
          ap.value = val;
        }
        break;
      }
      case 'width': {
        const ap = wpar('width');
        if (isModulated && ap) {
          targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
        } else if (isModulated) {
          targets[paramId] = { paramDef: paramDefs[paramId], customSetter: () => {} };
        } else if (ap && typeof val === 'number') {
          ap.value = val;
        }
        break;
      }
      case 'mix': {
        const ap = wpar('mix');
        if (isModulated) {
          if (ap) {
            targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          } else {
            // Fallback: drive convolver dry/wet gain pair
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => {
                const w = Math.max(0, Math.min(1, v));
                if (wetF && dryF) {
                  wetF.gain.value = w;
                  dryF.gain.value = 1 - w;
                }
              },
            };
          }
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          else if (wetF && dryF) {
            wetF.gain.value = val;
            dryF.gain.value = 1 - val;
          }
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
const R9_BUILDERS = {
  algo_reverb: buildAlgoReverb,
};

export default R9_BUILDERS;
