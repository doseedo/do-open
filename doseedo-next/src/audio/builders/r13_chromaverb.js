/**
 * R13 — ChromaVerb extended FDN reverb variants (smooth / strange / dense).
 *
 * Logic's ChromaVerb has 14 algorithms; R9 covers the basic 4
 * (room/hall/chamber/plate). This module adds three character variants:
 *
 *   fdn_smooth  — long, gentle reverb. High diffusion, gentle damping.
 *                 Think "Vocal Hall" / "Dark Hall".
 *   fdn_strange — modulated / non-stationary tail (chorused via per-line
 *                 LFOs on delay length). Think "Strange Room" / "Synth Hall".
 *   fdn_dense   — high-density reverb with aggressive early reflections.
 *                 Think "Bright Room" / "Drum Plate".
 *
 * Each is a separate node type sharing the same param surface as R9's
 * `algo_reverb`: `decay_time`, `pre_delay`, `damping`, `diffusion`, `width`,
 * `mix`. Per-variant character lives entirely in the worklet's tuning
 * constants (FDN order, base delay set, input AP cascade depth, modulation).
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildXxx(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: {...} } }
 *
 * Each builder tries to instantiate its dedicated worklet processor, and on
 * failure falls back to a primitive ConvolverNode + dry/wet pair so the
 * graph still binds and audio still flows. The fallback IR is tuned to
 * match the variant's gross spectral character (longer/darker for smooth,
 * brighter/short for dense, mid-length with random density for strange).
 */

const PROC_SMOOTH  = 'r13-fdn-smooth-processor';
const PROC_STRANGE = 'r13-fdn-strange-processor';
const PROC_DENSE   = 'r13-fdn-dense-processor';

function _safeWorklet(ctx, name, options = {}) {
  try {
    return new AudioWorkletNode(ctx, name, options);
  } catch (e) {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[R13] worklet ${name} unavailable, using fallback:`, e && e.message);
    }
    return null;
  }
}

// Synthetic exponentially-decaying noise IR tailored per variant. `decay`
// is the falloff exponent; higher = shorter tail. `tilt` skews the noise
// toward HF (positive) or LF (negative) by a single 1-pole pre-emphasis.
function _makeIR(ctx, durationSec, decay, tilt = 0) {
  const sr = ctx.sampleRate;
  const len = Math.max(1, Math.floor(sr * durationSec));
  const buf = ctx.createBuffer(2, len, sr);
  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch);
    let z = 0;
    const a = Math.max(-0.95, Math.min(0.95, tilt));
    for (let i = 0; i < len; i++) {
      const x = (Math.random() * 2 - 1);
      // 1-pole pre-emph: tilt>0 boosts HF (differentiator), tilt<0 boosts LF
      const y = x - a * z;
      z = x;
      data[i] = y * Math.exp((-decay * i) / len);
    }
  }
  return buf;
}

// Per-variant fallback IR shape: { duration, decay, tilt }
// Smooth = long & dark, Strange = mid & neutral, Dense = short & bright.
const FALLBACK_PROFILE = {
  fdn_smooth:  { duration: 4.5, decay: 2.5, tilt: -0.3 },
  fdn_strange: { duration: 2.8, decay: 3.0, tilt:  0.0 },
  fdn_dense:   { duration: 1.6, decay: 4.5, tilt:  0.4 },
};

/**
 * Common builder body shared by the three variants. The only thing that
 * differs is the worklet processor name and the fallback IR profile.
 */
function _buildVariant(variant, processorName, ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const input = ctx.createGain();
  const output = ctx.createGain();

  const initialDecay = (typeof params.decay_time === 'number') ? params.decay_time : 2.5;

  const worklet = _safeWorklet(ctx, processorName, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    parameterData: {
      decay_time: initialDecay,
    },
  });

  // ── Fallback path ────────────────────────────────────────────────────────
  let convolver = null, dryF = null, wetF = null, fallbackState = null;
  if (!worklet) {
    convolver = ctx.createConvolver();
    dryF = ctx.createGain();
    wetF = ctx.createGain();
    dryF.gain.value = 1 - 0.3;
    wetF.gain.value = 0.3;
    fallbackState = { ...FALLBACK_PROFILE[variant] };
    try { convolver.buffer = _makeIR(ctx, fallbackState.duration, fallbackState.decay, fallbackState.tilt); }
    catch (e) { /* ctx closed — leave buffer null */ }

    input.connect(dryF);
    input.connect(convolver);
    convolver.connect(wetF);
    dryF.connect(output);
    wetF.connect(output);
  } else {
    input.connect(worklet);
    worklet.connect(output);
  }

  const wpar = (name) => (worklet && worklet.parameters)
    ? (worklet.parameters.get(name) || null)
    : null;

  // ── Param wiring ─────────────────────────────────────────────────────────
  for (const [key, val] of Object.entries(params)) {
    const isModulated = typeof val === 'string' && val.startsWith('@');
    const paramId = isModulated ? val.slice(1) : null;

    switch (key) {
      case 'decay_time':
      case 'decay': {
        const ap = wpar('decay_time');
        if (isModulated) {
          if (ap) {
            targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          } else {
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => {
                fallbackState.decay = Math.max(0.1, 6 / Math.max(0.1, v));
                fallbackState.duration = Math.max(0.5, Math.min(8, v * 1.2));
                try { convolver.buffer = _makeIR(ctx, fallbackState.duration, fallbackState.decay, fallbackState.tilt); }
                catch (e) { /* ignore */ }
              },
            };
          }
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          else if (convolver) {
            fallbackState.duration = Math.max(0.5, Math.min(8, val * 1.2));
            fallbackState.decay = Math.max(0.1, 6 / Math.max(0.1, val));
            try { convolver.buffer = _makeIR(ctx, fallbackState.duration, fallbackState.decay, fallbackState.tilt); }
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

export function buildFdnSmooth(ctx, node, paramDefs) {
  return _buildVariant('fdn_smooth', PROC_SMOOTH, ctx, node, paramDefs);
}

export function buildFdnStrange(ctx, node, paramDefs) {
  return _buildVariant('fdn_strange', PROC_STRANGE, ctx, node, paramDefs);
}

export function buildFdnDense(ctx, node, paramDefs) {
  return _buildVariant('fdn_dense', PROC_DENSE, ctx, node, paramDefs);
}

// ── Default export: NODE_BUILDERS map ─────────────────────────────────────
const R13_BUILDERS = {
  fdn_smooth:  buildFdnSmooth,
  fdn_strange: buildFdnStrange,
  fdn_dense:   buildFdnDense,
};

export default R13_BUILDERS;
