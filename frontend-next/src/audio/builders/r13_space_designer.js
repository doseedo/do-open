/**
 * R13 — Space Designer (convolution_sd) builder
 *
 * Registers a NEW DSP node type `convolution_sd` that mirrors Logic Pro's
 * Space Designer convolution reverb. Built on top of the R1 `convolution`
 * primitive (ConvolverNode) but adds the IR-shaping operations Space Designer
 * exposes:
 *
 *   length        0–100% — truncate the IR to a fraction of its raw length
 *   attack_time   ms     — linear fade-in envelope at the IR start
 *   decay_time    ms     — exponential fade-out envelope across the IR tail
 *   predelay      ms     — silence prepended before the IR fires
 *   low_cut       Hz     — IIR highpass placed BEFORE the convolver
 *   high_cut      Hz     — IIR lowpass  placed BEFORE the convolver
 *   mix           0–1    — dry / wet
 *   reverse       bool   — reverse the source IR
 *   density       0–1    — IR resampling factor for grain / density variation
 *
 * Because `ConvolverNode.buffer` is read at the time it's assigned, every
 * IR-shape param mutation rebuilds the IR buffer and re-assigns it. The cost
 * is amortised across `setTargetAtTime`-rate knob drags via a small debounce.
 *
 * A default IR is generated synthetically (exponentially-decaying noise) on
 * construction so the node always renders audible reverb without an external
 * IR file. `loadIR(srcOrUrl)` is exposed on the returned object for runtime
 * IR swaps; the engine can grab it by reference.
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildConvolutionSD(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets, loadIR }
 *
 * Worklet path: a `r13-convolution-sd-processor` is reserved for a future
 * partitioned-FFT convolution implementation; for now we ship the
 * ConvolverNode-only path. If/when the worklet lands, callers using
 * `params.use_worklet=true` would route through it. Today that path is a
 * no-op upgrade hook.
 *
 * Author: Agent R13
 */

const R13_WORKLET_NAME = 'r13-convolution-sd-processor';

// ── Helpers ───────────────────────────────────────────────────────────────

function isAtBinding(v) {
  return typeof v === 'string' && v.startsWith('@');
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * Synthesised exponentially-decaying noise IR. Used as the default source
 * when no external IR has been loaded yet — guarantees the node makes audible
 * reverb on the first build.
 */
function makeDefaultSourceIR(ctx, durationSec = 2.5, decay = 3.0) {
  const sr = ctx.sampleRate;
  const len = Math.max(1, Math.floor(sr * durationSec));
  const buf = ctx.createBuffer(2, len, sr);
  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch);
    for (let i = 0; i < len; i++) {
      const t = i / len;
      data[i] = (Math.random() * 2 - 1) * Math.exp(-decay * t);
    }
  }
  return buf;
}

/**
 * Apply Space Designer IR shape on top of a SOURCE buffer. Returns a NEW
 * AudioBuffer suitable for assignment to `ConvolverNode.buffer`.
 *
 * Shape order (matches Logic's signal flow inside Space Designer):
 *   1. reverse the source
 *   2. resample by `density` (sub-sample for grainier feel; 1.0 = identity)
 *   3. truncate to `length` fraction of the resampled source
 *   4. apply linear attack envelope (`attack_time` ms fade-in)
 *   5. apply exponential decay envelope across the remainder (`decay_time` ms)
 *   6. prepend `predelay` ms of silence
 *
 * No allocation in process()-style loops — only called on param mutation.
 */
function shapeIR(ctx, sourceBuffer, shape) {
  if (!sourceBuffer) return null;
  const sr = ctx.sampleRate;
  const channels = sourceBuffer.numberOfChannels;
  const srcLen = sourceBuffer.length;

  const length    = clamp(shape.length    ?? 1.0, 0.001, 1.0);
  const attackMs  = Math.max(0, shape.attack_time ?? 0);
  const decayMs   = Math.max(0, shape.decay_time  ?? 0);
  const predelay  = Math.max(0, shape.predelay    ?? 0);
  const reverse   = !!shape.reverse;
  const density   = clamp(shape.density   ?? 1.0, 0.05, 4.0);

  // Density: stride through the source. >1 stretches (resamples up), <1 packs
  // (resamples down — denser grain). We keep it cheap (linear interp).
  const densityLen = Math.max(1, Math.floor(srcLen / density));

  // Apply length truncation to the density-resampled length
  const shapedLen = Math.max(1, Math.floor(densityLen * length));

  // Predelay (samples) prepended in front of the shaped tail
  const predelaySamps = Math.floor((predelay / 1000) * sr);
  const totalLen = shapedLen + predelaySamps;

  const out = ctx.createBuffer(channels, totalLen, sr);

  const attackSamps = Math.min(shapedLen, Math.floor((attackMs / 1000) * sr));
  // Decay envelope: exponential from 1 → ~ -60 dB over `decay_time` ms.
  // When decay_time=0 the envelope is disabled (shape stays flat post-attack)
  // — this matches Space Designer's "no decay shaping" behaviour where the
  // raw IR length / length-truncation alone determines the tail.
  const decayActive = decayMs > 0;
  const decaySamps = decayActive
    ? Math.min(shapedLen, Math.floor((decayMs / 1000) * sr))
    : 0;
  // Choose a decay coefficient so the envelope reaches ≈ -60 dB over decaySamps
  // (Sabine-style RT60 mapping).
  const decayCoef = decayActive && decaySamps > 0 ? (Math.log(1000) / decaySamps) : 0;

  for (let ch = 0; ch < channels; ch++) {
    const srcData = sourceBuffer.getChannelData(Math.min(ch, channels - 1));
    const outData = out.getChannelData(ch);
    // outData starts as zeros (from createBuffer) — predelay region is silence.
    for (let i = 0; i < shapedLen; i++) {
      // 1) density-resample with linear interpolation
      let srcIdx = i * density;
      // 2) reverse if requested (read from the tail)
      if (reverse) srcIdx = (srcLen - 1) - srcIdx;
      const i0 = Math.floor(srcIdx);
      const i1 = Math.min(srcLen - 1, Math.max(0, i0 + (reverse ? -1 : 1)));
      const frac = srcIdx - Math.floor(srcIdx);
      const a = srcData[Math.max(0, Math.min(srcLen - 1, i0))] || 0;
      const b = srcData[Math.max(0, Math.min(srcLen - 1, i1))] || 0;
      let s = a + (b - a) * (reverse ? -frac : frac);

      // 3) attack envelope (linear fade-in)
      if (attackSamps > 0 && i < attackSamps) {
        s *= i / attackSamps;
      }
      // 4) decay envelope (exponential fade-out over decaySamps)
      if (decayActive) {
        if (i < decaySamps) {
          s *= Math.exp(-decayCoef * i);
        } else {
          // past the explicit decay window: continue at the floor
          s *= Math.exp(-decayCoef * decaySamps);
        }
      }

      outData[predelaySamps + i] = s;
    }
  }
  return out;
}

// ── convolution_sd builder ────────────────────────────────────────────────
//
// Schema:
//   {
//     type: 'convolution_sd',
//     params: {
//       length:      0..1   | '@<id>',
//       attack_time: ms     | '@<id>',
//       decay_time:  ms     | '@<id>',
//       predelay:    ms     | '@<id>',
//       low_cut:     Hz     | '@<id>',
//       high_cut:    Hz     | '@<id>',
//       mix:         0..1   | '@<id>',
//       reverse:     bool   | '@<id>',
//       density:     0..1   | '@<id>',
//       ir_url:      string (URL/Blob/ArrayBuffer; optional — falls back to default IR)
//     }
//   }
export function buildConvolutionSD(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const input  = ctx.createGain();
  const output = ctx.createGain();
  const dryGain = ctx.createGain();
  const wetGain = ctx.createGain();

  // Pre-IR EQ stage: low_cut highpass → high_cut lowpass → convolver
  const lowCut  = ctx.createBiquadFilter();
  const highCut = ctx.createBiquadFilter();
  lowCut.type  = 'highpass';
  highCut.type = 'lowpass';
  lowCut.frequency.value  = (typeof params.low_cut  === 'number') ? params.low_cut  : 20;
  highCut.frequency.value = (typeof params.high_cut === 'number') ? params.high_cut : 20000;
  lowCut.Q.value = 0.707;
  highCut.Q.value = 0.707;

  const conv = ctx.createConvolver();
  conv.normalize = true;

  // Initial dry/wet
  const initialMix = (typeof params.mix === 'number') ? params.mix : 0.3;
  wetGain.gain.value = clamp(initialMix, 0, 1);
  dryGain.gain.value = 1 - clamp(initialMix, 0, 1);

  // Wire: input → dryGain → output
  //       input → lowCut → highCut → conv → wetGain → output
  input.connect(dryGain);
  dryGain.connect(output);
  input.connect(lowCut);
  lowCut.connect(highCut);
  highCut.connect(conv);
  conv.connect(wetGain);
  wetGain.connect(output);

  // ── IR state ────────────────────────────────────────────────────────────
  // sourceBuffer is the raw, unmodified IR. shape{} holds current params.
  // Whenever any shape param changes we run shapeIR(...) and reassign
  // conv.buffer. We synthesise a default source IR up front so the node
  // makes audible reverb without an external file.
  let sourceBuffer = null;
  try {
    sourceBuffer = makeDefaultSourceIR(ctx, 2.5, 3.0);
  } catch (e) { /* ctx may be closed in tests */ }

  const shape = {
    length:      (typeof params.length      === 'number') ? params.length      : 1.0,
    attack_time: (typeof params.attack_time === 'number') ? params.attack_time : 0,
    decay_time:  (typeof params.decay_time  === 'number') ? params.decay_time  : 0,
    predelay:    (typeof params.predelay    === 'number') ? params.predelay    : 0,
    reverse:     !!params.reverse,
    density:     (typeof params.density     === 'number') ? params.density     : 1.0,
  };

  // Debounce IR rebuilds so a rapid knob drag (50+ events/sec) doesn't thrash
  // the audio thread. setTimeout(0) coalesces all events from one tick.
  let _rebuildScheduled = false;
  const scheduleRebuild = () => {
    if (_rebuildScheduled) return;
    _rebuildScheduled = true;
    const fire = () => {
      _rebuildScheduled = false;
      try {
        if (sourceBuffer) {
          conv.buffer = shapeIR(ctx, sourceBuffer, shape);
        }
      } catch (e) {
        // eslint-disable-next-line no-console
        if (typeof console !== 'undefined') console.warn('[R13] IR rebuild failed', e);
      }
    };
    if (typeof queueMicrotask === 'function') queueMicrotask(fire);
    else if (typeof setTimeout === 'function') setTimeout(fire, 0);
    else fire();
  };

  // Synchronous rebuild — used by tests and the initial assignment so
  // conv.buffer is set before the engine pulls the first audio frame.
  const rebuildNow = () => {
    try {
      if (sourceBuffer) conv.buffer = shapeIR(ctx, sourceBuffer, shape);
    } catch (e) { /* ignore */ }
  };
  rebuildNow();

  // IR loader — supports raw URL strings, Blobs, ArrayBuffers, or {url}.
  // Sets `sourceBuffer` and fires a rebuild. The user-visible IR (conv.buffer)
  // is the SHAPED version, never the raw source.
  const loadIR = async (src) => {
    if (!src) return false;
    try {
      let arrayBuffer;
      if (src instanceof ArrayBuffer) {
        arrayBuffer = src;
      } else if (typeof Blob !== 'undefined' && src instanceof Blob) {
        arrayBuffer = await src.arrayBuffer();
      } else if (typeof src === 'string') {
        const resp = await fetch(src);
        arrayBuffer = await resp.arrayBuffer();
      } else if (src && typeof src.url === 'string') {
        const resp = await fetch(src.url);
        arrayBuffer = await resp.arrayBuffer();
      } else if (src && src.arrayBuffer) {
        arrayBuffer = await src.arrayBuffer();
      } else {
        return false;
      }
      const decoded = await ctx.decodeAudioData(arrayBuffer);
      sourceBuffer = decoded;
      rebuildNow();
      return true;
    } catch (err) {
      // eslint-disable-next-line no-console
      if (typeof console !== 'undefined') console.warn('[R13] IR load failed', err);
      return false;
    }
  };

  // ── Param wiring ────────────────────────────────────────────────────────

  // Helper: bind an IR-shape param to a customSetter that rewrites `shape`
  // and schedules a rebuild.
  const bindShapeParam = (paramId, paramDef, key) => {
    targets[paramId] = {
      paramDef,
      customSetter: (val) => {
        shape[key] = (key === 'reverse') ? !!val : val;
        scheduleRebuild();
      },
    };
  };

  for (const [k, v] of Object.entries(params)) {
    if (isAtBinding(v)) {
      const id = v.slice(1);
      const pd = paramDefs ? paramDefs[id] : undefined;
      switch (k) {
        case 'length':       bindShapeParam(id, pd, 'length'); break;
        case 'attack_time':  bindShapeParam(id, pd, 'attack_time'); break;
        case 'decay_time':   bindShapeParam(id, pd, 'decay_time'); break;
        case 'predelay':     bindShapeParam(id, pd, 'predelay'); break;
        case 'reverse':      bindShapeParam(id, pd, 'reverse'); break;
        case 'density':      bindShapeParam(id, pd, 'density'); break;
        case 'low_cut':
          targets[id] = { audioParam: lowCut.frequency, paramDef: pd };
          break;
        case 'high_cut':
          targets[id] = { audioParam: highCut.frequency, paramDef: pd };
          break;
        case 'mix':
        case 'wet':
          targets[id] = {
            paramDef: pd,
            customSetter: (val) => {
              const w = clamp(val, 0, 1);
              wetGain.gain.value = w;
              dryGain.gain.value = 1 - w;
            },
          };
          break;
        case 'ir_file':
        case 'ir_url':
        case 'impulse':
          targets[id] = { paramDef: pd, customSetter: (val) => { loadIR(val); } };
          break;
        default:
          break;
      }
    } else {
      // Literal value — already absorbed into shape{} / lowCut / highCut /
      // wetGain. Handle ir_url (string literal) here.
      if ((k === 'ir_file' || k === 'ir_url' || k === 'impulse') && v) {
        loadIR(v);
      }
    }
  }

  // Worklet upgrade hook — if the partitioned-FFT processor ships in the
  // future, callers can opt-in via `params.use_worklet=true` and we route
  // input through it instead. Today this is a no-op: we never construct it.
  // Kept here so downstream code can probe `node._r13_worklet_name`.
  // eslint-disable-next-line no-unused-vars
  const _workletHookName = R13_WORKLET_NAME;

  return {
    input,
    output,
    paramTargets: targets,
    loadIR,
    // Expose internals for tests + engine introspection. The engine doesn't
    // reach in — only the test harness does.
    _convolver: conv,
    _shape: shape,
    _rebuildNow: rebuildNow,
    _setSourceIR: (buf) => { sourceBuffer = buf; rebuildNow(); },
    _getSourceIR: () => sourceBuffer,
  };
}

// ── Default export: NODE_BUILDERS map ─────────────────────────────────────

const R13_BUILDERS = {
  convolution_sd: buildConvolutionSD,
};

export default R13_BUILDERS;

// Also export the helpers so tests can poke at them directly without having
// to drive the whole builder.
export { makeDefaultSourceIR, shapeIR };
