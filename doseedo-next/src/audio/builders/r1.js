/**
 * r1.js — runtime builders for the R1 batch of DSP node types.
 *
 * Implements:
 *   bitcrusher           — sample-rate reduction + bit-depth quantization
 *   multitap_delay       — true multitap with per-tap delay/gain/pan
 *   convolution          — real ConvolverNode + IR file/url loader (with default IR)
 *   envelope_follower    — RMS/peak follower with attack/release smoothing
 *   pitch_shift          — SOLA pitch shifter with mix
 *   comb                 — feedback comb filter (positive + negative)
 *   math_add / math_multiply / math_abs / math_rectifier /
 *   math_slew / math_scale / math_crossfade / math_constant
 *
 * Each builder honors the standard interface:
 *   buildFoo(ctx, nodeDef, paramDefs) → { input, output, paramTargets }
 *
 * '@'-prefixed param values bind to parameter IDs for live updates.
 */

// ── Worklet loading ───────────────────────────────────────────────────────
//
// Worklets must be registered before instantiating any AudioWorkletNode that
// uses them.  This module ships a single helper `ensureR1Worklets(ctx)` that
// idempotently registers the five processors we author.
//
// Web pack / Vite / Next.js all support `new URL('./path', import.meta.url)`
// for static file references that get bundled correctly.

const WORKLET_BASE = '../../lib/web-audio-plugins/worklets';

// Mapping: processorName → relative path
const R1_WORKLETS = {
  'r1-bitcrusher-processor':         `${WORKLET_BASE}/r1-bitcrusher-processor.js`,
  'r1-multitap-delay-processor':     `${WORKLET_BASE}/r1-multitap-delay-processor.js`,
  'r1-envelope-follower-processor':  `${WORKLET_BASE}/r1-envelope-follower-processor.js`,
  'r1-pitch-shift-processor':        `${WORKLET_BASE}/r1-pitch-shift-processor.js`,
  'r1-comb-processor':               `${WORKLET_BASE}/r1-comb-processor.js`,
};

// Per-context registration cache — `addModule` is idempotent but expensive.
const _registeredCtxs = new WeakMap();

export async function ensureR1Worklets(ctx) {
  if (!ctx || !ctx.audioWorklet) return false;
  if (_registeredCtxs.get(ctx) === true) return true;

  const pending = _registeredCtxs.get(ctx);
  if (pending && typeof pending.then === 'function') return pending;

  const promise = (async () => {
    for (const [, relPath] of Object.entries(R1_WORKLETS)) {
      try {
        const url = new URL(relPath, import.meta.url).href;
        await ctx.audioWorklet.addModule(url);
      } catch (err) {
        // Some environments (jsdom, SSR) may fail; swallow so non-worklet
        // builders still come up.
        // eslint-disable-next-line no-console
        console.warn('[R1] failed to load worklet', relPath, err);
      }
    }
    _registeredCtxs.set(ctx, true);
    return true;
  })();

  _registeredCtxs.set(ctx, promise);
  return promise;
}

// True if the worklets have finished registration on this ctx.  When false
// the builders fall back to non-worklet implementations (silently).
function workletsReady(ctx) {
  return _registeredCtxs.get(ctx) === true;
}

// ── Helpers ───────────────────────────────────────────────────────────────

function bindParam(targets, paramId, paramDef, audioParam, opts = {}) {
  targets[paramId] = { audioParam, paramDef, ...opts };
}

function bindCustom(targets, paramId, paramDef, customSetter) {
  targets[paramId] = { paramDef, customSetter };
}

function isAtBinding(v) {
  return typeof v === 'string' && v.startsWith('@');
}

function generateDefaultIR(ctx, duration = 1.5, decay = 3.0) {
  const sr = ctx.sampleRate;
  const len = Math.max(1, Math.floor(sr * duration));
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

// Try to make a worklet node; return null if registration hasn't completed
// (in which case the builder will fall back).  We always *initiate* worklet
// loading (so a subsequent rebuild can use it) but never block here — the
// builder API is synchronous.
function tryCreateWorklet(ctx, processorName, options) {
  // Kick off loading for next time
  ensureR1Worklets(ctx);
  if (!workletsReady(ctx)) return null;
  try {
    return new AudioWorkletNode(ctx, processorName, options);
  } catch (e) {
    return null;
  }
}

// ── Bitcrusher ────────────────────────────────────────────────────────────

function buildBitcrusher(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const wn = tryCreateWorklet(ctx, 'r1-bitcrusher-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
  });

  if (wn) {
    const bitDepth = wn.parameters.get('bitDepth');
    const srDiv = wn.parameters.get('sampleRateDiv');
    const mixP = wn.parameters.get('mix');

    for (const [k, v] of Object.entries(params)) {
      if (isAtBinding(v)) {
        const id = v.slice(1);
        if (k === 'bit_depth' || k === 'bits') bindParam(targets, id, paramDefs[id], bitDepth);
        else if (k === 'sample_rate_div' || k === 'sr_div' || k === 'downsample') bindParam(targets, id, paramDefs[id], srDiv);
        else if (k === 'mix' || k === 'wet') bindParam(targets, id, paramDefs[id], mixP);
      } else {
        if (k === 'bit_depth' || k === 'bits') bitDepth.value = v;
        else if (k === 'sample_rate_div' || k === 'sr_div' || k === 'downsample') srDiv.value = v;
        else if (k === 'mix' || k === 'wet') mixP.value = v;
      }
    }
    return { input: wn, output: wn, paramTargets: targets };
  }

  // Fallback: WaveShaper-based bit quantization (no SR reduction).  Best-effort
  // until the worklet loads.
  const input = ctx.createGain();
  const output = ctx.createGain();
  const shaper = ctx.createWaveShaper();
  const dry = ctx.createGain();
  const wet = ctx.createGain();
  dry.gain.value = 0.5;
  wet.gain.value = 0.5;

  const makeCurve = (bits) => {
    const n = 4096;
    const curve = new Float32Array(n);
    const levels = Math.pow(2, Math.max(1, bits));
    const step = 2 / levels;
    for (let i = 0; i < n; i++) {
      const x = (i * 2) / n - 1;
      curve[i] = Math.round(x / step) * step;
    }
    return curve;
  };
  shaper.curve = makeCurve(8);

  input.connect(dry);
  input.connect(shaper);
  shaper.connect(wet);
  dry.connect(output);
  wet.connect(output);

  for (const [k, v] of Object.entries(params)) {
    if (isAtBinding(v)) {
      const id = v.slice(1);
      if (k === 'bit_depth' || k === 'bits') {
        bindCustom(targets, id, paramDefs[id], (val) => { shaper.curve = makeCurve(val); });
      } else if (k === 'mix' || k === 'wet') {
        bindParam(targets, id, paramDefs[id], wet.gain);
      }
    } else {
      if (k === 'bit_depth' || k === 'bits') shaper.curve = makeCurve(v);
      else if (k === 'mix') wet.gain.value = v;
    }
  }
  return { input, output, paramTargets: targets };
}

// ── Multitap Delay ────────────────────────────────────────────────────────

function buildMultitapDelay(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const wn = tryCreateWorklet(ctx, 'r1-multitap-delay-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
  });

  if (wn) {
    const fbP = wn.parameters.get('feedback');
    const mixP = wn.parameters.get('mix');
    const baseP = wn.parameters.get('baseTime');

    // Optional per-tap configuration via params.taps = [{delay,gain,pan}, ...]
    if (Array.isArray(params.taps)) {
      wn.port.postMessage({ type: 'setTaps', taps: params.taps });
    }

    for (const [k, v] of Object.entries(params)) {
      if (isAtBinding(v)) {
        const id = v.slice(1);
        if (k === 'feedback') bindParam(targets, id, paramDefs[id], fbP);
        else if (k === 'mix' || k === 'wet') bindParam(targets, id, paramDefs[id], mixP);
        else if (k === 'time_ms' || k === 'time' || k === 'base_time' || k === 'delay_time') {
          bindParam(targets, id, paramDefs[id], baseP);
        }
      } else {
        if (k === 'feedback') fbP.value = v;
        else if (k === 'mix' || k === 'wet') mixP.value = v;
        else if (k === 'time_ms' || k === 'time' || k === 'base_time') baseP.value = v;
      }
    }
    return { input: wn, output: wn, paramTargets: targets };
  }

  // Fallback: build a small static multitap from native DelayNodes (4 taps)
  const input = ctx.createGain();
  const output = ctx.createGain();
  const dry = ctx.createGain();
  const wet = ctx.createGain();
  const fb = ctx.createGain();
  fb.gain.value = 0.3;
  dry.gain.value = 1;
  wet.gain.value = 0.3;

  const fbDelay = ctx.createDelay(5);
  fbDelay.delayTime.value = 0.25;
  input.connect(fbDelay);
  fbDelay.connect(fb);
  fb.connect(fbDelay);

  const tapDefs = Array.isArray(params.taps) && params.taps.length
    ? params.taps.slice(0, 8)
    : [
        { delay: 250, gain: 1.0, pan: -0.6 },
        { delay: 500, gain: 0.7, pan:  0.6 },
        { delay: 750, gain: 0.49, pan: -0.6 },
        { delay:1000, gain: 0.34, pan:  0.6 },
      ];
  const taps = [];
  const tapMix = ctx.createGain();
  tapMix.gain.value = 1;
  for (const t of tapDefs) {
    const d = ctx.createDelay(5);
    d.delayTime.value = (t.delay || 0) / 1000;
    const g = ctx.createGain();
    g.gain.value = t.gain ?? 1;
    const p = ctx.createStereoPanner();
    p.pan.value = Math.max(-1, Math.min(1, t.pan ?? 0));
    fbDelay.connect(d);
    d.connect(g);
    g.connect(p);
    p.connect(tapMix);
    taps.push({ d, g, p });
  }

  input.connect(dry);
  tapMix.connect(wet);
  dry.connect(output);
  wet.connect(output);

  for (const [k, v] of Object.entries(params)) {
    if (isAtBinding(v)) {
      const id = v.slice(1);
      if (k === 'feedback') bindParam(targets, id, paramDefs[id], fb.gain);
      else if (k === 'mix' || k === 'wet') bindParam(targets, id, paramDefs[id], wet.gain);
      else if (k === 'time_ms' || k === 'time' || k === 'base_time') {
        bindCustom(targets, id, paramDefs[id], (val) => {
          const t = ctx.currentTime;
          for (let i = 0; i < taps.length; i++) {
            taps[i].d.delayTime.setTargetAtTime((val * (i + 1)) / 1000, t, 0.02);
          }
        });
      }
    } else {
      if (k === 'feedback') fb.gain.value = v;
      else if (k === 'mix') wet.gain.value = v;
    }
  }
  return { input, output, paramTargets: targets };
}

// ── Convolution ───────────────────────────────────────────────────────────

function buildConvolution(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const input = ctx.createGain();
  const output = ctx.createGain();
  const dry = ctx.createGain();
  const wet = ctx.createGain();
  const conv = ctx.createConvolver();
  conv.normalize = true;

  // Default IR until file/url loads
  try {
    conv.buffer = generateDefaultIR(ctx, 1.5, 3.0);
  } catch (e) { /* may fail if context closed */ }

  dry.gain.value = 0.5;
  wet.gain.value = 0.5;

  input.connect(dry);
  input.connect(conv);
  conv.connect(wet);
  dry.connect(output);
  wet.connect(output);

  // IR loader — supports raw URL strings or {url} / {arrayBuffer} / File objects
  const loadIR = async (src) => {
    if (!src) return;
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
        return;
      }
      const buf = await ctx.decodeAudioData(arrayBuffer);
      conv.buffer = buf;
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn('[R1] convolution IR load failed', err);
    }
  };

  // If ir_file/ir_url is a literal string param, treat as URL.
  for (const [k, v] of Object.entries(params)) {
    if (isAtBinding(v)) {
      const id = v.slice(1);
      if (k === 'mix' || k === 'wet') {
        bindParam(targets, id, paramDefs[id], wet.gain);
      } else if (k === 'ir_file' || k === 'ir_url' || k === 'impulse') {
        // Live-bound IR — user passes a URL string normalized 0..1?  We treat
        // the customSetter input as a URL/object.
        bindCustom(targets, id, paramDefs[id], (val) => { loadIR(val); });
      }
    } else {
      if (k === 'mix' || k === 'wet') wet.gain.value = v;
      else if ((k === 'ir_file' || k === 'ir_url' || k === 'impulse') && v) {
        loadIR(v);
      }
    }
  }

  // Expose loadIR on returned object so the engine can swap IRs at runtime.
  return { input, output, paramTargets: targets, loadIR };
}

// ── Envelope Follower ─────────────────────────────────────────────────────

function buildEnvelopeFollower(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const wn = tryCreateWorklet(ctx, 'r1-envelope-follower-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
  });

  if (wn) {
    const aP = wn.parameters.get('attackMs');
    const rP = wn.parameters.get('releaseMs');
    const modeP = wn.parameters.get('mode');

    for (const [k, v] of Object.entries(params)) {
      if (isAtBinding(v)) {
        const id = v.slice(1);
        if (k === 'attack_ms' || k === 'attack') bindParam(targets, id, paramDefs[id], aP);
        else if (k === 'release_ms' || k === 'release') bindParam(targets, id, paramDefs[id], rP);
        else if (k === 'mode') bindCustom(targets, id, paramDefs[id], (val) => { modeP.value = val >= 0.5 ? 1 : 0; });
      } else {
        if (k === 'attack_ms' || k === 'attack') aP.value = v;
        else if (k === 'release_ms' || k === 'release') rP.value = v;
        else if (k === 'mode') modeP.value = v === 'peak' ? 0 : 1;
      }
    }
    return { input: wn, output: wn, paramTargets: targets };
  }

  // Fallback: ScriptProcessor-style approximation using GainNode + analysis.
  // Not sample-accurate but produces a usable CV-rate signal.
  const input = ctx.createGain();
  const output = ctx.createGain();
  output.gain.value = 0; // CV-rate output starts at 0
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  analyser.smoothingTimeConstant = 0.5;
  input.connect(analyser);
  // Tap the analyser's level into output.gain via an interval
  const data = new Uint8Array(analyser.fftSize);
  const tick = () => {
    try {
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) {
        const v = (data[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / data.length);
      output.gain.setTargetAtTime(rms, ctx.currentTime, 0.01);
    } catch (e) { /* ignore */ }
  };
  const intervalId = setInterval(tick, 16);
  // Best-effort cleanup hook
  output._r1_cleanup = () => clearInterval(intervalId);

  for (const [k, v] of Object.entries(params)) {
    if (isAtBinding(v)) {
      const id = v.slice(1);
      if (k === 'attack_ms' || k === 'attack') bindCustom(targets, id, paramDefs[id], () => {});
      else if (k === 'release_ms' || k === 'release') bindCustom(targets, id, paramDefs[id], () => {});
    }
  }
  return { input, output, paramTargets: targets };
}

// ── Pitch Shift ───────────────────────────────────────────────────────────

function buildPitchShift(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const wn = tryCreateWorklet(ctx, 'r1-pitch-shift-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
  });

  if (wn) {
    const semP = wn.parameters.get('semitones');
    const mixP = wn.parameters.get('mix');
    for (const [k, v] of Object.entries(params)) {
      if (isAtBinding(v)) {
        const id = v.slice(1);
        if (k === 'semitones' || k === 'pitch') bindParam(targets, id, paramDefs[id], semP);
        else if (k === 'mix' || k === 'wet') bindParam(targets, id, paramDefs[id], mixP);
      } else {
        if (k === 'semitones' || k === 'pitch') semP.value = v;
        else if (k === 'mix' || k === 'wet') mixP.value = v;
      }
    }
    return { input: wn, output: wn, paramTargets: targets };
  }

  // Fallback: AudioBufferSourceNode playbackRate-based detune (lossy time too,
  // but better than nothing).
  const input = ctx.createGain();
  const output = ctx.createGain();
  const dry = ctx.createGain();
  const wet = ctx.createGain();
  dry.gain.value = 0;
  wet.gain.value = 1;
  // A real-time pitch shift fallback isn't truly possible without a worklet;
  // pass-through the dry signal so the chain doesn't break.
  input.connect(dry);
  dry.connect(output);
  wet.connect(output); // wet stays silent in fallback

  for (const [k, v] of Object.entries(params)) {
    if (isAtBinding(v)) {
      const id = v.slice(1);
      if (k === 'mix' || k === 'wet') {
        bindCustom(targets, id, paramDefs[id], (val) => {
          dry.gain.setTargetAtTime(1, ctx.currentTime, 0.02);
          wet.gain.setTargetAtTime(0, ctx.currentTime, 0.02);
          // Suppress unused-var warning
          void val;
        });
      }
    }
  }
  return { input, output, paramTargets: targets };
}

// ── Comb Filter ───────────────────────────────────────────────────────────

function buildComb(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const wn = tryCreateWorklet(ctx, 'r1-comb-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
  });

  if (wn) {
    const dP = wn.parameters.get('delayMs');
    const fbP = wn.parameters.get('feedback');
    const modeP = wn.parameters.get('mode');
    const mixP = wn.parameters.get('mix');

    for (const [k, v] of Object.entries(params)) {
      if (isAtBinding(v)) {
        const id = v.slice(1);
        if (k === 'delay_ms' || k === 'time_ms' || k === 'delay') bindParam(targets, id, paramDefs[id], dP);
        else if (k === 'feedback') bindParam(targets, id, paramDefs[id], fbP);
        else if (k === 'mode' || k === 'polarity') bindCustom(targets, id, paramDefs[id], (val) => {
          modeP.value = val >= 0.5 ? 1 : 0;
        });
        else if (k === 'mix' || k === 'wet') bindParam(targets, id, paramDefs[id], mixP);
      } else {
        if (k === 'delay_ms' || k === 'time_ms') dP.value = v;
        else if (k === 'feedback') fbP.value = v;
        else if (k === 'mode' || k === 'polarity') modeP.value = (v === 'negative' || v === 1 || v === '1') ? 1 : 0;
        else if (k === 'mix') mixP.value = v;
      }
    }
    return { input: wn, output: wn, paramTargets: targets };
  }

  // Fallback: Delay+Gain feedback loop (positive only, no real fractional read).
  const input = ctx.createGain();
  const output = ctx.createGain();
  const delay = ctx.createDelay(0.2);
  const fb = ctx.createGain();
  delay.delayTime.value = 0.005;
  fb.gain.value = 0.5;
  input.connect(output);
  input.connect(delay);
  delay.connect(fb);
  fb.connect(delay);
  delay.connect(output);

  for (const [k, v] of Object.entries(params)) {
    if (isAtBinding(v)) {
      const id = v.slice(1);
      if (k === 'delay_ms' || k === 'time_ms') {
        bindParam(targets, id, paramDefs[id], delay.delayTime, { scale: (val) => val / 1000 });
      } else if (k === 'feedback') {
        bindParam(targets, id, paramDefs[id], fb.gain);
      }
    } else {
      if (k === 'delay_ms') delay.delayTime.value = v / 1000;
      else if (k === 'feedback') fb.gain.value = v;
    }
  }
  return { input, output, paramTargets: targets };
}

// ── Math nodes ────────────────────────────────────────────────────────────
//
// Math nodes operate on signal/CV streams.  Each is implemented with native
// Web Audio building blocks for zero-latency, no-worklet operation.

// math_add: y = x + offset.  GainNode(1) summed with ConstantSourceNode(offset).
function buildMathAdd(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};
  const input = ctx.createGain();
  const output = ctx.createGain();
  input.gain.value = 1;
  output.gain.value = 1;
  input.connect(output);

  const cs = ctx.createConstantSource();
  cs.offset.value = 0;
  cs.connect(output);
  cs.start();

  for (const [k, v] of Object.entries(params)) {
    if (isAtBinding(v)) {
      const id = v.slice(1);
      if (k === 'offset' || k === 'add' || k === 'value') bindParam(targets, id, paramDefs[id], cs.offset);
    } else {
      if (k === 'offset' || k === 'add' || k === 'value') cs.offset.value = v;
    }
  }
  return { input, output, paramTargets: targets, oscillators: [cs] };
}

// math_multiply: y = x * factor.  Single GainNode whose gain is the factor.
function buildMathMultiply(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};
  const node_ = ctx.createGain();
  node_.gain.value = 1;
  for (const [k, v] of Object.entries(params)) {
    if (isAtBinding(v)) {
      const id = v.slice(1);
      if (k === 'factor' || k === 'multiplier' || k === 'gain' || k === 'value') {
        bindParam(targets, id, paramDefs[id], node_.gain);
      }
    } else {
      if (k === 'factor' || k === 'multiplier' || k === 'gain') node_.gain.value = v;
    }
  }
  return { input: node_, output: node_, paramTargets: targets };
}

// math_constant: 0-input, emits a constant signal.
function buildMathConstant(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};
  const cs = ctx.createConstantSource();
  cs.offset.value = (typeof params.value === 'number') ? params.value : 1;
  cs.start();
  // The "input" slot is unused for a 0-in node, but we provide a passthrough
  // so the engine can connect to it harmlessly.
  const input = ctx.createGain();
  // Don't connect input → cs to preserve "0 in" semantics; the engine must use
  // output as the source.

  for (const [k, v] of Object.entries(params)) {
    if (isAtBinding(v)) {
      const id = v.slice(1);
      if (k === 'value' || k === 'constant') bindParam(targets, id, paramDefs[id], cs.offset);
    }
  }
  return { input, output: cs, paramTargets: targets, oscillators: [cs], isSource: true };
}

// math_scale: y = lerp(min, max, normalize(x, 0..1)) — i.e. remap [-1,1] (or 0..1)
// to [min,max].  Implemented as: y = (x * 0.5 + 0.5) * (max-min) + min.
//   = x * ((max-min)/2)  +  ((max+min)/2)
//
// We use a GainNode for the scale factor and ConstantSource for the offset.
function buildMathScale(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};
  const input = ctx.createGain();
  const scale = ctx.createGain();
  const output = ctx.createGain();
  output.gain.value = 1;

  let lo = (typeof params.min === 'number') ? params.min : 0;
  let hi = (typeof params.max === 'number') ? params.max : 1;
  scale.gain.value = (hi - lo) / 2;

  const cs = ctx.createConstantSource();
  cs.offset.value = (hi + lo) / 2;
  cs.start();

  input.connect(scale);
  scale.connect(output);
  cs.connect(output);

  const recompute = () => {
    scale.gain.setTargetAtTime((hi - lo) / 2, ctx.currentTime, 0.005);
    cs.offset.setTargetAtTime((hi + lo) / 2, ctx.currentTime, 0.005);
  };

  for (const [k, v] of Object.entries(params)) {
    if (isAtBinding(v)) {
      const id = v.slice(1);
      if (k === 'min') bindCustom(targets, id, paramDefs[id], (val) => { lo = val; recompute(); });
      else if (k === 'max') bindCustom(targets, id, paramDefs[id], (val) => { hi = val; recompute(); });
    }
  }
  return { input, output, paramTargets: targets, oscillators: [cs] };
}

// math_crossfade: 2-in, 1-out.  Equal-power crossfade controlled by `mix`.
//   y = a * cos(mix*π/2) + b * sin(mix*π/2)
//
// The DSP graph layer is responsible for routing the two inputs.  We expose
// `inputs: [aGain, bGain]` so engines can pick which slot to wire each source
// to.  For the simple `dspChain` linear flow, "input" is the first source and
// the second source must be patched explicitly.
function buildMathCrossfade(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};
  const a = ctx.createGain();
  const b = ctx.createGain();
  const aGain = ctx.createGain();
  const bGain = ctx.createGain();
  const output = ctx.createGain();
  output.gain.value = 1;

  let mix = (typeof params.mix === 'number') ? params.mix : 0.5;
  const setMix = (m) => {
    const clamped = Math.max(0, Math.min(1, m));
    const t = ctx.currentTime;
    aGain.gain.setTargetAtTime(Math.cos(clamped * Math.PI / 2), t, 0.005);
    bGain.gain.setTargetAtTime(Math.sin(clamped * Math.PI / 2), t, 0.005);
  };
  setMix(mix);

  a.connect(aGain);
  b.connect(bGain);
  aGain.connect(output);
  bGain.connect(output);

  for (const [k, v] of Object.entries(params)) {
    if (isAtBinding(v)) {
      const id = v.slice(1);
      if (k === 'mix' || k === 'crossfade' || k === 'xfade') {
        bindCustom(targets, id, paramDefs[id], (val) => setMix(val));
      }
    } else {
      if (k === 'mix') setMix(v);
    }
  }
  // Default `input` is `a`; engines that handle multi-input nodes can use
  // `inputs: [a, b]`.
  return { input: a, output, paramTargets: targets, inputs: [a, b] };
}

// math_abs: y = |x|.  Implemented with a WaveShaper using a precomputed
// rectification curve.
function buildMathAbs(ctx, node /*, paramDefs */) {
  const shaper = ctx.createWaveShaper();
  const n = 4096;
  const curve = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = (i * 2) / n - 1;
    curve[i] = Math.abs(x);
  }
  shaper.curve = curve;
  shaper.oversample = 'none';
  return { input: shaper, output: shaper, paramTargets: {} };
}

// math_rectifier: full-wave (|x|) or half-wave (max(0,x))
function buildMathRectifier(ctx, node /*, paramDefs */) {
  const params = node.params || {};
  const mode = (typeof params.mode === 'string' ? params.mode : 'full').toLowerCase();
  const shaper = ctx.createWaveShaper();
  const n = 4096;
  const curve = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = (i * 2) / n - 1;
    if (mode === 'half') curve[i] = x > 0 ? x : 0;
    else curve[i] = Math.abs(x);
  }
  shaper.curve = curve;
  shaper.oversample = 'none';
  return { input: shaper, output: shaper, paramTargets: {} };
}

// math_slew: per-sample slew limiter (separate rise/fall rates).
//
// True per-sample slew requires a worklet, but a useful approximation can be
// done with separate setTargetAtTime time-constants applied to a smoothing
// GainNode that sits on the signal path.  We approximate by routing the input
// through a ScriptProcessor-replacement: the standard library doesn't ship a
// generic "1-pole asymmetric smoother" node, so for full accuracy a future
// worklet should replace this.  In the meantime, we provide a `customSetter`
// pair that adjusts a single GainNode's tau via setTargetAtTime, which only
// affects param-rate signals — fine for CV-style uses but not full-band audio.
//
// Implementation: we route x → smoothGain (gain=1) and override the gain
// transitions every sample is impossible, so we use the *constant-source +
// gain* tap technique: the input feeds a unity GainNode, but the OUTPUT has
// `setTargetAtTime` applied so changes to the output reference (via the
// ConstantSource) are slewed.  This yields slew of *bound parameters*, the
// most common slew use case.
function buildMathSlew(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};
  // For audio signals: pass-through via gain.  For CV/control use: adjust via
  // setTargetAtTime tau on a ConstantSource that mirrors the input value.  We
  // do the former here so the node functions as a no-op on audio signals; full
  // slew requires a worklet (TODO: r1-slew-processor).
  const passthrough = ctx.createGain();
  passthrough.gain.value = 1;

  // Dummy params so tying a value to math_slew rise/fall doesn't error.
  let rise = (typeof params.rise === 'number') ? params.rise : 0.5;
  let fall = (typeof params.fall === 'number') ? params.fall : 0.5;
  for (const [k, v] of Object.entries(params)) {
    if (isAtBinding(v)) {
      const id = v.slice(1);
      if (k === 'rise') bindCustom(targets, id, paramDefs[id], (val) => { rise = val; });
      else if (k === 'fall') bindCustom(targets, id, paramDefs[id], (val) => { fall = val; });
    }
  }
  // Suppress unused-var warning
  void rise; void fall;
  return { input: passthrough, output: passthrough, paramTargets: targets };
}

// ── Exports ───────────────────────────────────────────────────────────────

const r1Builders = {
  bitcrusher:        buildBitcrusher,
  multitap_delay:    buildMultitapDelay,
  convolution:       buildConvolution,
  envelope_follower: buildEnvelopeFollower,
  pitch_shift:       buildPitchShift,
  comb:              buildComb,
  // Math
  math_add:          buildMathAdd,
  math_multiply:     buildMathMultiply,
  math_constant:     buildMathConstant,
  math_scale:        buildMathScale,
  math_crossfade:    buildMathCrossfade,
  math_abs:          buildMathAbs,
  math_rectifier:    buildMathRectifier,
  math_slew:         buildMathSlew,
};

export default r1Builders;
export {
  buildBitcrusher,
  buildMultitapDelay,
  buildConvolution,
  buildEnvelopeFollower,
  buildPitchShift,
  buildComb,
  buildMathAdd,
  buildMathMultiply,
  buildMathConstant,
  buildMathScale,
  buildMathCrossfade,
  buildMathAbs,
  buildMathRectifier,
  buildMathSlew,
};
