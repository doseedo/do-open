/**
 * WebAudioDSPEngine — maps dsplang JSON config to a live Web Audio graph.
 *
 * Usage:
 *   const engine = new WebAudioDSPEngine(dspConfig);
 *   await engine.loadAudioFile(file);   // or loadTestTone('noise')
 *   engine.play();
 *   engine.setParameter('cutoff', 0.6); // normalized 0-1
 *   engine.stop();
 *   engine.dispose();
 */

// ── Helpers ────────────────────────────────────────────────────────────────

function scaleParam(norm, param) {
  const min = param.min ?? 0;
  const max = param.max ?? 1;
  const skew = param.skew || 1;
  // skew < 1 → more resolution at low end (e.g. 0.3 for frequency)
  const shaped = Math.pow(Math.max(0, Math.min(1, norm)), 1 / skew);
  return min + (max - min) * shaped;
}

function generateImpulseResponse(ctx, duration = 2.5, decay = 2.0) {
  const sr = ctx.sampleRate;
  const len = sr * duration;
  const buf = ctx.createBuffer(2, len, sr);
  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch);
    for (let i = 0; i < len; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.exp(-decay * i / len);
    }
  }
  return buf;
}

function generateNoiseBuffer(ctx, duration = 8) {
  const sr = ctx.sampleRate;
  const len = sr * duration;
  const buf = ctx.createBuffer(2, len, sr);
  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch);
    for (let i = 0; i < len; i++) {
      data[i] = Math.random() * 2 - 1;
    }
  }
  return buf;
}

function generateSineBuffer(ctx, freq = 440, duration = 8) {
  const sr = ctx.sampleRate;
  const len = sr * duration;
  const buf = ctx.createBuffer(2, len, sr);
  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch);
    for (let i = 0; i < len; i++) {
      data[i] = Math.sin(2 * Math.PI * freq * i / sr) * 0.5;
    }
  }
  return buf;
}

function generateSweepBuffer(ctx, duration = 8) {
  const sr = ctx.sampleRate;
  const len = sr * duration;
  const buf = ctx.createBuffer(2, len, sr);
  const f0 = 80, f1 = 8000;
  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch);
    let phase = 0;
    for (let i = 0; i < len; i++) {
      const t = i / len;
      const freq = f0 * Math.pow(f1 / f0, t);
      phase += (2 * Math.PI * freq) / sr;
      data[i] = Math.sin(phase) * 0.5;
    }
  }
  return buf;
}

function generateDrumBuffer(ctx, duration = 8) {
  const sr = ctx.sampleRate;
  const len = sr * duration;
  const buf = ctx.createBuffer(2, len, sr);
  // Simple kick + hihat pattern
  const bpm = 120;
  const beatSamples = (60 / bpm) * sr;
  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch);
    for (let i = 0; i < len; i++) {
      const beatPos = (i % beatSamples) / beatSamples;
      const beatNum = Math.floor(i / beatSamples) % 4;
      // Kick on 1 and 3
      if ((beatNum === 0 || beatNum === 2) && beatPos < 0.08) {
        const env = Math.exp(-beatPos * 50);
        const freq = 60 * Math.exp(-beatPos * 30);
        data[i] += Math.sin(2 * Math.PI * freq * beatPos * (60 / bpm)) * env * 0.7;
      }
      // Hihat on every 8th
      const sub = (i % (beatSamples / 2)) / (beatSamples / 2);
      if (sub < 0.02) {
        data[i] += (Math.random() * 2 - 1) * Math.exp(-sub * 200) * 0.3;
      }
      // Snare on 2 and 4
      if ((beatNum === 1 || beatNum === 3) && beatPos < 0.06) {
        const env = Math.exp(-beatPos * 40);
        data[i] += (Math.random() * 2 - 1) * env * 0.4;
        data[i] += Math.sin(2 * Math.PI * 200 * beatPos * (60 / bpm)) * env * 0.3;
      }
    }
  }
  return buf;
}

// ── Filter type mapping ────────────────────────────────────────────────────

const FILTER_TYPE_MAP = {
  lowpass: 'lowpass',
  highpass: 'highpass',
  bandpass: 'bandpass',
  notch: 'notch',
  allpass: 'allpass',
  shelf_low: 'lowshelf',
  shelf_high: 'highshelf',
  parametric_eq: 'peaking',
};

// ── Node builders ──────────────────────────────────────────────────────────
// Each builder returns { input, output, paramTargets }
// paramTargets maps "@param_id" → { audioParam, scale? } for real-time binding

function buildFilter(ctx, node, paramDefs) {
  const filterType = FILTER_TYPE_MAP[node.type] || 'lowpass';
  const filter = ctx.createBiquadFilter();
  filter.type = filterType;
  filter.frequency.value = 1000;
  filter.Q.value = 1;

  const targets = {};
  const params = node.params || {};

  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'cutoff' || key === 'frequency' || key === 'freq') {
        targets[paramId] = { audioParam: filter.frequency, paramDef: paramDefs[paramId] };
      } else if (key === 'resonance' || key === 'q' || key === 'Q') {
        targets[paramId] = { audioParam: filter.Q, paramDef: paramDefs[paramId] };
      } else if (key === 'gain') {
        targets[paramId] = { audioParam: filter.gain, paramDef: paramDefs[paramId] };
      }
    } else {
      // Static value
      if (key === 'cutoff' || key === 'frequency' || key === 'freq') filter.frequency.value = val;
      else if (key === 'resonance' || key === 'q' || key === 'Q') filter.Q.value = val;
      else if (key === 'gain') filter.gain.value = val;
    }
  }

  return { input: filter, output: filter, paramTargets: targets };
}

function buildLadder(ctx, node, paramDefs) {
  // Approximate Moog ladder with 4 cascaded lowpass filters
  const filters = [];
  for (let i = 0; i < 4; i++) {
    const f = ctx.createBiquadFilter();
    f.type = 'lowpass';
    f.frequency.value = 1000;
    f.Q.value = 0;
    filters.push(f);
    if (i > 0) filters[i - 1].connect(f);
  }

  const targets = {};
  const params = node.params || {};
  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'cutoff' || key === 'frequency') {
        // Link all 4 filters to same param
        targets[paramId] = {
          audioParams: filters.map(f => f.frequency),
          paramDef: paramDefs[paramId],
          multi: true,
        };
      } else if (key === 'resonance') {
        targets[paramId] = {
          audioParams: [filters[3].Q],
          paramDef: paramDefs[paramId] || { min: 0, max: 20 },
          multi: true,
        };
      }
    } else {
      if (key === 'cutoff' || key === 'frequency') filters.forEach(f => { f.frequency.value = val; });
      else if (key === 'resonance') filters[3].Q.value = val;
    }
  }

  return { input: filters[0], output: filters[3], paramTargets: targets };
}

function buildGain(ctx, node, paramDefs) {
  const gain = ctx.createGain();
  gain.gain.value = 1;
  const targets = {};
  const params = node.params || {};

  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'gain' || key === 'level' || key === 'volume' || key === 'makeup' || key === 'makeup_gain') {
        targets[paramId] = { audioParam: gain.gain, paramDef: paramDefs[paramId] };
      }
    } else {
      if (key === 'gain' || key === 'level' || key === 'volume') gain.gain.value = val;
    }
  }

  return { input: gain, output: gain, paramTargets: targets };
}

function buildPan(ctx, node, paramDefs) {
  const pan = ctx.createStereoPanner();
  pan.pan.value = 0;
  const targets = {};
  const params = node.params || {};

  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'pan' || key === 'position') {
        targets[paramId] = { audioParam: pan.pan, paramDef: paramDefs[paramId] || { min: -1, max: 1 } };
      }
    } else {
      if (key === 'pan') pan.pan.value = val;
    }
  }

  return { input: pan, output: pan, paramTargets: targets };
}

function buildDelay(ctx, node, paramDefs) {
  // Delay with feedback loop
  const input = ctx.createGain();
  const output = ctx.createGain();
  const delay = ctx.createDelay(5.0);
  const feedback = ctx.createGain();
  const dry = ctx.createGain();
  const wet = ctx.createGain();

  delay.delayTime.value = 0.3;
  feedback.gain.value = 0.4;
  dry.gain.value = 1;
  wet.gain.value = 0.5;

  input.connect(dry);
  input.connect(delay);
  delay.connect(feedback);
  feedback.connect(delay);
  delay.connect(wet);
  dry.connect(output);
  wet.connect(output);

  const targets = {};
  const params = node.params || {};
  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'time_ms' || key === 'time' || key === 'delay_time') {
        targets[paramId] = {
          audioParam: delay.delayTime,
          paramDef: paramDefs[paramId],
          scale: (v) => v / 1000, // ms → seconds
        };
      } else if (key === 'feedback') {
        targets[paramId] = { audioParam: feedback.gain, paramDef: paramDefs[paramId] };
      } else if (key === 'mix' || key === 'wet' || key === 'delay_mix') {
        targets[paramId] = { audioParam: wet.gain, paramDef: paramDefs[paramId] };
      }
    } else {
      if (key === 'time_ms' || key === 'time' || key === 'delay_time') delay.delayTime.value = (val || 300) / 1000;
      else if (key === 'feedback') feedback.gain.value = val;
      else if (key === 'mix' || key === 'wet') wet.gain.value = val;
    }
  }

  return { input, output, paramTargets: targets };
}

function buildCompressor(ctx, node, paramDefs) {
  const comp = ctx.createDynamicsCompressor();
  comp.threshold.value = -24;
  comp.ratio.value = 4;
  comp.attack.value = 0.003;
  comp.release.value = 0.25;
  comp.knee.value = 6;

  const targets = {};
  const params = node.params || {};
  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'threshold') targets[paramId] = { audioParam: comp.threshold, paramDef: paramDefs[paramId] };
      else if (key === 'ratio') targets[paramId] = { audioParam: comp.ratio, paramDef: paramDefs[paramId] };
      else if (key === 'attack') targets[paramId] = { audioParam: comp.attack, paramDef: paramDefs[paramId], scale: (v) => v / 1000 };
      else if (key === 'release') targets[paramId] = { audioParam: comp.release, paramDef: paramDefs[paramId], scale: (v) => v / 1000 };
      else if (key === 'knee') targets[paramId] = { audioParam: comp.knee, paramDef: paramDefs[paramId] };
    } else {
      if (key === 'threshold') comp.threshold.value = val;
      else if (key === 'ratio') comp.ratio.value = val;
      else if (key === 'attack') comp.attack.value = (val || 3) / 1000;
      else if (key === 'release') comp.release.value = (val || 250) / 1000;
    }
  }

  // Makeup gain after compressor
  const makeup = ctx.createGain();
  makeup.gain.value = 1;
  comp.connect(makeup);

  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'makeup' || key === 'makeup_gain') {
        targets[paramId] = { audioParam: makeup.gain, paramDef: paramDefs[paramId] };
      }
    }
  }

  return { input: comp, output: makeup, paramTargets: targets };
}

function buildLimiter(ctx, node, paramDefs) {
  // Limiter = aggressive compressor
  const comp = ctx.createDynamicsCompressor();
  comp.threshold.value = -6;
  comp.ratio.value = 20;
  comp.attack.value = 0.001;
  comp.release.value = 0.1;
  comp.knee.value = 0;

  const targets = {};
  const params = node.params || {};
  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'threshold' || key === 'ceiling') targets[paramId] = { audioParam: comp.threshold, paramDef: paramDefs[paramId] };
      else if (key === 'release') targets[paramId] = { audioParam: comp.release, paramDef: paramDefs[paramId], scale: (v) => v / 1000 };
    }
  }

  return { input: comp, output: comp, paramTargets: targets };
}

function buildReverb(ctx, node, paramDefs) {
  const input = ctx.createGain();
  const output = ctx.createGain();
  const convolver = ctx.createConvolver();
  const dry = ctx.createGain();
  const wet = ctx.createGain();

  dry.gain.value = 1;
  wet.gain.value = 0.4;

  // Default impulse response
  convolver.buffer = generateImpulseResponse(ctx, 2.5, 2.0);

  input.connect(dry);
  input.connect(convolver);
  convolver.connect(wet);
  dry.connect(output);
  wet.connect(output);

  const targets = {};
  const params = node.params || {};

  // Store references for decay regeneration
  const reverbState = { duration: 2.5, decay: 2.0 };

  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'mix' || key === 'wet' || key === 'reverb_mix') {
        targets[paramId] = { audioParam: wet.gain, paramDef: paramDefs[paramId] };
      } else if (key === 'decay' || key === 'room_size' || key === 'size') {
        targets[paramId] = {
          paramDef: paramDefs[paramId],
          // Custom setter — regenerate IR when decay changes
          customSetter: (value) => {
            reverbState.decay = Math.max(0.1, 6 - value * 5); // higher value = longer decay
            reverbState.duration = Math.max(0.5, value * 5);
            try {
              convolver.buffer = generateImpulseResponse(ctx, reverbState.duration, reverbState.decay);
            } catch (e) { /* ignore if context is closed */ }
          },
        };
      } else if (key === 'damping') {
        // Damping approximated by a lowpass after convolver
        // Skip for now — just map to wet gain as secondary
      }
    } else {
      if (key === 'mix' || key === 'wet') wet.gain.value = val;
    }
  }

  return { input, output, paramTargets: targets };
}

function buildWaveshaper(ctx, node, paramDefs) {
  const input = ctx.createGain();
  const output = ctx.createGain();
  const shaper = ctx.createWaveShaper();
  const dry = ctx.createGain();
  const wet = ctx.createGain();

  dry.gain.value = 0.5;
  wet.gain.value = 0.5;

  // Generate a tanh distortion curve
  const makeDistortionCurve = (amount) => {
    const n = 44100;
    const curve = new Float32Array(n);
    const k = amount;
    for (let i = 0; i < n; i++) {
      const x = (i * 2) / n - 1;
      curve[i] = ((1 + k) * x) / (1 + k * Math.abs(x));
    }
    return curve;
  };

  shaper.curve = makeDistortionCurve(5);
  shaper.oversample = '4x';

  input.connect(dry);
  input.connect(shaper);
  shaper.connect(wet);
  dry.connect(output);
  wet.connect(output);

  const targets = {};
  const params = node.params || {};
  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'drive' || key === 'amount' || key === 'gain') {
        targets[paramId] = {
          paramDef: paramDefs[paramId],
          customSetter: (value) => {
            shaper.curve = makeDistortionCurve(value * 50);
          },
        };
      } else if (key === 'mix') {
        targets[paramId] = { audioParam: wet.gain, paramDef: paramDefs[paramId] };
      }
    }
  }

  return { input, output, paramTargets: targets };
}

function buildChorus(ctx, node, paramDefs) {
  const input = ctx.createGain();
  const output = ctx.createGain();
  const dry = ctx.createGain();
  const wet = ctx.createGain();
  const delay = ctx.createDelay(0.1);
  const lfo = ctx.createOscillator();
  const lfoGain = ctx.createGain();

  dry.gain.value = 0.7;
  wet.gain.value = 0.5;
  delay.delayTime.value = 0.015;
  lfo.frequency.value = 1.5;
  lfoGain.gain.value = 0.003;

  lfo.connect(lfoGain);
  lfoGain.connect(delay.delayTime);
  lfo.start();

  input.connect(dry);
  input.connect(delay);
  delay.connect(wet);
  dry.connect(output);
  wet.connect(output);

  const targets = {};
  const params = node.params || {};
  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'rate') targets[paramId] = { audioParam: lfo.frequency, paramDef: paramDefs[paramId] };
      else if (key === 'depth') targets[paramId] = { audioParam: lfoGain.gain, paramDef: paramDefs[paramId] };
      else if (key === 'mix') targets[paramId] = { audioParam: wet.gain, paramDef: paramDefs[paramId] };
    }
  }

  return { input, output, paramTargets: targets, oscillators: [lfo] };
}

function buildPhaser(ctx, node, paramDefs) {
  // Phaser with cascaded allpass filters modulated by LFO
  const input = ctx.createGain();
  const output = ctx.createGain();
  const dry = ctx.createGain();
  const wet = ctx.createGain();

  dry.gain.value = 0.6;
  wet.gain.value = 0.6;

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
  lfo.frequency.value = 0.5;
  lfoGain.gain.value = 500;
  lfo.connect(lfoGain);
  stages.forEach(ap => lfoGain.connect(ap.frequency));
  lfo.start();

  input.connect(dry);
  input.connect(stages[0]);
  stages[stages.length - 1].connect(wet);
  dry.connect(output);
  wet.connect(output);

  const targets = {};
  const params = node.params || {};
  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'rate') targets[paramId] = { audioParam: lfo.frequency, paramDef: paramDefs[paramId] };
      else if (key === 'depth') targets[paramId] = { audioParam: lfoGain.gain, paramDef: paramDefs[paramId] };
      else if (key === 'mix') targets[paramId] = { audioParam: wet.gain, paramDef: paramDefs[paramId] };
    }
  }

  return { input, output, paramTargets: targets, oscillators: [lfo] };
}

function buildFlanger(ctx, node, paramDefs) {
  // Flanger = short delay with LFO + feedback
  const input = ctx.createGain();
  const output = ctx.createGain();
  const dry = ctx.createGain();
  const wet = ctx.createGain();
  const delay = ctx.createDelay(0.02);
  const feedback = ctx.createGain();
  const lfo = ctx.createOscillator();
  const lfoGain = ctx.createGain();

  dry.gain.value = 0.7;
  wet.gain.value = 0.7;
  delay.delayTime.value = 0.005;
  feedback.gain.value = 0.6;
  lfo.frequency.value = 0.3;
  lfoGain.gain.value = 0.003;

  lfo.connect(lfoGain);
  lfoGain.connect(delay.delayTime);
  lfo.start();

  input.connect(dry);
  input.connect(delay);
  delay.connect(feedback);
  feedback.connect(delay);
  delay.connect(wet);
  dry.connect(output);
  wet.connect(output);

  const targets = {};
  const params = node.params || {};
  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'rate') targets[paramId] = { audioParam: lfo.frequency, paramDef: paramDefs[paramId] };
      else if (key === 'depth') targets[paramId] = { audioParam: lfoGain.gain, paramDef: paramDefs[paramId] };
      else if (key === 'feedback') targets[paramId] = { audioParam: feedback.gain, paramDef: paramDefs[paramId] };
      else if (key === 'mix') targets[paramId] = { audioParam: wet.gain, paramDef: paramDefs[paramId] };
    }
  }

  return { input, output, paramTargets: targets, oscillators: [lfo] };
}

function buildTremolo(ctx, node, paramDefs) {
  const input = ctx.createGain();
  const tremGain = ctx.createGain();
  tremGain.gain.value = 1;
  const lfo = ctx.createOscillator();
  const lfoGain = ctx.createGain();

  lfo.frequency.value = 4;
  lfoGain.gain.value = 0.5;
  lfo.connect(lfoGain);
  lfoGain.connect(tremGain.gain);
  lfo.start();

  input.connect(tremGain);

  const targets = {};
  const params = node.params || {};
  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'rate') targets[paramId] = { audioParam: lfo.frequency, paramDef: paramDefs[paramId] };
      else if (key === 'depth') targets[paramId] = { audioParam: lfoGain.gain, paramDef: paramDefs[paramId] };
    }
  }

  return { input, output: tremGain, paramTargets: targets, oscillators: [lfo] };
}

// ── Node builder registry ──────────────────────────────────────────────────

const NODE_BUILDERS = {
  lowpass: buildFilter, highpass: buildFilter, bandpass: buildFilter,
  notch: buildFilter, allpass: buildFilter, shelf_low: buildFilter,
  shelf_high: buildFilter, parametric_eq: buildFilter,
  ladder: buildLadder,
  gain: buildGain, pan: buildPan,
  delay: buildDelay, multitap_delay: buildDelay, ping_pong_delay: buildDelay,
  compressor: buildCompressor, limiter: buildLimiter,
  gate: buildCompressor, expander: buildCompressor,
  reverb: buildReverb, convolution: buildReverb,
  overdrive: buildWaveshaper, waveshaper: buildWaveshaper,
  saturation: buildWaveshaper, foldback: buildWaveshaper,
  chorus: buildChorus, phaser: buildPhaser, flanger: buildFlanger,
  tremolo: buildTremolo, ring_mod: buildTremolo,
  // Passthrough for unsupported types
  dc_blocker: buildGain, mix: buildGain, splitter: buildGain, merger: buildGain,
};

// ── Main Engine Class ──────────────────────────────────────────────────────

export default class WebAudioDSPEngine {
  constructor(dspConfig) {
    this.config = dspConfig;
    this.ctx = null;
    this.sourceNode = null;
    this.audioBuffer = null;
    this.nodes = [];
    this.paramTargets = {}; // paramId → target info
    this.paramValues = {};  // paramId → current normalized value
    this.oscillators = [];
    this.playing = false;
    this.loop = true;
    this.masterGain = null;
    this.analyser = null;

    // Build param definitions lookup
    this.paramDefs = {};
    if (dspConfig?.parameters) {
      for (const p of dspConfig.parameters) {
        this.paramDefs[p.id] = p;
        this.paramValues[p.id] = (p.default != null && p.max != null && p.min != null)
          ? (p.default - p.min) / (p.max - p.min) : 0.5;
      }
    }
  }

  _ensureContext() {
    if (!this.ctx || this.ctx.state === 'closed') {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  _buildGraph() {
    this._ensureContext();
    const ctx = this.ctx;

    // Clean up previous graph
    this._teardownGraph();

    // Master gain + analyser
    this.masterGain = ctx.createGain();
    this.masterGain.gain.value = 0.8;
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 256;
    this.masterGain.connect(this.analyser);
    this.analyser.connect(ctx.destination);

    const chain = this.config?.dspChain || [];
    if (chain.length === 0) {
      // No DSP — direct passthrough
      this._chainInput = this.masterGain;
      return;
    }

    // Build each node
    const builtNodes = [];
    for (const nodeDef of chain) {
      const builder = NODE_BUILDERS[nodeDef.type];
      if (!builder) {
        // Unknown type — passthrough gain
        const g = ctx.createGain();
        builtNodes.push({ input: g, output: g, paramTargets: {} });
        continue;
      }
      const built = builder(ctx, nodeDef, this.paramDefs);
      builtNodes.push(built);
      if (built.oscillators) this.oscillators.push(...built.oscillators);

      // Collect param targets
      for (const [paramId, target] of Object.entries(built.paramTargets)) {
        this.paramTargets[paramId] = target;
      }
    }

    // Connect nodes in series
    for (let i = 1; i < builtNodes.length; i++) {
      builtNodes[i - 1].output.connect(builtNodes[i].input);
    }

    // Last node → master gain
    builtNodes[builtNodes.length - 1].output.connect(this.masterGain);

    this._chainInput = builtNodes[0].input;
    this.nodes = builtNodes;

    // Apply initial parameter values
    for (const [paramId, normVal] of Object.entries(this.paramValues)) {
      this._applyParam(paramId, normVal);
    }
  }

  _teardownGraph() {
    this.oscillators.forEach(o => { try { o.stop(); } catch (e) {} });
    this.oscillators = [];
    this.nodes = [];
    this.paramTargets = {};
    if (this.sourceNode) {
      try { this.sourceNode.stop(); } catch (e) {}
      this.sourceNode = null;
    }
  }

  _applyParam(paramId, normValue) {
    const target = this.paramTargets[paramId];
    if (!target) return;

    if (target.customSetter) {
      const paramDef = target.paramDef || { min: 0, max: 1 };
      const scaled = scaleParam(normValue, paramDef);
      target.customSetter(scaled);
      return;
    }

    const paramDef = target.paramDef || { min: 0, max: 1 };
    let value = scaleParam(normValue, paramDef);
    if (target.scale) value = target.scale(value);

    const t = this.ctx?.currentTime || 0;
    if (target.multi && target.audioParams) {
      target.audioParams.forEach(ap => {
        ap.cancelScheduledValues(t);
        ap.setTargetAtTime(value, t, 0.02);
      });
    } else if (target.audioParam) {
      target.audioParam.cancelScheduledValues(t);
      target.audioParam.setTargetAtTime(value, t, 0.02);
    }
  }

  // ── Public API ─────────────────────────────────────────────────────────

  async loadAudioFile(file) {
    this._ensureContext();
    const arrayBuffer = await file.arrayBuffer();
    this.audioBuffer = await this.ctx.decodeAudioData(arrayBuffer);
  }

  async loadAudioUrl(url) {
    this._ensureContext();
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    this.audioBuffer = await this.ctx.decodeAudioData(arrayBuffer);
  }

  loadTestTone(type = 'drums') {
    this._ensureContext();
    switch (type) {
      case 'noise': this.audioBuffer = generateNoiseBuffer(this.ctx, 8); break;
      case 'sine': this.audioBuffer = generateSineBuffer(this.ctx, 440, 8); break;
      case 'sweep': this.audioBuffer = generateSweepBuffer(this.ctx, 8); break;
      case 'drums': this.audioBuffer = generateDrumBuffer(this.ctx, 8); break;
      default: this.audioBuffer = generateDrumBuffer(this.ctx, 8);
    }
  }

  play() {
    if (this.playing) this.stop();
    this._buildGraph();

    if (!this.audioBuffer) {
      this.loadTestTone('drums');
    }

    const source = this.ctx.createBufferSource();
    source.buffer = this.audioBuffer;
    source.loop = this.loop;
    source.connect(this._chainInput || this.masterGain);
    source.start();
    this.sourceNode = source;
    this.playing = true;

    source.onended = () => {
      if (!this.loop) {
        this.playing = false;
      }
    };
  }

  stop() {
    if (this.sourceNode) {
      try { this.sourceNode.stop(); } catch (e) {}
      this.sourceNode = null;
    }
    this.playing = false;
  }

  setParameter(paramId, normalizedValue) {
    this.paramValues[paramId] = normalizedValue;
    if (this.playing) {
      this._applyParam(paramId, normalizedValue);
    }
  }

  setMasterVolume(value) {
    if (this.masterGain) {
      this.masterGain.gain.setTargetAtTime(value, this.ctx.currentTime, 0.02);
    }
  }

  setLoop(loop) {
    this.loop = loop;
    if (this.sourceNode) {
      this.sourceNode.loop = loop;
    }
  }

  getAnalyserData() {
    if (!this.analyser) return null;
    const data = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteFrequencyData(data);
    return data;
  }

  getParameterIds() {
    return (this.config?.parameters || []).map(p => p.id);
  }

  getParameterDef(paramId) {
    return this.paramDefs[paramId];
  }

  dispose() {
    this.stop();
    this._teardownGraph();
    if (this.ctx && this.ctx.state !== 'closed') {
      this.ctx.close().catch(() => {});
    }
    this.ctx = null;
    this.audioBuffer = null;
  }
}
