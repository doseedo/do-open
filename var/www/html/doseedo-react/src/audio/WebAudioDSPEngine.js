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

function dbToLinear(db) {
  return Math.pow(10, db / 20);
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
        const pDef = paramDefs[paramId];
        const isDb = pDef && (pDef.unit === 'dB' || pDef.unit === 'db');
        targets[paramId] = { audioParam: gain.gain, paramDef: pDef, scale: isDb ? dbToLinear : undefined };
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

  return { input: comp, output: makeup, paramTargets: targets, compressorNode: comp };
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

  return { input: comp, output: comp, paramTargets: targets, compressorNode: comp };
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

// ── Instrument node builders ─────────────────────────────────────────────────

function buildOscillator(ctx, node, paramDefs) {
  // Oscillator that responds to noteOn/noteOff for instrument mode
  const output = ctx.createGain();
  output.gain.value = 0.5;
  const targets = {};
  const params = node.params || {};
  const waveType = params.waveform || params.type || 'sawtooth';

  // Store config for creating voices on noteOn
  const oscConfig = {
    type: typeof waveType === 'string' && !waveType.startsWith('@') ? waveType : 'sawtooth',
    detune: 0,
    unison: 1,
    unisonSpread: 15,
  };

  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'detune' || key === 'fine') {
        targets[paramId] = { paramDef: paramDefs[paramId], customSetter: (v) => { oscConfig.detune = v; } };
      } else if (key === 'gain' || key === 'level') {
        targets[paramId] = { audioParam: output.gain, paramDef: paramDefs[paramId] };
      }
    } else {
      if (key === 'detune' || key === 'fine') oscConfig.detune = val;
      if (key === 'unison') oscConfig.unison = val;
      if (key === 'unison_spread') oscConfig.unisonSpread = val;
    }
  }

  return { input: output, output, paramTargets: targets, oscConfig, isSource: true };
}

function buildNoiseGen(ctx, node, paramDefs) {
  const bufLen = ctx.sampleRate * 4;
  const buf = ctx.createBuffer(2, bufLen, ctx.sampleRate);
  const noiseType = (node.params?.type || 'white').toLowerCase();

  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch);
    if (noiseType === 'pink') {
      let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
      for (let i = 0; i < bufLen; i++) {
        const white = Math.random() * 2 - 1;
        b0 = 0.99886 * b0 + white * 0.0555179;
        b1 = 0.99332 * b1 + white * 0.0750759;
        b2 = 0.96900 * b2 + white * 0.1538520;
        b3 = 0.86650 * b3 + white * 0.3104856;
        b4 = 0.55000 * b4 + white * 0.5329522;
        b5 = -0.7616 * b5 - white * 0.0168980;
        data[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.11;
        b6 = white * 0.115926;
      }
    } else {
      for (let i = 0; i < bufLen; i++) data[i] = Math.random() * 2 - 1;
    }
  }

  const source = ctx.createBufferSource();
  source.buffer = buf;
  source.loop = true;
  const output = ctx.createGain();
  output.gain.value = 0.3;
  source.connect(output);
  source.start();

  const targets = {};
  const params = node.params || {};
  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'gain' || key === 'level') {
        targets[paramId] = { audioParam: output.gain, paramDef: paramDefs[paramId] };
      }
    }
  }

  return { input: output, output, paramTargets: targets, oscillators: [source], isSource: true };
}

function buildADSREnvelope(ctx, node, paramDefs) {
  const gain = ctx.createGain();
  gain.gain.value = 0;
  const targets = {};
  const params = node.params || {};

  const envConfig = { attack: 0.01, decay: 0.2, sustain: 0.7, release: 0.3 };

  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      targets[paramId] = {
        paramDef: paramDefs[paramId],
        customSetter: (v) => { envConfig[key] = v; },
      };
    } else {
      if (key in envConfig) envConfig[key] = val;
    }
  }

  return { input: gain, output: gain, paramTargets: targets, envConfig, isEnvelope: true };
}

function buildLFONode(ctx, node, paramDefs) {
  const lfo = ctx.createOscillator();
  const lfoGain = ctx.createGain();
  lfo.frequency.value = 2;
  lfoGain.gain.value = 0.5;
  lfo.type = (node.params?.shape || 'sine').toLowerCase();
  lfo.connect(lfoGain);
  lfo.start();

  // LFO output goes to a modulation target — pass through a gain node
  const output = ctx.createGain();
  output.gain.value = 1;
  lfoGain.connect(output);

  const targets = {};
  const params = node.params || {};
  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'rate' || key === 'frequency') targets[paramId] = { audioParam: lfo.frequency, paramDef: paramDefs[paramId] };
      else if (key === 'depth' || key === 'amount') targets[paramId] = { audioParam: lfoGain.gain, paramDef: paramDefs[paramId] };
    } else {
      if (key === 'rate' || key === 'frequency') lfo.frequency.value = val;
      else if (key === 'depth' || key === 'amount') lfoGain.gain.value = val;
    }
  }

  return { input: output, output, paramTargets: targets, oscillators: [lfo] };
}

function buildSubOscillator(ctx, node, paramDefs) {
  // Sub oscillator: same as oscillator but defaults to -1 or -2 octaves
  const result = buildOscillator(ctx, node, paramDefs);
  result.oscConfig.type = (node.params?.waveform || 'sine').toLowerCase();
  result.oscConfig.octaveOffset = node.params?.octave ?? -1;
  return result;
}

function buildMixer(ctx, node, paramDefs) {
  const gain = ctx.createGain();
  gain.gain.value = 1;
  const targets = {};
  const params = node.params || {};
  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'gain' || key === 'level') targets[paramId] = { audioParam: gain.gain, paramDef: paramDefs[paramId] };
    }
  }
  return { input: gain, output: gain, paramTargets: targets };
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
  // Instrument nodes
  oscillator: buildOscillator, osc: buildOscillator, saw: buildOscillator,
  square: buildOscillator, sine: buildOscillator, triangle: buildOscillator,
  sub_oscillator: buildSubOscillator, sub_osc: buildSubOscillator,
  noise: buildNoiseGen, noise_gen: buildNoiseGen, white_noise: buildNoiseGen, pink_noise: buildNoiseGen,
  adsr: buildADSREnvelope, envelope: buildADSREnvelope, amp_env: buildADSREnvelope, filter_env: buildADSREnvelope,
  lfo: buildLFONode,
  mixer: buildMixer, mix_bus: buildMixer,
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
      try {
        this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        this.contextError = null;
      } catch (err) {
        this.contextError = err.message || 'Failed to create AudioContext';
        return;
      }
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => {});
    }
  }

  getContextState() {
    return {
      state: this.ctx?.state,
      error: this.contextError,
    };
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

    // If a raw dspGraph with parallel routing exists, use it
    if (this.config?.dspGraph?.nodes?.length > 0 && this.config?.dspGraph?.edges?.length > 0) {
      this._buildGraphFromNodes();
      return;
    }

    const chain = this.config?.dspChain || [];
    if (chain.length === 0) {
      // No DSP — direct passthrough
      this._chainInput = this.masterGain;
      this._voiceInput = this.masterGain;
      return;
    }

    // Instrument mode: separate sources/envelopes from effects
    if (this.isInstrument) {
      this._buildInstrumentGraph(chain);
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
      if (built.compressorNode) {
        if (!this._compressorNodes) this._compressorNodes = [];
        this._compressorNodes.push(built.compressorNode);
      }

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

  _buildGraphFromNodes() {
    const ctx = this.ctx;
    const graph = this.config.dspGraph;
    const builtNodes = {};

    // Build each node
    for (const nodeDef of graph.nodes) {
      if (nodeDef.type === 'input' || nodeDef.type === 'output') {
        // Virtual input/output nodes
        const g = ctx.createGain();
        builtNodes[nodeDef.id] = { input: g, output: g, paramTargets: {} };
        continue;
      }
      const builder = NODE_BUILDERS[nodeDef.type];
      if (!builder) {
        const g = ctx.createGain();
        builtNodes[nodeDef.id] = { input: g, output: g, paramTargets: {} };
        continue;
      }
      const built = builder(ctx, nodeDef, this.paramDefs);
      builtNodes[nodeDef.id] = built;
      if (built.oscillators) this.oscillators.push(...built.oscillators);
      for (const [paramId, target] of Object.entries(built.paramTargets)) {
        this.paramTargets[paramId] = target;
      }
    }

    // Connect based on edges
    for (const edge of graph.edges) {
      const src = builtNodes[edge.source];
      const tgt = builtNodes[edge.target];
      if (src && tgt) {
        try { src.output.connect(tgt.input); } catch (e) { /* ignore connection errors */ }
      }
    }

    // Find input and output nodes
    const inputNode = builtNodes['input'] || builtNodes['audio_input'];
    const outputNode = builtNodes['output'] || builtNodes['audio_output'];

    if (outputNode) {
      outputNode.output.connect(this.masterGain);
    }

    this._chainInput = inputNode?.input || this.masterGain;
    this.nodes = Object.values(builtNodes);

    // Apply initial parameter values
    for (const [paramId, normVal] of Object.entries(this.paramValues)) {
      this._applyParam(paramId, normVal);
    }
  }

  _buildInstrumentGraph(chain) {
    const ctx = this.ctx;
    this._sourceConfigs = [];
    this._envelopeConfigs = [];

    const SOURCE_TYPES = new Set([
      'oscillator', 'osc', 'saw', 'square', 'sine', 'triangle',
      'sub_oscillator', 'sub_osc', 'noise', 'noise_gen', 'white_noise', 'pink_noise',
    ]);
    const ENVELOPE_TYPES = new Set(['adsr', 'envelope', 'amp_env', 'filter_env']);

    const effectDefs = [];

    for (const nodeDef of chain) {
      if (SOURCE_TYPES.has(nodeDef.type)) {
        const params = nodeDef.params || {};
        let waveType = params.waveform || params.type || 'sawtooth';
        if (typeof waveType === 'string' && waveType.startsWith('@')) waveType = 'sawtooth';
        // Override wave type for explicitly named types
        if (nodeDef.type === 'square') waveType = 'square';
        else if (nodeDef.type === 'sine') waveType = 'sine';
        else if (nodeDef.type === 'triangle') waveType = 'triangle';
        else if (nodeDef.type === 'saw') waveType = 'sawtooth';

        const cfg = {
          type: waveType,
          detune: 0, unison: 1, unisonSpread: 15, octaveOffset: 0, gainValue: 0.5,
        };

        if (nodeDef.type === 'sub_oscillator' || nodeDef.type === 'sub_osc') {
          cfg.type = (params.waveform || 'sine').toLowerCase();
          if (cfg.type.startsWith('@')) cfg.type = 'sine';
          cfg.octaveOffset = params.octave ?? -1;
        }

        if (nodeDef.type.includes('noise')) {
          cfg.isNoise = true;
          cfg.noiseType = (params.type || 'white').toLowerCase();
        }

        // Wire param bindings to update config for future notes
        for (const [key, val] of Object.entries(params)) {
          if (typeof val === 'string' && val.startsWith('@')) {
            const paramId = val.slice(1);
            if (key === 'detune' || key === 'fine') {
              this.paramTargets[paramId] = { paramDef: this.paramDefs[paramId], customSetter: (v) => { cfg.detune = v; } };
            } else if (key === 'gain' || key === 'level') {
              this.paramTargets[paramId] = { paramDef: this.paramDefs[paramId], customSetter: (v) => { cfg.gainValue = v; } };
            }
          } else {
            if (key === 'detune' || key === 'fine') cfg.detune = val;
            if (key === 'unison') cfg.unison = val;
            if (key === 'unison_spread') cfg.unisonSpread = val;
          }
        }

        this._sourceConfigs.push(cfg);
        continue;
      }

      if (ENVELOPE_TYPES.has(nodeDef.type)) {
        const params = nodeDef.params || {};
        const envCfg = { attack: 0.01, decay: 0.2, sustain: 0.7, release: 0.3 };
        for (const [key, val] of Object.entries(params)) {
          if (typeof val === 'string' && val.startsWith('@')) {
            const paramId = val.slice(1);
            this.paramTargets[paramId] = { paramDef: this.paramDefs[paramId], customSetter: (v) => { envCfg[key] = v; } };
          } else {
            if (key in envCfg) envCfg[key] = val;
          }
        }
        this._envelopeConfigs.push(envCfg);
        continue;
      }

      // Effect node — keep for static chain
      effectDefs.push(nodeDef);
    }

    // Build only effect nodes
    const effectNodes = [];
    for (const nodeDef of effectDefs) {
      const builder = NODE_BUILDERS[nodeDef.type];
      if (!builder) {
        const g = ctx.createGain();
        effectNodes.push({ input: g, output: g, paramTargets: {} });
        continue;
      }
      const built = builder(ctx, nodeDef, this.paramDefs);
      effectNodes.push(built);
      if (built.oscillators) this.oscillators.push(...built.oscillators);
      for (const [paramId, target] of Object.entries(built.paramTargets)) {
        this.paramTargets[paramId] = target;
      }
    }

    // Voice merge point → effect chain → master
    this._voiceInput = ctx.createGain();
    this._voiceInput.gain.value = 1;

    if (effectNodes.length > 0) {
      this._voiceInput.connect(effectNodes[0].input);
      for (let i = 1; i < effectNodes.length; i++) {
        effectNodes[i - 1].output.connect(effectNodes[i].input);
      }
      effectNodes[effectNodes.length - 1].output.connect(this.masterGain);
    } else {
      this._voiceInput.connect(this.masterGain);
    }

    this._chainInput = this._voiceInput;
    this.nodes = effectNodes;

    // Defaults if DSP chain didn't include explicit source/envelope
    if (this._envelopeConfigs.length === 0) {
      this._envelopeConfigs.push({ attack: 0.01, decay: 0.2, sustain: 0.7, release: 0.3 });
    }
    if (this._sourceConfigs.length === 0) {
      this._sourceConfigs.push({ type: 'sawtooth', detune: 0, unison: 1, unisonSpread: 15, octaveOffset: 0, gainValue: 0.5 });
    }

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
    this._compressorNodes = [];
    if (this.sourceNode) {
      try { this.sourceNode.stop(); } catch (e) {}
      this.sourceNode = null;
    }
    // Clean up instrument voices
    if (this._activeVoices) {
      for (const v of Object.values(this._activeVoices)) {
        v.voices.forEach(o => { try { o.stop(); } catch (e) {} });
        try { v.voiceGain.disconnect(); } catch (e) {}
      }
      this._activeVoices = {};
    }
    this._voiceInput = null;
    this._sourceConfigs = null;
    this._envelopeConfigs = null;
    this.stopMicInput();
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
    // For instruments, just ensure graph is built (notes are triggered via noteOn)
    if (this.isInstrument) {
      if (!this.masterGain) this._buildGraph();
      this.playing = true;
      return;
    }

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
    this.stopMicInput();
    this.playing = false;
  }

  setParameter(paramId, normalizedValue) {
    this.paramValues[paramId] = normalizedValue;
    // Apply immediately if graph is built (not just when playing — instruments need params between notes)
    if (this.masterGain) {
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

  // ── Microphone input (for FX plugins) ──────────────────────────────────

  async startMicInput() {
    this._ensureContext();
    if (!this.masterGain) this._buildGraph();
    if (this._micStream) this.stopMicInput();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this._micStream = stream;
    this._micSource = this.ctx.createMediaStreamSource(stream);
    this._micSource.connect(this._chainInput || this.masterGain);
    this.playing = true;
  }

  stopMicInput() {
    if (this._micSource) {
      try { this._micSource.disconnect(); } catch (e) {}
      this._micSource = null;
    }
    if (this._micStream) {
      this._micStream.getTracks().forEach(t => t.stop());
      this._micStream = null;
    }
  }

  getAnalyserData() {
    if (!this.analyser) return null;
    const data = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteFrequencyData(data);
    return data;
  }

  /** Returns gain reduction in dB (negative value) from the first compressor/limiter node, or 0 */
  getReduction() {
    if (!this._compressorNodes?.length) return 0;
    return this._compressorNodes[0].reduction || 0; // reduction is a float, always <= 0
  }

  /** Returns current output RMS level as 0-1 from the analyser */
  getOutputLevel() {
    if (!this.analyser) return 0;
    const data = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteFrequencyData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i];
    return sum / (data.length * 255); // normalized 0-1
  }

  getParameterIds() {
    return (this.config?.parameters || []).map(p => p.id);
  }

  getParameterDef(paramId) {
    return this.paramDefs[paramId];
  }

  // Preset support: snapshot / restore all param values
  getState() {
    return { ...this.paramValues };
  }

  setState(values) {
    if (!values) return;
    for (const [paramId, val] of Object.entries(values)) {
      this.setParameter(paramId, val);
    }
  }

  // ── Instrument mode (synth) ───────────────────────────────────────────

  get isInstrument() {
    return this.config?.pluginType === 'instrument' ||
      (this.config?.dspChain || []).some(n => NODE_BUILDERS[n.type] === buildOscillator || NODE_BUILDERS[n.type] === buildSubOscillator || NODE_BUILDERS[n.type] === buildNoiseGen);
  }

  noteOn(midiNote = 60, velocity = 0.8) {
    this._ensureContext();
    if (!this.masterGain) this._buildGraph();
    const ctx = this.ctx;
    const freq = 440 * Math.pow(2, (midiNote - 69) / 12);
    const t = ctx.currentTime;

    // Per-voice envelope config (read current values — may be updated by param knobs)
    const env = this._envelopeConfigs?.[0] || { attack: 0.01, decay: 0.2, sustain: 0.7, release: 0.3 };

    // Create per-voice envelope gain node
    const voiceGain = ctx.createGain();
    voiceGain.gain.setValueAtTime(0, t);
    voiceGain.gain.linearRampToValueAtTime(velocity, t + Math.max(0.001, env.attack));
    voiceGain.gain.linearRampToValueAtTime(
      velocity * (env.sustain ?? 0.7),
      t + Math.max(0.001, env.attack) + Math.max(0.001, env.decay)
    );
    voiceGain.connect(this._voiceInput || this.masterGain);

    // Create oscillators for each source config
    const voices = [];
    const sourceConfigs = this._sourceConfigs || [];

    for (const cfg of sourceConfigs) {
      if (cfg.isNoise) {
        // Per-voice noise buffer
        const bufLen = ctx.sampleRate * 4;
        const buf = ctx.createBuffer(2, bufLen, ctx.sampleRate);
        for (let ch = 0; ch < 2; ch++) {
          const data = buf.getChannelData(ch);
          if (cfg.noiseType === 'pink') {
            let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
            for (let i = 0; i < bufLen; i++) {
              const w = Math.random() * 2 - 1;
              b0 = 0.99886 * b0 + w * 0.0555179; b1 = 0.99332 * b1 + w * 0.0750759;
              b2 = 0.96900 * b2 + w * 0.1538520; b3 = 0.86650 * b3 + w * 0.3104856;
              b4 = 0.55000 * b4 + w * 0.5329522; b5 = -0.7616 * b5 - w * 0.0168980;
              data[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11;
              b6 = w * 0.115926;
            }
          } else {
            for (let i = 0; i < bufLen; i++) data[i] = Math.random() * 2 - 1;
          }
        }
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.loop = true;
        const noiseGain = ctx.createGain();
        noiseGain.gain.value = cfg.gainValue ?? 0.3;
        src.connect(noiseGain);
        noiseGain.connect(voiceGain);
        src.start(t);
        voices.push(src);
      } else {
        const count = cfg.unison || 1;
        for (let u = 0; u < count; u++) {
          const osc = ctx.createOscillator();
          osc.type = cfg.type || 'sawtooth';
          const octOffset = cfg.octaveOffset || 0;
          const uniDetune = count > 1 ? (u / (count - 1) - 0.5) * (cfg.unisonSpread || 15) : 0;
          osc.frequency.value = freq * Math.pow(2, octOffset);
          osc.detune.value = (cfg.detune || 0) + uniDetune;
          const oscGain = ctx.createGain();
          oscGain.gain.value = cfg.gainValue ?? 0.5;
          osc.connect(oscGain);
          oscGain.connect(voiceGain);
          osc.start(t);
          voices.push(osc);
        }
      }
    }

    this.playing = true;
    const voiceId = Date.now() + '-' + midiNote;
    if (!this._activeVoices) this._activeVoices = {};
    this._activeVoices[voiceId] = { voices, voiceGain, midiNote, env: { ...env } };
    return voiceId;
  }

  noteOff(voiceId) {
    if (!this._activeVoices?.[voiceId]) return;
    const { voices, voiceGain, env } = this._activeVoices[voiceId];
    const ctx = this.ctx;
    const t = ctx.currentTime;
    const release = Math.max(0.01, env?.release || 0.3);

    // Release this voice's envelope
    voiceGain.gain.cancelScheduledValues(t);
    voiceGain.gain.setValueAtTime(voiceGain.gain.value, t);
    voiceGain.gain.linearRampToValueAtTime(0, t + release);

    // Stop oscillators after release
    for (const osc of voices) {
      osc.stop(t + release + 0.05);
    }

    // Clean up after release completes
    setTimeout(() => {
      try { voiceGain.disconnect(); } catch (e) {}
    }, (release + 0.1) * 1000);

    delete this._activeVoices[voiceId];
  }

  // Keyboard mapping: computer keys → MIDI notes
  static KEYBOARD_MAP = {
    'a': 60, 'w': 61, 's': 62, 'e': 63, 'd': 64, 'f': 65, 't': 66,
    'g': 67, 'y': 68, 'h': 69, 'u': 70, 'j': 71, 'k': 72, 'o': 73,
    'l': 74, 'p': 75, ';': 76,
  };

  dispose() {
    // Stop active voices first (before teardown kills the graph)
    if (this._activeVoices) {
      for (const vid of Object.keys(this._activeVoices)) this.noteOff(vid);
    }
    this.stop();
    this._teardownGraph();
    if (this.ctx && this.ctx.state !== 'closed') {
      this.ctx.close().catch(() => {});
    }
    this.ctx = null;
    this.audioBuffer = null;
  }
}
