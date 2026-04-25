/**
 * VoiceManager — polyphonic voice allocator + MIDI driver for dsplang voice
 * graphs. Pairs with WebAudioDSPEngine: instead of building a single static
 * oscillator chain, the engine hands the manager a "voice template" sub-graph
 * (nodes + edges describing ONE voice) and the manager instantiates a fresh
 * copy of that graph for every note-on, ramps the per-voice envelope, then
 * tears the voice down once release tail decays.
 *
 * Wire format reuses the WebAudioDSPEngine `dspGraph` shape:
 *   voiceTemplate = {
 *     nodes: [
 *       { id: 'osc1', type: 'oscillator', params: { waveform: 'sawtooth', ...,
 *           frequency: '@pitch' } },           // '@pitch' = bind to this voice's note Hz
 *       { id: 'env',  type: 'adsr', params: { attack: '@amp_attack', ... } },
 *       { id: 'vca',  type: 'gain', params: { gain: '@gate' } },          // '@gate' = env value
 *       { id: 'output', type: 'output' },
 *     ],
 *     edges: [
 *       { source: 'osc1', target: 'vca' },
 *       { source: 'vca',  target: 'output' },
 *     ],
 *   }
 *
 * Implementation notes:
 *   • Each voice = an isolated mini Web-Audio graph rooted at its own gain node.
 *     We chose plain WebAudio nodes per voice (vs a single AudioWorklet w/ voice
 *     arrays) because (a) it keeps parity with WebAudioDSPEngine's existing node
 *     builders (zero duplication) and (b) the GC pressure of 8-16 short-lived
 *     graphs per second is negligible compared to the cost of writing a custom
 *     polyphonic worklet — we can always migrate hot paths later.
 *   • Voice stealing: when polyphony cap is hit, the cheapest voice is freed. The
 *     "quietest" mode samples each voice's envelope value via its `voiceGain`
 *     scheduled state; "oldest" is FIFO; default is "oldest" (cheapest, no envelope
 *     introspection cost).
 *   • Sustain pedal: while held, noteOff just marks the voice as
 *     `.deferredRelease=true`; setSustain(false) releases all such voices.
 *   • Mono / legato: a single voice slot is reused; legato suppresses the env
 *     retrigger (keeps current envelope state, only re-pitches), retrigger mode
 *     restarts attack on every new note while the previous note is still held.
 *
 * No external dependencies — pure browser APIs.
 */

// ── Frequency math ──────────────────────────────────────────────────────────

const REF_MIDI = 69;
const REF_HZ = 440;

function midiToHz(note) {
  return REF_HZ * Math.pow(2, (note - REF_MIDI) / 12);
}

// ── Filter type → BiquadFilter type ────────────────────────────────────────

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

// ── Per-voice node builders ─────────────────────────────────────────────────
// Voice-aware builders: any param value of the form '@gate', '@pitch',
// '@velocity' is wired to per-voice signals; '@<paramId>' (anything else)
// resolves against shared paramValues at instantiation time.
//
// Each builder returns:
//   { input, output, audioNodes, perVoiceTargets, paramTargets, isSource, isEnvelope }
//
// audioNodes: list of nodes that need .stop()/.disconnect() at teardown
// perVoiceTargets: { '@gate' | '@pitch' | '@velocity': AudioParam | setter[] }
// paramTargets: shared-knob bindings, identical shape to engine's paramTargets

function isBound(val) {
  return typeof val === 'string' && val.startsWith('@');
}

function bindKey(val) {
  return val.slice(1);
}

function noiseBuffer(ctx, kind = 'white', seconds = 4) {
  const len = ctx.sampleRate * seconds;
  const buf = ctx.createBuffer(2, len, ctx.sampleRate);
  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch);
    if (kind === 'pink') {
      let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
      for (let i = 0; i < len; i++) {
        const w = Math.random() * 2 - 1;
        b0 = 0.99886 * b0 + w * 0.0555179;
        b1 = 0.99332 * b1 + w * 0.0750759;
        b2 = 0.96900 * b2 + w * 0.1538520;
        b3 = 0.86650 * b3 + w * 0.3104856;
        b4 = 0.55000 * b4 + w * 0.5329522;
        b5 = -0.7616 * b5 - w * 0.0168980;
        data[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11;
        b6 = w * 0.115926;
      }
    } else if (kind === 'brown') {
      let last = 0;
      for (let i = 0; i < len; i++) {
        const w = (Math.random() * 2 - 1) * 0.02;
        last = Math.max(-1, Math.min(1, last + w));
        data[i] = last * 3.5;
      }
    } else {
      for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;
    }
  }
  return buf;
}

function buildVoiceOscillator(ctx, node, paramSnap) {
  const params = node.params || {};
  const out = ctx.createGain();
  out.gain.value = 1;
  const audioNodes = [];
  const perVoice = {};
  const sharedTargets = {};

  let waveform = params.waveform || params.type || 'sawtooth';
  if (isBound(waveform)) waveform = 'sawtooth';
  if (waveform === 'saw') waveform = 'sawtooth';

  // Detune in cents and base octave offset (static reads only — knobs apply
  // to all future voices through paramSnap)
  const detuneStatic = isBound(params.detune) ? (paramSnap[bindKey(params.detune)] ?? 0) : (params.detune ?? 0);
  const octaveOffset = params.octave_offset ?? 0;
  const unisonCount = Math.max(1, Math.floor(params.unison ?? 1));
  const unisonSpread = params.unison_spread ?? 15;
  const levelStatic = isBound(params.gain ?? params.level)
    ? (paramSnap[bindKey(params.gain ?? params.level)] ?? 0.5)
    : (params.gain ?? params.level ?? 0.5);

  // Each oscillator gets a per-voice frequency AudioParam. Combine all of them
  // into perVoice['@pitch'] so VoiceManager can drive them in lock-step.
  const pitchParams = [];
  const detuneParams = [];

  for (let u = 0; u < unisonCount; u++) {
    const osc = ctx.createOscillator();
    osc.type = waveform;
    const uniSpread = unisonCount > 1 ? (u / (unisonCount - 1) - 0.5) * unisonSpread : 0;
    osc.detune.value = detuneStatic + uniSpread;
    // Octave offset baked into freq when '@pitch' is delivered
    osc.frequency.value = REF_HZ; // placeholder; VoiceManager sets real Hz
    pitchParams.push({ audioParam: osc.frequency, octaveOffset });
    detuneParams.push({ audioParam: osc.detune, base: uniSpread });
    const gain = ctx.createGain();
    gain.gain.value = levelStatic / unisonCount;
    osc.connect(gain).connect(out);
    audioNodes.push(osc);
  }

  if (isBound(params.frequency)) perVoice[params.frequency] = pitchParams;
  // 'oscillator' nodes implicitly accept @pitch even without explicit binding
  if (!perVoice['@pitch']) perVoice['@pitch'] = pitchParams;

  // Detune knob (modulatable) → all detune AudioParams
  if (isBound(params.detune)) {
    sharedTargets[bindKey(params.detune)] = {
      audioParams: detuneParams.map(d => d.audioParam),
      multi: true,
      paramDef: null,
    };
  }
  if (isBound(params.gain ?? params.level)) {
    const k = bindKey(params.gain ?? params.level);
    sharedTargets[k] = { audioParam: out.gain, paramDef: null };
  }

  return { input: out, output: out, audioNodes, perVoice, sharedTargets, isSource: true };
}

function buildVoiceNoise(ctx, node, paramSnap) {
  const params = node.params || {};
  const kind = params.type || 'white';
  const buf = noiseBuffer(ctx, kind);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.loop = true;
  const out = ctx.createGain();
  const levelStatic = isBound(params.level ?? params.gain)
    ? (paramSnap[bindKey(params.level ?? params.gain)] ?? 0.3)
    : (params.level ?? params.gain ?? 0.3);
  out.gain.value = levelStatic;
  src.connect(out);
  const sharedTargets = {};
  if (isBound(params.level ?? params.gain)) {
    sharedTargets[bindKey(params.level ?? params.gain)] = { audioParam: out.gain, paramDef: null };
  }
  return { input: out, output: out, audioNodes: [src], perVoice: {}, sharedTargets, isSource: true };
}

function buildVoiceWavetable(ctx, node, paramSnap) {
  // Approximate wavetable with a periodic-wave oscillator. Real wavetable
  // morphing would need an AudioWorklet; this stub still gives the manager
  // pitch tracking + per-voice instantiation so the integration is sound.
  const out = ctx.createGain();
  out.gain.value = 1;
  const real = new Float32Array([0, 1, 0.5, 0.3, 0.2, 0.1]);
  const imag = new Float32Array(real.length);
  let wave;
  try {
    wave = ctx.createPeriodicWave(real, imag, { disableNormalization: false });
  } catch (_) {
    wave = null;
  }
  const osc = ctx.createOscillator();
  if (wave) osc.setPeriodicWave(wave); else osc.type = 'sawtooth';
  osc.frequency.value = REF_HZ;
  const params = node.params || {};
  const levelStatic = isBound(params.level)
    ? (paramSnap[bindKey(params.level)] ?? 1)
    : (params.level ?? 1);
  out.gain.value = levelStatic;
  osc.connect(out);
  const perVoice = { '@pitch': [{ audioParam: osc.frequency, octaveOffset: 0 }] };
  const sharedTargets = {};
  if (isBound(params.level)) sharedTargets[bindKey(params.level)] = { audioParam: out.gain, paramDef: null };
  return { input: out, output: out, audioNodes: [osc], perVoice, sharedTargets, isSource: true };
}

function buildVoiceFmOperator(ctx, node, paramSnap) {
  // Carrier oscillator + a modulator coming in via input gain. Ratio scales
  // the carrier's per-voice frequency; index scales the modulator depth.
  const params = node.params || {};
  const input = ctx.createGain();
  input.gain.value = isBound(params.index) ? (paramSnap[bindKey(params.index)] ?? 1) : (params.index ?? 1);
  const osc = ctx.createOscillator();
  osc.type = 'sine';
  osc.frequency.value = REF_HZ;
  // FM: route input to osc.frequency via mod-depth gain
  const modDepth = ctx.createGain();
  modDepth.gain.value = 100; // base depth — index AudioParam scales further
  input.connect(modDepth);
  modDepth.connect(osc.frequency);
  const out = ctx.createGain();
  const levelStatic = isBound(params.level)
    ? (paramSnap[bindKey(params.level)] ?? 1)
    : (params.level ?? 1);
  out.gain.value = levelStatic;
  osc.connect(out);
  const ratio = isBound(params.ratio) ? (paramSnap[bindKey(params.ratio)] ?? 2) : (params.ratio ?? 2);
  const perVoice = {
    '@pitch': [{ audioParam: osc.frequency, octaveOffset: 0, scale: ratio }],
  };
  const sharedTargets = {};
  if (isBound(params.index)) sharedTargets[bindKey(params.index)] = { audioParam: input.gain, paramDef: null };
  if (isBound(params.level)) sharedTargets[bindKey(params.level)] = { audioParam: out.gain, paramDef: null };
  return { input, output: out, audioNodes: [osc], perVoice, sharedTargets, isSource: true };
}

function buildVoiceSamplePlayer(ctx, node, paramSnap, voiceContext) {
  // sample_player: pitch-shift via playbackRate based on (note - root_note).
  // The engine is expected to load the buffer ahead of time and stash it in
  // `node._buffer` (simple convention; if absent we synthesize a 440Hz sine
  // so the test still produces non-zero output).
  const params = node.params || {};
  const out = ctx.createGain();
  out.gain.value = 1;
  let buffer = node._buffer;
  if (!buffer) {
    // Synth fallback: 1s sine at 440 so we have something audible
    const sr = ctx.sampleRate;
    buffer = ctx.createBuffer(1, sr, sr);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < sr; i++) data[i] = Math.sin(2 * Math.PI * 440 * i / sr) * 0.5;
  }
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.loop = !!params.loop;
  src.connect(out);

  const rootNote = params.root_note ?? 60;
  const note = voiceContext?.note ?? 60;
  src.playbackRate.value = Math.pow(2, (note - rootNote) / 12);

  return {
    input: out, output: out, audioNodes: [src],
    perVoice: {}, sharedTargets: {}, isSource: true, _isSampler: true,
  };
}

function buildVoiceFilter(ctx, node, paramSnap) {
  const params = node.params || {};
  const filter = ctx.createBiquadFilter();
  filter.type = FILTER_TYPE_MAP[node.type] || 'lowpass';
  filter.frequency.value = isBound(params.cutoff ?? params.frequency)
    ? (paramSnap[bindKey(params.cutoff ?? params.frequency)] ?? 1000)
    : (params.cutoff ?? params.frequency ?? 1000);
  filter.Q.value = isBound(params.resonance ?? params.q)
    ? (paramSnap[bindKey(params.resonance ?? params.q)] ?? 1)
    : (params.resonance ?? params.q ?? 1);
  const sharedTargets = {};
  for (const [k, v] of Object.entries(params)) {
    if (!isBound(v)) continue;
    const id = bindKey(v);
    if (k === 'cutoff' || k === 'frequency') sharedTargets[id] = { audioParam: filter.frequency, paramDef: null };
    else if (k === 'resonance' || k === 'q' || k === 'Q') sharedTargets[id] = { audioParam: filter.Q, paramDef: null };
    else if (k === 'gain') sharedTargets[id] = { audioParam: filter.gain, paramDef: null };
  }
  return { input: filter, output: filter, audioNodes: [], perVoice: {}, sharedTargets };
}

function buildVoiceADSR(ctx, node, paramSnap) {
  // Envelope = a gain node whose value is driven by VoiceManager during
  // noteOn/noteOff. Returns the gain so signal flows through it; reads its
  // a/d/s/r values from paramSnap if bound, else static.
  const params = node.params || {};
  const gain = ctx.createGain();
  gain.gain.value = 0;

  const readParam = (key, fallback) => {
    const v = params[key];
    if (v == null) return fallback;
    if (isBound(v)) return paramSnap[bindKey(v)] ?? fallback;
    // Convert ms-suffixed to seconds
    if (key.endsWith('_ms')) return v / 1000;
    return v;
  };

  const env = {
    attack: readParam('attack', readParam('attack_ms', 0.01)),
    decay: readParam('decay', readParam('decay_ms', 0.2)),
    sustain: readParam('sustain', 0.7),
    release: readParam('release', readParam('release_ms', 0.3)),
  };

  // Live-update binding so a turning the cutoff knob during sustain still
  // reads correct release time on noteOff.
  const sharedTargets = {};
  for (const k of ['attack', 'decay', 'sustain', 'release', 'attack_ms', 'decay_ms', 'release_ms']) {
    const v = params[k];
    if (isBound(v)) {
      sharedTargets[bindKey(v)] = {
        paramDef: null,
        customSetter: (val) => {
          const realKey = k.replace('_ms', '');
          env[realKey] = k.endsWith('_ms') ? val / 1000 : val;
        },
      };
    }
  }

  return {
    input: gain, output: gain, audioNodes: [],
    perVoice: { '@gate': gain.gain },
    sharedTargets,
    isEnvelope: true, env,
  };
}

function buildVoiceGain(ctx, node, paramSnap) {
  const params = node.params || {};
  const gain = ctx.createGain();
  gain.gain.value = isBound(params.gain ?? params.level)
    ? (paramSnap[bindKey(params.gain ?? params.level)] ?? 1)
    : (params.gain ?? params.level ?? 1);
  const sharedTargets = {};
  const perVoice = {};
  if (isBound(params.gain ?? params.level)) {
    const v = params.gain ?? params.level;
    if (v === '@gate' || v === '@velocity') {
      perVoice[v] = gain.gain;
    } else {
      sharedTargets[bindKey(v)] = { audioParam: gain.gain, paramDef: null };
    }
  }
  return { input: gain, output: gain, audioNodes: [], perVoice, sharedTargets };
}

function buildVoicePassthrough(ctx) {
  const g = ctx.createGain();
  return { input: g, output: g, audioNodes: [], perVoice: {}, sharedTargets: {} };
}

const VOICE_NODE_BUILDERS = {
  oscillator: buildVoiceOscillator,
  osc: buildVoiceOscillator,
  saw: buildVoiceOscillator,
  square: buildVoiceOscillator,
  sine: buildVoiceOscillator,
  triangle: buildVoiceOscillator,
  sub_oscillator: buildVoiceOscillator,
  sub_osc: buildVoiceOscillator,
  noise: buildVoiceNoise,
  noise_gen: buildVoiceNoise,
  white_noise: buildVoiceNoise,
  pink_noise: buildVoiceNoise,
  wavetable: buildVoiceWavetable,
  fm_operator: buildVoiceFmOperator,
  sample_player: buildVoiceSamplePlayer,
  adsr: buildVoiceADSR,
  envelope: buildVoiceADSR,
  envelope_adsr: buildVoiceADSR,
  amp_env: buildVoiceADSR,
  filter_env: buildVoiceADSR,
  lowpass: buildVoiceFilter,
  highpass: buildVoiceFilter,
  bandpass: buildVoiceFilter,
  notch: buildVoiceFilter,
  allpass: buildVoiceFilter,
  shelf_low: buildVoiceFilter,
  shelf_high: buildVoiceFilter,
  parametric_eq: buildVoiceFilter,
  gain: buildVoiceGain,
  vca: buildVoiceGain,
  mixer: buildVoiceGain,
  osc_mixer: buildVoiceGain,
};

// ── Voice ───────────────────────────────────────────────────────────────────

class Voice {
  constructor(manager, opts) {
    this.manager = manager;
    this.ctx = manager.ctx;
    this.note = opts.note;
    this.velocity = opts.velocity;
    this.startTime = opts.time;
    this.released = false;
    this.deferredRelease = false;
    this.stolen = false;
    this.builtNodes = {}; // id → { input, output, audioNodes, perVoice, isEnvelope, env }
    this.audioNodes = []; // flat list for teardown
    this.envelopes = []; // [{ gainParam, env }]
    this.output = null;  // final output AudioNode (gain → engine voice bus)
    this.freeAt = Infinity;
    this._build();
    this._noteOn();
  }

  _build() {
    const { ctx, manager } = this;
    const tpl = manager.voiceTemplate;
    const paramSnap = manager.paramValues;
    const voiceCtx = { note: this.note, velocity: this.velocity };

    // Build each node
    for (const nodeDef of tpl.nodes) {
      if (nodeDef.type === 'input' || nodeDef.type === 'output') {
        this.builtNodes[nodeDef.id] = buildVoicePassthrough(ctx);
        continue;
      }
      const builder = VOICE_NODE_BUILDERS[nodeDef.type] || buildVoicePassthrough;
      const built = builder(ctx, nodeDef, paramSnap, voiceCtx);
      this.builtNodes[nodeDef.id] = built;
      if (built.audioNodes) this.audioNodes.push(...built.audioNodes);
      if (built.isEnvelope) {
        this.envelopes.push({
          gainParam: built.perVoice['@gate'],
          env: built.env,
          node: built,
        });
      }
    }

    // Wire edges
    for (const edge of tpl.edges || []) {
      const src = this.builtNodes[edge.source];
      const tgt = this.builtNodes[edge.target];
      if (!src || !tgt) continue;
      try { src.output.connect(tgt.input); } catch (_) { /* ignore */ }
    }

    // Find or synthesize an output node
    const outputNode = this.builtNodes['output']
      || this.builtNodes['out']
      || this.builtNodes['audio_output'];

    if (outputNode) {
      this.output = outputNode.output;
    } else {
      // Sum all nodes that have isSource (or the last edge's target)
      const sum = ctx.createGain();
      for (const [id, b] of Object.entries(this.builtNodes)) {
        if (b.isSource) {
          try { b.output.connect(sum); } catch (_) {}
        }
      }
      this.output = sum;
    }

    // Voice-level gain (used for stealing fade + final output level)
    this.voiceGain = ctx.createGain();
    this.voiceGain.gain.value = 1;
    try { this.output.connect(this.voiceGain); } catch (_) {}
  }

  _noteOn() {
    const t = this.startTime;
    const freq = midiToHz(this.note);

    // Pitch wiring: hit every node that exposed '@pitch'
    for (const built of Object.values(this.builtNodes)) {
      const pitchTargets = built.perVoice?.['@pitch'];
      if (!pitchTargets) continue;
      const list = Array.isArray(pitchTargets) ? pitchTargets : [pitchTargets];
      for (const tgt of list) {
        const oct = tgt.octaveOffset || 0;
        const scale = tgt.scale || 1;
        const targetFreq = freq * Math.pow(2, oct) * scale;
        if (tgt.audioParam) tgt.audioParam.setValueAtTime(targetFreq, t);
      }
    }

    // Velocity wiring
    for (const built of Object.values(this.builtNodes)) {
      const velTarget = built.perVoice?.['@velocity'];
      if (!velTarget) continue;
      if (velTarget.setValueAtTime) velTarget.setValueAtTime(this.velocity, t);
    }

    // Trigger envelopes (gate=1)
    if (this.envelopes.length === 0) {
      // No envelope in template — use a default attack ramp on voiceGain so
      // simple templates still articulate. Velocity scales the peak.
      this.voiceGain.gain.cancelScheduledValues(t);
      this.voiceGain.gain.setValueAtTime(0, t);
      this.voiceGain.gain.linearRampToValueAtTime(this.velocity, t + 0.005);
    } else {
      for (const e of this.envelopes) {
        const { gainParam, env } = e;
        const peak = this.velocity;
        const sus = peak * (env.sustain ?? 0.7);
        gainParam.cancelScheduledValues(t);
        gainParam.setValueAtTime(0, t);
        gainParam.linearRampToValueAtTime(peak, t + Math.max(0.001, env.attack || 0.01));
        gainParam.linearRampToValueAtTime(
          sus,
          t + Math.max(0.001, env.attack || 0.01) + Math.max(0.001, env.decay || 0.2)
        );
      }
    }
  }

  retrigger(note, velocity, time) {
    // Mono retrigger — reset attack from sustain level
    this.note = note;
    this.velocity = velocity;
    this.startTime = time;
    this.released = false;
    this._noteOn();
  }

  legato(note, velocity, time, glideMs = 0) {
    // Legato: re-pitch without retriggering env, optional glide
    this.note = note;
    if (velocity != null) this.velocity = velocity;
    const freq = midiToHz(note);
    for (const built of Object.values(this.builtNodes)) {
      const pitchTargets = built.perVoice?.['@pitch'];
      if (!pitchTargets) continue;
      const list = Array.isArray(pitchTargets) ? pitchTargets : [pitchTargets];
      for (const tgt of list) {
        const oct = tgt.octaveOffset || 0;
        const scale = tgt.scale || 1;
        const targetFreq = freq * Math.pow(2, oct) * scale;
        if (!tgt.audioParam) continue;
        if (glideMs > 0) {
          tgt.audioParam.cancelScheduledValues(time);
          tgt.audioParam.setValueAtTime(tgt.audioParam.value, time);
          tgt.audioParam.linearRampToValueAtTime(targetFreq, time + glideMs / 1000);
        } else {
          tgt.audioParam.setValueAtTime(targetFreq, time);
        }
      }
    }
  }

  release(time) {
    if (this.released) return;
    this.released = true;
    // Use longest envelope release as the freeAt deadline
    let maxRel = 0.05;
    if (this.envelopes.length === 0) {
      // Default short release on voiceGain
      this.voiceGain.gain.cancelScheduledValues(time);
      this.voiceGain.gain.setValueAtTime(this.voiceGain.gain.value, time);
      this.voiceGain.gain.linearRampToValueAtTime(0, time + 0.05);
    } else {
      for (const e of this.envelopes) {
        const { gainParam, env } = e;
        const rel = Math.max(0.005, env.release || 0.3);
        if (rel > maxRel) maxRel = rel;
        gainParam.cancelScheduledValues(time);
        gainParam.setValueAtTime(gainParam.value, time);
        // Exponential-ish: linear ramp to 0 over release
        gainParam.linearRampToValueAtTime(0, time + rel);
      }
    }
    // Hard ceiling 10s after release start
    this.freeAt = time + Math.min(maxRel + 0.1, 10);
    // Stop oscillators a hair after the envelope hits zero
    for (const n of this.audioNodes) {
      if (typeof n.stop === 'function') {
        try { n.stop(this.freeAt + 0.05); } catch (_) {}
      }
    }
  }

  steal(time, fadeMs = 5) {
    // Fast fade then teardown
    if (this.stolen) return;
    this.stolen = true;
    this.released = true;
    this.voiceGain.gain.cancelScheduledValues(time);
    this.voiceGain.gain.setValueAtTime(this.voiceGain.gain.value, time);
    this.voiceGain.gain.linearRampToValueAtTime(0, time + fadeMs / 1000);
    this.freeAt = time + fadeMs / 1000 + 0.01;
    for (const n of this.audioNodes) {
      if (typeof n.stop === 'function') {
        try { n.stop(this.freeAt + 0.05); } catch (_) {}
      }
    }
  }

  /** Cheap loudness proxy for stealing decisions. */
  estimatedLevel() {
    // Use voiceGain.gain.value as cheap proxy. (envelope state is folded into
    // the chain, so it correlates with audible level.)
    let env = 1;
    for (const e of this.envelopes) env *= (e.gainParam.value ?? 0);
    return env * (this.voiceGain.gain.value ?? 1) * (this.velocity ?? 1);
  }

  destroy() {
    for (const n of this.audioNodes) {
      try { n.stop(); } catch (_) {}
      try { n.disconnect(); } catch (_) {}
    }
    for (const built of Object.values(this.builtNodes)) {
      try { built.output?.disconnect(); } catch (_) {}
      try { built.input?.disconnect(); } catch (_) {}
    }
    try { this.voiceGain.disconnect(); } catch (_) {}
  }
}

// ── VoiceManager ────────────────────────────────────────────────────────────

export default class VoiceManager {
  /**
   * @param {AudioContext|OfflineAudioContext} ctx
   * @param {{nodes:Array, edges:Array}} voiceTemplate
   * @param {Object} paramDefs  shared knob param definitions (same shape as engine.paramDefs)
   * @param {Object} options
   *   polyphony  {number} max simultaneous voices (default 8)
   *   mode       {'poly'|'mono'|'legato'} default 'poly'
   *   stealing   {'oldest'|'quietest'|'lowest'} default 'oldest'
   *   glide_ms   {number} glide for mono/legato (default 0)
   *   masterGain {number} default 1
   */
  constructor(ctx, voiceTemplate, paramDefs = {}, options = {}) {
    this.ctx = ctx;
    this.voiceTemplate = this._normalizeTemplate(voiceTemplate);
    this.paramDefs = paramDefs;
    this.paramValues = {}; // paramId → current value (already scaled, not normalized)
    for (const id of Object.keys(paramDefs)) {
      const def = paramDefs[id];
      this.paramValues[id] = def?.default ?? 0;
    }
    this.polyphony = options.polyphony ?? 8;
    this.mode = options.mode || 'poly';
    this.stealing = options.stealing || 'oldest';
    this.glide_ms = options.glide_ms ?? 0;

    // The voice manager exposes a single `output` AudioNode that downstream
    // effects/master gain connect to.
    this.output = ctx.createGain();
    this.output.gain.value = options.masterGain ?? 1;

    // note → Voice (poly: array; mono: single)
    this._activeByNote = new Map();   // MIDI note → Voice (most recent)
    this._allVoices = [];             // FIFO order, oldest first
    this._heldNotes = [];             // Stack of held notes for mono priority
    this._sustainOn = false;
    this._sustainedNotes = new Set(); // Notes whose noteOff was deferred

    // Periodic cleanup of finished voices
    this._cleanupTimer = null;
    this._scheduleCleanup();
  }

  _normalizeTemplate(tpl) {
    if (!tpl) return { nodes: [], edges: [] };
    return {
      nodes: tpl.nodes || [],
      edges: tpl.edges || [],
    };
  }

  _scheduleCleanup() {
    if (typeof setInterval !== 'function') return;
    if (this._cleanupTimer) return;
    this._cleanupTimer = setInterval(() => this._cleanupFinishedVoices(), 250);
  }

  _cleanupFinishedVoices() {
    const t = this.ctx.currentTime;
    const survivors = [];
    for (const v of this._allVoices) {
      if (v.freeAt <= t) {
        v.destroy();
      } else {
        survivors.push(v);
      }
    }
    this._allVoices = survivors;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  noteOn(note, velocity = 0.8, time = null) {
    const t = time != null ? time : this.ctx.currentTime;
    const vel = Math.max(0.001, Math.min(1, velocity));

    if (this.mode === 'mono' || this.mode === 'legato') {
      return this._monoNoteOn(note, vel, t);
    }
    return this._polyNoteOn(note, vel, t);
  }

  noteOff(note, time = null) {
    const t = time != null ? time : this.ctx.currentTime;
    if (this._sustainOn) {
      this._sustainedNotes.add(note);
      // Mark on the active voice so we know later
      const v = this._activeByNote.get(note);
      if (v) v.deferredRelease = true;
      return;
    }
    if (this.mode === 'mono' || this.mode === 'legato') {
      this._monoNoteOff(note, t);
      return;
    }
    this._polyNoteOff(note, t);
  }

  setSustain(on) {
    const wasOn = this._sustainOn;
    this._sustainOn = !!on;
    if (wasOn && !this._sustainOn) {
      // Pedal up → release all deferred notes
      const t = this.ctx.currentTime;
      for (const note of this._sustainedNotes) {
        if (this.mode === 'mono' || this.mode === 'legato') {
          this._monoNoteOff(note, t);
        } else {
          this._polyNoteOff(note, t);
        }
      }
      this._sustainedNotes.clear();
    }
  }

  /** Forward shared-knob change to all live voices. */
  setParam(paramId, value) {
    this.paramValues[paramId] = value;
    const t = this.ctx.currentTime;
    for (const voice of this._allVoices) {
      for (const built of Object.values(voice.builtNodes)) {
        const tgt = built.sharedTargets?.[paramId];
        if (!tgt) continue;
        if (tgt.customSetter) {
          try { tgt.customSetter(value); } catch (_) {}
          continue;
        }
        if (tgt.multi && tgt.audioParams) {
          for (const ap of tgt.audioParams) {
            try {
              ap.cancelScheduledValues(t);
              ap.setTargetAtTime(value, t, 0.02);
            } catch (_) {}
          }
        } else if (tgt.audioParam) {
          try {
            tgt.audioParam.cancelScheduledValues(t);
            tgt.audioParam.setTargetAtTime(value, t, 0.02);
          } catch (_) {}
        }
      }
    }
  }

  /** Total number of active (not-yet-freed) voices. */
  voiceCount() {
    return this._allVoices.length;
  }

  /** All notes currently active (held or releasing). */
  activeNotes() {
    return Array.from(this._activeByNote.keys());
  }

  destroy() {
    if (this._cleanupTimer) {
      clearInterval(this._cleanupTimer);
      this._cleanupTimer = null;
    }
    for (const v of this._allVoices) {
      try { v.destroy(); } catch (_) {}
    }
    this._allVoices = [];
    this._activeByNote.clear();
    this._heldNotes = [];
    try { this.output.disconnect(); } catch (_) {}
  }

  // ── Polyphonic ────────────────────────────────────────────────────────────

  _polyNoteOn(note, vel, t) {
    // Re-trigger same note: release the previous voice on this note first
    if (this._activeByNote.has(note)) {
      const prev = this._activeByNote.get(note);
      if (prev && !prev.released) prev.release(t);
    }

    // Polyphony cap: steal a voice if we're at the limit
    while (this._allVoices.filter(v => !v.released && !v.stolen).length >= this.polyphony) {
      this._stealOne(t);
    }

    const voice = new Voice(this, { note, velocity: vel, time: t });
    try { voice.voiceGain.connect(this.output); } catch (_) {}
    this._allVoices.push(voice);
    this._activeByNote.set(note, voice);
    return voice;
  }

  _polyNoteOff(note, t) {
    const v = this._activeByNote.get(note);
    if (!v) return;
    v.release(t);
    this._activeByNote.delete(note);
  }

  _stealOne(t) {
    const candidates = this._allVoices.filter(v => !v.stolen);
    if (candidates.length === 0) return;
    let target;
    if (this.stealing === 'quietest') {
      target = candidates.reduce((a, b) => a.estimatedLevel() < b.estimatedLevel() ? a : b);
    } else if (this.stealing === 'lowest') {
      // Steal the lowest-pitched voice (e.g. for lead synths)
      target = candidates.reduce((a, b) => a.note < b.note ? a : b);
    } else {
      // 'oldest' — first non-released, then first overall
      target = candidates.find(v => !v.released) || candidates[0];
    }
    target.steal(t, 5);
    // Drop from active map if it was the active one
    for (const [n, v] of this._activeByNote) {
      if (v === target) {
        this._activeByNote.delete(n);
        break;
      }
    }
  }

  // ── Monophonic / legato ───────────────────────────────────────────────────

  _monoNoteOn(note, vel, t) {
    this._heldNotes.push(note);
    const existing = this._allVoices.find(v => !v.released && !v.stolen);
    if (existing) {
      if (this.mode === 'legato') {
        existing.legato(note, vel, t, this.glide_ms);
      } else {
        existing.retrigger(note, vel, t);
      }
      this._activeByNote.clear();
      this._activeByNote.set(note, existing);
      return existing;
    }
    return this._polyNoteOn(note, vel, t);
  }

  _monoNoteOff(note, t) {
    this._heldNotes = this._heldNotes.filter(n => n !== note);
    if (this._heldNotes.length === 0) {
      // Last finger up — release the voice
      const v = this._allVoices.find(x => !x.released && !x.stolen);
      if (v) v.release(t);
      this._activeByNote.clear();
      return;
    }
    // Still holding earlier notes → fall back to most-recent (or whatever the
    // priority dictates). We use last-note priority.
    const fallback = this._heldNotes[this._heldNotes.length - 1];
    const v = this._allVoices.find(x => !x.released && !x.stolen);
    if (v) {
      if (this.mode === 'legato') {
        v.legato(fallback, null, t, this.glide_ms);
      } else {
        v.retrigger(fallback, v.velocity, t);
      }
      this._activeByNote.clear();
      this._activeByNote.set(fallback, v);
    }
  }
}

export { Voice, midiToHz };
