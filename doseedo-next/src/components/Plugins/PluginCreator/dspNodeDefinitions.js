// ── All supported DSP node types with their parameter schemas ──────────────
// v2: Added io specs, min/max bounds, modulatable flags, and skew hints
// Modulation targets use "nodeId.paramKey" notation (e.g. "filter1.cutoff")

export const NODE_CATEGORIES = {
  'Oscillators': [
    {
      type: 'oscillator', label: 'Oscillator',
      io: { in: 0, out: 1 },
      params: {
        waveform:    { default: 'sawtooth', options: ['sine','sawtooth','square','triangle'] },
        detune:      { default: 0, unit: 'cents', min: -100, max: 100, modulatable: true },
        gain:        { default: 0.5, min: 0, max: 1, modulatable: true },
        unison:      { default: 1, min: 1, max: 16 },
        unison_spread: { default: 15, unit: 'cents', min: 0, max: 100 },
      },
    },
    {
      type: 'sub_oscillator', label: 'Sub Oscillator',
      io: { in: 0, out: 1 },
      params: {
        waveform: { default: 'sine', options: ['sine','sawtooth','square','triangle'] },
        octave:   { default: -1, min: -2, max: 0 },
        gain:     { default: 0.4, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'noise', label: 'Noise Generator',
      io: { in: 0, out: 1 },
      params: {
        type: { default: 'white', options: ['white','pink'] },
        gain: { default: 0.3, min: 0, max: 1, modulatable: true },
      },
    },
  ],
  'Envelopes & Modulation': [
    {
      type: 'adsr', label: 'ADSR Envelope',
      io: { in: 1, out: 1 },
      params: {
        attack:  { default: 0.01, unit: 's', min: 0.001, max: 5, skew: 0.3, modulatable: true },
        decay:   { default: 0.2, unit: 's', min: 0.001, max: 5, skew: 0.3, modulatable: true },
        sustain: { default: 0.7, min: 0, max: 1, modulatable: true },
        release: { default: 0.3, unit: 's', min: 0.001, max: 10, skew: 0.3, modulatable: true },
      },
    },
    {
      type: 'lfo', label: 'LFO',
      io: { in: 0, out: 1 },
      params: {
        rate:  { default: 2, unit: 'Hz', min: 0.01, max: 50, skew: 0.3, modulatable: true },
        depth: { default: 0.5, min: 0, max: 1, modulatable: true },
        shape: { default: 'sine', options: ['sine','triangle','sawtooth','square'] },
      },
    },
  ],
  'Filters': [
    {
      type: 'lowpass', label: 'Lowpass',
      io: { in: 1, out: 1 },
      params: {
        cutoff:    { default: 1000, unit: 'Hz', min: 20, max: 20000, skew: 0.25, modulatable: true },
        resonance: { default: 0.5, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'highpass', label: 'Highpass',
      io: { in: 1, out: 1 },
      params: {
        cutoff:    { default: 1000, unit: 'Hz', min: 20, max: 20000, skew: 0.25, modulatable: true },
        resonance: { default: 0.5, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'bandpass', label: 'Bandpass',
      io: { in: 1, out: 1 },
      params: {
        cutoff:    { default: 1000, unit: 'Hz', min: 20, max: 20000, skew: 0.25, modulatable: true },
        resonance: { default: 0.5, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'notch', label: 'Notch',
      io: { in: 1, out: 1 },
      params: {
        cutoff:    { default: 1000, unit: 'Hz', min: 20, max: 20000, skew: 0.25, modulatable: true },
        resonance: { default: 0.5, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'allpass', label: 'Allpass',
      io: { in: 1, out: 1 },
      params: {
        cutoff:    { default: 1000, unit: 'Hz', min: 20, max: 20000, skew: 0.25, modulatable: true },
        resonance: { default: 0.5, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'ladder', label: 'Ladder',
      io: { in: 1, out: 1 },
      params: {
        cutoff:    { default: 2000, unit: 'Hz', min: 20, max: 20000, skew: 0.25, modulatable: true },
        resonance: { default: 0.3, min: 0, max: 1, modulatable: true },
        mode:      { default: 'LP24', options: ['LP12','LP24','HP12','HP24','BP12','BP24'] },
      },
    },
    {
      type: 'comb', label: 'Comb',
      io: { in: 1, out: 1 },
      params: {
        delay_ms: { default: 5, unit: 'ms', min: 0.1, max: 100, skew: 0.5, modulatable: true },
        feedback:  { default: 0.5, min: 0, max: 0.99, modulatable: true },
      },
    },
    {
      type: 'shelf_low', label: 'Low Shelf',
      io: { in: 1, out: 1 },
      params: {
        cutoff:  { default: 200, unit: 'Hz', min: 20, max: 2000, skew: 0.4, modulatable: true },
        gain_db: { default: 0, unit: 'dB', min: -18, max: 18, modulatable: true },
      },
    },
    {
      type: 'shelf_high', label: 'High Shelf',
      io: { in: 1, out: 1 },
      params: {
        cutoff:  { default: 8000, unit: 'Hz', min: 1000, max: 20000, skew: 0.4, modulatable: true },
        gain_db: { default: 0, unit: 'dB', min: -18, max: 18, modulatable: true },
      },
    },
    {
      type: 'parametric_eq', label: 'Parametric EQ',
      io: { in: 1, out: 1 },
      params: {
        freq:    { default: 1000, unit: 'Hz', min: 20, max: 20000, skew: 0.25, modulatable: true },
        gain_db: { default: 0, unit: 'dB', min: -18, max: 18, modulatable: true },
        q:       { default: 1, min: 0.1, max: 10, skew: 0.5, modulatable: true },
      },
    },
    {
      type: 'filter_morph', label: 'Filter Morph',
      io: { in: 1, out: 1 },
      params: {
        cutoff:    { default: 1000, unit: 'Hz', min: 20, max: 20000, skew: 0.25, modulatable: true },
        resonance: { default: 0.5, min: 0, max: 1, modulatable: true },
        morph:     { default: 0, min: 0, max: 1, modulatable: true },
      },
    },
  ],

  'Dynamics': [
    {
      type: 'compressor', label: 'Compressor',
      io: { in: 1, out: 1 },
      params: {
        threshold_db:   { default: -18, unit: 'dB', min: -60, max: 0 },
        ratio:          { default: 4, min: 1, max: 20 },
        attack_ms:      { default: 10, unit: 'ms', min: 0.1, max: 300, skew: 0.4 },
        release_ms:     { default: 150, unit: 'ms', min: 1, max: 2000, skew: 0.4 },
        makeup_db:      { default: 0, unit: 'dB', min: 0, max: 40, modulatable: true },
        sidechain_input: { default: false },
        knee_db:        { default: 6, unit: 'dB', min: 0, max: 24 },
      },
    },
    {
      type: 'limiter', label: 'Limiter',
      io: { in: 1, out: 1 },
      params: {
        threshold_db: { default: -3, unit: 'dB', min: -30, max: 0 },
        release_ms:   { default: 100, unit: 'ms', min: 1, max: 1000, skew: 0.4 },
      },
    },
    {
      type: 'gate', label: 'Gate',
      io: { in: 1, out: 1 },
      params: {
        threshold_db: { default: -40, unit: 'dB', min: -80, max: 0 },
        attack_ms:    { default: 1, unit: 'ms', min: 0.1, max: 100, skew: 0.4 },
        release_ms:   { default: 100, unit: 'ms', min: 1, max: 2000, skew: 0.4 },
      },
    },
    {
      type: 'expander', label: 'Expander',
      io: { in: 1, out: 1 },
      params: {
        threshold_db: { default: -40, unit: 'dB', min: -80, max: 0 },
        ratio:        { default: 2, min: 1, max: 10 },
        attack_ms:    { default: 5, unit: 'ms', min: 0.1, max: 300, skew: 0.4 },
        release_ms:   { default: 100, unit: 'ms', min: 1, max: 2000, skew: 0.4 },
      },
    },
    {
      type: 'envelope_follower', label: 'Env Follower',
      io: { in: 1, out: 0 }, // signal processor, outputs CV-like value
      params: {
        attack_ms:  { default: 10, unit: 'ms', min: 0.1, max: 500, skew: 0.4 },
        release_ms: { default: 100, unit: 'ms', min: 1, max: 2000, skew: 0.4 },
      },
    },
  ],

  'Time': [
    {
      type: 'delay', label: 'Delay',
      io: { in: 1, out: 1 },
      params: {
        time_ms:  { default: 350, unit: 'ms', min: 0, max: 5000, skew: 0.4, modulatable: true },
        feedback: { default: 0.4, min: 0, max: 0.99, modulatable: true },
        mix:      { default: 0.3, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'multitap_delay', label: 'Multitap Delay',
      io: { in: 1, out: 1 },
      params: {
        feedback: { default: 0.3, min: 0, max: 0.99, modulatable: true },
        mix:      { default: 0.3, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'ping_pong_delay', label: 'Ping-Pong',
      io: { in: 1, out: 2 },
      params: {
        time_ms:  { default: 300, unit: 'ms', min: 0, max: 5000, skew: 0.4, modulatable: true },
        feedback: { default: 0.4, min: 0, max: 0.99, modulatable: true },
        mix:      { default: 0.3, min: 0, max: 1, modulatable: true },
        spread:   { default: 0.8, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'reverb', label: 'Reverb',
      io: { in: 1, out: 2 },
      params: {
        room_size: { default: 0.5, min: 0, max: 1, modulatable: true },
        damping:   { default: 0.5, min: 0, max: 1, modulatable: true },
        width:     { default: 1, min: 0, max: 1, modulatable: true },
        mix:       { default: 0.3, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'convolution', label: 'Convolution',
      io: { in: 1, out: 2 },
      params: {
        ir_file: { default: '', type: 'file', accept: '.wav,.aiff,.flac' },
        mix:     { default: 0.5, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'algo_reverb', label: 'Algo Reverb',
      io: { in: 1, out: 2 },
      params: {
        algorithm:  { default: 'hall', options: ['room','hall','chamber','plate'] },
        decay_time: { default: 2.5, unit: 's',  min: 0.1, max: 20,  skew: 0.4, modulatable: true },
        pre_delay:  { default: 0,   unit: 'ms', min: 0,   max: 500, skew: 0.5, modulatable: true },
        damping:    { default: 0.4, min: 0, max: 1, modulatable: true },
        diffusion:  { default: 0.8, min: 0, max: 1, modulatable: true },
        width:      { default: 0.7, min: 0, max: 1, modulatable: true },
        mix:        { default: 0.3, min: 0, max: 1, modulatable: true },
      },
    },
  ],

  'Modulation': [
    {
      type: 'chorus', label: 'Chorus',
      io: { in: 1, out: 2 },
      params: {
        rate_hz: { default: 1.2, unit: 'Hz', min: 0.01, max: 20, skew: 0.4, modulatable: true },
        depth:   { default: 0.4, min: 0, max: 1, modulatable: true },
        mix:     { default: 0.5, min: 0, max: 1, modulatable: true },
        voices:  { default: 2, min: 2, max: 8, step: 1 },
      },
    },
    {
      type: 'flanger', label: 'Flanger',
      io: { in: 1, out: 1 },
      params: {
        rate_hz:  { default: 0.3, unit: 'Hz', min: 0.01, max: 20, skew: 0.4, modulatable: true },
        depth:    { default: 0.5, min: 0, max: 1, modulatable: true },
        feedback: { default: 0.5, min: -0.99, max: 0.99, modulatable: true },
        mix:      { default: 0.5, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'phaser', label: 'Phaser',
      io: { in: 1, out: 1 },
      params: {
        rate_hz:  { default: 0.5, unit: 'Hz', min: 0.01, max: 20, skew: 0.4, modulatable: true },
        depth:    { default: 0.5, min: 0, max: 1, modulatable: true },
        feedback: { default: 0.3, min: -0.99, max: 0.99, modulatable: true },
        mix:      { default: 0.5, min: 0, max: 1, modulatable: true },
        stages:   { default: 4, min: 2, max: 12, step: 2 },
      },
    },
    {
      type: 'tremolo', label: 'Tremolo',
      io: { in: 1, out: 1 },
      params: {
        rate_hz: { default: 4, unit: 'Hz', min: 0.01, max: 30, skew: 0.4, modulatable: true },
        depth:   { default: 0.5, min: 0, max: 1, modulatable: true },
        shape:   { default: 'sine', options: ['sine','triangle','square'] },
      },
    },
    {
      type: 'ring_mod', label: 'Ring Mod',
      io: { in: 1, out: 1 },
      params: {
        freq_hz: { default: 440, unit: 'Hz', min: 1, max: 10000, skew: 0.3, modulatable: true },
        mix:     { default: 0.5, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'lfo', label: 'LFO',
      io: { in: 0, out: 0 }, // modulation-only node
      params: {
        rate_hz:    { default: 1, unit: 'Hz', min: 0.01, max: 100, skew: 0.3, modulatable: false },
        shape:      { default: 'sine', options: ['sine','triangle','saw','square','random'] },
        depth:      { default: 0.5, min: 0, max: 1, modulatable: false },
        // target: use "nodeId.paramKey" notation, e.g. "filter1.cutoff"
        target:     { default: '', type: 'mod_target' },
        sync_to_bpm: { default: false },
        sync_rate:  { default: '1/4', options: ['1/16','1/8','1/4','1/2','1','2','4'] },
        phase:      { default: 0, unit: 'deg', min: 0, max: 360 },
      },
    },
    {
      type: 'macro', label: 'Macro',
      io: { in: 0, out: 0 }, // modulation-only node
      params: {
        value:    { default: 0.5, min: 0, max: 1 },
        // targets: use "nodeId.paramKey" notation
        target_1: { default: '', type: 'mod_target' },
        amount_1: { default: 1, min: -1, max: 1 },
        target_2: { default: '', type: 'mod_target' },
        amount_2: { default: 1, min: -1, max: 1 },
      },
    },
    {
      type: 'mod_envelope', label: 'Mod Envelope',
      io: { in: 0, out: 0 }, // modulation-only node, triggers on gate
      params: {
        attack_ms:  { default: 10, unit: 'ms', min: 0.1, max: 5000, skew: 0.3 },
        decay_ms:   { default: 300, unit: 'ms', min: 1, max: 5000, skew: 0.3 },
        sustain:    { default: 0.7, min: 0, max: 1 },
        release_ms: { default: 500, unit: 'ms', min: 1, max: 10000, skew: 0.3 },
        amount:     { default: 1, min: -1, max: 1 },
        target:     { default: '', type: 'mod_target' },
      },
    },
  ],

  'Distortion': [
    {
      type: 'overdrive', label: 'Overdrive',
      io: { in: 1, out: 1 },
      params: {
        drive: { default: 0.3, min: 0, max: 1, modulatable: true },
        tone:  { default: 0.6, min: 0, max: 1, modulatable: true },
        mix:   { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'waveshaper', label: 'Waveshaper',
      io: { in: 1, out: 1 },
      params: {
        curve:  { default: 'tanh', options: ['tanh','atan','cubic','hard_clip'] },
        amount: { default: 0.5, min: 0, max: 1, modulatable: true },
        mix:    { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'bitcrusher', label: 'Bitcrusher',
      io: { in: 1, out: 1 },
      params: {
        bit_depth:       { default: 8, min: 1, max: 24, step: 1, modulatable: true },
        sample_rate_div: { default: 4, min: 1, max: 64, step: 1, modulatable: true },
        mix:             { default: 0.5, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'saturation', label: 'Saturation',
      io: { in: 1, out: 1 },
      params: {
        amount:    { default: 0.4, min: 0, max: 1, modulatable: true },
        asymmetry: { default: 0, min: -1, max: 1, modulatable: true },
        mix:       { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'foldback', label: 'Foldback',
      io: { in: 1, out: 1 },
      params: {
        threshold: { default: 0.5, min: 0.01, max: 1, modulatable: true },
        mix:       { default: 0.5, min: 0, max: 1, modulatable: true },
      },
    },
  ],

  'Utility': [
    {
      type: 'gain', label: 'Gain',
      io: { in: 1, out: 1 },
      params: {
        gain_db: { default: 0, unit: 'dB', min: -60, max: 12, modulatable: true },
      },
    },
    {
      type: 'pan', label: 'Pan',
      io: { in: 1, out: 2 },
      params: {
        pan: { default: 0, min: -1, max: 1, modulatable: true },
      },
    },
    {
      type: 'mix', label: 'Dry/Wet',
      io: { in: 2, out: 1 },
      params: {
        dry: { default: 1, min: 0, max: 1, modulatable: true },
        wet: { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'dc_blocker', label: 'DC Blocker',
      io: { in: 1, out: 1 },
      params: {},
    },
    {
      type: 'sidechain', label: 'Sidechain Input',
      io: { in: 0, out: 1 }, // receives sidechain bus
      params: {
        gain_db: { default: 0, unit: 'dB', min: -60, max: 12, modulatable: true },
      },
    },
    {
      type: 'midi_cc', label: 'MIDI CC Map',
      io: { in: 0, out: 0 }, // MIDI-only node
      params: {
        cc_number: { default: 1, min: 0, max: 127, step: 1 },
        // target: use "nodeId.paramKey" notation
        target:    { default: '', type: 'mod_target' },
        min_val:   { default: 0, min: 0, max: 1 },
        max_val:   { default: 1, min: 0, max: 1 },
      },
    },
  ],

  'Synthesis': [
    {
      type: 'oscillator', label: 'Oscillator',
      io: { in: 0, out: 1 },
      params: {
        waveform: { default: 'saw', options: ['sine','saw','square','triangle','noise'] },
        detune:   { default: 0, unit: 'cents', min: -100, max: 100, modulatable: true },
        level:    { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'noise', label: 'Noise',
      io: { in: 0, out: 1 },
      params: {
        type:  { default: 'white', options: ['white','pink','brown'] },
        level: { default: 0.5, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'wavetable', label: 'Wavetable',
      io: { in: 0, out: 1 },
      params: {
        table:    { default: 'basic_shapes', type: 'wavetable_ref' },
        position: { default: 0.5, min: 0, max: 1, modulatable: true },
        level:    { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'fm_operator', label: 'FM Operator',
      io: { in: 1, out: 1 }, // in = modulator signal
      params: {
        ratio: { default: 2, min: 0.5, max: 16, skew: 0.5, modulatable: true },
        index: { default: 1, min: 0, max: 10, modulatable: true },
        level: { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'envelope_adsr', label: 'ADSR',
      io: { in: 0, out: 0 }, // modulation-only, targets amp or pitch
      params: {
        attack_ms:  { default: 10, unit: 'ms', min: 0.1, max: 5000, skew: 0.3 },
        decay_ms:   { default: 300, unit: 'ms', min: 1, max: 5000, skew: 0.3 },
        sustain:    { default: 0.7, min: 0, max: 1 },
        release_ms: { default: 500, unit: 'ms', min: 1, max: 10000, skew: 0.3 },
        target:     { default: 'amp', options: ['amp', 'pitch', 'filter'] },
      },
    },
    {
      type: 'sample_player', label: 'Sampler',
      io: { in: 0, out: 1 },
      params: {
        file:      { default: '', type: 'file', accept: '.wav,.aiff,.flac,.mp3' },
        root_note: { default: 60, min: 0, max: 127, step: 1 },
        loop:      { default: false },
        reverse:   { default: false },
      },
    },
    {
      type: 'sub_oscillator', label: 'Sub Osc',
      io: { in: 0, out: 1 },
      params: {
        octave:   { default: -1, options: [-2, -1] },
        waveform: { default: 'sine', options: ['sine', 'square', 'triangle'] },
        level:    { default: 0.5, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'unison', label: 'Unison',
      io: { in: 1, out: 1 },
      params: {
        voices:       { default: 1, min: 1, max: 8, step: 1 },
        detune_cents: { default: 25, unit: 'cents', min: 0, max: 100, modulatable: true },
        spread:       { default: 0.5, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'osc_mixer', label: 'Osc Mixer',
      io: { in: -1, out: 1 }, // -1 = variable (set by params.inputs)
      params: {
        inputs: { default: 2, min: 2, max: 8, step: 1 },
      },
    },
  ],

  'Routing': [
    {
      type: 'splitter', label: 'Splitter',
      io: { in: 1, out: -1 }, // -1 = variable
      params: {
        outputs: { default: 2, min: 2, max: 8, step: 1 },
      },
    },
    {
      type: 'merger', label: 'Merger',
      io: { in: -1, out: 1 }, // -1 = variable
      params: {
        inputs: { default: 2, min: 2, max: 8, step: 1 },
      },
    },
    {
      type: 'feedback_delay', label: 'Feedback Loop',
      io: { in: 1, out: 1 },
      params: {
        delay_samples: { default: 1, min: 1, max: 2048, step: 1 },
        feedback:      { default: 0.5, min: 0, max: 0.99, modulatable: true },
      },
    },
  ],

  'Math': [
    {
      type: 'math_add', label: 'Add',
      io: { in: 1, out: 1 },
      params: {
        offset: { default: 0, min: -1, max: 1, modulatable: true },
      },
    },
    {
      type: 'math_multiply', label: 'Multiply',
      io: { in: 1, out: 1 },
      params: {
        factor: { default: 1, min: -4, max: 4, modulatable: true },
      },
    },
    {
      type: 'math_constant', label: 'Constant',
      io: { in: 0, out: 1 },
      params: {
        value: { default: 1, min: 0, max: 1 },
      },
    },
    {
      type: 'math_scale', label: 'Scale',
      io: { in: 1, out: 1 },
      params: {
        min: { default: 0, min: -1, max: 1 },
        max: { default: 1, min: -1, max: 1 },
      },
    },
    {
      type: 'math_crossfade', label: 'Crossfade',
      io: { in: 2, out: 1 },
      params: {
        mix: { default: 0.5, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'math_abs', label: 'Abs',
      io: { in: 1, out: 1 },
      params: {},
    },
    {
      type: 'math_rectifier', label: 'Rectifier',
      io: { in: 1, out: 1 },
      params: {
        mode: { default: 'full', options: ['full', 'half'] },
      },
    },
    {
      type: 'math_slew', label: 'Slew Limiter',
      io: { in: 1, out: 1 },
      params: {
        rise: { default: 0.5, min: 0, max: 1, modulatable: true },
        fall: { default: 0.5, min: 0, max: 1, modulatable: true },
      },
    },
  ],

  'Spectral': [
    {
      // SOLA / time-domain pitch shifter (R1). Lower latency (~10ms),
      // gentler artifacts on small shifts; smears transients at extremes.
      type: 'pitch_shift', label: 'Pitch Shift',
      io: { in: 1, out: 1 },
      params: {
        semitones: { default: 0, min: -24, max: 24, step: 1, modulatable: true },
        mix:       { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      // Phase-vocoder pitch shifter (R5). Higher latency (~46ms @ 2048 FFT),
      // better quality at large shifts (±12 to ±24). Use this for creative effects.
      type: 'pitch_shift_pv', label: 'Pitch Shift (PV)',
      io: { in: 1, out: 1 },
      params: {
        semitones: { default: 0, min: -24, max: 24, step: 1, modulatable: true },
        mix:       { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'spectral_filter', label: 'Spectral Filter',
      io: { in: 1, out: 1 },
      params: {
        low_bin:  { default: 0, min: 0, max: 1, modulatable: true },
        high_bin: { default: 1, min: 0, max: 1, modulatable: true },
        mix:      { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'spectral_freeze', label: 'Spectral Freeze',
      io: { in: 1, out: 1 },
      params: {
        freeze: { default: 0, min: 0, max: 1, modulatable: true },
        mix:    { default: 0.5, min: 0, max: 1, modulatable: true },
      },
    },
  ],

  'Analysis': [
    {
      type: 'peak_meter', label: 'Peak Meter',
      io: { in: 1, out: 1 }, // pass-through
      params: {},
    },
    {
      type: 'rms_meter', label: 'RMS Meter',
      io: { in: 1, out: 1 }, // pass-through
      params: {
        window_ms: { default: 50, unit: 'ms', min: 1, max: 500 },
      },
    },
  ],

  'Circuit Models': [
    {
      type: 'circuit_fender_bassman', label: 'Fender Bassman',
      io: { in: 1, out: 1 },
      params: {
        gain:     { default: 1, min: 0.1, max: 5, modulatable: true },
        bass:     { default: 0.5, min: 0, max: 1, modulatable: true },
        mid:      { default: 0.5, min: 0, max: 1, modulatable: true },
        treble:   { default: 0.5, min: 0, max: 1, modulatable: true },
        presence: { default: 0.5, min: 0, max: 1, modulatable: true },
        master:   { default: 0.5, min: 0.05, max: 1, modulatable: true },
      },
    },
    {
      type: 'circuit_pultec_eq', label: 'Pultec EQ',
      io: { in: 1, out: 1 },
      params: {
        low_boost:  { default: 0.3, min: 0, max: 1, modulatable: true },
        low_atten:  { default: 0, min: 0, max: 1, modulatable: true },
        low_freq:   { default: 60, unit: 'Hz', min: 20, max: 200, skew: 0.4 },
        high_boost: { default: 0.3, min: 0, max: 1, modulatable: true },
        high_atten: { default: 0, min: 0, max: 1, modulatable: true },
        high_freq:  { default: 10000, unit: 'Hz', min: 3000, max: 16000, skew: 0.4 },
        output:     { default: 1, min: 0.5, max: 2, modulatable: true },
      },
    },
    {
      type: 'circuit_la2a', label: 'LA-2A Compressor',
      io: { in: 1, out: 1 },
      params: {
        peak_reduction: { default: 0.5, min: 0, max: 1, modulatable: true },
        gain:           { default: 0.5, min: 0, max: 1, modulatable: true },
        mode:           { default: 0, options: [0, 1] },  // 0=compress, 1=limit
        mix:            { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'circuit_1176', label: '1176 Limiter',
      io: { in: 1, out: 1 },
      params: {
        input:   { default: 0.5, min: 0, max: 1, modulatable: true },
        output:  { default: 0.5, min: 0, max: 1, modulatable: true },
        attack:  { default: 0.3, min: 0, max: 1, modulatable: false },
        release: { default: 0.5, min: 0, max: 1, modulatable: false },
        ratio:   { default: 0, options: [0, 0.33, 0.66, 1] },  // 4:1, 8:1, 12:1, 20:1
        mix:     { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'circuit_tape_machine', label: 'Tape Machine',
      io: { in: 1, out: 1 },
      params: {
        input_level: { default: 1.5, min: 0.5, max: 5, modulatable: true },
        speed:       { default: 0.5, min: 0, max: 1, modulatable: false },
        bias:        { default: 0.5, min: 0, max: 1, modulatable: false },
        wow_flutter: { default: 0.2, min: 0, max: 1, modulatable: true },
        mix:         { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'circuit_tube_preamp', label: 'Tube Preamp',
      io: { in: 1, out: 1 },
      params: {
        gain:         { default: 0.5, min: 0.1, max: 5, modulatable: true },
        stages:       { default: 2, min: 1, max: 3, step: 1 },
        bright:       { default: 0.3, min: 0, max: 1, modulatable: true },
        output_level: { default: 0.4, min: 0.05, max: 1, modulatable: true },
      },
    },
  ],

  'Analog Modeling': [
    {
      type: 'wdf_diode_clipper', label: 'Diode Clipper',
      io: { in: 1, out: 1 },
      params: {
        drive:     { default: 2, min: 0.5, max: 10, modulatable: true },
        ideality:  { default: 1.8, min: 1.0, max: 2.5, modulatable: false },
        symmetry:  { default: 0, min: -1, max: 1, modulatable: true },
        mix:       { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'wdf_tube_triode', label: 'Tube Triode',
      io: { in: 1, out: 1 },
      params: {
        drive:  { default: 1, min: 0.1, max: 5, modulatable: true },
        bias:   { default: -1.5, min: -4, max: 0, modulatable: true },
        mix:    { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'wdf_tube_amp', label: 'Tube Preamp',
      io: { in: 1, out: 1 },
      params: {
        gain:         { default: 1, min: 0.1, max: 5, modulatable: true },
        bias:         { default: -1.5, min: -4, max: 0, modulatable: false },
        stages:       { default: 2, min: 1, max: 3, step: 1 },
        output_level: { default: 0.3, min: 0.05, max: 1, modulatable: true },
        mix:          { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'wdf_tone_stack', label: 'Tone Stack',
      io: { in: 1, out: 1 },
      params: {
        bass:   { default: 0.5, min: 0, max: 1, modulatable: true },
        mid:    { default: 0.5, min: 0, max: 1, modulatable: true },
        treble: { default: 0.5, min: 0, max: 1, modulatable: true },
        mix:    { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'wdf_transformer', label: 'Transformer',
      io: { in: 1, out: 1 },
      params: {
        drive:      { default: 1, min: 0.1, max: 5, modulatable: true },
        saturation: { default: 0.5, min: 0.1, max: 1, modulatable: true },
        mix:        { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'wdf_tape_sat', label: 'Tape Saturation',
      io: { in: 1, out: 1 },
      params: {
        input_level: { default: 1.5, min: 0.5, max: 5, modulatable: true },
        bias:        { default: 0.5, min: 0, max: 1, modulatable: false },
        speed:       { default: 0.5, min: 0, max: 1, modulatable: false },
        head_bump:   { default: 0.5, min: 0, max: 1, modulatable: true },
        mix:         { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'wdf_rc_filter', label: 'RC Filter',
      io: { in: 1, out: 1 },
      params: {
        resistance:  { default: 10000, unit: '\u03A9', min: 100, max: 1000000, skew: 0.25 },
        capacitance: { default: 1e-8, unit: 'F', min: 1e-12, max: 1e-5, skew: 0.15 },
        mix:         { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'wdf_rlc_filter', label: 'RLC Filter',
      io: { in: 1, out: 1 },
      params: {
        resistance:  { default: 1000, unit: '\u03A9', min: 10, max: 100000, skew: 0.25 },
        inductance:  { default: 0.01, unit: 'H', min: 1e-6, max: 1, skew: 0.15 },
        capacitance: { default: 1e-7, unit: 'F', min: 1e-12, max: 1e-4, skew: 0.15 },
        mix:         { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'wdf_transistor_clipper', label: 'Transistor Clipper',
      io: { in: 1, out: 1 },
      params: {
        drive: { default: 2, min: 0.5, max: 10, modulatable: true },
        beta:  { default: 150, min: 50, max: 300, modulatable: false },
        fuzz:  { default: 0.5, min: 0, max: 1, modulatable: true },
        mix:   { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
    {
      type: 'wdf_power_supply_sag', label: 'Power Sag',
      io: { in: 1, out: 1 },
      params: {
        sag:      { default: 0.5, min: 0, max: 1, modulatable: true },
        recovery: { default: 0.05, unit: 's', min: 0.01, max: 0.5, skew: 0.4 },
        mix:      { default: 1, min: 0, max: 1, modulatable: true },
      },
    },
  ],
};

// Category colors for node headers
export const CATEGORY_COLORS = {
  Filters:    '#4fc3f7',
  Dynamics:   '#ff8a65',
  Time:       '#81c784',
  Modulation: '#ba68c8',
  Distortion: '#ef5350',
  Utility:    '#90a4ae',
  Synthesis:  '#ffd54f',
  Analysis:   '#7986cb',
  Routing:    '#26c6da',
  Math:       '#78909c',
  Spectral:   '#ab47bc',
  'Circuit Models': '#d4a76a',
  'Analog Modeling': '#e6a03c',
};

export function getCategoryForType(type) {
  for (const [cat, nodes] of Object.entries(NODE_CATEGORIES)) {
    if (nodes.some(n => n.type === type)) return cat;
  }
  return 'Utility';
}

export function getNodeSchema(type) {
  for (const nodes of Object.values(NODE_CATEGORIES)) {
    const found = nodes.find(n => n.type === type);
    if (found) return found;
  }
  return null;
}

// ── Stable node ID generation (survives page refresh via counter store) ──────
// Call initNodeIdCounter({ lastId }) on app load to restore from project state.
let _nodeIdCounter = 0;
export function initNodeIdCounter(savedState = {}) {
  _nodeIdCounter = savedState.lastId ?? 0;
}
export function nextNodeId(type) {
  _nodeIdCounter++;
  return `${type}_${_nodeIdCounter}`;
}
export function getNodeIdCounter() {
  return _nodeIdCounter;
}

// ── Helpers for JUCE codegen ─────────────────────────────────────────────────

/**
 * Returns all modulatable params for a node type as { paramKey, schema } pairs.
 */
export function getModulatableParams(type) {
  const schema = getNodeSchema(type);
  if (!schema) return [];
  return Object.entries(schema.params)
    .filter(([, p]) => p.modulatable)
    .map(([key, p]) => ({ key, ...p }));
}

/**
 * Returns the NormalisableRange args for a param: [min, max, skew].
 * Skew defaults to 1.0 (linear).
 */
export function getNormalisableRange(paramSchema) {
  const { min = 0, max = 1, skew = 1.0 } = paramSchema;
  return { min, max, skew };
}

/**
 * Resolves a mod_target string "nodeId.paramKey" into { nodeId, paramKey }.
 * Returns null if the string is empty or malformed.
 */
export function resolveModTarget(target) {
  if (!target || !target.includes('.')) return null;
  const [nodeId, paramKey] = target.split('.');
  return { nodeId, paramKey };
}
