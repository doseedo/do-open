# Agent 4: Modulation Effects

## Your Mission
You are responsible for implementing **4 modulation effects**: Chorus, Flanger, Phaser, and Tremolo/Auto Pan. These plugins add movement, depth, and animation to audio through time-varying parameter modulation.

## Plugins to Implement

### 1. Chorus
**Purpose**: Creates the illusion of multiple voices/instruments by layering slightly detuned delays

**Parameters**:
- `rate` (0.01 to 10 Hz, default: 0.5) - LFO speed
- `depth` (0 to 100%, default: 50) - modulation intensity
- `voices` (1 to 8, default: 4) - number of delay lines
- `spread` (0 to 100%, default: 50) - stereo width
- `feedback` (0 to 100%, default: 0)
- `mix` (0 to 100%, default: 50)
- `delay` (5 to 50 ms, default: 20) - base delay time

**Key Features**:
- Multiple delay lines (2-8 voices)
- Independent LFO phase per voice
- Stereo spread
- Feedback for richer sound

### 2. Flanger
**Purpose**: Jet-plane whoosh effect created by short delay with feedback and modulation

**Parameters**:
- `rate` (0.01 to 10 Hz, default: 0.5)
- `depth` (0 to 100%, default: 50)
- `feedback` (0 to 100%, default: 50)
- `delay` (0.5 to 10 ms, default: 3)
- `manual` (0 to 100%, default: 50) - static delay offset
- `sync` (boolean) - tempo sync
- `waveform` (sine, triangle)
- `mix` (0 to 100%, default: 50)

**Key Features**:
- Very short delay times
- High feedback for resonance
- Manual control for static flange
- Negative feedback option

### 3. Phaser
**Purpose**: Sweeping notches created by all-pass filters with LFO modulation

**Parameters**:
- `rate` (0.01 to 10 Hz, default: 0.5)
- `depth` (0 to 100%, default: 50)
- `feedback` (0 to 100%, default: 0)
- `stages` (4, 6, 8, 12) - number of all-pass filters
- `frequency` (200 to 8000 Hz) - center frequency
- `spread` (0 to 100%) - spacing between notches
- `waveform` (sine, triangle, square, sawtooth)
- `mix` (0 to 100%, default: 50)

**Key Features**:
- 4-12 stage all-pass filter cascade
- LFO modulates filter frequencies
- Feedback for resonance
- Multiple waveforms

### 4. Tremolo/Auto Pan
**Purpose**: Amplitude or pan modulation for rhythmic movement

**Parameters**:
- `rate` (0.01 to 40 Hz or sync to BPM)
- `depth` (0 to 100%, default: 50)
- `waveform` (sine, triangle, square, sawtooth, random)
- `phase` (0 to 360°, default: 0)
- `mode` (tremolo, pan) - amplitude or pan modulation
- `shape` (0 to 100%) - waveform shaping/skew
- `sync` (boolean) - tempo sync
- `stereo` (boolean) - phase offset between L/R

**Key Features**:
- Amplitude or panning modulation
- Multiple LFO waveforms
- Tempo sync
- Random waveform option
- Stereo phase offset

## Research Phase (Week 1)

### Essential Research Topics

1. **LFO (Low Frequency Oscillator) Design**
   - OscillatorNode for standard waveforms
   - Custom waveforms with PeriodicWave
   - Random/noise-based LFO
   - Phase offset for multiple LFOs
   - Tempo synchronization

2. **Chorus Algorithm**
   - Multiple short delay lines
   - Phase-distributed LFOs
   - Pseudo-random delay time variation
   - Stereo spreading techniques

3. **Flanger Physics**
   - Comb filtering from short delays
   - Feedback creates resonance peaks
   - Through-zero flanging (not easily done in Web Audio)
   - Manual sweep control

4. **All-Pass Filter Design**
   - All-pass filter characteristics (flat magnitude, varying phase)
   - Cascading multiple stages
   - Coefficient calculation for phaser
   - Q factor and resonance

5. **Amplitude and Pan Modulation**
   - GainNode modulation
   - StereoPannerNode modulation
   - Constant-power panning law
   - Waveform shaping (asymmetric modulation)

### Reference Materials
- **Ableton Manual**: Chorus, Flanger, Phaser, Auto Pan
- **Web Audio API**: OscillatorNode, PeriodicWave, BiquadFilterNode (allpass)
- **Tone.js**: Study modulation effect implementations
- **DSP**: "Digital Audio Effects" by Udo Zölzer (Chorus/Flanger/Phaser chapters)

### Code to Study
```javascript
// LFO setup
const lfo = context.createOscillator();
const lfoGain = context.createGain();

lfo.type = 'sine';
lfo.frequency.value = 0.5; // Hz
lfoGain.gain.value = 0.005; // Depth

// Connect LFO to delay time
lfo.connect(lfoGain);
lfoGain.connect(delay.delayTime);
lfo.start();

// All-pass filter for phaser
const allpass = context.createBiquadFilter();
allpass.type = 'allpass';
allpass.frequency.value = 1000;
allpass.Q.value = 1;

// Multiple LFOs with phase offset
const phaseOffset = (index / numVoices) * Math.PI * 2;
// Use custom waveform or delayed start
```

## Implementation Phase (Week 2-3)

### Architecture Pattern - Chorus

```javascript
class Chorus {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Dry/wet
    this.dryGain = audioContext.createGain();
    this.wetGain = audioContext.createGain();

    // Multiple voices (delay lines)
    this.voices = [];
    this.numVoices = 4;

    for (let i = 0; i < this.numVoices; i++) {
      this.voices.push(this.createVoice(i));
    }

    this.setupRouting();
    this.initialize(options);
  }

  createVoice(index) {
    const voice = {
      delay: this.context.createDelay(0.1),
      lfo: this.context.createOscillator(),
      lfoGain: this.context.createGain(),
      panner: this.context.createStereoPanner()
    };

    // Set phase offset for each voice
    const phaseOffset = (index / this.numVoices) * Math.PI * 2;

    // Base delay time
    voice.delay.delayTime.value = 0.020; // 20ms base

    // LFO depth
    voice.lfoGain.gain.value = 0.005; // 5ms modulation

    // Connect LFO
    voice.lfo.connect(voice.lfoGain);
    voice.lfoGain.connect(voice.delay.delayTime);
    voice.lfo.type = 'sine';
    voice.lfo.frequency.value = 0.5;

    // Start with phase offset (use custom waveform for precise phase)
    voice.lfo.start();

    // Stereo spread
    const pan = (index / (this.numVoices - 1)) * 2 - 1; // -1 to 1
    voice.panner.pan.value = pan;

    return voice;
  }

  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet paths (all voices in parallel)
    this.voices.forEach(voice => {
      this.input.connect(voice.delay);
      voice.delay.connect(voice.panner);
      voice.panner.connect(this.wetGain);
    });

    this.wetGain.connect(this.output);
  }

  setVoices(num) {
    // Dynamically change number of active voices
    this.numVoices = Math.min(Math.max(1, num), 8);
    // Rebuild voices array...
  }

  setRate(hz) {
    this.voices.forEach(voice => {
      voice.lfo.frequency.value = hz;
    });
  }

  setDepth(percent) {
    const depth = percent / 100;
    this.voices.forEach(voice => {
      voice.lfoGain.gain.value = depth * 0.010; // Max 10ms modulation
    });
  }

  setSpread(percent) {
    const spread = percent / 100;
    this.voices.forEach((voice, index) => {
      const pan = ((index / (this.voices.length - 1)) * 2 - 1) * spread;
      voice.panner.pan.value = pan;
    });
  }
}
```

### Architecture Pattern - Flanger

```javascript
class Flanger {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Core components
    this.delay = audioContext.createDelay(0.02); // Max 20ms
    this.feedback = audioContext.createGain();
    this.lfo = audioContext.createOscillator();
    this.lfoGain = audioContext.createGain();

    this.dryGain = audioContext.createGain();
    this.wetGain = audioContext.createGain();

    this.setupRouting();
    this.initialize(options);
  }

  setupRouting() {
    // Dry
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet with feedback
    this.input.connect(this.delay);
    this.delay.connect(this.feedback);
    this.feedback.connect(this.delay); // Feedback loop
    this.feedback.connect(this.wetGain);
    this.wetGain.connect(this.output);

    // LFO modulation
    this.lfo.connect(this.lfoGain);
    this.lfoGain.connect(this.delay.delayTime);
    this.lfo.type = 'sine';
    this.lfo.frequency.value = 0.5;
    this.lfo.start();
  }

  setDelay(ms) {
    // Base delay time
    this.delay.delayTime.value = ms / 1000;
  }

  setDepth(percent) {
    const depth = percent / 100;
    this.lfoGain.gain.value = depth * 0.003; // Max 3ms modulation
  }

  setFeedback(percent) {
    // Can be positive or negative
    const feedback = percent / 100;
    this.feedback.gain.value = feedback;
  }

  setManual(percent) {
    // Static delay offset
    const offset = percent / 100;
    const baseDelay = this.delay.delayTime.value;
    this.delay.delayTime.value = baseDelay + (offset * 0.005);
  }
}
```

### Architecture Pattern - Phaser

```javascript
class Phaser {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // All-pass filter stages
    this.stages = [];
    this.numStages = 6;

    for (let i = 0; i < this.numStages; i++) {
      const allpass = audioContext.createBiquadFilter();
      allpass.type = 'allpass';
      allpass.Q.value = 1;
      this.stages.push(allpass);
    }

    // LFO
    this.lfo = audioContext.createOscillator();
    this.lfoGain = audioContext.createGain();

    this.feedback = audioContext.createGain();
    this.dryGain = audioContext.createGain();
    this.wetGain = audioContext.createGain();

    this.setupRouting();
    this.initialize(options);
  }

  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path - cascade all-pass filters
    let current = this.input;
    this.stages.forEach(stage => {
      current.connect(stage);
      current = stage;
    });

    // Feedback from last stage to first
    current.connect(this.feedback);
    this.feedback.connect(this.stages[0]);

    // Output
    current.connect(this.wetGain);
    this.wetGain.connect(this.output);

    // LFO modulates all filter frequencies
    this.lfo.connect(this.lfoGain);
    this.stages.forEach(stage => {
      this.lfoGain.connect(stage.frequency);
    });

    this.lfo.type = 'sine';
    this.lfo.frequency.value = 0.5;
    this.lfo.start();
  }

  setStages(num) {
    // Change number of active stages (4, 6, 8, 12)
    this.numStages = num;
    // Rebuild filter chain...
  }

  setFrequency(hz) {
    // Set center frequency for all stages
    this.stages.forEach((stage, index) => {
      // Spread frequencies across spectrum
      const multiplier = Math.pow(2, index / 2);
      stage.frequency.value = hz * multiplier;
    });
  }

  setDepth(percent) {
    const depth = percent / 100;
    this.lfoGain.gain.value = depth * 2000; // Frequency modulation range
  }
}
```

### Architecture Pattern - Tremolo

```javascript
class Tremolo {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // LFO
    this.lfo = audioContext.createOscillator();
    this.lfoGain = audioContext.createGain();

    // For tremolo, modulate gain
    this.amplitudeModulator = audioContext.createGain();

    // For auto-pan, modulate pan
    this.panner = audioContext.createStereoPanner();

    this.mode = 'tremolo'; // or 'pan'

    this.setupRouting();
    this.initialize(options);
  }

  setupRouting() {
    if (this.mode === 'tremolo') {
      // Amplitude modulation
      this.input.connect(this.amplitudeModulator);
      this.amplitudeModulator.connect(this.output);

      // LFO controls amplitude
      this.lfo.connect(this.lfoGain);
      this.lfoGain.connect(this.amplitudeModulator.gain);
    } else {
      // Pan modulation
      this.input.connect(this.panner);
      this.panner.connect(this.output);

      // LFO controls pan
      this.lfo.connect(this.lfoGain);
      this.lfoGain.connect(this.panner.pan);
    }

    this.lfo.type = 'sine';
    this.lfo.frequency.value = 5; // Hz
    this.lfo.start();
  }

  setRate(hz) {
    this.lfo.frequency.value = hz;
  }

  setDepth(percent) {
    const depth = percent / 100;

    if (this.mode === 'tremolo') {
      // Depth controls amplitude variation
      this.amplitudeModulator.gain.value = 1 - (depth * 0.5);
      this.lfoGain.gain.value = depth * 0.5;
    } else {
      // Depth controls pan range
      this.lfoGain.gain.value = depth; // -1 to 1
    }
  }

  setWaveform(type) {
    this.lfo.type = type; // sine, triangle, square, sawtooth
  }
}
```

### Testing Checklist

- [ ] Chorus: Multiple voices create rich, detuned sound
- [ ] Chorus: Stereo spread works correctly
- [ ] Chorus: No clicks or artifacts
- [ ] Flanger: Characteristic "jet plane" sound
- [ ] Flanger: Feedback creates resonance
- [ ] Flanger: Manual control works
- [ ] Phaser: Sweeping notches are audible
- [ ] Phaser: Different stage counts change character
- [ ] Phaser: Feedback enhances resonance
- [ ] Tremolo: Smooth amplitude modulation
- [ ] Auto Pan: Smooth panning without clicks
- [ ] All: Tempo sync works accurately
- [ ] All: Parameter changes are smooth

## Deliverables

### Code Files
```
/modulation/
├── Chorus.js
├── Flanger.js
├── Phaser.js
├── Tremolo.js
└── README.md
```

### Example HTML
Create `/examples/modulation-showcase-example.html`:
- Chorus on vocals/guitars
- Flanger sweep demo
- Phaser on various sources
- Tremolo/Auto Pan rhythmic examples
- Combined modulation effects

## Success Criteria

✅ All 4 modulation plugins implemented
✅ LFO modulation is smooth and artifact-free
✅ Stereo effects work correctly
✅ Tempo sync is accurate
✅ Sound quality matches Ableton originals
✅ Code is modular and documented

Begin with LFO research, then implement each effect. Test for smooth modulation! 🌊
