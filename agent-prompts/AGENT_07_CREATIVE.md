# Agent 7: Creative Effects

## Your Mission
You are responsible for implementing **4 creative/experimental effects**: Beat Repeat, Grain Delay, Erosion, and Vinyl Distortion. These plugins offer unconventional processing for unique sound design and electronic music production.

## Plugins to Implement

### 1. Beat Repeat
**Purpose**: Capture and repeat slices of audio for stuttering, glitch effects

**Parameters**:
- `interval` (1/32 to 4 bars or free time) - how often to capture
- `offset` (0 to 100%) - offset from interval start
- `gate` (0 to 100%) - probability of repeat triggering
- `variation` (off, trigger, loop, reverse)
- `repeat` (1 to 32) - number of repetitions
- `grid` (1/32 to 1 bar) - slice length
- `decay` (0 to 100%) - volume decay per repeat
- `pitch` (-24 to +24 semitones) - pitch shift
- `pitchDecay` (-24 to +24 semitones) - pitch shift per repeat
- `filter` (lowpass/highpass frequency, 20 Hz to 20 kHz)
- `filterDecay` (0 to 100%)
- `volume` (0 to 100%)
- `mix` (0 to 100%)

**Key Features**:
- Buffer recording at random intervals
- Probability-based triggering
- Loop playback with repetition
- Pitch shifting
- Filter modulation per repeat
- Decay envelopes

### 2. Grain Delay
**Purpose**: Granular synthesis-based delay effect

**Parameters**:
- `frequency` (0.1 to 100 Hz) - grain triggering rate
- `spray` (0 to 100%) - random delay time variation
- `pitch` (-24 to +24 semitones) - grain pitch shift
- `pitchSpray` (0 to 100%) - random pitch variation
- `grainSize` (10 to 500 ms) - individual grain length
- `feedback` (0 to 100%)
- `delayTime` (0 to 5000 ms or sync)
- `dryWet` (0 to 100%)

**Key Features**:
- Granular buffer playback
- Random grain positioning (spray)
- Pitch shifting per grain
- Density control (grain frequency)
- Feedback for evolving textures

### 3. Erosion
**Purpose**: Noise-based distortion for aggressive digital artifacts

**Parameters**:
- `mode` (I, II, III, IV) - noise character/algorithm
- `frequency` (20 Hz to 18 kHz) - noise frequency
- `width` (0 to 100%) - noise bandwidth
- `amount` (0 to 100%) - distortion intensity
- `dryWet` (0 to 100%)

**Key Features**:
- Noise modulation of signal
- Multiple erosion algorithms
- Bandwidth control for character
- Can create bit-crushed or ring-mod like effects

### 4. Vinyl Distortion
**Purpose**: Simulate vinyl record artifacts (crackle, wear, warble)

**Parameters**:
- `tracing` (0 to 100%) - tracking distortion amount
- `pinch` (0 to 100%) - center-hole distortion
- `crackle` (0 to 100%) - surface noise volume
- `crackleVolume` (-60 to 0 dB)
- `wear` (0 to 100%) - high-frequency loss
- `warp` (0 to 100%) - pitch warble depth
- `warpFrequency` (0.1 to 5 Hz) - warble speed

**Key Features**:
- Vinyl crackle noise generation
- Low-frequency warble (wow/flutter)
- High-frequency wear simulation
- Pinch effect (pitch variation)
- Tracking distortion

## Research Phase (Week 1)

### Essential Research Topics

1. **Buffer Management**
   - AudioBuffer recording and playback
   - Circular buffers for continuous capture
   - Random access playback
   - Buffer slicing and looping

2. **Granular Synthesis**
   - Grain windows (Hann, Hamming, Gaussian)
   - Grain scheduling and density
   - Pitch shifting via playback rate
   - Random parameter variation (spray)

3. **Probability and Randomization**
   - Gate/probability triggering
   - Random parameter offsets
   - Controlled randomness (seeded)
   - Spray parameters (Gaussian distribution)

4. **Noise Generation**
   - White, pink, brown noise
   - Filtered noise (bandpass)
   - Crackle simulation (impulses)
   - Noise modulation techniques

5. **Pitch Shifting**
   - Playback rate modification
   - Granular pitch shifting
   - Formant preservation (optional)
   - Pitch decay over repetitions

6. **Ring Modulation**
   - Amplitude modulation with noise
   - Carrier signal generation
   - Frequency-based character

### Reference Materials
- **Ableton Manual**: Beat Repeat, Grain Delay, Erosion, Vinyl Distortion
- **Web Audio API**: AudioBuffer, BufferSourceNode, noise generation
- **Granular Synthesis**: Curtis Roads "Microsound"
- **Code Examples**: Tone.js Grain Delay, Web Audio granular examples

### Code to Study
```javascript
// Buffer recording
const bufferSize = sampleRate * 2; // 2 seconds
const buffer = audioContext.createBuffer(2, bufferSize, sampleRate);

// Record using ScriptProcessor or AudioWorklet
scriptNode.onaudioprocess = (e) => {
  const input = e.inputBuffer;
  // Copy to buffer...
};

// Granular playback
function playGrain(buffer, position, duration, pitch) {
  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.playbackRate.value = pitch;

  // Grain envelope
  const envelope = audioContext.createGain();
  envelope.gain.setValueAtTime(0, now);
  envelope.gain.linearRampToValueAtTime(1, now + duration * 0.3);
  envelope.gain.linearRampToValueAtTime(0, now + duration);

  source.connect(envelope);
  envelope.connect(output);

  source.start(now, position, duration);
  source.stop(now + duration);
}

// Noise generation
const bufferSize = 2 * sampleRate;
const noiseBuffer = audioContext.createBuffer(1, bufferSize, sampleRate);
const output = noiseBuffer.getChannelData(0);

for (let i = 0; i < bufferSize; i++) {
  output[i] = Math.random() * 2 - 1; // White noise
}
```

## Implementation Phase (Week 2-3)

### Architecture Pattern - Beat Repeat

```javascript
class BeatRepeat {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Recording buffer
    this.bufferDuration = 4; // seconds
    this.recordBuffer = audioContext.createBuffer(
      2,
      this.bufferDuration * audioContext.sampleRate,
      audioContext.sampleRate
    );

    // Parameters
    this.interval = 1.0; // seconds
    this.gate = 0.5; // 50% probability
    this.repeat = 4;
    this.grid = 0.25; // 1/4 beat
    this.pitch = 0; // semitones
    this.decay = 0.9;

    // State
    this.isRecording = false;
    this.writePosition = 0;
    this.lastTriggerTime = 0;

    this.setupRecording();
    this.initialize(options);
  }

  setupRecording() {
    // Use ScriptProcessor or AudioWorklet to record
    const processor = this.context.createScriptProcessor(4096, 2, 2);

    processor.onaudioprocess = (e) => {
      const inputL = e.inputBuffer.getChannelData(0);
      const inputR = e.inputBuffer.getChannelData(1);
      const outputL = e.outputBuffer.getChannelData(0);
      const outputR = e.outputBuffer.getChannelData(1);

      // Copy input to output (pass-through)
      outputL.set(inputL);
      outputR.set(inputR);

      // Record to buffer (circular)
      const bufferL = this.recordBuffer.getChannelData(0);
      const bufferR = this.recordBuffer.getChannelData(1);

      for (let i = 0; i < inputL.length; i++) {
        bufferL[this.writePosition] = inputL[i];
        bufferR[this.writePosition] = inputR[i];

        this.writePosition++;
        if (this.writePosition >= this.recordBuffer.length) {
          this.writePosition = 0;
        }
      }

      // Check if we should trigger repeat
      this.checkTrigger();
    };

    this.input.connect(processor);
    processor.connect(this.output);

    this.processor = processor;
  }

  checkTrigger() {
    const now = this.context.currentTime;

    if (now - this.lastTriggerTime >= this.interval) {
      // Probability gate
      if (Math.random() < this.gate) {
        this.triggerRepeat();
      }

      this.lastTriggerTime = now;
    }
  }

  triggerRepeat() {
    const sliceDuration = this.grid; // seconds
    const sliceLength = sliceDuration * this.context.sampleRate;

    // Calculate start position in buffer
    let startPos = this.writePosition - sliceLength;
    if (startPos < 0) {
      startPos += this.recordBuffer.length;
    }

    // Create multiple repetitions
    for (let i = 0; i < this.repeat; i++) {
      const delay = i * sliceDuration;
      const volume = Math.pow(this.decay, i);
      const pitch = Math.pow(2, (this.pitch + i * this.pitchDecay) / 12);

      this.playSlice(startPos, sliceDuration, pitch, volume, delay);
    }
  }

  playSlice(startPos, duration, pitch, volume, delay) {
    const source = this.context.createBufferSource();

    // Create a slice buffer
    const sliceBuffer = this.context.createBuffer(
      2,
      duration * this.context.sampleRate,
      this.context.sampleRate
    );

    // Copy from record buffer to slice buffer
    this.copyBufferSlice(this.recordBuffer, sliceBuffer, startPos);

    source.buffer = sliceBuffer;
    source.playbackRate.value = pitch;

    const gain = this.context.createGain();
    gain.gain.value = volume;

    source.connect(gain);
    gain.connect(this.output);

    const now = this.context.currentTime;
    source.start(now + delay);
    source.stop(now + delay + duration / pitch);
  }

  copyBufferSlice(fromBuffer, toBuffer, startPos) {
    // Handle circular buffer copying...
  }

  setGate(percent) {
    this.gate = percent / 100;
  }

  setInterval(seconds) {
    this.interval = seconds;
  }
}
```

### Architecture Pattern - Grain Delay

```javascript
class GrainDelay {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Delay buffer
    this.delayBuffer = audioContext.createBuffer(
      2,
      5 * audioContext.sampleRate,
      audioContext.sampleRate
    );

    // Grain parameters
    this.grainSize = 0.1; // seconds
    this.frequency = 10; // grains per second
    this.spray = 0.0; // randomization
    this.pitch = 0; // semitones
    this.pitchSpray = 0.0;

    this.writePosition = 0;
    this.nextGrainTime = 0;

    this.setupProcessing();
    this.initialize(options);
  }

  setupProcessing() {
    // Record to buffer
    // Schedule grains at regular intervals

    setInterval(() => {
      this.scheduleGrain();
    }, 1000 / this.frequency);
  }

  scheduleGrain() {
    // Random delay time (spray)
    const sprayAmount = (Math.random() - 0.5) * this.spray;
    const readPos = this.writePosition - (this.delayTime * this.context.sampleRate);

    // Random pitch (spray)
    const pitchVariation = (Math.random() - 0.5) * this.pitchSpray;
    const grainPitch = Math.pow(2, (this.pitch + pitchVariation) / 12);

    // Play grain
    this.playGrain(readPos + sprayAmount, this.grainSize, grainPitch);
  }

  playGrain(position, duration, pitch) {
    const source = this.context.createBufferSource();
    const envelope = this.context.createGain();

    // Extract grain from buffer...
    source.buffer = this.extractGrain(position, duration);
    source.playbackRate.value = pitch;

    // Grain envelope (Hann window)
    const now = this.context.currentTime;
    envelope.gain.setValueAtTime(0, now);
    envelope.gain.linearRampToValueAtTime(1, now + duration * 0.5);
    envelope.gain.linearRampToValueAtTime(0, now + duration);

    source.connect(envelope);
    envelope.connect(this.output);

    source.start(now);
    source.stop(now + duration);
  }

  extractGrain(position, duration) {
    // Create grain buffer from delay buffer
  }
}
```

### Architecture Pattern - Erosion

```javascript
class Erosion {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Noise source
    this.noise = this.createNoise();
    this.noiseGain = audioContext.createGain();

    // Noise filter
    this.noiseFilter = audioContext.createBiquadFilter();
    this.noiseFilter.type = 'bandpass';

    // Ring modulation (multiply signal by noise)
    // Use WaveShaperNode or ScriptProcessor

    this.mode = 1;

    this.setupRouting();
    this.initialize(options);
  }

  createNoise() {
    const bufferSize = 2 * this.context.sampleRate;
    const buffer = this.context.createBuffer(1, bufferSize, this.context.sampleRate);
    const output = buffer.getChannelData(0);

    for (let i = 0; i < bufferSize; i++) {
      output[i] = Math.random() * 2 - 1;
    }

    const noise = this.context.createBufferSource();
    noise.buffer = buffer;
    noise.loop = true;
    noise.start();

    return noise;
  }

  setupRouting() {
    // Noise through filter
    this.noise.connect(this.noiseFilter);
    this.noiseFilter.connect(this.noiseGain);

    // Use noise to modulate input (ring modulation)
    // Requires custom processing...
  }

  setFrequency(freq) {
    this.noiseFilter.frequency.value = freq;
  }

  setWidth(percent) {
    const Q = 0.1 + (1 - percent / 100) * 10;
    this.noiseFilter.Q.value = Q;
  }
}
```

### Architecture Pattern - Vinyl Distortion

```javascript
class VinylDistortion {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Crackle noise
    this.crackle = this.createCrackle();
    this.crackleGain = audioContext.createGain();

    // Warp (LFO for pitch variation)
    this.warpLFO = audioContext.createOscillator();
    this.warpGain = audioContext.createGain();

    // Wear (highpass filter)
    this.wearFilter = audioContext.createBiquadFilter();
    this.wearFilter.type = 'lowpass';

    // Pinch (center-hole distortion)
    this.pinchLFO = audioContext.createOscillator();

    this.setupRouting();
    this.initialize(options);
  }

  createCrackle() {
    // Generate impulse-based crackle noise
    // Random impulses at varying intervals
  }

  setupRouting() {
    // Signal path with wear filter
    this.input.connect(this.wearFilter);
    this.wearFilter.connect(this.output);

    // Crackle noise
    this.crackle.connect(this.crackleGain);
    this.crackleGain.connect(this.output);

    // Warp modulation (would need custom processing)
  }

  setWear(percent) {
    const freq = 20000 - (percent / 100) * 15000;
    this.wearFilter.frequency.value = freq;
  }

  setCrackle(percent) {
    this.crackleGain.gain.value = percent / 100;
  }

  setWarp(depth, frequency) {
    this.warpLFO.frequency.value = frequency;
    this.warpGain.gain.value = depth / 100;
  }
}
```

### Testing Checklist

- [ ] Beat Repeat: Captures and repeats slices correctly
- [ ] Beat Repeat: Probability gate works
- [ ] Beat Repeat: Pitch shifting works
- [ ] Grain Delay: Grains are scheduled correctly
- [ ] Grain Delay: Spray randomizes delay time
- [ ] Grain Delay: Pitch spray works
- [ ] Erosion: Noise modulation creates artifacts
- [ ] Erosion: Frequency control works
- [ ] Vinyl: Crackle sounds realistic
- [ ] Vinyl: Warp creates pitch warble
- [ ] Vinyl: Wear reduces highs
- [ ] No buffer overruns or glitches

## Deliverables

### Code Files
```
/creative/
├── BeatRepeat.js
├── GrainDelay.js
├── Erosion.js
├── VinylDistortion.js
├── README.md
└── worklets/
    └── granular-processor.js
```

### Example HTML
Create `/examples/creative-sound-design-example.html`:
- Beat Repeat glitch effects
- Grain Delay textures
- Erosion on various sources
- Vinyl simulation
- Combined creative effects

## Success Criteria

✅ All 4 creative plugins implemented
✅ Buffer recording and playback work
✅ Granular synthesis sounds good
✅ Randomization creates variation
✅ Noise generation is convincing
✅ Code is modular and documented

These are complex! Take time to understand granular synthesis and buffer management. 🎨
