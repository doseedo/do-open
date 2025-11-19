# Agent 5: Reverb & Spatial Effects

## Your Mission
You are responsible for implementing **3 spatial/reverb effects**: Reverb (algorithmic), Hybrid Reverb (convolution + algorithmic), and Echo. These plugins create acoustic spaces and ambient textures essential for depth and realism.

## Plugins to Implement

### 1. Reverb (Algorithmic)
**Purpose**: Create artificial room ambience using feedback delay networks

**Parameters**:
- `preDelay` (0 to 250 ms, default: 0)
- `decayTime` (0.1 to 20 seconds, default: 2)
- `size` (0 to 100%, default: 50) - room size
- `diffusion` (0 to 100%, default: 70) - echo density
- `damping` (0 to 100%, default: 50) - high-frequency absorption
- `modulation` (0 to 100%, default: 20) - subtle chorus effect
- `stereoWidth` (0 to 100%, default: 100)
- `earlyLevel` (-inf to 0 dB, default: -12) - early reflections level
- `tailLevel` (-inf to 0 dB, default: -6) - reverb tail level
- `mix` (0 to 100%, default: 30)

**Key Features**:
- Early reflections + diffuse tail
- Frequency-dependent damping
- Subtle modulation to avoid metallic sound
- Stereo width control

### 2. Hybrid Reverb
**Purpose**: Combine convolution reverb (realistic) with algorithmic tail (efficient)

**Parameters**:
- `impulseResponse` (file path or preset selection)
- `irLength` (0 to 100%, default: 100) - trim IR
- `predelay` (0 to 250 ms)
- `decayTime` (0.1 to 20 seconds) - algorithmic tail
- `crossover` (200 to 8000 Hz) - where convolution hands off to algorithmic
- `erLevel` (-inf to 0 dB) - early reflections from IR
- `tailLevel` (-inf to 0 dB) - algorithmic tail
- `damping` (0 to 100%)
- `mix` (0 to 100%)

**Key Features**:
- Load impulse response files (WAV)
- Convolution for early reflections
- Algorithmic tail for efficiency
- Crossover blend between IR and algo

### 3. Echo
**Purpose**: Complex delay with modulation, ducking, and reverb in feedback path

**Parameters**:
- `delayTimeL` (0 to 2000 ms or sync)
- `delayTimeR` (0 to 2000 ms or sync)
- `feedback` (0 to 100%)
- `channelMode` (left, right, ping-pong)
- `stereoOffset` (-50 to +50 ms)
- `modulation`:
  - `rate` (0 to 10 Hz)
  - `amount` (0 to 100%)
- `ducking`:
  - `threshold` (-60 to 0 dB)
  - `ratio` (1 to 10)
- `reverb`:
  - `amount` (0 to 100%)
  - `decay` (0.1 to 10 seconds)
- `filter`:
  - `highpass` (20 to 1000 Hz)
  - `lowpass` (1k to 20k Hz)
- `mix` (0 to 100%)

**Key Features**:
- Stereo delay with offset
- Modulation in feedback path
- Ducking (delay quiets when input is loud)
- Reverb in feedback for ambient tails
- Tempo sync

## Research Phase (Week 1)

### Essential Research Topics

1. **Algorithmic Reverb Design**
   - Schroeder reverberator (comb filters + all-pass)
   - Freeverb algorithm
   - Dattorro plate reverb
   - Feedback Delay Network (FDN)
   - Early reflections vs late reverb tail

2. **Convolution Reverb**
   - ConvolverNode usage
   - Impulse response loading (WAV files)
   - IR trimming and normalization
   - Latency considerations
   - Efficient convolution

3. **Diffusion Networks**
   - All-pass filter cascades
   - Hadamard matrices for mixing
   - Echo density and buildup
   - Avoiding flutter echoes

4. **Damping Implementation**
   - Frequency-dependent decay
   - Lowpass filters in feedback loops
   - Material absorption simulation
   - Air absorption modeling

5. **Reverb Modulation**
   - LFO on delay lines
   - Preventing metallic artifacts
   - Subtle pitch variation
   - Chorusing in reverb tail

6. **Ducking/Sidechain**
   - Envelope follower on input
   - Dynamic reverb level
   - Preventing mud in mixes

### Reference Materials
- **Ableton Manual**: Reverb, Hybrid Reverb, Echo
- **Web Audio API**: ConvolverNode, complex delay networks
- **Academic Papers**:
  - "Effect Design, Part 1: Reverberator and Other Filters" by Jon Dattorro
  - Schroeder's original reverb paper (1962)
  - Freeverb algorithm documentation
- **Impulse Responses**: OpenAIR library, Free IR libraries

### Code to Study
```javascript
// Convolution reverb
const convolver = context.createConvolver();

// Load impulse response
fetch('impulse-response.wav')
  .then(response => response.arrayBuffer())
  .then(arrayBuffer => context.decodeAudioData(arrayBuffer))
  .then(audioBuffer => {
    convolver.buffer = audioBuffer;
  });

// Simple Schroeder reverb structure
const combFilters = []; // Parallel comb filters
const allpassFilters = []; // Series all-pass filters

// Typical comb delay times (in samples at 44.1kHz)
const combDelays = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116];

// Typical all-pass delay times
const allpassDelays = [225, 556, 441, 341];
```

## Implementation Phase (Week 2-3)

### Architecture Pattern - Algorithmic Reverb

```javascript
class Reverb {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Pre-delay
    this.predelay = audioContext.createDelay(0.25);

    // Early reflections (simple delays)
    this.earlyReflections = this.createEarlyReflections();

    // Diffuse reverb tail (FDN or Freeverb-style)
    this.reverbTail = this.createReverbTail();

    this.dryGain = audioContext.createGain();
    this.wetGain = audioContext.createGain();
    this.earlyGain = audioContext.createGain();
    this.tailGain = audioContext.createGain();

    this.setupRouting();
    this.initialize(options);
  }

  createEarlyReflections() {
    // 6-8 short delays simulating room geometry
    const delays = [];
    const delayTimes = [0.019, 0.022, 0.027, 0.031, 0.037, 0.043];

    delayTimes.forEach((time, index) => {
      const delay = this.context.createDelay(0.1);
      const gain = this.context.createGain();
      const pan = this.context.createStereoPanner();

      delay.delayTime.value = time;
      gain.gain.value = 1 / (index + 1); // Decay over time
      pan.pan.value = (Math.random() * 2 - 1) * 0.5; // Random pan

      delays.push({ delay, gain, pan });
    });

    return delays;
  }

  createReverbTail() {
    // Freeverb-style: parallel comb filters + series all-pass

    const tail = {
      combFilters: [],
      allpassFilters: [],
      damping: []
    };

    // 8 parallel comb filters (4 per channel for stereo)
    const combDelayTimes = [
      0.0297, 0.0371, 0.0411, 0.0437,
      0.0050, 0.0017, 0.0098, 0.0122
    ];

    combDelayTimes.forEach(time => {
      const delay = this.context.createDelay(0.1);
      const feedback = this.context.createGain();
      const damping = this.context.createBiquadFilter();

      delay.delayTime.value = time;
      feedback.gain.value = 0.84; // Decay factor
      damping.type = 'lowpass';
      damping.frequency.value = 5000;

      tail.combFilters.push({ delay, feedback, damping });
    });

    // 4 series all-pass filters for diffusion
    const allpassTimes = [0.0051, 0.0126, 0.0100, 0.0077];

    allpassTimes.forEach(time => {
      const delay = this.context.createDelay(0.1);
      const feedback = this.context.createGain();
      const feedforward = this.context.createGain();

      delay.delayTime.value = time;
      feedback.gain.value = 0.5;
      feedforward.gain.value = -0.5;

      tail.allpassFilters.push({ delay, feedback, feedforward });
    });

    return tail;
  }

  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path: predelay → early reflections + reverb tail
    this.input.connect(this.predelay);

    // Early reflections
    this.earlyReflections.forEach(er => {
      this.predelay.connect(er.delay);
      er.delay.connect(er.gain);
      er.gain.connect(er.pan);
      er.pan.connect(this.earlyGain);
    });
    this.earlyGain.connect(this.output);

    // Reverb tail (comb filters in parallel)
    this.reverbTail.combFilters.forEach(comb => {
      this.predelay.connect(comb.delay);
      comb.delay.connect(comb.damping);
      comb.damping.connect(comb.feedback);
      comb.feedback.connect(comb.delay); // Feedback loop
      comb.feedback.connect(this.tailGain);
    });

    // All-pass filters in series for diffusion
    let current = this.tailGain;
    this.reverbTail.allpassFilters.forEach(ap => {
      current.connect(ap.delay);
      ap.delay.connect(ap.feedback);
      ap.feedback.connect(ap.delay); // Feedback
      current = ap.delay;
    });

    current.connect(this.wetGain);
    this.wetGain.connect(this.output);
  }

  setDecayTime(seconds) {
    // Adjust feedback gains to change decay
    const feedback = Math.pow(0.001, 1 / (seconds * this.context.sampleRate));
    this.reverbTail.combFilters.forEach(comb => {
      comb.feedback.gain.value = feedback;
    });
  }

  setDamping(percent) {
    // Adjust lowpass frequency
    const freq = 20000 * (1 - percent / 100);
    this.reverbTail.combFilters.forEach(comb => {
      comb.damping.frequency.value = freq;
    });
  }

  setSize(percent) {
    // Scale all delay times
    const scale = 0.5 + (percent / 100) * 1.5;
    // Adjust delay times...
  }
}
```

### Architecture Pattern - Hybrid Reverb

```javascript
class HybridReverb {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Convolution for early reflections
    this.convolver = audioContext.createConvolver();
    this.irGain = audioContext.createGain();

    // Algorithmic tail
    this.algorithmicTail = new Reverb(audioContext);

    // Crossover (highpass for tail)
    this.crossover = audioContext.createBiquadFilter();
    this.crossover.type = 'highpass';

    this.setupRouting();
    this.initialize(options);
  }

  async loadImpulseResponse(url) {
    try {
      const response = await fetch(url);
      const arrayBuffer = await response.arrayBuffer();
      const audioBuffer = await this.context.decodeAudioData(arrayBuffer);

      this.convolver.buffer = audioBuffer;
      return true;
    } catch (error) {
      console.error('Failed to load IR:', error);
      return false;
    }
  }

  setupRouting() {
    // IR path (early reflections)
    this.input.connect(this.convolver);
    this.convolver.connect(this.irGain);
    this.irGain.connect(this.output);

    // Algorithmic tail path
    this.input.connect(this.crossover);
    this.crossover.connect(this.algorithmicTail.input);
    this.algorithmicTail.connect(this.output);
  }

  setCrossover(freq) {
    this.crossover.frequency.value = freq;
  }
}
```

### Architecture Pattern - Echo

```javascript
class Echo {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Stereo delays
    this.delayL = audioContext.createDelay(2.0);
    this.delayR = audioContext.createDelay(2.0);

    // Feedback
    this.feedbackL = audioContext.createGain();
    this.feedbackR = audioContext.createGain();

    // Modulation
    this.lfo = audioContext.createOscillator();
    this.lfoGain = audioContext.createGain();

    // Ducking (sidechain compressor)
    this.ducker = audioContext.createDynamicsCompressor();

    // Reverb in feedback
    this.feedbackReverb = new Reverb(audioContext);

    // Filters
    this.highpass = audioContext.createBiquadFilter();
    this.lowpass = audioContext.createBiquadFilter();

    this.setupRouting();
    this.initialize(options);
  }

  setupRouting() {
    // Complex routing with modulation, ducking, reverb, filtering
    // Input → delays → filters → ducking → reverb → feedback
  }

  setDucking(threshold, ratio) {
    this.ducker.threshold.value = threshold;
    this.ducker.ratio.value = ratio;
  }
}
```

### Testing Checklist

- [ ] Reverb: Sounds natural, not metallic
- [ ] Reverb: Decay time is accurate
- [ ] Reverb: Damping affects high frequencies
- [ ] Reverb: Early reflections are distinct from tail
- [ ] Hybrid Reverb: IR loads correctly
- [ ] Hybrid Reverb: Crossover blends smoothly
- [ ] Echo: Ducking responds to input level
- [ ] Echo: Modulation is subtle and pleasant
- [ ] Echo: Reverb in feedback creates ambient tails
- [ ] No audio artifacts or feedback runaway
- [ ] CPU usage is reasonable

## Deliverables

### Code Files
```
/reverb/
├── Reverb.js
├── HybridReverb.js
├── Echo.js
├── README.md
└── impulse-responses/
    ├── small-room.wav
    ├── large-hall.wav
    └── plate.wav
```

### Example HTML
Create `/examples/spatial-effects-example.html`:
- Algorithmic reverb on various sources
- Hybrid reverb with different IRs
- Echo with ducking demo
- Reverb comparison (algorithmic vs convolution)

## Success Criteria

✅ All 3 spatial plugins implemented
✅ Reverb sounds natural and realistic
✅ Hybrid reverb loads and uses IRs correctly
✅ Echo has working ducking and modulation
✅ No feedback runaway or instability
✅ Code is modular and documented

Begin with reverb algorithm research. This is complex! Take time to understand FDNs. 🎭
