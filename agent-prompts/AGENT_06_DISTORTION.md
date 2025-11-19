# Agent 6: Distortion & Saturation

## Your Mission
You are responsible for implementing **4 distortion/saturation plugins**: Overdrive, Saturator, Distortion, and Redux. These plugins add harmonic richness, warmth, grit, and character through controlled non-linear processing.

## Plugins to Implement

### 1. Overdrive
**Purpose**: Tube-style soft clipping for warm, musical distortion

**Parameters**:
- `drive` (0 to 100%, default: 30) - amount of gain before saturation
- `tone` (0 to 100%, default: 50) - post-distortion tone control
- `bias` (-100 to +100%, default: 0) - asymmetric distortion
- `output` (-24 to +24 dB, default: 0) - makeup gain
- `mix` (0 to 100%, default: 100)

**Key Features**:
- Soft clipping (tanh, atan, or custom curves)
- Asymmetric distortion (even harmonics)
- Tone stack (pre/post EQ)
- Auto gain compensation option

### 2. Saturator
**Purpose**: Multi-mode saturation from subtle warmth to heavy distortion

**Parameters**:
- `drive` (0 to 100%, default: 0)
- `type` (warm, digital, analog, clip, foldback, sine-fold)
- `color` (0 to 100%) - additional harmonic emphasis
- `depth` (0 to 100%) - wet/dry character
- `dcFilter` (boolean, default: true) - remove DC offset
- `output` (-24 to +24 dB)
- `mix` (0 to 100%)

**Key Features**:
- Multiple saturation algorithms
- Pre/post filtering
- DC offset removal
- Oversampling to reduce aliasing

### 3. Distortion
**Purpose**: Hard clipping distortion with aggressive harmonic generation

**Parameters**:
- `drive` (0 to 100%, default: 50)
- `tone` (20 Hz to 20 kHz, default: 1000) - tone center frequency
- `toneWidth` (0.1 to 10, default: 1) - tone Q
- `filterPosition` (pre, post) - tone filter placement
- `clipType` (hard, soft, asymmetric, foldback)
- `output` (-24 to +24 dB)
- `mix` (0 to 100%)

**Key Features**:
- Hard clipping with pre/post filtering
- Tone stack for shaping distortion character
- Multiple clipping algorithms
- High gain capability

### 4. Redux
**Purpose**: Bit crushing and sample rate reduction for lo-fi digital artifacts

**Parameters**:
- `bitDepth` (1 to 16 bits, default: 8)
- `sampleRate` (50 to 44100 Hz, default: 22050) - downsampling amount
- `hardness` (0 to 100%) - quantization curve
- `dither` (0 to 100%) - noise to reduce quantization
- `jitter` (0 to 100%) - sample timing variation
- `mix` (0 to 100%)

**Key Features**:
- Bit depth reduction (quantization)
- Sample rate reduction simulation
- Dithering to reduce harsh artifacts
- Jitter for analog-style instability

## Research Phase (Week 1)

### Essential Research Topics

1. **WaveShaperNode Mastery**
   - Transfer function curves
   - Oversampling settings (none, 2x, 4x)
   - Float32Array curve generation
   - Curve resolution (typical 4096 samples)

2. **Distortion Algorithms**
   - Soft clipping: `tanh(x)`, `atan(x)`, `x / (1 + |x|)`
   - Hard clipping: `clamp(x, -1, 1)`
   - Asymmetric: `f(x + bias)` for even harmonics
   - Foldback: `fold(x)` for complex harmonics
   - Sine folding: `sin(x * π)`

3. **Oversampling**
   - Aliasing in non-linear processing
   - WaveShaperNode oversample parameter
   - Manual upsampling/downsampling
   - Anti-aliasing filters

4. **Bit Crushing**
   - Quantization: `round(x * levels) / levels`
   - Sample rate reduction simulation
   - Dithering techniques (TPDF, noise shaping)
   - Jitter implementation

5. **DC Offset Removal**
   - Highpass filter at very low frequency (<10 Hz)
   - Running average subtraction
   - Importance after asymmetric distortion

6. **Tone Stacks**
   - Pre-distortion filtering (shapes input spectrum)
   - Post-distortion filtering (shapes harmonics)
   - Classic guitar amp tone circuits

### Reference Materials
- **Ableton Manual**: Overdrive, Saturator, Distortion, Redux
- **Web Audio API**: WaveShaperNode, oversampling
- **DSP Books**: "Designing Audio Effect Plugins in C++" (Pirkle) - distortion chapter
- **Academic**: Papers on virtual analog modeling
- **Code Examples**: Tone.js distortion effects

### Code to Study
```javascript
// WaveShaperNode with soft clipping curve
const shaper = context.createWaveShaper();
const curve = new Float32Array(4096);
const deg = Math.PI / 180;

for (let i = 0; i < 4096; i++) {
  const x = (i * 2 / 4096) - 1; // -1 to 1
  // Soft clip with tanh
  curve[i] = Math.tanh(x * 3); // Drive of 3
}

shaper.curve = curve;
shaper.oversample = '4x'; // Reduce aliasing

// Bit crushing algorithm
function bitCrush(input, bitDepth) {
  const levels = Math.pow(2, bitDepth);
  return Math.round(input * levels) / levels;
}

// Sample rate reduction
let lastSample = 0;
let sampleCounter = 0;
const reduction = sampleRate / targetRate;

function reduceSampleRate(input) {
  sampleCounter++;
  if (sampleCounter >= reduction) {
    lastSample = input;
    sampleCounter = 0;
  }
  return lastSample; // Hold sample
}
```

## Implementation Phase (Week 2-3)

### Architecture Pattern - Overdrive

```javascript
class Overdrive {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Pre-gain (drive)
    this.preGain = audioContext.createGain();

    // Waveshaper (soft clipping)
    this.shaper = audioContext.createWaveShaper();
    this.shaper.oversample = '4x';

    // Tone control (post-distortion)
    this.toneFilter = audioContext.createBiquadFilter();
    this.toneFilter.type = 'lowshelf';

    // Output gain
    this.outputGain = audioContext.createGain();

    // Dry/wet
    this.dryGain = audioContext.createGain();
    this.wetGain = audioContext.createGain();

    this.setupRouting();
    this.initialize(options);
  }

  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path
    this.input.connect(this.preGain);
    this.preGain.connect(this.shaper);
    this.shaper.connect(this.toneFilter);
    this.toneFilter.connect(this.outputGain);
    this.outputGain.connect(this.wetGain);
    this.wetGain.connect(this.output);
  }

  setDrive(percent) {
    // Drive 0-100 maps to gain 1-20
    const drive = 1 + (percent / 100) * 19;
    this.preGain.gain.value = drive;

    // Update waveshaper curve with new drive
    this.updateCurve(drive);
  }

  updateCurve(drive, bias = 0) {
    const curve = new Float32Array(4096);

    for (let i = 0; i < 4096; i++) {
      let x = (i * 2 / 4096) - 1; // -1 to 1

      // Apply bias for asymmetric distortion
      x += bias;

      // Soft clipping with tanh
      curve[i] = Math.tanh(x * drive);
    }

    this.shaper.curve = curve;
  }

  setBias(percent) {
    const bias = percent / 100; // -1 to 1
    const drive = this.preGain.gain.value;
    this.updateCurve(drive, bias);
  }

  setTone(percent) {
    // Tone control: 0 = dark, 100 = bright
    const freq = 200 + (percent / 100) * 4800;
    this.toneFilter.frequency.value = freq;
    this.toneFilter.gain.value = (percent / 100) * 12 - 6;
  }
}
```

### Architecture Pattern - Saturator

```javascript
class Saturator {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    this.preGain = audioContext.createGain();
    this.shaper = audioContext.createWaveShaper();
    this.shaper.oversample = '4x';

    // DC filter
    this.dcFilter = audioContext.createBiquadFilter();
    this.dcFilter.type = 'highpass';
    this.dcFilter.frequency.value = 5; // 5 Hz highpass

    // Color (additional filtering)
    this.colorFilter = audioContext.createBiquadFilter();

    this.outputGain = audioContext.createGain();
    this.dryGain = audioContext.createGain();
    this.wetGain = audioContext.createGain();

    this.saturationType = 'warm';

    this.setupRouting();
    this.initialize(options);
  }

  setupRouting() {
    // Dry
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet
    this.input.connect(this.preGain);
    this.preGain.connect(this.shaper);
    this.shaper.connect(this.dcFilter);
    this.dcFilter.connect(this.colorFilter);
    this.colorFilter.connect(this.outputGain);
    this.outputGain.connect(this.wetGain);
    this.wetGain.connect(this.output);
  }

  setType(type) {
    this.saturationType = type;
    this.updateCurve();
  }

  updateCurve() {
    const curve = new Float32Array(4096);
    const drive = this.preGain.gain.value;

    for (let i = 0; i < 4096; i++) {
      const x = (i * 2 / 4096) - 1;

      switch (this.saturationType) {
        case 'warm':
          // Soft tanh saturation
          curve[i] = Math.tanh(x * drive);
          break;

        case 'digital':
          // Hard clipping
          curve[i] = Math.max(-1, Math.min(1, x * drive));
          break;

        case 'analog':
          // Asymmetric soft clip
          curve[i] = Math.tanh((x + 0.1) * drive);
          break;

        case 'clip':
          // Very hard clip
          curve[i] = x * drive > 0 ? 1 : -1;
          break;

        case 'foldback':
          // Foldback distortion
          const folded = x * drive;
          curve[i] = Math.abs(folded % 4 - 2) - 1;
          break;

        case 'sine-fold':
          // Sine folding
          curve[i] = Math.sin(x * drive * Math.PI);
          break;
      }
    }

    this.shaper.curve = curve;
  }

  setDrive(percent) {
    const drive = 1 + (percent / 100) * 9;
    this.preGain.gain.value = drive;
    this.updateCurve();
  }

  setColor(percent) {
    // Color adds harmonic emphasis
    const freq = 2000 + (percent / 100) * 6000;
    this.colorFilter.type = 'peaking';
    this.colorFilter.frequency.value = freq;
    this.colorFilter.Q.value = 2;
    this.colorFilter.gain.value = (percent / 100) * 6;
  }
}
```

### Architecture Pattern - Redux

```javascript
class Redux {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Use ScriptProcessorNode or AudioWorklet for bit crushing
    this.processor = null;

    this.bitDepth = 8;
    this.sampleRate = 22050;
    this.hardness = 1.0;
    this.ditherAmount = 0.0;
    this.jitterAmount = 0.0;

    this.setupProcessor();
    this.initialize(options);
  }

  setupProcessor() {
    // Use AudioWorklet for better performance
    // Fallback to ScriptProcessorNode if needed

    const bufferSize = 4096;
    this.processor = this.context.createScriptProcessor(bufferSize, 1, 1);

    this.processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      const output = e.outputBuffer.getChannelData(0);

      let sampleCounter = 0;
      let lastSample = 0;
      const reduction = this.context.sampleRate / this.sampleRate;

      for (let i = 0; i < input.length; i++) {
        // Sample rate reduction
        sampleCounter++;
        if (sampleCounter >= reduction) {
          // Jitter
          const jitter = (Math.random() * 2 - 1) * this.jitterAmount * reduction;
          if (sampleCounter >= reduction + jitter) {
            // Bit crushing
            const levels = Math.pow(2, this.bitDepth);

            // Dithering
            const dither = (Math.random() * 2 - 1) * this.ditherAmount / levels;

            // Quantize
            lastSample = Math.round((input[i] + dither) * levels) / levels;
            sampleCounter = 0;
          }
        }

        output[i] = lastSample;
      }
    };

    this.input.connect(this.processor);
    this.processor.connect(this.output);
  }

  setBitDepth(bits) {
    this.bitDepth = Math.max(1, Math.min(16, bits));
  }

  setSampleRate(rate) {
    this.sampleRate = Math.max(50, Math.min(this.context.sampleRate, rate));
  }

  setDither(percent) {
    this.ditherAmount = percent / 100;
  }

  setJitter(percent) {
    this.jitterAmount = percent / 100;
  }

  dispose() {
    this.processor.disconnect();
    this.input.disconnect();
    this.output.disconnect();
  }
}
```

### Testing Checklist

- [ ] Overdrive: Soft clipping sounds warm and musical
- [ ] Overdrive: Bias creates asymmetric distortion
- [ ] Overdrive: Tone control shapes character
- [ ] Saturator: All saturation types work correctly
- [ ] Saturator: No DC offset after distortion
- [ ] Saturator: Color parameter adds character
- [ ] Distortion: Hard clipping is aggressive
- [ ] Distortion: Pre/post filtering works
- [ ] Redux: Bit crushing is audible
- [ ] Redux: Sample rate reduction works
- [ ] Redux: Dithering reduces harshness
- [ ] All: Oversampling reduces aliasing
- [ ] All: No audio artifacts during parameter changes

## Deliverables

### Code Files
```
/distortion/
├── Overdrive.js
├── Saturator.js
├── Distortion.js
├── Redux.js
├── README.md
└── worklets/
    └── redux-processor.js (AudioWorklet version)
```

### Example HTML
Create `/examples/distortion-shootout-example.html`:
- Overdrive on various sources
- Saturator type comparison
- Distortion aggressiveness demo
- Redux lo-fi effects
- Distortion chain examples

## Success Criteria

✅ All 4 distortion plugins implemented
✅ Soft clipping sounds musical
✅ Hard clipping is aggressive
✅ Bit crushing creates lo-fi effect
✅ Oversampling reduces aliasing
✅ DC filtering works correctly
✅ Code is modular and documented

Begin with WaveShaperNode research. Study various clipping curves! 🎸
