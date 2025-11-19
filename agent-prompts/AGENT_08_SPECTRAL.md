# Agent 8: Spectral & Advanced Processing

## Your Mission
You are responsible for implementing **4 advanced spectral processing plugins**: Spectral Time, Spectral Resonator, Frequency Shifter, and Vocoder. These plugins manipulate audio in the frequency domain using FFT-based techniques for unique timbral transformations.

## Plugins to Implement

### 1. Spectral Time
**Purpose**: Time stretching and freezing without pitch change using phase vocoder

**Parameters**:
- `stretch` (0.1 to 4.0x, default: 1.0) - time stretch factor
- `freeze` (button/toggle) - freeze spectrum
- `blur` (0 to 100%) - spectral blurring/smearing
- `shift` (-24 to +24 semitones) - pitch shift
- `formant` (-4 to +4 semitones) - formant shift
- `residual` (0 to 100%) - transient preservation
- `mix` (0 to 100%)

**Key Features**:
- Phase vocoder algorithm
- Independent time and pitch control
- Spectral freeze for ambient textures
- Formant preservation/shifting
- Transient separation

### 2. Spectral Resonator
**Purpose**: Resonant comb filtering based on spectral analysis and pitch tracking

**Parameters**:
- `mode` (pitched, noise, hybrid)
- `pitch` (MIDI note or frequency) - resonant frequency
- `decay` (0.1 to 10 seconds) - resonance decay time
- `color` (0 to 100%) - harmonic emphasis
- `harmonics` (1 to 32) - number of harmonics
- `stretch` (0.5 to 2.0) - harmonic spacing
- `detune` (-100 to +100 cents)
- `attack` (0 to 500 ms)
- `release` (10 to 5000 ms)
- `mix` (0 to 100%)

**Key Features**:
- Pitched resonance (comb filtering)
- Harmonic series generation
- Envelope following
- Pitch tracking (optional)

### 3. Frequency Shifter
**Purpose**: Linear frequency shifting (not pitch shifting) using single-sideband modulation

**Parameters**:
- `frequency` (-5000 to +5000 Hz, default: 0) - shift amount
- `fine` (-100 to +100 Hz) - fine tuning
- `mode` (up, down, wide) - shift direction
- `wideAmount` (0 to 100%) - stereo spread for wide mode
- `drive` (0 to 100%) - harmonic saturation
- `mix` (0 to 100%)

**Key Features**:
- Single-sideband modulation
- Linear (not logarithmic) frequency shift
- Creates inharmonic spectra
- Ring modulation-like effects at extreme settings

### 4. Vocoder
**Purpose**: Multi-band vocoder for imposing modulator's spectral envelope on carrier

**Parameters**:
- `bands` (8, 16, 32, 40) - number of frequency bands
- `range` (low, mid, high, full) - frequency range
- `loFreq` (20 to 500 Hz) - lowest band
- `hiFreq` (2k to 20k Hz) - highest band
- `attack` (0.1 to 100 ms) - envelope follower attack
- `release` (10 to 500 ms) - envelope follower release
- `formant` (-4 to +4 semitones) - shift carrier formants
- `resonance` (0 to 100%) - emphasize formants
- `carrierSource` (internal, external) - carrier signal source
- `carrierType` (noise, saw, pulse) - if internal
- `upperBandLevel` (0 to 100%) - high-frequency emphasis
- `mix` (0 to 100%)

**Key Features**:
- Multi-band filtering (bank of bandpass filters)
- Envelope followers per band
- Internal/external carrier
- Formant shifting
- Unvoiced detection

## Research Phase (Week 1)

### Essential Research Topics

1. **FFT and STFT**
   - Fast Fourier Transform fundamentals
   - Short-Time Fourier Transform (STFT)
   - Window functions (Hann, Hamming, Blackman)
   - Overlap-add reconstruction
   - FFT size vs time/frequency resolution

2. **Phase Vocoder**
   - Phase unwrapping
   - Time stretching without pitch change
   - Pitch shifting algorithms
   - Spectral freezing
   - Formant preservation

3. **AudioWorklet for FFT**
   - FFT libraries for JavaScript (e.g., FFT.js, KISS FFT port)
   - Real-time FFT processing
   - Circular buffer management
   - Latency considerations

4. **Single-Sideband Modulation**
   - Hilbert transform
   - Complex multiplication
   - Upper/lower sideband selection
   - Ring modulation relationship

5. **Vocoder Design**
   - Filterbank design (constant-Q or linear)
   - Envelope follower per band
   - Attack/release characteristics
   - Carrier synthesis
   - Formant shifting techniques

6. **Comb Filtering**
   - Delay-based comb filters
   - Feedforward vs feedback
   - Harmonic series
   - Pitch-based delay time calculation

### Reference Materials
- **Ableton Manual**: Spectral Time, Spectral Resonator, Frequency Shifter, Vocoder
- **DSP Books**:
  - "Spectral Audio Signal Processing" by Julius O. Smith
  - "DAFX: Digital Audio Effects" (phase vocoder chapter)
- **Academic Papers**:
  - "Phase Vocoder Tutorial" by Miller Puckette
  - "New Digital Techniques for Vocoder Implementations"
- **Web Resources**:
  - Web Audio FFT examples
  - AudioWorklet documentation

### Code to Study
```javascript
// AudioWorklet for FFT processing
class SpectralProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.fftSize = 2048;
    this.hopSize = this.fftSize / 4;
    this.inputBuffer = new Float32Array(this.fftSize);
    this.outputBuffer = new Float32Array(this.fftSize);
    this.position = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0][0];
    const output = outputs[0][0];

    // Collect input samples
    // Perform FFT
    // Process spectrum
    // Perform IFFT
    // Overlap-add output

    return true;
  }
}

// Hilbert transform for frequency shifter
function hilbertTransform(signal) {
  // Apply 90-degree phase shift
  // Used in single-sideband modulation
}

// Comb filter for resonator
const delay = context.createDelay(1.0);
const feedback = context.createGain();

delay.delayTime.value = 1 / frequency; // Period
feedback.gain.value = 0.99; // Long decay

delay.connect(feedback);
feedback.connect(delay); // Feedback loop
```

## Implementation Phase (Week 2-3)

### Architecture Pattern - Spectral Time

```javascript
class SpectralTime {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // AudioWorklet for phase vocoder
    this.workletNode = null;

    // Parameters
    this.stretch = 1.0;
    this.freeze = false;
    this.blur = 0.0;

    this.setupWorklet();
    this.initialize(options);
  }

  async setupWorklet() {
    // Load and create AudioWorklet
    await this.context.audioWorklet.addModule('spectral-time-processor.js');

    this.workletNode = new AudioWorkletNode(
      this.context,
      'spectral-time-processor'
    );

    this.input.connect(this.workletNode);
    this.workletNode.connect(this.output);
  }

  setStretch(factor) {
    this.stretch = factor;
    this.workletNode.port.postMessage({
      type: 'stretch',
      value: factor
    });
  }

  setFreeze(enabled) {
    this.freeze = enabled;
    this.workletNode.port.postMessage({
      type: 'freeze',
      value: enabled
    });
  }

  setBlur(percent) {
    this.blur = percent / 100;
    this.workletNode.port.postMessage({
      type: 'blur',
      value: this.blur
    });
  }
}

// spectral-time-processor.js (AudioWorklet)
class SpectralTimeProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    this.fftSize = 4096;
    this.hopSize = this.fftSize / 4;
    this.stretch = 1.0;
    this.freeze = false;

    // FFT buffers
    this.inputBuffer = new Float32Array(this.fftSize);
    this.outputBuffer = new Float32Array(this.fftSize);

    // Phase vocoder state
    this.previousPhase = new Float32Array(this.fftSize / 2);
    this.sumPhase = new Float32Array(this.fftSize / 2);

    this.port.onmessage = (e) => {
      if (e.data.type === 'stretch') {
        this.stretch = e.data.value;
      } else if (e.data.type === 'freeze') {
        this.freeze = e.data.value;
      }
    };
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0][0];
    const output = outputs[0][0];

    if (!input) return true;

    // Phase vocoder algorithm
    // 1. Window and FFT input
    // 2. Calculate magnitude and phase
    // 3. Phase unwrapping
    // 4. Time stretch by modifying hop size
    // 5. IFFT and overlap-add

    // This is complex - requires FFT library integration

    return true;
  }
}
```

### Architecture Pattern - Spectral Resonator

```javascript
class SpectralResonator {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Comb filter for each harmonic
    this.harmonics = [];
    this.numHarmonics = 8;

    this.fundamentalFreq = 440; // Hz

    for (let i = 0; i < this.numHarmonics; i++) {
      this.harmonics.push(this.createHarmonic(i + 1));
    }

    this.setupRouting();
    this.initialize(options);
  }

  createHarmonic(harmonicNumber) {
    const delay = this.context.createDelay(1.0);
    const feedback = this.context.createGain();
    const gain = this.context.createGain();

    // Harmonic frequency
    const freq = this.fundamentalFreq * harmonicNumber;
    delay.delayTime.value = 1 / freq;

    // Decay
    feedback.gain.value = 0.95;

    // Amplitude decreases with harmonic number
    gain.gain.value = 1 / harmonicNumber;

    // Comb filter structure
    delay.connect(feedback);
    feedback.connect(delay);

    return { delay, feedback, gain, harmonicNumber };
  }

  setupRouting() {
    // Parallel comb filters
    this.harmonics.forEach(harmonic => {
      this.input.connect(harmonic.delay);
      harmonic.delay.connect(harmonic.gain);
      harmonic.gain.connect(this.output);
    });
  }

  setPitch(freq) {
    this.fundamentalFreq = freq;

    this.harmonics.forEach(harmonic => {
      const harmonicFreq = freq * harmonic.harmonicNumber;
      harmonic.delay.delayTime.value = 1 / harmonicFreq;
    });
  }

  setDecay(seconds) {
    // Convert decay time to feedback gain
    const feedback = Math.pow(0.001, 1 / (seconds * this.context.sampleRate));

    this.harmonics.forEach(harmonic => {
      harmonic.feedback.gain.value = feedback;
    });
  }

  setHarmonics(num) {
    this.numHarmonics = num;
    // Rebuild harmonic array...
  }
}
```

### Architecture Pattern - Frequency Shifter

```javascript
class FrequencyShifter {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Single-sideband modulation
    // Requires Hilbert transform (90-degree phase shift)
    // This is complex and typically needs AudioWorklet

    this.workletNode = null;
    this.shiftFrequency = 0;

    this.setupWorklet();
    this.initialize(options);
  }

  async setupWorklet() {
    await this.context.audioWorklet.addModule('frequency-shifter-processor.js');

    this.workletNode = new AudioWorkletNode(
      this.context,
      'frequency-shifter-processor'
    );

    this.input.connect(this.workletNode);
    this.workletNode.connect(this.output);
  }

  setFrequency(hz) {
    this.shiftFrequency = hz;
    this.workletNode.port.postMessage({
      type: 'frequency',
      value: hz
    });
  }
}

// frequency-shifter-processor.js
class FrequencyShifterProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    this.shiftFrequency = 0;
    this.phase = 0;

    // Hilbert transformer (all-pass filter cascade)
    // Or use FFT-based approach

    this.port.onmessage = (e) => {
      if (e.data.type === 'frequency') {
        this.shiftFrequency = e.data.value;
      }
    };
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0][0];
    const output = outputs[0][0];

    if (!input) return true;

    for (let i = 0; i < input.length; i++) {
      // 1. Generate quadrature (90-degree phase shifted) signal
      // 2. Modulate both in-phase and quadrature with sine/cosine
      // 3. Sum or difference for upper/lower sideband

      const inputSample = input[i];
      const quadrature = this.hilbert(inputSample); // 90-degree shift

      const phaseDelta = (2 * Math.PI * this.shiftFrequency) / sampleRate;
      this.phase += phaseDelta;

      const cosine = Math.cos(this.phase);
      const sine = Math.sin(this.phase);

      // Single-sideband modulation
      output[i] = inputSample * cosine - quadrature * sine;
    }

    return true;
  }

  hilbert(sample) {
    // Implement Hilbert transform
    // Can use all-pass filter cascade or FFT method
  }
}
```

### Architecture Pattern - Vocoder

```javascript
class Vocoder {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain(); // Modulator input
    this.carrierInput = audioContext.createGain();
    this.output = audioContext.createGain();

    // Band count
    this.numBands = 16;
    this.bands = [];

    // Internal carrier oscillator
    this.internalCarrier = audioContext.createOscillator();
    this.internalCarrier.type = 'sawtooth';
    this.internalCarrier.frequency.value = 220;
    this.internalCarrier.start();

    for (let i = 0; i < this.numBands; i++) {
      this.bands.push(this.createBand(i));
    }

    this.setupRouting();
    this.initialize(options);
  }

  createBand(index) {
    // Calculate band frequency (logarithmic spacing)
    const loFreq = 80;
    const hiFreq = 12000;
    const ratio = Math.pow(hiFreq / loFreq, 1 / this.numBands);
    const centerFreq = loFreq * Math.pow(ratio, index);

    // Modulator analysis filter
    const modulatorFilter = this.context.createBiquadFilter();
    modulatorFilter.type = 'bandpass';
    modulatorFilter.frequency.value = centerFreq;
    modulatorFilter.Q.value = 10;

    // Envelope follower (rectifier + lowpass)
    const rectifier = this.context.createWaveShaper();
    rectifier.curve = this.createRectifierCurve();

    const envelopeFilter = this.context.createBiquadFilter();
    envelopeFilter.type = 'lowpass';
    envelopeFilter.frequency.value = 20; // Smooth envelope

    // Carrier synthesis filter
    const carrierFilter = this.context.createBiquadFilter();
    carrierFilter.type = 'bandpass';
    carrierFilter.frequency.value = centerFreq;
    carrierFilter.Q.value = 10;

    // VCA (voltage-controlled amplifier)
    const vca = this.context.createGain();
    vca.gain.value = 0;

    return {
      centerFreq,
      modulatorFilter,
      rectifier,
      envelopeFilter,
      carrierFilter,
      vca
    };
  }

  setupRouting() {
    this.bands.forEach(band => {
      // Modulator path: input → bandpass → rectify → envelope → VCA gain
      this.input.connect(band.modulatorFilter);
      band.modulatorFilter.connect(band.rectifier);
      band.rectifier.connect(band.envelopeFilter);
      band.envelopeFilter.connect(band.vca.gain);

      // Carrier path: carrier → bandpass → VCA → output
      this.carrierInput.connect(band.carrierFilter);
      this.internalCarrier.connect(band.carrierFilter);
      band.carrierFilter.connect(band.vca);
      band.vca.connect(this.output);
    });
  }

  createRectifierCurve() {
    const curve = new Float32Array(4096);
    for (let i = 0; i < 4096; i++) {
      const x = (i * 2 / 4096) - 1;
      curve[i] = Math.abs(x); // Full-wave rectifier
    }
    return curve;
  }

  setBands(num) {
    this.numBands = num;
    // Rebuild band array...
  }

  setAttack(ms) {
    // Adjust envelope filter frequency for attack time
  }

  setRelease(ms) {
    // Adjust envelope filter frequency for release time
  }
}
```

### Testing Checklist

- [ ] Spectral Time: Time stretching works without pitch change
- [ ] Spectral Time: Freeze captures and holds spectrum
- [ ] Spectral Time: No excessive artifacts
- [ ] Spectral Resonator: Harmonics resonate correctly
- [ ] Spectral Resonator: Decay time works
- [ ] Frequency Shifter: Linear frequency shift (not pitch)
- [ ] Frequency Shifter: Creates inharmonic spectra
- [ ] Vocoder: Modulator controls carrier envelope
- [ ] Vocoder: Band count affects character
- [ ] Vocoder: Internal carrier works
- [ ] All: Latency is acceptable
- [ ] All: CPU usage is reasonable

## Deliverables

### Code Files
```
/spectral/
├── SpectralTime.js
├── SpectralResonator.js
├── FrequencyShifter.js
├── Vocoder.js
├── README.md
└── worklets/
    ├── spectral-time-processor.js
    ├── frequency-shifter-processor.js
    └── fft-lib.js (FFT library)
```

### Example HTML
Create `/examples/spectral-processing-example.html`:
- Spectral Time stretching/freezing
- Spectral Resonator on percussion
- Frequency Shifter inharmonic effects
- Vocoder with various carriers
- Combined spectral effects

## Success Criteria

✅ All 4 spectral plugins implemented
✅ FFT processing works in AudioWorklet
✅ Time stretching is artifact-free (reasonable quality)
✅ Frequency shifter creates linear shifts
✅ Vocoder imposes spectral envelope correctly
✅ Latency is acceptable
✅ Code is modular and documented

This is the most technically demanding set! FFT and phase vocoder are complex. Take time! 🌈
