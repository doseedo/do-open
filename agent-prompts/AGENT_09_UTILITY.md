# Agent 9: Utility & Analysis Tools

## Your Mission
You are responsible for implementing **4 utility and analysis plugins**: Utility, Spectrum Analyzer, Tuner, and Channel EQ. These are essential tools for gain staging, visualization, tuning, and basic frequency adjustments in any production environment.

## Plugins to Implement

### 1. Utility
**Purpose**: Essential gain, pan, phase, and stereo width control

**Parameters**:
- `gain` (-inf to +35 dB, default: 0)
- `panL` (-100 to +100, default: 0) - left channel pan
- `panR` (-100 to +100, default: 0) - right channel pan
- `width` (0 to 200%, default: 100) - stereo width (0 = mono, 200 = extra wide)
- `balance` (-100 to +100, default: 0) - L/R balance
- `phaseL` (boolean, default: false) - invert left phase
- `phaseR` (boolean, default: false) - invert right phase
- `mono` (boolean, default: false) - sum to mono
- `swap` (boolean, default: false) - swap L/R channels
- `dcFilter` (boolean, default: false) - remove DC offset
- `bassMono` (frequency: 0-500 Hz, default: 0/off) - mono below frequency

**Key Features**:
- Precise gain control with dB scaling
- Independent L/R panning
- Stereo width (Mid/Side processing)
- Phase inversion per channel
- Channel swap
- DC offset removal
- Bass mono (mono low frequencies)

### 2. Spectrum Analyzer
**Purpose**: Real-time FFT-based frequency spectrum display

**Parameters**:
- `fftSize` (512, 1024, 2048, 4096, 8192, 16384) - frequency resolution
- `smoothing` (0 to 100%, default: 80) - temporal smoothing
- `scale` (linear, logarithmic) - frequency axis
- `range` (-100 to 0 dB) - display range
- `peakHold` (boolean, default: false) - show peak values
- `freeze` (boolean, default: false) - freeze display
- `tilt` (-6 to +6 dB/oct) - tilt display curve
- `channels` (L, R, L+R, Mid, Side) - which channels to display

**Key Features**:
- Real-time FFT visualization
- Multiple FFT sizes for resolution control
- Smoothing for stable display
- Peak hold mode
- Freeze function
- Canvas or SVG rendering
- dB scale with adjustable range

### 3. Tuner
**Purpose**: Pitch detection and tuning reference

**Parameters**:
- `referenceFreq` (400 to 480 Hz, default: 440) - A4 reference
- `input` (auto, L, R) - which channel to analyze
- `mode` (chromatic, guitar, bass, etc.) - tuning mode
- `tolerance` (±50 cents) - detection threshold

**Display**:
- Current note (C, C#, D, etc.)
- Octave number
- Cents deviation (+/- from note)
- Frequency in Hz
- Visual indicator (meter or needle)

**Key Features**:
- Accurate pitch detection (autocorrelation or YIN algorithm)
- Chromatic tuning
- Adjustable reference frequency
- Cents deviation display
- Real-time updates

### 4. Channel EQ
**Purpose**: Simple low-cut and high-cut filters for mixing

**Parameters**:
- `lowCut` (off, 20 to 500 Hz) - highpass frequency
- `lowCutSlope` (6, 12, 18, 24, 36, 48 dB/oct) - filter steepness
- `highCut` (off, 2k to 20k Hz) - lowpass frequency
- `highCutSlope` (6, 12, 18, 24, 36, 48 dB/oct) - filter steepness

**Key Features**:
- Simple UI (just two filters)
- Multiple slope options
- Efficient processing
- Visual frequency response

## Research Phase (Week 1)

### Essential Research Topics

1. **dB to Gain Conversion**
   - `gain = 10^(dB / 20)`
   - `dB = 20 * log10(gain)`
   - -inf dB handling (gain = 0)

2. **Mid/Side Processing**
   - Mid = (L + R) / 2
   - Side = (L - R) / 2
   - Reconstruction: L = Mid + Side, R = Mid - Side
   - Stereo width control via Side gain

3. **Pan Law**
   - Constant power panning: -3dB center
   - Linear panning: -6dB center
   - Web Audio StereoPannerNode uses constant power

4. **AnalyserNode**
   - FFT computation
   - Frequency bin calculation
   - Time domain vs frequency domain
   - Smoothing time constant

5. **Pitch Detection Algorithms**
   - Autocorrelation (simple, robust)
   - YIN algorithm (more accurate)
   - FFT-based methods
   - Frequency to note conversion

6. **Filter Slopes**
   - 1st order = 6 dB/oct (single biquad)
   - 2nd order = 12 dB/oct (single biquad lowpass/highpass)
   - Higher orders = cascade multiple biquads
   - Butterworth vs Linkwitz-Riley

### Reference Materials
- **Ableton Manual**: Utility, Spectrum, Tuner, Channel EQ
- **Web Audio API**: AnalyserNode, getByteFrequencyData, getFloatFrequencyData
- **DSP**: Mid/Side processing, filter design
- **Pitch Detection**: YIN algorithm paper, autocorrelation methods

### Code to Study
```javascript
// dB to gain
function dbToGain(db) {
  if (db === -Infinity) return 0;
  return Math.pow(10, db / 20);
}

// Mid/Side processing
function stereoToMS(left, right) {
  const mid = (left + right) / 2;
  const side = (left - right) / 2;
  return { mid, side };
}

function MSToStereo(mid, side) {
  const left = mid + side;
  const right = mid - side;
  return { left, right };
}

// AnalyserNode
const analyser = context.createAnalyser();
analyser.fftSize = 2048;
const bufferLength = analyser.frequencyBinCount;
const dataArray = new Uint8Array(bufferLength);

analyser.getByteFrequencyData(dataArray);

// Autocorrelation pitch detection
function autoCorrelate(buffer, sampleRate) {
  let SIZE = buffer.length;
  let rms = 0;

  for (let i = 0; i < SIZE; i++) {
    let val = buffer[i];
    rms += val * val;
  }
  rms = Math.sqrt(rms / SIZE);

  if (rms < 0.01) return -1; // Too quiet

  // Autocorrelation
  let r1 = 0, r2 = SIZE - 1, threshold = 0.2;
  // ... autocorrelation implementation
}
```

## Implementation Phase (Week 2-3)

### Architecture Pattern - Utility

```javascript
class Utility {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Stereo split/merge
    this.splitter = audioContext.createChannelSplitter(2);
    this.merger = audioContext.createChannelMerger(2);

    // Individual channel processing
    this.gainL = audioContext.createGain();
    this.gainR = audioContext.createGain();

    this.panL = audioContext.createStereoPanner();
    this.panR = audioContext.createStereoPanner();

    // DC filter
    this.dcFilterL = audioContext.createBiquadFilter();
    this.dcFilterR = audioContext.createBiquadFilter();
    this.dcFilterL.type = 'highpass';
    this.dcFilterR.type = 'highpass';
    this.dcFilterL.frequency.value = 5;
    this.dcFilterR.frequency.value = 5;

    // Width (Mid/Side processing)
    this.widthProcessor = null; // Needs custom processing

    this.setupRouting();
    this.initialize(options);
  }

  setupRouting() {
    // Basic routing: input → splitter → gains → merger → output
    this.input.connect(this.splitter);

    this.splitter.connect(this.gainL, 0);
    this.splitter.connect(this.gainR, 1);

    this.gainL.connect(this.dcFilterL);
    this.gainR.connect(this.dcFilterR);

    this.dcFilterL.connect(this.merger, 0, 0);
    this.dcFilterR.connect(this.merger, 0, 1);

    this.merger.connect(this.output);
  }

  setGain(db) {
    const gain = this.dbToGain(db);
    this.gainL.gain.value = gain;
    this.gainR.gain.value = gain;
  }

  dbToGain(db) {
    if (db === -Infinity) return 0;
    return Math.pow(10, db / 20);
  }

  setPan(channel, value) {
    // value: -100 to +100
    const pan = value / 100; // -1 to 1

    if (channel === 'L') {
      // Pan left channel
    } else if (channel === 'R') {
      // Pan right channel
    }
  }

  setWidth(percent) {
    // 0 = mono, 100 = normal, 200 = extra wide
    // Requires Mid/Side processing
    // Need custom processing or AudioWorklet

    const width = percent / 100;

    // M/S processing:
    // - Extract mid and side
    // - Scale side by width
    // - Reconstruct L/R
  }

  setPhase(channel, invert) {
    if (invert) {
      if (channel === 'L') {
        this.gainL.gain.value *= -1;
      } else if (channel === 'R') {
        this.gainR.gain.value *= -1;
      }
    } else {
      // Restore normal polarity
      if (channel === 'L') {
        this.gainL.gain.value = Math.abs(this.gainL.gain.value);
      } else if (channel === 'R') {
        this.gainR.gain.value = Math.abs(this.gainR.gain.value);
      }
    }
  }

  setMono(enabled) {
    if (enabled) {
      // Sum to mono: (L + R) / 2 to both channels
      // Requires custom processing
    }
  }

  swapChannels(enabled) {
    if (enabled) {
      // Swap L and R
      // Reconnect splitter to merger with swapped indices
    }
  }

  setDCFilter(enabled) {
    // Enable/disable DC highpass filter
    if (!enabled) {
      this.dcFilterL.type = 'allpass';
      this.dcFilterR.type = 'allpass';
    } else {
      this.dcFilterL.type = 'highpass';
      this.dcFilterR.type = 'highpass';
    }
  }
}
```

### Architecture Pattern - Spectrum Analyzer

```javascript
class SpectrumAnalyzer {
  constructor(audioContext, canvasElement, options = {}) {
    this.context = audioContext;
    this.canvas = canvasElement;
    this.canvasContext = canvasElement.getContext('2d');

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Analyser node
    this.analyser = audioContext.createAnalyser();
    this.analyser.fftSize = 2048;
    this.analyser.smoothingTimeConstant = 0.8;

    // FFT data
    this.bufferLength = this.analyser.frequencyBinCount;
    this.dataArray = new Uint8Array(this.bufferLength);
    this.peakArray = new Uint8Array(this.bufferLength);

    // Display options
    this.freeze = false;
    this.peakHold = false;
    this.scale = 'logarithmic';

    this.setupRouting();
    this.startAnimation();
    this.initialize(options);
  }

  setupRouting() {
    this.input.connect(this.analyser);
    this.input.connect(this.output); // Pass-through
  }

  startAnimation() {
    const draw = () => {
      requestAnimationFrame(draw);

      if (!this.freeze) {
        this.analyser.getByteFrequencyData(this.dataArray);
      }

      // Update peak hold
      if (this.peakHold) {
        for (let i = 0; i < this.bufferLength; i++) {
          if (this.dataArray[i] > this.peakArray[i]) {
            this.peakArray[i] = this.dataArray[i];
          } else {
            this.peakArray[i] *= 0.99; // Slow decay
          }
        }
      }

      this.drawSpectrum();
    };

    draw();
  }

  drawSpectrum() {
    const WIDTH = this.canvas.width;
    const HEIGHT = this.canvas.height;
    const ctx = this.canvasContext;

    // Clear
    ctx.fillStyle = 'rgb(20, 20, 20)';
    ctx.fillRect(0, 0, WIDTH, HEIGHT);

    // Draw frequency bars
    const barWidth = (WIDTH / this.bufferLength) * 2.5;
    let barHeight;
    let x = 0;

    for (let i = 0; i < this.bufferLength; i++) {
      barHeight = (this.dataArray[i] / 255) * HEIGHT;

      // Color gradient based on frequency
      const hue = (i / this.bufferLength) * 360;
      ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;

      if (this.scale === 'logarithmic') {
        // Logarithmic frequency spacing
        // Map linear bins to log scale
      }

      ctx.fillRect(x, HEIGHT - barHeight, barWidth, barHeight);

      // Draw peak hold
      if (this.peakHold) {
        const peakHeight = (this.peakArray[i] / 255) * HEIGHT;
        ctx.fillStyle = 'white';
        ctx.fillRect(x, HEIGHT - peakHeight, barWidth, 2);
      }

      x += barWidth + 1;
    }

    // Draw frequency labels
    this.drawFrequencyLabels(ctx, WIDTH, HEIGHT);
  }

  drawFrequencyLabels(ctx, width, height) {
    ctx.fillStyle = 'white';
    ctx.font = '10px monospace';

    const frequencies = [100, 1000, 10000];
    frequencies.forEach(freq => {
      const x = this.frequencyToX(freq, width);
      ctx.fillText(`${freq}Hz`, x, height - 5);
    });
  }

  frequencyToX(freq, width) {
    const nyquist = this.context.sampleRate / 2;

    if (this.scale === 'logarithmic') {
      return width * Math.log(freq / 20) / Math.log(nyquist / 20);
    } else {
      return width * freq / nyquist;
    }
  }

  setFFTSize(size) {
    this.analyser.fftSize = size;
    this.bufferLength = this.analyser.frequencyBinCount;
    this.dataArray = new Uint8Array(this.bufferLength);
    this.peakArray = new Uint8Array(this.bufferLength);
  }

  setSmoothing(percent) {
    this.analyser.smoothingTimeConstant = percent / 100;
  }

  setFreeze(enabled) {
    this.freeze = enabled;
  }

  setPeakHold(enabled) {
    this.peakHold = enabled;
    if (!enabled) {
      this.peakArray.fill(0);
    }
  }
}
```

### Architecture Pattern - Tuner

```javascript
class Tuner {
  constructor(audioContext, displayElement, options = {}) {
    this.context = audioContext;
    this.display = displayElement;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Analyser for pitch detection
    this.analyser = audioContext.createAnalyser();
    this.analyser.fftSize = 2048;

    this.bufferLength = this.analyser.fftSize;
    this.buffer = new Float32Array(this.bufferLength);

    this.referenceFreq = 440; // A4

    this.setupRouting();
    this.startDetection();
    this.initialize(options);
  }

  setupRouting() {
    this.input.connect(this.analyser);
    this.input.connect(this.output);
  }

  startDetection() {
    const detect = () => {
      requestAnimationFrame(detect);

      this.analyser.getFloatTimeDomainData(this.buffer);

      const pitch = this.detectPitch(this.buffer, this.context.sampleRate);

      if (pitch !== -1) {
        const note = this.frequencyToNote(pitch);
        this.updateDisplay(note);
      }
    };

    detect();
  }

  detectPitch(buffer, sampleRate) {
    // Simple autocorrelation pitch detection
    let SIZE = buffer.length;
    let sumOfSquares = 0;

    for (let i = 0; i < SIZE; i++) {
      let val = buffer[i];
      sumOfSquares += val * val;
    }

    let rms = Math.sqrt(sumOfSquares / SIZE);
    if (rms < 0.01) return -1; // Signal too quiet

    // Autocorrelation
    let r1 = 0, r2 = SIZE - 1;
    let threshold = 0.2;

    // Find first zero crossing
    for (let i = 0; i < SIZE / 2; i++) {
      if (Math.abs(buffer[i]) < threshold) {
        r1 = i;
        break;
      }
    }

    // Find peak autocorrelation
    let maxCorr = 0;
    let maxLag = -1;

    for (let lag = r1; lag < SIZE / 2; lag++) {
      let corr = 0;

      for (let i = 0; i < SIZE / 2; i++) {
        corr += buffer[i] * buffer[i + lag];
      }

      if (corr > maxCorr) {
        maxCorr = corr;
        maxLag = lag;
      }
    }

    if (maxLag === -1) return -1;

    return sampleRate / maxLag;
  }

  frequencyToNote(frequency) {
    // Calculate note and cents from frequency
    const noteNum = 12 * (Math.log(frequency / this.referenceFreq) / Math.log(2));
    const roundedNote = Math.round(noteNum);
    const cents = Math.round((noteNum - roundedNote) * 100);

    const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const noteIndex = (roundedNote + 9 + 120) % 12; // A = 0
    const octave = Math.floor((roundedNote + 9) / 12) + 4;

    return {
      note: noteNames[noteIndex],
      octave: octave,
      cents: cents,
      frequency: frequency.toFixed(2)
    };
  }

  updateDisplay(noteInfo) {
    // Update HTML display with note, octave, cents, frequency
    this.display.innerHTML = `
      <div class="note">${noteInfo.note}${noteInfo.octave}</div>
      <div class="cents">${noteInfo.cents > 0 ? '+' : ''}${noteInfo.cents}¢</div>
      <div class="frequency">${noteInfo.frequency} Hz</div>
    `;
  }

  setReferenceFreq(freq) {
    this.referenceFreq = freq;
  }
}
```

### Architecture Pattern - Channel EQ

```javascript
class ChannelEQ {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Low cut (highpass) - cascade for higher slopes
    this.lowCutFilters = [];
    for (let i = 0; i < 4; i++) {
      const filter = audioContext.createBiquadFilter();
      filter.type = 'highpass';
      this.lowCutFilters.push(filter);
    }

    // High cut (lowpass) - cascade for higher slopes
    this.highCutFilters = [];
    for (let i = 0; i < 4; i++) {
      const filter = audioContext.createBiquadFilter();
      filter.type = 'lowpass';
      this.highCutFilters.push(filter);
    }

    this.setupRouting();
    this.initialize(options);
  }

  setupRouting() {
    // Chain filters
    let current = this.input;

    // Low cut filters in series
    this.lowCutFilters.forEach(filter => {
      current.connect(filter);
      current = filter;
    });

    // High cut filters in series
    this.highCutFilters.forEach(filter => {
      current.connect(filter);
      current = filter;
    });

    current.connect(this.output);
  }

  setLowCut(freq, slope = 12) {
    // slope: 6, 12, 18, 24, 36, 48 dB/oct
    const numFilters = slope / 12; // Each biquad is 12dB/oct for highpass

    for (let i = 0; i < this.lowCutFilters.length; i++) {
      if (i < numFilters) {
        this.lowCutFilters[i].frequency.value = freq;
        this.lowCutFilters[i].Q.value = 0.707; // Butterworth
      } else {
        // Bypass unused filters
        this.lowCutFilters[i].type = 'allpass';
      }
    }
  }

  setHighCut(freq, slope = 12) {
    const numFilters = slope / 12;

    for (let i = 0; i < this.highCutFilters.length; i++) {
      if (i < numFilters) {
        this.highCutFilters[i].frequency.value = freq;
        this.highCutFilters[i].Q.value = 0.707;
      } else {
        this.highCutFilters[i].type = 'allpass';
      }
    }
  }
}
```

### Testing Checklist

- [ ] Utility: Gain control is accurate in dB
- [ ] Utility: Pan works correctly
- [ ] Utility: Stereo width affects sides
- [ ] Utility: Phase inversion works
- [ ] Utility: DC filter removes offset
- [ ] Spectrum: Displays frequency spectrum
- [ ] Spectrum: FFT size changes resolution
- [ ] Spectrum: Peak hold works
- [ ] Spectrum: Freeze stops updates
- [ ] Tuner: Detects pitch accurately
- [ ] Tuner: Shows correct note and cents
- [ ] Tuner: Reference frequency adjustable
- [ ] Channel EQ: Low cut works
- [ ] Channel EQ: High cut works
- [ ] Channel EQ: Slopes are correct

## Deliverables

### Code Files
```
/utility/
├── Utility.js
├── SpectrumAnalyzer.js
├── Tuner.js
├── ChannelEQ.js
└── README.md
```

### Example HTML
Create `/examples/utility-tools-example.html`:
- Utility gain and pan demo
- Spectrum analyzer visualization
- Tuner with instrument input
- Channel EQ basic filtering
- Combined utility tools

## Success Criteria

✅ All 4 utility plugins implemented
✅ Utility has all features working
✅ Spectrum analyzer displays correctly
✅ Tuner detects pitch accurately
✅ Channel EQ filters work
✅ Visualizations are smooth
✅ Code is modular and documented

These are essential tools! Focus on accuracy and usability. 🛠️
