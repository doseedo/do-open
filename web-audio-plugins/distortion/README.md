# Distortion & Saturation Plugins

A comprehensive collection of Web Audio API distortion and saturation plugins for adding harmonic richness, warmth, grit, and character to your audio.

## Overview

This library provides **4 professional-grade distortion/saturation plugins**:

1. **Overdrive** - Tube-style soft clipping for warm, musical distortion
2. **Saturator** - Multi-mode saturation from subtle warmth to heavy distortion
3. **Distortion** - Hard clipping distortion with aggressive harmonic generation
4. **Redux** - Bit crushing and sample rate reduction for lo-fi digital artifacts

All plugins are built using the Web Audio API and support:
- ✅ Oversampling to reduce aliasing
- ✅ Dry/wet mix control
- ✅ Parameter automation
- ✅ Real-time processing
- ✅ Zero-latency operation

## Installation

```javascript
// Include the plugins in your HTML
<script src="distortion/Overdrive.js"></script>
<script src="distortion/Saturator.js"></script>
<script src="distortion/Distortion.js"></script>
<script src="distortion/Redux.js"></script>
```

## Quick Start

```javascript
// Create audio context
const audioContext = new AudioContext();

// Create an oscillator for testing
const osc = audioContext.createOscillator();
osc.frequency.value = 440; // A4

// Create Overdrive plugin
const overdrive = new Overdrive(audioContext, {
  drive: 50,
  tone: 60,
  mix: 100
});

// Connect: oscillator → overdrive → speakers
osc.connect(overdrive.input);
overdrive.connect(audioContext.destination);

// Start
osc.start();
```

---

## Plugin Reference

### 1. Overdrive

Tube-style soft clipping for warm, musical distortion.

#### Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `drive` | 0-100% | 30 | Amount of gain before saturation |
| `tone` | 0-100% | 50 | Post-distortion tone control (dark to bright) |
| `bias` | -100 to +100% | 0 | Asymmetric distortion for even harmonics |
| `output` | -24 to +24 dB | 0 | Makeup gain |
| `mix` | 0-100% | 100 | Dry/wet balance |

#### Curve Types

- `tanh` - Hyperbolic tangent (smooth, warm)
- `atan` - Arctangent (softer than tanh)
- `softClip` - Algebraic soft clipping

#### Usage Example

```javascript
const overdrive = new Overdrive(audioContext, {
  drive: 40,
  tone: 70,
  bias: 10,
  output: -3,
  mix: 80
});

// Change parameters
overdrive.setDrive(60);
overdrive.setTone(50);
overdrive.setBias(-20);
overdrive.setCurveType('atan');

// Get current parameters
const params = overdrive.getParams();
console.log(params);
```

#### Key Features

- **Soft clipping** using tanh, atan, or custom curves
- **Asymmetric distortion** creates even harmonics (richer sound)
- **Tone stack** shapes the character (lowshelf filter)
- **4x oversampling** reduces aliasing artifacts

---

### 2. Saturator

Multi-mode saturation from subtle warmth to heavy distortion.

#### Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `drive` | 0-100% | 0 | Saturation intensity |
| `type` | enum | 'warm' | Saturation algorithm |
| `color` | 0-100% | 0 | Harmonic emphasis (2-8 kHz) |
| `depth` | 0-100% | 100 | Wet/dry character control |
| `dcFilter` | boolean | true | Remove DC offset |
| `output` | -24 to +24 dB | 0 | Makeup gain |
| `mix` | 0-100% | 100 | Dry/wet balance |

#### Saturation Types

| Type | Description |
|------|-------------|
| `warm` | Soft tanh saturation (warm, musical) |
| `digital` | Hard clipping (aggressive digital) |
| `analog` | Asymmetric soft clip (analog circuit simulation) |
| `clip` | Very hard clip (extreme distortion) |
| `foldback` | Foldback distortion (complex harmonics) |
| `sine-fold` | Sine folding (musical harmonics) |

#### Usage Example

```javascript
const saturator = new Saturator(audioContext, {
  drive: 30,
  type: 'warm',
  color: 40,
  depth: 100,
  dcFilter: true
});

// Change saturation type
saturator.setType('analog');
saturator.setDrive(50);
saturator.setColor(60);

// Get parameters
console.log(saturator.getParams());
```

#### Key Features

- **6 saturation algorithms** for different sonic characters
- **DC offset removal** (5 Hz highpass filter)
- **Color filter** adds harmonic emphasis
- **Depth parameter** blends between dry and saturated

---

### 3. Distortion

Hard clipping distortion with aggressive harmonic generation.

#### Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `drive` | 0-100% | 50 | Distortion intensity |
| `tone` | 20 Hz - 20 kHz | 1000 | Tone center frequency |
| `toneWidth` | 0.1 - 10 | 1 | Tone Q (resonance) |
| `filterPosition` | 'pre' or 'post' | 'post' | Filter placement |
| `clipType` | enum | 'hard' | Clipping algorithm |
| `output` | -24 to +24 dB | 0 | Makeup gain |
| `mix` | 0-100% | 100 | Dry/wet balance |

#### Clipping Types

| Type | Description |
|------|-------------|
| `hard` | Brick wall limiting (aggressive) |
| `soft` | Tanh clipping (smoother) |
| `asymmetric` | Different threshold for +/- (tube-like) |
| `foldback` | Signal folds back at threshold |

#### Usage Example

```javascript
const distortion = new Distortion(audioContext, {
  drive: 70,
  tone: 1500,
  toneWidth: 2,
  filterPosition: 'pre',
  clipType: 'hard'
});

// Change parameters
distortion.setDrive(80);
distortion.setTone(2000);
distortion.setClipType('foldback');
distortion.setFilterPosition('post');
```

#### Key Features

- **High gain capability** (up to 50x drive)
- **Pre/post filtering** for tonal shaping
- **4 clipping algorithms** for different distortion characters
- **Tone stack** with adjustable frequency and Q

---

### 4. Redux

Bit crushing and sample rate reduction for lo-fi digital artifacts.

#### Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `bitDepth` | 1-16 bits | 8 | Bit depth reduction |
| `sampleRate` | 50-44100 Hz | 22050 | Target sample rate |
| `hardness` | 0-100% | 50 | Quantization curve (soft to hard) |
| `dither` | 0-100% | 0 | Noise to reduce quantization artifacts |
| `jitter` | 0-100% | 0 | Sample timing variation |
| `mix` | 0-100% | 100 | Dry/wet balance |

#### Usage Example

```javascript
const redux = new Redux(audioContext, {
  bitDepth: 4,
  sampleRate: 8000,
  hardness: 80,
  dither: 20,
  jitter: 10
});

// Change parameters
redux.setBitDepth(6);
redux.setSampleRate(11025);
redux.setDither(30);
redux.setJitter(15);
```

#### Key Features

- **Bit depth reduction** (1-16 bits) for digital degradation
- **Sample rate reduction** (50-44100 Hz) for lo-fi effect
- **TPDF dithering** reduces harsh quantization artifacts
- **Jitter** adds analog-style timing instability
- **Hardness parameter** controls quantization curve

#### AudioWorklet Version

For production use, Redux includes an AudioWorklet processor for better performance:

```javascript
// Load AudioWorklet processor
await audioContext.audioWorklet.addModule('distortion/worklets/redux-processor.js');

// Create AudioWorklet node
const reduxNode = new AudioWorkletNode(audioContext, 'redux-processor');

// Send parameters
reduxNode.port.postMessage({
  type: 'setParams',
  params: {
    bitDepth: 8,
    sampleRate: 22050,
    hardness: 50,
    dither: 20,
    jitter: 10
  }
});

// Connect
source.connect(reduxNode);
reduxNode.connect(audioContext.destination);
```

---

## Common API

All plugins share a common API:

### Methods

#### `connect(destination)`
Connect the plugin output to a destination node.

```javascript
overdrive.connect(audioContext.destination);
```

#### `disconnect()`
Disconnect from all destinations.

```javascript
overdrive.disconnect();
```

#### `setParams(params)`
Set multiple parameters at once.

```javascript
overdrive.setParams({
  drive: 60,
  tone: 70,
  mix: 80
});
```

#### `getParams()`
Get current parameter values.

```javascript
const params = overdrive.getParams();
console.log(params);
```

#### `dispose()`
Clean up and release resources.

```javascript
overdrive.dispose();
```

### Properties

All plugins have:
- `input` - Input gain node (connect sources here)
- `output` - Output gain node (connect to destination)

---

## Advanced Usage

### Chaining Effects

```javascript
const overdrive = new Overdrive(audioContext, { drive: 30 });
const saturator = new Saturator(audioContext, { type: 'analog', drive: 20 });
const redux = new Redux(audioContext, { bitDepth: 8, sampleRate: 22050 });

// Chain: source → overdrive → saturator → redux → speakers
source.connect(overdrive.input);
overdrive.connect(saturator.input);
saturator.connect(redux.input);
redux.connect(audioContext.destination);
```

### Parallel Processing

```javascript
const dry = audioContext.createGain();
const wet1 = audioContext.createGain();
const wet2 = audioContext.createGain();
const merger = audioContext.createGain();

const overdrive = new Overdrive(audioContext);
const distortion = new Distortion(audioContext);

// Split signal
source.connect(dry);
source.connect(overdrive.input);
source.connect(distortion.input);

// Mix parallel paths
dry.connect(merger);
overdrive.connect(wet1);
wet1.connect(merger);
distortion.connect(wet2);
wet2.connect(merger);

merger.connect(audioContext.destination);
```

### Parameter Automation

```javascript
const overdrive = new Overdrive(audioContext);

// Automate drive parameter
const now = audioContext.currentTime;
overdrive.preGain.gain.setValueAtTime(1, now);
overdrive.preGain.gain.linearRampToValueAtTime(10, now + 2);
overdrive.preGain.gain.linearRampToValueAtTime(1, now + 4);
```

---

## Technical Details

### Oversampling

All WaveShaperNode-based plugins (Overdrive, Saturator, Distortion) use **4x oversampling** to reduce aliasing artifacts caused by non-linear processing.

### DC Offset Removal

Saturator includes a **5 Hz highpass filter** to remove DC offset that can occur after asymmetric distortion.

### Bit Crushing Algorithm

Redux uses:
- **TPDF dithering** (Triangular Probability Density Function) for smooth quantization
- **Sample and hold** for sample rate reduction
- **Jitter** adds random timing variation to each sample

### Performance Notes

- **Overdrive, Saturator, Distortion**: Very efficient (WaveShaperNode)
- **Redux (ScriptProcessor)**: Higher CPU usage, suitable for moderate use
- **Redux (AudioWorklet)**: Recommended for production (separate audio thread)

---

## Browser Compatibility

- ✅ Chrome 66+ (full support including AudioWorklet)
- ✅ Firefox 76+ (full support including AudioWorklet)
- ✅ Safari 14.1+ (full support including AudioWorklet)
- ⚠️ Older browsers: use ScriptProcessorNode version of Redux

---

## Examples

See `/distortion/examples/distortion-shootout-example.html` for interactive demonstrations of all plugins.

---

## Research & References

### Distortion Algorithms
- Yeh, D. T., & Smith, J. O. (2006). "Simulating Guitar Distortion Circuits Using Wave Digital and Nonlinear State-Space Formulations"
- Pirkle, W. (2019). "Designing Audio Effect Plugins in C++" - Chapter on Distortion

### Web Audio API
- [WaveShaperNode Documentation](https://developer.mozilla.org/en-US/docs/Web/API/WaveShaperNode)
- [AudioWorklet Documentation](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet)

### Classic Guitar Pedals
- Ibanez Tube Screamer (overdrive)
- ProCo RAT (distortion)
- Boss DS-1 (distortion)
- Ableton Live: Redux device

---

## License

MIT License - feel free to use in your projects!

---

## Contributing

Contributions welcome! Please submit issues and pull requests.

### Future Enhancements
- [ ] Additional clipping curves (exponential, cubic, etc.)
- [ ] Pre-emphasis/de-emphasis filtering
- [ ] Multi-band distortion
- [ ] FFT-based harmonic analysis
- [ ] Visual feedback (waveform, spectrum analyzer)
- [ ] Preset system

---

**Happy distorting! 🎸**
