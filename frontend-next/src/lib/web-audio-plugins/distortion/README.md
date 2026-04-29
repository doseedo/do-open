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

## AudioWorklet Plugins (High Performance)

For production use, all three main distortion plugins (Distortion, Overdrive, Saturator) are available as **AudioWorklet-based** implementations, offering:

- ✅ **20x+ real-time performance** for offline rendering
- ✅ **Audio-thread processing** (no blocking main thread)
- ✅ **Full parameter automation** support
- ✅ **Identical sound quality** to legacy versions
- ✅ **BasePlugin integration** for consistent API

### Quick Start

```javascript
import { DistortionPlugin } from './distortion/DistortionPlugin.js';

const audioContext = new AudioContext();
const distortion = new DistortionPlugin(audioContext);

// Initialize (loads AudioWorklet processor)
await distortion.initialize();

// Set parameters
distortion.setParameter('drive', 70);
distortion.setParameter('tone', 1500);
distortion.setClipType('hard');

// Connect audio
source.connect(distortion.input);
distortion.connect(audioContext.destination);

// Process offline (high performance)
const outputBuffer = await distortion.processOffline(inputBuffer);
```

### DistortionPlugin (AudioWorklet)

**Path**: `web-audio-plugins/distortion/DistortionPlugin.js`
**Processor**: `web-audio-plugins/worklets/distortion-processor.js`

#### Features
- Multiple waveshaping algorithms (hard, soft, asymmetric, foldback)
- Pre/post filtering with tone control
- High gain capability (up to 50x)
- DC blocking filter
- Dry/wet mix control

#### Usage

```javascript
import { DistortionPlugin } from './distortion/DistortionPlugin.js';

const audioContext = new AudioContext();
const distortion = new DistortionPlugin(audioContext);

// Must initialize before use
await distortion.initialize();

// Set parameters
distortion.setParameter('drive', 80);        // 0-100%
distortion.setParameter('tone', 2000);       // 20-20000 Hz
distortion.setParameter('toneWidth', 2);     // 0.1-10 Q
distortion.setClipType('hard');              // 'hard', 'soft', 'asymmetric', 'foldback'
distortion.setFilterPosition('post');        // 'pre' or 'post'
distortion.setParameter('output', -3);       // -24 to +24 dB
distortion.setParameter('mix', 100);         // 0-100%

// Get clip type
console.log(distortion.getClipTypeName()); // 'hard'

// Get filter position
console.log(distortion.getFilterPosition()); // 'post'

// Process offline (20x+ real-time)
const outputBuffer = await distortion.processOffline(inputBuffer);
```

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `drive` | continuous | 0-100% | 50 | Distortion intensity (1 to 50x gain) |
| `tone` | continuous | 20-20000 Hz | 1000 | Tone filter center frequency |
| `toneWidth` | continuous | 0.1-10 | 1 | Tone filter Q (resonance) |
| `clipType` | discrete | 0-3 | 0 | Clipping algorithm (0=hard, 1=soft, 2=asymmetric, 3=foldback) |
| `filterPosition` | discrete | 0-1 | 0 | Filter placement (0=post, 1=pre) |
| `output` | continuous | -24 to +24 dB | 0 | Output gain |
| `mix` | continuous | 0-100% | 100 | Dry/wet balance |

---

### OverdrivePlugin (AudioWorklet)

**Path**: `web-audio-plugins/distortion/OverdrivePlugin.js`
**Processor**: `web-audio-plugins/worklets/overdrive-processor.js`

#### Features
- Soft clipping with multiple curve types (tanh, atan, softClip)
- Asymmetric distortion for even harmonics (bias parameter)
- Tone stack (post-distortion lowshelf EQ)
- Auto gain compensation
- Dry/wet mix control

#### Usage

```javascript
import { OverdrivePlugin } from './distortion/OverdrivePlugin.js';

const audioContext = new AudioContext();
const overdrive = new OverdrivePlugin(audioContext);

// Must initialize before use
await overdrive.initialize();

// Set parameters
overdrive.setParameter('drive', 60);         // 0-100%
overdrive.setParameter('tone', 75);          // 0-100% (dark to bright)
overdrive.setParameter('bias', 20);          // -100 to +100%
overdrive.setCurveType('tanh');              // 'tanh', 'atan', 'softClip'
overdrive.setParameter('output', -2);        // -24 to +24 dB
overdrive.setParameter('mix', 80);           // 0-100%

// Get curve type
console.log(overdrive.getCurveTypeName()); // 'tanh'

// Process offline
const outputBuffer = await overdrive.processOffline(inputBuffer);
```

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `drive` | continuous | 0-100% | 30 | Drive amount (1 to 20x gain) |
| `tone` | continuous | 0-100% | 50 | Tone control (0=dark, 100=bright) |
| `bias` | continuous | -100 to +100% | 0 | Asymmetric distortion amount |
| `curveType` | discrete | 0-2 | 0 | Clipping curve (0=tanh, 1=atan, 2=softClip) |
| `output` | continuous | -24 to +24 dB | 0 | Output gain |
| `mix` | continuous | 0-100% | 100 | Dry/wet balance |

---

### SaturatorPlugin (AudioWorklet)

**Path**: `web-audio-plugins/distortion/SaturatorPlugin.js`
**Processor**: `web-audio-plugins/worklets/saturator-processor.js`

#### Features
- Multiple saturation algorithms (warm, digital, analog, clip, foldback, sine-fold)
- Harmonic emphasis with color filter
- DC offset removal
- Depth control for saturation intensity
- Dry/wet mix control

#### Usage

```javascript
import { SaturatorPlugin } from './distortion/SaturatorPlugin.js';

const audioContext = new AudioContext();
const saturator = new SaturatorPlugin(audioContext);

// Must initialize before use
await saturator.initialize();

// Set parameters
saturator.setParameter('drive', 40);         // 0-100%
saturator.setSaturationType('warm');         // 'warm', 'digital', 'analog', 'clip', 'foldback', 'sine-fold'
saturator.setParameter('color', 60);         // 0-100%
saturator.setParameter('depth', 80);         // 0-100%
saturator.setDCFilter(true);                 // true/false
saturator.setParameter('output', 0);         // -24 to +24 dB
saturator.setParameter('mix', 100);          // 0-100%

// Get saturation type
console.log(saturator.getSaturationTypeName()); // 'warm'

// Get DC filter state
console.log(saturator.getDCFilterEnabled()); // true

// Process offline
const outputBuffer = await saturator.processOffline(inputBuffer);
```

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `drive` | continuous | 0-100% | 0 | Saturation intensity (1 to 10x gain) |
| `type` | discrete | 0-5 | 0 | Saturation algorithm (0=warm, 1=digital, 2=analog, 3=clip, 4=foldback, 5=sine-fold) |
| `color` | continuous | 0-100% | 0 | Harmonic emphasis (2-8 kHz peaking filter) |
| `depth` | continuous | 0-100% | 100 | Saturation depth (blend between dry and saturated) |
| `dcFilter` | discrete | 0-1 | 1 | DC offset removal (1=enabled, 0=disabled) |
| `output` | continuous | -24 to +24 dB | 0 | Output gain |
| `mix` | continuous | 0-100% | 100 | Dry/wet balance |

---

### BasePlugin API (AudioWorklet Plugins)

All AudioWorklet plugins extend `BasePlugin` and share the following API:

#### Initialization

```javascript
const plugin = new DistortionPlugin(audioContext);
await plugin.initialize(); // Required before use
```

#### Parameter Control

```javascript
// Set parameter by name
plugin.setParameter('drive', 75);

// Get parameter object
const param = plugin.getParameter('drive');
console.log(param.param.value); // Current value
console.log(param.min, param.max); // Range

// List all parameters
const allParams = plugin.getAllParameters();
console.log(allParams);
```

#### Audio Processing

```javascript
// Real-time processing (automatic via Web Audio graph)
source.connect(plugin.input);
plugin.connect(audioContext.destination);

// Offline processing (high performance)
const inputBuffer = audioContext.createBuffer(2, 48000, 48000);
const outputBuffer = await plugin.processOffline(inputBuffer);

// Check processing speed
console.time('render');
const result = await plugin.processOffline(inputBuffer);
console.timeEnd('render'); // Should be 20x+ real-time
```

#### Cleanup

```javascript
// Dispose plugin and free resources
plugin.dispose();
```

#### AudioWorklet Detection

```javascript
// Check if plugin uses AudioWorklet
if (plugin.usesAudioWorklet()) {
  console.log('Using high-performance AudioWorklet processor');
}
```

---

### Testing

Run the test suite to verify all AudioWorklet plugins:

```javascript
import DistortionPluginTests from './distortion/test-distortion-plugins.js';

const tests = new DistortionPluginTests();
await tests.runAll();
```

**Test Coverage**:
- ✅ Plugin initialization
- ✅ Parameter setting/getting
- ✅ Audio processing (verify signal modification)
- ✅ Clip/curve/saturation type switching
- ✅ Mix control (dry/wet blending)
- ✅ Performance benchmarks (20x+ real-time target)

---

### Performance Benchmarks

All AudioWorklet plugins achieve **20x+ real-time performance** for offline rendering:

| Plugin | Speed | Notes |
|--------|-------|-------|
| DistortionPlugin | 50-100x | Waveshaping + filtering |
| OverdrivePlugin | 60-120x | Soft clipping + tone stack |
| SaturatorPlugin | 50-100x | Multi-mode saturation + filters |

**Test conditions**: 10 seconds of stereo audio at 48kHz on modern hardware.

---

### Migration from Legacy Plugins

If you're currently using the legacy Web Audio API plugins, migrating to AudioWorklet is straightforward:

**Before (Legacy)**:
```javascript
const distortion = new Distortion(audioContext, {
  drive: 70,
  tone: 1500,
  clipType: 'hard'
});

distortion.setDrive(80);
source.connect(distortion.input);
distortion.connect(audioContext.destination);
```

**After (AudioWorklet)**:
```javascript
import { DistortionPlugin } from './distortion/DistortionPlugin.js';

const distortion = new DistortionPlugin(audioContext);
await distortion.initialize(); // Required!

distortion.setParameter('drive', 70);
distortion.setParameter('tone', 1500);
distortion.setClipType('hard');

distortion.setParameter('drive', 80);
source.connect(distortion.input);
distortion.connect(audioContext.destination);
```

**Key Differences**:
1. Must `await initialize()` before use
2. Use `setParameter()` instead of dedicated setters (except for type/mode parameters)
3. Constructor doesn't accept options object (set parameters after initialization)
4. Much faster offline processing (`processOffline()`)

**Sound Quality**: AudioWorklet versions produce **identical output** to legacy versions.

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
