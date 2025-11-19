# Utility & Analysis Tools

**Professional Web Audio API Plugins for Audio Production**

This collection provides 4 essential utility and analysis tools for audio production, mixing, and mastering. All plugins are built using the Web Audio API and provide professional-grade audio processing.

## 📦 Plugins Included

1. **Utility** - Essential gain, pan, phase, and stereo width control
2. **Spectrum Analyzer** - Real-time FFT-based frequency visualization
3. **Tuner** - Pitch detection and tuning reference
4. **Channel EQ** - Simple low-cut and high-cut filters

## 🚀 Quick Start

### Installation

Simply include the plugin files in your HTML:

```html
<script src="utility/Utility.js"></script>
<script src="utility/SpectrumAnalyzer.js"></script>
<script src="utility/Tuner.js"></script>
<script src="utility/ChannelEQ.js"></script>
```

### Basic Usage

```javascript
// Create audio context
const audioContext = new AudioContext();

// Create plugins
const utility = new Utility(audioContext, {
  gain: 0,
  width: 100,
  balance: 0
});

const spectrum = new SpectrumAnalyzer(
  audioContext,
  document.getElementById('canvas')
);

const tuner = new Tuner(
  audioContext,
  document.getElementById('display')
);

const channelEQ = new ChannelEQ(audioContext, {
  lowCutFreq: 80,
  lowCutSlope: 12
});

// Connect audio source → plugins → destination
audioSource.connect(utility.input);
utility.connect(channelEQ.input);
channelEQ.connect(spectrum.input);
channelEQ.connect(tuner.input);
spectrum.connect(audioContext.destination);
tuner.connect(audioContext.destination);
```

## 📖 Plugin Documentation

### 1. Utility

Essential gain, pan, phase, and stereo width control for precise audio manipulation.

#### Features

- Gain control: -∞ to +35 dB
- Independent L/R panning
- Stereo width: 0% (mono) to 200% (extra wide)
- L/R balance control
- Phase inversion per channel
- Channel swap (L↔R)
- DC offset removal filter
- Bass mono (mono low frequencies)

#### API

```javascript
const utility = new Utility(audioContext, options);

// Gain control
utility.setGain(db); // -Infinity to +35 dB

// Pan control (independent L/R)
utility.setPan('L', value); // -100 to +100
utility.setPan('R', value); // -100 to +100

// Stereo width
utility.setWidth(percent); // 0 to 200

// Balance
utility.setBalance(value); // -100 (left) to +100 (right)

// Phase inversion
utility.setPhase('L', true); // Invert left channel
utility.setPhase('R', true); // Invert right channel

// Mono
utility.setMono(true); // Sum to mono

// Channel swap
utility.setSwap(true); // Swap L/R

// DC filter
utility.setDCFilter(true); // Enable DC offset removal

// Bass mono
utility.setBassMono(frequency); // 0-500 Hz, 0 = off

// Get current state
const state = utility.getState();

// Connect/disconnect
utility.connect(destination);
utility.disconnect();

// Cleanup
utility.destroy();
```

#### Example

```javascript
const utility = new Utility(audioContext, {
  gain: -3,
  width: 120,
  balance: 0,
  dcFilter: true
});

// Apply gain staging
utility.setGain(-6);

// Widen stereo image
utility.setWidth(150);

// Fix phase issues
utility.setPhase('L', true);
```

---

### 2. Spectrum Analyzer

Real-time FFT-based frequency spectrum display with professional features.

#### Features

- Real-time FFT visualization
- FFT sizes: 512, 1024, 2048, 4096, 8192, 16384
- Temporal smoothing (0-100%)
- Linear or logarithmic frequency scale
- Adjustable dB range (-100 to 0 dB)
- Peak hold mode
- Freeze display
- Tilt compensation (-6 to +6 dB/octave)
- Multiple channel display modes (L, R, L+R, Mid, Side)

#### API

```javascript
const analyzer = new SpectrumAnalyzer(audioContext, canvasElement, options);

// FFT size
analyzer.setFFTSize(2048); // 512, 1024, 2048, 4096, 8192, 16384

// Smoothing
analyzer.setSmoothing(percent); // 0-100

// Frequency scale
analyzer.setScale('logarithmic'); // 'linear' or 'logarithmic'

// Peak hold
analyzer.setPeakHold(true);

// Freeze
analyzer.setFreeze(true);

// Tilt
analyzer.setTilt(tilt); // -6 to +6 dB/octave

// Channel mode
analyzer.setChannels('L+R'); // 'L', 'R', 'L+R', 'Mid', 'Side'

// Start/stop
analyzer.start();
analyzer.stop();

// Get current state
const state = analyzer.getState();

// Connect/disconnect
analyzer.connect(destination);
analyzer.disconnect();

// Cleanup
analyzer.destroy();
```

#### Example

```javascript
const canvas = document.getElementById('spectrum-canvas');
const analyzer = new SpectrumAnalyzer(audioContext, canvas, {
  fftSize: 4096,
  smoothing: 80,
  scale: 'logarithmic',
  peakHold: true
});

// Connect audio
audioSource.connect(analyzer.input);
analyzer.connect(audioContext.destination);

// Start visualization
analyzer.start();
```

---

### 3. Tuner

Pitch detection and tuning reference for musical instruments.

#### Features

- Accurate pitch detection (autocorrelation algorithm)
- Chromatic tuning (all 12 notes)
- Adjustable reference frequency (A4: 400-480 Hz)
- Cents deviation display (±50 cents)
- Real-time frequency display
- Visual tuning indicator
- Confidence rating

#### API

```javascript
const tuner = new Tuner(audioContext, displayElement, options);

// Reference frequency
tuner.setReferenceFreq(442); // 400-480 Hz (A4)

// Start/stop
tuner.start();
tuner.stop();

// Get current note
const note = tuner.getCurrentNote();
// Returns: { note, octave, cents, frequency, confidence }

// Get current state
const state = tuner.getState();

// Connect/disconnect
tuner.connect(destination);
tuner.disconnect();

// Cleanup
tuner.destroy();
```

#### Example

```javascript
const display = document.getElementById('tuner-display');
const tuner = new Tuner(audioContext, display, {
  referenceFreq: 440,
  minVolume: 0.01
});

// Connect microphone
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(tuner.input);
    tuner.connect(audioContext.destination);
    tuner.start();
  });

// Get current note information
const noteInfo = tuner.getCurrentNote();
console.log(`${noteInfo.note}${noteInfo.octave} ${noteInfo.cents}¢`);
```

---

### 4. Channel EQ

Simple low-cut and high-cut filters for mixing and mastering.

#### Features

- Low-cut (highpass): 20-500 Hz
- High-cut (lowpass): 2k-20k Hz
- Multiple slopes: 6, 12, 18, 24, 36, 48 dB/octave
- Butterworth filter response
- Frequency response curve export

#### API

```javascript
const channelEQ = new ChannelEQ(audioContext, options);

// Low cut (highpass)
channelEQ.setLowCut(frequency, slope);
// frequency: 20-500 Hz (0 = off)
// slope: 6, 12, 18, 24, 36, 48 dB/oct

// High cut (lowpass)
channelEQ.setHighCut(frequency, slope);
// frequency: 2000-20000 Hz (0 = off)
// slope: 6, 12, 18, 24, 36, 48 dB/oct

// Enable/disable
channelEQ.setLowCutEnabled(true);
channelEQ.setHighCutEnabled(true);

// Get frequency response
const gain = channelEQ.getFrequencyResponse(1000); // dB at 1kHz

// Get frequency response curve
const curve = channelEQ.getFrequencyResponseCurve(100);
// Returns: [{ frequency, gain }, ...]

// Get current state
const state = channelEQ.getState();

// Connect/disconnect
channelEQ.connect(destination);
channelEQ.disconnect();

// Cleanup
channelEQ.destroy();
```

#### Example

```javascript
const channelEQ = new ChannelEQ(audioContext, {
  lowCutFreq: 80,
  lowCutSlope: 12,
  highCutFreq: 10000,
  highCutSlope: 24
});

// Remove rumble
channelEQ.setLowCut(80, 12);

// Remove harshness
channelEQ.setHighCut(12000, 12);

// Get frequency response at 100 Hz
const response = channelEQ.getFrequencyResponse(100);
console.log(`100 Hz: ${response.toFixed(2)} dB`);
```

## 🎯 Use Cases

### Mixing & Mastering

```javascript
// Gain staging
utility.setGain(-3);

// Remove rumble
channelEQ.setLowCut(40, 24);

// Analyze frequency balance
analyzer.setPeakHold(true);
analyzer.start();
```

### Instrument Tuning

```javascript
// Setup tuner for guitar
const tuner = new Tuner(audioContext, display, {
  referenceFreq: 440,
  mode: 'chromatic'
});

// Connect microphone
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(tuner.input);
    tuner.start();
  });
```

### Stereo Imaging

```javascript
// Widen stereo field
utility.setWidth(150);

// Mono bass frequencies
utility.setBassMono(120);

// Visualize stereo image
analyzer.setChannels('Side');
```

### Problem Solving

```javascript
// Fix DC offset
utility.setDCFilter(true);

// Fix phase issues
utility.setPhase('L', true);

// Remove low-end rumble
channelEQ.setLowCut(40, 48);
```

## 📊 Examples

See the `examples/` directory for complete working examples:

- `utility-tools-example.html` - Comprehensive demo of all plugins

## 🔧 Technical Details

### Audio Processing

All plugins use the Web Audio API for real-time audio processing:

- **Utility**: GainNode, ChannelSplitterNode, ChannelMergerNode, BiquadFilterNode
- **Spectrum Analyzer**: AnalyserNode with FFT processing
- **Tuner**: AnalyserNode with autocorrelation pitch detection
- **Channel EQ**: Cascaded BiquadFilterNode for multi-slope filters

### Performance

- Efficient processing with minimal CPU usage
- Optimized FFT visualization (60 FPS)
- Real-time pitch detection with low latency
- Smart filter cascading for high-order slopes

### Compatibility

- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support (iOS 14.5+)
- Opera: Full support

## 📝 Best Practices

### Gain Staging

```javascript
// Always monitor levels
utility.setGain(-6); // Leave headroom

// Use spectrum analyzer to check balance
analyzer.start();
```

### Filter Usage

```javascript
// Use gentle slopes for transparent filtering
channelEQ.setLowCut(40, 12);

// Use steep slopes for problem frequencies
channelEQ.setLowCut(100, 48);
```

### Tuning

```javascript
// Ensure quiet environment
tuner.state.minVolume = 0.02;

// Use appropriate reference frequency
tuner.setReferenceFreq(442); // Orchestra tuning
```

## 🐛 Troubleshooting

### No audio output

- Check audio context is running: `audioContext.resume()`
- Verify connections: source → plugin → destination
- Check gain levels: `utility.setGain(0)`

### Spectrum analyzer not updating

- Call `analyzer.start()`
- Check canvas element exists
- Verify audio is connected to analyzer input

### Tuner not detecting pitch

- Increase input volume
- Lower `minVolume` threshold
- Check microphone permissions
- Ensure single note (not chords)

### Filters not working

- Verify frequency values are in range
- Check if filter is enabled
- Ensure proper slope value

## 🔬 Advanced Usage

### Custom Mid/Side Processing

```javascript
// Split into mid/side
analyzer.setChannels('Mid');
// Analyze mid channel

analyzer.setChannels('Side');
// Analyze side channel

// Adjust width
utility.setWidth(150); // Enhance side
```

### Cascaded Processing

```javascript
// Chain multiple utilities
const utility1 = new Utility(audioContext);
const utility2 = new Utility(audioContext);

source.connect(utility1.input);
utility1.connect(utility2.input);
utility2.connect(destination);
```

## 📚 References

- [Web Audio API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [AnalyserNode Documentation](https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode)
- [BiquadFilterNode Documentation](https://developer.mozilla.org/en-US/docs/Web/API/BiquadFilterNode)
- [YIN Pitch Detection Algorithm](http://audition.ens.fr/adc/pdf/2002_JASA_YIN.pdf)

## 📄 License

MIT License - Free for personal and commercial use

## 🙏 Credits

**Developer**: Agent 9 - Utility & Analysis Tools
**Part of**: Dø (Doseedo) AI Music Platform
**Built with**: Web Audio API

---

**Make Professional Sounding Music** 🎵
