# Spectral Processing Plugins

Advanced FFT-based audio effects for Web Audio API, providing professional-grade spectral manipulation tools.

## Overview

This library implements **4 advanced spectral processing plugins**:

1. **Spectral Time** - Phase vocoder for time stretching, pitch shifting, and spectral freezing
2. **Spectral Resonator** - Resonant comb filtering with harmonic control
3. **Frequency Shifter** - Linear frequency shifting using single-sideband modulation
4. **Vocoder** - Multi-band vocoder for spectral envelope transfer

All plugins are built using the Web Audio API and AudioWorklet for real-time, low-latency processing.

## Features

- ✅ Real-time FFT processing using AudioWorklet
- ✅ Phase vocoder algorithm for independent time/pitch control
- ✅ Harmonic resonance with configurable decay
- ✅ Single-sideband modulation for frequency shifting
- ✅ Multi-band vocoding with internal/external carrier
- ✅ Low latency and efficient CPU usage
- ✅ Modular architecture - use plugins independently or in chains
- ✅ Comprehensive parameter control

## Installation

### Browser (CDN)

```html
<!-- Load individual plugins -->
<script src="path/to/spectral/SpectralTime.js"></script>
<script src="path/to/spectral/SpectralResonator.js"></script>
<script src="path/to/spectral/FrequencyShifter.js"></script>
<script src="path/to/spectral/Vocoder.js"></script>
```

### Module Import

```javascript
import SpectralTime from './spectral/SpectralTime.js';
import SpectralResonator from './spectral/SpectralResonator.js';
import FrequencyShifter from './spectral/FrequencyShifter.js';
import Vocoder from './spectral/Vocoder.js';
```

## Quick Start

```javascript
// Create audio context
const audioContext = new AudioContext();

// Create a plugin
const spectralTime = new SpectralTime(audioContext);

// Wait for worklet to be ready (for worklet-based plugins)
await spectralTime.ready();

// Connect audio source
audioSource.connect(spectralTime.input);
spectralTime.connect(audioContext.destination);

// Set parameters
spectralTime.setStretch(2.0); // Slow down to half speed
spectralTime.setFreeze(true); // Freeze spectrum
spectralTime.setPitchShift(7); // Shift up 7 semitones
```

## Plugin Documentation

### 1. Spectral Time

Phase vocoder implementation for time stretching, pitch shifting, and spectral effects.

#### Features
- Independent time and pitch control
- Spectral freezing for ambient textures
- Spectral blurring/smearing
- Formant preservation
- Transient separation (residual)

#### Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `stretch` | 0.1 - 4.0 | 1.0 | Time stretch factor (1.0 = normal) |
| `freeze` | boolean | false | Freeze spectrum at current state |
| `blur` | 0 - 100% | 0% | Spectral blurring amount |
| `shift` | -24 to +24 | 0 | Pitch shift in semitones |
| `formant` | -4 to +4 | 0 | Formant shift in semitones |
| `residual` | 0 - 100% | 0% | Transient preservation |
| `mix` | 0 - 100% | 100% | Wet/dry mix |

#### Usage Example

```javascript
const spectralTime = new SpectralTime(audioContext, {
  stretch: 1.5,    // 1.5x slower
  blur: 20,        // 20% blur
  shift: 0,        // No pitch shift
  mix: 100         // 100% wet
});

await spectralTime.ready();

// Connect
source.connect(spectralTime.input);
spectralTime.connect(destination);

// Control in real-time
spectralTime.setStretch(0.5);  // 2x faster
spectralTime.setFreeze(true);   // Freeze
spectralTime.setBlur(50);       // 50% blur
spectralTime.setPitchShift(12); // +1 octave
```

#### Use Cases
- Time stretching without pitch change
- Ambient texture creation (freeze + blur)
- Pitch correction
- Special effects (extreme stretch, freeze)
- Formant shifting for vocal processing

---

### 2. Spectral Resonator

Resonant comb filtering based on harmonic series.

#### Features
- Pitched resonance using comb filters
- Configurable harmonic series
- Variable decay time
- Harmonic stretch and color control
- Envelope following capability

#### Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `mode` | string | 'pitched' | Resonance mode (pitched/noise/hybrid) |
| `pitch` | 0-127 or Hz | 440 Hz | Resonant frequency |
| `decay` | 0.1 - 10 s | 1.0 s | Resonance decay time |
| `color` | 0 - 100% | 50% | Harmonic emphasis |
| `harmonics` | 1 - 32 | 8 | Number of harmonics |
| `stretch` | 0.5 - 2.0 | 1.0 | Harmonic spacing |
| `detune` | -100 to +100 | 0 | Detune in cents |
| `attack` | 0 - 500 ms | 10 ms | Envelope attack time |
| `release` | 10 - 5000 ms | 100 ms | Envelope release time |
| `mix` | 0 - 100% | 50% | Wet/dry mix |

#### Usage Example

```javascript
const resonator = new SpectralResonator(audioContext, {
  pitch: 60,        // Middle C (MIDI)
  harmonics: 16,    // 16 harmonics
  decay: 2.0,       // 2 second decay
  color: 70,        // 70% color
  mix: 50          // 50% mix
});

// Connect
percussionTrack.connect(resonator.input);
resonator.connect(destination);

// Control
resonator.setPitch(440);       // A4 (440 Hz)
resonator.setHarmonics(32);    // More harmonics
resonator.setDecay(5.0);       // Long decay
resonator.setStretch(1.2);     // Stretched harmonics
```

#### Use Cases
- Resonant percussion
- Tonal enhancement
- Pitched resonance effects
- String simulation
- Spectral reinforcement

---

### 3. Frequency Shifter

Linear frequency shifting using single-sideband modulation.

#### Features
- Linear (not logarithmic) frequency shift
- Creates inharmonic spectra
- Upper/lower sideband selection
- Stereo widening mode
- Ring modulation-like effects

#### Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `frequency` | -5000 to +5000 Hz | 0 Hz | Shift amount |
| `fine` | -100 to +100 Hz | 0 Hz | Fine tuning |
| `mode` | string | 'up' | Shift mode (up/down/wide) |
| `wideAmount` | 0 - 100% | 50% | Stereo spread (wide mode) |
| `drive` | 0 - 100% | 0% | Harmonic saturation |
| `mix` | 0 - 100% | 100% | Wet/dry mix |

#### Usage Example

```javascript
const shifter = new FrequencyShifter(audioContext, {
  frequency: 100,   // +100 Hz shift
  mode: 'up',       // Upper sideband
  drive: 20,        // 20% drive
  mix: 100          // 100% wet
});

await shifter.ready();

// Connect
source.connect(shifter.input);
shifter.connect(destination);

// Control
shifter.setFrequency(200);     // +200 Hz
shifter.setMode('wide');       // Stereo spread
shifter.setWideAmount(80);     // 80% spread
shifter.setDrive(50);          // Add harmonics
```

#### Use Cases
- Inharmonic effects
- Stereo widening
- Detuning/chorus effects
- Ring modulation-like tones
- Special FX (sci-fi sounds)

---

### 4. Vocoder

Multi-band vocoder for imposing one signal's spectral envelope on another.

#### Features
- Multi-band filtering (8 to 40 bands)
- Internal carrier synthesis (noise/saw/pulse)
- External carrier input support
- Formant shifting
- Configurable frequency range
- Envelope follower per band

#### Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `bands` | 8/16/32/40 | 16 | Number of frequency bands |
| `range` | string | 'full' | Frequency range preset |
| `loFreq` | 20 - 500 Hz | 80 Hz | Lowest frequency |
| `hiFreq` | 2k - 20k Hz | 12k Hz | Highest frequency |
| `attack` | 0.1 - 100 ms | 10 ms | Envelope attack |
| `release` | 10 - 500 ms | 100 ms | Envelope release |
| `formant` | -4 to +4 | 0 | Formant shift (semitones) |
| `resonance` | 0 - 100% | 50% | Formant emphasis |
| `carrierSource` | string | 'internal' | internal/external |
| `carrierType` | string | 'saw' | noise/saw/pulse |
| `upperBandLevel` | 0 - 100% | 100% | High-frequency emphasis |
| `mix` | 0 - 100% | 100% | Wet/dry mix |

#### Usage Example

```javascript
const vocoder = new Vocoder(audioContext, {
  bands: 32,          // 32 bands
  carrierType: 'saw', // Sawtooth carrier
  attack: 5,          // Fast attack
  release: 50,        // Medium release
  mix: 100            // 100% wet
});

// Connect modulator (e.g., voice)
voiceInput.connect(vocoder.input);

// Connect external carrier (optional)
synthInput.connect(vocoder.carrierInput);

// Connect output
vocoder.connect(destination);

// Control
vocoder.setBands(40);              // More bands
vocoder.setCarrierType('noise');   // Noise carrier
vocoder.setFormant(2);             // +2 semitones
vocoder.setResonance(70);          // Emphasize formants
vocoder.setCarrierFrequency(110);  // A2 carrier
```

#### Use Cases
- Robot voice effects
- Vocal synthesis
- Talkbox simulation
- Spectral transfer
- Creative sound design

---

## Advanced Usage

### Chaining Plugins

```javascript
// Chain multiple effects
source
  .connect(spectralTime.input);

spectralTime.output
  .connect(resonator.input);

resonator.output
  .connect(shifter.input);

shifter.output
  .connect(vocoder.input);

vocoder.output
  .connect(audioContext.destination);
```

### Dynamic Parameter Control

```javascript
// Automate parameters over time
let time = 0;
setInterval(() => {
  time += 0.1;

  // LFO on pitch shift
  const pitchShift = Math.sin(time) * 12;
  spectralTime.setPitchShift(pitchShift);

  // Modulate resonator frequency
  const freq = 200 + Math.sin(time * 0.5) * 100;
  resonator.setPitch(freq);
}, 50);
```

### Cleanup

```javascript
// Dispose of plugins when done
spectralTime.dispose();
resonator.dispose();
shifter.dispose();
vocoder.dispose();
```

## Technical Details

### FFT Processing

- **FFT Size**: 2048-4096 samples (configurable)
- **Window Function**: Hann window (default)
- **Overlap**: 75% (4x overlap-add)
- **Latency**: ~100-200ms (depending on FFT size)

### Phase Vocoder Algorithm

The Spectral Time plugin uses a classic phase vocoder:

1. **Analysis**: STFT with windowing
2. **Phase Unwrapping**: Calculate instantaneous frequency
3. **Time Stretching**: Modify synthesis hop size
4. **Pitch Shifting**: Resample spectrum
5. **Synthesis**: Inverse FFT and overlap-add

### Single-Sideband Modulation

The Frequency Shifter uses Hilbert transform for quadrature generation:

```
output = input * cos(ωt) ± quadrature * sin(ωt)
```

Where `±` determines upper/lower sideband.

### Vocoder Architecture

Each band consists of:

1. **Analysis Filter**: Bandpass filter on modulator
2. **Envelope Follower**: Rectifier + lowpass filter
3. **Synthesis Filter**: Bandpass filter on carrier
4. **VCA**: Gain controlled by envelope

## Browser Compatibility

- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support (Safari 14.1+)
- Mobile: ✅ iOS Safari 14.5+, Chrome Android

**Requirements**:
- Web Audio API
- AudioWorklet support (for Spectral Time and Frequency Shifter)

## Performance

### CPU Usage (approximate)

| Plugin | CPU Usage | Latency |
|--------|-----------|---------|
| Spectral Time | Medium-High | ~150ms |
| Spectral Resonator | Low | <10ms |
| Frequency Shifter | Medium | ~50ms |
| Vocoder (16 bands) | Low-Medium | <10ms |
| Vocoder (40 bands) | Medium | <10ms |

### Optimization Tips

1. Use smaller FFT sizes for lower latency (trade-off: frequency resolution)
2. Reduce number of vocoder bands if CPU constrained
3. Disable unused plugins in the chain
4. Use lower sample rates (22.05kHz) for less critical applications

## Examples

See `/examples/spectral-processing-example.html` for a comprehensive interactive demo.

### Minimal Example

```html
<!DOCTYPE html>
<html>
<head>
  <title>Spectral Time Demo</title>
</head>
<body>
  <button id="start">Start</button>
  <input type="range" id="stretch" min="0.1" max="4" step="0.1" value="1">

  <script src="SpectralTime.js"></script>
  <script>
    const ctx = new AudioContext();
    const spectralTime = new SpectralTime(ctx);

    document.getElementById('start').addEventListener('click', async () => {
      await spectralTime.ready();

      const osc = ctx.createOscillator();
      osc.connect(spectralTime.input);
      spectralTime.connect(ctx.destination);
      osc.start();
    });

    document.getElementById('stretch').addEventListener('input', (e) => {
      spectralTime.setStretch(parseFloat(e.target.value));
    });
  </script>
</body>
</html>
```

## API Reference

### Common Methods

All plugins share these methods:

```javascript
// Connection
plugin.connect(destination)
plugin.disconnect()

// Cleanup
plugin.dispose()
```

Worklet-based plugins also have:

```javascript
// Wait for ready
await plugin.ready()
```

### Spectral Time API

```javascript
setStretch(factor)           // 0.1 - 4.0
setFreeze(enabled)           // boolean
setBlur(percent)             // 0 - 100
setPitchShift(semitones)     // -24 to +24
setFormantShift(semitones)   // -4 to +4
setResidual(percent)         // 0 - 100
setMix(percent)              // 0 - 100
```

### Spectral Resonator API

```javascript
setMode(mode)                // 'pitched', 'noise', 'hybrid'
setPitch(pitch)              // MIDI note (0-127) or Hz
setDecay(seconds)            // 0.1 - 10
setColor(percent)            // 0 - 100
setHarmonics(num)            // 1 - 32
setStretch(factor)           // 0.5 - 2.0
setDetune(cents)             // -100 to +100
setAttack(ms)                // 0 - 500
setRelease(ms)               // 10 - 5000
setMix(percent)              // 0 - 100
```

### Frequency Shifter API

```javascript
setFrequency(hz)             // -5000 to +5000
setFine(hz)                  // -100 to +100
setMode(mode)                // 'up', 'down', 'wide'
setWideAmount(percent)       // 0 - 100
setDrive(percent)            // 0 - 100
setMix(percent)              // 0 - 100
```

### Vocoder API

```javascript
setBands(num)                // 8, 16, 32, 40
setRange(range)              // 'low', 'mid', 'high', 'full'
setLoFreq(hz)                // 20 - 500
setHiFreq(hz)                // 2000 - 20000
setAttack(ms)                // 0.1 - 100
setRelease(ms)               // 10 - 500
setFormant(semitones)        // -4 to +4
setResonance(percent)        // 0 - 100
setCarrierSource(source)     // 'internal', 'external'
setCarrierType(type)         // 'noise', 'saw', 'pulse'
setCarrierFrequency(hz)      // Any frequency
setUpperBandLevel(percent)   // 0 - 100
setMix(percent)              // 0 - 100
```

## Troubleshooting

### Worklet Loading Issues

If you see errors loading AudioWorklet modules:

```javascript
// Ensure correct path to worklet files
const basePath = './spectral';
```

### No Sound Output

1. Check that audio context is started (requires user gesture)
2. Verify plugin is enabled and connected
3. Check mix parameter (should be > 0 for wet signal)
4. Ensure input source is active

### High Latency

- Reduce FFT size in worklet processor
- Use native Web Audio nodes where possible (Resonator, Vocoder)
- Avoid chaining too many worklet-based plugins

### CPU Issues

- Reduce vocoder band count
- Lower sample rate
- Disable unused plugins
- Use smaller FFT sizes

## Credits

**Author**: Agent 8 - Spectral Processing Specialist
**Built with**: Web Audio API, AudioWorklet
**Inspired by**: Ableton Live, Native Instruments, iZotope

## License

MIT License - Free to use in commercial and personal projects.

## Further Reading

### Academic Papers
- "Phase Vocoder Tutorial" - Miller Puckette
- "DAFX: Digital Audio Effects" - Udo Zölzer (Chapter on Phase Vocoder)
- "Spectral Audio Signal Processing" - Julius O. Smith III

### Web Resources
- [Web Audio API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [AudioWorklet Guide](https://developers.google.com/web/updates/2017/12/audio-worklet)
- [Phase Vocoder Explained](https://www.dsprelated.com/)

---

**Built with ❤️ for creative audio processing**
