# Modulation Effects Library

A comprehensive collection of Web Audio API-based modulation effects including Chorus, Flanger, Phaser, and Tremolo/Auto Pan. These effects add movement, depth, and animation to audio through time-varying parameter modulation.

## Overview

This library implements four professional-grade modulation effects:

1. **Chorus** - Creates rich, ensemble-like sounds by layering multiple delayed and detuned copies
2. **Flanger** - Produces the classic "jet plane" whoosh effect using short delays with feedback
3. **Phaser** - Creates sweeping notches using cascaded all-pass filters
4. **Tremolo/Auto Pan** - Modulates amplitude or stereo position for rhythmic movement

All effects are built using the Web Audio API and can be used in any modern browser that supports it.

## Installation

Simply include the effect files in your HTML:

```html
<script src="modulation/Chorus.js"></script>
<script src="modulation/Flanger.js"></script>
<script src="modulation/Phaser.js"></script>
<script src="modulation/Tremolo.js"></script>
```

Or import them as modules if using a bundler:

```javascript
import Chorus from './modulation/Chorus.js';
import Flanger from './modulation/Flanger.js';
import Phaser from './modulation/Phaser.js';
import Tremolo from './modulation/Tremolo.js';
```

## Quick Start

```javascript
// Create audio context
const audioContext = new AudioContext();

// Create an audio source (e.g., oscillator)
const oscillator = audioContext.createOscillator();
oscillator.frequency.value = 440; // A4

// Create a chorus effect
const chorus = new Chorus(audioContext, {
  rate: 0.5,
  depth: 50,
  voices: 4,
  mix: 50
});

// Connect: Source -> Chorus -> Destination
oscillator.connect(chorus.input);
chorus.connect(audioContext.destination);

// Start the oscillator
oscillator.start();
```

## Chorus

Creates the illusion of multiple voices/instruments by layering slightly detuned delays.

### Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `rate` | 0.01 - 10 Hz | 0.5 | LFO speed (modulation rate) |
| `depth` | 0 - 100% | 50 | Modulation intensity |
| `voices` | 1 - 8 | 4 | Number of delay lines |
| `spread` | 0 - 100% | 50 | Stereo width |
| `feedback` | 0 - 100% | 0 | Feedback amount for richer sound |
| `mix` | 0 - 100% | 50 | Dry/wet mix |
| `delay` | 5 - 50 ms | 20 | Base delay time |

### Usage Example

```javascript
const chorus = new Chorus(audioContext);

// Set parameters
chorus.setRate(0.8);        // Slower modulation
chorus.setDepth(70);        // More intense modulation
chorus.setVoices(6);        // More voices for richer sound
chorus.setSpread(80);       // Wider stereo image
chorus.setMix(40);          // 40% wet, 60% dry

// Connect to audio chain
source.connect(chorus.input);
chorus.connect(destination);

// Get current parameters
const params = chorus.getParams();
console.log(params);

// Cleanup when done
chorus.dispose();
```

### Key Features

- Multiple delay lines (1-8 voices) with independent LFO phases
- Stereo spread for wide soundstage
- Feedback control for enhanced depth
- Smooth, artifact-free modulation
- Equal-power dry/wet mixing

## Flanger

Jet-plane whoosh effect created by short delay with feedback and modulation.

### Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `rate` | 0.01 - 10 Hz | 0.5 | LFO speed |
| `depth` | 0 - 100% | 50 | Modulation intensity |
| `feedback` | -100 - 100% | 50 | Feedback amount (can be negative) |
| `delay` | 0.5 - 10 ms | 3 | Base delay time |
| `manual` | 0 - 100% | 50 | Static delay offset |
| `waveform` | sine, triangle, square, sawtooth | sine | LFO waveform |
| `mix` | 0 - 100% | 50 | Dry/wet mix |

### Usage Example

```javascript
const flanger = new Flanger(audioContext, {
  rate: 0.3,
  depth: 80,
  feedback: 60,
  waveform: 'triangle'
});

// Negative feedback for different character
flanger.setFeedback(-70);

// Manual control for static flange
flanger.setManual(75);

// Change LFO waveform
flanger.setWaveform('sine');

source.connect(flanger.input);
flanger.connect(destination);
```

### Key Features

- Very short delay times (0.5-10ms) for classic flanging
- High feedback capability for resonance
- Negative feedback option for different tonal character
- Manual control for static flange position
- Multiple LFO waveforms

## Phaser

Sweeping notches created by all-pass filters with LFO modulation.

### Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `rate` | 0.01 - 10 Hz | 0.5 | LFO speed |
| `depth` | 0 - 100% | 50 | Modulation intensity |
| `feedback` | 0 - 100% | 0 | Feedback for resonance |
| `stages` | 4, 6, 8, 12 | 6 | Number of all-pass filters |
| `frequency` | 200 - 8000 Hz | 1000 | Center frequency |
| `spread` | 0 - 100% | 50 | Spacing between notches |
| `waveform` | sine, triangle, square, sawtooth | sine | LFO waveform |
| `mix` | 0 - 100% | 50 | Dry/wet mix |

### Usage Example

```javascript
const phaser = new Phaser(audioContext);

// Configure phaser
phaser.setStages(8);           // 8-stage phaser (more pronounced)
phaser.setFrequency(1500);     // Higher center frequency
phaser.setSpread(70);          // Wide notch spacing
phaser.setFeedback(40);        // Add resonance
phaser.setWaveform('triangle'); // Triangle wave modulation

source.connect(phaser.input);
phaser.connect(destination);
```

### Key Features

- Configurable all-pass filter stages (4, 6, 8, or 12)
- Frequency spread control for notch spacing
- Feedback for enhanced resonance
- Multiple LFO waveforms
- Wide frequency range (200-8000 Hz)

## Tremolo / Auto Pan

Amplitude or pan modulation for rhythmic movement.

### Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `rate` | 0.01 - 40 Hz | 5 | LFO speed (can sync to BPM) |
| `depth` | 0 - 100% | 50 | Modulation intensity |
| `waveform` | sine, triangle, square, sawtooth | sine | LFO waveform |
| `phase` | 0 - 360° | 0 | Phase offset |
| `mode` | tremolo, pan | tremolo | Amplitude or pan modulation |
| `stereo` | boolean | false | Stereo phase offset for tremolo |

### Usage Example

```javascript
const tremolo = new Tremolo(audioContext, {
  mode: 'tremolo',
  rate: 6,
  depth: 70
});

// Switch to auto pan mode
tremolo.setMode('pan');
tremolo.setRate(2);
tremolo.setDepth(100);

// Enable stereo tremolo (phase offset L/R)
tremolo.setMode('tremolo');
tremolo.setStereo(true);

// Change waveform for different rhythmic feel
tremolo.setWaveform('square'); // Choppy, gated effect

source.connect(tremolo.input);
tremolo.connect(destination);
```

### Key Features

- Two modes: Tremolo (amplitude) and Auto Pan (stereo position)
- Multiple LFO waveforms for different rhythmic feels
- Stereo tremolo with phase offset between channels
- Fast rates up to 40 Hz for interesting effects
- Smooth, click-free modulation

## Common Methods

All effects share these common methods:

### `connect(destination)`
Connect the effect output to a destination node.

```javascript
effect.connect(audioContext.destination);
```

### `disconnect()`
Disconnect the effect output.

```javascript
effect.disconnect();
```

### `getParams()`
Get current parameter values.

```javascript
const params = effect.getParams();
console.log(params);
```

### `dispose()`
Clean up and release resources. Call this when you're done using the effect.

```javascript
effect.dispose();
```

## Effect Chaining

You can chain multiple effects together:

```javascript
const audioContext = new AudioContext();
const oscillator = audioContext.createOscillator();

const chorus = new Chorus(audioContext);
const flanger = new Flanger(audioContext);
const phaser = new Phaser(audioContext);

// Chain: Oscillator -> Chorus -> Flanger -> Phaser -> Output
oscillator.connect(chorus.input);
chorus.output.connect(flanger.input);
flanger.output.connect(phaser.input);
phaser.output.connect(audioContext.destination);

oscillator.start();
```

## Demo

See `examples/modulation-showcase-example.html` for a complete interactive demo of all four effects with real-time parameter control and visualization.

To run the demo:

1. Open `examples/modulation-showcase-example.html` in a modern web browser
2. Click "Start Audio" to begin
3. Toggle effects on/off and adjust parameters in real-time
4. Try different audio sources (oscillator, noise, microphone)

## Technical Details

### LFO Implementation

All modulation effects use the Web Audio API's OscillatorNode for smooth, efficient LFO generation. The Chorus effect uses custom PeriodicWave to create phase-offset LFOs for each voice.

### Delay-Based Effects

Chorus and Flanger use DelayNode with modulated delay times. The delay time is controlled by connecting an LFO (OscillatorNode) to the delayTime AudioParam.

### All-Pass Filters

The Phaser uses cascaded BiquadFilterNode instances configured as all-pass filters. These filters maintain constant magnitude response while varying phase, creating the characteristic notches when mixed with the dry signal.

### Amplitude Modulation

Tremolo uses GainNode modulation for amplitude changes. The LFO is offset to keep the signal always positive, preventing polarity inversion.

### Stereo Processing

Auto Pan uses StereoPannerNode, while stereo Tremolo uses ChannelSplitter/Merger with independent LFOs for left and right channels.

## Browser Compatibility

These effects require the Web Audio API and are compatible with:

- Chrome/Edge 35+
- Firefox 25+
- Safari 14.1+
- Opera 22+

## Performance Considerations

- **Chorus**: Multiple delay lines and LFOs. CPU usage scales with number of voices (1-8).
- **Flanger**: Lightweight single delay with feedback. Minimal CPU usage.
- **Phaser**: Multiple all-pass filters. CPU usage scales with stages (4-12).
- **Tremolo**: Very lightweight. Minimal CPU usage.

For best performance on mobile devices, limit the number of simultaneous effects and reduce complexity (fewer chorus voices, fewer phaser stages).

## Tips for Best Results

### Chorus
- Use 4-6 voices for natural chorus on vocals/guitars
- Increase feedback for richer, shimmer-like textures
- Adjust spread for wider or narrower stereo image
- Lower mix (20-40%) for subtle doubling effect

### Flanger
- Slow rate (0.1-0.5 Hz) for classic tape flanging
- Fast rate (2-5 Hz) for metallic, robotic sounds
- Negative feedback creates different harmonic character
- Try square/triangle waves for rhythmic swooshes

### Phaser
- 4-6 stages for subtle, musical phasing
- 8-12 stages for dramatic, pronounced effect
- Feedback adds resonance and "vowel-like" quality
- Experiment with different center frequencies on different sources

### Tremolo
- Sine wave for smooth, vintage tremolo
- Square wave for choppy, gated effects
- Triangle for linear amplitude sweep
- Auto Pan mode great for width and movement on synths

## Limitations

- Tempo sync is not yet implemented (placeholders exist in code)
- Random LFO waveform not yet implemented for Tremolo
- Phase offset requires custom waveform implementation
- Through-zero flanging is difficult with Web Audio API

## License

MIT License - See LICENSE file for details

## Credits

Developed as part of Agent 4: Modulation Effects for the Do music generation system.

References:
- Web Audio API Specification
- "Digital Audio Effects" by Udo Zölzer
- Tone.js effect implementations
- Ableton Live effect documentation

## Support

For issues, questions, or contributions, please refer to the main project repository.
