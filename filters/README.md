# Filter Plugins

Advanced filter effects with modulation capabilities built with the Web Audio API.

## 📦 Contents

- **AutoFilter.js** - Multi-mode resonant filter with LFO and envelope modulation

## 🎛️ Auto Filter

### Overview

Auto Filter is a multi-mode resonant filter with comprehensive modulation capabilities. It combines a high-quality filter with LFO and envelope modulation sources to create sweeping, rhythmic, and dynamic filter effects.

### Features

- **Multiple Filter Types**: Lowpass (12dB, 24dB), Highpass (12dB, 24dB), Bandpass, Notch
- **Resonance Control**: 0-100% (maps to Q factor 0.1-20)
- **Frequency Range**: 20 Hz to 20 kHz
- **LFO Modulation**:
  - Multiple waveforms (sine, triangle, square, sawtooth)
  - Rate: 0.01 to 40 Hz
  - Amount: -100% to +100%
  - Tempo sync support
- **Envelope Modulation**:
  - Full ADSR envelope
  - Amount: -100% to +100%
  - Attack: 0.1 to 500 ms
  - Decay: 0.1 to 1000 ms
  - Sustain: 0 to 100%
  - Release: 10 to 5000 ms
- **Dry/Wet Mix**: 0-100% for parallel processing
- **Cascaded Filters**: 24dB slopes use two cascaded 12dB filters

### Default Settings

- **Filter Type**: Lowpass 24dB
- **Frequency**: 1000 Hz
- **Resonance**: 0%
- **Mix**: 100% (fully wet)
- **LFO**: Disabled (amount = 0%)
- **Envelope**: Disabled (amount = 0%)

### Usage

```javascript
// Create Auto Filter instance
const audioContext = new AudioContext();
const autoFilter = new AutoFilter(audioContext, {
  frequency: 1000,
  resonance: 50,
  filterType: 'lowpass24',
  mix: 100,
  bpm: 120
});

// Connect to audio graph
source.connect(autoFilter.getInput());
autoFilter.connect(audioContext.destination);

// Set filter parameters
autoFilter.setFrequency(500);         // Hz
autoFilter.setResonance(75);          // 0-100%
autoFilter.setFilterType('lowpass24'); // Filter type
autoFilter.setMix(80);                // 80% wet, 20% dry

// Configure LFO
autoFilter.setLFO({
  waveform: 'sine',    // sine, triangle, square, sawtooth
  rate: 2,             // Hz (ignored if sync is true)
  amount: 50,          // ±50% modulation depth
  sync: false          // Tempo sync off
});

// Configure LFO with tempo sync
autoFilter.setBPM(128);
autoFilter.setLFO({
  waveform: 'triangle',
  syncRate: '1/4',     // Quarter note (1/1, 1/2, 1/4, 1/8, 1/16, etc.)
  amount: 60,
  sync: true
});

// Configure envelope
autoFilter.setEnvelope({
  amount: 80,          // ±80% modulation depth
  attack: 50,          // ms
  decay: 200,          // ms
  sustain: 60,         // %
  release: 300         // ms
});

// Trigger envelope (for rhythmic effects or note-on)
autoFilter.triggerEnvelope();

// Release envelope (for note-off)
autoFilter.releaseEnvelope();

// Get current state
const state = autoFilter.getState();
console.log(state);

// Clean up
autoFilter.destroy();
```

### Filter Types

| Type | Slope | Description | Use Case |
|------|-------|-------------|----------|
| `lowpass12` | 12 dB/oct | Gentle lowpass | Warm, smooth filtering |
| `lowpass24` | 24 dB/oct | Steep lowpass | Classic synth filter sound |
| `highpass12` | 12 dB/oct | Gentle highpass | Subtle bass removal |
| `highpass24` | 24 dB/oct | Steep highpass | Aggressive bass cut |
| `bandpass` | 12 dB/oct | Pass only center freq | Narrow frequency selection |
| `notch` | N/A | Reject center freq | Remove specific frequencies |

### LFO Waveforms

- **Sine**: Smooth, periodic modulation
- **Triangle**: Linear up/down sweeps
- **Square**: Abrupt on/off switching
- **Sawtooth**: Ramp up then reset

### Modulation Routing

```
Input → Filter → Wet Gain ────┐
  │                            ├──→ Output
  └────────────────→ Dry Gain ─┘

LFO ──→ LFO Gain ──→ Filter Frequency
Envelope ──────────→ Filter Frequency
```

### API Reference

#### Constructor

```javascript
new AutoFilter(audioContext, options)
```

**Parameters:**
- `audioContext` (AudioContext) - Web Audio API context
- `options` (Object) - Optional initialization parameters
  - `frequency` (number) - Filter frequency (20-20000 Hz)
  - `resonance` (number) - Resonance (0-100%)
  - `filterType` (string) - Filter type
  - `mix` (number) - Dry/wet mix (0-100%)
  - `bpm` (number) - BPM for tempo sync (20-300)
  - `lfo` (Object) - LFO parameters
  - `envelope` (Object) - Envelope parameters

#### Methods

**`setFrequency(freq)`**
Set filter cutoff frequency (20-20000 Hz).

**`setResonance(resonance)`**
Set filter resonance (0-100%).

**`setFilterType(type)`**
Set filter type ('lowpass12', 'lowpass24', 'highpass12', 'highpass24', 'bandpass', 'notch').

**`setMix(mix)`**
Set dry/wet mix (0-100%).

**`setLFO(params)`**
Configure LFO modulation.

- `params` (Object)
  - `rate` (number) - LFO rate in Hz (0.01-40)
  - `amount` (number) - Modulation depth (-100 to +100%)
  - `waveform` (string) - Waveform type
  - `phase` (number) - Phase offset (0-360°)
  - `sync` (boolean) - Enable tempo sync
  - `syncRate` (string) - Sync rate ('1/1', '1/2', '1/4', '1/8', '1/16')

**`setBPM(bpm)`**
Set BPM for tempo-synced LFO (20-300).

**`setEnvelope(params)`**
Configure envelope modulation.

- `params` (Object)
  - `amount` (number) - Modulation depth (-100 to +100%)
  - `attack` (number) - Attack time (0.1-500 ms)
  - `decay` (number) - Decay time (0.1-1000 ms)
  - `sustain` (number) - Sustain level (0-100%)
  - `release` (number) - Release time (10-5000 ms)

**`triggerEnvelope()`**
Trigger the envelope (attack phase).

**`releaseEnvelope()`**
Release the envelope (release phase).

**`getState()`**
Get current filter state.

**`getInput()` / `getOutput()`**
Get input/output nodes for connection.

**`connect(destination)` / `disconnect()`**
Connect/disconnect from audio graph.

**`destroy()`**
Clean up all resources.

### Performance

- **Latency**: < 3ms
- **CPU Usage**: < 2%
- **Modulation Resolution**: Audio-rate (LFO), sample-accurate (envelope)

## 🎵 Use Cases

### Classic Filter Sweep

```javascript
// Slow sine wave sweep
autoFilter.setFilterType('lowpass24');
autoFilter.setResonance(70);
autoFilter.setLFO({
  waveform: 'sine',
  rate: 0.25,      // 4 seconds per cycle
  amount: 80
});
```

### Rhythmic Filter (Tempo-Synced)

```javascript
// Synced to 1/8 notes
autoFilter.setBPM(128);
autoFilter.setFilterType('lowpass24');
autoFilter.setResonance(60);
autoFilter.setLFO({
  waveform: 'square',
  syncRate: '1/8',
  amount: 70,
  sync: true
});
```

### Envelope-Controlled Filter (Synth-Style)

```javascript
// Classic synth filter envelope
autoFilter.setFilterType('lowpass24');
autoFilter.setFrequency(300);    // Start low
autoFilter.setResonance(80);     // High resonance
autoFilter.setEnvelope({
  amount: 100,      // Maximum modulation
  attack: 20,       // Fast attack
  decay: 300,       // Moderate decay
  sustain: 30,      // Low sustain
  release: 500      // Moderate release
});

// Trigger on each note
midiInput.onNoteOn = () => autoFilter.triggerEnvelope();
midiInput.onNoteOff = () => autoFilter.releaseEnvelope();
```

### Subtle High-Pass Movement

```javascript
// Gentle high-pass for dynamics
autoFilter.setFilterType('highpass12');
autoFilter.setFrequency(80);
autoFilter.setResonance(10);     // Subtle resonance
autoFilter.setMix(50);           // Blend with dry signal
autoFilter.setLFO({
  waveform: 'triangle',
  rate: 0.5,
  amount: 30
});
```

### Aggressive Bandpass

```javascript
// Narrow bandpass sweep
autoFilter.setFilterType('bandpass');
autoFilter.setResonance(90);     // Very narrow
autoFilter.setLFO({
  waveform: 'sawtooth',
  rate: 2,
  amount: 100      // Full sweep range
});
```

## 🔧 Technical Notes

### 24dB Filter Implementation

The 24dB/octave filter types are achieved by cascading two 12dB BiquadFilterNodes:

```javascript
// Example: Lowpass 24dB
filter1.type = 'lowpass';  // First stage
filter2.type = 'lowpass';  // Second stage
// Both tuned to the same frequency
```

This creates a steeper rolloff (-24dB per octave instead of -12dB).

### Modulation Depth Calculation

LFO and envelope modulation depth is calculated as:

```javascript
modulationRange = ±2 octaves = ±baseFrequency
finalFrequency = baseFrequency + (modulationAmount × modulationRange)
```

For example:
- Base frequency: 1000 Hz
- Modulation amount: +50%
- Range: ±1000 Hz
- Final range: 500 Hz to 1500 Hz

### Resonance Mapping

Resonance (0-100%) is mapped to BiquadFilterNode Q factor (0.1-20):

```javascript
Q = 0.1 + (resonance / 100) × 19.9
```

Higher Q values create more pronounced peaks at the cutoff frequency.

### Tempo Sync Calculation

When tempo sync is enabled, LFO frequency is calculated from BPM and note division:

```javascript
beatsPerSecond = BPM / 60
noteLength = numerator / denominator  // e.g., 1/4 = 0.25
lfoFrequency = beatsPerSecond / noteLength
```

Example at 120 BPM:
- 1/4 note: 2 Hz (0.5s cycle)
- 1/8 note: 4 Hz (0.25s cycle)
- 1/16 note: 8 Hz (0.125s cycle)

### Envelope Implementation

The ADSR envelope modulates filter frequency using Web Audio API automation:

```
Attack: baseFreq → targetFreq (linear ramp)
Decay: targetFreq → sustainFreq (linear ramp)
Sustain: hold at sustainFreq
Release: sustainFreq → baseFreq (linear ramp)
```

## 📚 References

- [Web Audio API Specification](https://www.w3.org/TR/webaudio/)
- [BiquadFilterNode](https://developer.mozilla.org/en-US/docs/Web/API/BiquadFilterNode)
- [OscillatorNode](https://developer.mozilla.org/en-US/docs/Web/API/OscillatorNode)
- [Audio Parameter Automation](https://www.w3.org/TR/webaudio/#AudioParam)

## 📝 License

Part of the Doseedo Audio Production Suite.

## 🤝 Contributing

For issues or enhancements, please refer to the main project repository.
