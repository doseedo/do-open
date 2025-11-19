# Agent 2: EQ & Filters

## Your Mission
You are responsible for implementing **3 essential EQ and filter plugins**: EQ Eight (parametric), EQ Three (DJ-style), and Auto Filter (modulated filter). These are fundamental tools for frequency shaping in any mix.

## Plugins to Implement

### 1. EQ Eight
**Purpose**: Professional 8-band parametric equalizer for surgical frequency adjustments

**Parameters** (per band):
- `frequency` (20 Hz to 20 kHz, logarithmic)
- `gain` (-15 to +15 dB)
- `Q` (0.1 to 10, default: 0.71)
- `filterType` (bell, lowshelf, highshelf, lowpass, highpass, notch, bandpass)
- `enabled` (boolean per band)

**Global Parameters**:
- `globalGain` (-15 to +15 dB)
- `adaptive` (boolean) - adaptive Q based on gain
- `editMode` (single, left, right, stereo) - stereo independence

**Key Features**:
- 8 independent bands
- Multiple filter types per band
- Visual frequency response curve (optional)
- Stereo or mid/side processing
- Adaptive Q (higher gain = narrower Q)

### 2. EQ Three
**Purpose**: DJ-style 3-band EQ with kill switches

**Parameters**:
- `lowGain` (0 to 2, default: 1) - can kill to 0
- `midGain` (0 to 2, default: 1)
- `highGain` (0 to 2, default: 1)
- `lowFreq` (20 to 500 Hz, default: 250) - crossover frequency
- `highFreq` (2k to 10k Hz, default: 3500) - crossover frequency

**Key Features**:
- Kill switches (0 = complete removal)
- Smooth crossover filters (Linkwitz-Riley)
- Fast parameter changes for DJ performance
- No phase issues at crossover points

### 3. Auto Filter
**Purpose**: Multi-mode resonant filter with modulation sources

**Parameters**:
- `frequency` (20 Hz to 20 kHz, default: 1000)
- `resonance` (0 to 100%, default: 0)
- `filterType` (lowpass12, lowpass24, highpass12, highpass24, bandpass, notch)
- `envelope`:
  - `amount` (-100 to +100%, default: 0)
  - `attack` (0.1 to 500 ms)
  - `decay` (0.1 to 1000 ms)
  - `sustain` (0 to 100%)
  - `release` (10 to 5000 ms)
- `lfo`:
  - `amount` (-100 to +100%, default: 0)
  - `rate` (0.01 to 40 Hz or sync to BPM)
  - `waveform` (sine, triangle, square, sawtooth, random)
  - `phase` (0 to 360°)
- `mix` (0 to 100%, default: 100)

**Key Features**:
- Multiple filter slopes (12dB/oct, 24dB/oct)
- Envelope follower for frequency modulation
- LFO with multiple waveforms
- Tempo-synced LFO
- Sidechain envelope follower option

## Research Phase (Week 1)

### Essential Research Topics

1. **BiquadFilterNode Mastery**
   - All filter types (lowpass, highpass, bandpass, notch, allpass, peaking, lowshelf, highshelf)
   - Frequency response calculation
   - Q factor vs bandwidth relationship
   - Gain parameter behavior

2. **Filter Topologies**
   - Butterworth (maximally flat)
   - Linkwitz-Riley (perfect crossover)
   - Chebyshev (steeper rolloff)
   - State variable filters

3. **Parametric EQ Design**
   - Bell filter mathematics
   - Shelf filter design
   - Filter coefficient calculation
   - Gain scaling and headroom

4. **Crossover Filters**
   - Linkwitz-Riley 4th order
   - Phase alignment at crossover
   - Summing multiple bands

5. **Modulation Sources**
   - LFO implementation (OscillatorNode)
   - Envelope follower (signal analysis)
   - ADSR envelope generation
   - Modulation depth scaling

### Reference Materials
- **Ableton Manual**: EQ Eight, EQ Three, Auto Filter sections
- **Audio EQ Cookbook**: Robert Bristow-Johnson's formulas
- **Web Audio API**: BiquadFilterNode documentation
- **DSP Books**: "The Art of VA Filter Design" by Vadim Zavalishin

### Code to Study
```javascript
// Study filter coefficient calculation:
// Audio EQ Cookbook formulas
const omega = 2 * Math.PI * frequency / sampleRate;
const alpha = Math.sin(omega) / (2 * Q);

// Peaking EQ coefficients
const A = Math.pow(10, gain / 40);
const b0 = 1 + alpha * A;
const b1 = -2 * Math.cos(omega);
const b2 = 1 - alpha * A;
const a0 = 1 + alpha / A;
const a1 = -2 * Math.cos(omega);
const a2 = 1 - alpha / A;
```

## Implementation Phase (Week 2-3)

### Architecture Pattern - EQ Eight

```javascript
class EQEight {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // 8 bands of filtering
    this.bands = [];
    for (let i = 0; i < 8; i++) {
      this.bands.push({
        filter: audioContext.createBiquadFilter(),
        bypass: audioContext.createGain(),
        enabled: true
      });
    }

    // Default frequencies for 8 bands
    this.defaultFrequencies = [
      30, 90, 250, 700, 2000, 5000, 10000, 16000
    ];

    // Chain all bands
    this.setupFilterChain();

    this.initialize(options);
  }

  setupFilterChain() {
    let current = this.input;
    for (let i = 0; i < 8; i++) {
      current.connect(this.bands[i].filter);
      this.bands[i].filter.connect(this.bands[i].bypass);
      current = this.bands[i].bypass;
    }
    current.connect(this.output);
  }

  setBand(index, params) {
    const band = this.bands[index];

    if (params.frequency) {
      band.filter.frequency.value = params.frequency;
    }

    if (params.gain !== undefined) {
      band.filter.gain.value = params.gain;
    }

    if (params.Q) {
      band.filter.Q.value = params.Q;
    }

    if (params.filterType) {
      // Map to BiquadFilterNode types
      band.filter.type = this.mapFilterType(params.filterType);
    }

    if (params.enabled !== undefined) {
      this.enableBand(index, params.enabled);
    }
  }

  mapFilterType(type) {
    const typeMap = {
      'bell': 'peaking',
      'lowshelf': 'lowshelf',
      'highshelf': 'highshelf',
      'lowpass': 'lowpass',
      'highpass': 'highpass',
      'notch': 'notch',
      'bandpass': 'bandpass'
    };
    return typeMap[type] || 'peaking';
  }

  enableBand(index, enabled) {
    const band = this.bands[index];
    band.enabled = enabled;

    // Bypass by setting gain to 0 or filter to 0dB gain
    if (enabled) {
      band.bypass.gain.value = 1;
    } else {
      // Set filter to unity gain instead of bypassing
      band.filter.gain.value = 0;
    }
  }

  getFrequencyResponse(frequencies) {
    // Calculate combined frequency response
    // Useful for visualization
  }
}
```

### Architecture Pattern - EQ Three

```javascript
class EQThree {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Split into 3 bands
    this.lowBand = this.createBand(audioContext);
    this.midBand = this.createBand(audioContext);
    this.highBand = this.createBand(audioContext);

    // Crossover filters (Linkwitz-Riley)
    this.lowpass1 = audioContext.createBiquadFilter();
    this.lowpass2 = audioContext.createBiquadFilter();
    this.bandpass1 = audioContext.createBiquadFilter();
    this.bandpass2 = audioContext.createBiquadFilter();
    this.highpass1 = audioContext.createBiquadFilter();
    this.highpass2 = audioContext.createBiquadFilter();

    this.setupCrossovers();
    this.initialize(options);
  }

  createBand(context) {
    return {
      gain: context.createGain(),
      filter1: context.createBiquadFilter(),
      filter2: context.createBiquadFilter()
    };
  }

  setupCrossovers() {
    // Low band: 2x lowpass at lowFreq
    // Mid band: highpass at lowFreq + lowpass at highFreq
    // High band: 2x highpass at highFreq
    // This creates Linkwiz-Riley 4th order crossover
  }

  setLowGain(gain) {
    this.lowBand.gain.gain.value = gain;
  }

  setMidGain(gain) {
    this.midBand.gain.gain.value = gain;
  }

  setHighGain(gain) {
    this.highBand.gain.gain.value = gain;
  }

  setLowFrequency(freq) {
    // Update crossover frequency
    this.lowpass1.frequency.value = freq;
    this.lowpass2.frequency.value = freq;
    this.bandpass1.frequency.value = freq;
  }

  setHighFrequency(freq) {
    this.bandpass2.frequency.value = freq;
    this.highpass1.frequency.value = freq;
    this.highpass2.frequency.value = freq;
  }
}
```

### Architecture Pattern - Auto Filter

```javascript
class AutoFilter {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Main filter
    this.filter = audioContext.createBiquadFilter();

    // Modulation sources
    this.lfo = audioContext.createOscillator();
    this.lfoGain = audioContext.createGain();

    this.envelope = {
      attack: 10,
      decay: 100,
      sustain: 0.5,
      release: 100
    };
    this.envelopeAmount = 0;

    // Dry/wet for parallel processing
    this.wetGain = audioContext.createGain();
    this.dryGain = audioContext.createGain();

    this.setupRouting();
    this.initialize(options);
  }

  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path
    this.input.connect(this.filter);
    this.filter.connect(this.wetGain);
    this.wetGain.connect(this.output);

    // LFO modulation
    this.lfo.connect(this.lfoGain);
    this.lfoGain.connect(this.filter.frequency);

    this.lfo.start();
  }

  setFilterType(type) {
    // Map to appropriate BiquadFilterNode type
    // For 24dB, might need to cascade two 12dB filters
  }

  setLFO(params) {
    if (params.rate) {
      this.lfo.frequency.value = params.rate;
    }

    if (params.amount !== undefined) {
      const depth = params.amount / 100;
      this.lfoGain.gain.value = depth * 2000; // Scale to frequency modulation
    }

    if (params.waveform) {
      this.lfo.type = params.waveform;
    }
  }

  triggerEnvelope() {
    // Called when audio input detected
    // Modulate filter frequency with ADSR envelope
    const now = this.context.currentTime;
    const attack = this.envelope.attack / 1000;
    const decay = this.envelope.decay / 1000;

    // Attack
    this.filter.frequency.cancelScheduledValues(now);
    this.filter.frequency.setValueAtTime(this.filter.frequency.value, now);
    this.filter.frequency.linearRampToValueAtTime(
      this.filter.frequency.value * (1 + this.envelopeAmount),
      now + attack
    );

    // Decay to sustain
    this.filter.frequency.linearRampToValueAtTime(
      this.filter.frequency.value * (1 + this.envelopeAmount * this.envelope.sustain),
      now + attack + decay
    );
  }
}
```

### Testing Checklist

- [ ] EQ Eight: All 8 bands function independently
- [ ] EQ Eight: Frequency response is accurate (verify with test signals)
- [ ] EQ Eight: Q parameter behaves correctly
- [ ] EQ Eight: All filter types work (bell, shelf, pass, notch)
- [ ] EQ Three: Clean crossover (no phase issues)
- [ ] EQ Three: Kill switches completely remove frequency ranges
- [ ] EQ Three: Smooth parameter changes (no clicks)
- [ ] Auto Filter: All filter types work correctly
- [ ] Auto Filter: LFO modulates frequency smoothly
- [ ] Auto Filter: Envelope follower responds to input
- [ ] Auto Filter: Tempo sync works accurately
- [ ] No audio artifacts during parameter changes

## Deliverables

### Code Files
```
/eq/
├── EQEight.js
├── EQThree.js
├── README.md
/filters/
├── AutoFilter.js
├── README.md
```

### Example HTML
Create `/examples/eq-filter-example.html`:
- EQ Eight surgical EQ demonstration
- EQ Three DJ mixing example
- Auto Filter with LFO sweep
- Auto Filter with envelope follower
- Frequency response visualization (optional)

### Documentation
- Parameter descriptions with ranges
- Filter type explanations
- Usage examples
- Performance notes

## Performance Targets

- **Latency**: < 3ms per plugin
- **CPU**: < 2% per plugin
- **Accuracy**: Frequency response within 0.5dB of specification

## Success Criteria

✅ All 3 plugins implemented and tested
✅ EQ Eight has accurate parametric EQ
✅ EQ Three has clean crossovers
✅ Auto Filter modulation is smooth
✅ No phase issues or artifacts
✅ Code is modular and documented
✅ Examples demonstrate key features

Begin with BiquadFilterNode research, then build each plugin. Test frequency response accuracy! 🎚️
