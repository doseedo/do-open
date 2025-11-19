# Agent 1: Dynamics Processors

## Your Mission
You are responsible for implementing **4 professional-grade dynamics processing plugins**: Compressor, Gate, Limiter, and Glue Compressor. These are foundational tools for controlling dynamic range in audio production.

## Plugins to Implement

### 1. Compressor
**Purpose**: Reduce dynamic range by attenuating signals above threshold

**Parameters**:
- `threshold` (-60 to 0 dB, default: -24)
- `ratio` (1 to 20, default: 4)
- `attack` (0.1 to 100 ms, default: 10)
- `release` (10 to 1000 ms, default: 100)
- `knee` (0 to 12 dB, default: 0) - hard knee to soft knee
- `makeupGain` (0 to 24 dB, default: 0)
- `mix` (0 to 100%, default: 100) - parallel compression support
- `sidechain` (boolean, default: false)

**Key Features**:
- Sidechain input support
- Auto makeup gain option
- RMS and Peak detection modes
- Gain reduction metering

### 2. Gate
**Purpose**: Attenuate signals below threshold (noise gate)

**Parameters**:
- `threshold` (-60 to 0 dB, default: -32)
- `range` (0 to -60 dB, default: -60) - maximum attenuation
- `attack` (0.1 to 50 ms, default: 1)
- `hold` (0 to 500 ms, default: 10)
- `release` (10 to 2000 ms, default: 100)
- `sidechain` (boolean, default: false)

**Key Features**:
- Hold time to prevent chattering
- Sidechain support for ducking
- Hysteresis to prevent rapid on/off

### 3. Limiter
**Purpose**: Prevent signal from exceeding ceiling (brick wall limiting)

**Parameters**:
- `ceiling` (-20 to 0 dB, default: -0.3)
- `release` (10 to 1000 ms, default: 50)
- `lookahead` (0 to 10 ms, default: 5)

**Key Features**:
- True peak limiting
- Lookahead implementation (requires buffer delay)
- Ultra-fast attack time (<1ms)
- Gain reduction metering

### 4. Glue Compressor
**Purpose**: Vintage-style bus compressor for "gluing" mixes together

**Parameters**:
- `threshold` (-40 to 0 dB, default: -12)
- `ratio` (2, 4, 10, or infinity:1)
- `attack` (0.1, 0.3, 1, 3, 10, 30 ms) - stepped values
- `release` (0.1, 0.3, 0.6, 1.2 seconds or Auto) - stepped values
- `makeupGain` (0 to 20 dB, default: 0)
- `dryWet` (0 to 100%, default: 100)

**Key Features**:
- Vintage VCA-style compression character
- Soft knee
- Auto-release mode
- Peak mode switch

## Research Phase (Week 1)

### Essential Research Topics

1. **DynamicsCompressorNode Deep Dive**
   - Study Web Audio's built-in DynamicsCompressorNode
   - Identify limitations (no sidechain, limited control)
   - Decide when to use native vs custom implementation

2. **Compression Algorithms**
   - Feed-forward vs feedback detection
   - RMS vs Peak detection mathematics
   - Envelope follower design (attack/release curves)
   - Knee curve calculations (hard vs soft)

3. **Lookahead Limiting**
   - DelayNode-based lookahead buffer
   - Peak detection across lookahead window
   - Gain smoothing to prevent distortion

4. **Sidechain Routing**
   - External sidechain input implementation
   - Signal splitting and analysis
   - Independent processing paths

5. **Gain Reduction Metering**
   - Real-time GR calculation
   - Metering without affecting audio

### Reference Materials
- **Ableton Manual**: Read compressor, gate, limiter, glue compressor sections
- **DSP Guide**: Chapter on dynamics processing
- **Web Audio**: Study DynamicsCompressorNode source implementations
- **Research Papers**:
  - "Digital Dynamic Range Compressor Design" by Giannoulis et al.
  - Look into vintage compressor modeling

### Code to Study
```javascript
// Study Tone.js implementations:
// - Tone.Compressor
// - Tone.Gate
// - Tone.Limiter

// Understand envelope detection:
const attackCoeff = Math.exp(-1 / (attackTime * sampleRate));
const releaseCoeff = Math.exp(-1 / (releaseTime * sampleRate));
```

## Implementation Phase (Week 2-3)

### Architecture Pattern

```javascript
class Compressor {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();
    this.sidechainInput = audioContext.createGain();

    // Internal nodes
    this.detector = audioContext.createGain();
    this.compressorNode = audioContext.createDynamicsCompressor();

    // Parameters
    this.params = {
      threshold: this.compressorNode.threshold,
      ratio: this.compressorNode.ratio,
      attack: this.compressorNode.attack,
      release: this.compressorNode.release,
      knee: this.compressorNode.knee,
      makeupGain: this.output.gain,
      mix: { value: 1.0 } // For parallel compression
    };

    // State
    this.gainReduction = 0;
    this.bypassed = false;

    this.initialize(options);
  }

  initialize(options) {
    // Set up audio graph
    // Apply default or provided options
  }

  connect(destination) {
    this.output.connect(destination);
    return destination;
  }

  disconnect() {
    this.output.disconnect();
  }

  setParameter(name, value, time = 0) {
    // Handle parameter changes with automation
  }

  enableSidechain(external) {
    // Route sidechain input
  }

  getGainReduction() {
    // Return current GR for metering
    return this.gainReduction;
  }

  bypass(enabled) {
    this.bypassed = enabled;
    // Implement true bypass
  }

  dispose() {
    // Clean up all nodes
  }
}
```

### Custom Implementation Considerations

For advanced features, you may need AudioWorklet:

```javascript
// compressor-processor.js (AudioWorklet)
class CompressorProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    for (let channel = 0; channel < input.length; channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];

      for (let i = 0; i < inputChannel.length; i++) {
        // Custom compression algorithm
        // Full control over attack/release envelopes
        // Accurate gain reduction calculation
      }
    }

    return true;
  }
}
```

### Testing Checklist

- [ ] Compressor reduces peaks correctly
- [ ] Ratio parameter works accurately (measure with test tones)
- [ ] Attack/release times match specifications
- [ ] Knee smooths transition around threshold
- [ ] Makeup gain compensates for gain reduction
- [ ] Sidechain input triggers compression
- [ ] No clicks/pops during parameter changes
- [ ] Gain reduction metering is accurate
- [ ] Gate opens/closes cleanly without clicks
- [ ] Limiter prevents signal from exceeding ceiling
- [ ] Lookahead actually prevents true peaks
- [ ] Glue compressor has vintage character

## Deliverables

### Code Files
```
/dynamics/
├── Compressor.js
├── Gate.js
├── Limiter.js
├── GlueCompressor.js
├── README.md
└── worklets/
    └── compressor-processor.js (if needed)
```

### Example HTML
Create `/examples/dynamics-chain-example.html`:
- Demonstrate each plugin
- Show sidechain ducking
- Parallel compression example
- Mastering chain (Compressor → Limiter)
- Visual gain reduction meters

### Documentation
In `/dynamics/README.md`:
- Overview of each plugin
- Parameter descriptions with ranges
- Usage examples
- Known limitations
- Performance notes

## Performance Targets

- **Latency**: < 5ms per plugin (excluding lookahead)
- **CPU**: < 3% per plugin on modern hardware
- **Accuracy**: Gain reduction within 0.5dB of specification

## Success Criteria

✅ All 4 plugins implemented and tested
✅ Sidechain routing works correctly
✅ Parameters respond in real-time
✅ No audio artifacts (clicks, pops)
✅ Gain reduction metering accurate
✅ Lookahead limiter prevents true peaks
✅ Code is modular and well-documented
✅ Examples demonstrate key features

## Questions to Answer

1. Should I use Web Audio's DynamicsCompressorNode or build custom?
2. How do I implement accurate lookahead limiting?
3. What's the best way to calculate RMS vs Peak detection?
4. How can I create a sidechain input that works with the routing system?
5. What envelope follower algorithm provides the smoothest response?

## Resources

- [Ableton Compressor Manual](https://www.ableton.com/en/manual/live-audio-effect-reference/)
- [DynamicsCompressorNode Spec](https://www.w3.org/TR/webaudio/#dynamicscompressornode)
- [Tone.js Compressor Source](https://github.com/Tonejs/Tone.js/blob/dev/Tone/component/channel/Compressor.ts)
- [Digital Dynamics Processing Paper](https://www.semanticscholar.org/paper/Digital-Dynamic-Range-Compressor-Design%E2%80%94A-Tutorial-Giannoulis-Massberg/aa13fb8a5a8c2d5c4f2b3c4f1d7e8c9f5b6a7d8e)

Begin with research, then build incrementally. Test each plugin thoroughly before moving to the next. Good luck! 🎛️
