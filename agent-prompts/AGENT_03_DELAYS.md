# Agent 3: Time-Based Effects (Delay)

## Your Mission
You are responsible for implementing **3 delay-based effects**: Simple Delay, Ping Pong Delay, and Filter Delay. These plugins create echoes, rhythmic patterns, and spatial depth essential for music production.

## Plugins to Implement

### 1. Simple Delay
**Purpose**: Basic echo effect with feedback and filtering

**Parameters**:
- `delayTime` (0 to 5000 ms or sync to BPM: 1/32 to 4 bars)
- `feedback` (0 to 100%, default: 0)
- `filterType` (off, lowpass, highpass)
- `filterFreq` (20 Hz to 20 kHz, default: 5000)
- `mix` (0 to 100%, default: 50)
- `sync` (boolean, default: false) - tempo sync
- `pingPong` (boolean, default: false) - stereo bounce

**Key Features**:
- Tempo synchronization
- Feedback loop with filtering
- Stereo or mono operation
- Smooth parameter changes (no clicks)

### 2. Ping Pong Delay
**Purpose**: Stereo delay that bounces between left and right channels

**Parameters**:
- `delayTimeL` (0 to 5000 ms or sync)
- `delayTimeR` (0 to 5000 ms or sync)
- `feedback` (0 to 100%)
- `spread` (0 to 100%) - stereo width
- `filterFreq` (20 Hz to 20 kHz)
- `filterType` (off, lowpass, highpass)
- `mix` (0 to 100%)

**Key Features**:
- Independent L/R delay times
- Cross-feedback (L→R, R→L)
- Stereo spread control
- Tempo sync with musical divisions
- Filter in feedback path

### 3. Filter Delay
**Purpose**: 3 parallel delay lines with individual filtering and panning

**Parameters** (per tap, 3 taps total):
- `delayTime[1-3]` (0 to 5000 ms or sync)
- `feedback[1-3]` (0 to 100%)
- `pan[1-3]` (-100 to +100)
- `filterType[1-3]` (lowpass, highpass, bandpass)
- `filterFreq[1-3]` (20 Hz to 20 kHz)
- `filterQ[1-3]` (0.1 to 10)
- `volume[1-3]` (0 to 100%)

**Global Parameters**:
- `mix` (0 to 100%)
- `input` (-inf to +6 dB)

**Key Features**:
- 3 independent delay taps
- Individual filtering per tap
- Pan position per tap
- Parallel signal flow
- Tempo sync per tap

## Research Phase (Week 1)

### Essential Research Topics

1. **DelayNode Deep Dive**
   - Maximum delay time (Web Audio typically 1-3 seconds default)
   - Dynamic delay time changes (smooth transitions)
   - DelayNode precision and accuracy
   - Sample-accurate timing

2. **Feedback Loop Design**
   - Preventing runaway feedback
   - Gain compensation in feedback path
   - Filter integration in feedback
   - Stability considerations

3. **Tempo Synchronization**
   - BPM to milliseconds conversion
   - Musical note divisions (1/4, 1/8, 1/16, etc.)
   - Triplet and dotted note support
   - Smooth tempo changes

4. **Stereo Processing**
   - Ping-pong routing (L→R→L→R)
   - Stereo spread techniques
   - Pan law (constant power panning)
   - Haas effect and precedence

5. **Feedback Path Filtering**
   - Filter placement in feedback loop
   - High-frequency damping for natural decay
   - Resonance control
   - Multiple filters in series

### Reference Materials
- **Ableton Manual**: Simple Delay, Ping Pong Delay, Filter Delay
- **Web Audio API**: DelayNode specification
- **Tone.js**: Study delay implementations
- **DSP**: Delay line algorithms and feedback networks

### Code to Study
```javascript
// Basic delay with feedback
const delay = context.createDelay(5.0); // max 5 seconds
const feedback = context.createGain();
const wetGain = context.createGain();
const dryGain = context.createGain();

input.connect(delay);
delay.connect(feedback);
feedback.connect(delay); // Feedback loop
feedback.connect(wetGain);

// BPM sync calculation
function syncTimeToMS(division, bpm) {
  const beatDuration = 60000 / bpm; // One beat in ms
  const divisionMap = {
    '1/4': 1,
    '1/8': 0.5,
    '1/16': 0.25,
    '1/4T': 2/3, // Triplet
    '1/8D': 0.75 // Dotted
  };
  return beatDuration * divisionMap[division];
}
```

## Implementation Phase (Week 2-3)

### Architecture Pattern - Simple Delay

```javascript
class SimpleDelay {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    this.delay = audioContext.createDelay(5.0);
    this.feedback = audioContext.createGain();
    this.filter = audioContext.createBiquadFilter();

    this.wetGain = audioContext.createGain();
    this.dryGain = audioContext.createGain();

    // State
    this.bpm = 120;
    this.syncEnabled = false;

    this.setupRouting();
    this.initialize(options);
  }

  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path with feedback
    this.input.connect(this.delay);
    this.delay.connect(this.filter);
    this.filter.connect(this.feedback);

    // Feedback loop
    this.feedback.connect(this.delay);

    // Output
    this.feedback.connect(this.wetGain);
    this.wetGain.connect(this.output);
  }

  setDelayTime(ms, rampTime = 0.05) {
    // Smooth delay time changes to prevent artifacts
    const now = this.context.currentTime;
    this.delay.delayTime.cancelScheduledValues(now);
    this.delay.delayTime.setValueAtTime(this.delay.delayTime.value, now);
    this.delay.delayTime.linearRampToValueAtTime(ms / 1000, now + rampTime);
  }

  setFeedback(amount) {
    // Amount is 0-100, convert to gain
    // Use logarithmic scaling for natural feel
    const gain = amount / 100;
    this.feedback.gain.value = gain;
  }

  setMix(percent) {
    const wet = percent / 100;
    const dry = 1 - wet;

    this.wetGain.gain.value = wet;
    this.dryGain.gain.value = dry;
  }

  setFilter(type, frequency) {
    if (type === 'off') {
      this.filter.type = 'allpass';
    } else {
      this.filter.type = type; // 'lowpass' or 'highpass'
      this.filter.frequency.value = frequency;
    }
  }

  setSync(enabled, division = '1/4') {
    this.syncEnabled = enabled;
    if (enabled) {
      const ms = this.syncTimeToMS(division, this.bpm);
      this.setDelayTime(ms);
    }
  }

  setBPM(bpm) {
    this.bpm = bpm;
    if (this.syncEnabled) {
      // Update delay time based on new BPM
    }
  }

  syncTimeToMS(division, bpm) {
    const beatDuration = 60000 / bpm;
    const divisionMap = {
      '1/1': 4,
      '1/2': 2,
      '1/4': 1,
      '1/8': 0.5,
      '1/16': 0.25,
      '1/32': 0.125,
      '1/4T': 2/3,
      '1/8T': 1/3,
      '1/4D': 1.5,
      '1/8D': 0.75
    };
    return beatDuration * divisionMap[division];
  }

  dispose() {
    this.input.disconnect();
    this.delay.disconnect();
    this.feedback.disconnect();
    this.filter.disconnect();
    this.wetGain.disconnect();
    this.dryGain.disconnect();
    this.output.disconnect();
  }
}
```

### Architecture Pattern - Ping Pong Delay

```javascript
class PingPongDelay {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Separate L/R channels
    this.splitter = audioContext.createChannelSplitter(2);
    this.merger = audioContext.createChannelMerger(2);

    // L and R delay lines
    this.delayL = audioContext.createDelay(5.0);
    this.delayR = audioContext.createDelay(5.0);

    // Cross-feedback gains
    this.feedbackL = audioContext.createGain();
    this.feedbackR = audioContext.createGain();

    // Filters for each channel
    this.filterL = audioContext.createBiquadFilter();
    this.filterR = audioContext.createBiquadFilter();

    this.setupPingPongRouting();
    this.initialize(options);
  }

  setupPingPongRouting() {
    // Split input into L/R
    this.input.connect(this.splitter);

    // Left channel: delay → filter → feedback
    this.splitter.connect(this.delayL, 0);
    this.delayL.connect(this.filterL);
    this.filterL.connect(this.feedbackL);

    // Right channel: delay → filter → feedback
    this.splitter.connect(this.delayR, 1);
    this.delayR.connect(this.filterR);
    this.filterR.connect(this.feedbackR);

    // Cross-feedback: L feeds R, R feeds L
    this.feedbackL.connect(this.delayR);
    this.feedbackR.connect(this.delayL);

    // Output to merger
    this.feedbackL.connect(this.merger, 0, 0);
    this.feedbackR.connect(this.merger, 0, 1);

    this.merger.connect(this.output);
  }

  setSpread(amount) {
    // Spread controls the stereo width
    // 0 = mono, 100 = full stereo
    const spread = amount / 100;
    // Adjust pan or timing offset for spread
  }
}
```

### Architecture Pattern - Filter Delay

```javascript
class FilterDelay {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // 3 parallel delay taps
    this.taps = [];
    for (let i = 0; i < 3; i++) {
      this.taps.push(this.createTap(audioContext));
    }

    this.setupParallelTaps();
    this.initialize(options);
  }

  createTap(context) {
    return {
      delay: context.createDelay(5.0),
      feedback: context.createGain(),
      filter: context.createBiquadFilter(),
      pan: context.createStereoPanner(),
      volume: context.createGain()
    };
  }

  setupParallelTaps() {
    // Each tap runs in parallel
    this.taps.forEach((tap, index) => {
      // Input to delay
      this.input.connect(tap.delay);

      // Delay → filter → feedback
      tap.delay.connect(tap.filter);
      tap.filter.connect(tap.feedback);

      // Feedback loop
      tap.feedback.connect(tap.delay);

      // Output path: feedback → pan → volume → output
      tap.feedback.connect(tap.pan);
      tap.pan.connect(tap.volume);
      tap.volume.connect(this.output);
    });
  }

  setTap(index, params) {
    const tap = this.taps[index];

    if (params.delayTime !== undefined) {
      tap.delay.delayTime.value = params.delayTime / 1000;
    }

    if (params.feedback !== undefined) {
      tap.feedback.gain.value = params.feedback / 100;
    }

    if (params.pan !== undefined) {
      tap.pan.pan.value = params.pan / 100;
    }

    if (params.volume !== undefined) {
      tap.volume.gain.value = params.volume / 100;
    }

    if (params.filterFreq !== undefined) {
      tap.filter.frequency.value = params.filterFreq;
    }

    if (params.filterQ !== undefined) {
      tap.filter.Q.value = params.filterQ;
    }

    if (params.filterType !== undefined) {
      tap.filter.type = params.filterType;
    }
  }
}
```

### Testing Checklist

- [ ] Simple Delay: Feedback loop is stable (doesn't run away)
- [ ] Simple Delay: Filter in feedback works correctly
- [ ] Simple Delay: Tempo sync is accurate
- [ ] Simple Delay: Delay time changes are smooth (no clicks)
- [ ] Ping Pong: Bounces between L/R correctly
- [ ] Ping Pong: Cross-feedback creates decay
- [ ] Ping Pong: Stereo spread works
- [ ] Filter Delay: All 3 taps work independently
- [ ] Filter Delay: Panning per tap works
- [ ] Filter Delay: Filters per tap work
- [ ] No audio dropouts or glitches
- [ ] Maximum delay time is adequate (5 seconds)

## Deliverables

### Code Files
```
/delay/
├── SimpleDelay.js
├── PingPongDelay.js
├── FilterDelay.js
└── README.md
```

### Example HTML
Create `/examples/delay-rhythms-example.html`:
- Simple delay with tempo sync
- Ping pong stereo bounce demo
- Filter delay rhythmic patterns
- Dub-style delay (high feedback)
- Combined delay chains

## Performance Targets

- **Latency**: Inherent delay time (user-controlled)
- **CPU**: < 3% per plugin
- **Accuracy**: Delay time within 1ms of specification

## Success Criteria

✅ All 3 delay plugins implemented
✅ Feedback loops are stable
✅ Tempo sync works accurately
✅ No clicks during parameter changes
✅ Stereo imaging works correctly
✅ Code is modular and documented

Begin with DelayNode research, then implement feedback carefully. Test for stability! ⏱️
