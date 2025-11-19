# Time-Based Effects (Delay)

Professional delay-based audio effects built with the Web Audio API. These plugins create echoes, rhythmic patterns, and spatial depth essential for music production.

## Plugins

### 1. Simple Delay
Basic echo effect with feedback and filtering.

**Features:**
- Tempo synchronization (BPM sync with musical divisions)
- Feedback loop with filtering (lowpass/highpass)
- Stereo or mono operation
- Ping-pong stereo mode
- Smooth parameter changes (no clicks)

**Parameters:**
- `delayTime` (0-5000 ms) - Delay time
- `feedback` (0-100%) - Feedback amount
- `filterType` ('off', 'lowpass', 'highpass') - Filter type in feedback path
- `filterFreq` (20-20000 Hz) - Filter frequency
- `mix` (0-100%) - Dry/wet mix
- `sync` (boolean) - Tempo sync enable
- `division` ('1/4', '1/8', etc.) - Musical note division
- `pingPong` (boolean) - Stereo ping-pong mode

**Usage:**
```javascript
const audioContext = new AudioContext();
const delay = new SimpleDelay(audioContext, {
  delayTime: 250,
  feedback: 50,
  filterType: 'lowpass',
  filterFreq: 5000,
  mix: 50,
  sync: true,
  division: '1/4',
  pingPong: false
});

// Connect audio source
source.connect(delay.input);
delay.output.connect(audioContext.destination);

// Control parameters
delay.setDelayTime(500); // Set delay to 500ms
delay.setFeedback(70); // Set feedback to 70%
delay.setMix(60); // Set mix to 60% wet
delay.setBPM(120); // Set tempo to 120 BPM
delay.setSync(true, '1/8'); // Sync to eighth notes
delay.setPingPong(true); // Enable ping-pong mode
```

### 2. Ping Pong Delay
Stereo delay that bounces between left and right channels.

**Features:**
- Independent L/R delay times
- Cross-feedback (L→R, R→L)
- Stereo spread control
- Tempo sync with musical divisions
- Filter in feedback path

**Parameters:**
- `delayTimeL` (0-5000 ms) - Left channel delay
- `delayTimeR` (0-5000 ms) - Right channel delay
- `feedback` (0-100%) - Cross-feedback amount
- `spread` (0-100%) - Stereo width
- `filterFreq` (20-20000 Hz) - Filter frequency
- `filterType` ('off', 'lowpass', 'highpass') - Filter type
- `mix` (0-100%) - Dry/wet mix
- `syncL`/`syncR` (boolean) - Tempo sync per channel
- `divisionL`/`divisionR` - Musical divisions per channel

**Usage:**
```javascript
const pingPong = new PingPongDelay(audioContext, {
  delayTimeL: 250,
  delayTimeR: 125,
  feedback: 50,
  spread: 75,
  filterType: 'lowpass',
  filterFreq: 5000,
  mix: 60
});

// Connect audio source
source.connect(pingPong.input);
pingPong.output.connect(audioContext.destination);

// Control parameters
pingPong.setDelayTimeL(375); // Left delay: 375ms
pingPong.setDelayTimeR(188); // Right delay: 188ms
pingPong.setFeedback(60); // Feedback: 60%
pingPong.setSpread(90); // Stereo spread: 90%
pingPong.setBPM(140); // Tempo: 140 BPM
pingPong.setSyncL(true, '1/4'); // Left: quarter notes
pingPong.setSyncR(true, '1/8'); // Right: eighth notes
```

### 3. Filter Delay
Three parallel delay lines with individual filtering and panning.

**Features:**
- 3 independent delay taps
- Individual filtering per tap (lowpass, highpass, bandpass)
- Pan position per tap
- Parallel signal flow
- Tempo sync per tap
- Independent volume per tap

**Parameters (per tap):**
- `delayTime` (0-5000 ms) - Delay time
- `feedback` (0-100%) - Feedback amount
- `pan` (-100 to +100) - Pan position
- `filterType` ('lowpass', 'highpass', 'bandpass') - Filter type
- `filterFreq` (20-20000 Hz) - Filter frequency
- `filterQ` (0.1-10) - Filter resonance
- `volume` (0-100%) - Tap volume

**Global Parameters:**
- `mix` (0-100%) - Dry/wet mix
- `input` (-60 to +6 dB) - Input gain

**Usage:**
```javascript
const filterDelay = new FilterDelay(audioContext, {
  mix: 50,
  input: 0,
  tap1: {
    delayTime: 250,
    feedback: 30,
    pan: -50,
    filterType: 'lowpass',
    filterFreq: 2000,
    filterQ: 1.0,
    volume: 70
  },
  tap2: {
    delayTime: 500,
    feedback: 30,
    pan: 0,
    filterType: 'bandpass',
    filterFreq: 1000,
    filterQ: 2.0,
    volume: 70
  },
  tap3: {
    delayTime: 750,
    feedback: 30,
    pan: 50,
    filterType: 'highpass',
    filterFreq: 500,
    filterQ: 1.0,
    volume: 70
  }
});

// Connect audio source
source.connect(filterDelay.input);
filterDelay.output.connect(audioContext.destination);

// Control individual tap
filterDelay.setTap(0, {
  delayTime: 300,
  feedback: 40,
  pan: -70,
  filterFreq: 1500
});

// Control global parameters
filterDelay.setMix(60);
filterDelay.setInputGain(0);
filterDelay.setBPM(120);
```

## Musical Note Divisions

All delays support tempo sync with the following musical divisions:

- **Whole notes:** `4` (4 bars), `2` (2 bars), `1` (1 bar)
- **Standard notes:** `1/2`, `1/4`, `1/8`, `1/16`, `1/32`
- **Triplets:** `1/2T`, `1/4T`, `1/8T`, `1/16T`
- **Dotted notes:** `1/2D`, `1/4D`, `1/8D`, `1/16D`

## Examples

See `/examples/delay-rhythms-example.html` for interactive demonstrations:
- Simple delay with tempo sync
- Ping pong stereo bounce
- Filter delay rhythmic patterns
- Dub-style delay (high feedback)
- Combined delay chains

## Technical Details

### Feedback Stability
All plugins implement proper feedback gain limiting to prevent runaway feedback:
- Feedback values use exponential curves for natural feel
- Cross-feedback is reduced by 0.5× to maintain stability
- Filters in feedback path provide natural high-frequency damping

### Delay Time Changes
Smooth parameter changes prevent clicks and artifacts:
- Delay time changes use 50ms linear ramps by default
- All AudioParam changes are scheduled properly
- No audible glitches during parameter automation

### Performance
- **CPU Usage:** < 3% per plugin (typical)
- **Latency:** Inherent delay time (user-controlled)
- **Accuracy:** Delay time within 1ms of specification
- **Maximum Delay:** 5 seconds (configurable)

### Browser Compatibility
- Uses Web Audio API standard features
- Fallback panning for browsers without StereoPanner
- Tested on Chrome, Firefox, Safari, Edge

## Cleanup

Always call `dispose()` when done:
```javascript
delay.dispose();
pingPong.dispose();
filterDelay.dispose();
```

## License

MIT License - See LICENSE file for details
