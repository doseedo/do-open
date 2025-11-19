# Master Prompt: Ableton Suite Web Audio Effects Library

## Project Overview

You are part of a team creating a comprehensive JavaScript Web Audio API implementation of 20+ Ableton Live Suite stock effects/audio processing plugins. This library will be modular, allowing custom routing and efficient parameter changes in real-time.

## Core Requirements (All Agents)

### Architecture Principles
1. **Modular Design**: Each plugin must be a self-contained ES6 class
2. **Web Audio API**: Use modern Web Audio API (AudioWorklet when necessary)
3. **Custom Routing**: Support arbitrary signal flow and parallel/serial connections
4. **Parameter Management**: Real-time parameter changes with automation support
5. **Performance**: Optimized for low-latency, real-time audio processing
6. **TypeScript Ready**: Structure code to support TypeScript definitions

### Code Structure Standards
```javascript
class PluginName {
  constructor(audioContext, options = {}) {
    this.context = audioContext;
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();
    // Plugin-specific nodes
    this.params = {}; // Exposed parameters
    this.initialize(options);
  }

  initialize(options) { /* Setup audio graph */ }

  connect(destination) { /* Custom routing */ }
  disconnect() { /* Cleanup */ }

  setParameter(name, value, time = 0) { /* Automation support */ }

  getParameter(name) { /* Get current value */ }

  bypass(enabled) { /* Wet/dry bypass */ }

  dispose() { /* Resource cleanup */ }
}
```

### File Organization
```
/web-audio-effects/
├── /core/
│   ├── AudioNode.js (Base class)
│   ├── Router.js (Signal routing system)
│   ├── ParamAutomation.js (Parameter automation)
│   └── PresetManager.js (Preset management)
├── /dynamics/
├── /eq/
├── /filters/
├── /modulation/
├── /delay/
├── /reverb/
├── /distortion/
├── /creative/
├── /spectral/
├── /utility/
└── /examples/
```

### Research Requirements (All Agents)
1. Study Ableton's official documentation for your assigned plugins
2. Research Web Audio API best practices (2023-2025)
3. Review existing open-source implementations (Tone.js, Web Audio modules)
4. Understand DSP fundamentals for each effect type
5. Investigate AudioWorklet for CPU-intensive effects
6. Research parameter ranges, curves, and default values
7. Study signal flow and internal routing of each plugin

---

## Agent Task Assignments

## Agent 1: Dynamics Processors
**Plugins to Implement**: Compressor, Gate, Limiter, Glue Compressor (4 plugins)

### Task Description
Implement professional-grade dynamics processing plugins that form the backbone of mixing and mastering workflows. These plugins control the dynamic range of audio signals.

### Specific Requirements
- **Compressor**: Threshold, ratio, attack, release, knee, makeup gain, sidechain input
- **Gate**: Threshold, range, attack, hold, release, sidechain input
- **Limiter**: Ceiling, release, lookahead (requires AudioWorklet or delay-based workaround)
- **Glue Compressor**: Vintage-style bus compressor with attack, release, ratio, makeup, dry/wet

### Research Focus
- DynamicsCompressorNode API and limitations
- Custom compressor algorithms (feed-forward vs feedback)
- RMS vs peak detection methods
- Lookahead limiting techniques in Web Audio
- Sidechain routing implementation
- Accurate envelope detection (attack/release curves)
- Soft vs hard knee compression

### Deliverables
- `/dynamics/Compressor.js`
- `/dynamics/Gate.js`
- `/dynamics/Limiter.js`
- `/dynamics/GlueCompressor.js`
- `/examples/dynamics-chain-example.html`

---

## Agent 2: EQ & Filters
**Plugins to Implement**: EQ Eight, EQ Three, Auto Filter (3 plugins)

### Task Description
Create parametric and DJ-style equalizers plus a versatile auto-filter effect. EQs are fundamental for frequency shaping and tonal balance.

### Specific Requirements
- **EQ Eight**: 8-band parametric EQ with frequency, gain, Q, filter type per band (bell, shelf, highpass, lowpass, notch)
- **EQ Three**: DJ-style 3-band EQ (low/mid/high kill switches with gain control)
- **Auto Filter**: Multi-mode filter with envelope follower and LFO modulation (lowpass, highpass, bandpass, notch)

### Research Focus
- BiquadFilterNode comprehensive usage
- Multiple filter topologies (Butterworth, Linkwitz-Riley)
- Frequency response visualization
- Filter coefficient calculation for various types
- LFO implementation for Auto Filter
- Envelope follower design
- Q factor mapping and resonance control

### Deliverables
- `/eq/EQEight.js`
- `/eq/EQThree.js`
- `/filters/AutoFilter.js`
- `/examples/eq-filter-example.html`

---

## Agent 3: Time-Based Effects (Delay)
**Plugins to Implement**: Simple Delay, Ping Pong Delay, Filter Delay (3 plugins)

### Task Description
Implement delay-based effects for creating echoes, rhythmic patterns, and spatial depth. Delays are essential for creating space and rhythm in mixes.

### Specific Requirements
- **Simple Delay**: Delay time (sync/free), feedback, dry/wet, filter (highpass/lowpass in feedback)
- **Ping Pong Delay**: Stereo ping-pong with independent L/R times, feedback, spread
- **Filter Delay**: 3 parallel delay lines with individual filters, panning, feedback

### Research Focus
- DelayNode precision and tempo sync
- Feedback loop implementation without audio glitches
- Stereo field manipulation for ping-pong effect
- BPM synchronization and tempo mapping
- Filter integration in feedback paths
- Anti-aliasing in delay feedback loops
- Maximum delay time management

### Deliverables
- `/delay/SimpleDelay.js`
- `/delay/PingPongDelay.js`
- `/delay/FilterDelay.js`
- `/examples/delay-rhythms-example.html`

---

## Agent 4: Modulation Effects
**Plugins to Implement**: Chorus, Flanger, Phaser, Tremolo/Auto Pan (4 plugins)

### Task Description
Create modulation effects that add movement and depth through time-varying parameter changes. These effects are staples in music production for adding width and animation.

### Specific Requirements
- **Chorus**: Multiple delay lines with LFO modulation, depth, rate, feedback, dry/wet
- **Flanger**: Short delay with feedback and LFO, manual control, resonance
- **Phaser**: Multi-stage all-pass filters with LFO, feedback, stages (4/6/8/12)
- **Tremolo/Auto Pan**: Amplitude or pan modulation, rate, depth, waveform shapes

### Research Focus
- LFO implementation (sine, triangle, square, sawtooth, random)
- Multiple delay line management for chorus
- All-pass filter design for phaser
- Phase cancellation effects
- Feedback path stability
- Waveform shaping and smoothing
- Stereo width control

### Deliverables
- `/modulation/Chorus.js`
- `/modulation/Flanger.js`
- `/modulation/Phaser.js`
- `/modulation/Tremolo.js`
- `/examples/modulation-showcase-example.html`

---

## Agent 5: Reverb & Spatial Effects
**Plugins to Implement**: Reverb, Hybrid Reverb, Echo (3 plugins)

### Task Description
Implement spatial effects for creating realistic acoustic spaces and ambient textures. Reverb is crucial for depth and realism in any mix.

### Specific Requirements
- **Reverb**: Algorithmic reverb with decay time, pre-delay, diffusion, damping, dry/wet
- **Hybrid Reverb**: Convolution reverb + algorithmic tail, IR loader support
- **Echo**: Delay with modulation, ducking, reverb, rhythmic patterns

### Research Focus
- ConvolverNode for impulse response reverb
- Algorithmic reverb design (Schroeder, Freeverb, Dattorro)
- Impulse response loading and management
- Early reflections vs late reverb tail
- Diffusion network design
- Frequency-dependent damping
- Reverb ducking techniques

### Deliverables
- `/reverb/Reverb.js`
- `/reverb/HybridReverb.js`
- `/reverb/Echo.js`
- `/reverb/impulse-responses/` (sample IRs)
- `/examples/spatial-effects-example.html`

---

## Agent 6: Distortion & Saturation
**Plugins to Implement**: Overdrive, Saturator, Distortion, Redux (4 plugins)

### Task Description
Create harmonic distortion and saturation effects for adding warmth, grit, and character. These plugins shape tone through controlled harmonic generation.

### Specific Requirements
- **Overdrive**: Tube-style soft clipping, drive, tone, dry/wet
- **Saturator**: Multi-mode saturation (warm, digital, analog), drive, color, dry/wet
- **Distortion**: Hard clipping with pre/post filtering, drive, tone stack
- **Redux**: Bit crushing and sample rate reduction, bit depth, downsample amount

### Research Focus
- WaveShaperNode curve generation
- Various distortion algorithms (soft clip, hard clip, asymmetric, foldback)
- Oversampling techniques to reduce aliasing
- Bit crushing and sample rate reduction simulation
- Pre and post filtering (tone controls)
- DC offset removal
- Gain compensation

### Deliverables
- `/distortion/Overdrive.js`
- `/distortion/Saturator.js`
- `/distortion/Distortion.js`
- `/distortion/Redux.js`
- `/examples/distortion-shootout-example.html`

---

## Agent 7: Creative Effects
**Plugins to Implement**: Beat Repeat, Grain Delay, Erosion, Vinyl Distortion (4 plugins)

### Task Description
Implement experimental and creative effects for unique sound design. These plugins offer unconventional processing for electronic music production.

### Specific Requirements
- **Beat Repeat**: Buffer capture and repetition, chance, gate, pitch, filter, mix
- **Grain Delay**: Granular synthesis delay, grain size, density, pitch randomization, spray
- **Erosion**: Noise-based distortion, frequency, width, mode
- **Vinyl Distortion**: Vinyl simulation with crackle, wear, pinch effect

### Research Focus
- AudioBuffer manipulation and playback
- Granular synthesis techniques
- Buffer recording and loop points
- Randomization and probability
- Noise generation (white, pink, brown)
- Sample accurate timing for beat repeat
- Vinyl artifacts simulation

### Deliverables
- `/creative/BeatRepeat.js`
- `/creative/GrainDelay.js`
- `/creative/Erosion.js`
- `/creative/VinylDistortion.js`
- `/examples/creative-sound-design-example.html`

---

## Agent 8: Spectral & Advanced Processing
**Plugins to Implement**: Spectral Time, Spectral Resonator, Frequency Shifter, Vocoder (4 plugins)

### Task Description
Implement advanced spectral processing effects using FFT-based techniques. These plugins manipulate the frequency domain for unique timbral transformations.

### Specific Requirements
- **Spectral Time**: Time stretching without pitch change, freeze, blur
- **Spectral Resonator**: Resonant comb filtering based on spectral analysis, pitch tracking
- **Frequency Shifter**: Linear frequency shifting (not pitch shifting), frequency, fine, wide
- **Vocoder**: Multi-band vocoder with carrier/modulator, bands, formant shift, attack/release

### Research Focus
- AudioWorklet for FFT processing
- FFT/IFFT implementation or libraries
- Phase vocoder algorithms
- Single-sideband modulation for frequency shifting
- Comb filtering and resonator design
- Band-split vocoder architecture
- Spectral freezing techniques

### Deliverables
- `/spectral/SpectralTime.js`
- `/spectral/SpectralResonator.js`
- `/spectral/FrequencyShifter.js`
- `/spectral/Vocoder.js`
- `/spectral/worklets/` (AudioWorklet processors)
- `/examples/spectral-processing-example.html`

---

## Agent 9: Utility & Analysis Tools
**Plugins to Implement**: Utility, Spectrum Analyzer, Tuner, Channel EQ (4 plugins)

### Task Description
Create essential utility plugins for gain staging, analysis, and channel processing. These tools are fundamental for any production environment.

### Specific Requirements
- **Utility**: Gain, pan, phase invert (L/R), stereo width, mono switch, DC filter
- **Spectrum Analyzer**: Real-time FFT display, multiple resolutions, peak hold, freeze
- **Tuner**: Pitch detection, note display, cents deviation, reference frequency adjust
- **Channel EQ**: Simple low/high cut filters for mixing, frequency and slope control

### Research Focus
- AnalyserNode for visualization
- FFT size and window functions
- Pitch detection algorithms (autocorrelation, YIN, FFT-based)
- Stereo width manipulation techniques
- Mid/Side processing
- Canvas rendering for real-time displays
- Phase correlation metering

### Deliverables
- `/utility/Utility.js`
- `/utility/SpectrumAnalyzer.js`
- `/utility/Tuner.js`
- `/utility/ChannelEQ.js`
- `/examples/utility-tools-example.html`

---

## Agent 10: Integration & Routing System
**Plugins to Implement**: Master Routing, Preset Manager, Signal Flow Architecture (Framework)

### Task Description
Create the core infrastructure that ties all plugins together. This includes the routing system, preset management, and overall architecture that enables modular plugin usage.

### Specific Requirements
- **Master Router**: Signal flow graph, arbitrary plugin chains, parallel/serial routing, send/return
- **Preset Manager**: Save/load presets for individual plugins and chains, JSON format
- **Base Plugin Class**: Shared functionality for all plugins (connect, disconnect, bypass, automation)
- **Parameter Automation**: Timeline-based parameter changes, curves, automation recording
- **Performance Monitor**: CPU usage, buffer underruns, node count

### Research Focus
- Audio graph optimization
- Circular dependency prevention in routing
- Preset serialization best practices
- AudioParam automation scheduling
- Memory management and cleanup
- Module bundling strategies
- State management patterns

### Deliverables
- `/core/BasePlugin.js`
- `/core/Router.js`
- `/core/PresetManager.js`
- `/core/ParamAutomation.js`
- `/core/PerformanceMonitor.js`
- `/examples/master-routing-example.html`
- `/examples/full-mixing-console-example.html`

---

## Global Deliverables (All Agents)

### Documentation Requirements
1. **README.md** for your plugin category with:
   - Overview of each plugin
   - Parameter descriptions and ranges
   - Code examples
   - Known limitations

2. **API Documentation** (JSDoc comments):
   ```javascript
   /**
    * @class Compressor
    * @description Dynamic range compressor with sidechain support
    * @param {AudioContext} audioContext - Web Audio context
    * @param {Object} options - Configuration options
    * @param {number} options.threshold - Compression threshold in dB (-60 to 0)
    * @param {number} options.ratio - Compression ratio (1 to 20)
    */
   ```

3. **Example HTML files** demonstrating:
   - Basic usage
   - Parameter automation
   - Routing configurations
   - Preset loading

### Testing Requirements
1. Create test signals (sine waves, noise, impulses)
2. Verify parameter ranges work correctly
3. Test edge cases (extreme values, rapid changes)
4. Check for audio artifacts (clicks, pops, DC offset)
5. Measure CPU usage
6. Test in Chrome, Firefox, Safari

### Performance Targets
- Latency: < 10ms total (aim for buffer size flexibility)
- CPU: Each plugin should use < 5% CPU on average device
- Memory: Efficient cleanup, no memory leaks
- Real-time: No audio dropouts during parameter changes

---

## Research Resources

### Essential Reading
1. Web Audio API Specification (W3C)
2. "Designing Audio Effect Plugins in C++" by Will Pirkle (concepts apply to Web Audio)
3. Ableton Live Manual - Audio Effects section
4. JUCE Framework documentation (for DSP concepts)
5. Tone.js source code (reference implementations)
6. Web Audio API Advanced Topics (Google Developers)

### Reference Implementations
- Tone.js (https://tonejs.github.io/)
- Tuna.js (audio effects library)
- Pizzicato.js (simple audio library)
- Web Audio Modules (WAM) standard

### Tools for Research
- Ableton Live Suite (analyze stock plugins)
- Desmos or Wolfram Alpha (for transfer function visualization)
- Audacity (analyze effect outputs)
- Chrome DevTools Performance tab

---

## Communication & Collaboration

### Naming Conventions
- Classes: PascalCase (e.g., `Compressor`, `EQEight`)
- Files: Match class name (e.g., `Compressor.js`)
- Parameters: camelCase (e.g., `threshold`, `attackTime`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_DELAY_TIME`)

### Git Workflow
- Branch: `feature/plugin-category-name`
- Commit format: `[Category] Plugin Name - Description`
- Example: `[Dynamics] Compressor - Implement sidechain routing`

### Code Review Checklist
- [ ] Follows base class structure
- [ ] All parameters have validation
- [ ] Includes dispose() method
- [ ] No memory leaks (disconnects all nodes)
- [ ] Example HTML file included
- [ ] JSDoc comments complete
- [ ] Tested in multiple browsers
- [ ] CPU usage is acceptable

---

## Success Criteria

Your implementation is successful when:
1. ✅ All assigned plugins are fully functional
2. ✅ Plugins can be chained in arbitrary order
3. ✅ Parameters respond in real-time without artifacts
4. ✅ Code is clean, modular, and well-documented
5. ✅ Examples demonstrate core functionality
6. ✅ Sound quality is comparable to Ableton originals
7. ✅ Performance meets targets
8. ✅ Other agents can integrate with your plugins seamlessly

---

## Timeline Expectations

- **Week 1**: Research phase, architecture planning, base implementations
- **Week 2**: Core plugin development, initial testing
- **Week 3**: Refinement, optimization, cross-browser testing
- **Week 4**: Documentation, examples, integration with other agents

---

## Questions to Consider

Before starting, research and answer:
1. What are the DSP fundamentals behind my assigned effects?
2. Which Web Audio API nodes are most suitable for each plugin?
3. Where might I need AudioWorklet vs standard nodes?
4. What are the typical parameter ranges in professional audio software?
5. How can I prevent audio artifacts during parameter changes?
6. What are the CPU bottlenecks for my plugin type?
7. How do other Web Audio libraries implement similar effects?

---

## Final Notes

This is an ambitious project requiring deep understanding of both DSP and Web Audio API. Focus on:
- **Accuracy**: Match Ableton's sound and behavior as closely as possible
- **Performance**: Real-time audio requires efficient code
- **Modularity**: Your plugins will be used in combination with others
- **Documentation**: Others need to understand and use your work

Remember: Research first, code second. Understanding the theory behind each effect is crucial for quality implementation.

Good luck, and create something amazing! 🎵
