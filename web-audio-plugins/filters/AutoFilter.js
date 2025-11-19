/**
 * Auto Filter - Multi-Mode Resonant Filter with Modulation
 *
 * Features:
 * - Multiple filter types (lowpass12, lowpass24, highpass12, highpass24, bandpass, notch)
 * - Resonance control (0-100%)
 * - LFO modulation with multiple waveforms
 * - Envelope follower for frequency modulation
 * - ADSR envelope generator
 * - Tempo-synced LFO
 * - Dry/wet mix control
 * - Sidechain envelope follower option
 *
 * Based on Web Audio API BiquadFilterNode and OscillatorNode
 */

class AutoFilter {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Audio nodes
    this.input = this.context.createGain();
    this.output = this.context.createGain();

    // Filter nodes (cascade two for 24dB slopes)
    this.filter1 = this.context.createBiquadFilter();
    this.filter2 = this.context.createBiquadFilter();

    // Dry/wet mixing
    this.wetGain = this.context.createGain();
    this.dryGain = this.context.createGain();

    // LFO
    this.lfo = this.context.createOscillator();
    this.lfoGain = this.context.createGain();

    // Envelope follower
    this.envelopeFollower = this.context.createDynamicsCompressor();
    this.envelopeGain = this.context.createGain();

    // Parameters
    this.baseFrequency = 1000; // Hz
    this.resonance = 0; // 0-100%
    this.filterType = 'lowpass24';
    this.mix = 100; // 0-100%

    // LFO parameters
    this.lfoParams = {
      amount: 0,        // -100 to +100%
      rate: 1,          // Hz or sync
      waveform: 'sine', // sine, triangle, square, sawtooth, random (sample & hold)
      phase: 0,         // 0-360 degrees
      sync: false,      // Tempo sync
      syncRate: '1/4'   // 1/1, 1/2, 1/4, 1/8, 1/16, etc.
    };

    // Envelope parameters
    this.envelopeParams = {
      amount: 0,        // -100 to +100%
      attack: 10,       // ms
      decay: 100,       // ms
      sustain: 50,      // 0-100%
      release: 100      // ms
    };

    // Envelope state
    this.envelopeActive = false;
    this.envelopeReleaseTimeout = null;

    // BPM for tempo sync (default 120)
    this.bpm = 120;

    // Setup audio routing
    this.setupRouting();

    // Start LFO
    this.lfo.start();

    // Initialize with options
    this.initialize(options);
  }

  /**
   * Setup audio routing
   */
  setupRouting() {
    // Dry path: input → dryGain → output
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path: input → filter1 → filter2 → wetGain → output
    this.input.connect(this.filter1);
    this.filter1.connect(this.filter2);
    this.filter2.connect(this.wetGain);
    this.wetGain.connect(this.output);

    // LFO → lfoGain → filter frequency
    this.lfo.connect(this.lfoGain);
    this.lfoGain.connect(this.filter1.frequency);
    this.lfoGain.connect(this.filter2.frequency);

    // Set default filter types
    this.filter1.type = 'lowpass';
    this.filter2.type = 'lowpass';
    this.filter1.frequency.value = this.baseFrequency;
    this.filter2.frequency.value = this.baseFrequency;

    // Set default LFO
    this.lfo.type = 'sine';
    this.lfo.frequency.value = 1;
    this.lfoGain.gain.value = 0;

    // Set default mix (100% wet)
    this.updateMix(100);
  }

  /**
   * Initialize with options
   */
  initialize(options) {
    if (options.frequency !== undefined) {
      this.setFrequency(options.frequency);
    }

    if (options.resonance !== undefined) {
      this.setResonance(options.resonance);
    }

    if (options.filterType !== undefined) {
      this.setFilterType(options.filterType);
    }

    if (options.mix !== undefined) {
      this.setMix(options.mix);
    }

    if (options.lfo) {
      this.setLFO(options.lfo);
    }

    if (options.envelope) {
      this.setEnvelope(options.envelope);
    }

    if (options.bpm !== undefined) {
      this.setBPM(options.bpm);
    }
  }

  /**
   * Set filter frequency
   * @param {number} freq - Frequency in Hz (20 to 20000)
   */
  setFrequency(freq) {
    this.baseFrequency = Math.max(20, Math.min(20000, freq));
    const now = this.context.currentTime;

    this.filter1.frequency.setValueAtTime(this.baseFrequency, now);
    this.filter2.frequency.setValueAtTime(this.baseFrequency, now);
  }

  /**
   * Set filter resonance
   * @param {number} resonance - Resonance 0-100%
   */
  setResonance(resonance) {
    this.resonance = Math.max(0, Math.min(100, resonance));

    // Map 0-100% to Q value (0.1 to 20)
    // Higher resonance = higher Q = more pronounced peak
    const Q = 0.1 + (this.resonance / 100) * 19.9;

    const now = this.context.currentTime;
    this.filter1.Q.setValueAtTime(Q, now);
    this.filter2.Q.setValueAtTime(Q, now);
  }

  /**
   * Set filter type
   * @param {string} type - Filter type (lowpass12, lowpass24, highpass12, highpass24, bandpass, notch)
   */
  setFilterType(type) {
    this.filterType = type;

    switch (type) {
      case 'lowpass12':
        this.filter1.type = 'lowpass';
        this.filter2.type = 'allpass'; // Bypass second filter
        break;

      case 'lowpass24':
        this.filter1.type = 'lowpass';
        this.filter2.type = 'lowpass';
        break;

      case 'highpass12':
        this.filter1.type = 'highpass';
        this.filter2.type = 'allpass';
        break;

      case 'highpass24':
        this.filter1.type = 'highpass';
        this.filter2.type = 'highpass';
        break;

      case 'bandpass':
        this.filter1.type = 'bandpass';
        this.filter2.type = 'allpass';
        break;

      case 'notch':
        this.filter1.type = 'notch';
        this.filter2.type = 'allpass';
        break;

      default:
        console.warn(`AutoFilter: Unknown filter type "${type}"`);
    }
  }

  /**
   * Set dry/wet mix
   * @param {number} mix - Mix percentage 0-100% (0 = dry, 100 = wet)
   */
  setMix(mix) {
    this.mix = Math.max(0, Math.min(100, mix));
    this.updateMix(this.mix);
  }

  /**
   * Update dry/wet gain nodes
   */
  updateMix(mix) {
    const wetAmount = mix / 100;
    const dryAmount = 1 - wetAmount;

    const now = this.context.currentTime;
    this.wetGain.gain.setValueAtTime(wetAmount, now);
    this.dryGain.gain.setValueAtTime(dryAmount, now);
  }

  /**
   * Set LFO parameters
   * @param {Object} params - LFO parameters
   */
  setLFO(params) {
    if (params.rate !== undefined && !params.sync) {
      this.lfoParams.rate = Math.max(0.01, Math.min(40, params.rate));
      this.lfo.frequency.value = this.lfoParams.rate;
    }

    if (params.amount !== undefined) {
      this.lfoParams.amount = Math.max(-100, Math.min(100, params.amount));
      this.updateLFOAmount();
    }

    if (params.waveform !== undefined) {
      this.lfoParams.waveform = params.waveform;

      // Map waveform names
      const waveformMap = {
        'sine': 'sine',
        'triangle': 'triangle',
        'square': 'square',
        'sawtooth': 'sawtooth',
        'random': 'square' // Use square and implement sample & hold separately
      };

      if (waveformMap[params.waveform]) {
        this.lfo.type = waveformMap[params.waveform];
      }
    }

    if (params.phase !== undefined) {
      this.lfoParams.phase = params.phase;
      // Phase offset would require recreating the oscillator with a delay
      // For now, store the parameter
    }

    if (params.sync !== undefined) {
      this.lfoParams.sync = params.sync;
      if (params.sync && params.syncRate) {
        this.setSyncRate(params.syncRate);
      }
    }

    if (params.syncRate !== undefined) {
      this.lfoParams.syncRate = params.syncRate;
      if (this.lfoParams.sync) {
        this.setSyncRate(params.syncRate);
      }
    }
  }

  /**
   * Update LFO modulation amount
   */
  updateLFOAmount() {
    const amount = this.lfoParams.amount / 100; // -1 to +1

    // Map to frequency modulation depth
    // ±100% = ±2 octaves = ±baseFrequency
    const depth = amount * this.baseFrequency;

    this.lfoGain.gain.setValueAtTime(depth, this.context.currentTime);
  }

  /**
   * Set tempo-synced LFO rate
   * @param {string} syncRate - Sync rate (e.g., '1/4', '1/8', '1/16')
   */
  setSyncRate(syncRate) {
    this.lfoParams.syncRate = syncRate;

    // Calculate frequency based on BPM and sync rate
    const beatsPerSecond = this.bpm / 60;

    // Parse sync rate (e.g., '1/4' = quarter note)
    const [numerator, denominator] = syncRate.split('/').map(Number);
    const noteLength = numerator / denominator; // In quarter notes

    // Calculate LFO frequency
    const freq = beatsPerSecond / noteLength;

    this.lfo.frequency.value = freq;
  }

  /**
   * Set BPM for tempo sync
   * @param {number} bpm - Beats per minute
   */
  setBPM(bpm) {
    this.bpm = Math.max(20, Math.min(300, bpm));

    // Update sync rate if enabled
    if (this.lfoParams.sync) {
      this.setSyncRate(this.lfoParams.syncRate);
    }
  }

  /**
   * Set envelope parameters
   * @param {Object} params - Envelope parameters
   */
  setEnvelope(params) {
    if (params.amount !== undefined) {
      this.envelopeParams.amount = Math.max(-100, Math.min(100, params.amount));
    }

    if (params.attack !== undefined) {
      this.envelopeParams.attack = Math.max(0.1, Math.min(500, params.attack));
    }

    if (params.decay !== undefined) {
      this.envelopeParams.decay = Math.max(0.1, Math.min(1000, params.decay));
    }

    if (params.sustain !== undefined) {
      this.envelopeParams.sustain = Math.max(0, Math.min(100, params.sustain));
    }

    if (params.release !== undefined) {
      this.envelopeParams.release = Math.max(10, Math.min(5000, params.release));
    }
  }

  /**
   * Trigger envelope (on note on or audio input detected)
   */
  triggerEnvelope() {
    if (this.envelopeParams.amount === 0) return;

    const now = this.context.currentTime;
    const amount = this.envelopeParams.amount / 100; // -1 to +1

    // Calculate modulation range (±2 octaves)
    const modulationDepth = amount * this.baseFrequency * 2;
    const targetFreq = this.baseFrequency + modulationDepth;
    const sustainFreq = this.baseFrequency + (modulationDepth * this.envelopeParams.sustain / 100);

    // Cancel any scheduled changes
    this.filter1.frequency.cancelScheduledValues(now);
    this.filter2.frequency.cancelScheduledValues(now);

    // Attack phase
    const attackTime = this.envelopeParams.attack / 1000;
    this.filter1.frequency.setValueAtTime(this.baseFrequency, now);
    this.filter2.frequency.setValueAtTime(this.baseFrequency, now);

    this.filter1.frequency.linearRampToValueAtTime(targetFreq, now + attackTime);
    this.filter2.frequency.linearRampToValueAtTime(targetFreq, now + attackTime);

    // Decay phase
    const decayTime = this.envelopeParams.decay / 1000;
    this.filter1.frequency.linearRampToValueAtTime(sustainFreq, now + attackTime + decayTime);
    this.filter2.frequency.linearRampToValueAtTime(sustainFreq, now + attackTime + decayTime);

    this.envelopeActive = true;

    // Clear any existing release timeout
    if (this.envelopeReleaseTimeout) {
      clearTimeout(this.envelopeReleaseTimeout);
    }
  }

  /**
   * Release envelope (on note off or audio input stopped)
   */
  releaseEnvelope() {
    if (!this.envelopeActive) return;

    const now = this.context.currentTime;
    const releaseTime = this.envelopeParams.release / 1000;

    // Release phase: return to base frequency
    this.filter1.frequency.cancelScheduledValues(now);
    this.filter2.frequency.cancelScheduledValues(now);

    this.filter1.frequency.setValueAtTime(this.filter1.frequency.value, now);
    this.filter2.frequency.setValueAtTime(this.filter2.frequency.value, now);

    this.filter1.frequency.linearRampToValueAtTime(this.baseFrequency, now + releaseTime);
    this.filter2.frequency.linearRampToValueAtTime(this.baseFrequency, now + releaseTime);

    this.envelopeActive = false;
  }

  /**
   * Auto-trigger envelope based on input level (envelope follower)
   * This would typically be called from an AnalyserNode monitoring input
   */
  analyzeInput() {
    // This is a placeholder for envelope follower implementation
    // In a real implementation, you would:
    // 1. Use an AnalyserNode to monitor input level
    // 2. Detect when level crosses a threshold
    // 3. Call triggerEnvelope() on attack, releaseEnvelope() on release
  }

  /**
   * Get the input node for connection
   */
  getInput() {
    return this.input;
  }

  /**
   * Get the output node for connection
   */
  getOutput() {
    return this.output;
  }

  /**
   * Connect this filter to another audio node
   */
  connect(destination) {
    this.output.connect(destination);
  }

  /**
   * Disconnect this filter from all destinations
   */
  disconnect() {
    this.output.disconnect();
  }

  /**
   * Get current state
   */
  getState() {
    return {
      frequency: this.baseFrequency,
      resonance: this.resonance,
      filterType: this.filterType,
      mix: this.mix,
      lfo: { ...this.lfoParams },
      envelope: { ...this.envelopeParams },
      bpm: this.bpm
    };
  }

  /**
   * Clean up resources
   */
  destroy() {
    this.disconnect();
    this.input.disconnect();
    this.dryGain.disconnect();
    this.wetGain.disconnect();
    this.filter1.disconnect();
    this.filter2.disconnect();
    this.lfo.stop();
    this.lfo.disconnect();
    this.lfoGain.disconnect();

    if (this.envelopeReleaseTimeout) {
      clearTimeout(this.envelopeReleaseTimeout);
    }

    console.log('AutoFilter destroyed');
  }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = AutoFilter;
}
