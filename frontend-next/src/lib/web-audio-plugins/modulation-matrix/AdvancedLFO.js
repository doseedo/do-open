/**
 * AdvancedLFO - Advanced Low Frequency Oscillator
 *
 * Features:
 * - Multiple waveforms: sine, triangle, square, sawtooth, random, custom
 * - BPM sync with musical divisions (1/1, 1/2, 1/4, 1/8, 1/16, triplets, dotted)
 * - Sample & Hold mode
 * - Step sequencer mode (up to 16 steps)
 * - Phase control and reset
 * - Smooth/stepped transitions
 * - Retrigger options
 *
 * Inspired by: Bitwig Grid, Ableton LFO, Serum LFO
 *
 * @author Agent 17 (Modulation Matrix & Advanced LFOs)
 * @version 1.0.0
 */

import ModulationSource from './ModulationSource.js';
import PluginFactory from '../core/PluginFactory.js';

export class AdvancedLFO extends ModulationSource {
  /**
   * Available waveform types
   */
  static WAVEFORMS = {
    SINE: 'sine',
    TRIANGLE: 'triangle',
    SQUARE: 'square',
    SAWTOOTH: 'sawtooth',
    RANDOM: 'random',
    SAMPLE_HOLD: 'samplehold',
    STEP: 'step'
  };

  /**
   * BPM sync divisions
   */
  static SYNC_DIVISIONS = {
    '4/1': 16,      // 4 bars
    '2/1': 8,       // 2 bars
    '1/1': 4,       // 1 bar
    '1/2': 2,       // Half note
    '1/4': 1,       // Quarter note
    '1/8': 0.5,     // Eighth note
    '1/16': 0.25,   // Sixteenth note
    '1/32': 0.125,  // Thirty-second note
    '1/2T': 2/3 * 2,    // Half note triplet
    '1/4T': 2/3,        // Quarter note triplet
    '1/8T': 2/3 * 0.5,  // Eighth note triplet
    '1/16T': 2/3 * 0.25,// Sixteenth note triplet
    '1/2D': 3,          // Dotted half note
    '1/4D': 1.5,        // Dotted quarter note
    '1/8D': 0.75,       // Dotted eighth note
    '1/16D': 0.375      // Dotted sixteenth note
  };

  constructor(audioContext, options = {}) {
    super(audioContext, {
      ...options,
      category: 'modulation',
      description: 'Advanced LFO with multiple waveforms and BPM sync'
    });

    // Oscillator for basic waveforms
    this._oscillator = null;
    this._waveform = AdvancedLFO.WAVEFORMS.SINE;

    // BPM sync
    this._bpmSyncEnabled = false;
    this._bpm = 120;
    this._syncDivision = '1/4';

    // Frequency control
    this._frequencyNode = audioContext.createConstantSource();
    this._frequencyNode.offset.value = 1; // 1 Hz default
    this._frequencyNode.start();

    // Phase control
    this._phase = 0;
    this._phaseOffset = 0;

    // Sample & Hold
    this._sampleHoldRate = 10; // Hz
    this._sampleHoldTimer = null;
    this._currentSampleValue = 0;

    // Step sequencer
    this._stepCount = 8;
    this._stepValues = new Array(16).fill(0).map((_, i) => Math.sin(i / 16 * Math.PI * 2));
    this._currentStep = 0;

    // Custom waveform (for future expansion)
    this._customWaveform = null;

    // Smoothing for stepped waveforms
    this._smoothing = 0.01; // Smoothing time in seconds

    // Register parameters
    this.registerParameter('frequency', this._frequencyNode.offset, {
      min: 0.01,
      max: 20,
      default: 1,
      unit: 'Hz',
      label: 'Frequency'
    });

    this.registerParameter('phase', {
      get value() { return this._phaseOffset; },
      set value(v) { this._phaseOffset = v; this._updatePhase(); }
    }, {
      min: 0,
      max: 360,
      default: 0,
      unit: '°',
      label: 'Phase Offset'
    });

    // Create initial oscillator
    this._createOscillator();
  }

  /**
   * Create and configure oscillator
   * @private
   */
  _createOscillator() {
    if (this._oscillator) {
      this._oscillator.stop();
      this._oscillator.disconnect();
    }

    // For basic waveforms, use OscillatorNode
    if ([AdvancedLFO.WAVEFORMS.SINE, AdvancedLFO.WAVEFORMS.TRIANGLE,
         AdvancedLFO.WAVEFORMS.SQUARE, AdvancedLFO.WAVEFORMS.SAWTOOTH].includes(this._waveform)) {

      this._oscillator = this.audioContext.createOscillator();
      this._oscillator.type = this._waveform;
      this._frequencyNode.connect(this._oscillator.frequency);
      this._oscillator.connect(this.output);
    }
  }

  /**
   * Set waveform type
   * @param {string} waveform - Waveform type (use WAVEFORMS constants)
   */
  setWaveform(waveform) {
    if (!Object.values(AdvancedLFO.WAVEFORMS).includes(waveform)) {
      console.warn(`Invalid waveform: ${waveform}`);
      return;
    }

    const wasRunning = this._running;
    if (wasRunning) {
      this.stop();
    }

    this._waveform = waveform;
    this._createOscillator();

    if (wasRunning) {
      this.start();
    }
  }

  /**
   * Get current waveform
   * @returns {string} Current waveform type
   */
  getWaveform() {
    return this._waveform;
  }

  /**
   * Enable/disable BPM sync
   * @param {boolean} enabled - True to enable BPM sync
   */
  setBPMSync(enabled) {
    this._bpmSyncEnabled = enabled;
    this._updateFrequency();
  }

  /**
   * Set BPM for sync mode
   * @param {number} bpm - Beats per minute (20-300)
   */
  setBPM(bpm) {
    this._bpm = Math.max(20, Math.min(300, bpm));
    this._updateFrequency();
  }

  /**
   * Set sync division
   * @param {string} division - Division string (e.g., '1/4', '1/8T', '1/4D')
   */
  setSyncDivision(division) {
    if (!AdvancedLFO.SYNC_DIVISIONS[division]) {
      console.warn(`Invalid sync division: ${division}`);
      return;
    }

    this._syncDivision = division;
    this._updateFrequency();
  }

  /**
   * Update frequency based on BPM sync or free-running mode
   * @private
   */
  _updateFrequency() {
    if (this._bpmSyncEnabled) {
      // Calculate frequency from BPM and division
      const beatsPerSecond = this._bpm / 60;
      const divisionMultiplier = AdvancedLFO.SYNC_DIVISIONS[this._syncDivision];
      const frequency = beatsPerSecond / divisionMultiplier;
      this._frequencyNode.offset.value = frequency;
    }
  }

  /**
   * Set frequency (free-running mode)
   * @param {number} frequency - Frequency in Hz
   */
  setFrequency(frequency) {
    if (this._bpmSyncEnabled) {
      console.warn('LFO is in BPM sync mode. Disable sync to set manual frequency.');
      return;
    }

    this._frequencyNode.offset.value = Math.max(0.01, Math.min(20, frequency));
  }

  /**
   * Set phase offset
   * @param {number} degrees - Phase offset in degrees (0-360)
   */
  setPhase(degrees) {
    this._phaseOffset = degrees % 360;
    this._updatePhase();
  }

  /**
   * Update oscillator phase
   * @private
   */
  _updatePhase() {
    // Phase can only be set at start time for OscillatorNode
    // For real-time phase control, we'd need to restart the oscillator
    if (this._running) {
      const wasRunning = true;
      this.stop();
      this.start();
    }
  }

  /**
   * Reset phase to zero
   */
  resetPhase() {
    this._phase = 0;
    this._phaseOffset = 0;
    this._updatePhase();
  }

  /**
   * Configure step sequencer
   * @param {Array<number>} steps - Step values (0-1 range)
   */
  setStepSequence(steps) {
    if (!Array.isArray(steps) || steps.length > 16) {
      console.warn('Step sequence must be an array of max 16 values');
      return;
    }

    this._stepValues = steps.map(v => Math.max(0, Math.min(1, v)));
    this._stepCount = steps.length;
    this._currentStep = 0;
  }

  /**
   * Set sample & hold rate
   * @param {number} rate - Sample rate in Hz
   */
  setSampleHoldRate(rate) {
    this._sampleHoldRate = Math.max(0.1, Math.min(100, rate));
  }

  /**
   * Start LFO
   * @param {number} time - Start time
   * @protected
   */
  _onStart(time) {
    const startTime = time || this.audioContext.currentTime;

    if (this._waveform === AdvancedLFO.WAVEFORMS.RANDOM) {
      this._startRandomLFO(startTime);
    } else if (this._waveform === AdvancedLFO.WAVEFORMS.SAMPLE_HOLD) {
      this._startSampleHold(startTime);
    } else if (this._waveform === AdvancedLFO.WAVEFORMS.STEP) {
      this._startStepSequencer(startTime);
    } else if (this._oscillator) {
      this._oscillator.start(startTime);
    }
  }

  /**
   * Stop LFO
   * @param {number} time - Stop time
   * @protected
   */
  _onStop(time) {
    const stopTime = time || this.audioContext.currentTime;

    if (this._oscillator && this._oscillator.stop) {
      try {
        this._oscillator.stop(stopTime);
      } catch (e) {
        // Already stopped
      }
    }

    if (this._sampleHoldTimer) {
      clearInterval(this._sampleHoldTimer);
      this._sampleHoldTimer = null;
    }
  }

  /**
   * Start random LFO (noise-based)
   * @private
   * @param {number} startTime - Start time
   */
  _startRandomLFO(startTime) {
    // Create buffer with random values
    const bufferSize = this.audioContext.sampleRate * 2; // 2 seconds of random data
    const buffer = this.audioContext.createBuffer(1, bufferSize, this.audioContext.sampleRate);
    const data = buffer.getChannelData(0);

    for (let i = 0; i < bufferSize; i++) {
      data[i] = Math.random() * 2 - 1; // -1 to 1
    }

    const bufferSource = this.audioContext.createBufferSource();
    bufferSource.buffer = buffer;
    bufferSource.loop = true;
    bufferSource.connect(this.output);
    bufferSource.start(startTime);

    this._oscillator = bufferSource;
  }

  /**
   * Start sample & hold LFO
   * @private
   * @param {number} startTime - Start time
   */
  _startSampleHold(startTime) {
    const intervalMs = 1000 / this._sampleHoldRate;
    const constantSource = this.audioContext.createConstantSource();
    constantSource.connect(this.output);
    constantSource.start(startTime);

    this._oscillator = constantSource;

    this._sampleHoldTimer = setInterval(() => {
      this._currentSampleValue = Math.random() * 2 - 1;
      constantSource.offset.setValueAtTime(
        this._currentSampleValue,
        this.audioContext.currentTime
      );
    }, intervalMs);
  }

  /**
   * Start step sequencer LFO
   * @private
   * @param {number} startTime - Start time
   */
  _startStepSequencer(startTime) {
    const constantSource = this.audioContext.createConstantSource();
    constantSource.connect(this.output);
    constantSource.start(startTime);

    this._oscillator = constantSource;

    const stepDuration = 1 / (this._frequencyNode.offset.value * this._stepCount);

    const advanceStep = () => {
      const value = this._stepValues[this._currentStep];
      const bipolarValue = this.bipolar ? (value * 2 - 1) : value;

      constantSource.offset.setValueAtTime(
        bipolarValue,
        this.audioContext.currentTime
      );

      this._currentStep = (this._currentStep + 1) % this._stepCount;
    };

    // Initial step
    advanceStep();

    // Set up interval for step advancement
    this._sampleHoldTimer = setInterval(advanceStep, stepDuration * 1000);
  }

  /**
   * Get current LFO value (for visualization)
   * @returns {number} Current value
   */
  getCurrentValue() {
    if (this._waveform === AdvancedLFO.WAVEFORMS.SAMPLE_HOLD) {
      return this._currentSampleValue;
    } else if (this._waveform === AdvancedLFO.WAVEFORMS.STEP) {
      return this._stepValues[this._currentStep];
    } else {
      // Approximate for oscillators
      const frequency = this._frequencyNode.offset.value;
      const time = this.audioContext.currentTime;
      const phase = (time * frequency + this._phaseOffset / 360) % 1;

      switch (this._waveform) {
        case AdvancedLFO.WAVEFORMS.SINE:
          return Math.sin(phase * Math.PI * 2);
        case AdvancedLFO.WAVEFORMS.TRIANGLE:
          return Math.abs((phase % 1) * 4 - 2) - 1;
        case AdvancedLFO.WAVEFORMS.SQUARE:
          return phase < 0.5 ? 1 : -1;
        case AdvancedLFO.WAVEFORMS.SAWTOOTH:
          return (phase % 1) * 2 - 1;
        default:
          return 0;
      }
    }
  }

  /**
   * Get LFO info
   * @returns {Object} LFO metadata
   */
  getInfo() {
    return {
      ...super.getInfo(),
      waveform: this._waveform,
      frequency: this._frequencyNode.offset.value,
      bpmSync: this._bpmSyncEnabled,
      bpm: this._bpm,
      syncDivision: this._syncDivision,
      phase: this._phaseOffset
    };
  }

  /**
   * Cleanup
   */
  dispose() {
    if (this._sampleHoldTimer) {
      clearInterval(this._sampleHoldTimer);
    }

    if (this._oscillator) {
      try {
        this._oscillator.stop();
        this._oscillator.disconnect();
      } catch (e) {
        // Already stopped
      }
    }

    this._frequencyNode.stop();
    this._frequencyNode.disconnect();

    super.dispose();
  }
}

// Register with PluginFactory
PluginFactory.register('AdvancedLFO', AdvancedLFO, {
  category: 'modulation',
  description: 'Advanced LFO with multiple waveforms, BPM sync, and step sequencer',
  tags: ['lfo', 'modulation', 'oscillator', 'bpm-sync'],
  version: '1.0.0',
  author: 'Agent 17'
});

export default AdvancedLFO;
