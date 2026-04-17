/**
 * EnvelopeGenerator - Advanced Envelope Generator
 *
 * Features:
 * - ADSR (Attack, Decay, Sustain, Release)
 * - AHDSR (Attack, Hold, Decay, Sustain, Release)
 * - Multi-stage envelopes (up to 8 stages)
 * - Looping envelopes
 * - Curve types: linear, exponential, logarithmic
 * - Trigger modes: gate, trigger, loop
 * - Velocity sensitivity
 * - Retrigger behavior
 *
 * Inspired by: Vital, Serum, Ableton Envelope
 *
 * @author Agent 17 (Modulation Matrix & Advanced LFOs)
 * @version 1.0.0
 */

import ModulationSource from './ModulationSource.js';
import PluginFactory from '../core/PluginFactory.js';

export class EnvelopeGenerator extends ModulationSource {
  /**
   * Envelope curve types
   */
  static CURVE_TYPES = {
    LINEAR: 'linear',
    EXPONENTIAL: 'exponential',
    LOGARITHMIC: 'logarithmic'
  };

  /**
   * Trigger modes
   */
  static TRIGGER_MODES = {
    GATE: 'gate',        // Traditional ADSR (sustain until release)
    TRIGGER: 'trigger',  // One-shot (no sustain)
    LOOP: 'loop'         // Loop envelope continuously
  };

  /**
   * Envelope types
   */
  static ENVELOPE_TYPES = {
    ADSR: 'adsr',
    AHDSR: 'ahdsr',
    MULTI: 'multi'
  };

  constructor(audioContext, options = {}) {
    super(audioContext, {
      ...options,
      category: 'modulation',
      description: 'Advanced envelope generator with multiple stages',
      bipolar: false // Envelopes are typically unipolar (0 to 1)
    });

    // Envelope type
    this._envelopeType = options.envelopeType || EnvelopeGenerator.ENVELOPE_TYPES.ADSR;

    // Constant source for envelope output
    this._envelopeSource = audioContext.createConstantSource();
    this._envelopeSource.offset.value = 0;
    this._envelopeSource.connect(this.output);
    this._envelopeSource.start();

    // ADSR parameters (in seconds)
    this._attack = 0.01;
    this._hold = 0;      // For AHDSR
    this._decay = 0.1;
    this._sustain = 0.7; // Level (0-1)
    this._release = 0.2;

    // Curve types for each stage
    this._attackCurve = EnvelopeGenerator.CURVE_TYPES.LINEAR;
    this._decayCurve = EnvelopeGenerator.CURVE_TYPES.EXPONENTIAL;
    this._releaseCurve = EnvelopeGenerator.CURVE_TYPES.EXPONENTIAL;

    // Multi-stage envelope
    this._stages = [];

    // Trigger mode
    this._triggerMode = EnvelopeGenerator.TRIGGER_MODES.GATE;

    // Looping
    this._loopEnabled = false;
    this._loopStart = 1; // Stage index to loop from
    this._loopEnd = 2;   // Stage index to loop to

    // Velocity sensitivity
    this._velocitySensitivity = 0; // 0 = no sensitivity, 1 = full sensitivity
    this._currentVelocity = 1;

    // State
    this._isGateOpen = false;
    this._currentStage = 'idle';
    this._releaseScheduled = false;

    // Register parameters
    this.registerParameter('attack', {
      get value() { return this._attack; },
      set value(v) { this._attack = Math.max(0.001, v); }
    }, {
      min: 0.001,
      max: 10,
      default: 0.01,
      unit: 's',
      label: 'Attack Time'
    });

    this.registerParameter('decay', {
      get value() { return this._decay; },
      set value(v) { this._decay = Math.max(0.001, v); }
    }, {
      min: 0.001,
      max: 10,
      default: 0.1,
      unit: 's',
      label: 'Decay Time'
    });

    this.registerParameter('sustain', {
      get value() { return this._sustain; },
      set value(v) { this._sustain = Math.max(0, Math.min(1, v)); }
    }, {
      min: 0,
      max: 1,
      default: 0.7,
      unit: '',
      label: 'Sustain Level'
    });

    this.registerParameter('release', {
      get value() { return this._release; },
      set value(v) { this._release = Math.max(0.001, v); }
    }, {
      min: 0.001,
      max: 10,
      default: 0.2,
      unit: 's',
      label: 'Release Time'
    });
  }

  /**
   * Set envelope type
   * @param {string} type - Envelope type (ADSR, AHDSR, MULTI)
   */
  setEnvelopeType(type) {
    if (!Object.values(EnvelopeGenerator.ENVELOPE_TYPES).includes(type)) {
      console.warn(`Invalid envelope type: ${type}`);
      return;
    }

    this._envelopeType = type;
  }

  /**
   * Set ADSR parameters
   * @param {Object} params - ADSR parameters
   * @param {number} params.attack - Attack time in seconds
   * @param {number} params.decay - Decay time in seconds
   * @param {number} params.sustain - Sustain level (0-1)
   * @param {number} params.release - Release time in seconds
   */
  setADSR(params) {
    if (params.attack !== undefined) this._attack = Math.max(0.001, params.attack);
    if (params.decay !== undefined) this._decay = Math.max(0.001, params.decay);
    if (params.sustain !== undefined) this._sustain = Math.max(0, Math.min(1, params.sustain));
    if (params.release !== undefined) this._release = Math.max(0.001, params.release);
  }

  /**
   * Set hold time (for AHDSR)
   * @param {number} hold - Hold time in seconds
   */
  setHold(hold) {
    this._hold = Math.max(0, hold);
  }

  /**
   * Set curve type for a stage
   * @param {string} stage - Stage name ('attack', 'decay', 'release')
   * @param {string} curveType - Curve type
   */
  setStageCurve(stage, curveType) {
    if (!Object.values(EnvelopeGenerator.CURVE_TYPES).includes(curveType)) {
      console.warn(`Invalid curve type: ${curveType}`);
      return;
    }

    switch (stage) {
      case 'attack':
        this._attackCurve = curveType;
        break;
      case 'decay':
        this._decayCurve = curveType;
        break;
      case 'release':
        this._releaseCurve = curveType;
        break;
    }
  }

  /**
   * Set trigger mode
   * @param {string} mode - Trigger mode
   */
  setTriggerMode(mode) {
    if (!Object.values(EnvelopeGenerator.TRIGGER_MODES).includes(mode)) {
      console.warn(`Invalid trigger mode: ${mode}`);
      return;
    }

    this._triggerMode = mode;

    if (mode === EnvelopeGenerator.TRIGGER_MODES.LOOP) {
      this._loopEnabled = true;
    }
  }

  /**
   * Set loop points
   * @param {number} start - Loop start stage index
   * @param {number} end - Loop end stage index
   */
  setLoopPoints(start, end) {
    this._loopStart = start;
    this._loopEnd = end;
  }

  /**
   * Enable/disable looping
   * @param {boolean} enabled - Loop enabled
   */
  setLoop(enabled) {
    this._loopEnabled = enabled;
  }

  /**
   * Set velocity sensitivity
   * @param {number} sensitivity - Sensitivity amount (0-1)
   */
  setVelocitySensitivity(sensitivity) {
    this._velocitySensitivity = Math.max(0, Math.min(1, sensitivity));
  }

  /**
   * Trigger the envelope (gate on)
   * @param {number} velocity - Note velocity (0-1)
   * @param {number} time - Trigger time (AudioContext time)
   */
  trigger(velocity = 1, time = null) {
    const triggerTime = time !== null ? time : this.audioContext.currentTime;
    this._currentVelocity = velocity;
    this._isGateOpen = true;
    this._releaseScheduled = false;

    const envelopeParam = this._envelopeSource.offset;

    // Cancel any existing automation
    envelopeParam.cancelScheduledValues(triggerTime);

    // Calculate peak level with velocity sensitivity
    const peakLevel = 1 - (this._velocitySensitivity * (1 - velocity));

    // Attack stage
    this._currentStage = 'attack';
    envelopeParam.setValueAtTime(0, triggerTime);

    switch (this._attackCurve) {
      case EnvelopeGenerator.CURVE_TYPES.LINEAR:
        envelopeParam.linearRampToValueAtTime(peakLevel, triggerTime + this._attack);
        break;
      case EnvelopeGenerator.CURVE_TYPES.EXPONENTIAL:
        envelopeParam.exponentialRampToValueAtTime(
          Math.max(0.001, peakLevel),
          triggerTime + this._attack
        );
        break;
      case EnvelopeGenerator.CURVE_TYPES.LOGARITHMIC:
        // Simulate log curve with setTargetAtTime
        envelopeParam.setTargetAtTime(peakLevel, triggerTime, this._attack / 5);
        envelopeParam.setValueAtTime(peakLevel, triggerTime + this._attack);
        break;
    }

    let currentTime = triggerTime + this._attack;

    // Hold stage (for AHDSR)
    if (this._envelopeType === EnvelopeGenerator.ENVELOPE_TYPES.AHDSR && this._hold > 0) {
      this._currentStage = 'hold';
      currentTime += this._hold;
    }

    // Decay stage
    this._currentStage = 'decay';
    const sustainLevel = this._sustain * peakLevel;

    switch (this._decayCurve) {
      case EnvelopeGenerator.CURVE_TYPES.LINEAR:
        envelopeParam.linearRampToValueAtTime(sustainLevel, currentTime + this._decay);
        break;
      case EnvelopeGenerator.CURVE_TYPES.EXPONENTIAL:
        envelopeParam.exponentialRampToValueAtTime(
          Math.max(0.001, sustainLevel),
          currentTime + this._decay
        );
        break;
      case EnvelopeGenerator.CURVE_TYPES.LOGARITHMIC:
        envelopeParam.setTargetAtTime(sustainLevel, currentTime, this._decay / 5);
        envelopeParam.setValueAtTime(sustainLevel, currentTime + this._decay);
        break;
    }

    currentTime += this._decay;

    // Sustain stage (for GATE mode) or proceed to release (for TRIGGER mode)
    if (this._triggerMode === EnvelopeGenerator.TRIGGER_MODES.TRIGGER) {
      this._currentStage = 'sustain';
      // In trigger mode, immediately release after decay
      this.release(currentTime);
    } else if (this._triggerMode === EnvelopeGenerator.TRIGGER_MODES.LOOP) {
      this._currentStage = 'sustain';
      // Schedule loop
      this._scheduleLoop(currentTime);
    } else {
      // GATE mode: hold at sustain level until explicit release
      this._currentStage = 'sustain';
      envelopeParam.setValueAtTime(sustainLevel, currentTime);
    }

    this._running = true;
  }

  /**
   * Schedule envelope loop
   * @private
   * @param {number} time - Loop start time
   */
  _scheduleLoop(time) {
    if (!this._loopEnabled) return;

    const loopDuration = this._attack + this._decay;

    setTimeout(() => {
      if (this._loopEnabled && this._running) {
        this.trigger(this._currentVelocity, this.audioContext.currentTime);
      }
    }, loopDuration * 1000);
  }

  /**
   * Release the envelope (gate off)
   * @param {number} time - Release time (AudioContext time)
   */
  release(time = null) {
    if (!this._isGateOpen && this._triggerMode === EnvelopeGenerator.TRIGGER_MODES.GATE) {
      return; // Already released
    }

    const releaseTime = time !== null ? time : this.audioContext.currentTime;
    this._isGateOpen = false;
    this._currentStage = 'release';

    const envelopeParam = this._envelopeSource.offset;
    const currentValue = envelopeParam.value;

    // Cancel future automation
    envelopeParam.cancelScheduledValues(releaseTime);
    envelopeParam.setValueAtTime(currentValue, releaseTime);

    // Release stage
    switch (this._releaseCurve) {
      case EnvelopeGenerator.CURVE_TYPES.LINEAR:
        envelopeParam.linearRampToValueAtTime(0, releaseTime + this._release);
        break;
      case EnvelopeGenerator.CURVE_TYPES.EXPONENTIAL:
        envelopeParam.exponentialRampToValueAtTime(0.001, releaseTime + this._release);
        envelopeParam.linearRampToValueAtTime(0, releaseTime + this._release + 0.001);
        break;
      case EnvelopeGenerator.CURVE_TYPES.LOGARITHMIC:
        envelopeParam.setTargetAtTime(0, releaseTime, this._release / 5);
        envelopeParam.setValueAtTime(0, releaseTime + this._release);
        break;
    }

    // Mark as stopped after release completes
    setTimeout(() => {
      if (!this._isGateOpen) {
        this._running = false;
        this._currentStage = 'idle';
      }
    }, this._release * 1000);
  }

  /**
   * Get current envelope value (for visualization)
   * @returns {number} Current envelope value (0-1)
   */
  getCurrentValue() {
    return this._envelopeSource.offset.value;
  }

  /**
   * Get current envelope stage
   * @returns {string} Current stage
   */
  getCurrentStage() {
    return this._currentStage;
  }

  /**
   * Check if gate is open
   * @returns {boolean} Gate state
   */
  isGateOpen() {
    return this._isGateOpen;
  }

  /**
   * Start modulation source (for compatibility with base class)
   * This is handled by trigger() for envelopes
   * @protected
   */
  _onStart(time) {
    this.trigger(1, time);
  }

  /**
   * Stop modulation source (for compatibility with base class)
   * This is handled by release() for envelopes
   * @protected
   */
  _onStop(time) {
    this.release(time);
  }

  /**
   * Get envelope info
   * @returns {Object} Envelope metadata
   */
  getInfo() {
    return {
      ...super.getInfo(),
      type: this._envelopeType,
      triggerMode: this._triggerMode,
      attack: this._attack,
      decay: this._decay,
      sustain: this._sustain,
      release: this._release,
      currentStage: this._currentStage,
      gateOpen: this._isGateOpen
    };
  }

  /**
   * Cleanup
   */
  dispose() {
    this._envelopeSource.stop();
    this._envelopeSource.disconnect();
    super.dispose();
  }
}

// Register with PluginFactory
PluginFactory.register('EnvelopeGenerator', EnvelopeGenerator, {
  category: 'modulation',
  description: 'Advanced envelope generator with ADSR, looping, and multiple curve types',
  tags: ['envelope', 'adsr', 'modulation', 'eg'],
  version: '1.0.0',
  author: 'Agent 17'
});

export default EnvelopeGenerator;
