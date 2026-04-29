/**
 * TapeEmulation - Analog tape machine emulation
 *
 * @description
 * Emulates the sound characteristics of analog tape recorders including:
 * - Tape saturation and compression
 * - Wow & flutter (speed variations)
 * - Tape hiss
 * - Head bump (bass boost around 60-100Hz)
 * - High-frequency roll-off
 * - Hysteresis
 *
 * Inspired by: Studer A800, Ampex ATR-102, Sony APR-5000
 *
 * Based on research:
 * - "Modeling Analog Tape" (Parker, 2013)
 * - UAD Oxide & Ampex ATR-102 documentation
 * - Tape machine service manuals
 *
 * @example
 * const tape = new TapeEmulation(audioContext);
 * tape.setParameter('saturation', 50);
 * tape.setParameter('warmth', 70);
 * tape.setParameter('wowFlutter', 30);
 * input.connect(tape.input);
 * tape.connect(audioContext.destination);
 */

import { BasePlugin } from '../core/BasePlugin.js';
import PluginFactory from '../core/PluginFactory.js';
import AnalogModeling from './AnalogModeling.js';

export class TapeEmulation extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      ...options,
      category: 'vintage',
      description: 'Analog tape machine emulation with saturation, wow/flutter, and hiss'
    });

    this.sampleRate = audioContext.sampleRate;

    // Create audio processing chain
    this._createAudioGraph();

    // Internal state for wow/flutter and noise
    this.time = 0;
    this.previousOutput = 0;
    this.hissPhase = 0;
    this.humPhase = 0;
    this.vuLevel = 0;

    // Register parameters
    this._registerParameters();

    // Create script processor for analog modeling
    this._createProcessor();

    // Factory presets
    this.factoryPresets = this._createFactoryPresets();
  }

  /**
   * Create audio node graph
   * @private
   */
  _createAudioGraph() {
    // Input stage
    this.inputGain = this._trackNode(this.audioContext.createGain());

    // Head bump (bass boost) - biquad filter
    this.headBumpFilter = this._trackNode(this.audioContext.createBiquadFilter());
    this.headBumpFilter.type = 'lowshelf';
    this.headBumpFilter.frequency.value = 80;
    this.headBumpFilter.gain.value = 0;

    // High-frequency roll-off
    this.highCutFilter = this._trackNode(this.audioContext.createBiquadFilter());
    this.highCutFilter.type = 'lowpass';
    this.highCutFilter.frequency.value = 18000;
    this.highCutFilter.Q.value = 0.707;

    // Saturation/warmth stage (will be processed in script processor)
    this.saturationGain = this._trackNode(this.audioContext.createGain());

    // Output stage
    this.outputGain = this._trackNode(this.audioContext.createGain());
    this.outputGain.gain.value = 1.0;

    // Dry/wet mix
    this.dryGain = this._trackNode(this.audioContext.createGain());
    this.wetGain = this._trackNode(this.audioContext.createGain());
    this.dryGain.gain.value = 0.3;
    this.wetGain.gain.value = 0.7;
  }

  /**
   * Register plugin parameters
   * @private
   */
  _registerParameters() {
    // Saturation amount (0-100%)
    this.registerParameter('saturation', this.saturationGain.gain, {
      min: 0,
      max: 100,
      default: 30,
      unit: '%',
      label: 'Saturation',
      type: 'continuous'
    });

    // Warmth (affects head bump and high roll-off)
    this.registerParameter('warmth', this.headBumpFilter.gain, {
      min: 0,
      max: 100,
      default: 50,
      unit: '%',
      label: 'Warmth',
      type: 'continuous'
    });

    // Wow & flutter amount
    const wowFlutterParam = this.audioContext.createGain().gain;
    wowFlutterParam.value = 20;
    this.registerParameter('wowFlutter', wowFlutterParam, {
      min: 0,
      max: 100,
      default: 20,
      unit: '%',
      label: 'Wow & Flutter',
      type: 'continuous'
    });

    // Tape hiss amount
    const hissParam = this.audioContext.createGain().gain;
    hissParam.value = 10;
    this.registerParameter('hiss', hissParam, {
      min: 0,
      max: 100,
      default: 10,
      unit: '%',
      label: 'Tape Hiss',
      type: 'continuous'
    });

    // AC hum amount (50/60Hz)
    const humParam = this.audioContext.createGain().gain;
    humParam.value = 0;
    this.registerParameter('hum', humParam, {
      min: 0,
      max: 100,
      default: 0,
      unit: '%',
      label: 'AC Hum',
      type: 'continuous'
    });

    // Tape age (0=new, 100=very old)
    const ageParam = this.audioContext.createGain().gain;
    ageParam.value = 30;
    this.registerParameter('age', ageParam, {
      min: 0,
      max: 100,
      default: 30,
      unit: '%',
      label: 'Tape Age',
      type: 'continuous'
    });

    // Tape speed (affects frequency response)
    const speedParam = this.audioContext.createGain().gain;
    speedParam.value = 15; // 15 IPS
    this.registerParameter('speed', speedParam, {
      min: 7.5,
      max: 30,
      default: 15,
      unit: 'IPS',
      label: 'Tape Speed',
      type: 'discrete'
    });

    // Mix (dry/wet)
    this.registerParameter('mix', this.wetGain.gain, {
      min: 0,
      max: 100,
      default: 70,
      unit: '%',
      label: 'Mix',
      type: 'continuous'
    });
  }

  /**
   * Create script processor for analog modeling
   * @private
   */
  _createProcessor() {
    const bufferSize = 2048;
    this.processor = this.audioContext.createScriptProcessor(bufferSize, 2, 2);

    this.processor.onaudioprocess = (e) => {
      const inputL = e.inputBuffer.getChannelData(0);
      const inputR = e.inputBuffer.getChannelData(1);
      const outputL = e.outputBuffer.getChannelData(0);
      const outputR = e.outputBuffer.getChannelData(1);

      const saturation = this.getParameter('saturation') / 100;
      const warmth = this.getParameter('warmth') / 100;
      const wowFlutter = this.getParameter('wowFlutter') / 100;
      const hiss = this.getParameter('hiss') / 100;
      const hum = this.getParameter('hum') / 100;
      const age = this.getParameter('age') / 100;

      // Update filters based on warmth
      this.headBumpFilter.gain.value = warmth * 6; // Up to +6dB
      this.highCutFilter.frequency.value = 22000 - (warmth * 10000); // 22kHz to 12kHz

      for (let i = 0; i < inputL.length; i++) {
        // Time increment
        this.time += 1 / this.sampleRate;

        // Apply wow & flutter (pitch modulation)
        let sampleL = inputL[i];
        let sampleR = inputR[i];

        if (wowFlutter > 0.01) {
          const pitchVar = AnalogModeling.wowAndFlutter(this.time, wowFlutter);
          // Simple pitch shift approximation (delay-based)
          // In production, use a proper pitch shifter
          const delay = pitchVar / 1200; // Convert cents to delay factor
          sampleL = sampleL * (1 + delay * 0.01);
          sampleR = sampleR * (1 + delay * 0.01);
        }

        // Apply tape saturation
        sampleL = AnalogModeling.tapeSaturation(sampleL, saturation);
        sampleR = AnalogModeling.tapeSaturation(sampleR, saturation);

        // Apply hysteresis (magnetic tape effect)
        sampleL = AnalogModeling.hysteresis(sampleL, this.previousOutput, age * 0.5);
        sampleR = AnalogModeling.hysteresis(sampleR, this.previousOutput, age * 0.5);

        // Component aging
        if (age > 0.1) {
          sampleL = AnalogModeling.componentAging(sampleL, age);
          sampleR = AnalogModeling.componentAging(sampleR, age);
        }

        // Add tape hiss
        if (hiss > 0.01) {
          const hissNoise = AnalogModeling.tapeHiss() * hiss;
          sampleL += hissNoise;
          sampleR += hissNoise;
        }

        // Add AC hum
        if (hum > 0.01) {
          this.humPhase += (2 * Math.PI * 60) / this.sampleRate;
          if (this.humPhase > 2 * Math.PI) this.humPhase -= 2 * Math.PI;

          const humNoise = AnalogModeling.acHum(this.humPhase, 60) * hum;
          sampleL += humNoise;
          sampleR += humNoise;
        }

        // Add bias noise (very subtle)
        const biasNoise = AnalogModeling.biasNoise(age);
        sampleL += biasNoise;
        sampleR += biasNoise;

        // Update VU meter
        this.vuLevel = AnalogModeling.vuMeterBallistics(
          (Math.abs(sampleL) + Math.abs(sampleR)) / 2,
          this.vuLevel,
          this.sampleRate
        );

        // Store previous output for hysteresis
        this.previousOutput = (sampleL + sampleR) / 2;

        // Write to output
        outputL[i] = sampleL;
        outputR[i] = sampleR;
      }
    };

    this._trackNode(this.processor);
  }

  /**
   * Reconnect internal routing
   * @protected
   */
  _reconnect() {
    // Routing:
    // input -> inputGain -> dry/wet split
    //   dry -> dryGain -> output
    //   wet -> headBumpFilter -> highCutFilter -> processor -> wetGain -> output

    this.input.disconnect();

    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path
    this.input.connect(this.inputGain);
    this.inputGain.connect(this.headBumpFilter);
    this.headBumpFilter.connect(this.highCutFilter);
    this.highCutFilter.connect(this.processor);
    this.processor.connect(this.wetGain);
    this.wetGain.connect(this.output);
  }

  /**
   * Update mix (dry/wet)
   * @param {number} mix - Mix percentage (0-100)
   */
  setMix(mix) {
    const wet = mix / 100;
    const dry = 1 - wet;

    this.dryGain.gain.setTargetAtTime(dry, this.audioContext.currentTime, 0.01);
    this.wetGain.gain.setTargetAtTime(wet, this.audioContext.currentTime, 0.01);

    this.setParameter('mix', mix);
  }

  /**
   * Get VU meter level
   * @returns {number} VU level (0-1)
   */
  getVULevel() {
    return this.vuLevel;
  }

  /**
   * Create factory presets
   * @private
   * @returns {Array<Object>} Factory presets
   */
  _createFactoryPresets() {
    return [
      {
        name: 'Clean Tape',
        category: 'Subtle',
        description: 'Subtle tape coloration, minimal artifacts',
        parameters: {
          saturation: 15,
          warmth: 40,
          wowFlutter: 5,
          hiss: 5,
          hum: 0,
          age: 10,
          speed: 30,
          mix: 60
        }
      },
      {
        name: 'Warm & Thick',
        category: 'Processing',
        description: 'Warm, thick tape sound with moderate saturation',
        parameters: {
          saturation: 40,
          warmth: 70,
          wowFlutter: 20,
          hiss: 15,
          hum: 0,
          age: 30,
          speed: 15,
          mix: 75
        }
      },
      {
        name: 'Vintage Lo-Fi',
        category: 'Creative',
        description: 'Old, degraded tape sound with artifacts',
        parameters: {
          saturation: 60,
          warmth: 80,
          wowFlutter: 50,
          hiss: 40,
          hum: 20,
          age: 80,
          speed: 7.5,
          mix: 85
        }
      },
      {
        name: 'Studer A800',
        category: 'Emulation',
        description: 'Emulates Studer A800 2-inch tape machine',
        parameters: {
          saturation: 35,
          warmth: 60,
          wowFlutter: 10,
          hiss: 12,
          hum: 5,
          age: 20,
          speed: 15,
          mix: 70
        }
      },
      {
        name: 'Ampex 456',
        category: 'Emulation',
        description: 'Classic Ampex 456 tape formula',
        parameters: {
          saturation: 45,
          warmth: 65,
          wowFlutter: 15,
          hiss: 18,
          hum: 0,
          age: 25,
          speed: 15,
          mix: 75
        }
      },
      {
        name: 'Cassette Tape',
        category: 'Lo-Fi',
        description: 'Compact cassette tape sound',
        parameters: {
          saturation: 50,
          warmth: 85,
          wowFlutter: 35,
          hiss: 50,
          hum: 10,
          age: 60,
          speed: 7.5,
          mix: 80
        }
      },
      {
        name: 'Master Bus Glue',
        category: 'Mastering',
        description: 'Subtle tape glue for mastering',
        parameters: {
          saturation: 20,
          warmth: 50,
          wowFlutter: 3,
          hiss: 3,
          hum: 0,
          age: 15,
          speed: 30,
          mix: 50
        }
      },
      {
        name: 'Extreme Saturation',
        category: 'Creative',
        description: 'Heavy tape saturation and compression',
        parameters: {
          saturation: 85,
          warmth: 90,
          wowFlutter: 25,
          hiss: 30,
          hum: 0,
          age: 40,
          speed: 15,
          mix: 90
        }
      }
    ];
  }

  /**
   * Get factory preset by name
   * @param {string} name - Preset name
   * @returns {Object|null} Preset data
   */
  getFactoryPreset(name) {
    return this.factoryPresets.find(p => p.name === name) || null;
  }

  /**
   * Load factory preset
   * @param {string} name - Preset name
   * @param {number} morphTime - Morph time in seconds
   */
  loadFactoryPreset(name, morphTime = 0) {
    const preset = this.getFactoryPreset(name);
    if (preset) {
      for (const [param, value] of Object.entries(preset.parameters)) {
        this.setParameter(param, value, morphTime);
      }

      // Update mix
      if (preset.parameters.mix) {
        this.setMix(preset.parameters.mix);
      }
    }
  }

  /**
   * Clean up resources
   */
  dispose() {
    if (this.processor) {
      this.processor.onaudioprocess = null;
      this.processor.disconnect();
    }

    super.dispose();
  }
}

// Register with PluginFactory
PluginFactory.register('TapeEmulation', TapeEmulation, {
  category: 'vintage',
  description: 'Analog tape machine emulation with saturation, wow/flutter, hiss, and head bump',
  tags: ['tape', 'saturation', 'vintage', 'analog', 'warmth', 'lo-fi'],
  version: '1.0.0',
  author: 'Agent 18'
});

export default TapeEmulation;
