/**
 * AnalogConsole - Analog mixing console emulation
 *
 * @description
 * Emulates the sound characteristics of classic analog mixing consoles:
 * - Console saturation and coloration
 * - Transformer coloration (input/output transformers)
 * - Crosstalk between channels
 * - VU meter emulation
 * - Subtle harmonic distortion
 * - Analog summing characteristics
 *
 * Inspired by: SSL 4000E/G, Neve 8078, API 1604
 *
 * @example
 * const console = new AnalogConsole(audioContext);
 * console.setParameter('drive', 60);
 * console.setParameter('transformer', 70);
 * input.connect(console.input);
 * console.connect(audioContext.destination);
 */

import { BasePlugin } from '../core/BasePlugin.js';
import PluginFactory from '../core/PluginFactory.js';
import AnalogModeling from './AnalogModeling.js';

export class AnalogConsole extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      ...options,
      category: 'vintage',
      description: 'Analog mixing console emulation with saturation and transformer coloration'
    });

    this.sampleRate = audioContext.sampleRate;

    // Create audio graph
    this._createAudioGraph();

    // Internal state
    this.vuLevelL = 0;
    this.vuLevelR = 0;
    this.previousOutputL = 0;
    this.previousOutputR = 0;

    // Register parameters
    this._registerParameters();

    // Create processor
    this._createProcessor();

    // Factory presets
    this.factoryPresets = this._createFactoryPresets();
  }

  /**
   * Create audio node graph
   * @private
   */
  _createAudioGraph() {
    // Input gain
    this.inputGain = this._trackNode(this.audioContext.createGain());
    this.inputGain.gain.value = 1.0;

    // Output gain
    this.outputGain = this._trackNode(this.audioContext.createGain());
    this.outputGain.gain.value = 1.0;

    // Dry/wet mix
    this.dryGain = this._trackNode(this.audioContext.createGain());
    this.wetGain = this._trackNode(this.audioContext.createGain());
    this.dryGain.gain.value = 0.2;
    this.wetGain.gain.value = 0.8;
  }

  /**
   * Register parameters
   * @private
   */
  _registerParameters() {
    // Drive/Saturation amount
    const driveParam = this.audioContext.createGain().gain;
    driveParam.value = 50;
    this.registerParameter('drive', driveParam, {
      min: 0,
      max: 100,
      default: 50,
      unit: '%',
      label: 'Drive',
      type: 'continuous'
    });

    // Transformer coloration
    const transformerParam = this.audioContext.createGain().gain;
    transformerParam.value = 60;
    this.registerParameter('transformer', transformerParam, {
      min: 0,
      max: 100,
      default: 60,
      unit: '%',
      label: 'Transformer',
      type: 'continuous'
    });

    // Crosstalk amount
    const crosstalkParam = this.audioContext.createGain().gain;
    crosstalkParam.value = 15;
    this.registerParameter('crosstalk', crosstalkParam, {
      min: 0,
      max: 100,
      default: 15,
      unit: '%',
      label: 'Crosstalk',
      type: 'continuous'
    });

    // Console type
    const typeParam = this.audioContext.createGain().gain;
    typeParam.value = 0; // 0=SSL, 1=Neve, 2=API
    this.registerParameter('consoleType', typeParam, {
      min: 0,
      max: 2,
      default: 0,
      unit: '',
      label: 'Console Type',
      type: 'discrete'
    });

    // Noise amount
    const noiseParam = this.audioContext.createGain().gain;
    noiseParam.value = 5;
    this.registerParameter('noise', noiseParam, {
      min: 0,
      max: 100,
      default: 5,
      unit: '%',
      label: 'Noise',
      type: 'continuous'
    });

    // Mix
    this.registerParameter('mix', this.wetGain.gain, {
      min: 0,
      max: 100,
      default: 80,
      unit: '%',
      label: 'Mix',
      type: 'continuous'
    });
  }

  /**
   * Create processor
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

      const drive = this.getParameter('drive') / 100;
      const transformer = this.getParameter('transformer') / 100;
      const crosstalk = this.getParameter('crosstalk') / 100;
      const consoleType = Math.round(this.getParameter('consoleType'));
      const noise = this.getParameter('noise') / 100;

      for (let i = 0; i < inputL.length; i++) {
        let sampleL = inputL[i];
        let sampleR = inputR[i];

        // Apply console-specific saturation
        switch (consoleType) {
          case 0: // SSL (VCA, clean, punchy)
            sampleL = AnalogModeling.transistorSaturation(sampleL, drive * 0.6);
            sampleR = AnalogModeling.transistorSaturation(sampleR, drive * 0.6);
            break;

          case 1: // Neve (transformer, warm, colored)
            sampleL = AnalogModeling.tubeSaturation(sampleL, drive * 0.8);
            sampleR = AnalogModeling.tubeSaturation(sampleR, drive * 0.8);
            break;

          case 2: // API (discrete, punchy, aggressive)
            sampleL = AnalogModeling.transistorSaturation(sampleL, drive * 0.9);
            sampleR = AnalogModeling.transistorSaturation(sampleR, drive * 0.9);
            break;
        }

        // Apply transformer coloration (frequency-dependent gain)
        if (transformer > 0.01) {
          // Simplified: add harmonics via soft clipping
          const transformAmount = transformer * 0.3;
          sampleL = AnalogModeling.softClip(sampleL, 0.8 - transformAmount * 0.2);
          sampleR = AnalogModeling.softClip(sampleR, 0.8 - transformAmount * 0.2);
        }

        // Apply crosstalk
        if (crosstalk > 0.01) {
          const crosstalked = AnalogModeling.crosstalk(sampleL, sampleR, crosstalk);
          sampleL = crosstalked.left;
          sampleR = crosstalked.right;
        }

        // Add console noise
        if (noise > 0.01) {
          const biasNoise = AnalogModeling.biasNoise(noise);
          sampleL += biasNoise;
          sampleR += biasNoise;
        }

        // Update VU meters
        this.vuLevelL = AnalogModeling.vuMeterBallistics(sampleL, this.vuLevelL, this.sampleRate);
        this.vuLevelR = AnalogModeling.vuMeterBallistics(sampleR, this.vuLevelR, this.sampleRate);

        // Store previous output
        this.previousOutputL = sampleL;
        this.previousOutputR = sampleR;

        // Write to output
        outputL[i] = sampleL;
        outputR[i] = sampleR;
      }
    };

    this._trackNode(this.processor);
  }

  /**
   * Reconnect routing
   * @protected
   */
  _reconnect() {
    this.input.disconnect();

    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path
    this.input.connect(this.inputGain);
    this.inputGain.connect(this.processor);
    this.processor.connect(this.wetGain);
    this.wetGain.connect(this.output);
  }

  /**
   * Get VU meter levels
   * @returns {Object} {left, right} VU levels (0-1)
   */
  getVULevels() {
    return {
      left: this.vuLevelL,
      right: this.vuLevelR
    };
  }

  /**
   * Set console type
   * @param {string} type - Console type ('ssl', 'neve', 'api')
   */
  setConsoleType(type) {
    const typeMap = {
      'ssl': 0,
      'neve': 1,
      'api': 2
    };

    const typeValue = typeMap[type.toLowerCase()] || 0;
    this.setParameter('consoleType', typeValue);
  }

  /**
   * Create factory presets
   * @private
   */
  _createFactoryPresets() {
    return [
      {
        name: 'SSL Bus Glue',
        category: 'Emulation',
        description: 'SSL 4000E/G console sound',
        parameters: {
          drive: 45,
          transformer: 40,
          crosstalk: 10,
          consoleType: 0,
          noise: 5,
          mix: 75
        }
      },
      {
        name: 'Neve Warmth',
        category: 'Emulation',
        description: 'Neve 8078 warm console sound',
        parameters: {
          drive: 55,
          transformer: 75,
          crosstalk: 20,
          consoleType: 1,
          noise: 10,
          mix: 80
        }
      },
      {
        name: 'API Punch',
        category: 'Emulation',
        description: 'API 1604 punchy console sound',
        parameters: {
          drive: 60,
          transformer: 50,
          crosstalk: 15,
          consoleType: 2,
          noise: 8,
          mix: 70
        }
      },
      {
        name: 'Subtle Glue',
        category: 'Mixing',
        description: 'Subtle console glue for mix bus',
        parameters: {
          drive: 30,
          transformer: 40,
          crosstalk: 5,
          consoleType: 0,
          noise: 3,
          mix: 50
        }
      },
      {
        name: 'Heavy Coloration',
        category: 'Creative',
        description: 'Heavy console coloration',
        parameters: {
          drive: 80,
          transformer: 85,
          crosstalk: 30,
          consoleType: 1,
          noise: 15,
          mix: 90
        }
      }
    ];
  }

  /**
   * Load factory preset
   * @param {string} name - Preset name
   */
  loadFactoryPreset(name) {
    const preset = this.factoryPresets.find(p => p.name === name);
    if (preset) {
      for (const [param, value] of Object.entries(preset.parameters)) {
        this.setParameter(param, value);
      }
    }
  }

  /**
   * Dispose
   */
  dispose() {
    if (this.processor) {
      this.processor.onaudioprocess = null;
      this.processor.disconnect();
    }

    super.dispose();
  }
}

// Register with factory
PluginFactory.register('AnalogConsole', AnalogConsole, {
  category: 'vintage',
  description: 'Analog mixing console emulation (SSL, Neve, API)',
  tags: ['console', 'saturation', 'vintage', 'ssl', 'neve', 'api'],
  version: '1.0.0',
  author: 'Agent 18'
});

export default AnalogConsole;
