/**
 * VintageCompressorLA2A - LA-2A Optical Compressor Emulation
 *
 * @description
 * Emulates the Teletronix LA-2A optical compressor/limiter:
 * - Program-dependent attack/release (optical element)
 * - Smooth, musical compression
 * - Tube-based gain stage
 * - Fixed 3:1 ratio (approximate)
 * - Classic "peak reduction" control
 *
 * Based on: Teletronix LA-2A
 */

import { BasePlugin } from '../core/BasePlugin.js';
import PluginFactory from '../core/PluginFactory.js';
import AnalogModeling from './AnalogModeling.js';

export class VintageCompressorLA2A extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      ...options,
      category: 'vintage',
      description: 'LA-2A optical compressor with smooth, musical compression'
    });

    this.sampleRate = audioContext.sampleRate;
    this._createAudioGraph();

    // Optical element state (slower response)
    this.opticalState = 0;
    this.gainReduction = 0;

    this._registerParameters();
    this._createProcessor();
    this.factoryPresets = this._createFactoryPresets();
  }

  _createAudioGraph() {
    this.inputGain = this._trackNode(this.audioContext.createGain());
    this.outputGain = this._trackNode(this.audioContext.createGain());
  }

  _registerParameters() {
    // Peak reduction (threshold control)
    const peakParam = this.audioContext.createGain().gain;
    peakParam.value = 50;
    this.registerParameter('peakReduction', peakParam, {
      min: 0,
      max: 100,
      default: 50,
      unit: '%',
      label: 'Peak Reduction'
    });

    // Gain (makeup)
    this.registerParameter('gain', this.outputGain.gain, {
      min: -12,
      max: 24,
      default: 6,
      unit: 'dB',
      label: 'Gain'
    });

    // Limit mode (changes ratio to ~10:1)
    const limitParam = this.audioContext.createGain().gain;
    limitParam.value = 0;
    this.registerParameter('limit', limitParam, {
      min: 0,
      max: 1,
      default: 0,
      unit: '',
      label: 'Limit Mode',
      type: 'boolean'
    });
  }

  _createProcessor() {
    const bufferSize = 512;
    this.processor = this.audioContext.createScriptProcessor(bufferSize, 2, 2);

    this.processor.onaudioprocess = (e) => {
      const inputL = e.inputBuffer.getChannelData(0);
      const inputR = e.inputBuffer.getChannelData(1);
      const outputL = e.outputBuffer.getChannelData(0);
      const outputR = e.outputBuffer.getChannelData(1);

      const peakReduction = this.getParameter('peakReduction') / 100;
      const limitMode = this.getParameter('limit') > 0.5;

      // Optical attack/release (program-dependent)
      const baseAttack = 0.010; // 10ms
      const baseRelease = 0.060; // 60ms (can be up to 1-2 seconds)

      for (let i = 0; i < inputL.length; i++) {
        const inputLevel = Math.max(Math.abs(inputL[i]), Math.abs(inputR[i]));

        // Optical element simulation (slower, program-dependent)
        const attackTime = baseAttack * (1 + this.opticalState * 2);
        const releaseTime = baseRelease * (1 + this.opticalState * 20);

        const attackCoeff = Math.exp(-1 / (this.sampleRate * attackTime));
        const releaseCoeff = Math.exp(-1 / (this.sampleRate * releaseTime));

        if (inputLevel > this.opticalState) {
          this.opticalState = attackCoeff * this.opticalState + (1 - attackCoeff) * inputLevel;
        } else {
          this.opticalState = releaseCoeff * this.opticalState + (1 - releaseCoeff) * inputLevel;
        }

        // Threshold varies with peak reduction control
        const threshold = 0.3 + (1 - peakReduction) * 0.5;

        if (this.opticalState > threshold) {
          const excess = this.opticalState - threshold;
          const ratio = limitMode ? 10 : 3; // Compress or limit

          const compressed = threshold + excess / ratio;
          this.gainReduction = 1 - compressed / this.opticalState;
        } else {
          this.gainReduction = 0;
        }

        const gain = 1 - this.gainReduction;

        // Apply compression
        let sampleL = inputL[i] * gain;
        let sampleR = inputR[i] * gain;

        // Tube saturation
        sampleL = AnalogModeling.tubeSaturation(sampleL, this.gainReduction * 0.5);
        sampleR = AnalogModeling.tubeSaturation(sampleR, this.gainReduction * 0.5);

        outputL[i] = sampleL;
        outputR[i] = sampleR;
      }
    };

    this._trackNode(this.processor);
  }

  _reconnect() {
    this.input.disconnect();
    this.input.connect(this.inputGain);
    this.inputGain.connect(this.processor);
    this.processor.connect(this.outputGain);
    this.outputGain.connect(this.output);
  }

  getGainReduction() {
    return this.gainReduction;
  }

  _createFactoryPresets() {
    return [
      {
        name: 'Smooth Vocals',
        category: 'Vocal',
        parameters: { peakReduction: 60, gain: 8, limit: 0 }
      },
      {
        name: 'Bass Leveling',
        category: 'Bass',
        parameters: { peakReduction: 70, gain: 6, limit: 0 }
      },
      {
        name: 'Gentle Master',
        category: 'Mastering',
        parameters: { peakReduction: 40, gain: 4, limit: 0 }
      },
      {
        name: 'Heavy Limiting',
        category: 'Creative',
        parameters: { peakReduction: 85, gain: 12, limit: 1 }
      }
    ];
  }

  loadFactoryPreset(name) {
    const preset = this.factoryPresets.find(p => p.name === name);
    if (preset) {
      for (const [param, value] of Object.entries(preset.parameters)) {
        this.setParameter(param, value);
      }
    }
  }

  dispose() {
    if (this.processor) {
      this.processor.onaudioprocess = null;
    }
    super.dispose();
  }
}

PluginFactory.register('VintageCompressorLA2A', VintageCompressorLA2A, {
  category: 'vintage',
  description: 'LA-2A optical compressor with smooth, program-dependent response',
  tags: ['compressor', 'la-2a', 'optical', 'vintage', 'tube'],
  version: '1.0.0',
  author: 'Agent 18'
});

export default VintageCompressorLA2A;
