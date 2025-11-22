/**
 * VintageEQPultec - Pultec EQP-1A Tube EQ Emulation
 *
 * @description
 * Emulates the legendary Pultec EQP-1A passive tube equalizer:
 * - Low-frequency boost (20, 30, 60, 100Hz)
 * - Low-frequency attenuation (20, 30, 60, 100Hz)
 * - High-frequency boost (3, 4, 5, 8, 10, 12, 16kHz)
 * - High-frequency attenuation (5, 10, 20kHz)
 * - Tube-based gain stage with harmonic distortion
 * - Famous "low-end trick" (boost and cut same frequency)
 *
 * Based on: Pultec EQP-1A
 */

import { BasePlugin } from '../core/BasePlugin.js';
import PluginFactory from '../core/PluginFactory.js';
import AnalogModeling from './AnalogModeling.js';

export class VintageEQPultec extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      ...options,
      category: 'vintage',
      description: 'Pultec EQP-1A tube equalizer with sweet, musical curves'
    });

    this.sampleRate = audioContext.sampleRate;
    this._createAudioGraph();
    this._registerParameters();
    this._createProcessor();
    this.factoryPresets = this._createFactoryPresets();
  }

  _createAudioGraph() {
    // Low frequency boost
    this.lowBoostFilter = this._trackNode(this.audioContext.createBiquadFilter());
    this.lowBoostFilter.type = 'lowshelf';
    this.lowBoostFilter.frequency.value = 60;
    this.lowBoostFilter.gain.value = 0;

    // Low frequency attenuation
    this.lowCutFilter = this._trackNode(this.audioContext.createBiquadFilter());
    this.lowCutFilter.type = 'lowshelf';
    this.lowCutFilter.frequency.value = 60;
    this.lowCutFilter.gain.value = 0;

    // High frequency boost
    this.highBoostFilter = this._trackNode(this.audioContext.createBiquadFilter());
    this.highBoostFilter.type = 'peaking';
    this.highBoostFilter.frequency.value = 10000;
    this.highBoostFilter.Q.value = 1.0;
    this.highBoostFilter.gain.value = 0;

    // High frequency attenuation
    this.highCutFilter = this._trackNode(this.audioContext.createBiquadFilter());
    this.highCutFilter.type = 'highshelf';
    this.highCutFilter.frequency.value = 10000;
    this.highCutFilter.gain.value = 0;

    // Output gain (makeup/tube gain)
    this.outputGain = this._trackNode(this.audioContext.createGain());
    this.outputGain.gain.value = 1.0;
  }

  _registerParameters() {
    // Low Boost Frequency
    this.registerParameter('lowBoostFreq', this.lowBoostFilter.frequency, {
      min: 20,
      max: 100,
      default: 60,
      unit: 'Hz',
      label: 'Low Boost Freq',
      type: 'discrete'
    });

    // Low Boost Gain
    this.registerParameter('lowBoost', this.lowBoostFilter.gain, {
      min: 0,
      max: 18,
      default: 0,
      unit: 'dB',
      label: 'Low Boost'
    });

    // Low Attenuation Frequency
    this.registerParameter('lowCutFreq', this.lowCutFilter.frequency, {
      min: 20,
      max: 100,
      default: 60,
      unit: 'Hz',
      label: 'Low Atten Freq',
      type: 'discrete'
    });

    // Low Attenuation Amount
    const lowCutParam = this.audioContext.createGain().gain;
    lowCutParam.value = 0;
    this.registerParameter('lowAtten', lowCutParam, {
      min: 0,
      max: 18,
      default: 0,
      unit: 'dB',
      label: 'Low Atten'
    });

    // High Boost Frequency
    this.registerParameter('highBoostFreq', this.highBoostFilter.frequency, {
      min: 3000,
      max: 16000,
      default: 10000,
      unit: 'Hz',
      label: 'High Boost Freq',
      type: 'discrete'
    });

    // High Boost Gain
    this.registerParameter('highBoost', this.highBoostFilter.gain, {
      min: 0,
      max: 18,
      default: 0,
      unit: 'dB',
      label: 'High Boost'
    });

    // High Bandwidth
    this.registerParameter('highBandwidth', this.highBoostFilter.Q, {
      min: 0.4,
      max: 2.5,
      default: 1.0,
      unit: '',
      label: 'High Bandwidth'
    });

    // High Attenuation Frequency
    this.registerParameter('highCutFreq', this.highCutFilter.frequency, {
      min: 5000,
      max: 20000,
      default: 10000,
      unit: 'Hz',
      label: 'High Atten Freq',
      type: 'discrete'
    });

    // High Attenuation Amount
    const highCutParam = this.audioContext.createGain().gain;
    highCutParam.value = 0;
    this.registerParameter('highAtten', highCutParam, {
      min: 0,
      max: 18,
      default: 0,
      unit: 'dB',
      label: 'High Atten'
    });

    // Output gain
    this.registerParameter('output', this.outputGain.gain, {
      min: -12,
      max: 12,
      default: 0,
      unit: 'dB',
      label: 'Output'
    });

    // Tube saturation amount
    const tubeSatParam = this.audioContext.createGain().gain;
    tubeSatParam.value = 30;
    this.registerParameter('tubeSaturation', tubeSatParam, {
      min: 0,
      max: 100,
      default: 30,
      unit: '%',
      label: 'Tube Saturation'
    });
  }

  _createProcessor() {
    const bufferSize = 1024;
    this.processor = this.audioContext.createScriptProcessor(bufferSize, 2, 2);

    this.processor.onaudioprocess = (e) => {
      const inputL = e.inputBuffer.getChannelData(0);
      const inputR = e.inputBuffer.getChannelData(1);
      const outputL = e.outputBuffer.getChannelData(0);
      const outputR = e.outputBuffer.getChannelData(1);

      const tubeSat = this.getParameter('tubeSaturation') / 100;
      const lowAtten = this.getParameter('lowAtten');
      const highAtten = this.getParameter('highAtten');

      // Update low cut (attenuation)
      this.lowCutFilter.gain.value = -lowAtten;

      // Update high cut (attenuation)
      this.highCutFilter.gain.value = -highAtten;

      for (let i = 0; i < inputL.length; i++) {
        let sampleL = inputL[i];
        let sampleR = inputR[i];

        // Apply tube saturation (warmth)
        if (tubeSat > 0.01) {
          sampleL = AnalogModeling.tubeSaturation(sampleL, tubeSat * 0.5);
          sampleR = AnalogModeling.tubeSaturation(sampleR, tubeSat * 0.5);
        }

        outputL[i] = sampleL;
        outputR[i] = sampleR;
      }
    };

    this._trackNode(this.processor);
  }

  _reconnect() {
    this.input.disconnect();

    // Signal chain: input -> lowBoost -> lowCut -> highBoost -> highCut -> processor -> output
    this.input.connect(this.lowBoostFilter);
    this.lowBoostFilter.connect(this.lowCutFilter);
    this.lowCutFilter.connect(this.highBoostFilter);
    this.highBoostFilter.connect(this.highCutFilter);
    this.highCutFilter.connect(this.processor);
    this.processor.connect(this.outputGain);
    this.outputGain.connect(this.output);
  }

  _createFactoryPresets() {
    return [
      {
        name: 'Pultec Trick (60Hz)',
        category: 'Low End',
        description: 'Famous Pultec trick - boost and cut same frequency for tight low end',
        parameters: {
          lowBoostFreq: 60,
          lowBoost: 8,
          lowCutFreq: 60,
          lowAtten: 5,
          highBoostFreq: 10000,
          highBoost: 0,
          highBandwidth: 1.0,
          highCutFreq: 10000,
          highAtten: 0,
          output: 0,
          tubeSaturation: 30
        }
      },
      {
        name: 'Warm & Open',
        category: 'Mastering',
        description: 'Warm low end with open, silky highs',
        parameters: {
          lowBoostFreq: 30,
          lowBoost: 4,
          lowCutFreq: 20,
          lowAtten: 0,
          highBoostFreq: 12000,
          highBoost: 6,
          highBandwidth: 0.7,
          highCutFreq: 20000,
          highAtten: 0,
          output: -1,
          tubeSaturation: 40
        }
      },
      {
        name: 'Kick Drum Punch',
        category: 'Drums',
        description: 'Powerful kick drum presence',
        parameters: {
          lowBoostFreq: 60,
          lowBoost: 10,
          lowCutFreq: 30,
          lowAtten: 3,
          highBoostFreq: 5000,
          highBoost: 4,
          highBandwidth: 1.2,
          highCutFreq: 10000,
          highAtten: 0,
          output: 0,
          tubeSaturation: 25
        }
      },
      {
        name: 'Vocal Air',
        category: 'Vocal',
        description: 'Add air and presence to vocals',
        parameters: {
          lowBoostFreq: 100,
          lowBoost: 3,
          lowCutFreq: 60,
          lowAtten: 0,
          highBoostFreq: 12000,
          highBoost: 8,
          highBandwidth: 0.6,
          highCutFreq: 5000,
          highAtten: 0,
          output: 0,
          tubeSaturation: 35
        }
      },
      {
        name: 'Bass Boost',
        category: 'Bass',
        description: 'Massive low-end boost',
        parameters: {
          lowBoostFreq: 60,
          lowBoost: 12,
          lowCutFreq: 20,
          lowAtten: 0,
          highBoostFreq: 8000,
          highBoost: 0,
          highBandwidth: 1.0,
          highCutFreq: 10000,
          highAtten: 2,
          output: -2,
          tubeSaturation: 45
        }
      },
      {
        name: 'Sparkle Top',
        category: 'Mix',
        description: 'Silky high-end sparkle',
        parameters: {
          lowBoostFreq: 30,
          lowBoost: 0,
          lowCutFreq: 20,
          lowAtten: 0,
          highBoostFreq: 16000,
          highBoost: 10,
          highBandwidth: 0.5,
          highCutFreq: 20000,
          highAtten: 0,
          output: -1,
          tubeSaturation: 30
        }
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

PluginFactory.register('VintageEQPultec', VintageEQPultec, {
  category: 'vintage',
  description: 'Pultec EQP-1A tube equalizer with musical, passive EQ curves',
  tags: ['eq', 'pultec', 'tube', 'vintage', 'passive'],
  version: '1.0.0',
  author: 'Agent 18'
});

export default VintageEQPultec;
