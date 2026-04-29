/**
 * VintageEQNeve - Neve 1073 EQ Emulation
 *
 * @description
 * Emulates the classic Neve 1073 3-band EQ with high-pass filter:
 * - High-pass filter (50, 80, 160, 300Hz)
 * - Low-shelf (35, 60, 110, 220Hz) with switchable boost/cut
 * - Mid-band parametric EQ (12 frequencies) with selectable bandwidth
 * - High-shelf (12kHz) with boost/cut
 * - Transformer-based I/O stage with coloration
 * - Marinair transformer emulation
 *
 * Based on: Neve 1073 console module
 */

import { BasePlugin } from '../core/BasePlugin.js';
import PluginFactory from '../core/PluginFactory.js';
import AnalogModeling from './AnalogModeling.js';

export class VintageEQNeve extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      ...options,
      category: 'vintage',
      description: 'Neve 1073 EQ with legendary transformer coloration'
    });

    this.sampleRate = audioContext.sampleRate;
    this._createAudioGraph();
    this._registerParameters();
    this._createProcessor();
    this.factoryPresets = this._createFactoryPresets();
  }

  _createAudioGraph() {
    // High-pass filter
    this.hpFilter = this._trackNode(this.audioContext.createBiquadFilter());
    this.hpFilter.type = 'highpass';
    this.hpFilter.frequency.value = 50;
    this.hpFilter.Q.value = 0.707;

    // Low shelf
    this.lowShelf = this._trackNode(this.audioContext.createBiquadFilter());
    this.lowShelf.type = 'lowshelf';
    this.lowShelf.frequency.value = 60;
    this.lowShelf.gain.value = 0;

    // Mid parametric
    this.midPeak = this._trackNode(this.audioContext.createBiquadFilter());
    this.midPeak.type = 'peaking';
    this.midPeak.frequency.value = 1000;
    this.midPeak.Q.value = 1.0;
    this.midPeak.gain.value = 0;

    // High shelf
    this.highShelf = this._trackNode(this.audioContext.createBiquadFilter());
    this.highShelf.type = 'highshelf';
    this.highShelf.frequency.value = 12000;
    this.highShelf.gain.value = 0;

    // Output gain
    this.outputGain = this._trackNode(this.audioContext.createGain());
    this.outputGain.gain.value = 1.0;
  }

  _registerParameters() {
    // High-pass filter frequency
    this.registerParameter('hpFreq', this.hpFilter.frequency, {
      min: 0,
      max: 300,
      default: 0,
      unit: 'Hz',
      label: 'HPF',
      type: 'discrete'
    });

    // Low shelf frequency
    this.registerParameter('lowFreq', this.lowShelf.frequency, {
      min: 35,
      max: 220,
      default: 60,
      unit: 'Hz',
      label: 'Low Freq',
      type: 'discrete'
    });

    // Low shelf gain
    this.registerParameter('lowGain', this.lowShelf.gain, {
      min: -16,
      max: 16,
      default: 0,
      unit: 'dB',
      label: 'Low Gain'
    });

    // Mid frequency
    this.registerParameter('midFreq', this.midPeak.frequency, {
      min: 360,
      max: 7200,
      default: 1000,
      unit: 'Hz',
      label: 'Mid Freq'
    });

    // Mid gain
    this.registerParameter('midGain', this.midPeak.gain, {
      min: -18,
      max: 18,
      default: 0,
      unit: 'dB',
      label: 'Mid Gain'
    });

    // Mid Q (bandwidth)
    this.registerParameter('midQ', this.midPeak.Q, {
      min: 0.5,
      max: 3.0,
      default: 1.0,
      unit: '',
      label: 'Mid Q'
    });

    // High shelf gain
    this.registerParameter('highGain', this.highShelf.gain, {
      min: -16,
      max: 16,
      default: 0,
      unit: 'dB',
      label: 'High Gain'
    });

    // Output gain
    this.registerParameter('output', this.outputGain.gain, {
      min: -12,
      max: 12,
      default: 0,
      unit: 'dB',
      label: 'Output'
    });

    // Transformer coloration
    const transformerParam = this.audioContext.createGain().gain;
    transformerParam.value = 60;
    this.registerParameter('transformer', transformerParam, {
      min: 0,
      max: 100,
      default: 60,
      unit: '%',
      label: 'Transformer'
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

      const transformer = this.getParameter('transformer') / 100;

      for (let i = 0; i < inputL.length; i++) {
        let sampleL = inputL[i];
        let sampleR = inputR[i];

        // Marinair transformer coloration (warm, slightly saturated)
        if (transformer > 0.01) {
          // Input transformer
          sampleL = AnalogModeling.tubeSaturation(sampleL, transformer * 0.4);
          sampleR = AnalogModeling.tubeSaturation(sampleR, transformer * 0.4);

          // Output transformer
          sampleL = AnalogModeling.softClip(sampleL, 0.85);
          sampleR = AnalogModeling.softClip(sampleR, 0.85);
        }

        outputL[i] = sampleL;
        outputR[i] = sampleR;
      }
    };

    this._trackNode(this.processor);
  }

  _reconnect() {
    this.input.disconnect();

    // Signal chain: input -> hp -> low -> mid -> high -> processor -> output
    this.input.connect(this.hpFilter);
    this.hpFilter.connect(this.lowShelf);
    this.lowShelf.connect(this.midPeak);
    this.midPeak.connect(this.highShelf);
    this.highShelf.connect(this.processor);
    this.processor.connect(this.outputGain);
    this.outputGain.connect(this.output);
  }

  _createFactoryPresets() {
    return [
      {
        name: 'Classic Vocal',
        category: 'Vocal',
        description: 'Classic Neve vocal EQ',
        parameters: {
          hpFreq: 80,
          lowFreq: 110,
          lowGain: 3,
          midFreq: 3500,
          midGain: 4,
          midQ: 1.0,
          highGain: 6,
          output: 0,
          transformer: 70
        }
      },
      {
        name: 'Warm Bass',
        category: 'Bass',
        description: 'Warm, full bass sound',
        parameters: {
          hpFreq: 0,
          lowFreq: 60,
          lowGain: 6,
          midFreq: 700,
          midGain: -2,
          midQ: 1.5,
          highGain: -3,
          output: 0,
          transformer: 75
        }
      },
      {
        name: 'Punchy Drums',
        category: 'Drums',
        description: 'Punchy, present drum sound',
        parameters: {
          hpFreq: 50,
          lowFreq: 60,
          lowGain: 5,
          midFreq: 2500,
          midGain: 3,
          midQ: 0.8,
          highGain: 4,
          output: 0,
          transformer: 60
        }
      },
      {
        name: 'Mix Bus Warmth',
        category: 'Mastering',
        description: 'Subtle warmth for mix bus',
        parameters: {
          hpFreq: 0,
          lowFreq: 60,
          lowGain: 2,
          midFreq: 1000,
          midGain: 0,
          midQ: 1.0,
          highGain: 3,
          output: 0,
          transformer: 80
        }
      },
      {
        name: 'Acoustic Guitar',
        category: 'Guitar',
        description: 'Bright, present acoustic guitar',
        parameters: {
          hpFreq: 80,
          lowFreq: 110,
          lowGain: -2,
          midFreq: 4500,
          midGain: 4,
          midQ: 1.2,
          highGain: 6,
          output: 0,
          transformer: 55
        }
      },
      {
        name: 'Rock Snare',
        category: 'Drums',
        description: 'Big, fat rock snare',
        parameters: {
          hpFreq: 160,
          lowFreq: 220,
          lowGain: 4,
          midFreq: 3000,
          midGain: 6,
          midQ: 0.9,
          highGain: 5,
          output: 0,
          transformer: 65
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

PluginFactory.register('VintageEQNeve', VintageEQNeve, {
  category: 'vintage',
  description: 'Neve 1073 equalizer with legendary Marinair transformer sound',
  tags: ['eq', 'neve', '1073', 'vintage', 'transformer'],
  version: '1.0.0',
  author: 'Agent 18'
});

export default VintageEQNeve;
