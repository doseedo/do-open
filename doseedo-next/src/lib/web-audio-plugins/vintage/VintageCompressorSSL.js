/**
 * VintageCompressorSSL - SSL Bus Compressor Emulation
 *
 * @description
 * Emulates the SSL 4000 series bus compressor (VCA style):
 * - Glue compression for mix bus
 * - Fixed attack times (.1, .3, 1, 3, 10, 30ms)
 * - Fixed release times (.1, .3, .6, 1.2s + Auto)
 * - 2:1, 4:1, 10:1 ratios
 * - VCA-based gain reduction
 *
 * Based on: SSL 4000G Bus Compressor
 */

import { BasePlugin } from '../core/BasePlugin.js';
import PluginFactory from '../core/PluginFactory.js';
import AnalogModeling from './AnalogModeling.js';

export class VintageCompressorSSL extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      ...options,
      category: 'vintage',
      description: 'SSL bus compressor for mix glue'
    });

    this.sampleRate = audioContext.sampleRate;
    this._createAudioGraph();

    this.envelope = 0;
    this.gainReduction = 0;

    this._registerParameters();
    this._createProcessor();
    this.factoryPresets = this._createFactoryPresets();
  }

  _createAudioGraph() {
    this.inputGain = this._trackNode(this.audioContext.createGain());
    this.outputGain = this._trackNode(this.audioContext.createGain());
    this.makeupGain = this._trackNode(this.audioContext.createGain());
  }

  _registerParameters() {
    // Threshold
    const thresholdParam = this.audioContext.createGain().gain;
    thresholdParam.value = -10;
    this.registerParameter('threshold', thresholdParam, {
      min: -30,
      max: 0,
      default: -10,
      unit: 'dB',
      label: 'Threshold'
    });

    // Ratio
    const ratioParam = this.audioContext.createGain().gain;
    ratioParam.value = 4;
    this.registerParameter('ratio', ratioParam, {
      min: 2,
      max: 10,
      default: 4,
      unit: ':1',
      label: 'Ratio',
      type: 'discrete'
    });

    // Attack (.1, .3, 1, 3, 10, 30ms)
    const attackParam = this.audioContext.createGain().gain;
    attackParam.value = 3;
    this.registerParameter('attack', attackParam, {
      min: 0.1,
      max: 30,
      default: 3,
      unit: 'ms',
      label: 'Attack'
    });

    // Release (.1, .3, .6, 1.2s, Auto)
    const releaseParam = this.audioContext.createGain().gain;
    releaseParam.value = 300;
    this.registerParameter('release', releaseParam, {
      min: 100,
      max: 1200,
      default: 300,
      unit: 'ms',
      label: 'Release'
    });

    // Auto release
    const autoReleaseParam = this.audioContext.createGain().gain;
    autoReleaseParam.value = 1;
    this.registerParameter('autoRelease', autoReleaseParam, {
      min: 0,
      max: 1,
      default: 1,
      unit: '',
      label: 'Auto Release',
      type: 'boolean'
    });

    // Makeup gain
    this.registerParameter('makeup', this.makeupGain.gain, {
      min: 0,
      max: 20,
      default: 0,
      unit: 'dB',
      label: 'Makeup Gain'
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

      const thresholdDb = this.getParameter('threshold');
      const ratio = this.getParameter('ratio');
      const attackMs = this.getParameter('attack');
      const releaseMs = this.getParameter('release');
      const autoRelease = this.getParameter('autoRelease') > 0.5;

      const threshold = Math.pow(10, thresholdDb / 20);
      const attackTime = attackMs / 1000;
      const releaseTime = releaseMs / 1000;

      const attackCoeff = Math.exp(-1 / (this.sampleRate * attackTime));
      const releaseCoeff = Math.exp(-1 / (this.sampleRate * releaseTime));

      for (let i = 0; i < inputL.length; i++) {
        const inputLevel = Math.max(Math.abs(inputL[i]), Math.abs(inputR[i]));

        // Envelope follower
        if (inputLevel > this.envelope) {
          this.envelope = attackCoeff * this.envelope + (1 - attackCoeff) * inputLevel;
        } else {
          // Auto release: faster release with low signal
          const effectiveRelease = autoRelease ?
            releaseCoeff * (1 - inputLevel * 0.5) :
            releaseCoeff;

          this.envelope = effectiveRelease * this.envelope + (1 - effectiveRelease) * inputLevel;
        }

        // Compression calculation
        if (this.envelope > threshold) {
          const excess = this.envelope - threshold;
          const compressed = threshold + excess / ratio;
          this.gainReduction = 1 - compressed / this.envelope;
        } else {
          this.gainReduction = 0;
        }

        const gain = 1 - this.gainReduction;

        // Apply compression
        let sampleL = inputL[i] * gain;
        let sampleR = inputR[i] * gain;

        // VCA coloration (subtle)
        const vcaDrive = this.gainReduction * 0.3;
        sampleL = AnalogModeling.transistorSaturation(sampleL, vcaDrive);
        sampleR = AnalogModeling.transistorSaturation(sampleR, vcaDrive);

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
    this.processor.connect(this.makeupGain);
    this.makeupGain.connect(this.output);
  }

  getGainReduction() {
    return this.gainReduction;
  }

  _createFactoryPresets() {
    return [
      {
        name: 'Mix Bus Glue',
        category: 'Mix',
        parameters: { threshold: -8, ratio: 4, attack: 3, release: 300, autoRelease: 1, makeup: 3 }
      },
      {
        name: 'Drum Bus',
        category: 'Drums',
        parameters: { threshold: -10, ratio: 4, attack: 1, release: 100, autoRelease: 0, makeup: 4 }
      },
      {
        name: 'Gentle Glue',
        category: 'Subtle',
        parameters: { threshold: -15, ratio: 2, attack: 10, release: 600, autoRelease: 1, makeup: 2 }
      },
      {
        name: 'Aggressive Bus',
        category: 'Heavy',
        parameters: { threshold: -5, ratio: 10, attack: 0.3, release: 300, autoRelease: 1, makeup: 6 }
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

PluginFactory.register('VintageCompressorSSL', VintageCompressorSSL, {
  category: 'vintage',
  description: 'SSL bus compressor for mix glue and cohesion',
  tags: ['compressor', 'ssl', 'bus', 'vintage', 'vca', 'glue'],
  version: '1.0.0',
  author: 'Agent 18'
});

export default VintageCompressorSSL;
