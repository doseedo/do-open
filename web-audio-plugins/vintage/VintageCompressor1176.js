/**
 * VintageCompressor1176 - 1176 FET Compressor Emulation
 *
 * @description
 * Emulates the classic UREI 1176 FET (Field Effect Transistor) compressor:
 * - Ultra-fast attack times (20-800 microseconds)
 * - Fixed release times or auto-release
 * - 4:1, 8:1, 12:1, 20:1 ratios
 * - "All buttons in" mode (aggressive, distorted)
 * - FET saturation characteristics
 * - Analog VU meter ballistics
 *
 * Based on: UREI 1176LN
 *
 * @example
 * const comp = new VintageCompressor1176(audioContext);
 * comp.setParameter('ratio', 4);
 * comp.setParameter('attack', 400);
 * comp.setParameter('release', 600);
 */

import { BasePlugin } from '../core/BasePlugin.js';
import PluginFactory from '../core/PluginFactory.js';
import AnalogModeling from './AnalogModeling.js';

export class VintageCompressor1176 extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      ...options,
      category: 'vintage',
      description: '1176 FET compressor emulation with ultra-fast attack'
    });

    this.sampleRate = audioContext.sampleRate;

    // Create audio graph
    this._createAudioGraph();

    // Compression state
    this.gainReduction = 0;
    this.envelope = 0;

    // Register parameters
    this._registerParameters();

    // Create processor
    this._createProcessor();

    // Factory presets
    this.factoryPresets = this._createFactoryPresets();
  }

  _createAudioGraph() {
    this.inputGain = this._trackNode(this.audioContext.createGain());
    this.outputGain = this._trackNode(this.audioContext.createGain());
    this.makeupGain = this._trackNode(this.audioContext.createGain());
  }

  _registerParameters() {
    // Input gain
    this.registerParameter('input', this.inputGain.gain, {
      min: -12,
      max: 12,
      default: 0,
      unit: 'dB',
      label: 'Input'
    });

    // Ratio (4, 8, 12, 20, or 100 for "all buttons")
    const ratioParam = this.audioContext.createGain().gain;
    ratioParam.value = 4;
    this.registerParameter('ratio', ratioParam, {
      min: 4,
      max: 100,
      default: 4,
      unit: ':1',
      label: 'Ratio',
      type: 'discrete'
    });

    // Attack (20-800 microseconds)
    const attackParam = this.audioContext.createGain().gain;
    attackParam.value = 400;
    this.registerParameter('attack', attackParam, {
      min: 20,
      max: 800,
      default: 400,
      unit: 'μs',
      label: 'Attack'
    });

    // Release (50-1100 milliseconds)
    const releaseParam = this.audioContext.createGain().gain;
    releaseParam.value = 600;
    this.registerParameter('release', releaseParam, {
      min: 50,
      max: 1100,
      default: 600,
      unit: 'ms',
      label: 'Release'
    });

    // Output/Makeup gain
    this.registerParameter('output', this.makeupGain.gain, {
      min: -12,
      max: 24,
      default: 0,
      unit: 'dB',
      label: 'Output'
    });
  }

  _createProcessor() {
    const bufferSize = 256; // Small buffer for fast response
    this.processor = this.audioContext.createScriptProcessor(bufferSize, 2, 2);

    this.processor.onaudioprocess = (e) => {
      const inputL = e.inputBuffer.getChannelData(0);
      const inputR = e.inputBuffer.getChannelData(1);
      const outputL = e.outputBuffer.getChannelData(0);
      const outputR = e.outputBuffer.getChannelData(1);

      const ratio = this.getParameter('ratio');
      const attackUs = this.getParameter('attack');
      const releaseMs = this.getParameter('release');

      // Convert to coefficients
      const attackTime = attackUs / 1000000; // microseconds to seconds
      const releaseTime = releaseMs / 1000; // milliseconds to seconds

      const attackCoeff = Math.exp(-1 / (this.sampleRate * attackTime));
      const releaseCoeff = Math.exp(-1 / (this.sampleRate * releaseTime));

      for (let i = 0; i < inputL.length; i++) {
        // Peak detection (stereo link)
        const inputLevel = Math.max(Math.abs(inputL[i]), Math.abs(inputR[i]));

        // Envelope follower
        if (inputLevel > this.envelope) {
          this.envelope = attackCoeff * this.envelope + (1 - attackCoeff) * inputLevel;
        } else {
          this.envelope = releaseCoeff * this.envelope + (1 - releaseCoeff) * inputLevel;
        }

        // Compression calculation
        const threshold = 0.5; // Fixed threshold (typical for 1176)

        if (this.envelope > threshold) {
          const excess = this.envelope - threshold;

          // "All buttons in" mode
          if (ratio >= 100) {
            // Aggressive, distorted compression
            this.gainReduction = 1 - (threshold + excess * 0.1) / this.envelope;
          } else {
            // Standard compression
            const compressed = threshold + excess / ratio;
            this.gainReduction = 1 - compressed / this.envelope;
          }
        } else {
          this.gainReduction = 0;
        }

        const gain = 1 - this.gainReduction;

        // Apply compression
        let sampleL = inputL[i] * gain;
        let sampleR = inputR[i] * gain;

        // FET saturation
        const fetDrive = Math.min(1, this.gainReduction * 2);
        sampleL = AnalogModeling.transistorSaturation(sampleL, fetDrive);
        sampleR = AnalogModeling.transistorSaturation(sampleR, fetDrive);

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
        name: 'Vocal Compression',
        category: 'Vocal',
        parameters: { input: 0, ratio: 4, attack: 400, release: 600, output: 3 }
      },
      {
        name: 'Drum Bus',
        category: 'Drums',
        parameters: { input: 2, ratio: 8, attack: 50, release: 400, output: 2 }
      },
      {
        name: 'All Buttons In',
        category: 'Creative',
        parameters: { input: 4, ratio: 100, attack: 200, release: 550, output: 6 }
      },
      {
        name: 'Bass Tightening',
        category: 'Bass',
        parameters: { input: 0, ratio: 12, attack: 100, release: 800, output: 4 }
      },
      {
        name: 'Parallel Smash',
        category: 'Mix',
        parameters: { input: 6, ratio: 20, attack: 50, release: 300, output: 8 }
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

PluginFactory.register('VintageCompressor1176', VintageCompressor1176, {
  category: 'vintage',
  description: '1176 FET compressor with ultra-fast attack and multiple ratio modes',
  tags: ['compressor', '1176', 'fet', 'vintage', 'dynamics'],
  version: '1.0.0',
  author: 'Agent 18'
});

export default VintageCompressor1176;
