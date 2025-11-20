/**
 * Auto-Registration for All Plugins
 *
 * Registers all 27 effect plugins with PluginFactory for easy instantiation
 *
 * Usage:
 *   import { registerAllPlugins } from './web-audio-plugins/register-all.js';
 *   import PluginFactory from './web-audio-plugins/core/PluginFactory.js';
 *
 *   registerAllPlugins();
 *   const compressor = PluginFactory.create('Compressor', audioContext);
 */

import PluginFactory from './core/PluginFactory.js';

// Import all plugins
import Compressor from './dynamics/Compressor.js';
import Gate from './dynamics/Gate.js';
import Limiter from './dynamics/Limiter.js';
import GlueCompressor from './dynamics/GlueCompressor.js';

import EQEight from './eq/EQEight.js';
import EQThree from './eq/EQThree.js';

import AutoFilter from './filters/AutoFilter.js';

import SimpleDelay from './delay/SimpleDelay.js';
import PingPongDelay from './delay/PingPongDelay.js';
import FilterDelay from './delay/FilterDelay.js';

import Chorus from './modulation/Chorus.js';
import Flanger from './modulation/Flanger.js';
import Phaser from './modulation/Phaser.js';
import Tremolo from './modulation/Tremolo.js';

import Reverb from './reverb/Reverb.js';
import HybridReverb from './reverb/HybridReverb.js';
import Echo from './reverb/Echo.js';

import Overdrive from './distortion/Overdrive.js';
import Saturator from './distortion/Saturator.js';
import Distortion from './distortion/Distortion.js';
import Redux from './distortion/Redux.js';

import SpectralTime from './spectral/SpectralTime.js';
import SpectralResonator from './spectral/SpectralResonator.js';
import FrequencyShifter from './spectral/FrequencyShifter.js';
import Vocoder from './spectral/Vocoder.js';

import Utility from './utility/Utility.js';
import SpectrumAnalyzer from './utility/SpectrumAnalyzer.js';
import Tuner from './utility/Tuner.js';
import ChannelEQ from './utility/ChannelEQ.js';

import AdvancedLFO from './modulation-matrix/AdvancedLFO.js';
import EnvelopeGenerator from './modulation-matrix/EnvelopeGenerator.js';
import MacroControls from './modulation-matrix/MacroControls.js';
import ModulationMatrix from './modulation-matrix/ModulationMatrix.js';

import RingModulator from './creative/RingModulator.js';
import PitchShifter from './creative/PitchShifter.js';
import Granular from './creative/Granular.js';

/**
 * Register all plugins with PluginFactory
 * @returns {Object} Registration statistics
 */
export function registerAllPlugins() {
  const registrations = [];

  // Dynamics Processors
  PluginFactory.register('Compressor', Compressor, {
    category: 'Dynamics',
    description: 'Dynamic range compressor with sidechain support',
    tags: ['dynamics', 'compressor', 'mixing', 'mastering'],
    version: '1.0.0',
    author: 'Agent 1'
  });
  registrations.push('Compressor');

  PluginFactory.register('Gate', Gate, {
    category: 'Dynamics',
    description: 'Noise gate with hold and range controls',
    tags: ['dynamics', 'gate', 'noise-reduction'],
    version: '1.0.0',
    author: 'Agent 1'
  });
  registrations.push('Gate');

  PluginFactory.register('Limiter', Limiter, {
    category: 'Dynamics',
    description: 'Brick-wall limiter with lookahead',
    tags: ['dynamics', 'limiter', 'mastering'],
    version: '1.0.0',
    author: 'Agent 1'
  });
  registrations.push('Limiter');

  PluginFactory.register('GlueCompressor', GlueCompressor, {
    category: 'Dynamics',
    description: 'Vintage-style bus compressor',
    tags: ['dynamics', 'compressor', 'glue', 'bus'],
    version: '1.0.0',
    author: 'Agent 1'
  });
  registrations.push('GlueCompressor');

  // EQ & Filters
  PluginFactory.register('EQEight', EQEight, {
    category: 'EQ',
    description: '8-band parametric equalizer',
    tags: ['eq', 'equalizer', 'parametric', 'mixing'],
    version: '1.0.0',
    author: 'Agent 2'
  });
  registrations.push('EQEight');

  PluginFactory.register('EQThree', EQThree, {
    category: 'EQ',
    description: 'DJ-style 3-band EQ with kill switches',
    tags: ['eq', 'equalizer', 'dj', 'kill'],
    version: '1.0.0',
    author: 'Agent 2'
  });
  registrations.push('EQThree');

  PluginFactory.register('AutoFilter', AutoFilter, {
    category: 'Filter',
    description: 'Multi-mode filter with LFO and envelope follower',
    tags: ['filter', 'modulation', 'lfo', 'envelope'],
    version: '1.0.0',
    author: 'Agent 2'
  });
  registrations.push('AutoFilter');

  // Delays
  PluginFactory.register('SimpleDelay', SimpleDelay, {
    category: 'Time-Based',
    description: 'Basic delay with feedback and filtering',
    tags: ['delay', 'echo', 'time-based'],
    version: '1.0.0',
    author: 'Agent 3'
  });
  registrations.push('SimpleDelay');

  PluginFactory.register('PingPongDelay', PingPongDelay, {
    category: 'Time-Based',
    description: 'Stereo ping-pong delay effect',
    tags: ['delay', 'stereo', 'ping-pong'],
    version: '1.0.0',
    author: 'Agent 3'
  });
  registrations.push('PingPongDelay');

  PluginFactory.register('FilterDelay', FilterDelay, {
    category: 'Time-Based',
    description: '3-tap delay with independent filtering',
    tags: ['delay', 'filter', 'multi-tap'],
    version: '1.0.0',
    author: 'Agent 3'
  });
  registrations.push('FilterDelay');

  // Modulation Effects
  PluginFactory.register('Chorus', Chorus, {
    category: 'Modulation',
    description: 'Multi-voice chorus effect',
    tags: ['modulation', 'chorus', 'width'],
    version: '1.0.0',
    author: 'Agent 4'
  });
  registrations.push('Chorus');

  PluginFactory.register('Flanger', Flanger, {
    category: 'Modulation',
    description: 'Jet-plane flanger effect',
    tags: ['modulation', 'flanger', 'sweep'],
    version: '1.0.0',
    author: 'Agent 4'
  });
  registrations.push('Flanger');

  PluginFactory.register('Phaser', Phaser, {
    category: 'Modulation',
    description: 'Multi-stage phaser with all-pass filters',
    tags: ['modulation', 'phaser', 'sweep'],
    version: '1.0.0',
    author: 'Agent 4'
  });
  registrations.push('Phaser');

  PluginFactory.register('Tremolo', Tremolo, {
    category: 'Modulation',
    description: 'Amplitude/pan modulation effect',
    tags: ['modulation', 'tremolo', 'pan', 'amplitude'],
    version: '1.0.0',
    author: 'Agent 4'
  });
  registrations.push('Tremolo');

  // Reverb & Spatial
  PluginFactory.register('Reverb', Reverb, {
    category: 'Spatial',
    description: 'Algorithmic reverb with diffusion',
    tags: ['reverb', 'spatial', 'ambience'],
    version: '1.0.0',
    author: 'Agent 5'
  });
  registrations.push('Reverb');

  PluginFactory.register('HybridReverb', HybridReverb, {
    category: 'Spatial',
    description: 'Convolution + algorithmic hybrid reverb',
    tags: ['reverb', 'convolution', 'ir', 'spatial'],
    version: '1.0.0',
    author: 'Agent 5'
  });
  registrations.push('HybridReverb');

  PluginFactory.register('Echo', Echo, {
    category: 'Spatial',
    description: 'Complex echo with modulation and ducking',
    tags: ['delay', 'echo', 'spatial', 'modulation'],
    version: '1.0.0',
    author: 'Agent 5'
  });
  registrations.push('Echo');

  // Distortion & Saturation
  PluginFactory.register('Overdrive', Overdrive, {
    category: 'Distortion',
    description: 'Tube-style soft clipping overdrive',
    tags: ['distortion', 'overdrive', 'saturation', 'warmth'],
    version: '1.0.0',
    author: 'Agent 6'
  });
  registrations.push('Overdrive');

  PluginFactory.register('Saturator', Saturator, {
    category: 'Distortion',
    description: 'Multi-mode saturation plugin',
    tags: ['distortion', 'saturation', 'harmonics'],
    version: '1.0.0',
    author: 'Agent 6'
  });
  registrations.push('Saturator');

  PluginFactory.register('Distortion', Distortion, {
    category: 'Distortion',
    description: 'Hard clipping distortion',
    tags: ['distortion', 'clipping', 'aggressive'],
    version: '1.0.0',
    author: 'Agent 6'
  });
  registrations.push('Distortion');

  PluginFactory.register('Redux', Redux, {
    category: 'Distortion',
    description: 'Bit crushing and sample rate reduction',
    tags: ['distortion', 'bitcrush', 'lofi', 'digital'],
    version: '1.0.0',
    author: 'Agent 6'
  });
  registrations.push('Redux');

  // Spectral Processing
  PluginFactory.register('SpectralTime', SpectralTime, {
    category: 'Spectral',
    description: 'Time stretching with phase vocoder',
    tags: ['spectral', 'time-stretch', 'fft', 'phase-vocoder'],
    version: '1.0.0',
    author: 'Agent 8'
  });
  registrations.push('SpectralTime');

  PluginFactory.register('SpectralResonator', SpectralResonator, {
    category: 'Spectral',
    description: 'Resonant comb filtering',
    tags: ['spectral', 'resonator', 'comb', 'harmonics'],
    version: '1.0.0',
    author: 'Agent 8'
  });
  registrations.push('SpectralResonator');

  PluginFactory.register('FrequencyShifter', FrequencyShifter, {
    category: 'Spectral',
    description: 'Linear frequency shifting',
    tags: ['spectral', 'frequency-shift', 'ssb', 'ring-mod'],
    version: '1.0.0',
    author: 'Agent 8'
  });
  registrations.push('FrequencyShifter');

  PluginFactory.register('Vocoder', Vocoder, {
    category: 'Spectral',
    description: 'Multi-band vocoder',
    tags: ['spectral', 'vocoder', 'voice', 'synthesis'],
    version: '1.0.0',
    author: 'Agent 8'
  });
  registrations.push('Vocoder');

  // Utility & Analysis
  PluginFactory.register('Utility', Utility, {
    category: 'Utility',
    description: 'Gain, pan, width, and phase utility',
    tags: ['utility', 'gain', 'pan', 'width'],
    version: '1.0.0',
    author: 'Agent 9'
  });
  registrations.push('Utility');

  PluginFactory.register('SpectrumAnalyzer', SpectrumAnalyzer, {
    category: 'Analysis',
    description: 'Real-time FFT spectrum analyzer',
    tags: ['analysis', 'spectrum', 'fft', 'visualization'],
    version: '1.0.0',
    author: 'Agent 9'
  });
  registrations.push('SpectrumAnalyzer');

  PluginFactory.register('Tuner', Tuner, {
    category: 'Analysis',
    description: 'Pitch detection tuner',
    tags: ['analysis', 'tuner', 'pitch', 'frequency'],
    version: '1.0.0',
    author: 'Agent 9'
  });
  registrations.push('Tuner');

  PluginFactory.register('ChannelEQ', ChannelEQ, {
    category: 'Utility',
    description: 'Simple high/low cut filters',
    tags: ['eq', 'utility', 'filter', 'cut'],
    version: '1.0.0',
    author: 'Agent 9'
  });
  registrations.push('ChannelEQ');

  // Modulation Matrix
  PluginFactory.register('AdvancedLFO', AdvancedLFO, {
    category: 'Modulation',
    description: 'Advanced LFO with multiple waveforms and BPM sync',
    tags: ['modulation', 'lfo', 'automation'],
    version: '1.0.0',
    author: 'Agent 17'
  });
  registrations.push('AdvancedLFO');

  PluginFactory.register('EnvelopeGenerator', EnvelopeGenerator, {
    category: 'Modulation',
    description: 'ADSR/AHDSR envelope generator',
    tags: ['modulation', 'envelope', 'adsr'],
    version: '1.0.0',
    author: 'Agent 17'
  });
  registrations.push('EnvelopeGenerator');

  PluginFactory.register('MacroControls', MacroControls, {
    category: 'Modulation',
    description: '8 macro knobs for parameter mapping',
    tags: ['modulation', 'macro', 'control', 'mapping'],
    version: '1.0.0',
    author: 'Agent 17'
  });
  registrations.push('MacroControls');

  PluginFactory.register('ModulationMatrix', ModulationMatrix, {
    category: 'Modulation',
    description: 'Visual modulation routing matrix',
    tags: ['modulation', 'matrix', 'routing'],
    version: '1.0.0',
    author: 'Agent 17'
  });
  registrations.push('ModulationMatrix');

  // Creative Effects
  PluginFactory.register('RingModulator', RingModulator, {
    category: 'Creative',
    description: 'Ring modulation for inharmonic sidebands',
    tags: ['creative', 'ring-mod', 'modulation', 'experimental'],
    version: '1.0.0',
    author: 'Agent 9'
  });
  registrations.push('RingModulator');

  PluginFactory.register('PitchShifter', PitchShifter, {
    category: 'Creative',
    description: 'Time-domain pitch shifting effect',
    tags: ['creative', 'pitch-shift', 'transpose', 'harmony'],
    version: '1.0.0',
    author: 'Agent 9'
  });
  registrations.push('PitchShifter');

  PluginFactory.register('Granular', Granular, {
    category: 'Creative',
    description: 'Granular synthesis for evolving textures',
    tags: ['creative', 'granular', 'texture', 'experimental'],
    version: '1.0.0',
    author: 'Agent 9'
  });
  registrations.push('Granular');

  const stats = {
    totalRegistered: registrations.length,
    plugins: registrations,
    byCategory: PluginFactory.getStats().byCategory
  };

  console.log(`✅ Registered ${stats.totalRegistered} plugins with PluginFactory`);
  console.log('Categories:', stats.byCategory);

  return stats;
}

export default registerAllPlugins;
