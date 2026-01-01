/**
 * Auto-Registration for All Plugins
 * Fixed to use correct exports from index.js files
 */

import PluginFactory from './core/PluginFactory.js';

// Dynamics
import { Compressor, Gate, Limiter, Expander } from './dynamics/index.js';

// EQ
import { EQ, Filter, GraphicEQ } from './eq/index.js';

// Delay
import { SimpleDelay, PingPongDelay } from './delay/index.js';

// Modulation
import { Chorus, Flanger, Phaser, Tremolo } from './modulation/index.js';

// Reverb
import { Reverb, HybridReverb, Echo } from './reverb/index.js';

// Distortion
import { Overdrive, Saturator, Distortion, Redux } from './distortion/index.js';

// Spectral
import { SpectralTime, FrequencyShifter as SpectralFrequencyShifter } from './spectral/index.js';

// Creative - disabled due to CommonJS export issues
// import { RingModulator, PitchShifter, Granular } from './creative/index.js';

// Utility - disabled due to CommonJS export issues
// import { Gain, Pan, StereoWidth } from './utility/index.js';

// Analysis - disabled due to CommonJS export issues
// import { SpectrumAnalyzer } from './analysis/index.js';

// Modulation Matrix
import { AdvancedLFO, EnvelopeGenerator, MacroControls, ModulationMatrix } from './modulation-matrix/index.js';

export function registerAllPlugins() {
  // Dynamics
  PluginFactory.register('Compressor', Compressor, { category: 'Dynamics' });
  PluginFactory.register('Gate', Gate, { category: 'Dynamics' });
  PluginFactory.register('Limiter', Limiter, { category: 'Dynamics' });
  PluginFactory.register('Expander', Expander, { category: 'Dynamics' });

  // EQ - with aliases for compatibility
  PluginFactory.register('EQ', EQ, { category: 'EQ' });
  PluginFactory.register('EQThree', EQ, { category: 'EQ' });
  PluginFactory.register('Filter', Filter, { category: 'Filter' });
  PluginFactory.register('AutoFilter', Filter, { category: 'Filter' });
  PluginFactory.register('GraphicEQ', GraphicEQ, { category: 'EQ' });
  PluginFactory.register('EQEight', GraphicEQ, { category: 'EQ' });

  // Delay
  PluginFactory.register('SimpleDelay', SimpleDelay, { category: 'Time-Based' });
  PluginFactory.register('PingPongDelay', PingPongDelay, { category: 'Time-Based' });

  // Modulation
  PluginFactory.register('Chorus', Chorus, { category: 'Modulation' });
  PluginFactory.register('Flanger', Flanger, { category: 'Modulation' });
  PluginFactory.register('Phaser', Phaser, { category: 'Modulation' });
  PluginFactory.register('Tremolo', Tremolo, { category: 'Modulation' });

  // Reverb
  PluginFactory.register('Reverb', Reverb, { category: 'Spatial' });
  PluginFactory.register('HybridReverb', HybridReverb, { category: 'Spatial' });
  PluginFactory.register('Echo', Echo, { category: 'Spatial' });

  // Distortion
  PluginFactory.register('Overdrive', Overdrive, { category: 'Distortion' });
  PluginFactory.register('Saturator', Saturator, { category: 'Distortion' });
  PluginFactory.register('Distortion', Distortion, { category: 'Distortion' });
  PluginFactory.register('Redux', Redux, { category: 'Distortion' });

  // Spectral
  PluginFactory.register('SpectralTime', SpectralTime, { category: 'Spectral' });
  PluginFactory.register('FrequencyShifter', SpectralFrequencyShifter, { category: 'Spectral' });

  // Creative - disabled due to CommonJS export issues
  // PluginFactory.register('RingModulator', RingModulator, { category: 'Creative' });
  // PluginFactory.register('PitchShifter', PitchShifter, { category: 'Creative' });
  // PluginFactory.register('Granular', Granular, { category: 'Creative' });

  // Utility - disabled due to CommonJS export issues
  // PluginFactory.register('Gain', Gain, { category: 'Utility' });
  // PluginFactory.register('Pan', Pan, { category: 'Utility' });
  // PluginFactory.register('StereoWidth', StereoWidth, { category: 'Utility' });

  // Analysis - disabled due to CommonJS export issues
  // PluginFactory.register('SpectrumAnalyzer', SpectrumAnalyzer, { category: 'Analysis' });

  // Modulation Matrix
  PluginFactory.register('AdvancedLFO', AdvancedLFO, { category: 'Modulation' });
  PluginFactory.register('EnvelopeGenerator', EnvelopeGenerator, { category: 'Modulation' });
  PluginFactory.register('MacroControls', MacroControls, { category: 'Modulation' });
  PluginFactory.register('ModulationMatrix', ModulationMatrix, { category: 'Modulation' });

  console.log('Registered plugins:', PluginFactory.getPluginNames());
  return { totalRegistered: PluginFactory.getPluginCount() };
}

export default registerAllPlugins;
