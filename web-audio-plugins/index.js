/**
 * Web Audio Plugins - Main Entry Point
 *
 * A comprehensive library of 27 audio effect plugins + modulation system
 * All plugins extend BasePlugin and work with PluginFactory
 *
 * @version 1.0.0
 * @author Web Audio Plugins Team
 */

// ===== CORE INFRASTRUCTURE =====
export { default as BasePlugin } from './core/BasePlugin.js';
export { default as PluginFactory } from './core/PluginFactory.js';
export { default as Router } from './core/Router.js';
export { default as PresetManager } from './core/PresetManager.js';
export { default as ParamAutomation } from './core/ParamAutomation.js';
export { default as PerformanceMonitor } from './core/PerformanceMonitor.js';

// ===== MODULATION MATRIX =====
export {
  ModulationSource,
  AdvancedLFO,
  EnvelopeGenerator,
  MacroControls,
  ModulationMatrix
} from './modulation-matrix/index.js';

// ===== DYNAMICS PROCESSORS =====
export {
  Compressor,
  Gate,
  Limiter,
  GlueCompressor
} from './dynamics/index.js';

// ===== EQ & FILTERS =====
export { EQEight, EQThree } from './eq/index.js';
export { AutoFilter } from './filters/index.js';

// ===== DELAYS =====
export {
  SimpleDelay,
  PingPongDelay,
  FilterDelay
} from './delay/index.js';

// ===== MODULATION EFFECTS =====
export {
  Chorus,
  Flanger,
  Phaser,
  Tremolo
} from './modulation/index.js';

// ===== REVERB & SPATIAL =====
export {
  Reverb,
  HybridReverb,
  Echo
} from './reverb/index.js';

// ===== DISTORTION & SATURATION =====
export {
  Overdrive,
  Saturator,
  Distortion,
  Redux
} from './distortion/index.js';

// ===== SPECTRAL PROCESSING =====
export {
  SpectralTime,
  SpectralResonator,
  FrequencyShifter,
  Vocoder
} from './spectral/index.js';

// ===== UTILITY & ANALYSIS =====
export {
  Utility,
  SpectrumAnalyzer,
  Tuner,
  ChannelEQ
} from './utility/index.js';

/**
 * Library Statistics
 */
export const LIBRARY_INFO = {
  version: '1.0.0',
  totalPlugins: 38, // 27 effects + 6 core + 5 modulation matrix
  categories: {
    'Core Infrastructure': 6,
    'Modulation Matrix': 5,
    'Dynamics': 4,
    'EQ & Filters': 3,
    'Delays': 3,
    'Modulation Effects': 4,
    'Reverb & Spatial': 3,
    'Distortion & Saturation': 4,
    'Spectral Processing': 4,
    'Utility & Analysis': 4
  },
  features: [
    'Modular plugin architecture',
    'Custom signal routing',
    'Preset management',
    'Parameter automation',
    'Performance monitoring',
    'Real-time audio processing',
    'Browser-compatible (Chrome, Firefox, Safari)'
  ]
};
