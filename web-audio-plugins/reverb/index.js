/**
 * Reverb & Spatial Effects - Index
 *
 * Exports all reverb and spatial effects
 * Includes both legacy (native Web Audio) and modern (AudioWorklet) implementations
 */

// Legacy implementations (native Web Audio API)
export { default as Reverb } from './Reverb.js';
export { default as HybridReverb } from './HybridReverb.js';
export { default as Echo } from './Echo.js';

// Modern AudioWorklet implementations (recommended for production)
export { ReverbPlugin } from './ReverbPlugin.js';
export { ConvolutionReverbPlugin } from './ConvolutionReverbPlugin.js';
export { ReverbPresets } from './ReverbPlugin.js';
