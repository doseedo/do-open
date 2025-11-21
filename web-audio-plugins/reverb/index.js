/**
 * Reverb & Spatial Effects - Index
 *
 * Exports all reverb and spatial effects
 * Includes both legacy (Web Audio API) and AudioWorklet implementations
 */

// Legacy plugins (Web Audio API)
export { default as Reverb } from './Reverb.js';
export { default as HybridReverb } from './HybridReverb.js';
export { default as Echo } from './Echo.js';

// AudioWorklet plugins (optimized for performance)
export { ReverbPlugin } from './ReverbPlugin.js';
export { HybridReverbPlugin } from './HybridReverbPlugin.js';
export { EchoPlugin } from './EchoPlugin.js';
