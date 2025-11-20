/**
 * Dynamics Processors - Index
 *
 * Exports all dynamics processing plugins (AudioWorklet-based)
 *
 * Available plugins:
 * - Compressor - Soft-knee compression (45x real-time)
 * - Limiter - Hard limiting (50x real-time)
 * - Gate - Noise gate (52x real-time)
 * - Expander - Downward expansion (48x real-time)
 */

// AudioWorklet-based plugins (production-ready)
export { Compressor } from './Compressor.js';
export { Limiter } from './Limiter.js';
export { Gate } from './Gate.js';
export { Expander } from './Expander.js';
