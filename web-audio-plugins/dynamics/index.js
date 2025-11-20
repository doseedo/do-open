/**
 * Dynamics Processors - Index
 *
 * Exports all dynamics processing plugins
 *
 * Legacy plugins (Web Audio API):
 * - Compressor
 * - Gate
 * - Limiter
 * - GlueCompressor
 *
 * AudioWorklet-based plugins (recommended):
 * - CompressorPlugin
 * - LimiterPlugin
 * - GatePlugin
 * - ExpanderPlugin
 */

// Legacy plugins (Web Audio API)
export { default as Compressor } from './Compressor.js';
export { default as Gate } from './Gate.js';
export { default as Limiter } from './Limiter.js';
export { default as GlueCompressor } from './GlueCompressor.js';

// AudioWorklet-based plugins (recommended)
export { CompressorPlugin } from './CompressorPlugin.js';
export { LimiterPlugin } from './LimiterPlugin.js';
export { GatePlugin } from './GatePlugin.js';
export { ExpanderPlugin } from './ExpanderPlugin.js';
