/**
 * EQ - Index
 *
 * Exports all EQ plugins
 */

// Legacy plugins (Web Audio API nodes)
export { default as EQEight } from './EQEight.js';
export { default as EQThree } from './EQThree.js';

// New AudioWorklet-based plugins (high performance)
export { EQPlugin } from './EQPlugin.js';
export { GraphicEQPlugin } from './GraphicEQPlugin.js';
export { FilterPlugin } from './FilterPlugin.js';
