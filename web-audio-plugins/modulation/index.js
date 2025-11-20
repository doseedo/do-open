/**
 * Modulation Effects - Index
 *
 * Exports all modulation effects
 * Updated by Agent 4 to include AudioWorklet versions
 */

// Legacy Web Audio API versions (for backwards compatibility)
export { default as Chorus } from './Chorus.js';
export { default as Flanger } from './Flanger.js';
export { default as Phaser } from './Phaser.js';
export { default as Tremolo } from './Tremolo.js';

// Modern AudioWorklet versions (recommended for production)
export { default as ChorusPlugin } from './ChorusPlugin.js';
export { default as FlangerPlugin } from './FlangerPlugin.js';
export { default as PhaserPlugin } from './PhaserPlugin.js';
export { default as TremoloPlugin } from './TremoloPlugin.js';
