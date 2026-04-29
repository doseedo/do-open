/**
 * Modulation Effects - Index
 *
 * Exports all modulation effects (legacy and AudioWorklet versions)
 */

// Legacy implementations
export { default as Chorus } from './Chorus.js';
export { default as Flanger } from './Flanger.js';
export { default as Phaser } from './Phaser.js';
export { default as Tremolo } from './Tremolo.js';

// AudioWorklet implementations
export { ChorusPlugin } from './ChorusPlugin.js';
export { FlangerPlugin } from './FlangerPlugin.js';
export { PhaserPlugin } from './PhaserPlugin.js';
export { TremoloPlugin } from './TremoloPlugin.js';
