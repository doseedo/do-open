/**
 * Distortion & Saturation - Index
 *
 * Exports all distortion and saturation plugins
 * - Legacy plugins (Distortion, Overdrive, Saturator, Redux)
 * - AudioWorklet plugins (DistortionPlugin, OverdrivePlugin, SaturatorPlugin)
 */

// Legacy plugins (Web Audio API based)
export { default as Overdrive } from './OverdrivePlugin.js';
export { SaturatorPlugin as Saturator } from './Saturator.js';
export { DistortionPlugin as Distortion } from './Distortion.js';
export { ReduxPlugin as Redux } from './Redux.js';

// AudioWorklet plugins (high performance)
export { DistortionPlugin } from './DistortionPlugin.js';
export { OverdrivePlugin } from './OverdrivePlugin.js';
export { SaturatorPlugin } from './SaturatorPlugin.js';
