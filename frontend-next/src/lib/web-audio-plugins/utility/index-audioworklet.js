/**
 * Utility Plugins (AudioWorklet Version)
 * Modern, high-performance utility plugins using AudioWorklet
 *
 * Export all utility plugins for easy importing
 */

// Import plugins
import { GainPlugin } from './GainPlugin.js';
import { PanPlugin } from './PanPlugin.js';
import { PolarityPlugin } from './PolarityPlugin.js';
import { StereoWidthPlugin } from './StereoWidthPlugin.js';

// Export all plugins
export {
  GainPlugin,
  PanPlugin,
  PolarityPlugin,
  StereoWidthPlugin
};

// Default export for convenience
export default {
  GainPlugin,
  PanPlugin,
  PolarityPlugin,
  StereoWidthPlugin
};
