// Studio plugin: StereoWidthPlugin (mid/side stereo width)
// Strategy: custom_worklet wrap of stereo-width-processor.js. The
// existing worklet uses postMessage params (no AudioParam descriptors)
// — buildCustomWorklet falls back to message-based param setting
// transparently. No source mods needed.
import { workletPreset } from './_helpers.js';
export default workletPreset({
  name: 'Stereo Width',
  category: 'utility',
  description: 'Mid/side stereo width control (0% mono → 200% extra wide)',
  processorName: 'stereo-width-processor',
  processorUrl: '/web-audio-plugins/utility/worklets/stereo-width-processor.js',
  paramBindings: [
    { bindKey: 'width', paramId: 'width', label: 'Width', min: 0, max: 2, default: 1, unit: '×' },
  ],
});
