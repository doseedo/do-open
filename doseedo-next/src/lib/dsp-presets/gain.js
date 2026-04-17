// Studio plugin: Gain (utility)
// Strategy: native DSP-lang `gain` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Gain',
  category: 'utility',
  description: 'Linear gain stage',
  nodeType: 'gain',
  paramBindings: [
    { bindKey: 'gain', paramId: 'gain', label: 'Gain', min: -60, max: 24, default: 0, unit: 'dB' },
  ],
});
