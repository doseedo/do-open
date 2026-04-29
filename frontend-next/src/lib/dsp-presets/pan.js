// Studio plugin: Pan (utility)
// Strategy: native DSP-lang `pan` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Pan',
  category: 'utility',
  description: 'Stereo panner',
  nodeType: 'pan',
  paramBindings: [
    { bindKey: 'pan', paramId: 'pan', label: 'Pan', min: -1, max: 1, default: 0, unit: '' },
  ],
});
