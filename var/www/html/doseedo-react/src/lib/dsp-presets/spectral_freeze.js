// Studio plugin: Spectral Freeze
// Strategy: native DSP-lang `spectral_freeze` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Spectral Freeze',
  category: 'spectral',
  description: 'Freeze the spectrum at a moment in time',
  nodeType: 'spectral_freeze',
  paramBindings: [
    { bindKey: 'freeze', paramId: 'freeze', label: 'Freeze', min: 0, max: 1, default: 0 },
    { bindKey: 'mix',    paramId: 'mix',    label: 'Mix',    min: 0, max: 1, default: 0.5 },
  ],
});
