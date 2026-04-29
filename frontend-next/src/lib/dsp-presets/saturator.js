// Studio plugin: SaturatorPlugin
// Strategy: native DSP-lang `saturation` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Saturator',
  category: 'distortion',
  description: 'Hyperbolic-tangent saturation',
  nodeType: 'saturation',
  paramBindings: [
    { bindKey: 'amount', paramId: 'amount', label: 'Amount', min: 0, max: 1, default: 0.5 },
    { bindKey: 'mix',    paramId: 'mix',    label: 'Mix',    min: 0, max: 1, default: 1 },
  ],
});
