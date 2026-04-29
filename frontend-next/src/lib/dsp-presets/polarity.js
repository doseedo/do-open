// Studio plugin: Polarity (utility)
// Strategy: a single `gain` node whose constant gain value is set
// from the UI param. Inverting = setting gain to -1; bypass = +1.
// (No DSP-lang `math_multiply` needed; gain handles ±1 multiplication.)
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Polarity',
  category: 'utility',
  description: 'Invert signal polarity (×-1)',
  nodeType: 'gain',
  paramBindings: [
    // -1 = inverted, +1 = normal. UI exposes this as a 2-state switch.
    { bindKey: 'gain', paramId: 'invert', label: 'Invert', min: -1, max: 1, default: 1 },
  ],
});
