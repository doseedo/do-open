// Studio plugin: tube/triode preamps in the vintage folder
// Strategy: native DSP-lang `circuit_tube_preamp` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Tube Preamp',
  category: 'vintage',
  description: 'Tube preamp model',
  nodeType: 'circuit_tube_preamp',
  paramBindings: [
    { bindKey: 'drive',  paramId: 'drive',  label: 'Drive',  min: 0, max: 10, default: 3 },
    { bindKey: 'bias',   paramId: 'bias',   label: 'Bias',   min: 0, max: 10, default: 5 },
    { bindKey: 'output', paramId: 'output', label: 'Output', min: -24, max: 12, default: 0, unit: 'dB' },
  ],
});
