// Studio plugin: Expander (dynamics)
// Strategy: native DSP-lang `expander` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Expander',
  category: 'dynamics',
  description: 'Downward expander',
  nodeType: 'expander',
  paramBindings: [
    { bindKey: 'threshold', paramId: 'threshold', label: 'Threshold', min: -60, max: 0,    default: -30, unit: 'dB' },
    { bindKey: 'ratio',     paramId: 'ratio',     label: 'Ratio',     min: 1,   max: 10,   default: 2,   unit: ':1' },
    { bindKey: 'attack',    paramId: 'attack',    label: 'Attack',    min: 0.1, max: 100,  default: 5,   unit: 'ms', skew: 0.3 },
    { bindKey: 'release',   paramId: 'release',   label: 'Release',   min: 1,   max: 2000, default: 200, unit: 'ms', skew: 0.3 },
  ],
});
