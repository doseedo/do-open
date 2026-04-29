// Studio plugin: Gate (dynamics)
// Strategy: native DSP-lang `gate` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Gate',
  category: 'dynamics',
  description: 'Noise gate with threshold/range',
  nodeType: 'gate',
  paramBindings: [
    { bindKey: 'threshold', paramId: 'threshold', label: 'Threshold', min: -80, max: 0,    default: -40, unit: 'dB' },
    { bindKey: 'attack',    paramId: 'attack',    label: 'Attack',    min: 0.1, max: 100,  default: 1,   unit: 'ms', skew: 0.3 },
    { bindKey: 'release',   paramId: 'release',   label: 'Release',   min: 1,   max: 2000, default: 100, unit: 'ms', skew: 0.3 },
  ],
});
