// Studio plugin: Compressor (dynamics)
// Strategy: native DSP-lang preset wrapping the built-in `compressor`
// node. No worklet — the DSP-lang node uses the same DynamicsCompressor
// primitive that the studio plugin's worklet was emulating.

import { nativePreset } from './_helpers.js';

export default nativePreset({
  name: 'Compressor',
  category: 'dynamics',
  description: 'Soft-knee compressor (DynamicsCompressorNode)',
  nodeType: 'compressor',
  paramBindings: [
    { bindKey: 'threshold', paramId: 'threshold', label: 'Threshold', min: -60, max: 0,   default: -20, unit: 'dB' },
    { bindKey: 'ratio',     paramId: 'ratio',     label: 'Ratio',     min: 1,   max: 20,  default: 4,   unit: ':1' },
    { bindKey: 'attack',    paramId: 'attack',    label: 'Attack',    min: 0.1, max: 1000, default: 3,  unit: 'ms', skew: 0.3 },
    { bindKey: 'release',   paramId: 'release',   label: 'Release',   min: 1,   max: 2000, default: 250,unit: 'ms', skew: 0.3 },
    { bindKey: 'knee',      paramId: 'knee',      label: 'Knee',      min: 0,   max: 40,  default: 30,  unit: 'dB' },
    { bindKey: 'makeup',    paramId: 'makeup',    label: 'Makeup',    min: -12, max: 24,  default: 0,   unit: 'dB' },
  ],
});
