// Studio plugin: EQ (parametric peaking band)
// Strategy: native DSP-lang `parametric_eq` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Parametric EQ',
  category: 'eq',
  description: 'Single peaking band',
  nodeType: 'parametric_eq',
  paramBindings: [
    { bindKey: 'frequency', paramId: 'frequency', label: 'Frequency', min: 20, max: 20000, default: 1000, unit: 'Hz', skew: 0.25 },
    { bindKey: 'q',         paramId: 'q',         label: 'Q',         min: 0.1, max: 10,   default: 1 },
    { bindKey: 'gain',      paramId: 'gain',      label: 'Gain',      min: -24, max: 24,   default: 0, unit: 'dB' },
  ],
});
