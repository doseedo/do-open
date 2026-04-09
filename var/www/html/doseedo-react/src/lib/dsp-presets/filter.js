// Studio plugin: Filter (multi-mode biquad)
// Strategy: native DSP-lang `lowpass` node by default; the studio
// plugin's "type" knob (LP/HP/BP/notch) maps to morphing the filter
// node type at the host level. For a single preset we use lowpass.
// Authors who want HP/BP/notch can clone and swap the node type, or
// we can introduce a `filter_morph` preset later.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Lowpass Filter',
  category: 'eq',
  description: '12 dB/oct lowpass biquad',
  nodeType: 'lowpass',
  paramBindings: [
    { bindKey: 'cutoff',    paramId: 'cutoff',    label: 'Cutoff', min: 20, max: 20000, default: 1000, unit: 'Hz', skew: 0.25 },
    { bindKey: 'resonance', paramId: 'resonance', label: 'Q',      min: 0,  max: 1,     default: 0.5 },
  ],
});
