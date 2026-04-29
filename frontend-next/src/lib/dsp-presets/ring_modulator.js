// Studio plugin: RingModulatorPlugin
// Strategy: native DSP-lang `ring_mod` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Ring Modulator',
  category: 'modulation',
  description: 'Multiply input by carrier oscillator',
  nodeType: 'ring_mod',
  paramBindings: [
    { bindKey: 'frequency', paramId: 'frequency', label: 'Carrier', min: 1, max: 5000, default: 200, unit: 'Hz', skew: 0.3 },
    { bindKey: 'depth',     paramId: 'depth',     label: 'Depth',   min: 0, max: 1,    default: 1 },
    { bindKey: 'mix',       paramId: 'mix',       label: 'Mix',     min: 0, max: 1,    default: 1 },
  ],
});
