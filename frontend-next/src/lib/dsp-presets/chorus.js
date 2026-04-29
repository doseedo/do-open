// Studio plugin: ChorusPlugin
// Strategy: native DSP-lang `chorus` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Chorus',
  category: 'modulation',
  description: 'LFO-modulated multi-tap chorus',
  nodeType: 'chorus',
  paramBindings: [
    { bindKey: 'rate',  paramId: 'rate',  label: 'Rate',  min: 0.01, max: 10, default: 1.5, unit: 'Hz', skew: 0.4 },
    { bindKey: 'depth', paramId: 'depth', label: 'Depth', min: 0,    max: 1,  default: 0.5 },
    { bindKey: 'mix',   paramId: 'mix',   label: 'Mix',   min: 0,    max: 1,  default: 0.5 },
  ],
});
