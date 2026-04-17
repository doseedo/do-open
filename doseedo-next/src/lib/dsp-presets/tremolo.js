// Studio plugin: TremoloPlugin
// Strategy: native DSP-lang `tremolo` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Tremolo',
  category: 'modulation',
  description: 'Amplitude modulation by LFO',
  nodeType: 'tremolo',
  paramBindings: [
    { bindKey: 'rate',  paramId: 'rate',  label: 'Rate',  min: 0.1, max: 20, default: 5, unit: 'Hz', skew: 0.4 },
    { bindKey: 'depth', paramId: 'depth', label: 'Depth', min: 0,   max: 1,  default: 0.5 },
    { bindKey: 'shape', paramId: 'shape', label: 'Shape', min: 0,   max: 3,  default: 0 /* 0=sine 1=tri 2=saw 3=square */ },
  ],
});
