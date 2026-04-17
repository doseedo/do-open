// Studio plugin: ReverbPlugin (Schroeder algorithmic reverb)
// Strategy: custom_worklet wrap. The DSP-lang `reverb` node uses a
// different algorithm; to preserve the existing studio sound exactly,
// we host the existing reverb-processor.js worklet via custom_worklet.
//
// Source plugin: src/lib/web-audio-plugins/reverb/ReverbPlugin.js
// Source worklet: src/lib/web-audio-plugins/worklets/reverb-processor.js

import { workletPreset, workletUrl } from './_helpers.js';

export default workletPreset({
  name: 'Reverb',
  category: 'reverb',
  description: 'Algorithmic Schroeder reverb (parallel comb + series allpass)',
  processorName: 'reverb-processor',
  processorUrl: workletUrl('reverb-processor.js'),
  paramBindings: [
    { bindKey: 'preDelay',  paramId: 'preDelay',  label: 'Pre-Delay',  min: 0,    max: 0.2, default: 0,    unit: 's' },
    { bindKey: 'decayTime', paramId: 'decayTime', label: 'Decay',      min: 0.1,  max: 10,  default: 2.0,  unit: 's',  skew: 0.4 },
    { bindKey: 'size',      paramId: 'size',      label: 'Size',       min: 0,    max: 100, default: 50,   unit: '%' },
    { bindKey: 'diffusion', paramId: 'diffusion', label: 'Diffusion',  min: 0,    max: 100, default: 70,   unit: '%' },
    { bindKey: 'damping',   paramId: 'damping',   label: 'Damping',    min: 0,    max: 100, default: 50,   unit: '%' },
    { bindKey: 'mix',       paramId: 'mix',       label: 'Mix',        min: 0,    max: 100, default: 30,   unit: '%' },
  ],
});
