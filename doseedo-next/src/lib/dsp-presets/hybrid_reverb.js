// Studio plugin: HybridReverbPlugin (Schroeder + convolution)
// Strategy: custom_worklet wrap — combined algorithm doesn't map to a
// single DSP-lang node. (Could be expressed as a 2-node graph, but the
// existing worklet has the calibrated mix curves baked in.)
import { workletPreset, workletUrl } from './_helpers.js';
export default workletPreset({
  name: 'Hybrid Reverb',
  category: 'reverb',
  description: 'Algorithmic + convolution reverb blend',
  processorName: 'hybrid-reverb-processor',
  processorUrl: workletUrl('hybrid-reverb-processor.js'),
  paramBindings: [
    { bindKey: 'size',      paramId: 'size',      label: 'Size',      min: 0,   max: 100, default: 50 },
    { bindKey: 'damping',   paramId: 'damping',   label: 'Damping',   min: 0,   max: 100, default: 50 },
    { bindKey: 'algBlend',  paramId: 'algBlend',  label: 'Algo↔Conv', min: 0,   max: 1,   default: 0.5 },
    { bindKey: 'preDelay',  paramId: 'preDelay',  label: 'Pre-Delay', min: 0,   max: 0.2, default: 0.02, unit: 's' },
    { bindKey: 'mix',       paramId: 'mix',       label: 'Mix',       min: 0,   max: 100, default: 30 },
  ],
});
