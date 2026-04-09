// Studio plugin: ConvolutionReverbPlugin
// Strategy: native DSP-lang `convolution` node (Web Audio ConvolverNode).
// The DSP-lang convolution node loads an IR file and convolves; identical
// to what the studio plugin does. No worklet needed.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Convolution Reverb',
  category: 'reverb',
  description: 'Impulse-response convolution reverb',
  nodeType: 'convolution',
  fixedParams: { ir_file: '' /* IR set per-instance via UI */ },
  paramBindings: [
    { bindKey: 'mix', paramId: 'mix', label: 'Mix', min: 0, max: 1, default: 0.4, unit: '' },
  ],
});
