// Studio plugin: Redux (bitcrusher)
// Strategy: native DSP-lang `bitcrusher` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Redux',
  category: 'distortion',
  description: 'Bitcrusher / sample-rate reducer',
  nodeType: 'bitcrusher',
  paramBindings: [
    { bindKey: 'bits',         paramId: 'bits',         label: 'Bits',         min: 1, max: 16,    default: 8 },
    { bindKey: 'downsample',   paramId: 'downsample',   label: 'Downsample',   min: 1, max: 50,    default: 4 },
  ],
});
