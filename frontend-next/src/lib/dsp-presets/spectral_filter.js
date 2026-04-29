// Studio plugin: Spectral Filter
// Strategy: native DSP-lang `spectral_filter` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Spectral Filter',
  category: 'spectral',
  description: 'FFT-bin pass/stop filter',
  nodeType: 'spectral_filter',
  paramBindings: [
    { bindKey: 'low_bin',  paramId: 'low_bin',  label: 'Low Bin',  min: 0, max: 1, default: 0 },
    { bindKey: 'high_bin', paramId: 'high_bin', label: 'High Bin', min: 0, max: 1, default: 1 },
    { bindKey: 'mix',      paramId: 'mix',      label: 'Mix',      min: 0, max: 1, default: 1 },
  ],
});
