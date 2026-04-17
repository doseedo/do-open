// Studio plugin: VintageEQPultec
// Strategy: native DSP-lang `circuit_pultec_eq` node — already a
// modelled Pultec EQ in the DSP language. Direct map.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Pultec EQ',
  category: 'vintage',
  description: 'Pultec passive program EQ model',
  nodeType: 'circuit_pultec_eq',
  paramBindings: [
    { bindKey: 'low_boost',  paramId: 'low_boost',  label: 'Low Boost',  min: 0, max: 10, default: 0 },
    { bindKey: 'low_atten',  paramId: 'low_atten',  label: 'Low Atten',  min: 0, max: 10, default: 0 },
    { bindKey: 'high_boost', paramId: 'high_boost', label: 'High Boost', min: 0, max: 10, default: 0 },
    { bindKey: 'high_atten', paramId: 'high_atten', label: 'High Atten', min: 0, max: 10, default: 0 },
  ],
});
