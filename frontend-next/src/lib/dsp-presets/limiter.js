// Studio plugin: Limiter (dynamics)
// Strategy: native DSP-lang `limiter` node (aggressive DynamicsCompressor).
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Limiter',
  category: 'dynamics',
  description: 'Brick-wall limiter (high-ratio DynamicsCompressor)',
  nodeType: 'limiter',
  paramBindings: [
    { bindKey: 'ceiling', paramId: 'ceiling', label: 'Ceiling', min: -24, max: 0,    default: -0.3, unit: 'dB' },
    { bindKey: 'release', paramId: 'release', label: 'Release', min: 1,   max: 1000, default: 50,   unit: 'ms', skew: 0.3 },
  ],
});
