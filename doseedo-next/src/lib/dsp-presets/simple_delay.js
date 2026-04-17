// Studio plugin: SimpleDelay (delay)
// Strategy: native DSP-lang `delay` node (already includes feedback + dry/wet).
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Simple Delay',
  category: 'delay',
  description: 'Mono feedback delay with mix',
  nodeType: 'delay',
  paramBindings: [
    { bindKey: 'time',     paramId: 'time',     label: 'Time',     min: 0.01, max: 2,  default: 0.3, unit: 's', skew: 0.4 },
    { bindKey: 'feedback', paramId: 'feedback', label: 'Feedback', min: 0,    max: 0.95, default: 0.4, unit: '' },
    { bindKey: 'mix',      paramId: 'mix',      label: 'Mix',      min: 0,    max: 1,  default: 0.3, unit: '' },
  ],
});
