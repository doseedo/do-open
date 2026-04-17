// Studio plugin: EchoPlugin (multitap delay)
// Strategy: custom_worklet wrap (echo-processor.js) — DSP-lang's
// multitap_delay doesn't expose per-tap params; the worklet does.
import { workletPreset, workletUrl } from './_helpers.js';
export default workletPreset({
  name: 'Echo',
  category: 'delay',
  description: 'Multi-tap echo with independent tap times',
  processorName: 'echo-processor',
  processorUrl: workletUrl('echo-processor.js'),
  paramBindings: [
    { bindKey: 'time',     paramId: 'time',     label: 'Time',     min: 0.01, max: 2,    default: 0.4, unit: 's', skew: 0.4 },
    { bindKey: 'feedback', paramId: 'feedback', label: 'Feedback', min: 0,    max: 0.95, default: 0.4 },
    { bindKey: 'taps',     paramId: 'taps',     label: 'Taps',     min: 1,    max: 8,    default: 3 },
    { bindKey: 'mix',      paramId: 'mix',      label: 'Mix',      min: 0,    max: 1,    default: 0.3 },
  ],
});
