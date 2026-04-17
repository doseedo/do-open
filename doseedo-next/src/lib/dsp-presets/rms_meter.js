// Studio plugin: RMS Meter
// Strategy: native DSP-lang `rms_meter` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'RMS Meter',
  category: 'analysis',
  description: 'Pass-through RMS meter',
  nodeType: 'rms_meter',
  paramBindings: [
    { bindKey: 'window_ms', paramId: 'window_ms', label: 'Window', min: 1, max: 500, default: 50, unit: 'ms' },
  ],
});
