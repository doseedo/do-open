// Studio plugin: Meter (peak metering, pass-through insert)
// Strategy: native DSP-lang `peak_meter` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Peak Meter',
  category: 'analysis',
  description: 'Pass-through peak meter',
  nodeType: 'peak_meter',
  paramBindings: [],
});
