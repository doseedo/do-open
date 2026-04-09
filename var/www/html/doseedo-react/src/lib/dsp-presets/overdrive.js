// Studio plugin: OverdrivePlugin
// Strategy: native DSP-lang `overdrive` node (soft-clip waveshaper).
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Overdrive',
  category: 'distortion',
  description: 'Soft-clipping overdrive',
  nodeType: 'overdrive',
  paramBindings: [
    { bindKey: 'drive',  paramId: 'drive',  label: 'Drive',  min: 1,   max: 50, default: 5, unit: '×', skew: 0.4 },
    { bindKey: 'tone',   paramId: 'tone',   label: 'Tone',   min: 200, max: 12000, default: 4000, unit: 'Hz', skew: 0.4 },
    { bindKey: 'output', paramId: 'output', label: 'Output', min: -24, max: 12, default: 0, unit: 'dB' },
  ],
});
