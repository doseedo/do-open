// Studio plugin: PhaserPlugin
// Strategy: native DSP-lang `phaser` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Phaser',
  category: 'modulation',
  description: 'Cascaded allpass + LFO',
  nodeType: 'phaser',
  paramBindings: [
    { bindKey: 'rate',     paramId: 'rate',     label: 'Rate',     min: 0.01, max: 10,   default: 0.5, unit: 'Hz', skew: 0.4 },
    { bindKey: 'depth',    paramId: 'depth',    label: 'Depth',    min: 0,    max: 1,    default: 0.7 },
    { bindKey: 'feedback', paramId: 'feedback', label: 'Feedback', min: 0,    max: 0.95, default: 0.5 },
    { bindKey: 'stages',   paramId: 'stages',   label: 'Stages',   min: 2,    max: 12,   default: 6 },
    { bindKey: 'mix',      paramId: 'mix',      label: 'Mix',      min: 0,    max: 1,    default: 0.5 },
  ],
});
