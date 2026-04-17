// Studio plugin: TapeEmulation
// Strategy: native DSP-lang `circuit_tape_machine` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Tape',
  category: 'vintage',
  description: 'Analog tape machine emulation (saturation + wow/flutter)',
  nodeType: 'circuit_tape_machine',
  paramBindings: [
    { bindKey: 'drive',  paramId: 'drive',  label: 'Drive',  min: 0, max: 10, default: 5 },
    { bindKey: 'bias',   paramId: 'bias',   label: 'Bias',   min: 0, max: 10, default: 5 },
    { bindKey: 'wow',    paramId: 'wow',    label: 'Wow',    min: 0, max: 1,  default: 0.1 },
    { bindKey: 'flutter',paramId: 'flutter',label: 'Flutter',min: 0, max: 1,  default: 0.1 },
    { bindKey: 'speed',  paramId: 'speed',  label: 'Speed',  min: 0, max: 3,  default: 1 /* 0=7.5ips 1=15ips 2=30ips */ },
  ],
});
