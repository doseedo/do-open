// Studio plugin: PingPongDelay (delay)
// Strategy: native DSP-lang `ping_pong_delay` node.
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Ping-Pong Delay',
  category: 'delay',
  description: 'Stereo bouncing delay',
  nodeType: 'ping_pong_delay',
  paramBindings: [
    { bindKey: 'time',     paramId: 'time',     label: 'Time',     min: 0.01, max: 2,    default: 0.4, unit: 's', skew: 0.4 },
    { bindKey: 'feedback', paramId: 'feedback', label: 'Feedback', min: 0,    max: 0.95, default: 0.4, unit: '' },
    { bindKey: 'mix',      paramId: 'mix',      label: 'Mix',      min: 0,    max: 1,    default: 0.4, unit: '' },
  ],
});
