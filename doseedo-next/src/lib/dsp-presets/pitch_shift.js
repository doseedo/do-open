// Studio plugin: PitchShifter
// Strategy: native DSP-lang `pitch_shift` node — already in the
// language. (The studio's PitchShifter has its own inline worklet,
// but the DSP-lang node is feature-equivalent and avoids needing
// custom_worklet.)
import { nativePreset } from './_helpers.js';
export default nativePreset({
  name: 'Pitch Shift',
  category: 'spectral',
  description: 'Time-domain pitch shift in semitones',
  nodeType: 'pitch_shift',
  paramBindings: [
    { bindKey: 'semitones', paramId: 'semitones', label: 'Semitones', min: -24, max: 24, default: 0 },
    { bindKey: 'mix',       paramId: 'mix',       label: 'Mix',       min: 0,   max: 1,  default: 1 },
  ],
});
