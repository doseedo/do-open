/**
 * R13 — Phase Distortion processor (worklet stub).
 *
 * The default `phase_distortion` node implementation in
 * `src/audio/builders/r13_phase_distortion.js` is a stereo pair of
 * WaveShaperNodes whose `curve` Float32Arrays get regenerated on every
 * shape-param mutation. That covers all of Logic Pro's stock Phase Distortion
 * use-cases and gets us audio-faithful output through the existing R12
 * null-diff harness.
 *
 * This worklet is reserved for a *future* enhancement: a per-sample variant
 * that drives the curve-family index continuously (so `pd_curve` itself can
 * be modulated at audio rate, or interpolated between two families). The
 * WaveShaperNode approach can't do that — its curve is only re-applied
 * when assignment happens, and assignment is a control-rate operation. A
 * worklet is the only way to interpolate curve-family at sample rate.
 *
 * Until then this file simply registers the processor name so the engine
 * can `addModule(...)` it without error. The current builder does NOT
 * instantiate this worklet — see r13_phase_distortion.js, which uses the
 * native WaveShaperNode path exclusively.
 *
 * @author Doseedo R13
 */

/* eslint-disable no-undef */

// Mirror of curve-family enum in the builder. Keep order in sync.
const FAMILY_NAMES = ['saw', 'square', 'pulse', 'res1', 'res2', 'res3'];

class R13PhaseDistortionProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      // 0..(N-1) where N = FAMILY_NAMES.length. Float so we can crossfade.
      { name: 'family',     defaultValue: 0,   minValue: 0,    maxValue: FAMILY_NAMES.length - 1, automationRate: 'a-rate' },
      { name: 'amount',     defaultValue: 0.5, minValue: 0,    maxValue: 1,                       automationRate: 'a-rate' },
      { name: 'asymmetry',  defaultValue: 0,   minValue: -1,   maxValue: 1,                       automationRate: 'a-rate' },
      { name: 'pre_gain',   defaultValue: 1,   minValue: 0,    maxValue: 16,                      automationRate: 'a-rate' },
      { name: 'post_gain',  defaultValue: 1,   minValue: 0,    maxValue: 16,                      automationRate: 'a-rate' },
      { name: 'mix',        defaultValue: 1,   minValue: 0,    maxValue: 1,                       automationRate: 'a-rate' },
    ];
  }

  // Stub: passthrough. Real curve evaluation will live here when a
  // sample-accurate path is needed (e.g. for `pd_curve` modulation at
  // audio rate / curve-family interpolation).
  process(inputs, outputs /*, parameters */) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || !input.length || !output) return true;

    const channels = Math.min(input.length, output.length);
    for (let ch = 0; ch < channels; ch++) {
      const inCh = input[ch];
      const outCh = output[ch];
      if (!inCh || !outCh) continue;
      // Plain passthrough until the sample-accurate variant lands.
      outCh.set(inCh);
    }
    return true;
  }
}

registerProcessor('r13-phase-distortion-processor', R13PhaseDistortionProcessor);
