// Helper builders for DSP-lang preset manifests.
// Each preset returns the same shape as a /plugins/create dspConfig:
//   { dspChain: [...nodes], parameters: [...UI param defs], routing: {...} }

const STEREO_ROUTING = { input: 'stereo', chain: [], output: 'stereo' };

/**
 * Build a parameter definition for the UI param panel.
 *
 * @param {string} id        — unique within the preset
 * @param {string} label     — UI display label
 * @param {number} min
 * @param {number} max
 * @param {number} dflt      — default value (in native units)
 * @param {string=} unit     — 'Hz', 'dB', 'ms', '%', 's', etc.
 * @param {number=} skew     — 1.0 = linear, <1 = log skew (more resolution near min)
 */
export function param(id, label, min, max, dflt, unit, skew = 1.0) {
  return { id, label, min, max, default: dflt, unit, skew };
}

/**
 * Wrap an existing AudioWorkletProcessor as a one-node DSP-lang preset.
 * Used for studio plugins whose math can't be expressed with vanilla
 * Web Audio nodes (FFT, phase vocoders, granular, etc.).
 */
export function workletPreset({
  name,
  category,
  description,
  processorName,     // e.g. 'reverb-processor'
  processorUrl,      // e.g. '/lib/web-audio-plugins/worklets/reverb-processor.js'
  paramBindings,     // array of {bindKey, paramId, label, min, max, default, unit, skew?}
  inputChannels = 2,
  outputChannels = 2,
}) {
  const nodeParams = {
    processor_name: processorName,
    processor_url: processorUrl,
    input_channels: inputChannels,
    output_channels: outputChannels,
  };
  for (const b of paramBindings) {
    nodeParams[b.bindKey] = '@' + b.paramId;
  }
  return {
    name,
    category,
    description,
    dspConfig: {
      dspChain: [
        { type: 'custom_worklet', id: 'core', params: nodeParams },
      ],
      parameters: paramBindings.map(b =>
        param(b.paramId, b.label, b.min, b.max, b.default, b.unit, b.skew)
      ),
      routing: STEREO_ROUTING,
    },
  };
}

/**
 * Wrap a single existing DSP-lang node as a preset (no worklet).
 * Used for studio plugins whose algorithm is already covered.
 *
 *   nativePreset({
 *     name: 'Compressor',
 *     nodeType: 'compressor',
 *     paramBindings: [
 *       {bindKey: 'threshold', paramId: 'threshold', label: 'Threshold', min: -60, max: 0, default: -20, unit: 'dB'},
 *       ...
 *     ],
 *   })
 */
export function nativePreset({
  name,
  category,
  description,
  nodeType,
  fixedParams = {},   // params set as constants, not bound to UI
  paramBindings,
}) {
  const nodeParams = { ...fixedParams };
  for (const b of paramBindings) {
    nodeParams[b.bindKey] = '@' + b.paramId;
  }
  return {
    name,
    category,
    description,
    dspConfig: {
      dspChain: [
        { type: nodeType, id: 'core', params: nodeParams },
      ],
      parameters: paramBindings.map(b =>
        param(b.paramId, b.label, b.min, b.max, b.default, b.unit, b.skew)
      ),
      routing: STEREO_ROUTING,
    },
  };
}

/**
 * Wrap multiple DSP-lang nodes in a series chain as one preset.
 * Used when a studio plugin combines (e.g.) tone EQ + drive (Distortion
 * has pre/post biquads + waveshaper).
 *
 *   seriesPreset({
 *     name: 'Distortion',
 *     nodes: [
 *       { type: 'highpass', id: 'pre',  params: { cutoff: '@hpf' } },
 *       { type: 'waveshaper', id: 'shape', params: { drive: '@drive', curve: 'tanh' } },
 *       { type: 'lowpass',  id: 'post', params: { cutoff: '@tone' } },
 *     ],
 *     parameters: [param('hpf','HP',20,2000,80,'Hz',0.3), ...],
 *   })
 */
export function seriesPreset({ name, category, description, nodes, parameters }) {
  return {
    name,
    category,
    description,
    dspConfig: {
      dspChain: nodes,
      parameters,
      routing: STEREO_ROUTING,
    },
  };
}

// Resolve a worklet URL relative to the repo root. The runtime engine
// will pass this directly to ctx.audioWorklet.addModule().
export function workletUrl(file) {
  return `/lib/web-audio-plugins/worklets/${file}`;
}
