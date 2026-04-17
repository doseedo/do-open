// Studio plugin: DistortionPlugin (waveshaping + tone filters)
// Strategy: series chain — pre highpass → waveshaper → post lowpass.
// Reproduces the studio plugin's tone-shaping topology with native nodes.
import { seriesPreset, param } from './_helpers.js';
export default seriesPreset({
  name: 'Distortion',
  category: 'distortion',
  description: 'Waveshaper distortion with pre/post tone filters',
  nodes: [
    { type: 'highpass',   id: 'pre',   params: { cutoff: '@hp', resonance: 0.3 } },
    { type: 'waveshaper', id: 'shape', params: { drive: '@drive', curve: 'tanh' } },
    { type: 'lowpass',    id: 'post',  params: { cutoff: '@tone', resonance: 0.3 } },
    { type: 'gain',       id: 'level', params: { gain: '@output' } },
  ],
  parameters: [
    param('drive',  'Drive',  1,    50,    10,   '×',  0.4),
    param('hp',     'HP',     20,   2000,  80,   'Hz', 0.4),
    param('tone',   'Tone',   500,  20000, 8000, 'Hz', 0.4),
    param('output', 'Output', -24,  12,    0,    'dB'),
  ],
});
