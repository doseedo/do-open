// Studio plugin: GraphicEQ (10-band graphic equaliser)
// Strategy: series chain of `parametric_eq` nodes (one per band).
// Each band's gain is exposed as its own UI param; frequencies are
// fixed ISO octave centres.
import { seriesPreset, param } from './_helpers.js';

const BANDS = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];

const nodes = BANDS.map((freq, i) => ({
  type: 'parametric_eq',
  id: `band_${i}`,
  params: {
    frequency: freq,
    q: 1.4,
    gain: '@band_' + i,
  },
}));

const parameters = BANDS.map((freq, i) =>
  param('band_' + i, `${freq < 1000 ? freq : (freq / 1000) + 'k'} Hz`, -18, 18, 0, 'dB')
);

export default seriesPreset({
  name: 'Graphic EQ',
  category: 'eq',
  description: '10-band ISO graphic EQ',
  nodes,
  parameters,
});
