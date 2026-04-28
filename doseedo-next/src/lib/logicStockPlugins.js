/**
 * Authoritative list of Logic Pro stock-plugin names for the Add-plugin
 * autocomplete. Populated to match
 * `tools/calibration/auto_driver/registry.py` so the web side surfaces
 * the same vocabulary the desktop calibration harness understands.
 *
 * Why this lives in the JS tree instead of being fetched: the names
 * change on the order of "once per Logic major release" (rarely), and
 * the alternative — exposing the desktop's plugin menu cache through an
 * auth-service endpoint — would be a lot of plumbing for a list whose
 * maintenance cost is one PR per Logic version. A free-text input
 * supplements this list, so any plugin not listed here still
 * broadcasts correctly when typed; the list is just an autocomplete
 * convenience, not a gate.
 *
 * Categories are advisory — the Add-plugin picker shows the flat list
 * and lets the operator scan; categories are surfaced only if the rack
 * UI ever wants grouped suggestions.
 */

export const LOGIC_STOCK_PLUGINS = [
  // Dynamics
  'Compressor',
  'Limiter',
  'Adaptive Limiter',
  'Multipressor',
  'DeEsser 2',
  'Enveloper',
  'Noise Gate',
  'Expander',

  // EQ
  'Channel EQ',
  'Linear Phase EQ',
  'Match EQ',
  'Vintage Console EQ',
  'Vintage Graphic EQ',
  'Vintage Tube EQ',
  'Vintage 1073',
  'Vintage API',

  // Reverb
  'ChromaVerb',
  'Space Designer',
  'PlatinumVerb',
  'SilverVerb',

  // Delay
  'Tape Delay',
  'Stereo Delay',
  'Sample Delay',
  'Delay Designer',
  'Modulation Delay',

  // Modulation
  'Chorus',
  'Ensemble',
  'Flanger',
  'Phaser',
  'Microphaser',
  'Ringshifter',
  'Rotor Cabinet',
  'Scanner Vibrato',
  'Spreader',
  'Tremolo',

  // Distortion / saturation
  'Bitcrusher',
  'Clip Distortion',
  'Distortion',
  'Distortion II',
  'Overdrive',
  'Phase Distortion',
  'Tube Distortion',

  // Amp / pedalboard
  'Amp Designer',
  'Bass Amp Designer',
  'Vintage Amp Modeling',
  'Pedalboard',
  'Bass Amp',
  'Guitar Amp Pro',

  // Pitch / vocal
  'Pitch Correction',
  'Pitch Shifter',
  'Vocal Transformer',
  'Vocoder',

  // Spectral
  'Spectral Gate',
  'ESS',

  // Imaging / utilities
  'Direction Mixer',
  'Stereo Spread',
  'Gain',
  'Test Oscillator',
  'I/O',
  'Tuner',

  // Specialty
  'Auto Filter',
  'AutoFilter',
];

/** De-duplicate a name list while preserving insertion order. */
export function uniqueNames(...lists) {
  const seen = new Set();
  const out = [];
  for (const list of lists) {
    if (!Array.isArray(list)) continue;
    for (const n of list) {
      const trimmed = (n || '').trim();
      if (!trimmed || seen.has(trimmed)) continue;
      seen.add(trimmed);
      out.push(trimmed);
    }
  }
  return out;
}
