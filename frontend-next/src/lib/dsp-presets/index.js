// Registry of all studio plugins expressed as DSP-lang preset manifests.
//
// Loading a preset = handing its `dspConfig` to a fresh
// `WebAudioDSPEngine` instance. The studio's plugin-FX layer can use
// this registry instead of the old `BasePlugin` factory once we're
// ready to flip the FX-bypass switch.

import compressor       from './compressor.js';
import limiter          from './limiter.js';
import gate             from './gate.js';
import expander         from './expander.js';
import gain             from './gain.js';
import pan              from './pan.js';
import reverb           from './reverb.js';
import convolutionReverb from './convolution_reverb.js';
import hybridReverb     from './hybrid_reverb.js';
import simpleDelay      from './simple_delay.js';
import pingPongDelay    from './ping_pong_delay.js';
import echo             from './echo.js';
import chorus           from './chorus.js';
import flanger          from './flanger.js';
import phaser           from './phaser.js';
import tremolo          from './tremolo.js';
import ringModulator    from './ring_modulator.js';
import distortion       from './distortion.js';
import overdrive        from './overdrive.js';
import saturator        from './saturator.js';
import redux            from './redux.js';
import filter           from './filter.js';
import parametricEQ     from './parametric_eq.js';
import graphicEQ        from './graphic_eq.js';
import pultecEQ         from './pultec_eq.js';
import tapeEmulation    from './tape_emulation.js';
import tubePreamp       from './tube_preamp.js';
import spectralFilter   from './spectral_filter.js';
import spectralFreeze   from './spectral_freeze.js';
import pitchShift       from './pitch_shift.js';
import peakMeter        from './peak_meter.js';
import rmsMeter         from './rms_meter.js';
import polarity         from './polarity.js';
import stereoWidth      from './stereo_width.js';

export const DSP_PRESETS = {
  // Dynamics
  compressor, limiter, gate, expander,
  // Utility
  gain, pan, polarity, stereoWidth,
  // Reverb
  reverb, convolutionReverb, hybridReverb,
  // Delay
  simpleDelay, pingPongDelay, echo,
  // Modulation
  chorus, flanger, phaser, tremolo, ringModulator,
  // Distortion
  distortion, overdrive, saturator, redux,
  // EQ
  filter, parametricEQ, graphicEQ,
  // Spectral
  spectralFilter, spectralFreeze, pitchShift,
  // Analysis
  peakMeter, rmsMeter,
  // Vintage
  pultecEQ, tapeEmulation, tubePreamp,
};

export const PRESETS_BY_CATEGORY = {};
for (const [key, preset] of Object.entries(DSP_PRESETS)) {
  const cat = preset.category || 'misc';
  if (!PRESETS_BY_CATEGORY[cat]) PRESETS_BY_CATEGORY[cat] = [];
  PRESETS_BY_CATEGORY[cat].push({ key, ...preset });
}

/** Look up a preset by its registry key (e.g. 'compressor'). */
export function getPreset(key) {
  return DSP_PRESETS[key] || null;
}

/** All preset names for picker UIs. */
export function listPresets() {
  return Object.entries(DSP_PRESETS).map(([key, p]) => ({
    key,
    name: p.name,
    category: p.category,
    description: p.description,
  }));
}
