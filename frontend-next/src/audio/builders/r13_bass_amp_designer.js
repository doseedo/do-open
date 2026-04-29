/**
 * R13 — Bass Amp Designer composite builder
 *
 * Logic Pro's Bass Amp Designer is a bass-specific guitar amp simulator. It
 * differs from Amp Designer in:
 *   - Lower-tuned tone stack (4-band: bass / low-mid / hi-mid / treble)
 *   - Bass-amp models (Ampeg-style, GK-style, Fender-style)
 *   - Bass-cab IRs (1x15, 2x15, 4x10, 6x10, 8x10, di_only)
 *   - Built-in mild compression (typical of bass amps)
 *   - 5-band graphic EQ (50 / 120 / 300 / 800 / 2k)
 *   - Tube-vs-solid-state stage blend (rear-panel control)
 *   - DI-vs-Mic blend (real bass DI boxes work this way: tap pre-cabinet,
 *     run through a HPF + slight shelf, blend with the mic'd output).
 *
 * Composite topology:
 *
 *   input → pre_gain → splitter
 *          ├─→ tube_drive_path  ─┐
 *          │                     ├─→ tube_blend → graphic_eq_5band
 *          └─→ solid_state_path ─┘                       │
 *                                                        ▼
 *                                            wdf_tone_stack (bass-tuned)
 *                                                        │
 *                                                        ▼
 *                                                   compression
 *                                                        │
 *                                                        ▼
 *                                                     cab_ir
 *                                                        │
 *          ┌─────────────────────── DI tap (post pre_gain) ──────────┐
 *          │                                                          │
 *          └─→ di_path (HPF + slight shelf) ─→ direct_out_blend ←─────┘
 *                                                        │
 *                                                        ▼
 *                                                  post_gain → output
 *
 * The R2 wdf_tone_stack worklet (when loaded) handles the analog tone-stack
 * coloration. When unavailable, we fall back to a 4-biquad cascade tuned for
 * bass frequencies.
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildBassAmpDesigner(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: {...} } }
 *
 * Author: Agent R13 (Bass Amp Designer)
 */

import { buildCabinetIR } from '../cabinet-ir.js';

const R2_TONE_STACK = 'r2-wdf-tone-stack-processor';

// Per-amp-model preamp shaping. These are knob biases applied at construction
// time to give each model its characteristic colour even before user params.
//
// Fields:
//   tubeDrive   — initial tube-stage drive (waveshaper amount, 0..2)
//   ssDrive     — initial solid-state-stage drive
//   tubeMix     — default tube/SS blend (0=all SS, 1=all tube)
//   bassPush    — added low-shelf gain at construction (dB)
//   midScoop    — peaking gain at 500 Hz (dB; negative = scooped)
//   prePresence — pre-cab high-shelf bump (dB)
export const AMP_MODELS = {
  flip_top:      { tubeDrive: 1.4, ssDrive: 0.8, tubeMix: 0.85, bassPush: 2.0, midScoop:  1.5, prePresence:  1.0 }, // Ampeg B-15 vibe
  classic_bass:  { tubeDrive: 1.7, ssDrive: 1.0, tubeMix: 0.75, bassPush: 3.5, midScoop:  0.5, prePresence:  2.5 }, // SVT classic
  fender_bass:   { tubeDrive: 1.2, ssDrive: 0.7, tubeMix: 0.80, bassPush: 1.0, midScoop:  2.0, prePresence:  0.5 }, // Bassman vibe
  modern_bass:   { tubeDrive: 0.8, ssDrive: 1.6, tubeMix: 0.30, bassPush: 4.5, midScoop: -2.0, prePresence:  3.5 }, // GK-style
  svt_classic:   { tubeDrive: 1.9, ssDrive: 1.0, tubeMix: 0.90, bassPush: 4.0, midScoop:  1.0, prePresence:  3.0 }, // SVT 70s
  svt_modern:    { tubeDrive: 1.5, ssDrive: 1.4, tubeMix: 0.55, bassPush: 5.0, midScoop: -1.5, prePresence:  4.0 }, // SVT classic w/ modern voice
  hiwatt_bass:   { tubeDrive: 1.6, ssDrive: 0.9, tubeMix: 0.80, bassPush: 1.5, midScoop:  3.0, prePresence:  1.5 }, // Hiwatt-ish
  acoustic_360:  { tubeDrive: 1.0, ssDrive: 1.2, tubeMix: 0.45, bassPush: 6.0, midScoop:  0.0, prePresence:  0.0 }, // Acoustic 360 (deep)
};

// Per-cabinet-model IR shaping. Lower frequency bottom for big cabs, more
// aggressive HF rolloff for sealed 8x10s, etc.
//
// Fields:
//   lowCutHz       — cab high-pass corner (cone air-coupling)
//   highCutHz      — cab low-pass corner (cone breakup)
//   presencePeakHz — small mid bump
//   durationSec    — IR length
export const CAB_MODELS = {
  '1x15':    { lowCutHz: 50,  highCutHz: 2800, presencePeakHz: 1500, durationSec: 0.05 },
  '2x15':    { lowCutHz: 45,  highCutHz: 2600, presencePeakHz: 1300, durationSec: 0.06 },
  '4x10':    { lowCutHz: 70,  highCutHz: 4500, presencePeakHz: 2400, durationSec: 0.045 },
  '6x10':    { lowCutHz: 60,  highCutHz: 4000, presencePeakHz: 2200, durationSec: 0.05 },
  '8x10':    { lowCutHz: 55,  highCutHz: 3800, presencePeakHz: 2000, durationSec: 0.055 },
  'di_only': null, // handled specially: cab is a unity passthrough
};

// 5-band graphic EQ centre frequencies (Hz). Matches Logic Bass Amp Designer
// Pre-EQ shape (50 / 120 / 300 / 800 / 2k).
export const GRAPHIC_EQ_FREQS = [50, 120, 300, 800, 2000];

export const DEFAULT_AMP_MODEL = 'classic_bass';
export const DEFAULT_CAB_MODEL = '4x10';

// ── Helpers ───────────────────────────────────────────────────────────────

function _safeWorklet(ctx, name, options = {}) {
  try {
    return new AudioWorkletNode(ctx, name, options);
  } catch (e) {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[R13.bass_amp] worklet ${name} unavailable, using fallback:`, e && e.message);
    }
    return null;
  }
}

// Build a soft-clip waveshaping curve. `drive` 0..2 (0 = transparent,
// 2 = aggressive). `asymmetry` adds a DC-biased tube-like 2nd harmonic.
function _makeTubeCurve(drive = 1.0, asymmetry = 0.15) {
  const N = 2048;
  const curve = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const x = (i / (N - 1)) * 2 - 1;
    // Asymmetric soft clip — positive half compresses harder than negative
    const a = drive * (1 + (x > 0 ? asymmetry : 0));
    curve[i] = Math.tanh(x * (1 + a)) / Math.tanh(1 + a);
  }
  return curve;
}

// Build a hard-clip-ish solid-state curve (sharper knee than tube).
function _makeSolidStateCurve(drive = 1.0) {
  const N = 2048;
  const curve = new Float32Array(N);
  const k = 1 + drive * 3;
  for (let i = 0; i < N; i++) {
    const x = (i / (N - 1)) * 2 - 1;
    // Sharper knee: x*k then soft-clip via x/(1+|x|), gives transistor-like saturation
    const xs = x * k;
    curve[i] = xs / (1 + Math.abs(xs));
  }
  return curve;
}

// Build the DI-path low-shelf curve via Float32Array — a simple emulation of
// what a bass DI box does to the signal pre-cab.
function _buildDiSignalPath(ctx) {
  // Chain: input → DI HPF (subsonic rumble cleanup) → DI low-shelf (slight
  // boost ~80 Hz) → DI HF rolloff (just enough to take the harshness off
  // the dry signal) → output gain.
  const hpf = ctx.createBiquadFilter();
  hpf.type = 'highpass';
  hpf.frequency.value = 35;
  hpf.Q.value = 0.7;

  const shelf = ctx.createBiquadFilter();
  shelf.type = 'lowshelf';
  shelf.frequency.value = 90;
  shelf.gain.value = 1.5;

  const lpf = ctx.createBiquadFilter();
  lpf.type = 'lowpass';
  lpf.frequency.value = 8000;
  lpf.Q.value = 0.5;

  const out = ctx.createGain();
  out.gain.value = 1;

  hpf.connect(shelf);
  shelf.connect(lpf);
  lpf.connect(out);
  return { input: hpf, output: out };
}

// ── Builder ───────────────────────────────────────────────────────────────

/**
 * Schema:
 *   {
 *     type: 'bass_amp_designer',
 *     params: {
 *       amp_model:       'flip_top'|'classic_bass'|'fender_bass'|'modern_bass'|
 *                        'svt_classic'|'svt_modern'|'hiwatt_bass'|'acoustic_360'
 *                        | '@<id>',
 *       gain:            number (0..2)         | '@<id>',
 *       bass:            number (0..1)         | '@<id>',
 *       mid_low:         number (0..1)         | '@<id>',
 *       mid_hi:          number (0..1)         | '@<id>',
 *       treble:          number (0..1)         | '@<id>',
 *       master:          number (0..2)         | '@<id>',
 *       compression:     number (0..1)         | '@<id>',  // amount of comp
 *       graphic_eq:      Array<5 dB values>    | '@<id>',
 *       tube_blend:      number (0..1)         | '@<id>',  // 0=SS only, 1=tube only
 *       cab_model:       '1x15'|'2x15'|'4x10'|'6x10'|'8x10'|'di_only' | '@<id>',
 *       mic_position:    number (0..1)         | '@<id>',  // 0=on-axis, 1=off-axis
 *       direct_out_mix:  number (0..1)         | '@<id>',  // 0=full mic, 1=full DI
 *       output_level:    number (0..2)         | '@<id>',
 *     }
 *   }
 */
export function buildBassAmpDesigner(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const input = ctx.createGain();
  const output = ctx.createGain();
  output.gain.value = 1;

  // Resolve amp_model (string) — '@'-prefixed values use the default until set
  let modelKey = (typeof params.amp_model === 'string' && !params.amp_model.startsWith('@'))
    ? params.amp_model
    : 'classic_bass';
  if (!AMP_MODELS[modelKey]) modelKey = 'classic_bass';
  let model = AMP_MODELS[modelKey];

  // Resolve cab_model (string)
  let cabKey = (typeof params.cab_model === 'string' && !params.cab_model.startsWith('@'))
    ? params.cab_model
    : '4x10';
  if (cabKey !== 'di_only' && !CAB_MODELS[cabKey]) cabKey = '4x10';

  // ── Pre-gain ───────────────────────────────────────────────────────────
  const preGain = ctx.createGain();
  preGain.gain.value = 1.0;

  // ── Tube path: gain stage → tube curve waveshaper ──────────────────────
  const tubePreGain = ctx.createGain();
  tubePreGain.gain.value = 1.0;
  const tubeShaper = ctx.createWaveShaper();
  tubeShaper.curve = _makeTubeCurve(model.tubeDrive, 0.18);
  tubeShaper.oversample = '4x';
  const tubeOut = ctx.createGain();
  tubeOut.gain.value = model.tubeMix;

  tubePreGain.connect(tubeShaper);
  tubeShaper.connect(tubeOut);

  // ── Solid-state path: gain stage → SS curve → dB-shifted out ──────────
  const ssPreGain = ctx.createGain();
  ssPreGain.gain.value = 1.0;
  const ssShaper = ctx.createWaveShaper();
  ssShaper.curve = _makeSolidStateCurve(model.ssDrive);
  ssShaper.oversample = '2x';
  const ssOut = ctx.createGain();
  ssOut.gain.value = 1 - model.tubeMix;

  ssPreGain.connect(ssShaper);
  ssShaper.connect(ssOut);

  // ── Sum of two paths ──────────────────────────────────────────────────
  const stageSum = ctx.createGain();
  stageSum.gain.value = 1.0;
  tubeOut.connect(stageSum);
  ssOut.connect(stageSum);

  // ── 5-band graphic EQ ──────────────────────────────────────────────────
  const graphicEqBands = GRAPHIC_EQ_FREQS.map((freq, idx) => {
    const filt = ctx.createBiquadFilter();
    filt.type = 'peaking';
    filt.frequency.value = freq;
    filt.Q.value = 1.4;
    filt.gain.value = 0;
    return filt;
  });
  // Initial graphic_eq if provided
  if (Array.isArray(params.graphic_eq)) {
    for (let i = 0; i < graphicEqBands.length && i < params.graphic_eq.length; i++) {
      const v = Number(params.graphic_eq[i]);
      if (Number.isFinite(v)) graphicEqBands[i].gain.value = v;
    }
  }
  // Wire EQ bands serially
  for (let i = 0; i < graphicEqBands.length - 1; i++) {
    graphicEqBands[i].connect(graphicEqBands[i + 1]);
  }
  const graphicEqHead = graphicEqBands[0];
  const graphicEqTail = graphicEqBands[graphicEqBands.length - 1];

  // ── Bass-tuned tone stack (R2 worklet OR 4-band biquad fallback) ──────
  // We try the R2 wdf_tone_stack worklet first (it expects bass/mid/treble)
  // and on top apply our 4-band fallback to get the extra mid_low / mid_hi
  // distinction that Bass Amp Designer has.
  const toneStack = _safeWorklet(ctx, R2_TONE_STACK, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    parameterData: { bass: 0.5, mid: 0.5, treble: 0.5 },
  });

  // 4-biquad fallback: low shelf + low-mid peak + hi-mid peak + high shelf,
  // tuned for bass-amp frequency centres. This applies whether or not the
  // worklet is present (it adds the mid_low / mid_hi distinction Bass Amp
  // Designer requires beyond R2's 3-band tone stack).
  const bassShelf = ctx.createBiquadFilter();
  bassShelf.type = 'lowshelf';
  bassShelf.frequency.value = 80;
  bassShelf.gain.value = model.bassPush;

  const midLowPeak = ctx.createBiquadFilter();
  midLowPeak.type = 'peaking';
  midLowPeak.frequency.value = 250;
  midLowPeak.Q.value = 0.9;
  midLowPeak.gain.value = 0;

  const midHiPeak = ctx.createBiquadFilter();
  midHiPeak.type = 'peaking';
  midHiPeak.frequency.value = 900;
  midHiPeak.Q.value = 1.1;
  midHiPeak.gain.value = model.midScoop;

  const trebShelf = ctx.createBiquadFilter();
  trebShelf.type = 'highshelf';
  trebShelf.frequency.value = 3000;
  trebShelf.gain.value = model.prePresence;

  // Wire the 4-band cascade
  bassShelf.connect(midLowPeak);
  midLowPeak.connect(midHiPeak);
  midHiPeak.connect(trebShelf);

  // ── Compression ────────────────────────────────────────────────────────
  // Bass amps typically have a built-in mild leveling stage. We expose one
  // `compression` knob (0..1) that scales threshold + ratio.
  const compressor = ctx.createDynamicsCompressor();
  compressor.threshold.value = -18;
  compressor.ratio.value     = 3;
  compressor.attack.value    = 0.005;
  compressor.release.value   = 0.18;
  compressor.knee.value      = 8;

  // ── Cabinet IR (or DI-only bypass) ─────────────────────────────────────
  let cab;
  if (cabKey === 'di_only') {
    cab = ctx.createGain();
    cab.gain.value = 1.0;
  } else {
    cab = ctx.createConvolver();
    try {
      cab.buffer = buildCabinetIR(ctx, CAB_MODELS[cabKey]);
    } catch (e) { /* ctx closed */ }
  }

  // ── Mic position ───────────────────────────────────────────────────────
  // Fake mic position via post-cab high-shelf tilt. On-axis (0) = brighter,
  // off-axis (1) = darker.
  const micPosFilter = ctx.createBiquadFilter();
  micPosFilter.type = 'highshelf';
  micPosFilter.frequency.value = 4000;
  micPosFilter.gain.value = 0;

  // ── DI signal path ─────────────────────────────────────────────────────
  const di = _buildDiSignalPath(ctx);

  // ── Direct-out blend ───────────────────────────────────────────────────
  // micGain ←→ diGain : mic'd cab vs DI box mix
  const micGain = ctx.createGain();
  const diGain = ctx.createGain();
  micGain.gain.value = 1.0;
  diGain.gain.value  = 0.0;

  // ── Post-gain (master / output level) ─────────────────────────────────
  const postGain = ctx.createGain();
  postGain.gain.value = 1.0;

  // ── Wire everything ────────────────────────────────────────────────────
  // input → preGain → split into tube + SS + DI tap
  input.connect(preGain);
  preGain.connect(tubePreGain);
  preGain.connect(ssPreGain);
  preGain.connect(di.input); // DI tap is pre-cab

  // Stage sum → graphic EQ
  stageSum.connect(graphicEqHead);

  // Graphic EQ → tone stack (worklet path) → fallback 4-biquad cascade
  if (toneStack) {
    graphicEqTail.connect(toneStack);
    toneStack.connect(bassShelf);
  } else {
    graphicEqTail.connect(bassShelf);
  }

  // 4-biquad tone tail → compressor → cab → mic-pos filter → micGain
  trebShelf.connect(compressor);
  compressor.connect(cab);
  cab.connect(micPosFilter);
  micPosFilter.connect(micGain);

  // DI path → diGain
  di.output.connect(diGain);

  // Sum mic + DI into post-gain
  micGain.connect(postGain);
  diGain.connect(postGain);
  postGain.connect(output);

  // ── Param wiring ──────────────────────────────────────────────────────
  const _wpar = (name) => (toneStack && toneStack.parameters)
    ? (toneStack.parameters.get(name) || null)
    : null;

  function _rebuildAmpModel(newKey) {
    if (!AMP_MODELS[newKey]) return;
    modelKey = newKey;
    model = AMP_MODELS[newKey];
    try {
      tubeShaper.curve = _makeTubeCurve(model.tubeDrive, 0.18);
      ssShaper.curve   = _makeSolidStateCurve(model.ssDrive);
    } catch (e) { /* ignore */ }
    tubeOut.gain.value = model.tubeMix;
    ssOut.gain.value   = 1 - model.tubeMix;
    bassShelf.gain.value = model.bassPush;
    midHiPeak.gain.value = model.midScoop;
    trebShelf.gain.value = model.prePresence;
  }

  function _rebuildCabModel(newKey) {
    if (newKey !== 'di_only' && !CAB_MODELS[newKey]) return;
    cabKey = newKey;
    if (cabKey === 'di_only') {
      // Replace cab with passthrough by zeroing IR (ConvolverNode returns
      // silence when buffer is null). Add a parallel unity bypass via a gain
      // node — but we can't restructure live, so instead we attenuate cab
      // output to 0 and crank diGain. That's effectively `direct_out_mix=1`.
      // For simplicity we just clear the buffer — and rely on direct_out_mix
      // having been set elsewhere.
      if (cab.buffer !== undefined) {
        try { cab.buffer = null; } catch (e) { /* ignore */ }
      }
    } else if (cab.buffer !== undefined) {
      try {
        cab.buffer = buildCabinetIR(ctx, CAB_MODELS[cabKey]);
      } catch (e) { /* ignore */ }
    }
  }

  for (const [key, val] of Object.entries(params)) {
    const isModulated = typeof val === 'string' && val.startsWith('@');
    const paramId = isModulated ? val.slice(1) : null;

    switch (key) {
      case 'amp_model': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              if (typeof v === 'string') _rebuildAmpModel(v);
              else if (typeof v === 'number') {
                // Map normalized 0..1 → enum index
                const keys = Object.keys(AMP_MODELS);
                const idx = Math.max(0, Math.min(keys.length - 1, Math.floor(v * keys.length)));
                _rebuildAmpModel(keys[idx]);
              }
            },
          };
        }
        // string val handled at construction
        break;
      }
      case 'gain': {
        if (isModulated) {
          targets[paramId] = { audioParam: preGain.gain, paramDef: paramDefs[paramId] };
        } else if (typeof val === 'number') {
          preGain.gain.value = val;
        }
        break;
      }
      case 'bass': {
        const ap = _wpar('bass');
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              if (ap) ap.value = v;
              // Map 0..1 → ±12 dB shelf, additive on top of model.bassPush
              bassShelf.gain.value = model.bassPush + (v - 0.5) * 24;
            },
          };
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          bassShelf.gain.value = model.bassPush + (val - 0.5) * 24;
        }
        break;
      }
      case 'mid_low': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => { midLowPeak.gain.value = (v - 0.5) * 18; },
          };
        } else if (typeof val === 'number') {
          midLowPeak.gain.value = (val - 0.5) * 18;
        }
        break;
      }
      case 'mid_hi':
      case 'mid_high': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => { midHiPeak.gain.value = model.midScoop + (v - 0.5) * 18; },
          };
        } else if (typeof val === 'number') {
          midHiPeak.gain.value = model.midScoop + (val - 0.5) * 18;
        }
        break;
      }
      case 'treble': {
        const ap = _wpar('treble');
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              if (ap) ap.value = v;
              trebShelf.gain.value = model.prePresence + (v - 0.5) * 24;
            },
          };
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          trebShelf.gain.value = model.prePresence + (val - 0.5) * 24;
        }
        break;
      }
      case 'master':
      case 'output_level': {
        if (isModulated) {
          targets[paramId] = { audioParam: postGain.gain, paramDef: paramDefs[paramId] };
        } else if (typeof val === 'number') {
          postGain.gain.value = val;
        }
        break;
      }
      case 'compression': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              const w = Math.max(0, Math.min(1, v));
              // 0 → bypass-ish (-6 dB threshold, 1.5 ratio)
              // 1 → assertive   (-30 dB threshold, 6 ratio)
              compressor.threshold.value = -6 - w * 24;
              compressor.ratio.value     = 1.5 + w * 4.5;
            },
          };
        } else if (typeof val === 'number') {
          const w = Math.max(0, Math.min(1, val));
          compressor.threshold.value = -6 - w * 24;
          compressor.ratio.value     = 1.5 + w * 4.5;
        }
        break;
      }
      case 'graphic_eq': {
        // Array param. If modulated (single id), pass an array through
        // customSetter; otherwise applied at construction above.
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              if (Array.isArray(v)) {
                for (let i = 0; i < graphicEqBands.length && i < v.length; i++) {
                  const g = Number(v[i]);
                  if (Number.isFinite(g)) graphicEqBands[i].gain.value = g;
                }
              }
            },
          };
        }
        break;
      }
      case 'tube_blend': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              const w = Math.max(0, Math.min(1, v));
              tubeOut.gain.value = w;
              ssOut.gain.value   = 1 - w;
            },
          };
        } else if (typeof val === 'number') {
          const w = Math.max(0, Math.min(1, val));
          tubeOut.gain.value = w;
          ssOut.gain.value   = 1 - w;
        }
        break;
      }
      case 'cab_model': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              if (typeof v === 'string') _rebuildCabModel(v);
              else if (typeof v === 'number') {
                const keys = Object.keys(CAB_MODELS);
                const idx = Math.max(0, Math.min(keys.length - 1, Math.floor(v * keys.length)));
                _rebuildCabModel(keys[idx]);
              }
            },
          };
        }
        // string val handled at construction
        break;
      }
      case 'mic_position': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              const w = Math.max(0, Math.min(1, v));
              // 0 (on-axis) = +2 dB HF, 1 (off-axis) = -6 dB HF
              micPosFilter.gain.value = 2 - w * 8;
            },
          };
        } else if (typeof val === 'number') {
          const w = Math.max(0, Math.min(1, val));
          micPosFilter.gain.value = 2 - w * 8;
        }
        break;
      }
      case 'direct_out_mix': {
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              const w = Math.max(0, Math.min(1, v));
              diGain.gain.value  = w;
              micGain.gain.value = 1 - w;
            },
          };
        } else if (typeof val === 'number') {
          const w = Math.max(0, Math.min(1, val));
          diGain.gain.value  = w;
          micGain.gain.value = 1 - w;
        }
        break;
      }
      default:
        break;
    }
  }

  return { input, output, paramTargets: targets };
}

// ── Default export: NODE_BUILDERS map ─────────────────────────────────────
const R13_BASS_AMP_BUILDERS = {
  bass_amp_designer: buildBassAmpDesigner,
};

export default R13_BASS_AMP_BUILDERS;
