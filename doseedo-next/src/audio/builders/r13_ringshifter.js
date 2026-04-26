/**
 * R13 — Ringshifter (`ring_shift`) builder
 *
 * Logic Pro's Ringshifter combines two effects in one plugin:
 *
 *   1. Ring modulator — input × sine(2πf·t). Adds sum and difference
 *      frequencies, gives a metallic / inharmonic character.
 *   2. Frequency shifter — Hilbert-transform single-sideband modulator.
 *      Shifts ALL frequencies by a fixed amount in Hz (NOT a pitch shift —
 *      the result loses harmonic relationship). The classic Bode trick:
 *
 *        up   = real·cos(2πft) − imag·sin(2πft)
 *        down = real·cos(2πft) + imag·sin(2πft)
 *
 *      where (real, imag) is the analytic-signal pair produced by feeding
 *      input through a 90° phase difference network (Hilbert).
 *
 * The Hilbert is implemented as an IIR allpass cascade — see
 * `INTEGRATION_R13_RINGSHIFTER.md` for the cutoff tuning. The cascade is
 * cheap (8 first-order allpasses) and produces a near-90° phase difference
 * across ~80 Hz – 10 kHz. Outside that band the freq-shift output is
 * imperfect (residual sideband bleeds through) but still musical.
 *
 * Modes
 *   ring_mod        : pure ring modulation
 *   freq_shift_up   : SSB up-shift
 *   freq_shift_down : SSB down-shift
 *   both            : parallel sum of ring + freq_shift_up
 *
 * Modulation
 *   carrier frequency is modulated by an internal LFO (sine/triangle/square/
 *   random) with rate `lfo_rate` Hz and depth `lfo_depth` (% of `freq_hz`).
 *
 * Feedback
 *   `feedback` (0..1) routes the wet output back into the shift path before
 *   the next sample, giving the characteristic snowballing modulation tail.
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildRingshifter(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: {...} } }
 *
 * Following R9/R13 convention: if the worklet processor isn't registered
 * (engine hasn't called `audioWorklet.addModule(...)` yet) we fall back to a
 * primitive ring-mod-only path constructed from native nodes — a sine
 * OscillatorNode driving a GainNode whose gain is the input. The frequency
 * shifter is unavailable in fallback (Web Audio has no Hilbert primitive),
 * but `mode='ring_mod'` and `mode='both'` still produce audible ring-mod
 * content. Once the worklet loads the same builder picks up the full DSP.
 *
 * Author: Agent R13
 */

const PROC_NAME = 'r13-ringshifter-processor';

const MODE_NAME_TO_INDEX = {
  ring_mod:        0,
  freq_shift_up:   1,
  freq_shift_down: 2,
  both:            3,
};

const LFO_SHAPE_TO_INDEX = {
  sine:     0,
  triangle: 1,
  square:   2,
  random:   3,
};

function _modeIndex(value) {
  if (typeof value === 'number') return Math.max(0, Math.min(3, Math.round(value)));
  if (typeof value === 'string') {
    const idx = MODE_NAME_TO_INDEX[value.toLowerCase()];
    return idx == null ? 0 : idx;
  }
  return 0;
}

function _lfoShapeIndex(value) {
  if (typeof value === 'number') return Math.max(0, Math.min(3, Math.round(value)));
  if (typeof value === 'string') {
    const idx = LFO_SHAPE_TO_INDEX[value.toLowerCase()];
    return idx == null ? 0 : idx;
  }
  return 0;
}

function _safeWorklet(ctx, name, options = {}) {
  try {
    return new AudioWorkletNode(ctx, name, options);
  } catch (e) {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[R13/ringshifter] worklet ${name} unavailable, using fallback:`, e && e.message);
    }
    return null;
  }
}

/**
 * buildRingshifter
 *
 * Schema:
 *   {
 *     type: 'ring_shift',
 *     params: {
 *       mode:        'ring_mod' | 'freq_shift_up' | 'freq_shift_down' | 'both' | '@<id>',
 *       freq_hz:     0..5000      | '@<id>',
 *       lfo_rate:    0..10  Hz    | '@<id>',
 *       lfo_depth:   0..100 %     | '@<id>',
 *       lfo_shape:   'sine'|'triangle'|'square'|'random' | '@<id>',
 *       feedback:    0..1         | '@<id>',
 *       dry_mix:     0..1         | '@<id>',
 *       wet_mix:     0..1         | '@<id>',
 *       output_gain: 0..2         | '@<id>',
 *     }
 *   }
 */
export function buildRingshifter(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const input  = ctx.createGain();
  const output = ctx.createGain();

  const initialFreq = (typeof params.freq_hz === 'number') ? params.freq_hz : 220;
  const initialMode = _modeIndex(typeof params.mode === 'string' && params.mode.startsWith('@')
    ? 'ring_mod'
    : (params.mode ?? 'ring_mod'));
  const initialLfoShape = _lfoShapeIndex(typeof params.lfo_shape === 'string' && params.lfo_shape.startsWith('@')
    ? 'sine'
    : (params.lfo_shape ?? 'sine'));

  const worklet = _safeWorklet(ctx, PROC_NAME, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    parameterData: {
      mode:        initialMode,
      freq_hz:     initialFreq,
      lfo_shape:   initialLfoShape,
    },
  });

  // ── Fallback path (worklet missing) ──────────────────────────────────────
  // Build a primitive ring-modulator: input × sine. Output = dry + wet*ring.
  // Frequency-shifter is unavailable; for `mode=freq_shift_up/down` we
  // degrade gracefully to ring-mod so audio still flows.
  let osc = null, ringG = null, dryG = null, wetG = null, outG = null;
  let fbState = null;
  if (!worklet) {
    osc = ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.value = initialFreq;
    ringG = ctx.createGain();
    ringG.gain.value = 0; // multiplied: start from input × osc
    dryG = ctx.createGain();
    wetG = ctx.createGain();
    outG = ctx.createGain();

    const initDry = (typeof params.dry_mix === 'number') ? params.dry_mix : 0.5;
    const initWet = (typeof params.wet_mix === 'number') ? params.wet_mix : 0.5;
    const initGain = (typeof params.output_gain === 'number') ? params.output_gain : 1.0;
    dryG.gain.value = initDry;
    wetG.gain.value = initWet;
    outG.gain.value = initGain;

    // Multiplier trick: connect oscillator into ringG.gain (an AudioParam),
    // then route input through ringG. Resulting signal at ringG output
    // is input(t) × osc(t).
    osc.connect(ringG.gain);

    input.connect(dryG);
    input.connect(ringG);
    ringG.connect(wetG);
    dryG.connect(outG);
    wetG.connect(outG);
    outG.connect(output);

    try { osc.start(); } catch (e) { /* already started */ }
    fbState = { feedback: 0 };
  } else {
    // Worklet path: input → worklet → output. The worklet handles dry/wet
    // mix and output gain internally so they're addressable via AudioParams.
    input.connect(worklet);
    worklet.connect(output);
  }

  const wpar = (name) => (worklet && worklet.parameters)
    ? (worklet.parameters.get(name) || null)
    : null;

  // ── Param wiring ─────────────────────────────────────────────────────────
  for (const [key, val] of Object.entries(params)) {
    const isModulated = typeof val === 'string' && val.startsWith('@');
    const paramId = isModulated ? val.slice(1) : null;

    switch (key) {
      case 'mode': {
        const ap = wpar('mode');
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              let idx;
              if (typeof v === 'string') idx = _modeIndex(v);
              else if (v >= 0 && v <= 1) idx = Math.min(3, Math.floor(v * 4));
              else idx = _modeIndex(v);
              if (ap) ap.value = idx;
              // Fallback has no mode switch — ring-mod always.
            },
          };
        } else if (val !== undefined) {
          if (ap) ap.value = _modeIndex(val);
        }
        break;
      }
      case 'freq_hz': {
        const ap = wpar('freq_hz');
        if (isModulated) {
          if (ap) {
            targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          } else {
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => {
                if (osc) osc.frequency.setTargetAtTime(Math.max(0, Math.min(5000, v)), ctx.currentTime, 0.005);
              },
            };
          }
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          else if (osc) osc.frequency.value = Math.max(0, Math.min(5000, val));
        }
        break;
      }
      case 'lfo_rate': {
        const ap = wpar('lfo_rate');
        if (isModulated && ap) {
          targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
        } else if (isModulated) {
          // Fallback: no LFO available. Accept the binding as a no-op.
          targets[paramId] = { paramDef: paramDefs[paramId], customSetter: () => {} };
        } else if (ap && typeof val === 'number') {
          ap.value = val;
        }
        break;
      }
      case 'lfo_depth': {
        const ap = wpar('lfo_depth');
        if (isModulated && ap) {
          targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
        } else if (isModulated) {
          targets[paramId] = { paramDef: paramDefs[paramId], customSetter: () => {} };
        } else if (ap && typeof val === 'number') {
          ap.value = val;
        }
        break;
      }
      case 'lfo_shape': {
        const ap = wpar('lfo_shape');
        if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => {
              let idx;
              if (typeof v === 'string') idx = _lfoShapeIndex(v);
              else if (v >= 0 && v <= 1) idx = Math.min(3, Math.floor(v * 4));
              else idx = _lfoShapeIndex(v);
              if (ap) ap.value = idx;
            },
          };
        } else if (val !== undefined && ap) {
          ap.value = _lfoShapeIndex(val);
        }
        break;
      }
      case 'feedback': {
        const ap = wpar('feedback');
        if (isModulated && ap) {
          targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
        } else if (isModulated) {
          targets[paramId] = {
            paramDef: paramDefs[paramId],
            customSetter: (v) => { if (fbState) fbState.feedback = Math.max(0, Math.min(0.99, v)); },
          };
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          else if (fbState) fbState.feedback = Math.max(0, Math.min(0.99, val));
        }
        break;
      }
      case 'dry_mix': {
        const ap = wpar('dry_mix');
        if (isModulated) {
          if (ap) {
            targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          } else {
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => { if (dryG) dryG.gain.value = Math.max(0, Math.min(1, v)); },
            };
          }
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          else if (dryG) dryG.gain.value = Math.max(0, Math.min(1, val));
        }
        break;
      }
      case 'wet_mix': {
        const ap = wpar('wet_mix');
        if (isModulated) {
          if (ap) {
            targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          } else {
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => { if (wetG) wetG.gain.value = Math.max(0, Math.min(1, v)); },
            };
          }
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          else if (wetG) wetG.gain.value = Math.max(0, Math.min(1, val));
        }
        break;
      }
      case 'output_gain': {
        const ap = wpar('output_gain');
        if (isModulated) {
          if (ap) {
            targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          } else {
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => { if (outG) outG.gain.value = Math.max(0, Math.min(4, v)); },
            };
          }
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          else if (outG) outG.gain.value = Math.max(0, Math.min(4, val));
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
const R13_RINGSHIFTER_BUILDERS = {
  ring_shift: buildRingshifter,
};

export default R13_RINGSHIFTER_BUILDERS;
