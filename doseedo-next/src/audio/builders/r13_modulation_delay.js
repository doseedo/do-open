/**
 * R13 — Modulation Delay (Logic Pro stock parity).
 *
 * Logic's "Modulation Delay" sits between chorus, flanger, and slap-delay:
 * it's a longer-range modulated delay line with feedback through tape
 * saturation and a band-limit. Pipeline:
 *
 *   1. Variable delay line per channel (0.1..80 ms range).
 *   2. LFO modulates delay time (sine / triangle / random / square shapes).
 *   3. Feedback loop with HPF + LPF (low_cut / high_cut) and optional
 *      tape saturation (asymmetric soft-clip with bias for 2nd-harmonic).
 *   4. Wet/dry mix.
 *   5. Stereo: two delay lines with offset LFO phases (`stereo_phase`).
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildModulationDelay(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: {...} } }
 *
 * Schema:
 *   {
 *     type: 'modulation_delay',
 *     params: {
 *       delay_ms:        0.1..80    // base delay
 *       rate_hz:         0.05..10   // LFO speed
 *       depth:           0..100     // % LFO sweep on delay time
 *       feedback:        -100..100  // % (negative = inverted polarity)
 *       tape_saturation: 0..1       // dry/wet of nonlinear saturator
 *       lfo_shape:       'sine' | 'triangle' | 'random' | 'square' | 0..3
 *       stereo_phase:    0..360 deg // L/R LFO phase offset
 *       low_cut:         20..2000   Hz HPF in feedback path
 *       high_cut:        1000..20000 Hz LPF in feedback path
 *       mix:             0..1       // dry/wet
 *     }
 *   }
 *
 * Param values may be literals or '@<paramId>' bindings.
 *
 * Fallback: if the worklet processor isn't registered (worklet module not
 * loaded yet, or running in a non-browser test context), the builder returns
 * a primitive DelayNode-based path with no LFO modulation but functional
 * dry/wet mix and feedback. Audio still flows; modulation character won't
 * match Logic until the worklet activates.
 *
 * @author Doseedo R13
 */

const R13_PROCESSOR = 'r13-modulation-delay-processor';

// ── Helpers ────────────────────────────────────────────────────────────────

const SHAPE_NAME_TO_INDEX = {
  sine:     0,
  triangle: 1,
  random:   2,
  square:   3,
};

function _shapeIndex(value) {
  if (typeof value === 'number') return Math.max(0, Math.min(3, Math.round(value)));
  if (typeof value === 'string') {
    const idx = SHAPE_NAME_TO_INDEX[value.toLowerCase()];
    return idx == null ? 0 : idx;
  }
  return 0;
}

function _isModulated(v) {
  return typeof v === 'string' && v.startsWith('@');
}

function _safeWorklet(ctx, name, options) {
  try {
    return new AudioWorkletNode(ctx, name, options);
  } catch (e) {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[R13] worklet ${name} unavailable, using fallback:`, e && e.message);
    }
    return null;
  }
}

// ── Builder ────────────────────────────────────────────────────────────────

export function buildModulationDelay(ctx, nodeDef, paramDefs) {
  const params = (nodeDef && nodeDef.params) || {};
  const targets = {};

  const input = ctx.createGain();
  const output = ctx.createGain();

  // Compute initial parameterData. Modulated entries fall back to defaults.
  const initial = {
    delay_ms:        (typeof params.delay_ms === 'number')        ? params.delay_ms        : 8.0,
    rate_hz:         (typeof params.rate_hz === 'number')         ? params.rate_hz         : 0.5,
    depth:           (typeof params.depth === 'number')           ? params.depth           : 30,
    feedback:        (typeof params.feedback === 'number')        ? params.feedback        : 0,
    tape_saturation: (typeof params.tape_saturation === 'number') ? params.tape_saturation : 0,
    lfo_shape:       _shapeIndex(_isModulated(params.lfo_shape) ? 'sine' : (params.lfo_shape ?? 'sine')),
    stereo_phase:    (typeof params.stereo_phase === 'number')    ? params.stereo_phase    : 90,
    low_cut:         (typeof params.low_cut === 'number')         ? params.low_cut         : 50,
    high_cut:        (typeof params.high_cut === 'number')        ? params.high_cut        : 12000,
    mix:             (typeof params.mix === 'number')             ? params.mix             : 0.5,
  };

  const worklet = _safeWorklet(ctx, R13_PROCESSOR, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    parameterData: initial,
  });

  // ── Fallback path ────────────────────────────────────────────────────────
  // Static DelayNode + feedback gain + dry/wet bus. No LFO modulation in
  // fallback (DelayNode delayTime can't be swept at sample rate without a
  // ConstantSourceNode driver, which is overkill for the fallback). Engine
  // schema stays consistent.
  let dryGainF = null, wetGainF = null, delayNodeF = null, fbGainF = null;
  if (!worklet) {
    delayNodeF = ctx.createDelay(0.2);
    delayNodeF.delayTime.value = Math.max(0.0001, initial.delay_ms / 1000);

    fbGainF = ctx.createGain();
    fbGainF.gain.value = (initial.feedback / 100) * 0.9;

    dryGainF = ctx.createGain();
    wetGainF = ctx.createGain();
    dryGainF.gain.value = Math.cos(initial.mix * Math.PI / 2);
    wetGainF.gain.value = Math.sin(initial.mix * Math.PI / 2);

    input.connect(dryGainF);
    input.connect(delayNodeF);
    delayNodeF.connect(fbGainF);
    fbGainF.connect(delayNodeF);
    delayNodeF.connect(wetGainF);
    dryGainF.connect(output);
    wetGainF.connect(output);
  } else {
    input.connect(worklet);
    worklet.connect(output);
  }

  const wpar = (name) => (worklet && worklet.parameters)
    ? (worklet.parameters.get(name) || null)
    : null;

  // ── Param wiring ─────────────────────────────────────────────────────────
  // Bind a numeric param either to the worklet AudioParam (when present) or
  // to the fallback path's analogue. `transform` lets callers (e.g. mix,
  // feedback) reshape the value before assignment.
  const bindNumeric = (key, paramName, fallbackSetter, transform) => {
    const val = params[key];
    const ap = wpar(paramName);
    if (val === undefined) return;
    if (_isModulated(val)) {
      const id = val.slice(1);
      if (ap) {
        if (!transform) {
          targets[id] = { audioParam: ap, paramDef: paramDefs[id] };
        } else {
          targets[id] = {
            paramDef: paramDefs[id],
            customSetter: (v) => { ap.value = transform(v); },
          };
        }
      } else if (fallbackSetter) {
        targets[id] = {
          paramDef: paramDefs[id],
          customSetter: (v) => fallbackSetter(transform ? transform(v) : v),
        };
      } else {
        targets[id] = { paramDef: paramDefs[id], customSetter: () => {} };
      }
    } else if (typeof val === 'number') {
      if (ap) ap.value = transform ? transform(val) : val;
      else if (fallbackSetter) fallbackSetter(transform ? transform(val) : val);
    }
  };

  bindNumeric('delay_ms', 'delay_ms',
    (v) => { if (delayNodeF) delayNodeF.delayTime.value = Math.max(0.0001, v / 1000); });

  bindNumeric('rate_hz', 'rate_hz', null);
  bindNumeric('depth',   'depth',   null);

  bindNumeric('feedback', 'feedback',
    (v) => { if (fbGainF) fbGainF.gain.value = (v / 100) * 0.9; });

  bindNumeric('tape_saturation', 'tape_saturation', null,
    (v) => Math.max(0, Math.min(1, v)));

  bindNumeric('stereo_phase', 'stereo_phase', null);
  bindNumeric('low_cut', 'low_cut', null);
  bindNumeric('high_cut', 'high_cut', null);

  bindNumeric('mix', 'mix',
    (v) => {
      if (!dryGainF || !wetGainF) return;
      const c = Math.max(0, Math.min(1, v));
      dryGainF.gain.value = Math.cos(c * Math.PI / 2);
      wetGainF.gain.value = Math.sin(c * Math.PI / 2);
    });

  // lfo_shape: accepts strings ('sine'/'triangle'/'random'/'square') OR enum
  // 0..3. Bind through a customSetter so we can normalise either form.
  {
    const val = params.lfo_shape;
    const ap = wpar('lfo_shape');
    if (val !== undefined) {
      if (_isModulated(val)) {
        const id = val.slice(1);
        targets[id] = {
          paramDef: paramDefs[id],
          customSetter: (v) => {
            const idx = _shapeIndex(v);
            if (ap) ap.value = idx;
          },
        };
      } else if (ap) {
        ap.value = _shapeIndex(val);
      }
    }
  }

  return { input, output, paramTargets: targets, workletNode: worklet };
}

// ── Default export: NODE_BUILDERS map ─────────────────────────────────────
const R13_BUILDERS = {
  modulation_delay: buildModulationDelay,
};

export default R13_BUILDERS;
