/**
 * R13 — Pitch Correction (Logic Pro stock parity)
 *
 * Registers `pitch_correct` as a NEW node type backed by a single AudioWorklet
 * (`r13-pitch-correct-processor`) that fuses YIN pitch detection, scale-snap
 * quantisation, and PSOLA pitch shift.
 *
 * Builder contract:
 *   buildPitchCorrect(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: {...} } }
 *
 * Schema (matches the Logic Pitch Correction surface):
 *   {
 *     type: 'pitch_correct',
 *     params: {
 *       key:                 0..11           // 0=C, 1=C#, ..., 11=B
 *       scale:               'major' | 'minor' | 'chromatic' | 'custom'
 *       scale_mask:          uint12          // used when scale === 'custom'
 *       response_ms:         0..500          // smoothing time-constant
 *       correction_amount:   0..1            // 1 = full snap, 0 = bypass
 *       formant_preserve:    0|1             // TODO — currently no-op
 *       mix:                 0..1            // dry/wet
 *     }
 *   }
 *
 * Param values may be literals or '@<paramId>' bindings (live-modulated).
 *
 * Fallback: if the worklet processor isn't registered yet (or worklets are
 * unavailable, e.g. SSR / jsdom), the builder returns a passthrough Gain
 * node with no-op param targets. Pitch correction is fundamentally not
 * feasible without sample-accurate processing, so the fallback is
 * intentionally a passthrough — see INTEGRATION_R13_PITCH_CORRECTION.md.
 *
 * @author Doseedo R13
 */

const R13_PROCESSOR = 'r13-pitch-correct-processor';

// Keep in sync with r13-pitch-correct-processor.js SCALE_MASKS.
export const R13_SCALE_MASKS = {
  major:     0b101010110101, // C D E F G A B
  minor:     0b010110101101, // C D Eb F G Ab Bb (natural minor)
  chromatic: 0b111111111111,
};

function _scaleNameToMask(scale, customMask) {
  if (typeof scale === 'string') {
    const k = scale.toLowerCase();
    if (k === 'custom') {
      const m = (typeof customMask === 'number') ? Math.round(customMask) : 0;
      return Math.max(0, Math.min(0xFFF, m));
    }
    if (R13_SCALE_MASKS[k] != null) return R13_SCALE_MASKS[k];
  }
  return R13_SCALE_MASKS.chromatic;
}

function _safeWorklet(ctx, name, options) {
  try {
    return new AudioWorkletNode(ctx, name, options);
  } catch (e) {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[R13] worklet ${name} unavailable, falling back to passthrough:`,
                    e && e.message);
    }
    return null;
  }
}

function _isModulated(v) {
  return typeof v === 'string' && v.startsWith('@');
}

export function buildPitchCorrect(ctx, nodeDef, paramDefs) {
  const params = (nodeDef && nodeDef.params) || {};
  const targets = {};

  const input  = ctx.createGain();
  const output = ctx.createGain();

  // Resolve initial constants for parameterData. Modulated entries → defaults.
  const initialKey =
    (typeof params.key === 'number') ? Math.max(0, Math.min(11, Math.round(params.key))) : 0;

  let initialMask;
  if (_isModulated(params.scale_mask)) {
    initialMask = R13_SCALE_MASKS.chromatic;
  } else if (typeof params.scale_mask === 'number') {
    initialMask = Math.max(0, Math.min(0xFFF, Math.round(params.scale_mask)));
  } else {
    initialMask = _scaleNameToMask(params.scale, params.scale_mask);
  }

  const initialResp = (typeof params.response_ms === 'number') ? params.response_ms : 50;
  const initialCorr = (typeof params.correction_amount === 'number') ? params.correction_amount : 1;
  const initialForm = params.formant_preserve ? 1 : 0;
  const initialMix  = (typeof params.mix === 'number') ? params.mix : 1;

  const worklet = _safeWorklet(ctx, R13_PROCESSOR, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    parameterData: {
      key:                initialKey,
      scale_mask:         initialMask,
      response_ms:        initialResp,
      correction_amount:  initialCorr,
      formant_preserve:   initialForm,
      mix:                initialMix,
    },
    processorOptions: {
      analysisWindow: 2048,
      yinThreshold: 0.15,
      minF0Hz: 70,
      maxF0Hz: 1100,
    },
  });

  if (!worklet) {
    // Passthrough fallback. Bind any '@' params as no-ops so the engine
    // still resolves the modulation graph without errors.
    input.connect(output);
    for (const [, val] of Object.entries(params)) {
      if (_isModulated(val)) {
        const id = val.slice(1);
        targets[id] = { paramDef: paramDefs[id], customSetter: () => {} };
      }
    }
    return { input, output, paramTargets: targets };
  }

  input.connect(worklet);
  worklet.connect(output);

  const wpar = (name) => {
    if (!worklet.parameters) return null;
    return worklet.parameters.get(name) || null;
  };

  // Helper: bind '@'-modulated param to AudioParam directly when the
  // param surface is a simple numeric AudioParam.
  const bindNumericParam = (key, paramName, transform) => {
    const val = params[key];
    const ap  = wpar(paramName);
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
      } else {
        targets[id] = { paramDef: paramDefs[id], customSetter: () => {} };
      }
    } else if (ap && typeof val === 'number') {
      ap.value = transform ? transform(val) : val;
    }
  };

  // ── Param wiring ──────────────────────────────────────────────────
  bindNumericParam('key', 'key', (v) => Math.max(0, Math.min(11, Math.round(v))));
  bindNumericParam('response_ms', 'response_ms');
  bindNumericParam('correction_amount', 'correction_amount');
  bindNumericParam('formant_preserve', 'formant_preserve', (v) => v ? 1 : 0);
  bindNumericParam('mix', 'mix');

  // scale + scale_mask: 'scale' is a name, 'scale_mask' is the raw uint12.
  // If either is modulated we install a setter that recomputes the mask.
  const scaleAp = wpar('scale_mask');
  let cachedScale     = (typeof params.scale === 'string') ? params.scale : null;
  let cachedCustomMsk = (typeof params.scale_mask === 'number') ? params.scale_mask : null;

  const applyMask = () => {
    if (!scaleAp) return;
    const m = _scaleNameToMask(cachedScale, cachedCustomMsk);
    scaleAp.value = m;
  };

  if (_isModulated(params.scale)) {
    const id = params.scale.slice(1);
    targets[id] = {
      paramDef: paramDefs[id],
      customSetter: (v) => {
        // Accept numeric enum (0..3) or string
        if (typeof v === 'string') {
          cachedScale = v;
        } else if (typeof v === 'number') {
          // 0=major, 1=minor, 2=chromatic, 3=custom (fallback contract)
          const enumNames = ['major', 'minor', 'chromatic', 'custom'];
          cachedScale = enumNames[Math.max(0, Math.min(3, Math.round(v)))] || 'chromatic';
        }
        applyMask();
      },
    };
  } else if (typeof params.scale === 'string') {
    cachedScale = params.scale;
    // Apply only if no explicit numeric scale_mask present and no @ on it.
    if (!_isModulated(params.scale_mask) && typeof params.scale_mask !== 'number') {
      applyMask();
    }
  }

  if (_isModulated(params.scale_mask)) {
    const id = params.scale_mask.slice(1);
    if (scaleAp) {
      targets[id] = {
        paramDef: paramDefs[id],
        customSetter: (v) => {
          cachedCustomMsk = (typeof v === 'number') ? v : 0;
          // If the user set a literal mask we treat scale as 'custom' for
          // resolution; otherwise the previous cachedScale stands.
          const prev = cachedScale;
          cachedScale = 'custom';
          applyMask();
          cachedScale = prev || 'custom';
        },
      };
    } else {
      targets[id] = { paramDef: paramDefs[id], customSetter: () => {} };
    }
  } else if (typeof params.scale_mask === 'number' && scaleAp) {
    // Literal mask — apply directly. (Already factored into initialMask
    // via parameterData, but reapply for safety in case the AudioParam
    // diverged from the constructor data.)
    scaleAp.value = Math.max(0, Math.min(0xFFF, Math.round(params.scale_mask)));
  }

  return { input, output, paramTargets: targets, workletNode: worklet };
}

const R13_BUILDERS = {
  pitch_correct: buildPitchCorrect,
};

export default R13_BUILDERS;
