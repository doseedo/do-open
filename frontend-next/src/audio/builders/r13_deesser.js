/**
 * R13 — DeEsser 2 (Logic Pro stock parity)
 *
 * Registers `deesser` as a NEW node type backed by a single AudioWorklet
 * (`r13-deesser-processor`) that fuses sibilance-band detection (cascade
 * highpass + lowpass), asymmetric envelope follower, and a dynamic peaking
 * biquad cut driven by the envelope's overshoot above threshold.
 *
 * Logic's DeEsser 2 is a frequency-selective dynamics processor — it leaves
 * the rest of the spectrum untouched and only reduces gain in the sibilant
 * band when the detection envelope crosses threshold.
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildDeEsser(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: {...} } }
 *
 * Schema:
 *   {
 *     type: 'deesser',
 *     params: {
 *       freq_low:     1500..10000  Hz   (default 4000)
 *       freq_high:    5000..15000  Hz   (default 9000)
 *       threshold_db: -60..0       dB   (default -28)
 *       range_db:     0..24        dB   (default 12)  — max dynamic cut
 *       attack_ms:    0.1..10      ms   (default 1.5)
 *       release_ms:   10..200      ms   (default 40)
 *       q:            0.5..10            (default 2.0)
 *       monitor:      0|1                (default 0)  — emit detection tap
 *     }
 *   }
 *
 * Param values may be literals or '@<paramId>' bindings (live-modulated).
 *
 * Fallback: if AudioWorkletNode isn't available (SSR / older Safari / before
 * `audioWorklet.addModule(...)` resolves) we degrade to a static-cut
 * BiquadFilterNode peaking EQ centred at `(freq_low * freq_high)^0.5` with
 * gain = `-range_db * 0.5` (a sensible mid-amount cut so material isn't left
 * unprocessed). The detection envelope is unavailable in this path so the
 * cut becomes time-invariant; modulated params still bind so the engine's
 * graph build never fails.
 *
 * @author Doseedo R13
 */

const R13_DEESSER_PROCESSOR = 'r13-deesser-processor';

function _safeWorklet(ctx, name, options) {
  try {
    return new AudioWorkletNode(ctx, name, options);
  } catch (e) {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[R13] worklet ${name} unavailable, using static peaking-biquad fallback:`,
                    e && e.message);
    }
    return null;
  }
}

function _isModulated(v) {
  return typeof v === 'string' && v.startsWith('@');
}

function _clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function _centerHz(fLow, fHigh) {
  // Geometric mean — same convention as the worklet
  return Math.sqrt(Math.max(20, fLow * fHigh));
}

export function buildDeEsser(ctx, nodeDef, paramDefs) {
  const params = (nodeDef && nodeDef.params) || {};
  const targets = {};

  const input  = ctx.createGain();
  const output = ctx.createGain();

  // Resolve initial constants (modulated entries → defaults)
  const initialFreqLow     = _isModulated(params.freq_low)     ? 4000  : (typeof params.freq_low     === 'number' ? params.freq_low     : 4000);
  const initialFreqHigh    = _isModulated(params.freq_high)    ? 9000  : (typeof params.freq_high    === 'number' ? params.freq_high    : 9000);
  const initialThresholdDb = _isModulated(params.threshold_db) ? -28   : (typeof params.threshold_db === 'number' ? params.threshold_db : -28);
  const initialRangeDb     = _isModulated(params.range_db)     ? 12    : (typeof params.range_db     === 'number' ? params.range_db     : 12);
  const initialAttackMs    = _isModulated(params.attack_ms)    ? 1.5   : (typeof params.attack_ms    === 'number' ? params.attack_ms    : 1.5);
  const initialReleaseMs   = _isModulated(params.release_ms)   ? 40    : (typeof params.release_ms   === 'number' ? params.release_ms   : 40);
  const initialQ           = _isModulated(params.q)            ? 2.0   : (typeof params.q            === 'number' ? params.q            : 2.0);
  const initialMonitor     = _isModulated(params.monitor)      ? 0
                              : (params.monitor === true || params.monitor === 1 ? 1 : 0);

  const worklet = _safeWorklet(ctx, R13_DEESSER_PROCESSOR, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    parameterData: {
      freq_low:     initialFreqLow,
      freq_high:    initialFreqHigh,
      threshold_db: initialThresholdDb,
      range_db:     initialRangeDb,
      attack_ms:    initialAttackMs,
      release_ms:   initialReleaseMs,
      q:            initialQ,
      monitor:      initialMonitor,
    },
  });

  // ── Fallback path ───────────────────────────────────────────────────────
  // BiquadFilterNode peaking EQ — a static-cut approximation. The detection
  // envelope is unavailable here; we apply half the configured range_db as
  // a constant cut so material is still treated. The graph stays buildable
  // and live param drags still resolve (they update centre/Q/gain on the
  // biquad directly).
  let fallbackPeak = null;
  if (!worklet) {
    let peak = null;
    try {
      peak = ctx.createBiquadFilter();
      peak.type = 'peaking';
      peak.frequency.value = _centerHz(initialFreqLow, initialFreqHigh);
      peak.Q.value = initialQ;
      peak.gain.value = -0.5 * initialRangeDb;
    } catch (e) {
      // ctx without BiquadFilterNode (very minimal stubs) — wire passthrough
      input.connect(output);
      // Bind any '@'-modulated params as no-ops so the engine still resolves
      // the modulation graph without errors.
      for (const [, val] of Object.entries(params)) {
        if (_isModulated(val)) {
          const id = val.slice(1);
          targets[id] = { paramDef: paramDefs[id], customSetter: () => {} };
        }
      }
      return { input, output, paramTargets: targets };
    }

    fallbackPeak = peak;
    input.connect(peak);
    peak.connect(output);

    // ── Param wiring (fallback) ──
    // The peaking biquad has freq / Q / gain AudioParams. range_db / threshold_db
    // / attack_ms / release_ms / freq_low / freq_high don't have direct biquad
    // analogues, so we maintain cached scalar state and recompute the biquad
    // setpoints on every relevant edit.
    const cache = {
      freq_low:     initialFreqLow,
      freq_high:    initialFreqHigh,
      range_db:     initialRangeDb,
      q:            initialQ,
      monitor:      initialMonitor,
    };
    const reapply = () => {
      try {
        peak.frequency.value = _centerHz(cache.freq_low, cache.freq_high);
        peak.Q.value         = _clamp(cache.q, 0.05, 100);
        peak.gain.value      = cache.monitor ? 0 : (-0.5 * cache.range_db);
      } catch (_) { /* ignore */ }
    };

    for (const [key, val] of Object.entries(params)) {
      if (!_isModulated(val)) continue;
      const id = val.slice(1);
      switch (key) {
        case 'freq_low':
          targets[id] = { paramDef: paramDefs[id], customSetter: (v) => {
            cache.freq_low = _clamp(v, 1500, 10000); reapply();
          }};
          break;
        case 'freq_high':
          targets[id] = { paramDef: paramDefs[id], customSetter: (v) => {
            cache.freq_high = _clamp(v, 5000, 15000); reapply();
          }};
          break;
        case 'q':
          targets[id] = { paramDef: paramDefs[id], customSetter: (v) => {
            cache.q = _clamp(v, 0.5, 10); reapply();
          }};
          break;
        case 'range_db':
          targets[id] = { paramDef: paramDefs[id], customSetter: (v) => {
            cache.range_db = _clamp(v, 0, 24); reapply();
          }};
          break;
        case 'monitor':
          targets[id] = { paramDef: paramDefs[id], customSetter: (v) => {
            cache.monitor = v ? 1 : 0; reapply();
          }};
          break;
        // threshold_db / attack_ms / release_ms have no static-biquad analogue;
        // bind as no-ops so the engine doesn't surface an unbound paramId.
        case 'threshold_db':
        case 'attack_ms':
        case 'release_ms':
          targets[id] = { paramDef: paramDefs[id], customSetter: () => {} };
          break;
        default:
          targets[id] = { paramDef: paramDefs[id], customSetter: () => {} };
          break;
      }
    }

    return { input, output, paramTargets: targets, fallbackPeak };
  }

  // ── Worklet path ────────────────────────────────────────────────────────
  input.connect(worklet);
  worklet.connect(output);

  const wpar = (name) => {
    if (!worklet.parameters) return null;
    return worklet.parameters.get(name) || null;
  };

  // Bind '@' param to AudioParam (or customSetter when the surface needs a transform)
  const bindNumeric = (key, paramName, transform) => {
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

  bindNumeric('freq_low',     'freq_low');
  bindNumeric('freq_high',    'freq_high');
  bindNumeric('threshold_db', 'threshold_db');
  bindNumeric('range_db',     'range_db');
  bindNumeric('attack_ms',    'attack_ms');
  bindNumeric('release_ms',   'release_ms');
  bindNumeric('q',            'q');
  bindNumeric('monitor',      'monitor', (v) => (v ? 1 : 0));

  return { input, output, paramTargets: targets, workletNode: worklet };
}

// ── Default export: NODE_BUILDERS map ─────────────────────────────────────
const R13_DEESSER_BUILDERS = {
  deesser: buildDeEsser,
};

export default R13_DEESSER_BUILDERS;
