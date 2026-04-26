/**
 * R13 — Multipressor (4-band parallel compressor) builder
 *
 * Registers a NEW DSP node type `multipressor` that mirrors Logic Pro's
 * Multipressor: a 4-band multiband compressor where each band has its own
 * threshold / ratio / attack / release / makeup gain. The four bands are
 * split via Linkwitz-Riley 4th-order crossovers and summed phase-coherent
 * at the output, so when no band is compressing the dry signal is faithfully
 * reconstructed (up to crossover allpass distortion).
 *
 * Pipeline (non-worklet path):
 *
 *                 ┌─ LR-LP@xo1 ─ band0 ─┐
 *                 ├─ BP(xo1..xo2) ─ band1 ─┤
 *   input ──── ↳ ─                          ─ Σ ── master_gain ── output
 *                 ├─ BP(xo2..xo3) ─ band2 ─┤
 *                 └─ LR-HP@xo3 ─ band3 ─┘
 *
 * Each "band" is `BiquadFilterNode chain → DynamicsCompressorNode → makeupGain`.
 * BiquadFilters are configured as Linkwitz-Riley 4th-order (two cascaded
 * 2nd-order Butterworth filters at the same cutoff, q=0.7071). For
 * band-pass slots (band1, band2) we cascade a LR-HP at the lower edge with
 * a LR-LP at the upper edge.
 *
 * Pure LR4 has the property that LP(f) + HP(f) = ALLPASS(f), so summing
 * the bands is allpass-equivalent to the input. Cascaded LR4s for the
 * mid bands accept an extra phase rotation but stay close to flat.
 *
 * Worklet path: a `r13-multipressor-processor` is reserved for a single-
 * pass multi-band implementation that does the full LR4 split + per-band
 * compression in one ScriptProcessor-style render quantum (lower CPU,
 * tighter latency). If the worklet module isn't loaded yet (engine
 * bootstrap hasn't called `audioWorklet.addModule(...)`), the builder
 * silently uses the BiquadFilter+DynamicsCompressor fallback.
 *
 * Lookahead: the global `lookahead_ms` shifts the dry input stream by
 * the requested amount (DelayNode) before splitting. This is the same
 * approximation Logic exposes — it doesn't see "the future", it just
 * delays the audio so the detector sees peaks slightly ahead of the
 * processed sample.
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildMultipressor(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: {...} } }
 *
 * Author: Agent R13 (Tier 2 / Multipressor)
 */

const R13_MULTIPRESSOR_PROCESSOR = 'r13-multipressor-processor';

const NUM_BANDS = 4;
const DEFAULT_CROSSOVERS = [120, 800, 4000];

// LR-4 = two cascaded Butterworth (q = 1/sqrt(2)) at the same cutoff
const BUTTER_Q = Math.SQRT1_2;

// ── Helpers ───────────────────────────────────────────────────────────────

function isAtBinding(v) {
  return typeof v === 'string' && v.startsWith('@');
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function dbToLinear(db) {
  return Math.pow(10, db / 20);
}

function _safeWorklet(ctx, name, options = {}) {
  try {
    return new AudioWorkletNode(ctx, name, options);
  } catch (e) {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[R13.multipressor] worklet ${name} unavailable, using fallback:`, e && e.message);
    }
    return null;
  }
}

/**
 * Build a Linkwitz-Riley 4th-order lowpass/highpass = two cascaded biquads
 * with q = 0.7071. Returns { input, output, frequency: [param, param] }
 * so the calling code can drive both biquads' frequency in lockstep.
 */
function _buildLR4(ctx, type, freqHz) {
  const a = ctx.createBiquadFilter();
  const b = ctx.createBiquadFilter();
  a.type = type; b.type = type;
  a.Q.value = BUTTER_Q; b.Q.value = BUTTER_Q;
  a.frequency.value = freqHz; b.frequency.value = freqHz;
  a.connect(b);
  return {
    input: a,
    output: b,
    setFrequency(f) { a.frequency.value = f; b.frequency.value = f; },
    frequencyParams: [a.frequency, b.frequency],
  };
}

/**
 * Apply a Logic-style scalar parameter onto a DynamicsCompressorNode.
 * Tolerates several parameter spellings (threshold_db, threshold, etc.)
 * to match the rest of the dsplang.
 */
function _bindCompressorScalar(comp, makeupGain, key, value) {
  if (typeof value !== 'number') return;
  switch (key) {
    case 'threshold_db':
    case 'threshold':
      comp.threshold.value = clamp(value, -100, 0);
      break;
    case 'ratio':
      comp.ratio.value = clamp(value, 1, 50);
      break;
    case 'attack_ms':
    case 'attack':
      comp.attack.value = clamp(value / 1000, 0, 1);
      break;
    case 'release_ms':
    case 'release':
      comp.release.value = clamp(value / 1000, 0.001, 2);
      break;
    case 'knee_db':
    case 'knee':
      comp.knee.value = clamp(value, 0, 40);
      break;
    case 'gain_db':
    case 'makeup_db':
    case 'makeup':
      makeupGain.gain.value = dbToLinear(value);
      break;
    default:
      break;
  }
}

// ── Builder ───────────────────────────────────────────────────────────────

export function buildMultipressor(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  // ── Try worklet path first ──────────────────────────────────────────────
  const worklet = _safeWorklet(ctx, R13_MULTIPRESSOR_PROCESSOR, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    parameterData: {
      crossover_1: typeof params.crossover_1 === 'number' ? params.crossover_1 : DEFAULT_CROSSOVERS[0],
      crossover_2: typeof params.crossover_2 === 'number' ? params.crossover_2 : DEFAULT_CROSSOVERS[1],
      crossover_3: typeof params.crossover_3 === 'number' ? params.crossover_3 : DEFAULT_CROSSOVERS[2],
      lookahead_ms: typeof params.lookahead_ms === 'number' ? params.lookahead_ms : 0,
      output_gain: typeof params.output_gain === 'number' ? params.output_gain : 0,
    },
  });

  // Common output stage (both paths)
  const input = ctx.createGain();
  const output = ctx.createGain();

  if (worklet) {
    // Worklet path: route input → worklet → output. The worklet is
    // expected to expose AudioParams for crossovers + per-band
    // {threshold,ratio,attack,release,gain,bypass} + master output_gain
    // + lookahead_ms.
    input.connect(worklet);
    worklet.connect(output);

    const wpar = (name) => (worklet.parameters && worklet.parameters.get(name)) || null;

    // Direct-set non-modulated params
    if (typeof params.output_gain === 'number') {
      const ap = wpar('output_gain');
      if (ap) ap.value = params.output_gain;
    }

    // Wire @-bindings — if the worklet exposes the named param we bind
    // directly; otherwise install a customSetter no-op so the engine
    // doesn't crash on a missing target.
    for (const [key, val] of Object.entries(params)) {
      if (!isAtBinding(val)) continue;
      const paramId = val.slice(1);
      const ap = wpar(key);
      if (ap) {
        const isMs = key.endsWith('_ms') && key !== 'lookahead_ms';
        targets[paramId] = {
          audioParam: ap,
          paramDef: paramDefs[paramId],
          ...(isMs ? { scale: (v) => v / 1000 } : {}),
        };
      } else {
        targets[paramId] = {
          paramDef: paramDefs[paramId],
          customSetter: () => { /* worklet doesn't expose this — silent */ },
        };
      }
    }

    return { input, output, paramTargets: targets, multipressorWorklet: worklet };
  }

  // ── Fallback path: BiquadFilter LR4 + DynamicsCompressor × 4 ────────────

  // Lookahead delay (applied once before the splitter)
  const lookaheadMs = typeof params.lookahead_ms === 'number' ? params.lookahead_ms : 0;
  const lookahead = ctx.createDelay(0.05); // 50 ms cap — Logic cap is 10
  lookahead.delayTime.value = clamp(lookaheadMs / 1000, 0, 0.05);
  input.connect(lookahead);

  // Crossover frequencies (defaults / numeric overrides)
  const xo = [
    typeof params.crossover_1 === 'number' ? params.crossover_1 : DEFAULT_CROSSOVERS[0],
    typeof params.crossover_2 === 'number' ? params.crossover_2 : DEFAULT_CROSSOVERS[1],
    typeof params.crossover_3 === 'number' ? params.crossover_3 : DEFAULT_CROSSOVERS[2],
  ];

  // Build the 4 bands. For each, an "entry" GainNode is the band's input
  // (so we can connect lookahead → bandEntry without re-cascading).
  //
  //   band[0] = LR-LP @ xo[0]
  //   band[1] = LR-HP @ xo[0] → LR-LP @ xo[1]
  //   band[2] = LR-HP @ xo[1] → LR-LP @ xo[2]
  //   band[3] = LR-HP @ xo[2]
  const bandLP = [_buildLR4(ctx, 'lowpass',  xo[0]),
                  _buildLR4(ctx, 'lowpass',  xo[1]),
                  _buildLR4(ctx, 'lowpass',  xo[2])];
  const bandHP = [_buildLR4(ctx, 'highpass', xo[0]),
                  _buildLR4(ctx, 'highpass', xo[1]),
                  _buildLR4(ctx, 'highpass', xo[2])];

  // Band entries: for serial chains we route lookahead → entry → ... → comp
  const bandInputs = [
    ctx.createGain(), ctx.createGain(), ctx.createGain(), ctx.createGain(),
  ];
  // Band 0: lookahead → bandLP[0] (LP @ xo0)
  lookahead.connect(bandInputs[0]);
  bandInputs[0].connect(bandLP[0].input);

  // Band 1: lookahead → bandHP[0] (HP @ xo0) → bandLP[1] (LP @ xo1)
  lookahead.connect(bandInputs[1]);
  bandInputs[1].connect(bandHP[0].input);
  bandHP[0].output.connect(bandLP[1].input);

  // Band 2: lookahead → bandHP[1] (HP @ xo1) → bandLP[2] (LP @ xo2)
  lookahead.connect(bandInputs[2]);
  bandInputs[2].connect(bandHP[1].input);
  bandHP[1].output.connect(bandLP[2].input);

  // Band 3: lookahead → bandHP[2] (HP @ xo2)
  lookahead.connect(bandInputs[3]);
  bandInputs[3].connect(bandHP[2].input);

  // Per-band compressor + makeup-gain stage. Default values match Logic's
  // Multipressor opening defaults reasonably closely (band-progressive
  // thresholds, 3:1 ratio, mild attack/release).
  const bands = [];
  const DEFAULT_BAND_PARAMS = [
    { threshold_db: -22, ratio: 3, attack_ms: 20, release_ms: 200, gain_db: 0, bypass: false },
    { threshold_db: -20, ratio: 3, attack_ms: 15, release_ms: 150, gain_db: 0, bypass: false },
    { threshold_db: -18, ratio: 3, attack_ms: 10, release_ms: 120, gain_db: 0, bypass: false },
    { threshold_db: -16, ratio: 3, attack_ms: 5,  release_ms: 100, gain_db: 0, bypass: false },
  ];
  for (let i = 0; i < NUM_BANDS; i++) {
    const comp = ctx.createDynamicsCompressor();
    const makeup = ctx.createGain();
    const bypassWet = ctx.createGain();
    const bypassDry = ctx.createGain();
    bypassDry.gain.value = 0; // bypass routes around comp; default OFF
    bypassWet.gain.value = 1;
    comp.knee.value = 6;

    // Apply defaults
    const dflt = DEFAULT_BAND_PARAMS[i];
    _bindCompressorScalar(comp, makeup, 'threshold_db', dflt.threshold_db);
    _bindCompressorScalar(comp, makeup, 'ratio',        dflt.ratio);
    _bindCompressorScalar(comp, makeup, 'attack_ms',    dflt.attack_ms);
    _bindCompressorScalar(comp, makeup, 'release_ms',   dflt.release_ms);
    _bindCompressorScalar(comp, makeup, 'gain_db',      dflt.gain_db);

    // Apply numeric param overrides (per-band)
    const prefix = `band${i + 1}_`;
    const altPrefix = `band[${i + 1}]_`;
    const altPrefix2 = `band${i}_`; // 0-indexed, also accepted
    for (const [key, val] of Object.entries(params)) {
      const matchStr = key.startsWith(prefix)
        ? key.slice(prefix.length)
        : key.startsWith(altPrefix)
          ? key.slice(altPrefix.length)
          : (key.startsWith(altPrefix2) ? key.slice(altPrefix2.length) : null);
      if (!matchStr) continue;
      if (typeof val === 'number') {
        if (matchStr === 'bypass') {
          bypassDry.gain.value = val ? 1 : 0;
          bypassWet.gain.value = val ? 0 : 1;
        } else {
          _bindCompressorScalar(comp, makeup, matchStr, val);
        }
      }
    }

    // Routing: bandLP_or_HP output → splitTap → (compressor → makeup → bypassWet) and (→ bypassDry)
    const splitTap = ctx.createGain();
    if (i === 0) bandLP[0].output.connect(splitTap);
    else if (i === 1) bandLP[1].output.connect(splitTap);
    else if (i === 2) bandLP[2].output.connect(splitTap);
    else bandHP[2].output.connect(splitTap);

    splitTap.connect(comp);
    comp.connect(makeup);
    makeup.connect(bypassWet);
    splitTap.connect(bypassDry);

    bands.push({ comp, makeup, bypassWet, bypassDry, splitTap });
  }

  // Sum bands → master gain → output
  const summer = ctx.createGain();
  for (const b of bands) {
    b.bypassWet.connect(summer);
    b.bypassDry.connect(summer);
  }
  const masterGain = ctx.createGain();
  masterGain.gain.value = typeof params.output_gain === 'number' ? dbToLinear(params.output_gain) : 1.0;
  summer.connect(masterGain);
  masterGain.connect(output);

  // ── Modulated (@-bound) param wiring ──────────────────────────────────
  for (const [key, val] of Object.entries(params)) {
    if (!isAtBinding(val)) continue;
    const paramId = val.slice(1);
    const pDef = paramDefs[paramId];

    // crossover_N → drive both biquads' frequency
    if (key === 'crossover_1' || key === 'crossover_2' || key === 'crossover_3') {
      const idx = Number(key.split('_')[1]) - 1;
      targets[paramId] = {
        paramDef: pDef,
        customSetter: (v) => {
          const f = clamp(v, 20, 20000);
          // Update the LP at xo[idx] AND the HP at xo[idx] in lockstep
          bandLP[idx].setFrequency(f);
          bandHP[idx].setFrequency(f);
        },
      };
      continue;
    }

    // output_gain (master), in dB
    if (key === 'output_gain') {
      const isDb = pDef?.unit === 'dB' || pDef?.unit === 'db';
      targets[paramId] = {
        audioParam: masterGain.gain,
        paramDef: pDef,
        scale: isDb ? dbToLinear : undefined,
      };
      continue;
    }

    // lookahead_ms (global)
    if (key === 'lookahead_ms') {
      targets[paramId] = {
        audioParam: lookahead.delayTime,
        paramDef: pDef,
        scale: (v) => clamp((v || 0) / 1000, 0, 0.05),
      };
      continue;
    }

    // Per-band params: band1_threshold_db / band[1]_threshold_db / band0_threshold_db
    let bandIdx = -1, sub = null;
    for (let i = 0; i < NUM_BANDS; i++) {
      const p1 = `band${i + 1}_`, p2 = `band[${i + 1}]_`, p3 = `band${i}_`;
      if (key.startsWith(p1)) { bandIdx = i; sub = key.slice(p1.length); break; }
      if (key.startsWith(p2)) { bandIdx = i; sub = key.slice(p2.length); break; }
      if (key.startsWith(p3)) { bandIdx = i; sub = key.slice(p3.length); break; }
    }
    if (bandIdx < 0 || !sub) continue;
    const band = bands[bandIdx];

    switch (sub) {
      case 'threshold_db':
      case 'threshold':
        targets[paramId] = { audioParam: band.comp.threshold, paramDef: pDef };
        break;
      case 'ratio':
        targets[paramId] = { audioParam: band.comp.ratio, paramDef: pDef };
        break;
      case 'attack_ms':
      case 'attack':
        targets[paramId] = { audioParam: band.comp.attack, paramDef: pDef, scale: (v) => clamp(v / 1000, 0, 1) };
        break;
      case 'release_ms':
      case 'release':
        targets[paramId] = { audioParam: band.comp.release, paramDef: pDef, scale: (v) => clamp(v / 1000, 0.001, 2) };
        break;
      case 'knee_db':
      case 'knee':
        targets[paramId] = { audioParam: band.comp.knee, paramDef: pDef };
        break;
      case 'gain_db':
      case 'makeup':
      case 'makeup_db': {
        const isDb = pDef?.unit === 'dB' || pDef?.unit === 'db' || sub.endsWith('_db');
        targets[paramId] = {
          audioParam: band.makeup.gain,
          paramDef: pDef,
          scale: isDb ? dbToLinear : undefined,
        };
        break;
      }
      case 'bypass':
        targets[paramId] = {
          paramDef: pDef,
          customSetter: (v) => {
            const on = !!v;
            band.bypassDry.gain.value = on ? 1 : 0;
            band.bypassWet.gain.value = on ? 0 : 1;
          },
        };
        break;
      default:
        // Unknown subparam — install a no-op so binding doesn't fail
        targets[paramId] = { paramDef: pDef, customSetter: () => {} };
        break;
    }
  }

  return {
    input,
    output,
    paramTargets: targets,
    // Internal references for tests / debugging
    multipressorBands: bands,
    multipressorCrossovers: { lp: bandLP, hp: bandHP },
    multipressorLookahead: lookahead,
    multipressorMasterGain: masterGain,
  };
}

// ── Default export: NODE_BUILDERS map ─────────────────────────────────────
const R13_MULTIPRESSOR_BUILDERS = {
  multipressor: buildMultipressor,
};

export default R13_MULTIPRESSOR_BUILDERS;
