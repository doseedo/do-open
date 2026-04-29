/**
 * R13 — Vintage EQ Collection (1073 + API)
 *
 * Implements two new DSP node types modelled on Logic's Vintage EQ Collection:
 *
 *   `vintage_1073` — Neon-style, modeled on the Neve 1073 console EQ.
 *                    Passive LCR low-cut + low shelf + mid bell + fixed 12 kHz
 *                    high shelf, plus an inductor-saturation waveshaper.
 *
 *   `vintage_api`  — Punchy-style, modeled on the API 550A. 4 bands of
 *                    proportional-Q peaking filters at 12 fixed frequencies,
 *                    plus an inductor-saturation waveshaper.
 *
 * **Pultec already lives in r4.js (`circuit_pultec_eq`).** R13 does not
 * touch that path — the two are parallel circuit models.
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildVintageXxx(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: { audioParam?, paramDef, customSetter? } } }
 *
 * Following the convention from R4/R9: if the optional worklet processor
 * isn't registered yet on the AudioContext (i.e. the engine hasn't called
 * `audioWorklet.addModule(...)` for the R13 processors), we fall back to
 * BiquadFilterNodes + a static WaveShaper. That fallback is fully audible
 * and tracks the linear EQ response — only the per-band Q-vs-gain curve is
 * approximated rather than continuous. Once the worklet is loaded the
 * builder picks up the real per-block coefficient updates with no code
 * change.
 *
 * Author: Agent R13
 */

// ── Worklet processor names (optional path; safe to be absent) ────────────
const R13_VINTAGE_1073 = 'r13-vintage-1073-processor';
const R13_VINTAGE_API  = 'r13-vintage-api-processor';

// ── 1073 frequency tables ─────────────────────────────────────────────────
// Indexed knobs: position N → Hz. Position 0 means "off" for HPF.
// Source: Neve 1073 channel module spec, Neumann/AMS Neve manuals.
const N1073_LOW_CUT_FREQS  = [0, 50, 80, 160, 300];          // 0 = OFF (bypass)
const N1073_LOW_SHELF_FREQS = [35, 60, 110, 220];            // Hz (last = 220 added; 1073 stock is 35/60/110, leave 220 for parity headroom — selectable via index 3)
// NOTE: Stock 1073 has 3 LF settings (35/60/110). We expose 4 to match the
// "Vintage EQ" plug-in's superset; calibration can ignore index 3 if needed.
const N1073_MID_FREQS       = [360, 700, 1600, 3200, 4800, 7200]; // Hz
const N1073_HIGH_SHELF_HZ   = 12000;                         // fixed
const N1073_MID_Q_PRESETS   = { broad: 0.7, medium: 1.4, narrow: 2.8 };

// ── API 550A / 560 fixed frequency table ──────────────────────────────────
// Per the brief: 50/100/200/400/800/1500/3000/5000/7500/12500/15000/20000.
const API_FREQ_TABLE = [50, 100, 200, 400, 800, 1500, 3000, 5000, 7500, 12500, 15000, 20000];

// ── Q-vs-gain curves ──────────────────────────────────────────────────────
// 1073 mid bell — flat-Q "broad" mode is gentle (Q≈0.7) at low gain and tightens
// to ~1.5 at extreme boost/cut. We pre-compute as a remap function so a `gain`
// change updates the BiquadFilterNode's Q at the same time.
function n1073MidQ(gainDb, qPreset = 'medium') {
  const baseQ = N1073_MID_Q_PRESETS[qPreset] ?? N1073_MID_Q_PRESETS.medium;
  // Q proportional to |gain|/18, capped: Q_min = 0.5*base, Q_max = 1.6*base
  const norm = Math.min(1, Math.abs(gainDb) / 18);
  const factor = 0.5 + norm * 1.1;        // 0.5 → 1.6
  return baseQ * factor;
}

// API 550 — proportional Q: at low gain Q≈0.6 (very broad), at full ±12 dB Q≈2.5.
// This is the classic "the more you push it the more it cuts/boosts a tighter slice"
// behaviour described in API marketing & Massenburg's notes on the original design.
function apiProportionalQ(gainDb) {
  const norm = Math.min(1, Math.abs(gainDb) / 12);
  return 0.6 + norm * (2.5 - 0.6);        // 0.6 .. 2.5
}

// ── Saturation curve (3rd-order soft clip with even-harmonic asymmetry) ──
// Used to model inductor core saturation at extreme settings. Drive 0..1
// where 0 ≈ inaudible, 1 ≈ pleasant warm crunch.
function makeInductorCurve(drive) {
  const len = 2048;
  const curve = new Float32Array(len);
  const k = 1 + drive * 4;            // pre-gain; 1..5
  for (let i = 0; i < len; i++) {
    const x = (i / (len - 1)) * 2 - 1;
    const sx = x * k;
    // 3rd-order soft clip, slight even-harmonic bias (asymmetric 0.04 offset)
    const y = (sx - (sx * sx * sx) / 3) * (1 - drive * 0.05) + (sx * sx) * 0.04 * drive;
    // Tame to ±1
    curve[i] = Math.max(-1, Math.min(1, y * (1 / Math.max(1, k * 0.6))));
  }
  return curve;
}

// ── Internal helpers ──────────────────────────────────────────────────────
function _safeWorklet(ctx, name, options = {}) {
  try {
    return new AudioWorkletNode(ctx, name, options);
  } catch (e) {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[R13] worklet ${name} unavailable, using fallback:`, e && e.message);
    }
    return null;
  }
}

function _isModulated(v) {
  return typeof v === 'string' && v.startsWith('@');
}

// Look up index in an array. Accepts integer indices or floats (rounded).
// Out-of-range clamps to nearest valid index.
function _indexedLookup(table, idx) {
  if (idx == null) return table[0];
  const i = Math.max(0, Math.min(table.length - 1, Math.round(Number(idx))));
  return table[i];
}

// ── Build: vintage_1073 ───────────────────────────────────────────────────
//
// Topology (signal flow):
//   input → highpass (4 cutoffs + off)
//         → low shelf (3-4 freqs, ±18 dB)
//         → mid bell  (5-6 freqs, ±18 dB, broad/medium/narrow Q)
//         → high shelf @ 12 kHz fixed (±18 dB)
//         → inductor saturation (waveshaper, drive = inductor_saturation)
//         → output_gain
//
// Param map (all `paramDefs` are scaled by engine before customSetter fires):
//   low_cut_freq      indexed 0..4  → off / 50 / 80 / 160 / 300 Hz
//   low_shelf_freq    indexed 0..3  → 35 / 60 / 110 / 220 Hz
//   low_shelf_gain    -18..+18 dB
//   mid_freq          indexed 0..5  → 360 / 700 / 1600 / 3200 / 4800 / 7200 Hz
//   mid_gain          -18..+18 dB
//   mid_q             0..1 → broad / medium / narrow (3-bin)
//   high_shelf_gain   -18..+18 dB  (band fixed at 12 kHz)
//   inductor_saturation 0..1
//   output_gain       0..2 (linear)
//
export function buildVintage1073(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const input  = ctx.createGain();
  const output = ctx.createGain();

  // Linear EQ chain (the workhorse audible path)
  const hpf       = ctx.createBiquadFilter();
  hpf.type        = 'highpass';
  hpf.frequency.value = 20;     // pre-set "off" — 20 Hz subsonic
  hpf.Q.value     = 0.707;

  const lowShelf  = ctx.createBiquadFilter();
  lowShelf.type   = 'lowshelf';
  lowShelf.frequency.value = 60;
  lowShelf.gain.value = 0;

  const midBell   = ctx.createBiquadFilter();
  midBell.type    = 'peaking';
  midBell.frequency.value = 700;
  midBell.Q.value = N1073_MID_Q_PRESETS.medium;
  midBell.gain.value = 0;

  const highShelf = ctx.createBiquadFilter();
  highShelf.type  = 'highshelf';
  highShelf.frequency.value = N1073_HIGH_SHELF_HZ;
  highShelf.gain.value = 0;

  // Inductor saturation
  const inductor = ctx.createWaveShaper();
  inductor.curve = makeInductorCurve(0); // start clean
  inductor.oversample = '2x';

  const outGain = ctx.createGain();
  outGain.gain.value = 1;

  // Optional worklet (per-block coeff updates → continuous Q-vs-gain). The
  // builder still places the biquad chain as the audible fallback path; if
  // the worklet is present we wrap it as an extra parallel "shape" stage so
  // either path produces audio. Keeping both wired keeps the audio graph
  // stable across hot reload.
  const worklet = _safeWorklet(ctx, R13_VINTAGE_1073, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    parameterData: { ind_drive: 0 },
  });

  // Internal mid-Q preset state (mid_gain reads this)
  let midQPreset = 'medium';
  let lowCutIdx  = 0;

  // Wire chain: input → hpf → lowShelf → midBell → highShelf → inductor → outGain → output
  // If a worklet is loaded, splice it after highShelf so it sees the linear curve.
  input.connect(hpf);
  hpf.connect(lowShelf);
  lowShelf.connect(midBell);
  midBell.connect(highShelf);
  let post = highShelf;
  if (worklet) {
    highShelf.connect(worklet);
    post = worklet;
  }
  post.connect(inductor);
  inductor.connect(outGain);
  outGain.connect(output);

  const wpar = (name) => (worklet && worklet.parameters)
    ? (worklet.parameters.get(name) || null)
    : null;

  // ── Param wiring ─────────────────────────────────────────────────────────
  for (const [key, val] of Object.entries(params)) {
    const isMod = _isModulated(val);
    const paramId = isMod ? val.slice(1) : null;
    const def = isMod ? (paramDefs[paramId] || {}) : null;

    switch (key) {
      case 'low_cut_freq': {
        const apply = (v) => {
          const idx = Math.round(Number(v) || 0);
          lowCutIdx = idx;
          const f = _indexedLookup(N1073_LOW_CUT_FREQS, idx);
          // index 0 = OFF → push HPF below audio band
          hpf.frequency.value = f > 0 ? f : 10;
        };
        if (isMod) targets[paramId] = { paramDef: def, customSetter: apply };
        else if (val !== undefined) apply(val);
        break;
      }
      case 'low_shelf_freq': {
        const apply = (v) => {
          const idx = Math.round(Number(v) || 0);
          lowShelf.frequency.value = _indexedLookup(N1073_LOW_SHELF_FREQS, idx);
        };
        if (isMod) targets[paramId] = { paramDef: def, customSetter: apply };
        else if (val !== undefined) apply(val);
        break;
      }
      case 'low_shelf_gain': {
        if (isMod) {
          targets[paramId] = {
            paramDef: def,
            customSetter: (v) => { lowShelf.gain.value = Math.max(-18, Math.min(18, v)); },
          };
        } else if (typeof val === 'number') {
          lowShelf.gain.value = Math.max(-18, Math.min(18, val));
        }
        break;
      }
      case 'mid_freq': {
        const apply = (v) => {
          const idx = Math.round(Number(v) || 0);
          midBell.frequency.value = _indexedLookup(N1073_MID_FREQS, idx);
        };
        if (isMod) targets[paramId] = { paramDef: def, customSetter: apply };
        else if (val !== undefined) apply(val);
        break;
      }
      case 'mid_gain': {
        if (isMod) {
          targets[paramId] = {
            paramDef: def,
            customSetter: (v) => {
              const g = Math.max(-18, Math.min(18, v));
              midBell.gain.value = g;
              midBell.Q.value   = n1073MidQ(g, midQPreset);
            },
          };
        } else if (typeof val === 'number') {
          midBell.gain.value = Math.max(-18, Math.min(18, val));
          midBell.Q.value = n1073MidQ(midBell.gain.value, midQPreset);
        }
        break;
      }
      case 'mid_q': {
        // 3-bin: 0..0.33 = broad, 0.33..0.66 = medium, 0.66..1 = narrow
        const apply = (v) => {
          if (typeof v === 'string') {
            midQPreset = (v in N1073_MID_Q_PRESETS) ? v : 'medium';
          } else {
            const n = Number(v) || 0;
            midQPreset = n < 0.34 ? 'broad' : n < 0.67 ? 'medium' : 'narrow';
          }
          // Re-apply with the existing gain
          midBell.Q.value = n1073MidQ(midBell.gain.value, midQPreset);
        };
        if (isMod) targets[paramId] = { paramDef: def, customSetter: apply };
        else if (val !== undefined) apply(val);
        break;
      }
      case 'high_shelf_gain': {
        if (isMod) {
          targets[paramId] = {
            paramDef: def,
            customSetter: (v) => { highShelf.gain.value = Math.max(-18, Math.min(18, v)); },
          };
        } else if (typeof val === 'number') {
          highShelf.gain.value = Math.max(-18, Math.min(18, val));
        }
        break;
      }
      case 'inductor_saturation': {
        const ap = wpar('ind_drive');
        const apply = (v) => {
          const drive = Math.max(0, Math.min(1, v));
          inductor.curve = makeInductorCurve(drive);
          if (ap) ap.value = drive;
        };
        if (isMod) targets[paramId] = { paramDef: def, customSetter: apply };
        else if (val !== undefined) apply(val);
        break;
      }
      case 'output_gain': {
        if (isMod) {
          targets[paramId] = { audioParam: outGain.gain, paramDef: def };
        } else if (typeof val === 'number') {
          outGain.gain.value = val;
        }
        break;
      }
      default:
        break;
    }
  }

  return { input, output, paramTargets: targets };
}

// ── Build: vintage_api ────────────────────────────────────────────────────
//
// Topology: 4 peaking BiquadFilterNodes serially, each with its own freq/gain
// and proportional-Q remap. Then inductor saturation, then output gain.
//
// Param map (4 bands × {freq, gain} + global):
//   band1_freq, band1_gain, band2_freq, band2_gain, band3_freq, band3_gain, band4_freq, band4_gain,
//   inductor_saturation, output_gain
//
// `bandN_freq` is an index into `API_FREQ_TABLE` (0..11).
// `bandN_gain` is dB in the range [-12, +12]. Q is computed automatically.
//
// Default frequency assignments (low/lo-mid/hi-mid/high) are 100/400/3000/12500 Hz —
// a sensible 4-band split into the API table. Calibration can override these
// via the dspGraph `params` block.
export function buildVintageAPI(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const input  = ctx.createGain();
  const output = ctx.createGain();

  // 4 peaking biquads
  const bands = [];
  const defaultFreqIdx = [1, 3, 6, 9]; // 100, 400, 3000, 12500 Hz
  for (let i = 0; i < 4; i++) {
    const bq = ctx.createBiquadFilter();
    bq.type = 'peaking';
    bq.frequency.value = API_FREQ_TABLE[defaultFreqIdx[i]];
    bq.Q.value = apiProportionalQ(0);
    bq.gain.value = 0;
    bands.push(bq);
  }

  const inductor = ctx.createWaveShaper();
  inductor.curve = makeInductorCurve(0);
  inductor.oversample = '2x';

  const outGain = ctx.createGain();
  outGain.gain.value = 1;

  const worklet = _safeWorklet(ctx, R13_VINTAGE_API, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    parameterData: { ind_drive: 0 },
  });

  // Wire the chain: input → b1 → b2 → b3 → b4 → [worklet?] → inductor → outGain → output
  input.connect(bands[0]);
  for (let i = 0; i < 3; i++) bands[i].connect(bands[i + 1]);
  let post = bands[3];
  if (worklet) {
    bands[3].connect(worklet);
    post = worklet;
  }
  post.connect(inductor);
  inductor.connect(outGain);
  outGain.connect(output);

  const wpar = (name) => (worklet && worklet.parameters)
    ? (worklet.parameters.get(name) || null)
    : null;

  // Helpers to update one band — gain change must also recompute Q.
  const setBandGain = (idx, dB) => {
    const g = Math.max(-12, Math.min(12, dB));
    bands[idx].gain.value = g;
    bands[idx].Q.value    = apiProportionalQ(g);
  };
  const setBandFreq = (idx, freqIdx) => {
    bands[idx].frequency.value = _indexedLookup(API_FREQ_TABLE, freqIdx);
  };

  for (const [key, val] of Object.entries(params)) {
    const isMod = _isModulated(val);
    const paramId = isMod ? val.slice(1) : null;
    const def = isMod ? (paramDefs[paramId] || {}) : null;

    // Match `bandN_freq` / `bandN_gain` (N = 1..4)
    const m = /^band([1-4])_(freq|gain)$/.exec(key);
    if (m) {
      const idx = parseInt(m[1], 10) - 1;
      const which = m[2];
      if (which === 'freq') {
        const apply = (v) => setBandFreq(idx, v);
        if (isMod) targets[paramId] = { paramDef: def, customSetter: apply };
        else if (val !== undefined) apply(val);
      } else {
        const apply = (v) => setBandGain(idx, v);
        if (isMod) targets[paramId] = { paramDef: def, customSetter: apply };
        else if (typeof val === 'number') apply(val);
      }
      continue;
    }

    switch (key) {
      case 'inductor_saturation': {
        const ap = wpar('ind_drive');
        const apply = (v) => {
          const drive = Math.max(0, Math.min(1, v));
          inductor.curve = makeInductorCurve(drive);
          if (ap) ap.value = drive;
        };
        if (isMod) targets[paramId] = { paramDef: def, customSetter: apply };
        else if (val !== undefined) apply(val);
        break;
      }
      case 'output_gain': {
        if (isMod) {
          targets[paramId] = { audioParam: outGain.gain, paramDef: def };
        } else if (typeof val === 'number') {
          outGain.gain.value = val;
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
const R13_BUILDERS = {
  vintage_1073: buildVintage1073,
  vintage_api:  buildVintageAPI,
};

export default R13_BUILDERS;

// Named exports for tables (handy for tests & calibration)
export const _internals = {
  N1073_LOW_CUT_FREQS,
  N1073_LOW_SHELF_FREQS,
  N1073_MID_FREQS,
  N1073_HIGH_SHELF_HZ,
  N1073_MID_Q_PRESETS,
  API_FREQ_TABLE,
  n1073MidQ,
  apiProportionalQ,
  makeInductorCurve,
};
