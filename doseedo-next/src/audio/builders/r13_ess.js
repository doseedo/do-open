/**
 * R13 — ESS (Enhanced Stereo Spread) — Logic Pro stock parity
 *
 * Registers `ess_stereo_spread` as a NEW node type implementing Logic's
 * Enhanced Stereo Spread / Stereo Spread family: a multi-band M/S processor
 * that can widen or narrow the stereo image independently in three frequency
 * bands, with optional Haas-style delays on the side signal and a "mono
 * below N Hz" safety net.
 *
 * Pipeline (all done with Web Audio primitives — no worklet required):
 *
 *   1. ChannelSplitter(2)                 — break L/R input
 *   2. Gain matrix → M-bus, S-bus         — M = 0.5·L + 0.5·R,  S = 0.5·L − 0.5·R
 *   3. Per-bus 3-band crossover           — LowPass / BandPass / HighPass biquads
 *      The M bands sum back to a single M-out (we don't re-gain the M bands
 *      because S-only multiplication is the canonical "width" definition).
 *      The S bands each go through:
 *         (a) per-band width gain (0..2)
 *         (b) optional per-band Haas delay (DelayNode)
 *      then sum to S-out.
 *   4. Optional `mono_below_hz` safety: a low-pass copy of S → a NEGATIVE
 *      gain summed back into S, cancelling the side signal under that freq.
 *      Effectively a high-pass on the side bus.
 *   5. master_width                       — single multiplier on S-out
 *   6. Re-encode L/R using a 2-input ChannelMerger + gain matrix:
 *         L = M + S
 *         R = M − S
 *      (Implemented as: M-out → both merger.0 and merger.1 with gain 1;
 *       S-out → merger.0 with gain +1 and merger.1 with gain −1.)
 *   7. output_gain                        — final scalar
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildESS(ctx, nodeDef, paramDefs) → { input, output, paramTargets }
 *
 * Param values may be literals or '@<paramId>' bindings (live-modulated).
 * Width params are 0..200 (% — 100=neutral); we convert to a 0..2 multiplier
 * internally. The builder is built entirely from native AudioNodes so there
 * is no separate worklet path; the "fallback" is the same code path.
 *
 * @author Doseedo R13
 */

// ── Defaults ──────────────────────────────────────────────────────────────
const DEFAULTS = {
  crossover_low:    250,    // Hz — bass / mid split
  crossover_high:   2500,   // Hz — mid / high split
  bass_width:       100,    // % — 100 = neutral
  mid_width:        100,
  high_width:       100,
  bass_delay_ms:    0,      // 0..30 ms Haas on S in bass band
  high_delay_ms:    0,      // 0..15 ms Haas on S in high band
  master_width:     100,    // %
  mono_below_hz:    0,      // 0 = disabled
  output_gain:      1.0,    // linear
};

const MAX_BASS_DELAY_S  = 0.030;
const MAX_HIGH_DELAY_S  = 0.015;

function _isModulated(v) {
  return typeof v === 'string' && v.startsWith('@');
}

function _clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function _pctToMul(pct) {
  // 0..200% → 0..2 linear multiplier
  return _clamp(pct / 100, 0, 2);
}

/**
 * buildESS — instantiate the multi-band M/S widener graph.
 *
 * Schema:
 *   {
 *     type: 'ess_stereo_spread',
 *     params: {
 *       crossover_low:   100..400,    // Hz
 *       crossover_high:  1000..6000,  // Hz
 *       bass_width:      0..200,      // %
 *       mid_width:       0..200,
 *       high_width:      0..200,
 *       bass_delay_ms:   0..30,
 *       high_delay_ms:   0..15,
 *       master_width:    0..200,
 *       mono_below_hz:   0|20..400,   // 0 disables
 *       output_gain:     linear,
 *     }
 *   }
 */
export function buildESS(ctx, nodeDef, paramDefs) {
  const params = (nodeDef && nodeDef.params) || {};
  const targets = {};

  // ── Outer I/O ─────────────────────────────────────────────────────────
  // Input is a stereo gain so upstream connections are simple. We accept
  // a 2-channel input regardless of upstream channel count (the splitter
  // below enforces it).
  const input  = ctx.createGain();
  const output = ctx.createGain();

  // ── 1. Split L / R ────────────────────────────────────────────────────
  const splitter = ctx.createChannelSplitter(2);
  // channelCountMode 'explicit' + channelCount=2 makes the splitter respect
  // the L/R layout regardless of any upmix/downmix in front.
  input.channelCount = 2;
  input.channelCountMode = 'explicit';
  input.channelInterpretation = 'speakers';
  input.connect(splitter);

  // ── 2. M / S encode via gain matrix ───────────────────────────────────
  // M = 0.5·L + 0.5·R   →   gainLM (0.5) + gainRM (0.5) summed into mBus
  // S = 0.5·L − 0.5·R   →   gainLS (+0.5) + gainRS (−0.5) summed into sBus
  const gainLM = ctx.createGain(); gainLM.gain.value =  0.5;
  const gainRM = ctx.createGain(); gainRM.gain.value =  0.5;
  const gainLS = ctx.createGain(); gainLS.gain.value =  0.5;
  const gainRS = ctx.createGain(); gainRS.gain.value = -0.5;

  const mBus = ctx.createGain();
  const sBus = ctx.createGain();

  // splitter.connect(destination, outputIndex)
  splitter.connect(gainLM, 0);
  splitter.connect(gainRM, 1);
  splitter.connect(gainLS, 0);
  splitter.connect(gainRS, 1);

  gainLM.connect(mBus);
  gainRM.connect(mBus);
  gainLS.connect(sBus);
  gainRS.connect(sBus);

  // ── 3. Per-bus 3-band crossover ───────────────────────────────────────
  // For each bus we build:
  //   low band:  LP @ crossover_low
  //   mid band:  HP @ crossover_low → LP @ crossover_high   (band-pass)
  //   high band: HP @ crossover_high
  //
  // We use BiquadFilterNode at default Q (Butterworth, 0.707) — these
  // crossover filters are linear, so summing the bands recovers the
  // original signal flat (within ~3 dB at the crossover points — same
  // approximation Logic uses for its split-band M/S processors).

  const initLow  = (typeof params.crossover_low  === 'number') ? params.crossover_low  : DEFAULTS.crossover_low;
  const initHigh = (typeof params.crossover_high === 'number') ? params.crossover_high : DEFAULTS.crossover_high;

  // M-bus crossover: low / mid / high
  const mLP = ctx.createBiquadFilter();
  mLP.type = 'lowpass';
  mLP.frequency.value = initLow;

  const mBP_hp = ctx.createBiquadFilter();
  mBP_hp.type = 'highpass';
  mBP_hp.frequency.value = initLow;

  const mBP_lp = ctx.createBiquadFilter();
  mBP_lp.type = 'lowpass';
  mBP_lp.frequency.value = initHigh;

  const mHP = ctx.createBiquadFilter();
  mHP.type = 'highpass';
  mHP.frequency.value = initHigh;

  mBus.connect(mLP);
  mBus.connect(mBP_hp); mBP_hp.connect(mBP_lp);
  mBus.connect(mHP);

  // M bands all have unity gain (M is unaffected by width — only S is
  // scaled). Sum bands directly into mOut.
  const mOut = ctx.createGain();
  mLP.connect(mOut);
  mBP_lp.connect(mOut);
  mHP.connect(mOut);

  // S-bus crossover: low / mid / high
  const sLP = ctx.createBiquadFilter();
  sLP.type = 'lowpass';
  sLP.frequency.value = initLow;

  const sBP_hp = ctx.createBiquadFilter();
  sBP_hp.type = 'highpass';
  sBP_hp.frequency.value = initLow;

  const sBP_lp = ctx.createBiquadFilter();
  sBP_lp.type = 'lowpass';
  sBP_lp.frequency.value = initHigh;

  const sHP = ctx.createBiquadFilter();
  sHP.type = 'highpass';
  sHP.frequency.value = initHigh;

  sBus.connect(sLP);
  sBus.connect(sBP_hp); sBP_hp.connect(sBP_lp);
  sBus.connect(sHP);

  // ── Per-band S width gains + optional Haas delays ─────────────────────
  const sBassDelay = ctx.createDelay(MAX_BASS_DELAY_S);
  sBassDelay.delayTime.value = (typeof params.bass_delay_ms === 'number')
    ? _clamp(params.bass_delay_ms, 0, 30) / 1000
    : DEFAULTS.bass_delay_ms / 1000;

  const sHighDelay = ctx.createDelay(MAX_HIGH_DELAY_S);
  sHighDelay.delayTime.value = (typeof params.high_delay_ms === 'number')
    ? _clamp(params.high_delay_ms, 0, 15) / 1000
    : DEFAULTS.high_delay_ms / 1000;

  const sBassGain = ctx.createGain();
  sBassGain.gain.value = (typeof params.bass_width === 'number')
    ? _pctToMul(params.bass_width) : _pctToMul(DEFAULTS.bass_width);

  const sMidGain = ctx.createGain();
  sMidGain.gain.value = (typeof params.mid_width === 'number')
    ? _pctToMul(params.mid_width) : _pctToMul(DEFAULTS.mid_width);

  const sHighGain = ctx.createGain();
  sHighGain.gain.value = (typeof params.high_width === 'number')
    ? _pctToMul(params.high_width) : _pctToMul(DEFAULTS.high_width);

  // S band routing: filter → delay → gain → sOut
  sLP.connect(sBassDelay); sBassDelay.connect(sBassGain);
  sBP_lp.connect(sMidGain);                              // mid has no Haas
  sHP.connect(sHighDelay); sHighDelay.connect(sHighGain);

  const sOutBeforeMono = ctx.createGain();
  sBassGain.connect(sOutBeforeMono);
  sMidGain.connect(sOutBeforeMono);
  sHighGain.connect(sOutBeforeMono);

  // ── 4. mono_below_hz safety net ───────────────────────────────────────
  // Equivalent to high-passing the side bus at mono_below_hz. We do it as
  // (S - LP(S)) which is mathematically identical to a 1st-order HP and
  // costs only a single biquad + a flip-sign gain.
  //
  // Disabled state (param=0): we keep the LP cutoff at a sane value (the
  // default 250 Hz) but zero out `monoNeg.gain` so the subtraction term
  // is silent and S passes through unchanged. This is cleaner than
  // setting the LP cutoff to 1 Hz (which still passes DC) and avoids
  // introducing an HP that's accidentally on at construction time.
  const monoLP = ctx.createBiquadFilter();
  monoLP.type = 'lowpass';
  const initMono = (typeof params.mono_below_hz === 'number') ? params.mono_below_hz : DEFAULTS.mono_below_hz;
  monoLP.frequency.value = initMono > 0 ? initMono : 250;

  const monoNeg = ctx.createGain();
  // Gain = -1 only when the safety net is enabled (param > 0).
  monoNeg.gain.value = initMono > 0 ? -1 : 0;

  const sOut = ctx.createGain();
  sOutBeforeMono.connect(sOut);                 // pass S through
  sOutBeforeMono.connect(monoLP);
  monoLP.connect(monoNeg);
  monoNeg.connect(sOut);                        // subtract LP(S) from S

  // ── 5. master_width on S ──────────────────────────────────────────────
  const masterS = ctx.createGain();
  masterS.gain.value = (typeof params.master_width === 'number')
    ? _pctToMul(params.master_width) : _pctToMul(DEFAULTS.master_width);
  sOut.connect(masterS);

  // ── 6. Re-encode L/R via ChannelMerger ────────────────────────────────
  // L = M + S → merger.0
  // R = M − S → merger.1
  const merger = ctx.createChannelMerger(2);

  const mToL = ctx.createGain(); mToL.gain.value = 1;
  const mToR = ctx.createGain(); mToR.gain.value = 1;
  const sToL = ctx.createGain(); sToL.gain.value =  1;
  const sToR = ctx.createGain(); sToR.gain.value = -1;

  mOut.connect(mToL);   mToL.connect(merger, 0, 0);
  mOut.connect(mToR);   mToR.connect(merger, 0, 1);
  masterS.connect(sToL); sToL.connect(merger, 0, 0);
  masterS.connect(sToR); sToR.connect(merger, 0, 1);

  // ── 7. output_gain → output ───────────────────────────────────────────
  const outG = ctx.createGain();
  outG.gain.value = (typeof params.output_gain === 'number') ? params.output_gain : DEFAULTS.output_gain;
  merger.connect(outG);
  outG.connect(output);

  // ── Param wiring ──────────────────────────────────────────────────────
  // Helper: bind a single AudioParam (or pair of AudioParams) to a
  // modulated paramId. `transform` rescales the input value before assignment.
  const bindAudioParam = (key, audioParam, transform) => {
    const val = params[key];
    if (val === undefined) return;
    if (_isModulated(val)) {
      const id = val.slice(1);
      targets[id] = {
        paramDef: paramDefs && paramDefs[id],
        customSetter: (v) => {
          const t = transform ? transform(v) : v;
          if (typeof audioParam.setTargetAtTime === 'function') {
            // Use a short ramp to avoid clicks under live param drag.
            try { audioParam.setTargetAtTime(t, ctx.currentTime || 0, 0.005); }
            catch (e) { audioParam.value = t; }
          } else {
            audioParam.value = t;
          }
        },
      };
    } else if (typeof val === 'number') {
      audioParam.value = transform ? transform(val) : val;
    }
  };

  // Crossover frequencies: each setter has to update BOTH the M-bus and
  // S-bus filter pair, so we use customSetter.
  if (_isModulated(params.crossover_low)) {
    const id = params.crossover_low.slice(1);
    targets[id] = {
      paramDef: paramDefs && paramDefs[id],
      customSetter: (v) => {
        const f = _clamp(v, 20, 1000);
        mLP.frequency.value = f;
        mBP_hp.frequency.value = f;
        sLP.frequency.value = f;
        sBP_hp.frequency.value = f;
      },
    };
  } else if (typeof params.crossover_low === 'number') {
    // Already applied at construction.
  }

  if (_isModulated(params.crossover_high)) {
    const id = params.crossover_high.slice(1);
    targets[id] = {
      paramDef: paramDefs && paramDefs[id],
      customSetter: (v) => {
        const f = _clamp(v, 200, 20000);
        mBP_lp.frequency.value = f;
        mHP.frequency.value = f;
        sBP_lp.frequency.value = f;
        sHP.frequency.value = f;
      },
    };
  }

  // Per-band widths: 0..200% → 0..2 multiplier on the S-band gain.
  bindAudioParam('bass_width',  sBassGain.gain, _pctToMul);
  bindAudioParam('mid_width',   sMidGain.gain,  _pctToMul);
  bindAudioParam('high_width',  sHighGain.gain, _pctToMul);
  bindAudioParam('master_width', masterS.gain,  _pctToMul);

  // Haas delays: ms → seconds, clamped to MaxDelay.
  bindAudioParam('bass_delay_ms', sBassDelay.delayTime,
    (v) => _clamp(v, 0, 30) / 1000);
  bindAudioParam('high_delay_ms', sHighDelay.delayTime,
    (v) => _clamp(v, 0, 15) / 1000);

  // mono_below_hz: 0 → disabled (zero out monoNeg so S - 0 = S).
  // Modulated path needs to update BOTH the LP cutoff AND the neg gain.
  if (_isModulated(params.mono_below_hz)) {
    const id = params.mono_below_hz.slice(1);
    targets[id] = {
      paramDef: paramDefs && paramDefs[id],
      customSetter: (v) => {
        if (v > 0) {
          monoLP.frequency.value = _clamp(v, 1, 20000);
          monoNeg.gain.value = -1;
        } else {
          // Keep the cutoff sane — only the gain disables the path.
          monoNeg.gain.value = 0;
        }
      },
    };
  } else if (typeof params.mono_below_hz === 'number') {
    // Already applied at construction.
  }

  // output_gain: linear scalar (could be dB if calibration prefers).
  bindAudioParam('output_gain', outG.gain);

  return {
    input,
    output,
    paramTargets: targets,
    // Expose internal nodes for tests / tooling that wants to introspect.
    _internal: {
      mBus, sBus, mOut, sOut, masterS,
      sBassGain, sMidGain, sHighGain,
      sBassDelay, sHighDelay,
      monoLP,
      mLP, mHP, mBP_hp, mBP_lp,
      sLP, sHP, sBP_hp, sBP_lp,
      splitter, merger,
      gainLM, gainRM, gainLS, gainRS,
      mToL, mToR, sToL, sToR,
      outG,
    },
  };
}

// ── Default export: NODE_BUILDERS map ─────────────────────────────────────
const R13_ESS_BUILDERS = {
  ess_stereo_spread: buildESS,
};

export default R13_ESS_BUILDERS;
