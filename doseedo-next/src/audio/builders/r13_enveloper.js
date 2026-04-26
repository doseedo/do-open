/**
 * R13 — Enveloper (transient shaper) builder
 *
 * Registers a NEW DSP node type `enveloper` that mirrors Logic Pro's stock
 * Enveloper plugin — a transient designer (think SPL Transient Designer /
 * Waves TransX). Two parallel envelope followers (a fast one tracking
 * transients, a slow one tracking the body) drive independent attack and
 * sustain gain curves applied to the dry signal.
 *
 * Algorithm (mirrors the worklet — see `r13-enveloper-processor.js`):
 *   1. Two parallel asymmetric one-pole envelope followers on |x|:
 *        - Fast: ~1 ms attack / ~10 ms release   (catches transients)
 *        - Slow: ~30 ms attack / ~300 ms release (tracks body)
 *   2. Attack envelope  = max(0, fast - slow)   — spikes during transients
 *      Sustain envelope = max(0, slow - 0.6*fast) — body between transients
 *      (Both auto-normalised by a slow-tracking peak so the gain math is
 *       independent of input level.)
 *   3. attack_gain  = 1 + attack_amount  * attack_env_norm
 *      sustain_gain = 1 + sustain_amount * sustain_env_norm
 *      total_gain   = attack_gain * sustain_gain
 *   4. y = mix * (x * total_gain * output_gain) + (1 - mix) * x
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildEnveloper(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: {...} } }
 *
 * Worklet path is preferred. If the processor isn't registered yet on the
 * AudioContext (worklet module not loaded — common during SSR/test or first
 * frame of a graph) we fall back to a primitive path built from a
 * DynamicsCompressor configured as a transient-emphasis stage plus a static
 * makeup gain. The fallback honours the same param surface so the engine
 * sees the same shape regardless of which path is alive — once the worklet
 * loads, the next graph build picks up the real DSP with no code change.
 *
 * Param surface:
 *   attack             -100..+100  (% — boost transients (+) or soften (-))
 *   sustain            -100..+100  (% — boost sustain or reduce)
 *   attack_time_ms     0.1..10     (fast follower attack)
 *   attack_release_ms  1..50       (fast follower release)
 *   sustain_time_ms    10..200     (slow follower attack)
 *   sustain_release_ms 100..2000   (slow follower release)
 *   output_gain        -12..+12    (dB)
 *   mix                0..1        (dry/wet)
 *
 * Author: Agent R13
 */

const R13_ENVELOPER = 'r13-enveloper-processor';

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

function _dbToGain(db) {
  return Math.pow(10, db / 20);
}

/**
 * Translate the Enveloper `attack` param (-100..+100 %) and `sustain` param
 * to a fallback compressor's threshold + ratio. The fallback is a coarse
 * approximation — the worklet is the real path. We just want graph
 * continuity + audible behaviour change as knobs move.
 *
 * Positive attack → fast attack/short release compressor with low threshold
 * (a transient-emphasis trick).  Negative attack → soft compression that
 * rounds peaks (longer attack).
 */
function _fallbackComp(ctx, attackPct, sustainPct) {
  const c = ctx.createDynamicsCompressor();
  // Default neutral
  c.threshold.value = -24;
  c.knee.value = 6;
  c.ratio.value = 2;
  c.attack.value = 0.003;
  c.release.value = 0.25;

  // Skew based on attack %: +100 ⇒ fast/aggressive, -100 ⇒ slow/round
  const a = Math.max(-100, Math.min(100, attackPct || 0)) / 100;
  c.attack.value  = Math.max(0.0005, 0.005 - a * 0.0045);  // 0.5 ms .. 9.5 ms
  c.release.value = Math.max(0.05,   0.25 - a * 0.20);     // 50 ms .. 450 ms

  const s = Math.max(-100, Math.min(100, sustainPct || 0)) / 100;
  // Negative sustain → tighten ratio (gates sustain). Positive sustain → expand
  // by lowering ratio (let sustain through harder). Bounded for safety.
  c.ratio.value = Math.max(1.5, 3 + s * -1.5);
  return c;
}

// ── enveloper builder ─────────────────────────────────────────────────────
export function buildEnveloper(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const input  = ctx.createGain();
  const output = ctx.createGain();

  // Initial param values from the node config (used to seed the worklet
  // OR the fallback path so first-block audio reflects user intent).
  const initial = {
    attack:             typeof params.attack             === 'number' ? params.attack             : 0,
    sustain:            typeof params.sustain            === 'number' ? params.sustain            : 0,
    attack_time_ms:     typeof params.attack_time_ms     === 'number' ? params.attack_time_ms     : 1.0,
    attack_release_ms:  typeof params.attack_release_ms  === 'number' ? params.attack_release_ms  : 10.0,
    sustain_time_ms:    typeof params.sustain_time_ms    === 'number' ? params.sustain_time_ms    : 30.0,
    sustain_release_ms: typeof params.sustain_release_ms === 'number' ? params.sustain_release_ms : 300.0,
    output_gain:        typeof params.output_gain        === 'number' ? params.output_gain        : 0,
    mix:                typeof params.mix                === 'number' ? params.mix                : 1.0,
  };

  const worklet = _safeWorklet(ctx, R13_ENVELOPER, {
    numberOfInputs:  1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    parameterData: {
      attack:             initial.attack,
      sustain:            initial.sustain,
      attack_time_ms:     initial.attack_time_ms,
      attack_release_ms:  initial.attack_release_ms,
      sustain_time_ms:    initial.sustain_time_ms,
      sustain_release_ms: initial.sustain_release_ms,
      output_gain:        initial.output_gain,
      mix:                initial.mix,
    },
  });

  // ── Fallback path ────────────────────────────────────────────────────────
  // input → [dry_path, wet_path(comp → makeup)] → output. Mix gain crossfades.
  let fbComp = null, fbMakeup = null, fbDry = null, fbWet = null;
  if (!worklet) {
    fbComp   = _fallbackComp(ctx, initial.attack, initial.sustain);
    fbMakeup = ctx.createGain();
    fbMakeup.gain.value = _dbToGain(initial.output_gain);
    fbDry = ctx.createGain();
    fbWet = ctx.createGain();
    const m = Math.max(0, Math.min(1, initial.mix));
    fbDry.gain.value = 1 - m;
    fbWet.gain.value = m;

    input.connect(fbDry);
    input.connect(fbComp);
    fbComp.connect(fbMakeup);
    fbMakeup.connect(fbWet);
    fbDry.connect(output);
    fbWet.connect(output);
  } else {
    input.connect(worklet);
    worklet.connect(output);
  }

  const wpar = (name) => (worklet && worklet.parameters)
    ? (worklet.parameters.get(name) || null)
    : null;

  // Helper for the simple "AudioParam if worklet, customSetter no-op otherwise"
  // wiring used by the timing params on the fallback path.
  function _bindWorkletParam(paramId, paramName, key) {
    const ap = wpar(paramName);
    if (ap) {
      targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
    } else {
      // Fallback: timing constants are baked into the comp at build time;
      // we accept the param so the engine binds, but no live re-tune.
      targets[paramId] = { paramDef: paramDefs[paramId], customSetter: () => {} };
    }
  }

  // ── Param wiring ─────────────────────────────────────────────────────────
  for (const [key, val] of Object.entries(params)) {
    const isModulated = _isModulated(val);
    const paramId = isModulated ? val.slice(1) : null;

    switch (key) {
      case 'attack': {
        const ap = wpar('attack');
        if (isModulated) {
          if (ap) {
            targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          } else {
            // Fallback: rebuild fbComp's attack/release on each tweak
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => {
                const a = Math.max(-100, Math.min(100, v)) / 100;
                if (fbComp) {
                  try { fbComp.attack.value  = Math.max(0.0005, 0.005 - a * 0.0045); } catch (e) {}
                  try { fbComp.release.value = Math.max(0.05,   0.25  - a * 0.20);   } catch (e) {}
                }
              },
            };
          }
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          // fallback already seeded at construction
        }
        break;
      }
      case 'sustain': {
        const ap = wpar('sustain');
        if (isModulated) {
          if (ap) {
            targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          } else {
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => {
                const s = Math.max(-100, Math.min(100, v)) / 100;
                if (fbComp) {
                  try { fbComp.ratio.value = Math.max(1.5, 3 + s * -1.5); } catch (e) {}
                }
              },
            };
          }
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
        }
        break;
      }
      case 'attack_time_ms':
        if (isModulated) _bindWorkletParam(paramId, 'attack_time_ms', key);
        else if (typeof val === 'number' && wpar('attack_time_ms')) wpar('attack_time_ms').value = val;
        break;
      case 'attack_release_ms':
        if (isModulated) _bindWorkletParam(paramId, 'attack_release_ms', key);
        else if (typeof val === 'number' && wpar('attack_release_ms')) wpar('attack_release_ms').value = val;
        break;
      case 'sustain_time_ms':
        if (isModulated) _bindWorkletParam(paramId, 'sustain_time_ms', key);
        else if (typeof val === 'number' && wpar('sustain_time_ms')) wpar('sustain_time_ms').value = val;
        break;
      case 'sustain_release_ms':
        if (isModulated) _bindWorkletParam(paramId, 'sustain_release_ms', key);
        else if (typeof val === 'number' && wpar('sustain_release_ms')) wpar('sustain_release_ms').value = val;
        break;
      case 'output_gain': {
        const ap = wpar('output_gain');
        if (isModulated) {
          if (ap) {
            targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          } else {
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => { if (fbMakeup) fbMakeup.gain.value = _dbToGain(v); },
            };
          }
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          else if (fbMakeup) fbMakeup.gain.value = _dbToGain(val);
        }
        break;
      }
      case 'mix': {
        const ap = wpar('mix');
        if (isModulated) {
          if (ap) {
            targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
          } else {
            targets[paramId] = {
              paramDef: paramDefs[paramId],
              customSetter: (v) => {
                const w = Math.max(0, Math.min(1, v));
                if (fbWet && fbDry) {
                  fbWet.gain.value = w;
                  fbDry.gain.value = 1 - w;
                }
              },
            };
          }
        } else if (typeof val === 'number') {
          if (ap) ap.value = val;
          else if (fbWet && fbDry) {
            const w = Math.max(0, Math.min(1, val));
            fbWet.gain.value = w;
            fbDry.gain.value = 1 - w;
          }
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
const R13_ENVELOPER_BUILDERS = {
  enveloper: buildEnveloper,
};

export default R13_ENVELOPER_BUILDERS;
