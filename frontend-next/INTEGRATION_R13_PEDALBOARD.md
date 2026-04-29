# R13 — Pedalboard composite builder

**Author:** Agent R13
**Date:** 2026-04-25
**Status:** shipped
**Roadmap row:** Tier 2 / Pedalboard

---

## Goal

Logic Pro's **Pedalboard** is a chainable stompbox container with 24+ pedal
models, drag-to-reorder cable routing, and per-pedal bypass. The TIER 2
parity work is **not** to re-author every pedal's DSP from scratch — most of
the 24 stomp models map cleanly onto an existing R-round node, calibrated
once and re-used everywhere. R13's contribution is the *composite* builder
that takes a `pedals: [...]` config and chains the appropriate sub-builders
serially with bypass + level staging.

---

## Files

| File | Role |
|---|---|
| `src/audio/builders/r13_pedalboard.js` | Composite builder + 24 sub-pedal builders. Registers `pedalboard` in `NODE_BUILDERS`. |
| `src/audio/WebAudioDSPEngine.js` | Imports `r13Builders`, spreads into `NODE_BUILDERS`. |
| `tests/r13_pedalboard.test.js` | Smoke + chain-construction (5+ pedals) tests. |
| `tools/calibration/configs/pedalboard-web.json` (desktop) | Topology + per-sub-type schema for the calibration auto-driver. |

---

## Node type registered

Single new node type: **`pedalboard`**.

```js
{
  type: 'pedalboard',
  params: {
    pedals: [
      { type: 'overdrive_pedal', drive: 0.6, tone: 0.5, level: 0.7 },
      { type: 'distortion_pedal', drive: 0.8, tone: 0.4, bypass: false },
      { type: 'delay_pedal', time: 250, feedback: 0.3, mix: 0.4 },
    ],
    mix: 1.0,  // optional global wet/dry around the whole chain
  }
}
```

Returned shape: `{ input, output, paramTargets, pedals: [...] }` — same
contract as every other R-round builder, plus a `pedals` array of per-slot
`{ type, slot, status, input, output, wetGain, dryGain, join }` so callers
(calibration harness, dev panels) can introspect.

### Pedal sub-types (24)

| Sub-type | Underlying node(s) | Params |
|---|---|---|
| `overdrive_pedal` | R2 `wdf_tube_triode` + tone tilt biquads | drive, tone, level |
| `distortion_pedal` | R2 `wdf_diode_clipper` + tone tilt | drive, tone, level |
| `fuzz_pedal` | R2 `wdf_transistor_clipper` + tone tilt | drive, tone, fuzz, level |
| `clean_boost_pedal` | native `GainNode` | gain (or level) |
| `compressor_pedal` | native `DynamicsCompressorNode` + makeup gain | threshold, ratio, attack, release, level |
| `eq_pedal` | 3 stacked biquads (lowshelf/peaking/highshelf) | bass, mid, treble |
| `wah_pedal` | bandpass biquad swept by `position` | position, q |
| `auto_wah_pedal` | bandpass + R1 `envelope_follower` modulating cutoff | sensitivity, attack, release, q |
| `phaser_pedal` | 4× allpass biquads + LFO | rate, depth, mix |
| `flanger_pedal` | short DelayNode + LFO + feedback | rate, depth, feedback, mix |
| `chorus_pedal` | DelayNode + LFO (single voice) | rate, depth, mix |
| `tremolo_pedal` | GainNode amplitude-modulated by LFO | rate, depth |
| `vibrato_pedal` | DelayNode + LFO, **no dry signal** | rate, depth |
| `delay_pedal` | R1 `multitap_delay` (single tap) | time, feedback, mix |
| `tape_delay_pedal` | R1 `multitap_delay` → R3 `wdf_tape_sat` | time, feedback, saturation, mix |
| `reverb_pedal` | R9 `algo_reverb` (room algo by default) | algorithm, decay, mix |
| `octave_pedal` | R1 `pitch_shift` semitones=−12 | semitones, mix |
| `pitch_shifter_pedal` | R5 `pitch_shift_pv` (falls back to R1) | semitones, mix |
| `ring_mod_pedal` | input × sine carrier + dry/wet | frequency, mix |
| `filter_pedal` | switchable LP/HP/BP biquad | mode, cutoff, resonance |
| `limiter_pedal` | DynamicsCompressor pinned high-ratio | ceiling, release |
| `gate_pedal` | env follower → WaveShaper threshold → VCA | threshold, release |
| `noise_gate_pedal` | same as gate, lower default threshold | threshold, release |
| `bitcrusher_pedal` | R1 `bitcrusher` | bits, downsample, mix |

12 of these are direct delegations (`_delegate(ctx, rNBuilders, …)`). The
remaining 12 use native primitives or compose two existing nodes — none
re-implement DSP that an existing R-round already provides.

---

## Engine wiring

Two lines in `src/audio/WebAudioDSPEngine.js`:

```js
import r13Builders from './builders/r13_pedalboard.js';
// …
const NODE_BUILDERS = {
  // …
  ...r13Builders,   // pedalboard (24 stomp models composed of R1/R2/R3/R5/R9 sub-stages)
};
```

That's it. `pedalboard` is now a first-class node type for any `dspGraph`,
`dspChain`, or plugin mapping JSON.

---

## Bypass & live control

Each pedal slot is wrapped in an equal-power crossfade pair:

```
prevOut ─┬─→ built.input ──→ built.output ──→ wetGain ─┐
         └─────────────────────────────────→ dryGain ──┴─→ join → next
```

`bypass: true` in the config sets `wetGain=0, dryGain=1` at construction.
Live toggle is exposed as `paramTargets['bypass_<i>']` with a `customSetter`
that ramps the two gains via `setTargetAtTime` (no clicks).

Per-pedal sub-targets surface at `paramTargets['pedal_<i>_<sub>']` —
e.g. `pedal_0_drive`, `pedal_2_mix`. The calibration harness translates
Logic's dotted address (`pedals[2].mix`) to the underscored form before
calling `engine.setParameter`.

---

## Calibration strategy

Pedalboard is a *container*, so the auto-driver harness calibrates per
pedal **sub-type**, not per Pedalboard instance. The plan in
`tools/calibration/configs/pedalboard-web.json`:

1. For each sub-type, instantiate Pedalboard with one pedal of that type
   and all-defaults elsewhere.
2. Sweep that pedal's params at 11 steps each.
3. Fit curves once → re-use mapping for every chain position.

Chain-order matters for non-linear pedals (drive→delay vs delay→drive),
so the harness also runs A/B order tests for distortion+modulation pairs
and asserts null-diff ≤ −40 dB across moderate gain levels.

---

## Tests

`tests/r13_pedalboard.test.js` covers:

- Empty pedalboard returns input→output passthrough.
- Single-pedal chain builds + connects without throw.
- 5+ pedal serial chain (all 24 types in the kitchen-sink test).
- Bypass crossfade ramps wet/dry correctly.
- Unknown pedal type fails open (passthrough, not crash).
- `pedals[]` introspection lists slot status correctly.

---

## Limits / known gaps

- **No M/S routing.** Logic's Pedalboard supports A/B parallel split via
  the cable view. R13 is serial-only. Parallel routing belongs in a
  follow-up node (`pedal_split` perhaps) and is rare in practice.
- **No tempo-sync delay.** `delay_pedal.time` is in ms only. Tempo-sync
  needs a tempo input; trivial to add via an extra param once the engine
  exposes `playbackTempo` as a control source.
- **Cabinet block.** Logic includes amp + cabinet stages inside Pedalboard
  in some presets. Those map to R4 `circuit_*` composites — the user can
  add an `amp_designer` node downstream of the pedalboard slot, but a
  single combined node is left for the Amp Designer roadmap row.
- **Pitch-shifter latency.** R5 phase-vocoder has ~50 ms latency; R1 SOLA
  has ~10 ms. Use the R1 fallback in real-time chains (`pedal_priority:
  'low_latency'` is a future toggle).
- **`gate_pedal` accuracy.** The WaveShaper-based threshold-detect is
  cheap but coarse. A dedicated `r1-gate-processor` worklet would be
  more accurate; deferred until the gate's null-diff regresses.

---

## Acceptance

- ✅ `pedalboard` resolves in `NODE_BUILDERS`.
- ✅ All 24 sub-types build without throw.
- ✅ Smoke test green (offline render, non-zero RMS).
- ⏳ Per-sub-type calibration mappings — 24 mappings to author by the
  auto-driver harness.
- ⏳ R12 golden tests — one per representative sub-type.
