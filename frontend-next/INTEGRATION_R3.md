# R3 — WDF Passive / Saturation Primitives

Implements 5 Wave Digital Filter style nodes for the Doseedo web DSP graph
runtime so that Logic stock plugins (Tape Delay, Exciter, ChromaGlow) can be
graph-compiled.

## Files added

| Purpose | Path |
|---|---|
| Tape sat worklet | `src/lib/web-audio-plugins/worklets/r3-wdf-tape-sat-processor.js` |
| Transformer worklet | `src/lib/web-audio-plugins/worklets/r3-wdf-transformer-processor.js` |
| RC filter worklet | `src/lib/web-audio-plugins/worklets/r3-wdf-rc-filter-processor.js` |
| RLC filter worklet | `src/lib/web-audio-plugins/worklets/r3-wdf-rlc-filter-processor.js` |
| Power-sag worklet | `src/lib/web-audio-plugins/worklets/r3-wdf-power-supply-sag-processor.js` |
| Builders (native nodes) | `src/audio/builders/r3.js` |

## Node coverage

| Node type | Builder | Implementation strategy |
|---|---|---|
| `wdf_tape_sat` | `buildWdfTapeSat` | input gain → asymmetric soft-knee `WaveShaperNode` (4× OS) → low-shelf `BiquadFilter` (head bump) → speed-mapped LP `BiquadFilter` → 5 Hz HP DC blocker → wet/dry |
| `wdf_transformer` | `buildWdfTransformer` | drive gain → atan-curve `WaveShaperNode` (4× OS, blended with dry by `saturation`) → DC blocker |
| `wdf_rc_filter` | `buildWdfRCFilter` | `BiquadFilter(lowpass, Q=0.5)` with `frequency = 1/(2π·R·C)` |
| `wdf_rlc_filter` | `buildWdfRLCFilter` | `BiquadFilter(bandpass)` with `frequency = 1/(2π·√(LC))`, `Q = (1/R)·√(L/C)` |
| `wdf_power_supply_sag` | `buildWdfPowerSupplySag` | full-wave rectifier → asymmetric 1-pole envelope follower (5 ms attack, `recovery` s release) → gain reduction `1 - sag·env`, clamped → DC blocker. `ScriptProcessorNode(256)` for the sample loop |

## Wiring into `WebAudioDSPEngine`

`WebAudioDSPEngine.js` exposes `NODE_BUILDERS` near line 788. Merge in R3
without touching that file directly — the engine loader (or whatever wraps it
in this branch) should do something like:

```js
import R3_BUILDERS from '../builders/r3.js';
Object.assign(NODE_BUILDERS, R3_BUILDERS);
```

The builders return the standard `{ input, output, paramTargets }` shape that
the engine already understands. `paramTargets` contains:

- direct `audioParam` bindings for `input_level`, `drive`
- `customSetter` entries for `bias`, `speed`, `head_bump`, `mix`,
  `resistance`, `inductance`, `capacitance`, `saturation`, `sag`, `recovery`

so existing parameter-update code works unmodified.

## Parameter map

### `wdf_tape_sat`

| Param | Range | Effect |
|---|---|---|
| `input_level` | 0.5–5 | pre-saturator gain |
| `bias` | 0–1 | curve asymmetry (DC-corrected); 0.5 = symmetric, even harmonics scale with deviation |
| `speed` | 0–1 | LP cutoff 2.5 kHz (slow/warm) → 18 kHz (fast/bright) |
| `head_bump` | 0–1 | 0–8 dB low-shelf @ 110 Hz |
| `mix` | 0–1 | wet amount |

### `wdf_transformer`

| Param | Range | Effect |
|---|---|---|
| `drive` | 0.1–5 | pre-gain |
| `saturation` | 0.1–1 | atan curve weight vs. soft-clip pass-through |
| `mix` | 0–1 | wet |

### `wdf_rc_filter`

| Param | Range | Effect |
|---|---|---|
| `resistance` | 100–1 MΩ | with `capacitance`, sets cutoff `1/(2πRC)` |
| `capacitance` | 1 pF–10 µF | (clamped to Nyquist) |
| `mix` | 0–1 | wet |

### `wdf_rlc_filter`

| Param | Range | Effect |
|---|---|---|
| `resistance` | 10 Ω–100 kΩ | damping → `Q = (1/R)·√(L/C)` |
| `inductance` | 1 µH–1 H | with `capacitance` sets `f0 = 1/(2π√(LC))` |
| `capacitance` | 1 pF–100 µF | as above |
| `mix` | 0–1 | wet |

### `wdf_power_supply_sag`

| Param | Range | Effect |
|---|---|---|
| `sag` | 0–1 | max GR depth (`1 - sag` is the gain floor) |
| `recovery` | 10–500 ms | reservoir-cap recharge time constant |
| `mix` | 0–1 | wet |

## Worklet registration (optional, for higher fidelity)

The companion worklets (sample-accurate, no `ScriptProcessor` deprecation,
and a-rate parameters where it matters) can be registered ahead of engine
construction:

```js
const ctx = new AudioContext();
await Promise.all([
  ctx.audioWorklet.addModule('/worklets/r3-wdf-tape-sat-processor.js'),
  ctx.audioWorklet.addModule('/worklets/r3-wdf-transformer-processor.js'),
  ctx.audioWorklet.addModule('/worklets/r3-wdf-rc-filter-processor.js'),
  ctx.audioWorklet.addModule('/worklets/r3-wdf-rlc-filter-processor.js'),
  ctx.audioWorklet.addModule('/worklets/r3-wdf-power-supply-sag-processor.js'),
]);
```

The current builders default to native nodes for portability; switching
specific builders to AudioWorkletNodes is a follow-up if/when the engine
gains an `await ctx.audioWorklet.addModule(...)` pre-init step.

Processor names registered:

- `r3-wdf-tape-sat-processor`
- `r3-wdf-transformer-processor`
- `r3-wdf-rc-filter-processor`
- `r3-wdf-rlc-filter-processor`
- `r3-wdf-power-supply-sag-processor`

## Logic-plugin compile mapping (suggested)

| Logic plugin | Suggested node chain |
|---|---|
| Tape Delay | `delay → wdf_tape_sat (input_level≈1.5, bias≈0.55, speed by tape-speed knob, head_bump≈0.4)` |
| Exciter | `wdf_rc_filter (HP via inverted topology) → wdf_transformer (drive≈2, saturation≈0.7) → mix` |
| ChromaGlow | `wdf_transformer + wdf_tape_sat parallel → wdf_power_supply_sag (sag≈0.3, recovery≈80ms)` |

## Known limitations

- `wdf_transformer` is a v1 atan model — full Jiles-Atherton B-H hysteresis is
  a TODO. The 1-sample `_prev` term in the worklet introduces only a hint of
  hysteretic lag.
- `wdf_rc_filter` uses a `BiquadFilter` rather than a true 1st-order filter
  (Web Audio doesn't ship one). Below the cutoff the response is essentially
  flat as desired; far above, slope is 12 dB/oct rather than 6 dB/oct in the
  builder version. The standalone worklet is a true first-order bilinear LP.
- `wdf_power_supply_sag` builder uses `ScriptProcessorNode` (deprecated). The
  worklet replaces this when the host registers it. Latency is one 256-sample
  block (≈5.3 ms @ 48 kHz).
- Bias asymmetry curve in tape-sat is DC-corrected at the curve level; very
  hot signals can still pump the post-stage low-shelf — DC blocker compensates.
- `wdf_rlc_filter` implements a band-pass tap (across R). LP and HP taps
  (across L and C respectively) are not exposed; add new node types if those
  are needed for specific Logic-plugin compilation targets.

## DC handling

Every nonlinear stage (tape sat, transformer, sag) terminates in a 5 Hz
high-pass DC blocker (Q = 0.7071). The biased tape-sat curve also pre-removes
the static DC offset at curve-build time so we don't lean on the post-stage
HP for asymmetric bias removal.
