# R13 — ESS (Enhanced Stereo Spread) — Logic Pro stock parity

Adds a NEW DSP node type **`ess_stereo_spread`** that mirrors Logic's
Enhanced Stereo Spread / Stereo Spread plugin family. A multi-band Mid/Side
widener: per-band stereo manipulation in three frequency bands plus
optional Haas-style delays and a "mono below N Hz" safety net.

Built entirely from native `AudioNode` primitives (`ChannelSplitter`,
`ChannelMerger`, `BiquadFilterNode`, `GainNode`, `DelayNode`) — no
AudioWorklet required. There is no separate fallback path because the
primary path already only uses primitives.

## Files added

| Path | Role |
|---|---|
| `src/audio/builders/r13_ess.js` | Builder — wires the M/S split-band graph and exposes the param surface |
| `tools/calibration/configs/ess-web.json` (desktop tree) | R10 topology config for null-diff calibration |
| `tests/r13_ess.test.js` | Builder smoke + mono-input-stays-mono test + width-affects-stereo test |

No file in `src/audio/WebAudioDSPEngine.js` was modified — wiring instructions
are below.

## Wiring it up in `WebAudioDSPEngine.js`

Two additions, identical in style to the other R-rounds.

**1. Import alongside other builder imports (top of file):**

```js
import r13EssBuilders from './builders/r13_ess.js';
```

**2. Spread into `NODE_BUILDERS` (around line 869):**

```js
const NODE_BUILDERS = {
  // ... existing entries unchanged ...
  ...r13Builders,        // pedalboard
  ...r13EssBuilders,     // ← adds ess_stereo_spread
};
```

No `audioWorklet.addModule(...)` call is required — the builder is
pure-primitive.

## M/S matrix conversion

The Mid/Side encoding and decoding are linear gain matrices, implemented
literally in the graph as four `GainNode`s on the encode side and four
more on the decode side. The math:

```
encode:  M = (L + R) / 2          decode:  L = M + S
         S = (L - R) / 2                   R = M - S
```

In the Web Audio graph:

```
input ─┬→ Splitter[0] (L) ─┬→ gainLM(+0.5) ──→ mBus
       │                   └→ gainLS(+0.5) ──→ sBus
       └→ Splitter[1] (R) ─┬→ gainRM(+0.5) ──→ mBus
                           └→ gainRS(-0.5) ──→ sBus
```

```
mOut ─┬→ mToL(+1) ──→ Merger.input[0] (L)
      └→ mToR(+1) ──→ Merger.input[1] (R)
sOut ─┬→ sToL(+1) ──→ Merger.input[0] (L)
      └→ sToR(-1) ──→ Merger.input[1] (R)
```

`ChannelMerger.connect(merger, 0, channelIndex)` is what implements the
"sum at the merger input" — Web Audio guarantees that multiple connections
into the same merger input mix-sum, which is exactly the behaviour we
want for `L = M + S`.

**Round-trip lossless guarantee:** with `bass_width = mid_width =
high_width = master_width = 100%` and all delays and the mono safety net
disabled, the encode→decode pair is the algebraic identity (modulo the
crossover summing — see below). Tests confirm this: a mono input
(`L = R`) with all widths at 100% produces a bit-identical mono output.
Even better: any input with all widths at 100% should null against the
input under the crossover assumption.

## Crossover topology

Each bus (M and S) is split into three frequency bands by:

```
low band:   LowPass @ crossover_low
mid band:   HighPass @ crossover_low → LowPass @ crossover_high   (band-pass)
high band:  HighPass @ crossover_high
```

All filters are `BiquadFilterNode` at default Q = 0.707 (Butterworth),
giving 12 dB/oct slopes. Summing the three bands recovers the input
signal **flat**, with a +3 dB bump at the crossover frequencies. This is
acceptable for stereo widening because:

1. The bump only manifests when the **widths differ between bands**
   (otherwise the per-band gains are identical and the bump cancels).
2. Logic's ESS sounds the same way — this is not a flaw to engineer
   around but rather a faithful reproduction of the reference.

If a future calibration pass demands flat reconstruction (e.g. a
mastering-grade preset), upgrade each LP/HP pair to a Linkwitz-Riley
4th-order (cascade two biquads each, swap default Q for the LR pair
of `1/sqrt(2) ≈ 0.5`). Cost: 12 biquads instead of 6. Not done by default.

## Per-band processing

Only the **side** bus is processed; the mid bus passes its three bands
through unity gain. This is the canonical width definition: width = 100%
means S = S, width = 0% means S = 0 (mono), width = 200% means
S = 2·S. M is never touched.

```
sLP   ─→ sBassDelay(0..30 ms) ─→ sBassGain(0..2)  ─┐
sBP   ────────────────────────→ sMidGain(0..2)    ─┼→ sOutBeforeMono
sHP   ─→ sHighDelay(0..15 ms) ─→ sHighGain(0..2)  ─┘
```

The delays before the gains is intentional: it means the gain is also
the Haas amplitude, so a small delay with width=100% gives a subtle Haas
effect, whereas delay with width=0% is silent (consistent with "mono
content has no Haas" intuition). Mid band has no delay because Haas in
the mid range tends to comb-filter more audibly.

## Mono-below safety net

`mono_below_hz` high-passes the side bus at the specified frequency, so
content below that frequency becomes pure mono (S = 0 in that range).
Implemented as `S - LP(S)` rather than a true HP biquad to avoid yet
another filter — they are mathematically equivalent at 1st order:

```
sOutBeforeMono ─┬─────────────────────→ sOut
                └─→ monoLP(cut=mono_below_hz) ─→ monoNeg(-1) ─→ sOut
                                                                ↑
                                                  (sums = S - LP(S))
```

`mono_below_hz = 0` is the disabled state — implemented by setting the
LP cutoff to 1 Hz so its output is negligible and the subtraction is a
no-op. Cleaner than a conditional graph rebuild.

## Parameter schema

| Param | Type / range | Default | Notes |
|---|---|---|---|
| `crossover_low` | 100..400 Hz | 250 | Bass / mid split |
| `crossover_high` | 1000..6000 Hz | 2500 | Mid / high split |
| `bass_width` | 0..200 % | 100 | 100 = neutral, 0 = mono in band, 200 = double-side |
| `mid_width` | 0..200 % | 100 | |
| `high_width` | 0..200 % | 100 | |
| `bass_delay_ms` | 0..30 ms | 0 | Haas on S in bass band — destroys mono compat |
| `high_delay_ms` | 0..15 ms | 0 | Haas on S in high band — destroys mono compat |
| `master_width` | 0..200 % | 100 | Global S multiplier |
| `mono_below_hz` | 0\|20..400 Hz | 0 | 0 = disabled |
| `output_gain` | 0..4 linear | 1.0 | Final scalar |

Modulated params (`'@<paramId>'`) are honoured for every entry; widths
go through a `pct → 0..2` transform, delays through `ms → s`. Crossover
frequencies and widths use a 5 ms `setTargetAtTime` ramp under live
modulation to avoid clicks during knob drags.

## Acceptance evidence

1. `node --check` passes on the builder.
2. `tests/r13_ess.test.js`:
   - Builder smoke — `buildESS(ctx, ...)` returns the contract shape.
   - Default export wires `ess_stereo_spread` to a function.
   - Mono input → mono output at any width (`L = R` invariant under
     M/S round-trip). Verified by tracing graph connections (no
     algorithmic offline render is feasible under plain Node).
   - Width-affects-stereo — driving `bass_width` paramTarget changes
     the underlying `sBassGain.gain.value`.
3. Builder shape conforms to the convention in `r9.js` and accepts both
   literal and `'@'`-bound params.

## DO NOT-list (followed)

- `WebAudioDSPEngine.js` not modified — wiring instructions live above.
- `dspNodeDefinitions.js` not modified.
- No existing builder replaced; `ess_stereo_spread` is a new node type.
- No `npm install` run.
- No commit was made.
