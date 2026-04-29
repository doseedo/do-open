# R13 — Adaptive Limiter

Adds a NEW DSP node type **`adaptive_limiter`** backed by a single AudioWorklet
processor implementing Logic's "transparent" mastering limiter with an
adaptive release. The existing `limiter` node is untouched.

## Files added

| Path | Role |
|---|---|
| `src/lib/web-audio-plugins/worklets/r13-adaptive-limiter-processor.js` | AudioWorklet — multi-stage limiter (pre-gain → lookahead → peak detect → adaptive release calc → gain follower → soft clip → brickwall). 4× polyphase FIR upsampler for true-peak detection. |
| `src/audio/builders/r13_adaptive_limiter.js` | Composite builder. Wraps the worklet (with a DynamicsCompressorNode fallback when the worklet isn't yet registered). Exposes `adaptive_limiter` in `R13_ADAPTIVE_LIMITER_BUILDERS`. |
| `tools/calibration/configs/adaptive-limiter-web.json` (desktop tree) | Calibration topology + Logic param hints. |

## Wiring it up in `WebAudioDSPEngine.js`

Two single-line additions, in the same style as R9 / R8 / R4.

**1. At the top of the file, alongside other builder imports:**

```js
import R13_ADAPTIVE_LIMITER_BUILDERS from './builders/r13_adaptive_limiter.js';
```

**2. Inside the `NODE_BUILDERS` map (around line 829), spread it in:**

```js
const NODE_BUILDERS = {
  // ... existing entries unchanged ...
  ...r9Builders,    // algo_reverb (FDN, 4 algos)
  ...R13_ADAPTIVE_LIMITER_BUILDERS,   // adaptive_limiter
};
```

**3. Register the worklet module before the graph is built.** Wherever the
engine boots its AudioContext (typically `_ensureContext()` /
`_ensurePhase1Worklets()`), add:

```js
await ctx.audioWorklet.addModule(
  new URL('../lib/web-audio-plugins/worklets/r13-adaptive-limiter-processor.js', import.meta.url)
);
```

The builder's `_safeWorklet()` falls back to a `DynamicsCompressorNode` in
brickwall configuration if the processor isn't registered — schema renders
and audio still flows.

## Parameter schema

| Param                | Type / Range          | Default | Notes |
|----------------------|-----------------------|---------|-------|
| `gain`               | 0–24 dB               | 0       | Pre-gain into the limiter |
| `out_ceiling`        | -30..0 dB             | -0.3    | Output brickwall ceiling |
| `lookahead_ms`       | 1–12 ms               | 5       | Per-channel lookahead delay; sets the attack time-constant (lookahead/3) |
| `release_min_ms`     | 1–50 ms               | 5       | Release floor on dense passages |
| `release_max_ms`     | 100–2000 ms           | 500     | Release ceiling on sparse passages |
| `release_adaptation` | 0–1                   | 0.7     | Crossfade between fixed (`release_max`) and fully adaptive |
| `true_peak`          | 0/1                   | 1       | 4× polyphase oversample for ISP detection |
| `soft_clip_amount`   | 0–1                   | 0.3     | Blend of 3rd-order soft-clip vs. linear before final brickwall |
| `link_lr`            | 0/1                   | 1       | Couple both channels through one envelope |

## Adaptive-release algorithm

The adaptive release is the differentiator vs. Logic's regular `Limiter`.

**Sliding window of peak-over-ceiling events.** A `Uint8Array` of length
`windowSec * sampleRate` (default 150 ms) holds 1/0 per sample for "did this
sample exceed the ceiling?". A running `eventCount` is updated incrementally
as the ring rotates — O(1) per sample, no scan.

**Density.** `density_norm = min(1, eventCount / maxEventsInWindow)`. The
saturation point `maxEventsInWindow = maxEventsPerSec * windowSec` defaults to
30 events / 150 ms (≈ 200/sec). At density=0 (sparse, e.g. an acoustic guitar
strum) → release_max. At density=1 (a brickwall master with content
constantly hitting the ceiling) → release_min.

**Release time mapping.**
```
release_adapt = release_min + (release_max - release_min) * (1 - density_norm)
release_final = (1 - adaptation) * release_max + adaptation * release_adapt
```

The `release_adaptation` knob lets the surface degrade gracefully to a
fixed-release limiter at adaptation=0 — useful for A/B testing against the
non-adaptive baseline, and a sane operating point if calibration finds
adaptation hurts null-diff on a particular preset.

**Why a sliding window, not an EMA of |peak|?** EMA-based density estimates
(common in adaptive-release commercial limiters) bias toward recent peaks
and chatter under transient bursts. The discrete-event sliding window is
piecewise-stable: the density only changes when an event enters/leaves the
window, so the release time-constant doesn't jitter on every sample. This
matches Logic's "transparent" reputation — release stays put until the
content character actually changes.

## Pipeline summary

```
input (L/R)
   │
   ├── × pre_gain (10^(gain/20))                           ← `gain` param
   │
   ├── lookahead delay (ring, lookahead_ms)
   │
   ├── peak detector
   │     └── if true_peak: 4× polyphase upsample, max |sub-sample|
   │     └── else:         |x|
   │
   ├── adaptive release calc (this section, above)
   │
   ├── gain follower
   │     attack  = exp(-1 / ((lookahead_ms/3000) * sr))
   │     release = exp(-1 / (release_final_sec * sr))
   │     target  = ceiling / max(peak, ceiling)            ← clamped to ≤1
   │
   ├── apply gain to delayed sample
   │
   ├── soft-clip stage (3rd-order, blended by soft_clip_amount)
   │
   └── final brickwall hard-clamp at out_ceiling           ← guarantees ≤ ceiling
```

## True-peak detection

A 32-tap windowed-sinc FIR is split into 4 polyphase phases (8 taps each).
For every input sample we evaluate the 4 sub-phase outputs and take the max
absolute value. Cheap (32 MACs / sample), audible difference vs. naive
`|x|` on bright drum content where ISP can be 1–3 dB above the sample peaks.

When `true_peak=0`, FIR history is still advanced (so toggling it back on
mid-track doesn't glitch).

## Latency

Lookahead delay = `lookahead_ms` (default 5 ms ≈ 240 samples @ 48k). The
calibration mapping should record `latency_samples` in the eventual
`/plugin-mappings/{id}.json` so the playback scheduler compensates.

## Fallback path (no worklet registered)

The builder constructs a `DynamicsCompressorNode` (knee=0, ratio=20,
attack=1ms, release=100ms) wrapped between a pre-gain (driven by `gain`)
and a passthrough post-gain. `out_ceiling` maps onto `threshold`. The
adaptive-release params are no-ops; `release_max_ms` drives the
DynamicsCompressorNode's release as a coarse approximation. The fallback
preserves the ceiling-enforcement contract (peaks ≤ ceiling) and is what
the test harness exercises (Node has no AudioWorklet support).

## Calibration plan

5 presets — transparent / vocal / drums / mix-bus / brickwall-master — at
two input levels (-12 dBFS and -3 dBFS). Curve-fit `gain`, `out_ceiling`,
`lookahead_ms` per-preset. Logic's single `Release` knob maps onto the
triple `(release_min_ms, release_max_ms, release_adaptation)` — recommend
piecewise breakpoints with adaptation pinned at 0.7. Target ≤ -40 dB RMS
null-diff across all 5 presets. Window length (150 ms) is the most likely
calibration knob to expose if null-diff regresses.

## Tests

`tests/r13_adaptive_limiter.test.js` — builder smoke (constructs without
throwing on a Node-mock AudioContext) + ceiling-enforcement regression
(input at -1 dBFS sine, ceiling at -6 dBFS, output peak ≤ -6 dBFS).
