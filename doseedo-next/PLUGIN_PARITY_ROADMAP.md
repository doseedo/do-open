# Logic Stock Plugin → Web DSP Parity Roadmap

**Goal:** every Logic Pro stock plugin runs natively in the Doseedo web
runtime (`WebAudioDSPEngine` + `PluginAdapter`), with real-time
knob-draggable parameters and audio-faithful output (R12 null-diff
≤ −40 dB RMS vs. a Logic bounce of the same input + preset).

**Why this matters:** today the web DAW falls back to bounce-cache audio
for any plugin without a `/plugin-mappings/{id}.json`. Bounce-cache means
no real-time editing, no live collab, no cheap session re-render. Native
parity unlocks all three. It also makes the offline `--au-4cc` bounce
path useful for stock plugins (which Apple doesn't expose as registered
AudioComponents — see `docs/logicx_format.md` § "kind=1 payload").

**Snapshot owner:** keep this doc updated as mappings ship. Each
plugin row's `status` column is the source of truth for coverage.

---

## What's already shipped

These are landed in `main` (commits `e4aa8859` Phase 1 + `5964b808` Phase 2A,
desktop `b292eec`). **Don't repeat work below — this is the foundation
the rest of the roadmap builds on.**

### DSP graph language

`WebAudioDSPEngine.NODE_BUILDERS` recognises these node types:

| Round | Node types | Notes |
|---|---|---|
| Legacy | `delay, reverb, chorus, flanger, phaser, distortion, overdrive, saturator, tremolo, compressor, expander, gate, limiter, hybrid_reverb, echo` | Pre-R1 worklets, retained |
| **R1** | `bitcrusher, multitap_delay, convolution, envelope_follower, pitch_shift, comb, math_{add,multiply,abs,rectifier,slew,scale,crossfade,constant}` | Worklets + non-worklet fallbacks |
| **R2** | WDF: `wdf_diode_clipper, wdf_transistor_clipper, wdf_tube_triode, wdf_tube_amp, wdf_tone_stack` | Wave-Digital-Filter analog modeling |
| **R3** | WDF: `wdf_tape_sat, wdf_transformer, wdf_rc_filter, wdf_rlc_filter, wdf_power_supply_sag` | Continuation of R2 |
| **R4** | `circuit_fender_bassman, circuit_pultec_eq` (+ more circuit models) | Composite analog circuits |
| **R5** | `pitch_shift_pv, spectral_filter, spectral_freeze` | Phase-vocoder + spectral |
| **R8** | `sidechain, compressor_sc, gate_sc` | Sidechained dynamics |
| **R9** | `algo_reverb` | FDN reverb, 4 algos (room/hall/chamber/plate) |
| **a3** | (sampler primitives via builder, not standalone nodes) | EXS24 sampler engine |

### Synthesis primitives

- `VoiceManager` (R6) — polyphonic voice allocation, MIDI note routing.
- `ModRouter` (R7) — mod matrix, LFO/envelope → AudioParam routing.
- `MidiInput` — Web MIDI bridge.
- `EXS24Reader` + `Keymap` + `SamplePlayer` (Phase 2A) — Logic EXS24
  sampler decoding + zoned playback. Foundation for all sample-based
  instrument mappings.
- `WebAudioDSPEngine` instrument mode (Phase 2A) — engine constructed
  from a synth-flavored dspChain instead of an effect chain.

### Live integration loop (A5)

```
Logic knob drag
  → doo_hook AUEventListener (kind=7 patcher) → /tmp/doo_live_deltas.sock
  → chat_server.py _param_delta_relay_loop → /ws broadcast `param_delta`
  → StudioDev's lifted useAgentWebSocket holds wsRef
  → useLiveParamDeltas → liveTrackChainRegistry lookup
  → slot.setLogicParam → engine.setParameter (setTargetAtTime ramp)
  → audible in ~20 ms
```

### Calibration + validation

- **R10** — calibration harness: drives Logic + headless web render +
  null-diff. One mapping authored to date: `Compressor (154)`.
- **R12** — `dspMetrics.js` (RMS / peak / 1/3-octave spectral diff),
  `goldenTests.js` (registry + runner), `ValidationPanel.js` at
  `/dev/validation` for human-in-loop tuning. Fallback adapter for
  CI when no real PluginAdapter is on the path.
- **R11** — `PluginAdapter.js` resolves Logic plugin records →
  WebAudioDSPEngine slots via `/plugin-mappings/{id}.json` mapping JSONs.

---

## Coverage matrix

Status legend:
- 🟢 **shipped** — mapping JSON merged + R12 golden(s) green.
- 🟡 **calibration in progress** — DSP exists, mapping being authored.
- 🔵 **DSP exists, calibration not started**.
- 🔴 **needs new DSP node(s)** — see the per-plugin notes.
- ⚫️ **out of scope for this phase** (Mastering Assistant, etc).

### Tier 1 — calibration only (no new DSP)

These decompose into existing R1–R9 nodes. Per-plugin work: bounce N
presets + curve-fit per param + author `web_topology` + R12 null-diff.
Estimated 30 min – 2 h each once the auto-driver harness lands.

| Plugin | Logic ID | Status | Topology hint |
|---|---|---|---|
| Compressor | 154 | 🟢 | dynamics: side-chain envelope → gain VCA |
| Limiter | tbd | 🔵 | hard limit + lookahead delay |
| Noise Gate | tbd | 🔵 | env-follower → gate threshold |
| Channel EQ | tbd | 🔵 | 8× biquad cascade, lowshelf/highshelf/peaking/cut |
| Single Band EQ | tbd | 🔵 | 1× biquad |
| Linear Phase EQ | tbd | 🔵 | FIR (R5 spectral foundation) |
| Fat EQ | tbd | 🔵 | 5× biquad cascade |
| AutoFilter | tbd | 🔵 | LFO → filter cutoff (ModRouter) |
| Distortion II | tbd | 🔵 | R2 wdf_diode_clipper or wdf_transistor_clipper |
| Clip Distortion | tbd | 🔵 | hard clip + LP |
| Overdrive | tbd | 🔵 | R2 wdf_tube_triode |
| Bitcrusher | tbd | 🔵 | R1 bitcrusher direct |
| Stereo Delay | tbd | 🔵 | R1 multitap_delay × 2 |
| Tape Delay | tbd | 🔵 | R1 multitap + R3 wdf_tape_sat in feedback loop |
| Sample Delay | tbd | 🔵 | static delay line |
| Echo | tbd | 🔵 | R1 multitap |
| Chorus | tbd | 🔵 | legacy chorus-processor.js |
| Ensemble | tbd | 🔵 | 8× chorus voices |
| Flanger | tbd | 🔵 | legacy flanger-processor.js |
| Microphaser | tbd | 🔵 | small-stage phaser |
| Phaser | tbd | 🔵 | legacy phaser-processor.js |
| Tremolo | tbd | 🔵 | legacy tremolo-processor.js |
| Spreader | tbd | 🔵 | mid/side gain matrix |
| Stereo Spread | tbd | 🔵 | mid/side EQ |
| Direction Mixer | tbd | 🔵 | M/S decode + width |
| Gain | tbd | 🔵 | trivial gain node |
| Stereo Pan | tbd | 🔵 | 2-channel pan law |
| Loudness Meter | tbd | 🔵 | analysis-only (no audible output) |
| Level Meter | tbd | 🔵 | analysis-only |
| Correlation Meter | tbd | 🔵 | analysis-only |
| Tuner | tbd | 🔵 | pitch-detect, no DSP path |
| ChromaVerb (basic algos) | tbd | 🔵 | R9 algo_reverb (room/hall/chamber/plate covered) |
| Silver Verb | tbd | 🔵 | smaller FDN |
| EnVerb | tbd | 🔵 | shorter FDN with env modulation |
| PlatinumVerb | tbd | 🔵 | longer FDN |
| AVerb | tbd | 🔵 | algorithmic — fits R9 |
| GoldVerb | tbd | 🔵 | algorithmic — fits R9 |

**Tier 1 total:** ~37 plugins. Realistic completion with auto-driver
harness: **3–5 weeks of focused calibration** (1 engineer).

### Tier 2 — needs new DSP nodes

Each row is a small R-round (worklet + builder + integration doc + ≥1
mapping). Estimated 1–3 weeks per row.

| Plugin | Status | New nodes needed |
|---|---|---|
| ChromaVerb (full algorithm set) | 🔴 | `fdn_smooth`, `fdn_strange`, `fdn_dense` — variants beyond R9's 4 algos |
| Space Designer | 🔴 | `convolution_sd` — IR truncation, decay envelope, EQ, length scaling on top of R1 convolution |
| Pitch Correction | 🔴 | `pitch_correct` — formant-preserving PSOLA + scale-snap quantizer |
| Vocoder | 🔴 | `vocoder` — analysis filter bank + carrier osc + N-band envelope routing |
| Vintage EQ Collection | 🔴 | passive LCR models for 1073, API (R4 has Pultec already) |
| Pedalboard | 🔴 | 24 stomp models — most fit R2/R3/R4 once each is calibrated as a sub-node |
| Amp Designer | 🔴 | composite: R2 preamp + R3 transformer + R3 power supply + cab IR loader |
| Bass Amp Designer | 🔴 | composite, similar to Amp Designer |
| Vintage Amp Modeling | 🔴 | composite, similar |
| Match EQ | 🔴 | `match_eq` — long-FFT spectrum matching + smoothing + render mode |
| Spectral Gate | 🔴 | `spectral_gate` — per-band threshold (R5 spectral foundation) |
| Enveloper | 🔴 | `enveloper` — transient shaper (attack/sustain) |
| Adaptive Limiter | 🔴 | `adaptive_limiter` — multi-stage limiting with adaptive release |
| Multipressor | 🔴 | `multipressor` — 4-band parallel compressor (use R8 sidechain × 4) |
| DeEsser 2 | 🔴 | `deesser` — dynamic peaking EQ |
| Modulation Delay | 🔴 | tape-style chorus/flanger combo |
| Ringshifter | 🔴 | `ring_shift` — ring mod + freq shift |
| Phase Distortion | 🔴 | `phase_distortion` — Casio-style PD osc |
| ESS (Enhanced Stereo Spread) | 🔴 | larger M/S manipulation |
| Mastering Assistant | ⚫️ | ML-driven; ship as offline-only |

**Tier 2 total:** ~20 plugins / DSP rounds. Realistic completion:
**5–6 months** (1 DSP engineer, full-time).

### Tier 3 — instruments (months each)

Each is a synthesis core, not a graph wiring exercise.

| Instrument | Status | New primitives needed |
|---|---|---|
| Quick Sampler / EXS24 | 🟡 | foundation in (`SamplePlayer` + `EXS24Reader` + `Keymap`); zone-mod routing + mapping JSON to finish |
| Drum Kit Designer | 🟡 | sampler core (covered) + per-cell processing chain |
| Drum Machine Designer | 🟡 | sampler core (covered) + per-cell synth slot |
| Ultrabeat | 🔴 | multi-engine drum cells (sample + synth + phys-mod). Hard scheduling, easy DSP |
| ES1 | 🔴 | `analog_osc` (PolyBLEP saw/square/triangle/pulse), Moog ladder filter |
| ES2 | 🔴 | `analog_osc` × 3 + wavetable osc + ladder filter + multi-mode filter |
| Retro Synth | 🔴 | (analog/wavetable/FM/sync modes — ES2-class core) |
| Mono/Poly | 🔴 | `analog_osc` × 4 + ladder filter |
| ES E / M / P | 🔴 | (covered by analog VA round) |
| EFM1 | 🔴 | `fm_op` × 2 — small project |
| Drum Synth | 🔴 | drum-cell synthesis from R1 + envelopes (cheap) |
| Vintage Electric Piano | 🔴 | sampler core + tine/reed velocity zones + tremolo + cabinet |
| Vintage Clav | 🔴 | sampler core + filter + wah + amp model |
| Vintage Mellotron | 🔴 | sampler core (cheap once sampler holds) |
| Vintage B3 | 🔴 | drawbar additive synth + Leslie rotor + tube preamp |
| **Sculpture** | 🔴 | `modal_string` (8 modes, stiffness/damping/decay), `exciter` (pluck/bow/strike), `resonant_body`, 2D pickup. **2–3 months dedicated** |
| **Alchemy** | 🔴 | granular + additive + spectral resynth + sampler, morphable presets. **Multi-quarter project** |

**Tier 3 total:** ~17 instruments. Realistic completion:
**6–9 months** for everything except Sculpture + Alchemy (each their
own dedicated project).

---

## Infrastructure that must scale alongside

These are blockers / leverage multipliers — invest before plugin #20.

### 1. Auto-driver calibration harness (✅ landed 2026-04-25)

The R10 harness (`calibrate.py` + `logic_driver.py` + `web_renderer.py` +
`curve_fit.py` + `scoring.py`) handled one plugin at a time and wrote
mappings to a gitignored staging dir. The auto-driver layer scales it
to N plugins and adds a publish step into the web tree.

**Location:** `tools/calibration/auto_driver/` (in the desktop tree).

**Modules:**

| File | Purpose |
|---|---|
| `registry.py` | Declarative `PluginEntry(plugin_id, plugin_name, web_topology, logic_project, signals, param_sweeps, combined_sweeps, tier, notes)` rows — the source of truth for the backlog. |
| `state.py` | JSON-backed per-plugin run state at `auto_driver/state.json`. Atomic writes (rename-after-tempfile) so Ctrl-C never corrupts. Schema-versioned, history-trimmed deque per plugin, corruption recovery. |
| `batch.py` | CLI orchestrator — `--list`, `--plugin <name>`, `--tier N`, `--all --resume`, `--force`, `--dry-run`. Spawns `calibrate.py` per plugin (clean process — one crash never poisons the whole run). Pre-flight checks for missing topology / project paths surface as `blocked` instead of running for two minutes then dying. |
| `publish.py` | Idempotent copy from the staging dir to `Do/doseedo-next/public/plugin-mappings/{id}.json`, with `index.json` auto-update so `PluginAdapter.load()` picks new mappings up. Validates each mapping JSON's shape + plugin_id match before promoting. |
| `tests/` | 19 unit tests covering registry filters, state round-trip + corruption recovery + atomic-write sanity, publish dry-run + validation. ~6 ms. No Logic or Chromium needed → CI-ready. |

**Curve fitting:** already automatic in `curve_fit.py:fit_param` —
tries linear/log/exp/piecewise, picks best R². No new code needed.

**What's deferred** (documented in the README's "What this layer
deliberately does NOT do" section):
- Programmatic Logic preset enumeration from `~/Library/Audio/Presets/Logic/<plugin>/*.cst`.
- Auto-generation of web-topology configs (hand-authored per plugin family for now).
- Auto-construction of the calibration `.logicx` per plugin (operator
  builds one per plugin — `Cal_<Plugin>.logicx`).

These are nice-to-haves; the harness is functional without them.

**Verification:** 19/19 unit tests pass; `--list` and `--dry-run`
smoke-test green without Logic running.

**Usage in 30 seconds:**

```bash
cd /Users/hydroadmin/Downloads/doseedo-desktop
python -m tools.calibration.auto_driver.batch --list
python -m tools.calibration.auto_driver.batch --plugin "Compressor"
python -m tools.calibration.auto_driver.publish
```

See `tools/calibration/auto_driver/README.md` for the full how-to,
including how to add a plugin to the backlog (one-page checklist).

### 2. CI gate

Wire `goldenTests.runAll(...)` into a GitHub Action that runs on every
PR. Fail the build if any preset's null-diff regresses past its
threshold. R12 doc already sketches the script.

`.github/workflows/plugin-goldens.yml`:
```yaml
- run: cd doseedo-next && npm ci
- run: cd doseedo-next && node scripts/run-goldens.mjs
```

The script lives at `doseedo-next/scripts/run-goldens.mjs` (template
in `INTEGRATION_R12.md` § "CI hook"). Emits JSON of failed entries +
exit 1 on any fail.

**Block the long tail with this**: even mapping #5 should be guarded.

### 3. Engine pooling + memory budget

`PluginAdapter` instantiates one `WebAudioDSPEngine` per slot. At 30
tracks × 8 inserts = 240 engines, each with its own ConvolverNode IR,
delay buffers, biquad chain. Memory profile:

- Empirical at the time of writing (Tier 1, no instruments): ~200 MB
  with all 240 engines warm. With instruments → projected 500–800 MB.
- The `_engineCache` Map in PluginAdapter is wired but unpopulated.

**Fix:** key the cache by `(plugin_id, param_state_hash)` (xxhash of
the param Float32Array). Identical states share AudioNodes via
`WeakRef`. Invalidation runs on the playback `stop()` path, before
the next `play()` builds chains.

### 4. Mapping schema versioning

Today `web_topology` is implicit — it's whatever WebAudioDSPEngine
accepts. As R-rounds add nodes / change parameter shapes, old mappings
will drift.

**Fix:** add `schema_version: "1.0"` to every mapping JSON. Bump on
breaking changes. Migration table in `PluginAdapter.js` translates
old → new at load time. Do this BEFORE shipping mapping #10 — easier
to add a field early than retrofit.

### 5. Bidirectional delta channel

A5 today: Logic → web only. For collab where multiple users drive the
same instance, web → Logic must work too. The doo_helper IPC already
supports `AudioUnitSetParameter` — just plumb `slot.setParam(...)`
calls from the web UI back over the WS to a new server-side relay.

**Risk:** echo loops. Solve via origin tagging on each param event
(Logic-origin events skip the round-trip back to Logic).

### 6. Worklet pre-warming

`_ensurePhase1Worklets` is now safe (Promise rejections suppressed
post the production-readiness pass) but still lazy. First-build of a
graph using R1/R8 worklets falls back to the non-worklet path while
`addModule` resolves.

**Fix:** call `await ctx.audioWorklet.addModule(...)` for the entire
worklet set inside `useAudioPlayback`'s `_initContext` `useEffect`,
gated on the first user gesture (which is when `AudioContext.resume()`
fires). Add a `worklet_ready` Promise on the context and have
PluginAdapter `await` it before instantiating an engine.

### 7. AU 4CC registry workaround for stock plugins (offline path)

Stock Logic plugins aren't in the system AudioComponent registry, so
`render_plugin.swift` falls through to dry render for them. The
diff-based collab flow handles that (owner sends `wet - dry`, peer
reconstructs from diff), but it's a workaround.

**Two options:**
- **(a)** Reverse-engineer Logic's `MADSPPlugInPublic.framework`
  loader and call it from `render_plugin.swift` with a synthesized
  ClassInfo plist. High-risk, fragile across Logic versions.
- **(b)** Accept the workaround for stock plugins — the live web-DSP
  path is the real win, and the bounce-cache path is acceptable for
  unmapped plugins via the existing diff flow.

**Recommendation:** (b). The roadmap is already enough work without
adding (a). Revisit if a real customer pain emerges.

---

## Pragmatic sequencing

If we're optimizing for **time to meaningful coverage** (covers the
median Logic project), not 100% parity:

1. ✅ **Auto-driver harness** — `tools/calibration/auto_driver/`. **Done 2026-04-25.**
2. **Top-20 effects by usage** (Tier 1 + early Tier 2):
   Channel EQ, Limiter, ChromaVerb, Tape Delay, Stereo Delay, Pedalboard,
   Bitcrusher, Distortion II, AutoFilter, Tremolo, Chorus, Phaser,
   Flanger, Pitch Correction, Vocoder, Match EQ, DeEsser, Direction
   Mixer, Multipressor, Space Designer.
   (4–6 weeks with auto-driver.)
3. **Sampler-family instruments** (Quick Sampler, EXS24, Drum Kit
   Designer, Drum Machine Designer). Foundation already in — finishes
   sample-based MIDI tracks. (3–4 weeks.)
4. **Analog VA round** — ES1, ES2, Retro Synth, Mono/Poly, ES E/M/P
   in one shared R-round (`analog_osc` + ladder + multi-mode filter).
   (4–6 weeks.)
5. **CI gate + engine pool + schema versioning** — lock in
   regression-proofing. (1–2 weeks.)
6. **EFM1 + Drum Synth + Vocoder** — quick wins, fill out coverage.
   (3 weeks.)
7. **Vintage instruments** (EP, Clav, Mellotron, B3) — sampler-core
   reuse + their characteristic processing chains. (4 weeks.)
8. **Sculpture** — dedicated 2–3 month DSP project.
9. **Alchemy** — multi-quarter, optional. Most users don't depend on it.
10. **Long tail** — niche effects, Mastering Assistant decision, AVerb
    / GoldVerb completionism.

**Realistic milestones:**

- **6 months focused work** = Tier 1 + Tier 2 + sampler + analog VA →
  meaningful coverage for the median Logic project. **Ship-able.**
- **12–18 months** = 100% parity including Sculpture + Alchemy.

---

## How to add a new mapping (concrete steps, today)

Until the auto-driver harness lands, here's the manual process. This
is the contract every Tier 1 plugin will follow.

1. **Capture references.** Bounce 3–5 representative presets from
   Logic with the same input source (`gen://drums` or a known WAV).
   Drop into `public/assets/golden/{plugin_id}_{preset}.wav`.
2. **Pick web_topology.** Hand-author the dspChain + parameters in a
   draft `public/plugin-mappings/{plugin_id}.json`. Reference
   `public/plugin-mappings/154.json` for the schema (`web_topology`,
   `param_map[]`, `param_map_by_name`).
3. **Param-sweep each Logic param.** With Logic open and the plugin
   instantiated, drive each param across its range (e.g. 11 steps).
   The kind=7 patcher can automate this. Capture the bounce at each
   step.
4. **Fit the curve.** For each `param_map[]` entry, pick `curve` ∈
   {linear, log, exp, piecewise}. For piecewise, supply 3–7
   `breakpoints` (logic_value, web_value pairs). The curve must
   minimize R12 RMS null-diff across all swept presets.
5. **Validate in `/dev/validation`.** Load the source + a Logic-bounced
   reference, dial in the params, click "Render web side". The
   "RMS null-diff" cell should go green (≤ −40 dB). If it doesn't,
   either the curve is wrong, or the topology is missing a node.
6. **Save golden.** Click "Save golden" in the panel. Persist the
   resulting entry into `DEFAULT_GOLDEN_TESTS` in
   `src/lib/goldenTests.js` so CI catches regressions.
7. **PR.** One mapping JSON + N golden entries + the WAVs. Reviewer
   checks: schema validates, golden runs green locally.

---

## Risks / open questions

- **Logic version drift.** Apple updates stock plugin DSP between
  Logic releases (rarely, but it happens). A null-diff that's green
  today may regress after a Logic update. Mitigation: document the
  Logic version each mapping was calibrated against; flag null-diff
  regressions in CI as version drift candidates.
- **Sample rate handling.** Logic projects are typically 44.1k or 48k.
  Web AudioContext defaults to system rate (often 48k). All R12
  metrics are sample-rate-agnostic but some R2/R3 WDF models have
  rate-dependent calibration constants. Audit before shipping
  cross-rate.
- **Stereo width / mid-side encoding.** Several Logic plugins
  (Direction Mixer, Stereo Spread) work in M/S domain. The web
  AudioGraph is L/R native. Need an explicit M/S encode/decode node
  pair in the vocabulary — easy add but currently missing.
- **Latency parity.** Logic plugins have known PDC (plugin delay
  compensation) values. The web side must match for clip alignment.
  Add a `latency_samples` field to the mapping schema; the playback
  scheduler already has compensation infra.
- **Preset compatibility.** A Logic preset (.cst) is an NSKeyedArchiver
  blob. We'll want a `.cst` → mapping-param-state translator so
  end-users can drag a Logic preset into the web DAW. Out of scope
  for plugin parity itself but worth flagging.
- **Mastering Assistant.** ML-driven; not a fixed-DSP plugin. Decision:
  ship as desktop-only ("preview in web, render on desktop") rather
  than try to clone the inference model. Revisit when there's enough
  data to train one.

---

## Pointers

- `INTEGRATION_R1.md … INTEGRATION_R12.md` — per-round integration specs.
- `INTEGRATION_A5.md` (desktop tree) — live param-delta channel design.
- `docs/logicx_format.md` (desktop tree) — Logic project file decode.
- `tools/calibration/` (desktop tree) — R10 calibration harness origin.
- `public/plugin-mappings/154.json` — reference mapping (Compressor).
- `src/lib/PluginAdapter.js` — runtime resolver.
- `src/audio/WebAudioDSPEngine.js` — DSP runtime + NODE_BUILDERS.
- `src/lib/goldenTests.js` + `src/components/Plugins/ValidationPanel/` — R12 fidelity QA.
- `src/lib/liveTrackChainRegistry.js` + `src/hooks/useLiveParamDeltas.js` — A5 frontend.
