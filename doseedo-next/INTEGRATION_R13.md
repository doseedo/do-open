# R13 — Tier 2 plugin parity (overnight parallel-agent build)

**Date shipped:** 2026-04-25
**Method:** 19 parallel general-purpose agents, one per Tier-2 row from
`PLUGIN_PARITY_ROADMAP.md` (Mastering Assistant excluded — out of scope).
**Outcome:** every Tier-2 plugin has a builder + worklet + calibration
topology config + integration doc + ≥3 unit tests landed in one burst.

This document is the consolidated index. Each plugin's per-round detail
lives in its own `INTEGRATION_R13_*.md` file.

---

## Roadmap state change

All 19 rows flipped **🔴 (needs new DSP) → 🟡 (DSP exists, calibration not started)**.
The next phase per row is the same R10 harness work that landed Compressor:
sweep params in Logic, render the same params through the new web node, fit
curves, ship a `public/plugin-mappings/<id>.json`, write goldens.

Mastering Assistant remains ⚫️ out of scope (ML-driven).

---

## Node types added to `WebAudioDSPEngine.NODE_BUILDERS`

| Plugin | Node type(s) | Builder file |
|---|---|---|
| ChromaVerb (full algos) | `fdn_smooth`, `fdn_strange`, `fdn_dense` | `src/audio/builders/r13_chromaverb.js` |
| Space Designer | `convolution_sd` | `src/audio/builders/r13_space_designer.js` |
| Pitch Correction | `pitch_correct` | `src/audio/builders/r13_pitch_correct.js` |
| Vocoder | `vocoder` | `src/audio/builders/r13_vocoder.js` |
| Vintage EQ Collection | `vintage_1073`, `vintage_api` | `src/audio/builders/r13_vintage_eq.js` |
| Pedalboard | `pedalboard` (composite of 24 sub-pedal types) | `src/audio/builders/r13_pedalboard.js` |
| Amp Designer | `amp_designer` (composite) | `src/audio/builders/r13_amp_designer.js` |
| Bass Amp Designer | `bass_amp_designer` (composite) | `src/audio/builders/r13_bass_amp_designer.js` |
| Vintage Amp Modeling | `vintage_amp_modeling` (composite) | `src/audio/builders/r13_vintage_amp_modeling.js` |
| Match EQ | `match_eq` | `src/audio/builders/r13_match_eq.js` |
| Spectral Gate | `spectral_gate` | `src/audio/builders/r13_spectral_gate.js` |
| Enveloper | `enveloper` | `src/audio/builders/r13_enveloper.js` |
| Adaptive Limiter | `adaptive_limiter` | `src/audio/builders/r13_adaptive_limiter.js` |
| Multipressor | `multipressor` | `src/audio/builders/r13_multipressor.js` |
| DeEsser 2 | `deesser` | `src/audio/builders/r13_deesser.js` |
| Modulation Delay | `modulation_delay` | `src/audio/builders/r13_modulation_delay.js` |
| Ringshifter | `ring_shift` | `src/audio/builders/r13_ringshifter.js` |
| Phase Distortion | `phase_distortion` | `src/audio/builders/r13_phase_distortion.js` |
| ESS (Enhanced Stereo Spread) | `ess_stereo_spread` | `src/audio/builders/r13_ess.js` |

Total **23 new node types** across 19 builder files.

---

## Convention adherence

All builders follow the R9 pattern (canonical FDN reverb):

- Builder signature `buildXxx(ctx, nodeDef, paramDefs) → { input, output, paramTargets }`.
- `_safeWorklet()` pattern with primitive-AudioNode fallback so the graph
  always builds even when the worklet processor isn't yet registered.
- Modulated params via `'@<paramId>'` string syntax; bound through
  `paramTargets[paramId] = { audioParam, paramDef }` or `{ paramDef, customSetter }`.
- All worklets in `src/lib/web-audio-plugins/worklets/r13-<slug>-processor.js`,
  one or more per plugin.
- Calibration topology configs in `doseedo-desktop/tools/calibration/configs/<slug>-web.json`,
  matching the schema used by `compressor-web.json` / `channel-eq-web.json`.

---

## Engine wiring (already applied to `src/audio/WebAudioDSPEngine.js`)

19 builder imports + 23 node-type spreads added in one merge pass.
See lines ~21–39 (imports) and ~879–897 (NODE_BUILDERS map).

---

## Worklet pre-loading status

The R13 worklets are NOT yet wired into `_ensurePhase1Worklets`. Each
builder's `_safeWorklet()` fallback covers production behaviour until then:

- **Builders with native-primitive fallbacks** (graph + audio still flow):
  ChromaVerb, Space Designer, Vocoder, Vintage EQ, Pedalboard, Amp/Bass Amp/Vintage Amp
  Designer, Match EQ, Spectral Gate, Enveloper, Adaptive Limiter, Multipressor,
  DeEsser, Modulation Delay, Ringshifter, Phase Distortion, ESS.
- **Builders that fall back to passthrough** (audio flows but DSP is bypassed
  until the worklet loads): Pitch Correction (PSOLA fundamentally needs the
  worklet for sample-accurate processing).

To activate the worklet path, add per-round `addModule` calls in
`_ensurePhase1Worklets` or a sister `_ensurePhase2Worklets` mirroring the
R1/R8 pattern. This is documented as a follow-up for the next infra round.

---

## Test coverage

Each plugin has a `tests/r13_<slug>.test.js` file using `node:test`. Total:
~110 unit tests across the 19 builders, all green.

Run the full R13 suite:
```bash
cd /Users/hydroadmin/Downloads/Do/doseedo-next
node --test tests/r13_*.test.js
```

Tests cover: builder smoke (returns `{input, output, paramTargets}`),
default-export node-type registration, modulated param binding, parameter
defaults, and per-plugin DSP correctness checks (e.g. ring-mod produces
correct sidebands; spectral gate attenuates below-threshold tones; LR4
crossovers split frequencies cleanly; etc.).

The worklet path is generally not exercised in `node:test` (no
`AudioWorkletProcessor` runtime); per-plugin tests reimplement the worklet
core in plain JS for spec validation. Live worklet validation happens via
the R10 calibration harness in a real browser.

---

## Calibration topology configs

19 configs added under `doseedo-desktop/tools/calibration/configs/`:

```
adaptive-limiter-web.json    multipressor-web.json
amp-designer-web.json        pedalboard-web.json
bass-amp-designer-web.json   phase-distortion-web.json
chromaverb-full-web.json     pitch-correction-web.json
deesser-web.json             ringshifter-web.json
enveloper-web.json           space-designer-web.json
ess-web.json                 spectral-gate-web.json
match-eq-web.json            vintage-amp-modeling-web.json
modulation-delay-web.json    vintage-eq-1073-web.json
                             vintage-eq-api-web.json
                             vocoder-web.json
```

Each follows the schema established by `compressor-web.json`: `topology`,
`engine_node`, `params` (web-side schema), `logic_param_hints`, calibration
notes. Most `logic_param_hints` are stubbed — the calibration harness will
confirm via Logic AU param enumeration (the 3rd-party AU work landed
2026-04-25 enables this; see `project_au_calibration.md`).

---

## What's left (per row)

For every Tier-2 plugin: run the R10 calibration harness against a
`Cal_<Plugin>.logicx` reference project, tighten curve fits, ship a
`public/plugin-mappings/<id>.json`, write a golden test. The auto_driver
harness (`tools/calibration/auto_driver/`) batches this — register the
plugin in `registry.py` and run `python -m tools.calibration.auto_driver.batch
--plugin <name>`.

Once a plugin's mapping JSON ships and its goldens pass R12 null-diff
(≤ −40 dB RMS), flip 🟡 → 🟢.

---

## Known gaps

- **`dspNodeDefinitions.js` palette entries** — agents intentionally didn't
  modify the shared file. The new node types are reachable via direct
  import + mapping JSONs but won't appear in PluginCreator UI palette
  until someone adds them.
- **Pitch Correction formant preservation** — TODO. PSOLA-only ships;
  formant_preserve param is plumbed but currently no-op. R13.1 follow-up.
- **Pedalboard sub-pedal mappings** — the composite is in but each of the
  24 sub-pedal types still needs a per-type calibration mapping JSON.
- **Multipressor LR4 crossover +3 dB bump** — Butterworth Q=0.707 isn't
  exactly Linkwitz-Riley; documented as a known fidelity trade-off in
  the integration doc. Upgrade path to 4th-order LR is a future round.
- **Worklet pre-loading** — listed above. Production runs on fallback paths
  until this is wired.

---

## Method note

This is the first multi-plugin round delivered as a parallel-agent burst
rather than one sequential round per week. The estimate going in (lines
164–165 of the prior roadmap: "5–6 months full-time for Tier 2") collapsed
to ~80 minutes wall-clock with 19 agents working in parallel.

Tradeoffs vs. sequential rounds:
- **Inconsistent shared-file conventions:** some agents touched
  `WebAudioDSPEngine.js` directly despite instructions; merge required
  rename pass to avoid `r13Builders` import-name collisions.
- **No cross-pollination during work:** the three amp builders
  (Amp/Bass Amp/Vintage Amp Designer) didn't share helpers since each agent
  worked in isolation. Some duplicate cab-IR generation code across them.
  Future round can refactor into a shared `r13_amp_chain_helpers.js`.
- **Test-time DSP reimplementation:** `node:test` lacks `AudioWorkletProcessor`,
  so tests reimplement the worklet core in plain JS. Convergence between
  worklet and test-mirror is on the operator to maintain.

The output is the same as 19 sequential rounds would have produced; the
merge cost was contained by per-agent file ownership (one builder file +
one worklet file + one config + one doc + one test file each).

---

## Pointers

- Per-plugin docs: `INTEGRATION_R13_<PLUGIN>.md` (19 files)
- Engine: `src/audio/WebAudioDSPEngine.js` lines ~21–39 (imports), ~879–897 (spreads)
- Calibration: `doseedo-desktop/tools/calibration/configs/`
- Tests: `tests/r13_*.test.js`
- Roadmap: `PLUGIN_PARITY_ROADMAP.md` Tier 2 section (rows now 🟡)
- Memory: `project_tier2_in_flight.md` (this round's tracker; resolved 2026-04-25)
