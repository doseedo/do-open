# R12 — Validation panel + golden-test infrastructure

Agent: **R12** (validation / fidelity QA)
Status: panel + lib + dev route shipped. Goldens registry seeded with 5
Compressor presets — assets are user-supplied, no auto-fetch on import.

## Files added

| Purpose | Path |
|---|---|
| Validation UI | `src/components/Plugins/ValidationPanel/ValidationPanel.js` |
| Validation styles | `src/components/Plugins/ValidationPanel/ValidationPanel.module.css` |
| Pure metrics lib | `src/lib/dspMetrics.js` |
| Golden registry + runner | `src/lib/goldenTests.js` |
| Page (canonical) | `src/app/dev/validation/page.js` |
| Page (real route) | `app/dev/validation/page.js` |

The R12 spec called for the page at `src/app/dev/validation/page.js` but
the Next.js App Router root in this project is `app/`, not `src/app/`.
Both files exist; `app/dev/validation/page.js` is a one-line re-export of
the canonical body in `src/app/dev/validation/page.js`. Tear down the
re-export once the App Router root is unified.

## Accessing the validation tool

In dev:

```bash
cd /Users/hydroadmin/Downloads/Do/doseedo-next
npm run dev
# open http://localhost:3000/dev/validation
```

In prod (after Vercel deploy of `main`): https://doseedo.com/dev/validation.
The route is **not gated by Clerk** and **not linked from any user-facing
nav** — it's an internal QA surface. If you want it gated, wrap the panel
component in `<SignedIn>` from `@clerk/nextjs` inside the page.

The panel mounts on the client only (`ssr: false`) because it instantiates
an `AudioContext` and reads file uploads via `FileReader`/`File.arrayBuffer()`.

## Component layout

```
┌──────────────────────┬───────────────────────────────────┐
│ CONTROLS (320 px)    │ VISUALS                           │
│                      │                                   │
│ Plugin selector      │ Reference waveform   (canvas)     │
│ Source-signal kind   │ Web-DSP waveform     (canvas)     │
│   noise / sweep /    │ Null waveform        (canvas)     │
│   impulse / drums /  │ ─────────────────────────────────  │
│   custom upload      │ Metrics grid                      │
│ Reference WAV upload │   Ref RMS  Web RMS                │
│                      │   RMS Δ    Peak Δ                 │
│ Per-param sliders    │ ─────────────────────────────────  │
│ (from mapping JSON)  │ 1/3-octave spectral diff (canvas) │
│                      │ ─────────────────────────────────  │
│ [Render web side]    │ Goldens table                     │
│ [Save golden]        │   id / preset / RMS Δ / status    │
│ [Run all goldens]    │                                   │
└──────────────────────┴───────────────────────────────────┘
```

## Workflow

1. Pick a plugin (defaults to `154` — the Logic Compressor mapping that
   R10/R11 calibrated). Sliders auto-populate from
   `/plugin-mappings/{id}.json`.
2. Pick a source signal — a built-in generator or a custom WAV upload.
3. Upload the Logic-bounced reference WAV.
4. Tweak params, click **Render web side**. The web DSP path runs the
   source through R11's `PluginAdapter.render(...)` (or
   `simpleCompressor` if R11 isn't present yet — see "Adapter mode" pill
   in the Plugin section of the panel).
5. Inspect the three waveforms + metrics. The "RMS null-diff" cell is
   colour-coded — green ≤ −40 dB, amber ≤ −20 dB, red otherwise.
6. **Save golden** to lock in the current `(plugin_id, params, source,
   ref)` tuple as a regression baseline. The threshold is auto-set to
   `ceil(observed_rms_diff_db) + 6` to give 6 dB of headroom.
7. **Run all goldens** to evaluate the entire registry against the live
   adapter — useful on a release branch before bouncing the next build.

## Metric formulas

All in `src/lib/dspMetrics.js`. Pure JS, zero deps.

| Metric              | Formula                                                                       |
|---------------------|-------------------------------------------------------------------------------|
| `rmsDb(buf)`        | `20 · log10( sqrt( sum(b[i]²) / N ) )`                                        |
| `peakDb(buf)`       | `20 · log10( max(|b[i]|) )`                                                   |
| `rmsDiffDb(r, w)`   | `20 · log10( sqrt( sum( (r[i]−w[i])² ) / N ) )`                               |
| `peakDiffDb(r, w)`  | `20 · log10( max(|r[i]−w[i]|) )`                                              |
| `nullDiff(r, w)`    | `Float32Array` of `r[i] − w[i]`                                               |
| `thirdOctaveSpectralDiff(r, w, sr)` | per-band `20 · log10( |avg(|FFT(r)|) − avg(|FFT(w)|)| )` over ISO 1/3-oct bins |

All dB outputs floor at `DB_FLOOR = -240` so identical buffers don't propagate
`-Infinity`. The 1/3-octave bands are the standard ISO centres
(20 Hz … 20 kHz).

The FFT is a vanilla radix-2 Cooley-Tukey, fed a Hann-windowed segment
sized at the largest power-of-two ≤ buffer length (default cap not
enforced — caller can pass an explicit `fftSize`).

## dB null-diff thresholds (convention)

| RMS null-diff | Verdict                                                   |
|---------------|-----------------------------------------------------------|
| ≤ −60 dB      | Bit-equivalent / floating-point noise floor              |
| ≤ −40 dB      | Production-quality match — ship                           |
| ≤ −30 dB      | Audibly close, character intact                           |
| ≤ −20 dB      | Clearly different timbre but same envelope shape          |
| > −20 dB      | Mapping is broken or running on the wrong source          |

The default per-preset `expected_max_diff_db` in
`DEFAULT_GOLDEN_TESTS` (`src/lib/goldenTests.js`) reflects how much
character a given preset has — limiter-style presets (heavy nonlinearity)
get a looser threshold than gentle bus compression because tiny
differences in detector ballistics show up as larger sample-domain diffs.

## Golden-test schema

```ts
type GoldenTest = {
  id:                  string;            // unique within registry
  plugin_id:           string;            // matches /plugin-mappings/{id}.json
  preset_name:         string;
  params:              { [paramId: string]: number };  // 0..1
  source_audio_url:    string;            // see "Asset URL conventions"
  ref_bounce_url:      string;
  expected_max_diff_db: number;           // pass threshold (RMS null-diff)
  notes?:              string;
};
```

`runAll(audioContext, pluginAdapter, opts)` returns:

```ts
type GoldenResult = {
  id:           string;
  plugin_id:    string;
  preset:       string;
  pass:         boolean;        // diff_db ≤ threshold_db
  diff_db:      number | null;  // RMS null-diff (dB)
  peak_diff_db: number | null;
  threshold_db: number;
  error:        string | null;
};
```

### Asset URL conventions

The runner is asset-agnostic — it delegates to `opts.resolveBuffer(url) →
Promise<Float32Array>`. The default UI resolver in `ValidationPanel.js`
recognises three forms:

| URL prefix   | Behaviour                                                                |
|--------------|--------------------------------------------------------------------------|
| `gen://kind` | Synthesize on the fly (`gen://drums`, `gen://noise`, `gen://sweep`, …)   |
| `file://...` | Refuses to fetch — these are flagged as "user-supplied, drop into public" |
| `/assets/...` or `https://...` | Standard `fetch` + `decodeAudioData`                                     |

If you commit reference WAVs into the repo, place them under
`public/assets/golden/` and reference them as
`/assets/golden/154_gentle_master.wav`. The default registry uses this
convention. If you'd rather keep WAVs out of git, use the panel's "upload
reference" path and **Save golden** — the resulting entry has
`file://<filename>` paths and is informational only (the next CI run will
report a missing asset for that entry, which is the correct behaviour).

## Adding a new golden test

1. Bounce the Logic plugin at the chosen preset to a WAV. Drop the WAV
   into `public/assets/golden/` (committed).
2. Either:
   - **In code**: append a literal entry to `DEFAULT_GOLDEN_TESTS` in
     `src/lib/goldenTests.js` (preferred — it's a static registry).
   - **In UI**: load the source + ref into the panel, dial in the params,
     click "Render", click "Save golden". Then call
     `goldenTestSet.toJSON()` from devtools and persist the result into
     `DEFAULT_GOLDEN_TESTS` to lock it in for CI.
3. Adjust `expected_max_diff_db` if the preset's nonlinearity warrants a
   looser threshold than the auto-`+6` headroom.

## R11 dependency / fallback adapter

The harness expects a `pluginAdapter` exposing this contract:

```ts
adapter.render({
  pluginId:     string,
  sourceBuffer: Float32Array,
  sampleRate:   number,
  params:       { [paramId: string]: number },   // 0..1
}): Promise<Float32Array>;
```

R11's actual `PluginAdapter` (in `src/lib/PluginAdapter.js`) is geared
toward live studio hosting and exposes `instantiate(logicPlugin)` that
returns a slot `{ engine, input, output, setLogicParam, dispose }`. The
panel **auto-bridges** that API into the offline-render contract by
constructing R11 against an `OfflineAudioContext`, feeding the source
buffer into `slot.input`, calling `offCtx.startRendering()`, and reading
channel 0. See `buildR11Bridge` in `ValidationPanel.js`.

The "Adapter mode" indicator in the Plugin section of the panel reads:

| Value          | Meaning                                                        |
|----------------|----------------------------------------------------------------|
| `r11`          | An adapter with a native `.render()` method was found.         |
| `r11-bridge`   | R11's `PluginAdapter` class found; offline-render via bridge.  |
| `fallback`     | No R11 — uses `simpleCompressor` mock (acceptance only).       |

The fallback (`fallbackPluginAdapter` in `src/lib/goldenTests.js`) is
intentionally not faithful to Logic — it exists so the end-to-end harness
compiles and so a real Logic bounce loaded into the panel produces a
clearly non-zero diff (i.e. you can confirm the panel is wired up).

**Note on goldens-on-CI**: the headless runner script needs to bridge
R11 the same way the panel does. The `OfflineAudioContext` API is
available in `node-web-audio-api` ≥ 0.20 — pass an instance through
when constructing R11's `PluginAdapter({ ctx: offlineCtx })` and the
same flow works server-side.

## CI hook

There is no Vercel/GitHub Actions CI step yet — Doseedo currently relies on
`vercel build` for static checks and on manual smoke-testing for runtime
behaviour. To wire goldens into CI when one exists, use the same
`runAll` API headlessly (Node 20 + `node-web-audio-api`, or a Puppeteer
driver against `/dev/validation`):

```js
// scripts/run-goldens.mjs (suggested location for the CI hook)
import { GoldenTestSet, fallbackPluginAdapter } from '../src/lib/goldenTests.js';
import { promises as fs } from 'fs';

const ctx = new (await import('node-web-audio-api')).AudioContext();
const adapter = await loadRealOrFallbackAdapter(); // see comment block
const set = new GoldenTestSet();
const results = await set.runAll(ctx, adapter, {
  resolveBuffer: async (url) => {
    if (url.startsWith('/assets/')) {
      const ab = await fs.readFile(`public${url}`);
      const buf = await ctx.decodeAudioData(ab.buffer);
      return buf.getChannelData(0).slice();
    }
    throw new Error(`unhandled url: ${url}`);
  },
});

const failed = results.filter((r) => !r.pass || r.error);
if (failed.length > 0) {
  console.error(JSON.stringify(failed, null, 2));
  process.exit(1);
}
```

Plug that script into a new GitHub Actions workflow (or
`vercel.json` `buildCommand`) once the project gains a CI runner. The
runner-side resolver is intentionally the only piece that needs to be
swapped — the registry, the runner, and the metrics are all environment-
agnostic.

## Acceptance checklist

- [x] `dspMetrics.js` returns `0` (i.e. `DB_FLOOR`) for `rmsDiffDb`/
  `peakDiffDb` when ref == web; `DB_FLOOR` when both are silent;
  finite values otherwise.
- [x] `goldenTests.runAll(...)` runs against either R11 PluginAdapter or
  the fallback `simpleCompressor` adapter — `adapterKind` pill in the UI
  reports which.
- [x] `ValidationPanel` mounts client-only and renders without runtime
  errors when no mapping/ref/source is loaded.
- [x] `/dev/validation` page registers under the App Router root.
- [x] No existing components were modified.
