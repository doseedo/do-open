/**
 * goldenTests.js — registry + runner for plugin-fidelity golden tests.
 *
 * A "golden test" pairs a Logic-bounced reference WAV against the output of
 * the Doseedo web DSP runtime when fed the same source signal at the same
 * parameter values. The runner returns a pass/fail per test based on an
 * RMS null-diff threshold (lower dB == closer to bit-identical).
 *
 * Default registry: 5 tests covering Logic Compressor (plugin_id "154") at
 * five common factory presets. Each entry only carries metadata + paths;
 * the actual audio assets are NOT fetched at module import time. The user
 * supplies them at run time via the ValidationPanel UI (or a CI hook
 * passes them in directly to runAll()).
 *
 *   import goldenTestSet, { GoldenTestSet } from '@/lib/goldenTests';
 *
 *   const results = await goldenTestSet.runAll(audioContext, pluginAdapter, {
 *     resolveBuffer: async (url) => audioCtx.decodeAudioData(...),
 *   });
 *
 * The runner prefers a real `pluginAdapter` (R11) when one is provided.
 * If R11 hasn't shipped yet, callers may pass a simple shim object of the
 * shape `{ render: async (sourceArr, sampleRate, pluginId, params) => Float32Array }`
 * — see `simpleCompressor` at the bottom of this file for a usable mock.
 *
 * Schema (one entry):
 *
 *   {
 *     id:                  string,        // unique within registry
 *     plugin_id:           string,        // matches /plugin-mappings/{id}.json
 *     preset_name:         string,        // human-readable
 *     params:              { paramId: 0..1, … },
 *     source_audio_url:    string,        // user-supplied path
 *     ref_bounce_url:      string,        // user-supplied path
 *     expected_max_diff_db: number,       // pass threshold (e.g. -40 dB)
 *     notes?:              string,
 *   }
 */

import { rmsDiffDb, peakDiffDb, alignBuffers } from './dspMetrics';

// ── Default registry ──────────────────────────────────────────────────────
// Compressor presets — the mappings live under /plugin-mappings/154.json
// (plugin_id "154" was R11's first calibration target). Asset URLs point
// to /assets/golden/... — the user drops the WAVs there manually; this
// file never tries to fetch them at import time.

export const DEFAULT_GOLDEN_TESTS = [
  {
    id: 'compressor.gentle_master',
    plugin_id: '154',
    preset_name: 'Gentle Master Bus (4:1, 5ms / 100ms, -12 dB threshold)',
    params: { threshold: 0.4, ratio: 0.30, attack: 0.10, release: 0.40, makeup: 0.30 },
    source_audio_url: '/assets/golden/source_drum_loop.wav',
    ref_bounce_url:   '/assets/golden/154_gentle_master.wav',
    expected_max_diff_db: -36,
    notes: 'VCA-style gentle bus compression. Fast attack, medium release.',
  },
  {
    id: 'compressor.vocal_leveler',
    plugin_id: '154',
    preset_name: 'Vocal Leveler (3:1, 30ms / 200ms, -18 dB threshold)',
    params: { threshold: 0.30, ratio: 0.22, attack: 0.30, release: 0.55, makeup: 0.40 },
    source_audio_url: '/assets/golden/source_vocal_phrase.wav',
    ref_bounce_url:   '/assets/golden/154_vocal_leveler.wav',
    expected_max_diff_db: -34,
    notes: 'Lookahead off, opto-style timing.',
  },
  {
    id: 'compressor.drum_smash',
    plugin_id: '154',
    preset_name: 'Drum Smash (10:1, 1ms / 50ms, -24 dB threshold)',
    params: { threshold: 0.20, ratio: 0.85, attack: 0.02, release: 0.20, makeup: 0.55 },
    source_audio_url: '/assets/golden/source_drum_loop.wav',
    ref_bounce_url:   '/assets/golden/154_drum_smash.wav',
    expected_max_diff_db: -28,
    notes: 'Aggressive parallel-style smash. High GR, fast everything.',
  },
  {
    id: 'compressor.bass_glue',
    plugin_id: '154',
    preset_name: 'Bass Glue (2:1, 15ms / 150ms, -10 dB threshold)',
    params: { threshold: 0.50, ratio: 0.18, attack: 0.20, release: 0.45, makeup: 0.20 },
    source_audio_url: '/assets/golden/source_bass_riff.wav',
    ref_bounce_url:   '/assets/golden/154_bass_glue.wav',
    expected_max_diff_db: -38,
    notes: 'Subtle glue compression on a sustained bass.',
  },
  {
    id: 'compressor.limiter_brick',
    plugin_id: '154',
    preset_name: 'Limiter Brick (20:1, 0.5ms / 30ms, -3 dB threshold)',
    params: { threshold: 0.85, ratio: 0.95, attack: 0.0, release: 0.10, makeup: 0.10 },
    source_audio_url: '/assets/golden/source_drum_loop.wav',
    ref_bounce_url:   '/assets/golden/154_limiter_brick.wav',
    expected_max_diff_db: -24,
    notes: 'Limiter mode — large nonlinearity, expect higher diff floor.',
  },
];

// ── The class ─────────────────────────────────────────────────────────────

export class GoldenTestSet {
  /**
   * @param {Array<object>=} initialTests   defaults to DEFAULT_GOLDEN_TESTS
   */
  constructor(initialTests) {
    this.tests = (initialTests || DEFAULT_GOLDEN_TESTS).map((t) => ({ ...t }));
  }

  add(test) {
    if (!test || !test.id) throw new Error('GoldenTestSet.add: test.id is required');
    if (this.tests.some((t) => t.id === test.id)) {
      throw new Error(`GoldenTestSet.add: duplicate id "${test.id}"`);
    }
    this.tests.push({ ...test });
    return this;
  }

  remove(id) {
    this.tests = this.tests.filter((t) => t.id !== id);
    return this;
  }

  get(id)  { return this.tests.find((t) => t.id === id) || null; }
  list()   { return this.tests.map((t) => ({ ...t })); }
  size()   { return this.tests.length; }

  /**
   * Run every test against the supplied plugin adapter.
   *
   * @param {AudioContext} audioContext
   * @param {object} pluginAdapter
   *   either R11's PluginAdapter (must expose
   *   `render({ pluginId, sourceBuffer, sampleRate, params }) → Float32Array`)
   *   or a shim with the same `render` signature. See `simpleCompressor`.
   * @param {object=} opts
   *   {
   *     resolveBuffer:  async (url) → Float32Array          // required if assets aren't pre-loaded
   *     onProgress:     (i, total, result) → void           // optional UI hook
   *   }
   * @returns {Promise<Array<{
   *   id, plugin_id, preset, pass, diff_db, peak_diff_db, threshold_db, error?
   * }>>}
   */
  async runAll(audioContext, pluginAdapter, opts = {}) {
    if (!pluginAdapter || typeof pluginAdapter.render !== 'function') {
      throw new Error('GoldenTestSet.runAll: pluginAdapter.render is required');
    }
    const sampleRate = audioContext?.sampleRate || 48000;
    const results = [];

    for (let i = 0; i < this.tests.length; i++) {
      const t = this.tests[i];
      const out = {
        id: t.id,
        plugin_id: t.plugin_id,
        preset: t.preset_name,
        pass: false,
        diff_db: null,
        peak_diff_db: null,
        threshold_db: t.expected_max_diff_db,
        error: null,
      };

      try {
        const sourceBuf = await this._resolve(t.source_audio_url, opts);
        const refBuf    = await this._resolve(t.ref_bounce_url, opts);
        const webBuf    = await pluginAdapter.render({
          pluginId: t.plugin_id,
          sourceBuffer: sourceBuf,
          sampleRate,
          params: t.params,
        });
        const [refAligned, webAligned] = alignBuffers(refBuf, webBuf);
        out.diff_db      = rmsDiffDb(refAligned, webAligned);
        out.peak_diff_db = peakDiffDb(refAligned, webAligned);
        out.pass         = out.diff_db <= t.expected_max_diff_db;
      } catch (err) {
        out.error = err && err.message ? err.message : String(err);
      }

      results.push(out);
      if (typeof opts.onProgress === 'function') {
        opts.onProgress(i + 1, this.tests.length, out);
      }
    }

    return results;
  }

  async _resolve(url, opts) {
    if (opts && typeof opts.resolveBuffer === 'function') {
      const v = await opts.resolveBuffer(url);
      if (!v) throw new Error(`resolveBuffer returned no data for ${url}`);
      return v;
    }
    throw new Error(
      `goldenTests: no resolveBuffer provided — caller must supply a function ` +
      `that turns "${url}" into a Float32Array (this module does not fetch).`
    );
  }

  /** Serialize the registry — handy for committing into version control. */
  toJSON() {
    return { version: 1, tests: this.list() };
  }

  static fromJSON(json) {
    if (!json || !Array.isArray(json.tests)) {
      throw new Error('GoldenTestSet.fromJSON: expected { tests: [] }');
    }
    return new GoldenTestSet(json.tests);
  }
}

// ── R11 PluginAdapter dependency / mock ──────────────────────────────────
// If R11 has shipped, callers should `import { PluginAdapter } from
// '@/audio/PluginAdapter'`. Until then, the simple compressor below mimics
// the adapter contract well enough that goldens can be wired up and run
// end-to-end. This is intentionally NOT a faithful Logic-compressor model
// — it exists so the harness compiles and the validation UI demonstrates a
// non-zero diff against any real Logic bounce.

/**
 * `simpleCompressor(source: Float32Array, params, sampleRate) → Float32Array`
 * — soft-knee feed-forward compressor. Used as a fallback when no
 * pluginAdapter is provided.
 *
 *   params = {
 *     threshold:  0..1  → mapped to -60..0 dB
 *     ratio:      0..1  → mapped to 1:1 .. 20:1
 *     attack:     0..1  → 0.1 .. 200 ms
 *     release:    0..1  → 5 .. 1000 ms
 *     makeup:     0..1  → 0..18 dB
 *   }
 */
export function simpleCompressor(source, params = {}, sampleRate = 48000) {
  const src = source instanceof Float32Array ? source : Float32Array.from(source || []);
  const out = new Float32Array(src.length);
  const thrDb = -60 + (params.threshold ?? 0.5) * 60;       // -60..0
  const ratio = 1 + (params.ratio ?? 0.5) * 19;             // 1..20
  const attMs = 0.1 + (params.attack ?? 0.1) * 199.9;
  const relMs = 5   + (params.release ?? 0.4) * 995;
  const makeupDb = (params.makeup ?? 0) * 18;
  const attCoef = Math.exp(-1 / (sampleRate * (attMs / 1000)));
  const relCoef = Math.exp(-1 / (sampleRate * (relMs / 1000)));
  const makeupLin = Math.pow(10, makeupDb / 20);
  let env = 0;
  for (let i = 0; i < src.length; i++) {
    const inAbs = Math.abs(src[i]);
    const target = inAbs;
    const coef = target > env ? attCoef : relCoef;
    env = target + (env - target) * coef;
    const envDb = env > 0 ? 20 * Math.log10(env) : -240;
    let gainDb = 0;
    if (envDb > thrDb) gainDb = (thrDb - envDb) * (1 - 1 / ratio);
    out[i] = src[i] * Math.pow(10, gainDb / 20) * makeupLin;
  }
  return out;
}

/** A fallback adapter usable as the second arg to `runAll()`. */
export const fallbackPluginAdapter = {
  render: async ({ pluginId, sourceBuffer, sampleRate, params }) => {
    // Single-plugin mock — pluginId is informational here.
    void pluginId;
    return simpleCompressor(sourceBuffer, params, sampleRate);
  },
};

// ── Default singleton ────────────────────────────────────────────────────

const goldenTestSet = new GoldenTestSet();
export default goldenTestSet;
