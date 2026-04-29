/**
 * ValidationPanel.js — A/B compare a Logic-bounced reference WAV against
 * the web DSP runtime's render of the same signal at the same params.
 *
 * Internal-only tool, mounted at /dev/validation by
 * `app/dev/validation/page.js`. Not linked anywhere in the user-facing app.
 *
 * Layout:
 *
 *   ┌──────────────┬──────────────────────────────────────┐
 *   │  CONTROLS    │  REFERENCE WAVEFORM                  │
 *   │  (320px)     │  WEB-DSP WAVEFORM                    │
 *   │              │  NULL (ref - web)                    │
 *   │  Plugin      │  ──────────────────────────────────  │
 *   │  Source sig  │  METRICS [RMS / Peak / 1/3-oct bars] │
 *   │  Ref upload  │  ──────────────────────────────────  │
 *   │  Param       │  GOLDEN TESTS table                  │
 *   │  sliders     │                                      │
 *   │  [Render]    │                                      │
 *   │  [Save G]    │                                      │
 *   │  [Run All]   │                                      │
 *   └──────────────┴──────────────────────────────────────┘
 *
 * The component does not modify any existing component — it only consumes
 * `dspMetrics` and `goldenTests` from `src/lib/`, and (when present) the
 * R11 PluginAdapter via dynamic import. If R11 hasn't shipped, we fall
 * back to `simpleCompressor` from goldenTests.js.
 */

'use client';

import React, {
  useState, useEffect, useMemo, useRef, useCallback,
} from 'react';
import styles from './ValidationPanel.module.css';
import {
  rmsDb, peakDb, rmsDiffDb, peakDiffDb, nullDiff,
  thirdOctaveSpectralDiff, poolMaxAbs, DB_FLOOR,
} from '../../../lib/dspMetrics';
import goldenTestSet, {
  GoldenTestSet, fallbackPluginAdapter, simpleCompressor,
} from '../../../lib/goldenTests';

// ── Source-signal generators ─────────────────────────────────────────────

function genWhiteNoise(sampleRate, duration = 2) {
  const n = Math.floor(sampleRate * duration);
  const buf = new Float32Array(n);
  for (let i = 0; i < n; i++) buf[i] = (Math.random() * 2 - 1) * 0.5;
  return buf;
}

function genSineSweep(sampleRate, duration = 2, f0 = 80, f1 = 8000) {
  const n = Math.floor(sampleRate * duration);
  const buf = new Float32Array(n);
  let phase = 0;
  for (let i = 0; i < n; i++) {
    const t = i / n;
    const f = f0 * Math.pow(f1 / f0, t);
    phase += (2 * Math.PI * f) / sampleRate;
    buf[i] = Math.sin(phase) * 0.5;
  }
  return buf;
}

function genImpulse(sampleRate, duration = 2) {
  const n = Math.floor(sampleRate * duration);
  const buf = new Float32Array(n);
  buf[0] = 1.0;
  return buf;
}

function genDrumLoop(sampleRate, duration = 2, bpm = 120) {
  const n = Math.floor(sampleRate * duration);
  const buf = new Float32Array(n);
  const beatSamples = (60 / bpm) * sampleRate;
  for (let i = 0; i < n; i++) {
    const beatNum = Math.floor(i / beatSamples) % 4;
    const beatPos = (i % beatSamples) / beatSamples;
    if ((beatNum === 0 || beatNum === 2) && beatPos < 0.08) {
      const env = Math.exp(-beatPos * 50);
      buf[i] += Math.sin(2 * Math.PI * 60 * (beatPos * (60 / bpm))) * env * 0.7;
    }
    if ((beatNum === 1 || beatNum === 3) && beatPos < 0.06) {
      const env = Math.exp(-beatPos * 40);
      buf[i] += (Math.random() * 2 - 1) * env * 0.4;
    }
  }
  return buf;
}

const SOURCE_GENERATORS = {
  noise:   { label: 'White noise',  gen: genWhiteNoise },
  sweep:   { label: 'Sine sweep',   gen: genSineSweep  },
  impulse: { label: 'Impulse',      gen: genImpulse    },
  drums:   { label: 'Drum loop',    gen: genDrumLoop   },
  custom:  { label: 'File upload',  gen: null          },
};

// ── Mapping fetcher ──────────────────────────────────────────────────────

async function fetchPluginMapping(pluginId) {
  if (!pluginId) return null;
  try {
    const r = await fetch(`/plugin-mappings/${encodeURIComponent(pluginId)}.json`, {
      cache: 'no-store',
    });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

async function fetchMappingIndex() {
  try {
    const r = await fetch('/plugin-mappings/index.json', { cache: 'no-store' });
    if (!r.ok) return { mappings: [] };
    return await r.json();
  } catch {
    return { mappings: [] };
  }
}

// ── Adapter resolution (R11 if shipped, fallback otherwise) ─────────────
//
// R11 ships `PluginAdapter` as a class whose entry-point is
// `instantiate(logicPlugin) → { engine, input, output, setLogicParam, ... }`.
// That contract is geared toward LIVE-graph hosting (the studio mounts the
// returned slot into an active AudioContext). The validation harness
// instead needs offline `(source, params) → Float32Array` rendering, so
// here we bridge: build the slot inside an `OfflineAudioContext`, drive
// it with a buffer source, render, and grab channel 0.

async function buildR11Bridge() {
  let mod = null;
  try {
    mod = await import('../../../lib/PluginAdapter.js');
  } catch {
    return null;
  }
  const Cls = mod && (mod.default || mod.PluginAdapter);
  if (!Cls || typeof Cls !== 'function') return null;

  return {
    render: async ({ pluginId, sourceBuffer, sampleRate, params }) => {
      const src = sourceBuffer instanceof Float32Array
        ? sourceBuffer
        : Float32Array.from(sourceBuffer || []);
      const dur = Math.max(1, src.length);
      const Offline = (typeof window !== 'undefined') &&
        (window.OfflineAudioContext || window.webkitOfflineAudioContext);
      if (!Offline) throw new Error('OfflineAudioContext unavailable');

      const offCtx = new Offline(1, dur, sampleRate);
      const adapter = new Cls({ ctx: offCtx });

      // R11 caches mappings by plugin_id and lazily fetches if needed.
      const slot = await adapter.instantiate({
        plugin_id: pluginId,
        plugin_name: pluginId,
        parameters: Object.entries(params || {}).map(([id, value]) => ({
          id, name: id, value,
        })),
      });

      // No mapping — caller should fall back. Returning a zero buffer
      // would silently mask this; throwing is the correct behaviour.
      if (!slot) {
        throw new Error(`R11: no mapping for plugin_id=${pluginId}`);
      }

      // Feed a buffer source into the slot.
      const sBuf = offCtx.createBuffer(1, src.length, sampleRate);
      sBuf.getChannelData(0).set(src);
      const sNode = offCtx.createBufferSource();
      sNode.buffer = sBuf;
      sNode.connect(slot.input);
      slot.output.connect(offCtx.destination);
      sNode.start(0);

      const rendered = await offCtx.startRendering();
      try { slot.dispose(); } catch { /* noop */ }
      return rendered.getChannelData(0).slice();
    },
  };
}

async function resolvePluginAdapter() {
  const bridge = await buildR11Bridge();
  if (bridge) return { adapter: bridge, kind: 'r11-bridge' };
  return { adapter: fallbackPluginAdapter, kind: 'fallback' };
}

// ── Decode helpers ───────────────────────────────────────────────────────

async function decodeFile(file, audioCtx) {
  const ab = await file.arrayBuffer();
  const buf = await audioCtx.decodeAudioData(ab);
  // Take channel 0 — diff metrics are mono.
  return buf.getChannelData(0).slice();
}

async function fetchAndDecode(url, audioCtx) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  const ab = await r.arrayBuffer();
  const buf = await audioCtx.decodeAudioData(ab);
  return buf.getChannelData(0).slice();
}

// ── Canvas helpers ───────────────────────────────────────────────────────

function drawWaveform(canvas, samples, color = '#4f86d6') {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = '#0a0c10';
  ctx.fillRect(0, 0, w, h);
  if (!samples || samples.length === 0) {
    ctx.fillStyle = '#5b6271';
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('— no data —', w / 2, h / 2);
    return;
  }
  // Centerline
  ctx.strokeStyle = '#1d2128';
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();

  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < samples.length; i++) {
    const x = (i / (samples.length - 1)) * w;
    const v = Math.max(-1, Math.min(1, samples[i]));
    const y = (1 - (v + 1) / 2) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawSpectrumBars(canvas, bandData) {
  // bandData: [{ freq, diffDb }]
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = '#0a0c10';
  ctx.fillRect(0, 0, w, h);
  if (!bandData || bandData.length === 0) return;
  const minDb = -80, maxDb = 0;
  const barW = w / bandData.length;
  ctx.fillStyle = '#4f86d6';
  bandData.forEach((b, i) => {
    const clamped = Math.max(minDb, Math.min(maxDb, b.diffDb));
    const norm = (clamped - minDb) / (maxDb - minDb);
    const barH = norm * h;
    ctx.fillRect(i * barW + 1, h - barH, barW - 2, barH);
  });
}

// ── Main component ───────────────────────────────────────────────────────

const ValidationPanel = () => {
  // Audio context (lazy)
  const audioCtxRef = useRef(null);
  const ensureCtx = useCallback(() => {
    if (!audioCtxRef.current) {
      const Ctx = (typeof window !== 'undefined') &&
        (window.AudioContext || window.webkitAudioContext);
      if (!Ctx) return null;
      audioCtxRef.current = new Ctx();
    }
    if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume().catch(() => {});
    }
    return audioCtxRef.current;
  }, []);

  // Mapping registry
  const [mappingIndex, setMappingIndex] = useState({ mappings: [] });
  const [pluginId, setPluginId] = useState('154');
  const [mapping, setMapping] = useState(null);
  const [paramValues, setParamValues] = useState({});

  // Source signal
  const [sourceKind, setSourceKind] = useState('drums');
  const [customSource, setCustomSource] = useState(null);   // Float32Array
  const [customSourceName, setCustomSourceName] = useState('');

  // Reference audio
  const [refBuffer, setRefBuffer] = useState(null);
  const [refName, setRefName] = useState('');

  // Render output
  const [webBuffer, setWebBuffer] = useState(null);
  const [diffBuffer, setDiffBuffer] = useState(null);
  const [spectralBands, setSpectralBands] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [adapterKind, setAdapterKind] = useState('unknown');
  const [busy, setBusy] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  // Goldens runner
  const goldensRef = useRef(goldenTestSet);
  const [goldenResults, setGoldenResults] = useState([]);
  const [runningGoldens, setRunningGoldens] = useState(false);

  // Canvas refs
  const refWaveCanvas = useRef(null);
  const webWaveCanvas = useRef(null);
  const diffWaveCanvas = useRef(null);
  const spectrumCanvas = useRef(null);

  // ── Bootstrap mapping list ─────────────────────────────────────────────
  useEffect(() => {
    fetchMappingIndex().then(setMappingIndex);
  }, []);

  // ── Load selected mapping ──────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    fetchPluginMapping(pluginId).then((m) => {
      if (cancelled) return;
      setMapping(m);
      // Seed params at mapping defaults — fall back to 0.5 for any param.
      const next = {};
      const params = (m && m.parameters) || (m && m.params) || [];
      for (const p of params) {
        const norm = (p.default != null && p.min != null && p.max != null)
          ? (p.default - p.min) / (p.max - p.min) : 0.5;
        next[p.id || p.name] = norm;
      }
      setParamValues(next);
    });
    return () => { cancelled = true; };
  }, [pluginId]);

  // ── Resolve adapter once on mount ──────────────────────────────────────
  const adapterRef = useRef(null);
  useEffect(() => {
    let cancelled = false;
    resolvePluginAdapter().then(({ adapter, kind }) => {
      if (cancelled) return;
      adapterRef.current = adapter;
      setAdapterKind(kind);
    });
    return () => { cancelled = true; };
  }, []);

  // ── Source-buffer generator ────────────────────────────────────────────
  const buildSourceBuffer = useCallback(() => {
    const ctx = ensureCtx();
    const sr = ctx?.sampleRate || 48000;
    if (sourceKind === 'custom') {
      return customSource;
    }
    const def = SOURCE_GENERATORS[sourceKind];
    if (!def || !def.gen) return null;
    return def.gen(sr, 2);
  }, [sourceKind, customSource, ensureCtx]);

  // ── Param control ──────────────────────────────────────────────────────
  const handleParamChange = useCallback((pid, val) => {
    setParamValues((s) => ({ ...s, [pid]: val }));
  }, []);

  // ── Reference upload ───────────────────────────────────────────────────
  const handleRefUpload = useCallback(async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    setErrorMsg('');
    const ctx = ensureCtx();
    if (!ctx) { setErrorMsg('No AudioContext available'); return; }
    try {
      const data = await decodeFile(file, ctx);
      setRefBuffer(data);
      setRefName(file.name);
    } catch (err) {
      setErrorMsg(`Failed to decode reference: ${err.message || err}`);
    }
  }, [ensureCtx]);

  // ── Custom source upload ───────────────────────────────────────────────
  const handleSourceUpload = useCallback(async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    setErrorMsg('');
    const ctx = ensureCtx();
    if (!ctx) { setErrorMsg('No AudioContext available'); return; }
    try {
      const data = await decodeFile(file, ctx);
      setCustomSource(data);
      setCustomSourceName(file.name);
      setSourceKind('custom');
    } catch (err) {
      setErrorMsg(`Failed to decode source: ${err.message || err}`);
    }
  }, [ensureCtx]);

  // ── Render web-side ────────────────────────────────────────────────────
  const handleRender = useCallback(async () => {
    setErrorMsg('');
    setBusy(true);
    try {
      const ctx = ensureCtx();
      const sr = ctx?.sampleRate || 48000;
      const src = buildSourceBuffer();
      if (!src) throw new Error('No source signal — pick a generator or upload a file.');

      const adapter = adapterRef.current || fallbackPluginAdapter;
      const out = await adapter.render({
        pluginId, sourceBuffer: src, sampleRate: sr, params: paramValues,
      });
      const webOut = out instanceof Float32Array ? out : Float32Array.from(out || []);
      setWebBuffer(webOut);

      // Compute metrics + diff vs reference (if a ref is loaded).
      if (refBuffer && refBuffer.length > 0) {
        const diff = nullDiff(refBuffer, webOut);
        setDiffBuffer(diff);
        setMetrics({
          refRmsDb:    rmsDb(refBuffer),
          webRmsDb:    rmsDb(webOut),
          rmsDiffDb:   rmsDiffDb(refBuffer, webOut),
          peakDiffDb:  peakDiffDb(refBuffer, webOut),
        });
        setSpectralBands(thirdOctaveSpectralDiff(refBuffer, webOut, sr));
      } else {
        setDiffBuffer(null);
        setMetrics({
          refRmsDb: null,
          webRmsDb: rmsDb(webOut),
          rmsDiffDb: null,
          peakDiffDb: null,
        });
        setSpectralBands(null);
      }
    } catch (err) {
      setErrorMsg(`Render failed: ${err.message || err}`);
    } finally {
      setBusy(false);
    }
  }, [pluginId, paramValues, refBuffer, buildSourceBuffer, ensureCtx]);

  // ── Save golden ────────────────────────────────────────────────────────
  const handleSaveGolden = useCallback(() => {
    if (!metrics || metrics.rmsDiffDb == null) {
      setErrorMsg('Cannot save golden — render against a reference first.');
      return;
    }
    const id = `${pluginId}.${Date.now()}`;
    try {
      goldensRef.current.add({
        id,
        plugin_id: pluginId,
        preset_name: `User-saved ${new Date().toISOString()}`,
        params: { ...paramValues },
        source_audio_url: customSourceName ? `file://${customSourceName}` : `gen://${sourceKind}`,
        ref_bounce_url: refName ? `file://${refName}` : 'unknown',
        // Add 6 dB of headroom over the observed diff so transient noise
        // doesn't trip false failures.
        expected_max_diff_db: Math.ceil(metrics.rmsDiffDb + 6),
        notes: 'Saved from ValidationPanel',
      });
      setGoldenResults((r) => r.concat({
        id, plugin_id: pluginId, preset: 'Saved', pass: true,
        diff_db: metrics.rmsDiffDb, peak_diff_db: metrics.peakDiffDb,
        threshold_db: Math.ceil(metrics.rmsDiffDb + 6), error: null,
      }));
    } catch (err) {
      setErrorMsg(`Save golden failed: ${err.message || err}`);
    }
  }, [pluginId, paramValues, metrics, refName, customSourceName, sourceKind]);

  // ── Run all goldens ────────────────────────────────────────────────────
  const handleRunGoldens = useCallback(async () => {
    setRunningGoldens(true);
    setErrorMsg('');
    try {
      const ctx = ensureCtx();
      const adapter = adapterRef.current || fallbackPluginAdapter;
      const results = await goldensRef.current.runAll(ctx, adapter, {
        resolveBuffer: async (url) => {
          // Heuristic: file:// = no auto-fetch (those are user-supplied paths
          // outside the public/ tree), gen:// = synthesize, http(s) or
          // /assets path = fetch + decode.
          if (url.startsWith('gen://')) {
            const kind = url.slice('gen://'.length);
            const def = SOURCE_GENERATORS[kind];
            if (def && def.gen) return def.gen(ctx.sampleRate, 2);
            throw new Error(`unknown gen:// kind ${kind}`);
          }
          if (url.startsWith('file://')) {
            throw new Error(`file:// asset (${url}) — drop into public/assets/golden/ and rename ref_bounce_url`);
          }
          return fetchAndDecode(url, ctx);
        },
        onProgress: (i, total, result) => {
          setGoldenResults((existing) => {
            const map = new Map(existing.map((e) => [e.id, e]));
            map.set(result.id, result);
            return Array.from(map.values());
          });
        },
      });
      setGoldenResults(results);
    } catch (err) {
      setErrorMsg(`Goldens failed: ${err.message || err}`);
    } finally {
      setRunningGoldens(false);
    }
  }, [ensureCtx]);

  // ── Canvas painters ────────────────────────────────────────────────────
  useEffect(() => {
    if (refBuffer) drawWaveform(refWaveCanvas.current, poolMaxAbs(refBuffer, 2000), '#5fcf75');
    else drawWaveform(refWaveCanvas.current, null);
  }, [refBuffer]);

  useEffect(() => {
    if (webBuffer) drawWaveform(webWaveCanvas.current, poolMaxAbs(webBuffer, 2000), '#4f86d6');
    else drawWaveform(webWaveCanvas.current, null);
  }, [webBuffer]);

  useEffect(() => {
    if (diffBuffer) drawWaveform(diffWaveCanvas.current, poolMaxAbs(diffBuffer, 2000), '#d65f5f');
    else drawWaveform(diffWaveCanvas.current, null);
  }, [diffBuffer]);

  useEffect(() => {
    drawSpectrumBars(spectrumCanvas.current, spectralBands);
  }, [spectralBands]);

  // ── Param list resolution ──────────────────────────────────────────────
  const paramList = useMemo(() => {
    if (!mapping) return [];
    const ps = mapping.parameters || mapping.params || [];
    return ps.map((p) => ({
      id:   p.id || p.name,
      name: p.name || p.id,
      min:  p.min ?? 0,
      max:  p.max ?? 1,
      unit: p.unit || '',
    }));
  }, [mapping]);

  // ── Metric helpers ─────────────────────────────────────────────────────
  const fmtDb = (db) => {
    if (db == null) return '—';
    if (!isFinite(db) || db <= DB_FLOOR + 1) return '−∞';
    return `${db.toFixed(2)} dB`;
  };
  const diffClass = (db) => {
    if (db == null) return '';
    if (db <= -40) return styles.metricGood;
    if (db <= -20) return styles.metricWarn;
    return styles.metricBad;
  };

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div className={styles.panel}>
      {/* Left column: controls */}
      <div className={styles.column}>
        <div className={styles.section}>
          <h2 className={styles.h2}>Plugin</h2>
          <select
            className={styles.select}
            value={pluginId}
            onChange={(e) => setPluginId(e.target.value)}
          >
            {(mappingIndex.mappings || []).map((id) => (
              <option key={id} value={id}>
                {id} {mapping && mapping.plugin_id === id && mapping.name ? `— ${mapping.name}` : ''}
              </option>
            ))}
            {!(mappingIndex.mappings || []).includes(pluginId) && (
              <option value={pluginId}>{pluginId} (manual)</option>
            )}
          </select>
          {!mapping && (
            <div className={styles.help}>
              No <code>/plugin-mappings/{pluginId}.json</code> found —
              parameter sliders below will be empty until R10/R11 ships one.
              Adapter mode: <strong>{adapterKind}</strong>.
            </div>
          )}
          {mapping && mapping.name && (
            <div className={styles.help}>
              {mapping.name} — adapter: <strong>{adapterKind}</strong>
            </div>
          )}
        </div>

        <div className={styles.section}>
          <h2 className={styles.h2}>Source signal</h2>
          <select
            className={styles.select}
            value={sourceKind}
            onChange={(e) => setSourceKind(e.target.value)}
          >
            {Object.entries(SOURCE_GENERATORS).map(([k, v]) => (
              <option key={k} value={k}>{v.label}</option>
            ))}
          </select>
          {sourceKind === 'custom' && (
            <div className={styles.row} style={{ marginTop: 8 }}>
              <label className={styles.fileBtn}>
                Upload source
                <input type="file" accept="audio/*" onChange={handleSourceUpload} />
              </label>
              <span className={styles.fileName}>
                {customSourceName || 'no file'}
              </span>
            </div>
          )}
        </div>

        <div className={styles.section}>
          <h2 className={styles.h2}>Reference (Logic bounce)</h2>
          <div className={styles.row}>
            <label className={styles.fileBtn}>
              Upload .wav
              <input type="file" accept="audio/wav,audio/*" onChange={handleRefUpload} />
            </label>
            <span className={styles.fileName}>{refName || 'no file'}</span>
          </div>
        </div>

        <div className={styles.section}>
          <h2 className={styles.h2}>Parameters</h2>
          {paramList.length === 0 && (
            <div className={styles.help}>No params for this mapping.</div>
          )}
          {paramList.map((p) => {
            const val = paramValues[p.id] ?? 0.5;
            const scaled = p.min + (p.max - p.min) * val;
            return (
              <div key={p.id} className={styles.paramRow}>
                <span className={styles.paramLabel}>{p.name}</span>
                <input
                  type="range"
                  className={styles.slider}
                  min={0} max={1} step={0.001} value={val}
                  onChange={(e) => handleParamChange(p.id, parseFloat(e.target.value))}
                />
                <span className={styles.paramValue}>
                  {scaled.toFixed(2)}{p.unit ? ` ${p.unit}` : ''}
                </span>
              </div>
            );
          })}
        </div>

        <div className={styles.section}>
          <h2 className={styles.h2}>Actions</h2>
          <div className={styles.row}>
            <button
              className={styles.btn}
              onClick={handleRender}
              disabled={busy}
            >
              {busy ? 'Rendering…' : 'Render web side'}
            </button>
          </div>
          <div className={styles.row}>
            <button
              className={`${styles.btn} ${styles.btnGhost}`}
              onClick={handleSaveGolden}
              disabled={!metrics || metrics.rmsDiffDb == null}
            >
              Save golden
            </button>
          </div>
          <div className={styles.row}>
            <button
              className={`${styles.btn} ${styles.btnGhost}`}
              onClick={handleRunGoldens}
              disabled={runningGoldens}
            >
              {runningGoldens ? 'Running…' : `Run all goldens (${goldensRef.current.size()})`}
            </button>
          </div>
          {errorMsg && (
            <div className={`${styles.help} ${styles.statusFail}`}>{errorMsg}</div>
          )}
        </div>
      </div>

      {/* Right column: visuals + metrics + goldens table */}
      <div className={styles.column}>
        <div className={styles.section}>
          <h2 className={styles.h2}>Waveforms</h2>
          <div className={styles.canvasWrap} style={{ marginBottom: 6 }}>
            <span className={styles.canvasLabel}>REFERENCE</span>
            <canvas ref={refWaveCanvas} width={900} height={120} />
          </div>
          <div className={styles.canvasWrap} style={{ marginBottom: 6 }}>
            <span className={styles.canvasLabel}>WEB DSP</span>
            <canvas ref={webWaveCanvas} width={900} height={120} />
          </div>
          <div className={styles.canvasWrap}>
            <span className={styles.canvasLabel}>NULL (REF − WEB)</span>
            <canvas ref={diffWaveCanvas} width={900} height={120} />
          </div>
        </div>

        <div className={styles.section}>
          <h2 className={styles.h2}>Metrics</h2>
          <div className={styles.metricsGrid}>
            <div className={styles.metricCard}>
              <div className={styles.metricLabel}>Ref RMS</div>
              <div className={styles.metricValue}>{fmtDb(metrics?.refRmsDb)}</div>
            </div>
            <div className={styles.metricCard}>
              <div className={styles.metricLabel}>Web RMS</div>
              <div className={styles.metricValue}>{fmtDb(metrics?.webRmsDb)}</div>
            </div>
            <div className={styles.metricCard}>
              <div className={styles.metricLabel}>RMS null-diff</div>
              <div className={`${styles.metricValue} ${diffClass(metrics?.rmsDiffDb)}`}>
                {fmtDb(metrics?.rmsDiffDb)}
              </div>
            </div>
            <div className={styles.metricCard}>
              <div className={styles.metricLabel}>Peak null-diff</div>
              <div className={`${styles.metricValue} ${diffClass(metrics?.peakDiffDb)}`}>
                {fmtDb(metrics?.peakDiffDb)}
              </div>
            </div>
          </div>
          <div style={{ marginTop: 10 }}>
            <h2 className={styles.h2}>1/3-octave spectral diff</h2>
            <div className={styles.canvasWrap}>
              <canvas ref={spectrumCanvas} width={900} height={80} />
            </div>
          </div>
        </div>

        <div className={styles.section}>
          <h2 className={styles.h2}>Goldens</h2>
          {goldenResults.length === 0 && (
            <div className={styles.help}>
              No results yet. Click "Run all goldens" to execute the
              registry against the resolved adapter ({adapterKind}).
              Ref/source assets must live under{' '}
              <code>/public/assets/golden/</code> for fetch-based tests.
            </div>
          )}
          {goldenResults.length > 0 && (
            <table className={styles.goldenTable}>
              <thead>
                <tr>
                  <th>id</th>
                  <th>preset</th>
                  <th>RMS Δ dB</th>
                  <th>peak Δ dB</th>
                  <th>thr dB</th>
                  <th>status</th>
                </tr>
              </thead>
              <tbody>
                {goldenResults.map((r) => (
                  <tr key={r.id}>
                    <td>{r.id}</td>
                    <td>{r.preset}</td>
                    <td>{fmtDb(r.diff_db)}</td>
                    <td>{fmtDb(r.peak_diff_db)}</td>
                    <td>{r.threshold_db}</td>
                    <td className={r.error ? styles.statusErr : (r.pass ? styles.statusPass : styles.statusFail)}>
                      {r.error ? `ERR (${r.error})` : (r.pass ? 'PASS' : 'FAIL')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
};

export default ValidationPanel;
