import React, { useEffect, useRef, useState } from 'react';

/**
 * Verify page — served at /verify.
 *
 * Public utility: drop an audio file, get back the on-chain attestation
 * record for the generation that produced it. Backed by an inaudible
 * watermark embedded at gen-time and a Polygon attestation referenced by
 * the watermark seed.
 *
 * Aesthetic matches /plans (paper/workbench, Inter / JetBrains Mono / Lora
 * on warm off-white). App.js wraps with <LeftSidebar/> so we only render
 * the content region.
 *
 * No detector or chain lookup wired yet — the verifyFile() call is a stub
 * that returns mocked positive/negative/error results so the UI can be
 * exercised end-to-end. When the real detector ships, replace the body of
 * verifyFile() with a fetch to /api/verify (auth Cloud Run service).
 */

const C = {
  bg: '#e8e6e1',
  surface: '#f2f0ea',
  surface2: '#dcd9d1',
  ink: '#15181c',
  inkSoft: 'rgba(21,24,28,0.66)',
  inkMute: 'rgba(21,24,28,0.40)',
  inkFaint: 'rgba(21,24,28,0.22)',
  rule: 'rgba(21,24,28,0.14)',
  ruleStrong: 'rgba(21,24,28,0.30)',
  accent: '#1d4c7a',
  warm: '#c94f2c',
  good: '#2f7d4f',
  purple: '#AAB0EE',
  sans: '"Inter",system-ui,sans-serif',
  mono: '"JetBrains Mono",ui-monospace,Menlo,monospace',
  head: '"Lora",Georgia,serif',
};

const arrowPath = 'M5 12h14 M13 6l6 6-6 6';
const Arrow = ({ size = 13, stroke = 1.8, color = 'currentColor' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
    <path d={arrowPath} />
  </svg>
);

function Topbar() {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 14,
        padding: '0 36px',
        height: 48,
        borderBottom: `1px solid ${C.rule}`,
        background: C.surface,
        fontFamily: C.mono,
        fontSize: 10,
        letterSpacing: 0.6,
        textTransform: 'uppercase',
        color: C.inkMute,
        flexWrap: 'wrap',
      }}
    >
      <span>Dashboard</span>
      <span style={{ color: C.inkFaint }}>/</span>
      <span style={{ color: C.inkSoft }}>Provenance</span>
      <span style={{ color: C.inkFaint }}>/</span>
      <span>
        <strong style={{ color: C.inkSoft, fontWeight: 500 }}>Verify</strong>
      </span>
      <span style={{ color: C.inkFaint }}>·</span>
      <span>watermark · polygon attestation</span>
      <div style={{ flex: 1 }} />
      <span>
        registry · <strong style={{ color: C.inkSoft, fontWeight: 500 }}>polygon</strong>
      </span>
    </div>
  );
}

function SectHead({ title, count, right }) {
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 14, marginBottom: 18, flexWrap: 'wrap' }}>
      <h2 style={{ fontFamily: C.head, fontSize: 20, fontWeight: 600, letterSpacing: -0.3, margin: 0 }}>{title}</h2>
      {count && (
        <span style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase', color: C.inkMute }}>
          {count}
        </span>
      )}
      <div style={{ flex: 1 }} />
      {right}
    </div>
  );
}

function Hero() {
  return (
    <section style={{ marginBottom: 40, paddingBottom: 28, borderBottom: `1px solid ${C.rule}` }}>
      <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.8, textTransform: 'uppercase', color: C.inkMute, marginBottom: 12 }}>
        § Verify · Provenance lookup
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1.3fr) minmax(0,1fr)', gap: 60, alignItems: 'end' }}>
        <h1 className="page-title">
          Did <span style={{ color: C.accent }}>doseedo</span> make this?
        </h1>
        <div style={{ fontFamily: C.sans, fontSize: 13, color: C.inkSoft, lineHeight: 1.6, paddingBottom: 4, maxWidth: 420 }}>
          Drop an audio file. We’ll listen for the watermark we embed at generation time, look up the matching attestation on Polygon, and tell you when, by whom, and with which model — or that we’ve never seen it before.
        </div>
      </div>
    </section>
  );
}

// Server-backed verifier — uploads to /api/verify which proxies to the
// Modal watermark detector and resolves the seed against the Neon
// attestation registry. See:
//   doseedo-next/app/api/verify/route.ts
//   modal/modal_watermark.py
//   contracts/DoseedoAttestationRegistry.sol
async function verifyFile(file) {
  const form = new FormData();
  form.append('file', file);
  const r = await fetch('/api/verify', { method: 'POST', body: form });
  if (r.status === 429) {
    throw new Error('Rate limit hit — wait a minute and try again');
  }
  if (r.status === 413) {
    throw new Error('File too large — 100 MB max');
  }
  if (!r.ok) {
    let detail = '';
    try {
      const j = await r.json();
      detail = j && j.error ? ` — ${j.error}` : '';
    } catch (_) { /* ignore */ }
    throw new Error(`Detector unreachable (status ${r.status})${detail}`);
  }
  return r.json();
}

function formatBytes(n) {
  if (!n) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function DropZone({ onFile, busy }) {
  const inputRef = useRef(null);
  const [over, setOver] = useState(false);

  const onDrop = (e) => {
    e.preventDefault();
    setOver(false);
    if (busy) return;
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) onFile(file);
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); if (!busy) setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      onClick={() => !busy && inputRef.current && inputRef.current.click()}
      style={{
        border: `1px dashed ${over ? C.accent : C.ruleStrong}`,
        background: over ? 'rgba(29,76,122,.05)' : C.surface,
        padding: '40px 32px',
        cursor: busy ? 'progress' : 'pointer',
        textAlign: 'center',
        transition: 'background .12s, border-color .12s',
      }}
    >
      <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.8, textTransform: 'uppercase', color: C.inkMute, marginBottom: 10 }}>
        DSD-VERIFY · Drop or browse
      </div>
      <div style={{ fontFamily: C.head, fontSize: 22, fontWeight: 600, letterSpacing: -0.3, marginBottom: 6 }}>
        Drop an audio file to verify
      </div>
      <div style={{ fontFamily: C.sans, fontSize: 12, color: C.inkSoft, lineHeight: 1.5, maxWidth: 520, margin: '0 auto' }}>
        WAV, MP3, FLAC, Opus, Ogg up to 100 MB. Your file is uploaded to the doseedo detector, scanned for the watermark, and discarded the moment the scan completes — nothing is retained. If a watermark is found we resolve the match against the Polygon attestation registry and surface the record.
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,.wav,.mp3,.flac,.opus,.ogg,.m4a"
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files && e.target.files[0];
          if (f) onFile(f);
          e.target.value = '';
        }}
      />
    </div>
  );
}

function ResultPanel({ state, file, result, error, onReset }) {
  if (state === 'idle') return null;

  const headColor =
    state === 'verified' ? C.good :
    state === 'not_found' ? C.inkMute :
    state === 'error' ? C.warm :
    C.inkSoft;

  const headLabel =
    state === 'scanning' ? 'Scanning…' :
    state === 'verified' ? 'Verified · doseedo creation' :
    state === 'not_found' ? 'No doseedo watermark detected' :
    state === 'error' ? 'Verification failed' :
    '';

  const rows = [];
  if (state === 'verified' && result) {
    rows.push(['Generation ID', result.generationId]);
    rows.push(['Generated at', new Date(result.generatedAt).toUTCString()]);
    rows.push(['Model', result.model]);
    rows.push(['Tier at gen-time', result.tier]);
    rows.push(['Attribution', result.attribution]);
    rows.push(['Wallet', result.wallet || '— not bound']);
    rows.push(['Attestation hash', result.attestationHash]);
    rows.push(['Polygon tx', result.attestationTx, result.polygonScanUrl]);
    rows.push(['Watermark confidence', `${(result.watermarkConfidence * 100).toFixed(1)}%`]);
  }

  return (
    <section style={{ marginBottom: 40 }}>
      <div style={{ background: C.surface, border: `1px solid ${C.rule}` }}>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0,1fr) auto',
            gap: 16,
            padding: '14px 18px',
            background: C.surface2,
            borderBottom: `1px solid ${C.rule}`,
            alignItems: 'center',
          }}
        >
          <div>
            <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.7, textTransform: 'uppercase', color: headColor, marginBottom: 2 }}>
              {headLabel}
            </div>
            <div style={{ fontFamily: C.sans, fontSize: 13, color: C.ink, fontWeight: 500 }}>
              {file ? file.name : '—'}
              <span style={{ fontFamily: C.mono, fontSize: 11, color: C.inkMute, marginLeft: 10 }}>
                {file ? formatBytes(file.size) : ''}
              </span>
            </div>
          </div>
          <button
            type="button"
            onClick={onReset}
            disabled={state === 'scanning'}
            style={{
              padding: '6px 12px',
              fontFamily: C.mono,
              fontSize: 10,
              letterSpacing: 0.7,
              textTransform: 'uppercase',
              background: C.bg,
              color: state === 'scanning' ? C.inkFaint : C.inkSoft,
              border: `1px solid ${C.rule}`,
              cursor: state === 'scanning' ? 'not-allowed' : 'pointer',
            }}
          >
            Verify another
          </button>
        </div>

        {state === 'scanning' && (
          <div style={{ padding: '24px 18px', fontFamily: C.mono, fontSize: 11, color: C.inkSoft, letterSpacing: 0.3 }}>
            <div style={{ marginBottom: 6 }}>· extracting watermark payload</div>
            <div style={{ marginBottom: 6, color: C.inkMute }}>· hashing seed → attestation lookup</div>
            <div style={{ color: C.inkMute }}>· querying polygon registry</div>
          </div>
        )}

        {state === 'not_found' && (
          <div style={{ padding: '20px 18px' }}>
            <div style={{ fontFamily: C.sans, fontSize: 13, color: C.inkSoft, lineHeight: 1.6, maxWidth: 640 }}>
              The detector ran cleanly but found no doseedo watermark in this file. Either it wasn’t generated by doseedo, or the watermark was destroyed by aggressive re-encoding (very low-bitrate MP3, vocoder reprocessing, drastic time-stretch).
            </div>
            <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase', color: C.inkMute, marginTop: 10 }}>
              Result · DSD-VERIFY-NEG · no chain lookup performed
            </div>
          </div>
        )}

        {state === 'error' && (
          <div style={{ padding: '20px 18px' }}>
            <div style={{ fontFamily: C.sans, fontSize: 13, color: C.warm, lineHeight: 1.6, maxWidth: 640 }}>
              {error || 'Verification could not complete. Please try again in a moment.'}
            </div>
          </div>
        )}

        {state === 'verified' && rows.length > 0 && (
          <div>
            {rows.map(([k, v, href], i) => (
              <div
                key={i}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '180px 1fr',
                  gap: 14,
                  padding: '11px 18px',
                  borderTop: i === 0 ? 'none' : `1px dashed ${C.rule}`,
                  alignItems: 'baseline',
                }}
              >
                <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase', color: C.inkMute }}>
                  {k}
                </div>
                <div style={{ fontFamily: C.mono, fontSize: 12, color: C.ink, fontFeatureSettings: '"tnum"', wordBreak: 'break-all' }}>
                  {href ? (
                    <a href={href} target="_blank" rel="noreferrer" style={{ color: C.accent, textDecoration: 'none', borderBottom: `1px solid ${C.accent}55` }}>
                      {v} ↗
                    </a>
                  ) : (
                    v
                  )}
                </div>
              </div>
            ))}
            <div
              style={{
                padding: '12px 18px',
                borderTop: `1px solid ${C.rule}`,
                background: 'rgba(47,125,79,.06)',
                fontFamily: C.mono,
                fontSize: 10,
                letterSpacing: 0.6,
                textTransform: 'uppercase',
                color: C.good,
              }}
            >
              ✓ chain match · doseedo stands behind this generation
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

function HowItWorks() {
  const steps = [
    {
      n: '01',
      head: 'Watermark at gen-time',
      body: 'Every doseedo generation gets an inaudible watermark embedded directly into the audio. It survives MP3 compression, EQ, time-stretching, and analog re-recording. It does not survive aggressive vocoder reprocessing — that’s the one limit.',
    },
    {
      n: '02',
      head: 'Attestation on Polygon',
      body: 'At the same moment, a keccak256 hash of the watermark seed plus generation metadata (model, timestamp, tier) is committed to the doseedo registry contract on Polygon mainnet. The on-chain footprint is one bytes32 + a metadata URI pointing back here — no audio, no personal data, just the proof anchor.',
    },
    {
      n: '03',
      head: 'Anyone can verify',
      body: 'Drop the file here. The detector extracts the watermark payload and looks up the matching record. If it’s ours, you get the gen ID, timestamp, and chain proof. If not, the file never leaves your browser — no watermark, no upload.',
    },
  ];
  return (
    <section style={{ marginBottom: 40 }}>
      <SectHead title="How verification works" count="3 steps · public registry" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 14 }}>
        {steps.map((s) => (
          <div key={s.n} style={{ background: C.surface, border: `1px solid ${C.rule}`, padding: '18px 20px' }}>
            <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.8, textTransform: 'uppercase', color: C.inkMute, marginBottom: 6 }}>
              Step {s.n}
            </div>
            <div style={{ fontFamily: C.head, fontSize: 18, fontWeight: 600, letterSpacing: -0.3, marginBottom: 8 }}>
              {s.head}
            </div>
            <div style={{ fontFamily: C.sans, fontSize: 13, color: C.inkSoft, lineHeight: 1.55 }}>
              {s.body}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function FAQ() {
  const items = [
    { q: 'Does verification upload my file?', a: 'Yes — the watermark detector runs on our infra, so the file is uploaded over TLS to the doseedo scanner, scanned, and discarded immediately. We log the scan result (found/not-found, confidence, duration) but never the audio itself. If you need offline verification, we can ship a CLI binary on request — email support.' },
    { q: 'Can the watermark be removed?', a: 'It’s designed to survive normal post-production: MP3 / Opus encoding, EQ, time-stretch, normalisation, even analog re-recording. Aggressive neural vocoder reprocessing or extreme bandwidth limiting can destroy it. That’s a known trade-off — the signal has to stay inaudible.' },
    { q: 'What does the on-chain record actually contain?', a: 'A hash of the watermark seed, the generation timestamp, the model version, and the tier at gen-time. No user identity, no audio, no PII. If a user explicitly chose to publish to a wallet, the wallet address is also recorded.' },
    { q: 'Why Polygon?', a: 'Cheap finality and EVM tooling. Each attestation is a single tx on Polygon mainnet — committing one keccak256 hash + a short metadata URI through the doseedo registry contract. Reorg-safe with a finality re-check pass; failures retry with exponential backoff.' },
    { q: 'Can I opt out of the watermark?', a: 'On Studio and Power, yes — clean exports are a per-generation toggle. On Pro it’s on by default but can be turned off for individual exports. Free tier exports are always watermarked. The attestation always exists; only the audio-side mark is configurable.' },
  ];
  const [open, setOpen] = useState(0);
  return (
    <section style={{ marginBottom: 40 }}>
      <SectHead title="Frequently asked" count="5 · provenance team" />
      <div style={{ background: C.surface, border: `1px solid ${C.rule}` }}>
        {items.map((it, i) => (
          <div key={i} style={{ borderTop: i > 0 ? `1px solid ${C.rule}` : 'none' }}>
            <button
              type="button"
              onClick={() => setOpen(open === i ? -1 : i)}
              style={{
                width: '100%',
                display: 'grid',
                gridTemplateColumns: 'auto 1fr auto',
                gap: 16,
                alignItems: 'baseline',
                padding: '14px 18px',
                cursor: 'pointer',
                textAlign: 'left',
                background: 'transparent',
                border: 'none',
                color: 'inherit',
              }}
            >
              <span style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.6, color: C.inkMute }}>
                Q.{String(i + 1).padStart(2, '0')}
              </span>
              <span style={{ fontFamily: C.sans, fontSize: 14, fontWeight: 500, color: C.ink }}>{it.q}</span>
              <span
                style={{
                  fontFamily: C.mono,
                  fontSize: 13,
                  color: C.inkSoft,
                  transform: open === i ? 'rotate(45deg)' : 'none',
                  transition: 'transform .15s',
                }}
              >
                +
              </span>
            </button>
            {open === i && (
              <div style={{ padding: '0 18px 16px', display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 16 }}>
                <span style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.6, color: C.inkMute }}>
                  A.{String(i + 1).padStart(2, '0')}
                </span>
                <span style={{ fontFamily: C.sans, fontSize: 13, color: C.inkSoft, lineHeight: 1.6, maxWidth: 720 }}>
                  {it.a}
                </span>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

function Closing() {
  return (
    <section style={{ marginBottom: 20 }}>
      <div
        style={{
          background: C.ink,
          color: C.bg,
          padding: '28px 32px',
          display: 'grid',
          gridTemplateColumns: 'minmax(0,1fr) auto',
          gap: 32,
          alignItems: 'center',
        }}
      >
        <div>
          <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.8, textTransform: 'uppercase', color: 'rgba(232,230,225,.5)', marginBottom: 10 }}>
            § Registry
          </div>
          <div style={{ fontFamily: C.head, fontSize: 28, fontWeight: 600, letterSpacing: -0.6, lineHeight: 1.15, maxWidth: 640 }}>
            Provenance you can prove. <span style={{ color: C.purple }}>On every generation, by default.</span>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <a
            href="/plans"
            style={{
              padding: '12px 18px',
              fontFamily: C.mono,
              fontSize: 11,
              fontWeight: 600,
              letterSpacing: 0.8,
              textTransform: 'uppercase',
              background: 'transparent',
              border: '1px solid rgba(232,230,225,.3)',
              color: C.bg,
              cursor: 'pointer',
              textDecoration: 'none',
              display: 'inline-flex',
              alignItems: 'center',
            }}
          >
            See plans
          </a>
          <a
            href="https://polygonscan.com/address/0x0000000000000000000000000000000000000000"
            target="_blank"
            rel="noreferrer"
            style={{
              padding: '12px 18px',
              fontFamily: C.mono,
              fontSize: 11,
              fontWeight: 600,
              letterSpacing: 0.8,
              textTransform: 'uppercase',
              background: C.purple,
              border: 'none',
              color: C.ink,
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 10,
              textDecoration: 'none',
            }}
          >
            <span>View registry on Polygon</span>
            <Arrow size={13} stroke={1.8} color={C.ink} />
          </a>
        </div>
      </div>
      <div
        style={{
          marginTop: 10,
          display: 'flex',
          justifyContent: 'space-between',
          fontFamily: C.mono,
          fontSize: 10,
          letterSpacing: 0.5,
          textTransform: 'uppercase',
          color: C.inkMute,
          flexWrap: 'wrap',
          gap: 8,
        }}
      >
        <span>doseedo provenance · polygon · 2026</span>
        <span>open registry · public lookup · no account required</span>
      </div>
    </section>
  );
}

// Inject the workbench Google Fonts once — Inter / JetBrains Mono / Lora.
function useWorkbenchFonts() {
  useEffect(() => {
    const href =
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&family=Lora:wght@500;600;700&display=swap';
    if (typeof document === 'undefined') return;
    if (document.querySelector(`link[href="${href}"]`)) return;
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = href;
    document.head.appendChild(link);
  }, []);
}

const Verify = () => {
  const [state, setState] = useState('idle');
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  useWorkbenchFonts();

  const onFile = async (f) => {
    setFile(f);
    setResult(null);
    setError(null);
    setState('scanning');
    try {
      const r = await verifyFile(f);
      setResult(r.status === 'verified' ? r : null);
      setState(r.status === 'verified' ? 'verified' : 'not_found');
    } catch (e) {
      setError(e.message || 'unknown error');
      setState('error');
    }
  };

  const onReset = () => {
    setState('idle');
    setFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <main
      style={{
        minWidth: 0,
        display: 'flex',
        flexDirection: 'column',
        background: C.bg,
        color: C.ink,
        fontFamily: C.sans,
        fontSize: 13,
        minHeight: '100vh',
        flex: 1,
      }}
    >
      <Topbar />
      <div style={{ flex: 1, overflow: 'auto', padding: '36px 40px 80px', maxWidth: 1200, width: '100%', boxSizing: 'border-box' }}>
        <Hero />
        <section style={{ marginBottom: 40 }}>
          <SectHead
            title="Verify a file"
            count="audio · drop or browse"
            right={
              <span style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase', color: C.inkMute }}>
                ref. DSD-VERIFY-01
              </span>
            }
          />
          <DropZone onFile={onFile} busy={state === 'scanning'} />
        </section>
        <ResultPanel state={state} file={file} result={result} error={error} onReset={onReset} />
        <HowItWorks />
        <FAQ />
        <Closing />
      </div>
    </main>
  );
};

export default Verify;
