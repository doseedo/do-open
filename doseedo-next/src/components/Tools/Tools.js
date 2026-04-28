import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import VocalHarmonizerTool from './VocalHarmonizerTool';
import VoiceToInstrumentTool from './VoiceToInstrumentTool';
import LyricEditTool from './LyricEditTool';
import StemSeparationTool from './StemSeparationTool';
// Video-to-music lives in the studio drop zone (StudioDev/StudioDevVideo.js)
// now — the standalone tool is retired. The tool registry entry below is
// kept as a Studio launcher so the dashboard tile still works.
import SampleRegeneratorTool from './SampleRegeneratorTool';
import BeatGeneratorTool from './BeatGeneratorTool';

/**
 * Tools landing page — workbench-themed grid of AI utilities.
 *
 * Two modes:
 *   1. No tool selected: themed landing (Hero / filters / card grid /
 *      Workflow strip / Closing strip) — matches daw/tools.jsx.
 *   2. Tool selected: reuses the existing scroll-bar + per-tool UI so the
 *      actual tools (VocalHarmonizer, etc.) keep working unchanged.
 *
 * App.js wraps this component with <LeftSidebar/> so we drop the theme's
 * own sidebar and only render the content region.
 */

const C = {
  bg: '#e8e6e1',
  surface: '#f2f0ea',
  surface2: '#dcd9d1',
  surface3: '#cfccc3',
  ink: '#15181c',
  inkSoft: 'rgba(21,24,28,0.66)',
  inkMute: 'rgba(21,24,28,0.40)',
  inkFaint: 'rgba(21,24,28,0.22)',
  rule: 'rgba(21,24,28,0.14)',
  ruleStrong: 'rgba(21,24,28,0.30)',
  accent: '#1d4c7a',
  warm: '#c94f2c',
  ok: '#2f6b4e',
  purple: '#AAB0EE',
  sans: '"Inter",system-ui,sans-serif',
  mono: '"JetBrains Mono",ui-monospace,Menlo,monospace',
  head: '"Lora",Georgia,serif',
};

const arrowPath = 'M5 12h14 M13 6l6 6-6 6';

const Arrow = ({ size = 13, stroke = 1.8, color = 'currentColor' }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke={color}
    strokeWidth={stroke}
    strokeLinecap="round"
    strokeLinejoin="round"
    style={{ flexShrink: 0 }}
  >
    <path d={arrowPath} />
  </svg>
);

// Load workbench fonts once — same family + weights as /plans.
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

function Topbar({ availableCount }) {
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
      <span style={{ color: C.inkSoft }}>Create</span>
      <span style={{ color: C.inkFaint }}>/</span>
      <span>
        <strong style={{ color: C.inkSoft, fontWeight: 500 }}>Tools</strong>
      </span>
      <span style={{ color: C.inkFaint }}>·</span>
      <span>{availableCount} tools · no session needed</span>
      <div style={{ flex: 1 }} />
      <span
        style={{
          padding: '2px 6px',
          border: `1px solid ${C.rule}`,
          borderRadius: 2,
          fontFamily: C.mono,
          fontSize: 9,
          color: C.inkSoft,
          letterSpacing: 0.3,
          textTransform: 'none',
        }}
      >
        ⌘K
      </span>
    </div>
  );
}

function Hero() {
  return (
    <section style={{ marginBottom: 40, paddingBottom: 28, borderBottom: `1px solid ${C.rule}` }}>
      <div
        style={{
          fontFamily: C.mono,
          fontSize: 10,
          letterSpacing: 0.8,
          textTransform: 'uppercase',
          color: C.inkMute,
          marginBottom: 12,
        }}
      >
        § Tools · Standalone AI utilities
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1.3fr) minmax(0,1fr)', gap: 60, alignItems: 'end' }}>
        <h1 className="page-title">
          Individual tools, <span style={{ color: C.accent }}>no studio required.</span>
        </h1>
        <div style={{ fontFamily: C.sans, fontSize: 13, color: C.inkSoft, lineHeight: 1.6, paddingBottom: 4, maxWidth: 420 }}>
          One-shot utilities for when you don’t need a whole session. Drop a file, pick a tool, take the result back to wherever you were working.
        </div>
      </div>
    </section>
  );
}

// ---------- Per-tool illustrations ----------
function ToolVis({ kind }) {
  const common = { width: '100%', height: '100%', display: 'block' };
  if (kind === 'harmonizer') {
    return (
      <svg viewBox="0 0 200 80" style={common} preserveAspectRatio="none">
        {[0, 1, 2].map((r) => {
          const y = 20 + r * 20;
          const phase = r * 0.6;
          const pts = Array.from({ length: 40 }, (_, i) => {
            const x = i * 5;
            const yy = y + Math.sin(i * 0.4 + phase) * (8 - r * 2);
            return `${x},${yy}`;
          }).join(' ');
          return (
            <polyline
              key={r}
              points={pts}
              fill="none"
              stroke={r === 0 ? C.ink : r === 1 ? C.purple : C.accent}
              strokeWidth={r === 0 ? 1.4 : 1}
              opacity={r === 0 ? 1 : 0.8}
            />
          );
        })}
      </svg>
    );
  }
  if (kind === 'v2m') {
    return (
      <svg viewBox="0 0 200 80" style={common} preserveAspectRatio="none">
        <rect x="10" y="14" width="70" height="52" fill={C.ink} opacity=".08" stroke={C.ink} strokeWidth="1" />
        <path d="M38 30l22 12-22 12z" fill={C.ink} />
        <path d="M90 40h20" stroke={C.ink} strokeWidth="1.4" strokeLinecap="round" />
        <path d="M104 36l6 4-6 4" stroke={C.ink} strokeWidth="1.4" fill="none" strokeLinejoin="round" strokeLinecap="round" />
        {Array.from({ length: 18 }).map((_, i) => {
          const h = 4 + (Math.sin(i * 0.7) * 12 + 14);
          return <rect key={i} x={120 + i * 4} y={40 - h / 2} width="2" height={h} fill={C.accent} />;
        })}
      </svg>
    );
  }
  if (kind === 'lyric') {
    return (
      <svg viewBox="0 0 200 80" style={common} preserveAspectRatio="none">
        {[22, 36, 50, 64].map((y, i) => (
          <g key={i}>
            <line x1="12" y1={y} x2={12 + [140, 100, 120, 70][i]} y2={y} stroke={C.ink} strokeWidth={i === 1 ? 2 : 1} opacity={i === 1 ? 1 : 0.35} />
            {i === 1 && <rect x={12 + 100 - 2} y={y - 8} width="3" height="16" fill={C.purple} />}
          </g>
        ))}
        <text x="160" y="24" fontFamily="JetBrains Mono,monospace" fontSize="8" fill={C.inkMute}>
          AI
        </text>
      </svg>
    );
  }
  if (kind === 'v2i') {
    return (
      <svg viewBox="0 0 200 80" style={common} preserveAspectRatio="none">
        <circle cx="30" cy="34" r="8" fill="none" stroke={C.ink} strokeWidth="1.4" />
        <path d="M30 42v8 M22 50h16 M22 34a8 8 0 0016 0" fill="none" stroke={C.ink} strokeWidth="1.4" strokeLinecap="round" />
        <path d="M60 40h30" stroke={C.ink} strokeWidth="1.4" strokeLinecap="round" />
        <path d="M84 36l6 4-6 4" stroke={C.ink} strokeWidth="1.4" fill="none" strokeLinejoin="round" strokeLinecap="round" />
        <rect x="110" y="22" width="78" height="36" fill="none" stroke={C.ink} strokeWidth="1" />
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <line key={i} x1={110 + (i + 1) * 13} y1="22" x2={110 + (i + 1) * 13} y2="58" stroke={C.ink} strokeWidth=".8" />
        ))}
        {[0, 2, 3, 5].map((i) => (
          <rect key={i} x={110 + i * 13 + 9} y="22" width="8" height="22" fill={C.ink} />
        ))}
      </svg>
    );
  }
  if (kind === 'regen') {
    return (
      <svg viewBox="0 0 200 80" style={common} preserveAspectRatio="none">
        {Array.from({ length: 30 }).map((_, i) => {
          const h = 4 + Math.abs(Math.sin(i * 0.6)) * 30;
          return <rect key={i} x={10 + i * 3} y={40 - h / 2} width="1.8" height={h} fill={C.inkFaint} />;
        })}
        <path d="M100 40a20 20 0 1 1 -6 -14" fill="none" stroke={C.purple} strokeWidth="2" strokeLinecap="round" />
        <path d="M94 26l-4 2 2 4" fill="none" stroke={C.purple} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        {Array.from({ length: 30 }).map((_, i) => {
          const h = 4 + Math.abs(Math.sin(i * 0.6 + 1)) * 36;
          return <rect key={i} x={120 + i * 2.6} y={40 - h / 2} width="1.8" height={h} fill={C.ink} />;
        })}
      </svg>
    );
  }
  if (kind === 'stem') {
    return (
      <svg viewBox="0 0 200 80" style={common} preserveAspectRatio="none">
        {Array.from({ length: 40 }).map((_, i) => {
          const h = 6 + Math.abs(Math.sin(i * 0.45)) * 26;
          return <rect key={i} x={10 + i * 2.2} y={40 - h / 2} width="1.4" height={h} fill={C.ink} opacity=".4" />;
        })}
        <path d="M100 15v50" stroke={C.purple} strokeWidth="1.2" strokeDasharray="3 2" />
        {[
          { y: 18, c: C.warm },
          { y: 32, c: C.ok },
          { y: 46, c: C.accent },
          { y: 60, c: C.purple },
        ].map((r, i) => (
          <g key={i}>
            <rect x="110" y={r.y - 5} width="80" height="10" fill={r.c} opacity=".18" />
            {Array.from({ length: 20 }).map((_, j) => {
              const h = 2 + Math.abs(Math.sin(j * 0.7 + i)) * 6;
              return <rect key={j} x={112 + j * 4} y={r.y - h / 2} width="1.4" height={h} fill={r.c} />;
            })}
          </g>
        ))}
      </svg>
    );
  }
  if (kind === 'beat') {
    const pattern = [
      [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];
    const colors = [C.warm, C.ink, C.accent];
    return (
      <svg viewBox="0 0 200 80" style={common} preserveAspectRatio="none">
        {[20, 36, 52].map((y, r) => (
          <g key={r}>
            {pattern[r].map((on, c) => (
              <rect
                key={c}
                x={12 + c * 11}
                y={y - 5}
                width="9"
                height="10"
                fill={on ? colors[r] : 'none'}
                stroke={on ? colors[r] : C.rule}
                strokeWidth="1"
              />
            ))}
          </g>
        ))}
      </svg>
    );
  }
  return null;
}

// ---------- Themed tool card ----------
function ToolCard({ tool, onClick }) {
  const available = tool.status === 'available';
  const beta = tool.status === 'beta';
  return (
    <div
      style={{
        background: C.surface,
        border: `1px solid ${C.rule}`,
        padding: 0,
        display: 'flex',
        flexDirection: 'column',
        cursor: 'pointer',
        transition: 'border-color .12s',
        position: 'relative',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = C.ruleStrong;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = C.rule;
      }}
      onClick={onClick}
    >
      <div
        style={{
          position: 'absolute',
          top: -1,
          right: -1,
          background: C.ink,
          color: C.purple,
          fontFamily: C.mono,
          fontSize: 9,
          fontWeight: 600,
          letterSpacing: 0.5,
          padding: '3px 7px',
          zIndex: 2,
        }}
      >
        {tool.sku}
      </div>

      <div
        style={{
          height: 120,
          background: C.bg,
          borderBottom: `1px solid ${C.rule}`,
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            position: 'absolute',
            inset: 0,
            opacity: 0.35,
            backgroundImage: `linear-gradient(to right, ${C.rule} 1px, transparent 1px), linear-gradient(to bottom, ${C.rule} 1px, transparent 1px)`,
            backgroundSize: '10px 10px',
          }}
        />
        <div style={{ position: 'absolute', inset: 0, padding: 14 }}>
          <ToolVis kind={tool.vis} />
        </div>
      </div>

      <div style={{ padding: '16px 18px 14px', display: 'flex', flexDirection: 'column', gap: 6, flex: 1 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontFamily: C.mono, fontSize: 9, letterSpacing: 0.7, textTransform: 'uppercase', color: C.inkMute }}>
            {tool.category}
          </span>
          <span style={{ width: 4, height: 4, background: C.inkFaint }} />
          {available && (
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 5,
                fontFamily: C.mono,
                fontSize: 9,
                letterSpacing: 0.7,
                textTransform: 'uppercase',
                color: C.ok,
              }}
            >
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: C.ok }} />
              Available
            </span>
          )}
          {beta && (
            <span
              style={{
                fontFamily: C.mono,
                fontSize: 9,
                letterSpacing: 0.7,
                textTransform: 'uppercase',
                color: C.purple,
                padding: '1px 5px',
                border: `1px solid ${C.purple}55`,
                background: 'rgba(170,176,238,.12)',
              }}
            >
              Beta
            </span>
          )}
          {!available && !beta && (
            <span style={{ fontFamily: C.mono, fontSize: 9, letterSpacing: 0.7, textTransform: 'uppercase', color: C.inkMute }}>
              Soon
            </span>
          )}
        </div>

        <div style={{ fontFamily: C.head, fontSize: 19, fontWeight: 600, letterSpacing: -0.3, lineHeight: 1.15, marginTop: 2 }}>
          {tool.name}
        </div>
        <div style={{ fontFamily: C.sans, fontSize: 12, color: C.inkSoft, lineHeight: 1.5, flex: 1 }}>{tool.desc}</div>

        <div
          style={{
            display: 'flex',
            gap: 8,
            marginTop: 8,
            fontFamily: C.mono,
            fontSize: 9,
            letterSpacing: 0.4,
            color: C.inkMute,
            textTransform: 'uppercase',
            flexWrap: 'wrap',
          }}
        >
          <span>In · {tool.input}</span>
          <span style={{ color: C.inkFaint }}>→</span>
          <span>Out · {tool.output}</span>
          <span style={{ flex: 1 }} />
          <span>~{tool.time}</span>
        </div>

        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onClick();
          }}
          style={{
            marginTop: 12,
            padding: '9px 12px',
            background: C.ink,
            color: C.bg,
            border: 'none',
            fontFamily: C.mono,
            fontSize: 10,
            fontWeight: 600,
            letterSpacing: 0.8,
            textTransform: 'uppercase',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            cursor: 'pointer',
          }}
        >
          <span>Open tool</span>
          <Arrow size={12} color={C.bg} stroke={1.8} />
        </button>
      </div>
    </div>
  );
}

// ---------- Tools grid with filter ----------
function ToolsGrid({ tools, filter, setFilter, onToolClick }) {
  const filters = ['All', 'Available', 'Audio', 'Video', 'Lyrics', 'Beta'];
  const filtered =
    filter === 'All'
      ? tools
      : filter === 'Available'
      ? tools.filter((t) => t.status === 'available')
      : filter === 'Beta'
      ? tools.filter((t) => t.status === 'beta')
      : tools.filter((t) => t.category === filter);

  return (
    <section style={{ marginBottom: 40 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 18, flexWrap: 'wrap' }}>
        <h2 style={{ fontFamily: C.head, fontSize: 20, fontWeight: 600, letterSpacing: -0.3, margin: 0 }}>
          Creative tools
        </h2>
        <span
          style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase', color: C.inkMute }}
        >
          {filtered.length} of {tools.length} shown
        </span>
        <div style={{ flex: 1 }} />
        <div style={{ display: 'inline-flex', border: `1px solid ${C.rule}`, background: C.surface, flexWrap: 'wrap' }}>
          {filters.map((f, i) => (
            <button
              key={f}
              type="button"
              onClick={() => setFilter(f)}
              style={{
                padding: '5px 11px',
                fontFamily: C.mono,
                fontSize: 10,
                letterSpacing: 0.6,
                textTransform: 'uppercase',
                background: filter === f ? C.ink : 'transparent',
                color: filter === f ? C.bg : C.inkSoft,
                border: 'none',
                borderRight: i < filters.length - 1 ? `1px solid ${C.rule}` : 'none',
                cursor: 'pointer',
              }}
            >
              {f}
            </button>
          ))}
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 14 }}>
        {filtered.map((t) => (
          <ToolCard key={t.id} tool={t} onClick={() => onToolClick(t)} />
        ))}
      </div>
    </section>
  );
}

function Workflow() {
  const steps = [
    { n: '01', k: 'Drop input', d: 'Audio, video, text, or a voice memo. Drag straight from Finder or record in-page.' },
    { n: '02', k: 'Set one or two knobs', d: 'Each tool exposes only the controls that matter. Sensible defaults; tweak if you care.' },
    { n: '03', k: 'Download result', d: 'Take the output to your DAW, your timeline, or back into a doseedo session — your choice.' },
  ];
  return (
    <section style={{ marginBottom: 40 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 14, marginBottom: 18, flexWrap: 'wrap' }}>
        <h2 style={{ fontFamily: C.head, fontSize: 20, fontWeight: 600, letterSpacing: -0.3, margin: 0 }}>
          How each tool works
        </h2>
        <span
          style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase', color: C.inkMute }}
        >
          3 steps · no session
        </span>
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: 0,
          border: `1px solid ${C.rule}`,
          background: C.surface,
        }}
      >
        {steps.map((s, i) => (
          <div
            key={i}
            style={{
              padding: '24px 22px',
              borderLeft: i > 0 ? `1px solid ${C.rule}` : 'none',
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
            }}
          >
            <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.8, textTransform: 'uppercase', color: C.purple }}>
              {s.n}
            </div>
            <div style={{ fontFamily: C.head, fontSize: 18, fontWeight: 600, letterSpacing: -0.2, lineHeight: 1.2 }}>
              {s.k}
            </div>
            <div style={{ fontFamily: C.sans, fontSize: 12, color: C.inkSoft, lineHeight: 1.55 }}>{s.d}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

function Closing({ onOpenStudio }) {
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
          <div
            style={{
              fontFamily: C.mono,
              fontSize: 10,
              letterSpacing: 0.8,
              textTransform: 'uppercase',
              color: 'rgba(232,230,225,.5)',
              marginBottom: 10,
            }}
          >
            § Need more than one tool?
          </div>
          <div style={{ fontFamily: C.head, fontSize: 28, fontWeight: 600, letterSpacing: -0.6, lineHeight: 1.15, maxWidth: 680 }}>
            Chain the tools inside a <span style={{ color: C.purple }}>full doseedo session.</span>
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
            href="/studio"
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
            <span>Open the studio</span>
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
        <span>doseedo tools · standalone</span>
        <span>free tier: 3 runs per tool per day</span>
      </div>
    </section>
  );
}

// Authoritative tool list. Each entry pairs the themed card metadata with
// the real component to mount in the selected-tool view. Keep `id` stable
// so deep-links, analytics, and existing tool-bar state survive edits.
const TOOLS = [
  {
    id: 'vocal-harmonizer',
    sku: 'T-01',
    name: 'Vocal Harmonizer',
    desc: 'Generate beautiful harmony tracks from your vocals — pick an interval, choose a voicing, render stems.',
    category: 'Audio',
    status: 'available',
    vis: 'harmonizer',
    input: 'vocal WAV',
    output: '3× stem',
    time: '40s',
    icon: 'fa-music',
    component: VocalHarmonizerTool,
  },
  {
    id: 'video-to-music',
    sku: 'T-02',
    name: 'Video to Music',
    desc: 'Transform your videos into scored soundtracks. Drop a clip, pick a mood, get a frame-synced track.',
    category: 'Video',
    status: 'available',
    vis: 'v2m',
    input: 'mp4/mov',
    output: 'WAV + stems',
    time: '90s',
    icon: 'fa-video',
    // Routes to /studio when launched from the dashboard tile. The actual
    // upload + scoring UI lives in StudioDev/StudioDevVideo.js.
    route: '/studio',
  },
  {
    id: 'lyric-edit',
    sku: 'T-03',
    name: 'Lyric Edit',
    desc: 'Edit and generate lyrics with AI assistance. Rhyme-aware, meter-aware, and keeps your voice.',
    category: 'Lyrics',
    status: 'available',
    vis: 'lyric',
    input: 'text',
    output: 'text',
    time: '5s',
    icon: 'fa-pen-to-square',
    component: LyricEditTool,
  },
  {
    id: 'voice-to-instrument',
    sku: 'T-04',
    name: 'Voice to Instrument',
    desc: 'Convert voice recordings to instrumental tracks — hum a line, get a clean synth, piano, or guitar part.',
    category: 'Audio',
    status: 'available',
    vis: 'v2i',
    input: 'voice memo',
    output: 'MIDI + WAV',
    time: '30s',
    icon: 'fa-microphone-lines',
    component: VoiceToInstrumentTool,
  },
  {
    id: 'sample-regenerator',
    sku: 'T-05',
    name: 'Sample Regenerator',
    desc: 'Regenerate and enhance audio samples — up-rez, re-pitch, or refresh without losing the original character.',
    category: 'Audio',
    status: 'beta',
    vis: 'regen',
    input: 'WAV / MP3',
    output: 'WAV',
    time: '25s',
    icon: 'fa-rotate',
    component: SampleRegeneratorTool,
  },
  {
    id: 'stem-separation',
    sku: 'T-06',
    name: 'Stem Separation',
    desc: 'Separate any audio into individual stems — vocals, drums, bass, and harmonic content. Studio-grade.',
    category: 'Audio',
    status: 'available',
    vis: 'stem',
    input: 'any audio',
    output: '4× stems',
    time: '60s',
    icon: 'fa-layer-group',
    component: StemSeparationTool,
  },
  {
    id: 'beat-generator',
    sku: 'T-07',
    name: 'Beat Generator',
    desc: 'Generate custom beats and rhythms. Choose a feel, set BPM, dial in swing, and export the MIDI pattern.',
    category: 'Audio',
    status: 'beta',
    vis: 'beat',
    input: 'BPM + style',
    output: 'MIDI + WAV',
    time: '10s',
    icon: 'fa-drum',
    component: BeatGeneratorTool,
  },
];

const Tools = () => {
  const navigate = useNavigate();
  const [selectedTool, setSelectedTool] = useState(null);
  const [filter, setFilter] = useState('All');
  useWorkbenchFonts();

  const availableCount = TOOLS.filter((t) => t.status === 'available').length;

  // ---------- Selected-tool view ----------
  // The workbench-themed tool components (VocalHarmonizerTool, etc.)
  // render their own Topbar + Header + BottomBar via <ToolShell/>, so no
  // outer chrome needed here — just give the component the full container
  // and let it fill it. The onBack prop wires the shell's "Back to tools"
  // button back to this parent.
  if (selectedTool) {
    const ToolComponent = selectedTool.component;
    if (!ToolComponent) return null;
    return (
      <div
        style={{
          // Sidebar offset — clears the 220px LeftSidebar so the tool
          // shell isn't overlapped on the left.
          position: 'absolute',
          top: 0, right: 0, bottom: 0, left: 220,
          display: 'flex',
          flexDirection: 'column',
          background: '#e8e6e1',
          minHeight: 0,
          minWidth: 0,
        }}
      >
        <ToolComponent tool={selectedTool} onBack={() => setSelectedTool(null)} />
      </div>
    );
  }

  // ---------- Themed landing (no tool selected) ----------
  return (
    <main
      style={{
        // 220px clears the fixed LeftSidebar on dashboard routes.
        marginLeft: 220,
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
      <Topbar availableCount={availableCount} />
      <div
        style={{
          flex: 1,
          overflow: 'auto',
          padding: '36px 40px 80px',
          maxWidth: 1200,
          width: '100%',
          margin: '0 auto',
          boxSizing: 'border-box',
        }}
      >
        <Hero />
        <ToolsGrid
          tools={TOOLS}
          filter={filter}
          setFilter={setFilter}
          onToolClick={(tool) => {
            // Tools with a `route` (e.g. video-to-music → /studio) navigate
            // away rather than rendering an in-place tool shell.
            if (tool.route) navigate(tool.route);
            else setSelectedTool(tool);
          }}
        />
        <Workflow />
        <Closing />
      </div>
    </main>
  );
};

export default Tools;
