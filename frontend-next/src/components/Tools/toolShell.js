/**
 * Shared workbench shell + primitives for all /tools subpages.
 *
 * Layout (per daw/tool-harmonizer.jsx):
 *
 *   ┌─ Topbar (breadcrumb · SKU · version · SESSION ACTIVE pill · Docs) ┐
 *   │  Header (tool name · meta · runs/avg/last)                         │
 *   │  3-column grid: Input ▸ Controls ▸ Output                          │
 *   │  BottomBar (cost · engine · render progress · Render/Cancel)       │
 *   └────────────────────────────────────────────────────────────────────┘
 *
 * Each tool passes its own Input/Controls/Output content in; this module
 * owns everything else. Preserves the existing { tool, onBack } props the
 * legacy Tools.js scroll-bar passes, and the existing backend wiring in
 * each tool file stays untouched.
 *
 * Pass `layout="wide"` for tools without an audio-in/audio-out triple
 * (Lyric Edit, Beat Generator input column). That collapses the grid to
 * a single wide column so the tool can render its own internal layout.
 */
import React, { useEffect } from 'react';

export const C = {
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
  lcd: '#6ee7a1',
  sans: '"Inter",system-ui,sans-serif',
  mono: '"JetBrains Mono",ui-monospace,Menlo,monospace',
  head: '"Lora",Georgia,serif',
};

// ---------- Primitives ----------

export const Ic = ({ d, size = 14, stroke = 1.7, fill = 'none', color = 'currentColor' }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill={fill}
    stroke={color}
    strokeWidth={stroke}
    strokeLinecap="round"
    strokeLinejoin="round"
    style={{ flexShrink: 0 }}
  >
    <path d={d} />
  </svg>
);

export const P = {
  arrow: 'M5 12h14 M13 6l6 6-6 6',
  play: 'M6 4l14 8-14 8z',
  pause: 'M6 4h4v16H6z M14 4v4h4v-4z M14 4h4v16h-4z',
  dl: 'M12 3v12 M7 11l5 5 5-5 M4 21h16',
  close: 'M6 6l12 12 M18 6L6 18',
  dice: 'M5 5h14v14H5z M9 9h.01 M15 15h.01 M15 9h.01 M9 15h.01 M12 12h.01',
  plus: 'M12 5v14 M5 12h14',
  minus: 'M5 12h14',
  reset: 'M20 12a8 8 0 11-3-6.2 M20 3v6h-6',
};

// Stable deterministic waveform renderer. `seed` varies the curve so
// different calls don't overlap. `bars` controls density.
export function WaveformSVG({ color = C.ink, opacity = 1, seed = 0, height = 60, bars = 120 }) {
  return (
    <svg viewBox={`0 0 ${bars * 2} ${height}`} width="100%" height={height} preserveAspectRatio="none">
      {Array.from({ length: bars }).map((_, i) => {
        const h =
          Math.abs(Math.sin(i * 0.18 + seed) * 0.6 + Math.sin(i * 0.07 + seed * 2) * 0.4) *
            (height * 0.8) +
          2;
        return (
          <rect key={i} x={i * 2} y={(height - h) / 2} width="1.3" height={h} fill={color} opacity={opacity} />
        );
      })}
    </svg>
  );
}

// Analog-style knob. `value` 0..100. Optional onChange for interactivity
// (drag handled by the caller if desired; click/scroll is out of scope).
export function Knob({ value = 50, label, unit }) {
  const angle = -135 + (value / 100) * 270;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
      <div style={{ width: 56, height: 56, position: 'relative' }}>
        <div
          style={{
            position: 'absolute',
            inset: 0,
            borderRadius: '50%',
            background: C.surface2,
            border: `1px solid ${C.ruleStrong}`,
            boxShadow: 'inset 0 1px 0 rgba(255,255,255,.6), inset 0 -1px 0 rgba(0,0,0,.05)',
          }}
        />
        <div style={{ position: 'absolute', inset: 6, borderRadius: '50%', background: C.ink }} />
        <div
          style={{
            position: 'absolute',
            left: '50%',
            top: '50%',
            width: 2,
            height: 18,
            background: C.lcd,
            transformOrigin: '50% 0',
            transform: `translate(-50%,0) rotate(${angle}deg)`,
            boxShadow: `0 0 6px ${C.lcd}`,
          }}
        />
        {[-135, -90, -45, 0, 45, 90, 135].map((a) => (
          <div
            key={a}
            style={{
              position: 'absolute',
              left: '50%',
              top: '50%',
              width: 1,
              height: 3,
              background: C.inkMute,
              transformOrigin: `0 -28px`,
              transform: `translate(-50%,-100%) rotate(${a}deg)`,
            }}
          />
        ))}
      </div>
      {label && (
        <div
          style={{
            fontFamily: C.mono,
            fontSize: 9,
            letterSpacing: 0.7,
            textTransform: 'uppercase',
            color: C.inkMute,
            textAlign: 'center',
          }}
        >
          {label}
        </div>
      )}
      <div style={{ fontFamily: C.mono, fontSize: 11, letterSpacing: 0.3, color: C.ink, fontWeight: 500 }}>
        {value}
        {unit || ''}
      </div>
    </div>
  );
}

// Load workbench Google Fonts once per page. Same set as /plans and
// /tools landing — injects on mount, de-dupes by href.
export function useWorkbenchFonts() {
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

// ---------- Panel wrapper ----------

// Uses the ◆ / ◇ / ● header markers from the design for input / controls /
// output respectively. `marker` overrides the glyph; `status` renders a
// right-aligned pill on the panel header.
export function Panel({ title, marker = '◆', status, children, padding = 0 }) {
  return (
    <div
      style={{
        background: C.surface,
        border: `1px solid ${C.rule}`,
        display: 'flex',
        flexDirection: 'column',
        minWidth: 0,
      }}
    >
      <div
        style={{
          padding: '11px 16px',
          borderBottom: `1px solid ${C.rule}`,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          fontFamily: C.mono,
          fontSize: 10,
          letterSpacing: 0.7,
          textTransform: 'uppercase',
          color: C.inkMute,
          minHeight: 38,
        }}
      >
        <span style={{ color: C.purple }}>{marker}</span>
        <span>{title}</span>
        <span style={{ flex: 1 }} />
        {status}
      </div>
      <div style={{ padding, display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
        {children}
      </div>
    </div>
  );
}

// ---------- Topbar / Header / BottomBar ----------

function Topbar({ tool }) {
  const name = tool?.name || 'Tool';
  const sku = tool?.sku || '—';
  const version = tool?.version || 'v2.4.1';
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 14,
        padding: '0 28px',
        height: 48,
        borderBottom: `1px solid ${C.rule}`,
        background: C.surface,
        fontFamily: C.mono,
        fontSize: 10,
        letterSpacing: 0.6,
        textTransform: 'uppercase',
        color: C.inkMute,
        whiteSpace: 'nowrap',
        flexShrink: 0,
        overflow: 'hidden',
      }}
    >
      <span>Dashboard</span>
      <span style={{ color: C.inkFaint }}>/</span>
      <span>Create</span>
      <span style={{ color: C.inkFaint }}>/</span>
      <span>Tools</span>
      <span style={{ color: C.inkFaint }}>/</span>
      <span>
        <strong style={{ color: C.ink, fontWeight: 500 }}>{name}</strong>
      </span>
      <span style={{ color: C.inkFaint }}>·</span>
      <span style={{ color: C.ok, display: 'inline-flex', alignItems: 'center', gap: 5 }}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: C.ok }} />
        SESSION ACTIVE
      </span>
      <div style={{ flex: 1 }} />
      <span>SKU · {sku}</span>
      <span style={{ color: C.inkFaint }}>·</span>
      <span>{version}</span>
    </div>
  );
}

function Header({ tool, subtitle, description, meta }) {
  return (
    <div
      style={{
        padding: '22px 28px 18px',
        borderBottom: `1px solid ${C.rule}`,
        background: C.bg,
        display: 'grid',
        gridTemplateColumns: 'minmax(0,1fr) auto',
        gap: 30,
        alignItems: 'end',
        flexShrink: 0,
      }}
    >
      <div style={{ minWidth: 0 }}>
        <div
          style={{
            fontFamily: C.mono,
            fontSize: 10,
            letterSpacing: 0.8,
            textTransform: 'uppercase',
            color: C.inkMute,
            marginBottom: 8,
          }}
        >
          § {tool?.sku || '—'} · {tool?.category || 'Tool'} · {tool?.version || 'v2.4.1'}
        </div>
        <h1 className="page-title">
          {tool?.name || 'Tool'}
          {subtitle && (
            <span style={{ color: C.inkMute, fontSize: 18, fontWeight: 500 }}> · {subtitle}</span>
          )}
        </h1>
        {description && (
          <div
            style={{
              fontFamily: C.sans,
              fontSize: 12,
              color: C.inkSoft,
              marginTop: 8,
              maxWidth: 640,
              lineHeight: 1.5,
            }}
          >
            {description}
          </div>
        )}
      </div>
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
          fontFamily: C.mono,
          fontSize: 10,
          letterSpacing: 0.6,
          textTransform: 'uppercase',
          color: C.inkMute,
          textAlign: 'right',
        }}
      >
        {(meta || []).map((m, i) => (
          <div key={i}>
            {m.k} · <span style={{ color: C.ink }}>{m.v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function BottomBar({
  running,
  progress = 0,
  statusMessage,
  primaryLabel = 'Render',
  secondaryLabel,
  onPrimary,
  onSecondary,
  onCancel,
  primaryDisabled,
  cost = '1 run',
  engineLabel = 'ready',
  onBack,
}) {
  const pct = Math.max(0, Math.min(1, progress));
  return (
    <div
      style={{
        borderTop: `1px solid ${C.rule}`,
        background: C.surface,
        padding: '14px 28px',
        display: 'grid',
        gridTemplateColumns: 'auto minmax(0,1fr) auto',
        gap: 20,
        alignItems: 'center',
        flexShrink: 0,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <div>
          <div
            style={{
              fontFamily: C.mono,
              fontSize: 9,
              letterSpacing: 0.7,
              textTransform: 'uppercase',
              color: C.inkMute,
              marginBottom: 4,
            }}
          >
            Estimated cost
          </div>
          <div style={{ fontFamily: C.mono, fontSize: 14, fontWeight: 500 }}>
            {cost} <span style={{ color: C.inkMute }}>/ free</span>
          </div>
        </div>
        <div style={{ width: 1, height: 32, background: C.rule }} />
        <div>
          <div
            style={{
              fontFamily: C.mono,
              fontSize: 9,
              letterSpacing: 0.7,
              textTransform: 'uppercase',
              color: C.inkMute,
              marginBottom: 4,
            }}
          >
            Engine
          </div>
          <div
            style={{
              fontFamily: C.mono,
              fontSize: 11,
              fontWeight: 500,
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: C.lcd,
                boxShadow: `0 0 6px ${C.lcd}`,
              }}
            />
            {engineLabel}
          </div>
        </div>
      </div>

      <div
        style={{
          padding: '10px 14px',
          background: C.bg,
          border: `1px solid ${C.rule}`,
          display: 'flex',
          alignItems: 'center',
          gap: 14,
          minWidth: 0,
        }}
      >
        <div
          style={{
            fontFamily: C.mono,
            fontSize: 10,
            letterSpacing: 0.7,
            textTransform: 'uppercase',
            color: running ? C.warm : C.ok,
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            flexShrink: 0,
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              background: running ? C.warm : C.ok,
              borderRadius: '50%',
            }}
          />
          {running ? 'Rendering' : 'Ready'}
        </div>
        {running ? (
          <>
            <div
              style={{
                flex: 1,
                height: 4,
                background: C.surface2,
                position: 'relative',
                overflow: 'hidden',
                minWidth: 0,
              }}
            >
              <div
                style={{
                  position: 'absolute',
                  left: 0,
                  top: 0,
                  bottom: 0,
                  width: `${pct * 100}%`,
                  background: C.ink,
                  transition: 'width .2s ease-out',
                }}
              />
            </div>
            <span
              style={{
                fontFamily: C.mono,
                fontSize: 10,
                color: C.inkSoft,
                letterSpacing: 0.5,
                flexShrink: 0,
              }}
            >
              {Math.round(pct * 100)}%
            </span>
          </>
        ) : (
          <span
            style={{
              flex: 1,
              fontFamily: C.mono,
              fontSize: 10,
              color: C.inkMute,
              letterSpacing: 0.4,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              minWidth: 0,
            }}
            title={statusMessage || ''}
          >
            {statusMessage || 'Configure inputs and press render.'}
          </span>
        )}
      </div>

      <div style={{ display: 'flex', gap: 8 }}>
        {onBack && (
          <button
            type="button"
            onClick={onBack}
            style={{
              padding: '11px 14px',
              border: `1px solid ${C.rule}`,
              background: 'transparent',
              color: C.inkSoft,
              fontFamily: C.mono,
              fontSize: 10,
              letterSpacing: 0.8,
              textTransform: 'uppercase',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Back to tools
          </button>
        )}
        {secondaryLabel && onSecondary && (
          <button
            type="button"
            onClick={onSecondary}
            style={{
              padding: '11px 14px',
              border: `1px solid ${C.rule}`,
              background: 'transparent',
              color: C.inkSoft,
              fontFamily: C.mono,
              fontSize: 10,
              letterSpacing: 0.8,
              textTransform: 'uppercase',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            {secondaryLabel}
          </button>
        )}
        <button
          type="button"
          onClick={running ? onCancel : onPrimary}
          disabled={!running && primaryDisabled}
          style={{
            padding: '11px 22px',
            background: running ? C.warm : C.ink,
            color: C.bg,
            border: 'none',
            fontFamily: C.mono,
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: 0.8,
            textTransform: 'uppercase',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            whiteSpace: 'nowrap',
            cursor: !running && primaryDisabled ? 'not-allowed' : 'pointer',
            opacity: !running && primaryDisabled ? 0.5 : 1,
          }}
        >
          {running ? 'Cancel render' : primaryLabel}
          {!running && <Ic d={P.arrow} size={12} color={C.bg} stroke={1.8} />}
        </button>
      </div>
    </div>
  );
}

// ---------- Public shell ----------

/**
 * ToolShell props:
 *   tool          — { name, sku, category, version, description } (meta)
 *   subtitle      — optional inline suffix on the h1 (e.g. "single-take mode")
 *   description   — optional second-line description under the title
 *   meta          — optional [{k,v}] rows shown right-aligned in the header
 *   running       — bool: render in progress
 *   progress      — 0..1
 *   statusMessage — free text shown next to the engine pill when idle
 *   primaryLabel  — BottomBar primary button label
 *   onPrimary     — primary click handler
 *   onCancel      — click handler when running=true
 *   secondaryLabel, onSecondary — optional secondary button
 *   primaryDisabled — disable primary (respected only when not running)
 *   layout        — "3col" (default) | "wide"
 *   left, center, right — content for the 3 columns (3col)
 *   body          — content for the single wide column (wide)
 *   onBack        — optional Back-to-tools handler
 */
export function ToolShell({
  tool,
  subtitle,
  description,
  meta,
  running,
  progress,
  statusMessage,
  primaryLabel,
  secondaryLabel,
  onPrimary,
  onSecondary,
  onCancel,
  primaryDisabled,
  layout = '3col',
  left,
  center,
  right,
  body,
  onBack,
  cost,
  engineLabel,
}) {
  useWorkbenchFonts();
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        background: C.bg,
        color: C.ink,
        fontFamily: C.sans,
        fontSize: 13,
      }}
    >
      <Topbar tool={tool} />
      <Header tool={tool} subtitle={subtitle} description={description} meta={meta} />
      <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
        {layout === 'wide' ? (
          <div style={{ padding: '18px 28px 22px' }}>{body}</div>
        ) : (
          <div
            style={{
              padding: '18px 28px 22px',
              display: 'grid',
              gridTemplateColumns:
                'minmax(240px, 1fr) minmax(260px, 1.1fr) minmax(280px, 1.2fr)',
              gap: 12,
              minHeight: 0,
            }}
          >
            {left}
            {center}
            {right}
          </div>
        )}
      </div>
      <BottomBar
        running={running}
        progress={progress}
        statusMessage={statusMessage}
        primaryLabel={primaryLabel}
        secondaryLabel={secondaryLabel}
        onPrimary={onPrimary}
        onSecondary={onSecondary}
        onCancel={onCancel}
        primaryDisabled={primaryDisabled}
        cost={cost}
        engineLabel={engineLabel}
        onBack={onBack}
      />
    </div>
  );
}

// ---------- Reusable chrome pieces tools often need ----------

// Cream drop-zone with optional file card once a file is loaded.
export function DropZone({ file, onPick, onRemove, accept = 'audio/*', hint, icon = '◆' }) {
  const inputRef = React.useRef(null);
  const onDragOver = (e) => {
    e.preventDefault();
  };
  const onDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) onPick(f);
  };
  if (file) {
    return (
      <div
        style={{
          padding: '14px 16px',
          display: 'grid',
          gridTemplateColumns: '28px 1fr auto',
          gap: 12,
          alignItems: 'center',
        }}
      >
        <div
          style={{
            width: 28,
            height: 28,
            background: C.ink,
            color: C.purple,
            display: 'grid',
            placeItems: 'center',
            fontFamily: C.mono,
            fontSize: 8,
            fontWeight: 600,
            letterSpacing: 0.5,
          }}
        >
          {icon}
        </div>
        <div style={{ minWidth: 0 }}>
          <div
            style={{
              fontFamily: C.mono,
              fontSize: 11,
              fontWeight: 500,
              letterSpacing: 0.3,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            {file.name}
          </div>
          <div
            style={{
              fontFamily: C.mono,
              fontSize: 9,
              letterSpacing: 0.5,
              color: C.inkMute,
              textTransform: 'uppercase',
              marginTop: 2,
            }}
          >
            {Math.round((file.size || 0) / 1024)} KB · {file.type || 'file'}
          </div>
        </div>
        <button
          type="button"
          onClick={onRemove}
          style={{
            width: 24,
            height: 24,
            border: `1px solid ${C.rule}`,
            background: 'transparent',
            display: 'grid',
            placeItems: 'center',
            color: C.inkSoft,
            cursor: 'pointer',
          }}
          title="Remove file"
        >
          <Ic d={P.close} size={11} />
        </button>
      </div>
    );
  }
  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDrop={onDrop}
      onDragOver={onDragOver}
      style={{
        margin: 14,
        padding: '28px 16px',
        border: `1px dashed ${C.ruleStrong}`,
        background: 'transparent',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 8,
        cursor: 'pointer',
        textAlign: 'center',
        color: C.inkSoft,
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onPick(f);
        }}
      />
      <div
        style={{
          fontFamily: C.mono,
          fontSize: 10,
          letterSpacing: 0.8,
          textTransform: 'uppercase',
        }}
      >
        Drop file · or click
      </div>
      {hint && (
        <div
          style={{
            fontFamily: C.mono,
            fontSize: 9,
            letterSpacing: 0.5,
            color: C.inkMute,
            textTransform: 'uppercase',
          }}
        >
          {hint}
        </div>
      )}
    </div>
  );
}

// Chip-row selector (pill style).
export function ChipRow({ options, value, onChange, width = 'auto' }) {
  return (
    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', width }}>
      {options.map((opt) => {
        const v = typeof opt === 'string' ? opt : opt.value;
        const l = typeof opt === 'string' ? opt : opt.label;
        const active = value === v;
        return (
          <button
            key={v}
            type="button"
            onClick={() => onChange(v)}
            style={{
              padding: '5px 10px',
              background: active ? C.ink : 'transparent',
              color: active ? C.bg : C.inkSoft,
              border: `1px solid ${active ? C.ink : C.rule}`,
              fontFamily: C.mono,
              fontSize: 10,
              letterSpacing: 0.6,
              textTransform: 'uppercase',
              whiteSpace: 'nowrap',
              cursor: 'pointer',
            }}
          >
            {l}
          </button>
        );
      })}
    </div>
  );
}

// Monospace inline label + value row inside panels.
export function FieldLabel({ children, style }) {
  return (
    <div
      style={{
        fontFamily: C.mono,
        fontSize: 9,
        letterSpacing: 0.7,
        textTransform: 'uppercase',
        color: C.inkMute,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

// Range slider with monospace labels above / below.
export function Slider({ label, value, min, max, step = 1, onChange, unit = '', leftLabel, rightLabel }) {
  return (
    <div style={{ padding: '16px 16px 12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <FieldLabel>{label}</FieldLabel>
        <span style={{ fontFamily: C.mono, fontSize: 12, fontWeight: 500, color: C.ink }}>
          {value}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: '100%', margin: '10px 0 4px' }}
      />
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontFamily: C.mono,
          fontSize: 9,
          color: C.inkFaint,
        }}
      >
        <span>{leftLabel ?? min}</span>
        <span>{rightLabel ?? max}</span>
      </div>
    </div>
  );
}

// Small styled <select>. Use when a dropdown is unavoidable.
export function Select({ value, onChange, options, style }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        padding: '7px 10px',
        background: C.bg,
        border: `1px solid ${C.rule}`,
        color: C.ink,
        fontFamily: C.mono,
        fontSize: 11,
        letterSpacing: 0.3,
        width: '100%',
        ...style,
      }}
    >
      {options.map((o) => {
        const v = typeof o === 'string' ? o : o.value;
        const l = typeof o === 'string' ? o : o.label;
        return (
          <option key={v} value={v}>
            {l}
          </option>
        );
      })}
    </select>
  );
}

// LCD-style readout (ink bg, green text).
export function LCD({ primary, secondary, style }) {
  return (
    <div
      style={{
        padding: '7px 10px',
        background: C.ink,
        color: C.lcd,
        fontFamily: C.mono,
        fontSize: 12,
        fontWeight: 500,
        letterSpacing: 0.6,
        display: 'flex',
        justifyContent: 'space-between',
        whiteSpace: 'nowrap',
        ...style,
      }}
    >
      <span>{primary}</span>
      {secondary && <span style={{ color: C.inkFaint }}>{secondary}</span>}
    </div>
  );
}
