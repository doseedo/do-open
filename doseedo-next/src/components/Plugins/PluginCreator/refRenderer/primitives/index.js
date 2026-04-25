/**
 * Primitive components — Knob, VSlider, LED, Chip, Select, SegDigit.
 * Ported from /Users/hydroadmin/Downloads/plugin editor/helix/knobs.jsx
 * and /Users/hydroadmin/Downloads/plugin editor/strata/app.jsx.
 *
 * Parametric — never hard-coded to a plugin's palette. Colors come from
 * CSS custom properties set by the DSL renderer (see palette.css).
 */

import React, { useState, useRef, useEffect } from 'react';

// ────────────────────────────────────────────────────────────────
// Arc helpers for the rotary knob

function polarToCartesian(cx, cy, r, angleDeg) {
  const rad = (angleDeg - 90) * (Math.PI / 180);
  return [cx + Math.cos(rad) * r, cy + Math.sin(rad) * r];
}

export function arcPath(cx, cy, r, a1, a2) {
  const [x1, y1] = polarToCartesian(cx, cy, r, a1);
  const [x2, y2] = polarToCartesian(cx, cy, r, a2);
  const large = Math.abs(a2 - a1) > 180 ? 1 : 0;
  const sweep = a2 > a1 ? 1 : 0;
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${r} ${r} 0 ${large} ${sweep} ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

// ────────────────────────────────────────────────────────────────
// Knob — 270° rotary with value arc, tick, and optional bipolar layout

const ACCENT_VAR = {
  primary: 'var(--accent)',
  secondary: 'var(--accent2, var(--accent))',
  ok: 'var(--ok)',
  warn: 'var(--warn)',
};

function formatKnobValue(v, format, min = 0, max = 1) {
  const raw = min + v * (max - min);
  switch (format) {
    case 'percent': return `${Math.round(v * 100)}%`;
    case 'db':      return `${(v * 24 - 12).toFixed(1)}dB`;
    case 'semi':    return `${Math.round(v * 24 - 12)}`;
    case 'hz':      return `${Math.round(20 + v * (20000 - 20))}`;
    case 'ms':      return `${Math.round(v * 5000)}ms`;
    case 'count':   return `${Math.round(raw)}`;
    case 'pan':     return raw === 0.5 ? 'C' : raw > 0.5 ? `R${Math.round((raw - 0.5) * 200)}` : `L${Math.round((0.5 - raw) * 200)}`;
    case 'raw':
    default:        return raw.toFixed(2);
  }
}

export function Knob({
  label,
  value = 0.5,
  size = 34,
  bipolar = false,
  format = 'raw',
  accent = 'primary',
  primary = false,
  showValue = true,
}) {
  const [v, setV] = useState(value);
  const drag = useRef(null);
  useEffect(() => { setV(value); }, [value]);

  const startA = -135, endA = 135;
  const angle = startA + v * (endA - startA);

  const r = size / 2 - 3;
  const cx = size / 2, cy = size / 2;
  const accentColor = ACCENT_VAR[accent] || ACCENT_VAR.primary;

  const trackD = arcPath(cx, cy, r, startA, endA);
  const valD = bipolar
    ? arcPath(cx, cy, r, Math.min(0, angle), Math.max(0, angle))
    : arcPath(cx, cy, r, startA, angle);

  const [tickX, tickY] = polarToCartesian(cx, cy, r - 2, angle);
  const [tickInX, tickInY] = polarToCartesian(cx, cy, r - 10, angle);

  const onDown = (e) => {
    e.preventDefault();
    drag.current = { y: e.clientY, v };
    const onMove = (ev) => {
      const dy = drag.current.y - ev.clientY;
      setV(Math.max(0, Math.min(1, drag.current.v + dy / 150)));
    };
    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };

  return (
    <div className={`rr-knob ${primary ? 'rr-knob--primary' : ''}`} onMouseDown={onDown}>
      <svg width={size} height={size} style={{ cursor: 'ns-resize', display: 'block' }}>
        <path d={trackD} stroke="var(--line)" strokeWidth="2" fill="none" strokeLinecap="round" />
        <path d={valD}   stroke={accentColor} strokeWidth="2" fill="none" strokeLinecap="round" />
        <circle cx={cx} cy={cy} r={r - 5} fill="var(--bg-3)" stroke="var(--line)" strokeWidth="0.5" />
        <line x1={tickInX} y1={tickInY} x2={tickX} y2={tickY} stroke={accentColor} strokeWidth="2" strokeLinecap="round" />
      </svg>
      {label && <div className="rr-knob__label">{label}</div>}
      {showValue && <div className="rr-knob__value">{formatKnobValue(v, format)}</div>}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// Vertical slider

export function VSlider({ label, value = 0.7, height = 70, accent = 'primary' }) {
  const [v, setV] = useState(value);
  const drag = useRef(null);
  useEffect(() => { setV(value); }, [value]);

  const accentColor = ACCENT_VAR[accent] || ACCENT_VAR.primary;

  const onDown = (e) => {
    e.preventDefault();
    drag.current = { y: e.clientY, v };
    const onMove = (ev) => {
      const dy = drag.current.y - ev.clientY;
      setV(Math.max(0, Math.min(1, drag.current.v + dy / height)));
    };
    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };

  return (
    <div className="rr-vslider" style={{ height: height + 18 }}>
      <div className="rr-vslider__track" onMouseDown={onDown}>
        <div
          className="rr-vslider__fill"
          style={{ height: `${v * 100}%`, background: accentColor, opacity: 0.85 }}
        />
        <div
          className="rr-vslider__cap"
          style={{ bottom: `${v * 100}%`, borderColor: accentColor }}
        />
      </div>
      {label && (
        <div
          className="rr-vslider__label"
          dangerouslySetInnerHTML={{ __html: label }}
        />
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// LED dot

export function LED({ on = false, color = 'ok', size = 7 }) {
  const tone =
    color === 'amber' ? 'var(--accent)' :
    color === 'red'   ? 'var(--warn)' :
    color === 'blue'  ? 'var(--info, oklch(0.72 0.10 230))' :
                        'var(--ok, oklch(0.75 0.14 155))';
  return (
    <div
      className="rr-led"
      style={{
        width: size,
        height: size,
        background: on ? tone : 'var(--bg-4)',
        boxShadow: on ? `0 0 6px ${tone}` : 'inset 0 0 0 1px var(--line)',
      }}
    />
  );
}

// ────────────────────────────────────────────────────────────────
// Chip — used as tab, dest-badge, tiny pill button

export function Chip({ label, on = false, muted = false, amber = false }) {
  return (
    <span
      className={`rr-chip ${on ? 'is-on' : ''} ${muted ? 'is-muted' : ''} ${amber ? 'is-amber' : ''}`}
    >
      {label}
    </span>
  );
}

// ────────────────────────────────────────────────────────────────
// Select — visual-only dropdown

export function Select({ label, value, amber = false, width }) {
  return (
    <div
      className={`rr-select ${amber ? 'is-amber' : ''}`}
      style={width ? { width, justifyContent: 'space-between' } : null}
    >
      {label && <span className="rr-select__label">{label}</span>}
      <span className="rr-select__value">{value}</span>
      <span className="rr-select__caret">▾</span>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// 7-segment digit — ported from strata/app.jsx

const SEG_MAP = {
  '0': 'abcdef', '1': 'bc', '2': 'abged', '3': 'abgcd', '4': 'fgbc',
  '5': 'afgcd', '6': 'afgecd', '7': 'abc', '8': 'abcdefg', '9': 'abcdfg',
  '-': 'g', ' ': '', '.': '.',
};

const SEG_PATHS = {
  a: 'M4 2 L8 2 L36 2 L40 6 L36 10 L8 10 L4 6 Z',
  b: 'M42 4 L46 8 L46 36 L42 40 L38 36 L38 12 Z',
  c: 'M42 44 L46 48 L46 76 L42 80 L38 76 L38 52 Z',
  d: 'M4 82 L8 78 L36 78 L40 82 L36 86 L8 86 Z',
  e: 'M2 44 L6 48 L6 76 L2 80 L-2 76 L-2 48 Z',
  f: 'M2 4 L6 8 L6 36 L2 40 L-2 36 L-2 8 Z',
  g: 'M4 44 L8 40 L36 40 L40 44 L36 48 L8 48 Z',
};

export function SegDigit({ ch, size = 56, on = true }) {
  const segs = SEG_MAP[ch] || '';
  const w = size * 0.6, h = size;
  const segCol = (s) => (segs.includes(s) && on ? 'var(--led)' : 'var(--led-off, oklch(0.2 0.01 60))');
  const segGlow = (s) => (segs.includes(s) && on ? 'drop-shadow(0 0 3px var(--led-glow, var(--led)))' : 'none');
  return (
    <svg width={w} height={h} viewBox="-4 0 50 90" style={{ display: 'block' }}>
      {['a', 'b', 'c', 'd', 'e', 'f', 'g'].map((s) => (
        <path key={s} d={SEG_PATHS[s]} fill={segCol(s)} style={{ filter: segGlow(s) }} />
      ))}
    </svg>
  );
}

export function SegDot({ size = 56, on = true }) {
  return (
    <svg width={size * 0.18} height={size} viewBox="0 0 12 90" style={{ display: 'block' }}>
      <circle
        cx="6" cy="82" r="4"
        fill={on ? 'var(--led)' : 'var(--led-off, oklch(0.2 0.01 60))'}
        style={{ filter: on ? 'drop-shadow(0 0 3px var(--led-glow, var(--led)))' : 'none' }}
      />
    </svg>
  );
}

export function SegDisplay({ value = '2.4', size = 52 }) {
  const out = [];
  for (let i = 0; i < value.length; i++) {
    if (value[i] === '.') out.push(<SegDot key={i} size={size} />);
    else out.push(<SegDigit key={i} ch={value[i]} size={size} />);
  }
  return <div className="rr-seg-display">{out}</div>;
}

// ────────────────────────────────────────────────────────────────
// Stereo meter — ported from strata

export function StereoMeter({ lChannel = 5, rChannel = 4, segs = 7 }) {
  const lit = [Math.min(segs, lChannel), Math.min(segs, rChannel)];
  return (
    <div className="rr-meter">
      <div className="rr-meter__scale">
        <span>24</span><span>18</span><span>12</span><span>6</span><span>0dB</span>
      </div>
      {['L', 'R'].map((ch, idx) => (
        <div className="rr-meter__row" key={ch}>
          <span className="rr-meter__ch">{ch}</span>
          <div className="rr-meter__bar">
            {Array.from({ length: segs }, (_, i) => (
              <div
                key={i}
                className={`rr-meter__seg ${i < lit[idx] ? 'is-on' : ''}`}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// Program button (large filled circle + label + number) — STRATA

export function ProgramButton({ n, name, active, onClick }) {
  return (
    <div className="rr-progbtn">
      <div className="rr-progbtn__num">{n}</div>
      <div className="rr-progbtn__name">{name}</div>
      <button
        className={`rr-progbtn__btn ${active ? 'is-active' : ''}`}
        onClick={onClick}
      />
    </div>
  );
}

export function ParamButton({ label, ledOn, active, onClick }) {
  return (
    <div className="rr-parambtn">
      <div className="rr-parambtn__led-wrap">
        <div className={`rr-parambtn__led ${ledOn ? 'is-on' : ''}`} />
      </div>
      <button
        className={`rr-parambtn__btn ${active ? 'is-active' : ''}`}
        onClick={onClick}
      />
      <div
        className="rr-parambtn__label"
        dangerouslySetInnerHTML={{ __html: label }}
      />
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// CharacterKnob — big cream-faced hero knob for lo-fi/color processors.
// Has: optional bypass LED square in top-left, 60-tick grooved rim,
// single darker indicator line, label below. Larger than Knob (default
// 78px vs 34px) and visually distinct — reads as the hero of a row.

export function CharacterKnob({
  label,
  value = 0.5,
  active = true,
  tint = 'var(--accent)',
  size = 78,
  showLed = true,
}) {
  const [v, setV] = useState(value);
  const drag = useRef(null);
  useEffect(() => { setV(value); }, [value]);

  const angle = -135 + v * 270;
  const r = size / 2 - 4;
  const cx = size / 2, cy = size / 2;

  const ticks = [];
  for (let i = 0; i < 60; i++) {
    const a = (i / 60) * 360 - 90;
    const rad = (a - 90) * Math.PI / 180;
    const inner = r + 2;
    const outer = r + 5;
    ticks.push(
      <line
        key={i}
        x1={cx + Math.cos(rad) * inner}
        y1={cy + Math.sin(rad) * inner}
        x2={cx + Math.cos(rad) * outer}
        y2={cy + Math.sin(rad) * outer}
        stroke="var(--ink-faint)"
        strokeWidth="0.8"
      />
    );
  }

  const indRad = (angle - 90) * Math.PI / 180;
  const indX = cx + Math.cos(indRad) * (r - 8);
  const indY = cy + Math.sin(indRad) * (r - 8);

  const onDown = (e) => {
    e.preventDefault();
    drag.current = { y: e.clientY, v };
    const onMove = (ev) => {
      const dy = drag.current.y - ev.clientY;
      setV(Math.max(0, Math.min(1, drag.current.v + dy / 150)));
    };
    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };

  return (
    <div className="rr-character">
      {showLed && (
        <div
          className={`rr-character__led ${active ? 'is-on' : ''}`}
          style={active ? { background: tint, boxShadow: `0 0 4px ${tint}` } : null}
        />
      )}
      <svg width={size + 12} height={size + 12} style={{ cursor: 'ns-resize', display: 'block' }} onMouseDown={onDown}>
        <g transform={`translate(6, 6)`}>
          {ticks}
          <circle
            cx={cx} cy={cy} r={r - 2}
            fill="var(--ink)"
            stroke="var(--line)"
            strokeWidth="1"
          />
          <circle
            cx={cx} cy={cy} r={r - 6}
            fill="none"
            stroke="var(--line-soft)"
            strokeWidth="0.5"
          />
          <line
            x1={cx} y1={cy} x2={indX} y2={indY}
            stroke="var(--bg-0)"
            strokeWidth="2.5"
            strokeLinecap="round"
          />
          <circle cx={cx} cy={cy} r="2" fill="var(--bg-0)" />
        </g>
      </svg>
      <div className="rr-character__label" style={{ color: tint }}>{label}</div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// MorphBar — dual-label horizontal track ("WOW ↔ FLUTTER")

export function MorphBar({ a, b, position = 0.5 }) {
  return (
    <div className="rr-morph">
      <span className="rr-morph__label rr-morph__label--a">{a}</span>
      <div className="rr-morph__track">
        <div className="rr-morph__dot" style={{ left: `${position * 100}%` }} />
      </div>
      <span className="rr-morph__label rr-morph__label--b">{b}</span>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// FluxLane — tiny waveform strip for character modules

export function FluxLane() {
  return (
    <div className="rr-flux">
      <svg width="100%" height="12" viewBox="0 0 100 12" preserveAspectRatio="none">
        <path
          d="M0 6 Q 10 2 20 6 T 40 6 T 60 6 T 80 6 T 100 6"
          stroke="var(--ink-faint)"
          strokeWidth="0.8"
          fill="none"
          opacity="0.6"
        />
      </svg>
      <span className="rr-flux__label">FLUX</span>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// Tiny geometric glyphs — ported wholesale from helix/knobs.jsx

export const Icon = {
  chev: (d) => (
    <svg width="10" height="10" viewBox="0 0 10 10">
      <path
        d={d === 'l' ? 'M6 2 L3 5 L6 8' : d === 'r' ? 'M4 2 L7 5 L4 8' : d === 'd' ? 'M2 4 L5 7 L8 4' : 'M2 6 L5 3 L8 6'}
        stroke="currentColor" strokeWidth="1.2" fill="none" strokeLinecap="round"
      />
    </svg>
  ),
  sine: () => (
    <svg width="16" height="10" viewBox="0 0 16 10">
      <path d="M1 5 Q4 1 8 5 T15 5" stroke="currentColor" strokeWidth="1.2" fill="none" />
    </svg>
  ),
  saw: () => (
    <svg width="16" height="10" viewBox="0 0 16 10">
      <path d="M1 8 L7 2 L7 8 L13 2 L13 8" stroke="currentColor" strokeWidth="1.2" fill="none" />
    </svg>
  ),
  sq: () => (
    <svg width="16" height="10" viewBox="0 0 16 10">
      <path d="M1 8 L1 3 L5 3 L5 8 L9 8 L9 3 L13 3 L13 8" stroke="currentColor" strokeWidth="1.2" fill="none" />
    </svg>
  ),
  tri: () => (
    <svg width="16" height="10" viewBox="0 0 16 10">
      <path d="M1 8 L5 2 L9 8 L13 2" stroke="currentColor" strokeWidth="1.2" fill="none" />
    </svg>
  ),
};
