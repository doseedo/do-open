/**
 * Procedural displays — pure SVG, all styling via CSS custom properties.
 * Ported 1:1 from /Users/hydroadmin/Downloads/plugin editor/helix/displays.jsx
 * and a new FilterCurve/EnvCurve that takes cutoff/res/atk/dec/sus/rel
 * as props (the reference had them hard-coded).
 */

import React from 'react';

// ────────────────────────────────────────────────────────────────
// Wavetable — stacked flyover (OSC A, B)

export function WavetableStacked({ layers = 14, variant = 'sin' }) {
  const lines = [];
  for (let i = 0; i < layers; i++) {
    const yOff = 8 + i * 6;
    const amp = 22 - i * 1.1;
    const phase = i * 0.4;
    let d = '';
    for (let x = 0; x <= 240; x += 4) {
      const t = x / 240;
      let y;
      if (variant === 'saw') {
        y = yOff + amp * (2 * (t - Math.floor(t + 0.5))) * (1 - i * 0.04);
      } else {
        y = yOff + amp * Math.sin(t * Math.PI * 3 + phase) * Math.exp(-t * 0.3) * (1 - i * 0.04);
      }
      d += (x === 0 ? 'M' : 'L') + x + ' ' + y.toFixed(2) + ' ';
    }
    const opacity = 0.25 + (i / layers) * 0.75;
    lines.push(<path key={i} d={d} stroke="var(--accent)" strokeWidth="1" fill="none" opacity={opacity} />);
  }
  return (
    <svg width="100%" height="100%" viewBox="0 0 240 110" preserveAspectRatio="none">
      {[0, 1, 2, 3].map((i) => (
        <line key={i} x1="0" x2="240" y1={i * 28} y2={i * 28} stroke="var(--line-soft)" strokeWidth="0.5" />
      ))}
      {lines}
    </svg>
  );
}

// ────────────────────────────────────────────────────────────────
// Wavetable — stepped morph

export function WavetableStepped({ layers = 10 }) {
  const lines = [];
  for (let i = 0; i < layers; i++) {
    const yOff = 12 + i * 9;
    const amp = 18 - i * 0.6;
    const steps = 8 + i;
    let d = '';
    for (let s = 0; s <= steps; s++) {
      const x = (s / steps) * 240;
      const y = yOff - amp + (s % 2) * amp * 2 * (1 - s / steps);
      d += (s === 0 ? 'M' : 'L') + x + ' ' + y.toFixed(2) + ' ';
    }
    const opacity = 0.35 + (i / layers) * 0.65;
    lines.push(<path key={i} d={d} stroke="var(--accent)" strokeWidth="1" fill="none" opacity={opacity} />);
  }
  return (
    <svg width="100%" height="100%" viewBox="0 0 240 110" preserveAspectRatio="none">
      {[0, 1, 2, 3].map((i) => (
        <line key={i} x1="0" x2="240" y1={i * 28} y2={i * 28} stroke="var(--line-soft)" strokeWidth="0.5" />
      ))}
      {lines}
    </svg>
  );
}

// ────────────────────────────────────────────────────────────────
// Granular sample

export function GranularSample({ cursor = 0.23 }) {
  const bars = [];
  for (let i = 0; i < 120; i++) {
    const t = i / 120;
    const env = Math.pow(1 - Math.abs(t - 0.3) * 2, 2) * (1 - t * 0.5);
    const noise = (Math.sin(i * 1.7) * 0.5 + Math.sin(i * 0.83) * 0.5) * 0.5 + 0.5;
    const h = Math.max(2, Math.abs(env * noise) * 45);
    bars.push(<rect key={i} x={i * 2.1} y={55 - h} width="1.3" height={h * 2} fill="var(--ok)" opacity={0.85} />);
  }
  const cur = cursor * 260;
  return (
    <svg width="100%" height="100%" viewBox="0 0 260 110" preserveAspectRatio="none">
      <line x1="0" x2="260" y1="55" y2="55" stroke="var(--line-soft)" strokeWidth="0.5" />
      {bars}
      <line x1={cur} x2={cur} y1="5" y2="105" stroke="var(--accent)" strokeWidth="1" opacity="0.8" />
      <rect x={cur - 3} y="3" width="6" height="4" fill="var(--accent)" />
    </svg>
  );
}

// ────────────────────────────────────────────────────────────────
// Noise waveform

export function NoiseWave() {
  const bars = [];
  for (let i = 0; i < 60; i++) {
    const h = (Math.sin(i * 1.3) * 0.5 + Math.cos(i * 2.1) * 0.5 + ((i * 17) % 7) / 7) * 25 + 8;
    bars.push(<rect key={i} x={i * 2} y={55 - h / 2} width="1.2" height={h} fill="var(--ink-dim)" opacity={0.7} />);
  }
  return (
    <svg width="100%" height="100%" viewBox="0 0 120 110" preserveAspectRatio="none">
      <line x1="0" x2="120" y1="55" y2="55" stroke="var(--line-soft)" strokeWidth="0.5" />
      {bars}
    </svg>
  );
}

// ────────────────────────────────────────────────────────────────
// Filter response curve — parametric: cutoff/res/hue

export function FilterCurve({ cutoff = 0.4, res = 0.4, hue = 'amber' }) {
  const pts = [];
  const c = Math.max(0.05, Math.min(0.95, cutoff));
  for (let i = 0; i <= 80; i++) {
    const x = i / 80;
    const cutoffEffect = 1 / (1 + Math.pow(x / c, 8));
    const resonance = Math.exp(-Math.pow((x - c) / 0.05, 2)) * (0.3 + res);
    const y = Math.max(0.02, Math.min(1, cutoffEffect + resonance));
    pts.push([x * 160, 85 - y * 75]);
  }
  const d = pts.map((p, i) => (i === 0 ? 'M' : 'L') + p[0] + ' ' + p[1].toFixed(1)).join(' ');
  const fill = d + ` L 160 90 L 0 90 Z`;
  const color = hue === 'warn' ? 'var(--warn)' : hue === 'ok' ? 'var(--ok)' : 'var(--accent)';
  return (
    <svg width="100%" height="100%" viewBox="0 0 160 95" preserveAspectRatio="none">
      {[0, 1, 2, 3].map((i) => (
        <line key={i} x1="0" x2="160" y1={20 + i * 18} y2={20 + i * 18} stroke="var(--line-soft)" strokeWidth="0.5" />
      ))}
      <path d={fill} fill={color} opacity="0.12" />
      <path d={d} stroke={color} strokeWidth="1.3" fill="none" />
    </svg>
  );
}

// ────────────────────────────────────────────────────────────────
// ADSR envelope curve — parametric

export function EnvCurve({ atk = 0.15, dec = 0.25, sus = 0.6, rel = 0.35 }) {
  const x0 = 20, x1 = 320;
  const yTop = 35, yBot = 135;
  const span = x1 - x0;
  const ax = x0 + atk * span * 0.25;
  const dx = ax + dec * span * 0.25;
  const sx = x0 + span * 0.72;
  const rx = x1;
  const ay = yTop;
  const dy = yTop + (1 - sus) * (yBot - yTop);
  const sy = dy;
  const ry = yBot;

  const d = `M ${x0} ${yBot} L ${ax} ${ay} Q ${ax + 8} ${ay + 5} ${dx} ${dy} L ${sx} ${sy} Q ${sx + 30} ${sy + 10} ${rx} ${ry}`;

  return (
    <svg width="100%" height="100%" viewBox="0 0 340 150" preserveAspectRatio="none">
      <defs>
        <linearGradient id="rr-env-grad" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%"   stopColor="var(--accent)" stopOpacity="0.4" />
          <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, 1, 2, 3].map((i) => (
        <line key={i} x1="20" x2="320" y1={25 + i * 37} y2={25 + i * 37} stroke="var(--line-soft)" strokeWidth="0.5" />
      ))}
      {[0, 1, 2, 3, 4].map((i) => (
        <line key={`v${i}`} x1={20 + i * 75} x2={20 + i * 75} y1="20" y2="140" stroke="var(--line-soft)" strokeWidth="0.5" />
      ))}
      <path d={`${d} L ${rx} ${yBot} L ${x0} ${yBot} Z`} fill="url(#rr-env-grad)" />
      <path d={d} stroke="var(--accent)" strokeWidth="1.5" fill="none" />
      {[[ax, ay], [dx, dy], [sx, sy], [rx, ry]].map(([cx, cy], i) => (
        <circle key={i} cx={cx} cy={cy} r="3" fill="var(--accent)" />
      ))}
    </svg>
  );
}

// ────────────────────────────────────────────────────────────────
// LFO curve — accepts [x,y] points, 0..1 normalized

export function LFOCurve({ points }) {
  const defaultPts = [
    [10, 90], [40, 40], [90, 20], [140, 30], [190, 70],
    [220, 110], [190, 140], [140, 130], [100, 100], [60, 120], [30, 140],
  ];
  const pts = points && points.length > 2 ? points.map(([x, y]) => [x * 240, y * 160]) : defaultPts;
  const d = pts.map((p, i) => (i === 0 ? 'M' : 'L') + p[0] + ' ' + p[1]).join(' ');
  return (
    <svg width="100%" height="100%" viewBox="0 0 240 160" preserveAspectRatio="xMidYMid meet">
      <line x1="0" x2="240" y1="80" y2="80" stroke="var(--line-soft)" strokeWidth="0.5" />
      <path d={d} stroke="var(--accent)" strokeWidth="1.3" fill="none" />
      {pts.map((p, i) => (
        <circle
          key={i}
          cx={p[0]}
          cy={p[1]}
          r="3"
          fill={i === 0 || i === pts.length - 1 ? 'var(--accent)' : 'var(--bg-1)'}
          stroke="var(--accent)"
          strokeWidth="1.3"
        />
      ))}
    </svg>
  );
}

// ────────────────────────────────────────────────────────────────
// Velocity curve

export function VelocityCurve({ curve = 1 }) {
  // curve < 1 ⇒ soft, > 1 ⇒ aggressive. Draws y = x^curve on [0,1].
  const pts = [];
  for (let i = 0; i <= 24; i++) {
    const t = i / 24;
    const y = Math.pow(t, curve);
    pts.push([10 + t * 100, 125 - y * 110]);
  }
  const d = pts.map((p, i) => (i === 0 ? 'M' : 'L') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
  return (
    <svg width="100%" height="100%" viewBox="0 0 120 140" preserveAspectRatio="none">
      {[0, 1, 2, 3].map((i) => (
        <line key={i} x1="0" x2="120" y1={10 + i * 40} y2={10 + i * 40} stroke="var(--line-soft)" strokeWidth="0.5" />
      ))}
      <path d={d} stroke="var(--ok)" strokeWidth="1.5" fill="none" />
      <circle cx={pts[0][0]} cy={pts[0][1]} r="2.5" fill="var(--ok)" />
      <circle cx={pts[pts.length - 1][0]} cy={pts[pts.length - 1][1]} r="2.5" fill="var(--ok)" />
    </svg>
  );
}

// ────────────────────────────────────────────────────────────────
// XY pad with constellation backdrop — granular/performance UIs.

export function XYPadConstellation({ cursor = [0.55, 0.45] }) {
  const W = 240;
  const c = W / 2;
  // Seeded starfield so it's stable across renders
  const nodes = [];
  let s = 12345;
  const rand = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  for (let i = 0; i < 60; i++) {
    const ang = rand() * Math.PI * 2;
    const dist = 20 + rand() * (W / 2 - 30);
    nodes.push({
      x: c + Math.cos(ang) * dist,
      y: c + Math.sin(ang) * dist,
      b: 0.3 + rand() * 0.7,
    });
  }
  const cx = cursor[0] * W;
  const cy = cursor[1] * W;
  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${W} ${W}`} preserveAspectRatio="xMidYMid meet">
      <rect width={W} height={W} fill="var(--bg-0)" rx="4" />
      <circle cx={c} cy={c} r={W / 2 - 8} fill="none" stroke="var(--line-soft)" strokeWidth="1" opacity="0.4" />
      <circle cx={c} cy={c} r={W / 2 - 30} fill="none" stroke="var(--line-soft)" strokeWidth="1" opacity="0.3" />
      {nodes.map((n, i) => (
        <g key={i}>
          <line x1={c} y1={c} x2={n.x} y2={n.y} stroke="var(--ink-faint)" strokeWidth="0.5" opacity={n.b * 0.25} />
          <circle cx={n.x} cy={n.y} r="1.2" fill="var(--ink-dim)" opacity={n.b} />
        </g>
      ))}
      <circle cx={cx} cy={cy} r="9" fill="none" stroke="var(--accent)" strokeWidth="1.5" />
      <circle cx={cx} cy={cy} r="3" fill="var(--accent)" />
    </svg>
  );
}

export function XYPadGrid({ cursor = [0.5, 0.5] }) {
  const W = 240;
  const cx = cursor[0] * W;
  const cy = cursor[1] * W;
  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${W} ${W}`} preserveAspectRatio="xMidYMid meet">
      <rect width={W} height={W} fill="var(--bg-0)" rx="4" />
      {[0, 1, 2, 3].map((i) => (
        <line key={`h${i}`} x1="0" x2={W} y1={(i + 1) * W / 5} y2={(i + 1) * W / 5} stroke="var(--line-soft)" strokeWidth="0.5" />
      ))}
      {[0, 1, 2, 3].map((i) => (
        <line key={`v${i}`} y1="0" y2={W} x1={(i + 1) * W / 5} x2={(i + 1) * W / 5} stroke="var(--line-soft)" strokeWidth="0.5" />
      ))}
      <circle cx={cx} cy={cy} r="9" fill="none" stroke="var(--accent)" strokeWidth="1.5" />
      <circle cx={cx} cy={cy} r="3" fill="var(--accent)" />
    </svg>
  );
}

// ────────────────────────────────────────────────────────────────
// ModLane display — step or curve mode

export function ModLane({ points = [0.3, 0.7, 0.3, 0.8, 0.2, 0.5, 0.2, 0.8, 0.4, 0.6, 0.3, 0.7], mode = 'step', color = 'var(--accent)' }) {
  const W = 260, H = 120;
  const stepX = (i) => 8 + (i / Math.max(1, points.length - 1)) * (W - 16);
  const y = (v) => 10 + (1 - v) * (H - 20);
  let path = '';
  if (mode === 'step') {
    points.forEach((v, i) => {
      const x = stepX(i);
      if (i === 0) path += `M ${x} ${y(v)}`;
      else path += ` L ${x} ${y(points[i - 1])} L ${x} ${y(v)}`;
    });
  } else {
    points.forEach((v, i) => {
      const x = stepX(i);
      if (i === 0) path += `M ${x} ${y(v)}`;
      else {
        const xp = stepX(i - 1), yp = y(points[i - 1]);
        const xc = (xp + x) / 2;
        path += ` Q ${xc} ${yp} ${x} ${y(v)}`;
      }
    });
  }
  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
      <rect width={W} height={H} fill="var(--bg-0)" rx="3" />
      <line x1="0" x2={W} y1={H / 2} y2={H / 2} stroke="var(--line-soft)" strokeWidth="0.5" />
      <path d={path} fill="none" stroke={color} strokeWidth="1.8" />
      {points.map((v, i) => (
        <circle key={i} cx={stepX(i)} cy={y(v)} r="3" fill={color} />
      ))}
    </svg>
  );
}

// ────────────────────────────────────────────────────────────────
// Keyboard strip

export function Keyboard({ octaves = 4, pressed = [] }) {
  const whitesPerOct = 7;
  const totalWhites = octaves * whitesPerOct;
  const whiteW = 100 / totalWhites;
  const blackPattern = [true, true, false, true, true, true, false]; // after C,D,F,G,A
  const whiteNames = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];

  const pressedSet = new Set(pressed);

  const whites = [];
  const blacks = [];
  let midi = 24; // C1-ish
  for (let o = 0; o < octaves; o++) {
    for (let w = 0; w < whitesPerOct; w++) {
      const idx = o * whitesPerOct + w;
      const x = idx * whiteW;
      whites.push(
        <rect
          key={`w${idx}`}
          x={`${x}%`} y="0"
          width={`${whiteW}%`}
          height="100%"
          fill={pressedSet.has(midi) ? 'var(--accent)' : 'var(--ink)'}
          stroke="var(--line)"
          strokeWidth="0.3"
        />
      );
      // one black key per white except at E, B
      if (blackPattern[w]) {
        const bx = x + whiteW * 0.7;
        const bw = whiteW * 0.6;
        blacks.push(
          <rect
            key={`b${idx}`}
            x={`${bx}%`} y="0"
            width={`${bw}%`}
            height="62%"
            fill={pressedSet.has(midi + 1) ? 'var(--accent)' : 'var(--bg-0)'}
            stroke="var(--line)"
            strokeWidth="0.3"
          />
        );
      }
      midi += blackPattern[w] ? 2 : 1;
    }
  }

  return (
    <svg width="100%" height="100%" preserveAspectRatio="none" style={{ display: 'block' }}>
      {whites}
      {blacks}
    </svg>
  );
}
