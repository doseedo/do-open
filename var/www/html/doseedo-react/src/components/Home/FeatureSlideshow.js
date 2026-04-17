/**
 * FeatureSlideshow — vanilla React port of /home/ani/slide1-5
 * All shared constants sourced from /home/ani/shared:
 *   C, Icons, instrumentPool, seededRandom, WaveformSVG,
 *   NoiseWaveform, glassPanel, SlideFrame, TransportBar, Timeline, TrackRow
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import styles from './FeatureSlideshow.module.css';

// ═══════════════════════════════════════════════════════════════
// Shared — from /home/ani/shared
// ═══════════════════════════════════════════════════════════════
const C = {
  text: '#fff', textSec: '#ccc', textMuted: '#888', textDim: '#555',
};

const sz = { width: 14, height: 14, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round', strokeLinejoin: 'round' };
const Icons = {
  play:    <svg {...sz}><polygon points="5 3 19 12 5 21" fill="currentColor" stroke="none"/></svg>,
  pause:   <svg {...sz}><rect x="6" y="4" width="4" height="16" fill="currentColor" stroke="none"/><rect x="14" y="4" width="4" height="16" fill="currentColor" stroke="none"/></svg>,
  stop:    <svg {...sz}><rect x="6" y="6" width="12" height="12" rx="1" fill="currentColor" stroke="none"/></svg>,
  mic:     <svg {...sz}><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>,
  violin:  <svg {...sz}><path d="M12 2v6M9 8h6M7 12c0 2.8 2.2 5 5 5s5-2.2 5-5-2.2-5-5-5-5 2.2-5 5z"/><path d="M12 17v5"/></svg>,
  piano:   <svg {...sz}><rect x="2" y="4" width="20" height="16" rx="2"/><line x1="7" y1="4" x2="7" y2="14"/><line x1="12" y1="4" x2="12" y2="14"/><line x1="17" y1="4" x2="17" y2="14"/><line x1="2" y1="14" x2="22" y2="14"/></svg>,
  wind:    <svg {...sz}><path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"/></svg>,
  drums:   <svg {...sz}><ellipse cx="12" cy="10" rx="9" ry="5"/><path d="M3 10v4c0 2.8 4 5 9 5s9-2.2 9-5v-4"/><line x1="7" y1="3" x2="10" y2="8"/><line x1="17" y1="3" x2="14" y2="8"/></svg>,
  bass:    <svg {...sz}><path d="M19 3l2 2M16 6l-2 2M11 11c-2.5 2.5-6 2-6 2s-.5-3.5 2-6l7-7 3 3-6 6z"/><circle cx="7" cy="17" r="3"/></svg>,
  synth:   <svg {...sz}><rect x="2" y="6" width="20" height="12" rx="2"/><line x1="6" y1="10" x2="6" y2="14"/><line x1="10" y1="10" x2="10" y2="14"/><line x1="14" y1="10" x2="14" y2="14"/><line x1="18" y1="10" x2="18" y2="14"/></svg>,
  wand:    <svg {...sz}><path d="M15 4V2M15 16v-2M8 9h2M20 9h2M17.8 11.8l1.4 1.4M17.8 6.2l1.4-1.4M12.2 11.8l-1.4 1.4M3 21l9-9"/></svg>,
  split:   <svg {...sz}><path d="M16 3h5v5M4 20L21 3M21 16v5h-5M15 15l6 6M4 4l5 5"/></svg>,
  sliders: <svg {...sz}><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>,
};

// instrumentPool from shared
const instrumentPool = [
  { name: 'Vocals',  icon: Icons.mic,    color: 'rgba(168,127,255,0.7)', colorBg: 'rgba(168,127,255,0.06)' },
  { name: 'Strings', icon: Icons.violin, color: 'rgba(102,126,234,0.7)', colorBg: 'rgba(102,126,234,0.06)' },
  { name: 'Piano',   icon: Icons.piano,  color: 'rgba(72,202,228,0.7)',  colorBg: 'rgba(72,202,228,0.06)'  },
  { name: 'Winds',   icon: Icons.wind,   color: 'rgba(100,220,150,0.7)', colorBg: 'rgba(100,220,150,0.06)' },
  { name: 'Drums',   icon: Icons.drums,  color: 'rgba(255,165,0,0.7)',   colorBg: 'rgba(255,165,0,0.06)'   },
  { name: 'Bass',    icon: Icons.bass,   color: 'rgba(255,100,100,0.7)', colorBg: 'rgba(255,100,100,0.06)' },
  { name: 'Synth',   icon: Icons.synth,  color: 'rgba(200,100,255,0.7)', colorBg: 'rgba(200,100,255,0.06)' },
];

// seededRandom from shared
function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

// WaveformSVG path from shared
function waveformPath(seed, width, height) {
  const rng = seededRandom(seed);
  const mid = height / 2;
  let d = '';
  for (let x = 0; x < width; x += 3) {
    const env = 0.4 + 0.6 * Math.sin((x / width) * Math.PI);
    const burst = rng() > 0.87 ? 1.3 : 1;
    const amp = rng() * env * (height * 0.42) * burst;
    d += `M${x},${mid - amp}V${mid + amp}`;
  }
  return d;
}

// WaveformSVG component from shared
function WaveformSVG({ color, seed, width, height }) {
  const path = waveformPath(seed, width, height);
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <path d={path} stroke={color} strokeWidth={2} fill="none" strokeLinecap="round" />
    </svg>
  );
}

// NoiseWaveform canvas from shared
function NoiseWaveform({ width, height, color = 'rgba(139,92,246,0.6)', settling = false }) {
  const canvasRef = useRef(null);
  const animRef = useRef(0);
  const prevAmps = useRef(null);
  const startRef = useRef(Date.now());

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.scale(dpr, dpr);
    const lineCount = Math.floor(width / 4);
    const centerY = height / 2;
    if (!prevAmps.current) prevAmps.current = new Array(lineCount).fill(0);

    const animate = () => {
      const elapsed = (Date.now() - startRef.current) / 1000;
      ctx.clearRect(0, 0, width, height);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      for (let i = 0; i < lineCount; i++) {
        const x = (i / lineCount) * width;
        const prev = prevAmps.current[i] || 0;
        let amp;
        if (settling) {
          const p = Math.min(elapsed / 0.5, 1);
          const ease = 1 - Math.pow(1 - p, 3);
          amp = prev * (1 - ease);
        } else {
          const target = (Math.random() - 0.5) * height * 0.6;
          amp = prev + (target - prev) * 0.08;
        }
        prevAmps.current[i] = amp;
        ctx.beginPath();
        ctx.moveTo(x, centerY - amp);
        ctx.lineTo(x, centerY + amp);
        ctx.stroke();
      }
      if (!settling || elapsed < 0.6) animRef.current = requestAnimationFrame(animate);
    };
    animate();
    return () => cancelAnimationFrame(animRef.current);
  }, [width, height, color, settling]);

  useEffect(() => {
    if (settling) startRef.current = Date.now();
  }, [settling]);

  return <canvas ref={canvasRef} style={{ width, height, display: 'block' }} />;
}

// glassPanel helper from shared
function glassPanelStyle(glowRGB) {
  return {
    background: 'rgba(10,10,16,0.5)',
    backdropFilter: 'blur(32px) saturate(170%)',
    WebkitBackdropFilter: 'blur(32px) saturate(170%)',
    borderRadius: 12,
    border: '1px solid rgba(255,255,255,0.1)',
    boxShadow: `0 6px 24px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,255,255,0.08),0 0 40px rgba(${glowRGB},0.06)`,
  };
}

// SlideFrame from shared
function SlideFrame({ glowRGB, headline, copy, children }) {
  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden', background: '#050508', fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", userSelect: 'none' }}>
      <div style={{ position: 'absolute', inset: -40, pointerEvents: 'none', background: `radial-gradient(ellipse at 50% 38%, rgba(${glowRGB},0.25) 0%, rgba(${glowRGB},0.1) 30%, rgba(${glowRGB},0.04) 55%, transparent 78%)` }} />
      <div style={{ position: 'absolute', top: 32, left: 28, right: 28, bottom: 106 }}>
        {children}
      </div>
      <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '14px 28px 22px', background: 'linear-gradient(to top, rgba(5,5,8,0.97) 0%, rgba(5,5,8,0.75) 60%, transparent 100%)' }}>
        <div style={{ fontSize: 19, fontWeight: 700, color: '#fff', marginBottom: 4, textShadow: `0 0 40px rgba(${glowRGB},0.4)`, letterSpacing: '-0.2px' }}>{headline}</div>
        <div style={{ fontSize: 12.5, color: 'rgba(255,255,255,0.5)', lineHeight: 1.5 }}>{copy}</div>
      </div>
    </div>
  );
}

// TransportBar from shared
function TransportBar({ glowRGB, timeRef }) {
  return (
    <div style={{ height: 32, display: 'flex', alignItems: 'center', padding: '0 10px', gap: 6, background: 'rgba(8,8,14,0.3)', borderBottom: '1px solid rgba(255,255,255,0.08)', borderRadius: '12px 12px 0 0', flexShrink: 0 }}>
      <div style={{ width: 24, height: 24, borderRadius: 6, border: `1px solid rgba(${glowRGB},0.5)`, background: `rgba(${glowRGB},0.2)`, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{Icons.pause}</div>
      <div style={{ width: 24, height: 24, borderRadius: 6, border: '1px solid rgba(255,255,255,0.1)', background: 'rgba(255,255,255,0.06)', color: '#ccc', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{Icons.stop}</div>
      <div ref={timeRef} style={{ color: '#fff', fontSize: 11, fontWeight: 600, minWidth: 40, padding: '0 6px', background: 'rgba(6,6,12,0.2)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 6, height: 24, display: 'flex', alignItems: 'center', justifyContent: 'center', fontVariantNumeric: 'tabular-nums' }}>0:00</div>
      <div style={{ flex: 1 }} />
      <div style={{ padding: '3px 8px', borderRadius: 6, fontSize: 10, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: '#888' }}>120 BPM</div>
    </div>
  );
}

// Timeline from shared
function Timeline({ labelW = 100 }) {
  return (
    <div style={{ height: 24, display: 'flex', borderBottom: '1px solid rgba(255,255,255,0.06)', background: 'rgba(6,6,12,0.15)', flexShrink: 0 }}>
      <div style={{ width: labelW, minWidth: labelW, background: 'rgba(6,6,12,0.15)', borderRight: '1px solid rgba(255,255,255,0.06)', display: 'flex', alignItems: 'center', paddingLeft: 10 }}>
        <span style={{ fontSize: 9, color: '#555' }}>+ Add Track</span>
      </div>
      <div style={{ flex: 1, position: 'relative', background: 'rgba(6,6,10,0.1)' }}>
        {[0, 5, 10, 15, 20, 25, 30].map(t => (
          <div key={t} style={{ position: 'absolute', left: `${(t / 30) * 100}%`, top: 0, height: '100%' }}>
            <span style={{ position: 'absolute', top: 4, left: 3, color: '#555', fontSize: 8, whiteSpace: 'nowrap' }}>{t}s</span>
            <div style={{ position: 'absolute', bottom: 0, left: 0, width: 1, height: 8, background: '#333' }} />
          </div>
        ))}
      </div>
    </div>
  );
}

// TrackRow from shared
function TrackRow({ track, labelW = 100, trackH = 52, glowRGB }) {
  const clipW = Math.max(50, track.widthFrac * 420 - 6);
  const startX = labelW + track.startFrac * 420 + 4;
  return (
    <div style={{ position: 'relative', height: trackH, borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
      <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: labelW, display: 'flex', alignItems: 'center', padding: '0 6px', gap: 6, background: 'rgba(8,8,14,0.2)', backdropFilter: 'blur(20px)', borderRight: '1px solid rgba(255,255,255,0.06)' }}>
        <div style={{ color: track.color, opacity: 0.7, flexShrink: 0 }}>{track.icon}</div>
        <span style={{ fontSize: 10, color: '#ccc', fontWeight: 500 }}>{track.name}</span>
      </div>
      <div style={{ position: 'absolute', top: 3, left: startX, width: clipW, height: trackH - 6, overflow: 'hidden', borderRadius: 8 }}>
        {track.isPlaceholder ? (
          <div style={{ position: 'absolute', inset: 0, borderRadius: 8, background: `rgba(${glowRGB || '139,92,246'},0.04)`, border: `1px solid rgba(${glowRGB || '139,92,246'},0.12)` }}>
            <NoiseWaveform width={Math.max(50, clipW - 4)} height={trackH - 10} color={track.color} />
          </div>
        ) : (
          <div style={{ position: 'absolute', inset: 0, background: `linear-gradient(180deg, ${track.colorBg}, ${track.colorBg.replace('0.06', '0.02')})`, backdropFilter: 'blur(14px)', borderLeft: `2px solid ${track.color}`, borderRight: `1px solid ${track.color.replace('0.7', '0.12')}`, borderTop: `1px solid ${track.color.replace('0.7', '0.15')}`, borderBottom: `1px solid ${track.color.replace('0.7', '0.05')}`, borderRadius: 8, boxShadow: '0 4px 16px rgba(0,0,0,0.3),inset 0 1px 0 rgba(255,255,255,0.08)' }}>
            <div style={{ position: 'absolute', left: 6, top: '50%', transform: 'translateY(-50%)' }}>
              <WaveformSVG color={track.color} seed={track.seed} width={Math.max(50, clipW - 12)} height={trackH - 18} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Slide 1 — "Turn your songs back into sessions."
// from /home/ani/slide1 — rotating waveform → splits into 4 stems
// GLOW: 120,60,200  stemDefs: Vocals, Celeste, Strings, Winds
// ═══════════════════════════════════════════════════════════════
const stemDefs = [
  { name: 'Vocals',  rgb: [236, 72,  153], color: 'rgba(236,72,153,0.7)',  colorBg: 'rgba(236,72,153,0.06)',  icon: Icons.mic,    seed: 501 },
  { name: 'Celeste', rgb: [59,  130, 246], color: 'rgba(59,130,246,0.7)',  colorBg: 'rgba(59,130,246,0.06)',  icon: Icons.piano,  seed: 502 },
  { name: 'Strings', rgb: [16,  185, 129], color: 'rgba(16,185,129,0.7)',  colorBg: 'rgba(16,185,129,0.06)',  icon: Icons.violin, seed: 503 },
  { name: 'Winds',   rgb: [245, 158, 11],  color: 'rgba(245,158,11,0.7)',  colorBg: 'rgba(245,158,11,0.06)',  icon: Icons.wind,   seed: 504 },
];

function buildStemWaveData(barCount) {
  const n = stemDefs.length;
  const stemSeeds = [101, 247, 389, 513];
  const stemAmps = [];
  for (let s = 0; s < n; s++) {
    const rng = seededRandom(stemSeeds[s]);
    const amps = [];
    for (let b = 0; b < barCount; b++) {
      const pos = b / barCount;
      let env;
      if (s === 0) env = 0.2 + 0.8 * Math.pow(Math.sin(pos * Math.PI), 1.2);
      else if (s === 1) env = 0.1 + 0.9 * Math.pow(Math.sin(pos * Math.PI * 3 + 0.5), 2) * 0.7;
      else if (s === 2) env = 0.15 + 0.85 * Math.pow(pos, 0.6) * Math.sin(pos * Math.PI * 0.9);
      else env = 0.1 + 0.9 * (0.5 + 0.5 * Math.cos(pos * Math.PI * 2 - Math.PI)) * 0.6;
      amps.push(rng() * env * (rng() > 0.88 ? 1.4 : 1));
    }
    stemAmps.push(amps);
  }
  const masterAmps = [];
  for (let b = 0; b < barCount; b++) {
    let mx = 0;
    for (let s = 0; s < n; s++) if (stemAmps[s][b] > mx) mx = stemAmps[s][b];
    masterAmps.push(mx);
  }
  const peak = Math.max(...masterAmps);
  for (let b = 0; b < barCount; b++) {
    masterAmps[b] /= peak;
    for (let s = 0; s < n; s++) stemAmps[s][b] /= peak;
  }
  return { masterAmps, stemAmps };
}

function RotatingWaveCanvas({ startTime, freeze, width, height }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(0);
  const dataRef = useRef(null);
  const CYCLE_MS = 1300;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const barW = 3, barGap = 1.5, barR = barW / 2;
    const barCount = Math.floor(width / (barW + barGap));
    if (!dataRef.current || dataRef.current.masterAmps.length !== barCount) {
      dataRef.current = buildStemWaveData(barCount);
    }
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    const { masterAmps, stemAmps } = dataRef.current;
    const n = stemDefs.length;
    const centerY = height / 2;
    const maxAmp = height * 0.44;

    function smooth(amps, r = 3) {
      return amps.map((_, i) => {
        let sum = 0, cnt = 0;
        for (let j = Math.max(0, i - r); j <= Math.min(amps.length - 1, i + r); j++) { sum += amps[j]; cnt++; }
        return sum / cnt;
      });
    }
    function drawWave(rawAmps, style) {
      const amps = smooth(rawAmps);
      ctx.fillStyle = style;
      for (let b = 0; b < barCount; b++) {
        const x = b * (barW + barGap);
        if (x > width) break;
        const h = amps[b] * maxAmp;
        if (h < 1) continue;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x, centerY - h, barW, h * 2, barR);
        else ctx.rect(x, centerY - h, barW, h * 2);
        ctx.fill();
      }
    }

    const animate = () => {
      const t = (Date.now() - startTime) / CYCLE_MS;
      ctx.clearRect(0, 0, width, height);
      drawWave(masterAmps, 'rgba(255,255,255,0.1)');
      if (!freeze) {
        for (let s = 0; s < n; s++) {
          const ph = (((t - s / n) % 1) + 1) % 1;
          const alpha = Math.max(0, Math.cos(ph * 2 * Math.PI));
          if (alpha > 0.01) {
            const [r, g, b] = stemDefs[s].rgb;
            drawWave(stemAmps[s], `rgba(${r},${g},${b},${alpha * 0.85})`);
          }
        }
      } else {
        drawWave(masterAmps, 'rgba(168,127,255,0.5)');
      }
      rafRef.current = requestAnimationFrame(animate);
    };
    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [width, height, startTime, freeze]);

  return <canvas ref={canvasRef} style={{ display: 'block' }} />;
}

function Slide1({ active }) {
  const GLOW = '120,60,200';
  const LW = 100, TH = 52, CYCLE = 6000, ROTATE_TIME = 1300;
  const [phase, setPhase] = useState('rotating');
  const [revealedStems, setRevealedStems] = useState([]);
  const startRef = useRef(Date.now());
  const timersRef = useRef([]);

  const clearTimers = useCallback(() => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
  }, []);

  useEffect(() => {
    if (!active) { clearTimers(); return; }

    const run = () => {
      clearTimers();
      setPhase('rotating');
      setRevealedStems([]);
      startRef.current = Date.now();

      const t = (ms, fn) => { const id = setTimeout(fn, ms); timersRef.current.push(id); };
      t(ROTATE_TIME, () => setPhase('splitting'));
      t(ROTATE_TIME + 700, () => setPhase('generating'));
      stemDefs.forEach((_, i) => t(ROTATE_TIME + 1400 + i * 400, () => setRevealedStems(p => [...p, i])));
      t(ROTATE_TIME + 1400 + stemDefs.length * 400 + 200, () => setPhase('done'));
      timersRef.current.push(setTimeout(run, CYCLE));
    };
    run();
    return clearTimers;
  }, [active, clearTimers]);

  const showMix = phase === 'rotating';
  const isSplit = phase === 'splitting' || phase === 'generating' || phase === 'done';

  return (
    <SlideFrame glowRGB={GLOW} headline="Turn your songs back into sessions." copy="State-of-the-art source separation and reverse FX models allow instant stems with equivalent dry recordings — from only a master recording.">
      <div style={{ position: 'absolute', inset: 0, ...glassPanelStyle(GLOW), overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <TransportBar glowRGB={GLOW} />
        <Timeline labelW={LW} />
        <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          {/* Row backgrounds */}
          {[0, 1, 2, 3].map(i => (
            <div key={i} style={{ position: 'absolute', top: i * TH, left: 0, right: 0, height: TH, borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: LW, background: 'rgba(8,8,14,0.2)', borderRight: '1px solid rgba(255,255,255,0.06)' }} />
            </div>
          ))}

          {/* Full Mix track */}
          {showMix && (
            <>
              <div style={{ position: 'absolute', top: TH * 0.5, left: 0, width: LW, height: TH * 3, display: 'flex', alignItems: 'center', padding: '0 6px', gap: 6, background: 'rgba(8,8,14,0.4)', backdropFilter: 'blur(20px)', borderRight: '1px solid rgba(255,255,255,0.06)', zIndex: 2, transition: 'opacity 0.4s ease', opacity: showMix ? 1 : 0 }}>
                <div style={{ color: 'rgba(168,127,255,0.7)', opacity: 0.7 }}>{Icons.mic}</div>
                <span style={{ fontSize: 10, color: '#ccc', fontWeight: 500 }}>Full Mix</span>
              </div>
              <div style={{ position: 'absolute', top: TH * 0.5 + 3, left: LW + 8, right: 8, height: TH * 3 - 6, borderRadius: 8, overflow: 'hidden', background: 'linear-gradient(180deg,rgba(168,127,255,0.08),rgba(168,127,255,0.02))', borderLeft: '2px solid rgba(168,127,255,0.7)', borderRight: '1px solid rgba(168,127,255,0.12)', borderTop: '1px solid rgba(168,127,255,0.15)', boxShadow: '0 4px 16px rgba(0,0,0,0.3),inset 0 1px 0 rgba(255,255,255,0.08)' }}>
                <RotatingWaveCanvas startTime={startRef.current} freeze={false} width={720} height={TH * 3 - 10} />
              </div>
            </>
          )}

          {/* Status badge */}
          {(phase === 'rotating' || phase === 'splitting') && (
            <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', display: 'flex', alignItems: 'center', gap: 8, padding: '7px 14px', borderRadius: 20, background: `rgba(${GLOW},0.15)`, backdropFilter: 'blur(20px)', border: `1px solid rgba(${GLOW},0.3)`, zIndex: 10, color: '#fff', fontSize: 11, fontWeight: 600 }}>
              <span style={{ color: `rgba(${GLOW},0.9)` }}>{phase === 'rotating' ? Icons.wand : Icons.split}</span>
              {phase === 'rotating' ? 'Analyzing stems...' : 'Separating stems...'}
              <div className={styles.spinner} style={{ borderTopColor: `rgba(${GLOW},0.8)` }} />
            </div>
          )}

          {/* Individual stem tracks */}
          {isSplit && stemDefs.map((stem, i) => {
            const isRevealed = revealedStems.includes(i);
            return (
              <div key={stem.name} style={{ position: 'absolute', top: i * TH, left: 0, right: 0, height: TH, opacity: 1, transition: 'opacity 0.3s ease' }}>
                <div style={{ position: 'absolute', top: 0, left: 0, width: LW, height: TH, display: 'flex', alignItems: 'center', padding: '0 6px', gap: 6, background: 'rgba(8,8,14,0.4)', backdropFilter: 'blur(20px)', borderRight: '1px solid rgba(255,255,255,0.06)', borderBottom: '1px solid rgba(255,255,255,0.04)', zIndex: 2 }}>
                  <div style={{ color: stem.color, opacity: 0.7 }}>{stem.icon}</div>
                  <span style={{ fontSize: 10, color: '#ccc', fontWeight: 500 }}>{stem.name}</span>
                </div>
                <div style={{ position: 'absolute', top: 3, left: LW + 8, right: 8, height: TH - 6, borderRadius: 8, overflow: 'hidden' }}>
                  <div style={{ position: 'absolute', inset: 0, borderRadius: 8, background: `rgba(${GLOW},0.04)`, border: `1px solid rgba(${GLOW},0.12)`, opacity: isRevealed ? 0 : 1, transition: 'opacity 0.4s ease' }}>
                    {!isRevealed && <NoiseWaveform width={500} height={TH - 10} color={stem.color} settling={isRevealed} />}
                  </div>
                  <div style={{ position: 'absolute', inset: 0, opacity: isRevealed ? 1 : 0, transition: 'opacity 0.5s ease', background: `linear-gradient(180deg,${stem.colorBg},${stem.colorBg.replace('0.06', '0.02')})`, backdropFilter: 'blur(14px)', borderLeft: `2px solid ${stem.color}`, borderRight: `1px solid ${stem.color.replace('0.7', '0.12')}`, borderTop: `1px solid ${stem.color.replace('0.7', '0.15')}`, borderBottom: `1px solid ${stem.color.replace('0.7', '0.05')}`, borderRadius: 8, boxShadow: `0 4px 16px rgba(0,0,0,0.3),inset 0 1px 0 rgba(255,255,255,0.08)` }}>
                    <div style={{ position: 'absolute', left: 6, top: '50%', transform: 'translateY(-50%)' }}>
                      <WaveformSVG color={stem.color} seed={stem.seed} width={700} height={TH - 20} />
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </SlideFrame>
  );
}

// ═══════════════════════════════════════════════════════════════
// Slide 2 — "Track-aware generation."
// from /home/ani/slide2 — playhead + selection → noise → resolved waveforms
// GLOW: 100,160,255  TRACKS: Piano, Bass, Drums, Trumpet
// ═══════════════════════════════════════════════════════════════
const s2Tracks = [
  { name: 'Piano',   color: 'rgba(72,202,228,0.8)',  colorBg: 'rgba(72,202,228,0.07)',  icon: Icons.piano, seed: 701, genSeed: 801 },
  { name: 'Bass',    color: 'rgba(255,100,100,0.8)',  colorBg: 'rgba(255,100,100,0.07)', icon: Icons.bass,  seed: 702, genSeed: 802 },
  { name: 'Drums',   color: 'rgba(255,165,0,0.8)',    colorBg: 'rgba(255,165,0,0.07)',   icon: Icons.drums, seed: 703, genSeed: 803 },
  { name: 'Trumpet', color: 'rgba(245,158,11,0.8)',   colorBg: 'rgba(245,158,11,0.07)',  icon: Icons.wind,  seed: 704, genSeed: 804 },
];
const S2_GLOW = '100,160,255';
const S2_LW = 100, S2_TH = 52, S2_PAD = 8;
const SEL_START = 0.35, SEL_END = 0.78;
const TRACK_AREA_W = 700;

function Slide2({ active }) {
  const [phase, setPhase] = useState('idle');
  const [resolved, setResolved] = useState([]);
  const [selW, setSelW] = useState(0);
  const playheadRef = useRef(null);
  const playheadCapRef = useRef(null);
  const timeRef = useRef(null);
  const spinnerRef = useRef(null);
  const rafRef = useRef(0);
  const timersRef = useRef([]);
  const playStart = useRef(Date.now());
  const spinAngle = useRef(0);
  const lastNow = useRef(0);

  const clearAll = useCallback(() => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
    cancelAnimationFrame(rafRef.current);
  }, []);

  useEffect(() => {
    if (!active) { clearAll(); return; }

    const rafTick = (now) => {
      const dt = now - (lastNow.current || now);
      lastNow.current = now;
      const elapsed = (now - playStart.current) / 1000;
      const ph = (elapsed % 30) / 30;
      const phPx = S2_LW + S2_PAD + ph * TRACK_AREA_W;
      if (playheadRef.current) playheadRef.current.style.left = phPx + 'px';
      if (playheadCapRef.current) playheadCapRef.current.style.left = (phPx - 4.5) + 'px';
      if (timeRef.current) timeRef.current.textContent = `0:${String(Math.floor(ph * 30)).padStart(2, '0')}`;
      spinAngle.current = (spinAngle.current + dt * 0.36) % 360;
      if (spinnerRef.current) spinnerRef.current.style.transform = `rotate(${spinAngle.current}deg)`;
      rafRef.current = requestAnimationFrame(rafTick);
    };
    rafRef.current = requestAnimationFrame(rafTick);

    const t = (ms, fn) => { const id = setTimeout(fn, ms); timersRef.current.push(id); };
    const run = () => {
      clearAll();
      setPhase('idle');
      setResolved([]);
      setSelW(0);
      playStart.current = Date.now() - (SEL_START * 30000 - 6250);
      lastNow.current = 0;
      rafRef.current = requestAnimationFrame(rafTick);

      t(1500, () => {
        setPhase('selecting');
        const start = Date.now();
        const maxW = (SEL_END - SEL_START) * TRACK_AREA_W;
        const grow = () => {
          const p = Math.min(1, (Date.now() - start) / 800);
          const ease = 1 - Math.pow(1 - p, 3);
          setSelW(maxW * ease);
          if (p < 1) timersRef.current.push(setTimeout(grow, 16));
        };
        grow();
      });
      t(3000, () => setPhase('generating'));
      t(4500, () => setPhase('resolving'));
      s2Tracks.forEach((_, i) => t(4500 + i * 300, () => setResolved(p => [...p, i])));
      t(6500, () => { setPhase('playing'); setSelW(0); });
      timersRef.current.push(setTimeout(run, 10000));
    };
    run();
    return clearAll;
  }, [active, clearAll]);

  const selStartPx = S2_LW + S2_PAD + SEL_START * TRACK_AREA_W;
  const maxSelW = (SEL_END - SEL_START) * TRACK_AREA_W;

  return (
    <SlideFrame glowRGB={S2_GLOW} headline="Track-aware generation." copy="Select any region across your session. Doseedo reads every existing track and generates new material that fits seamlessly.">
      <div style={{ position: 'absolute', inset: 0, ...glassPanelStyle(S2_GLOW), overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <TransportBar glowRGB={S2_GLOW} timeRef={timeRef} />
        <Timeline labelW={S2_LW} />
        <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          {s2Tracks.map((t, i) => {
            const beforeW = SEL_START * TRACK_AREA_W - 2;
            const genW = maxSelW - 2;
            const afterLeft = selStartPx + maxSelW + 2;
            const afterW = Math.max(0, S2_LW + S2_PAD + TRACK_AREA_W - afterLeft - S2_PAD);
            const isResolved = resolved.includes(i);
            return (
              <div key={t.name} style={{ position: 'absolute', top: i * S2_TH, left: 0, right: 0, height: S2_TH, borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: S2_LW, display: 'flex', alignItems: 'center', padding: '0 6px', gap: 6, background: 'rgba(8,8,14,0.2)', backdropFilter: 'blur(20px)', borderRight: '1px solid rgba(255,255,255,0.06)' }}>
                  <div style={{ color: t.color, opacity: 0.7 }}>{t.icon}</div>
                  <span style={{ fontSize: 10, color: '#ccc', fontWeight: 500 }}>{t.name}</span>
                </div>
                {/* Before clip */}
                <div style={{ position: 'absolute', top: 3, left: S2_LW + S2_PAD, width: beforeW, height: S2_TH - 6, borderRadius: 8, background: `linear-gradient(180deg,${t.colorBg},${t.colorBg.replace('0.07', '0.02')})`, backdropFilter: 'blur(14px)', borderLeft: `2px solid ${t.color}`, overflow: 'hidden' }}>
                  <div style={{ position: 'absolute', left: 4, top: '50%', transform: 'translateY(-50%)' }}>
                    <WaveformSVG color={t.color} seed={t.seed} width={Math.max(40, beforeW - 8)} height={S2_TH - 18} />
                  </div>
                </div>
                {/* Gen region */}
                <div style={{ position: 'absolute', top: 3, left: selStartPx, width: genW, height: S2_TH - 6, borderRadius: 8, overflow: 'hidden' }}>
                  {(phase === 'generating' || phase === 'resolving') && !isResolved && (
                    <div style={{ position: 'absolute', inset: 0, borderRadius: 8, background: `rgba(${S2_GLOW},0.04)`, border: `1px solid rgba(${S2_GLOW},0.12)` }}>
                      <NoiseWaveform width={Math.max(40, genW - 4)} height={S2_TH - 10} color={t.color} />
                    </div>
                  )}
                  {isResolved && (
                    <div style={{ position: 'absolute', inset: 0, borderRadius: 8, background: `linear-gradient(180deg,${t.colorBg},${t.colorBg.replace('0.07', '0.02')})`, backdropFilter: 'blur(14px)', borderLeft: `2px solid ${t.color}`, overflow: 'hidden' }}>
                      <div style={{ position: 'absolute', left: 4, top: '50%', transform: 'translateY(-50%)' }}>
                        <WaveformSVG color={t.color} seed={t.genSeed} width={Math.max(40, genW - 8)} height={S2_TH - 18} />
                      </div>
                    </div>
                  )}
                </div>
                {/* After clip */}
                {afterW > 20 && (
                  <div style={{ position: 'absolute', top: 3, left: afterLeft, width: afterW, height: S2_TH - 6, borderRadius: 8, background: `linear-gradient(180deg,${t.colorBg},${t.colorBg.replace('0.07', '0.02')})`, backdropFilter: 'blur(14px)', borderLeft: `2px solid ${t.color}`, overflow: 'hidden' }}>
                    <div style={{ position: 'absolute', left: 4, top: '50%', transform: 'translateY(-50%)' }}>
                      <WaveformSVG color={t.color} seed={t.seed + 100} width={Math.max(20, afterW - 8)} height={S2_TH - 18} />
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {/* Selection box */}
          {selW > 0 && (
            <div style={{ position: 'absolute', top: 0, bottom: 0, left: selStartPx, width: selW, border: '1px solid rgba(255,255,255,0.25)', background: 'rgba(255,255,255,0.04)', pointerEvents: 'none', zIndex: 4, borderRadius: 3 }} />
          )}

          {/* Playhead */}
          <div ref={playheadRef} style={{ position: 'absolute', top: 0, bottom: 0, width: 1, background: 'rgba(255,255,255,0.7)', pointerEvents: 'none', zIndex: 5, left: S2_LW + S2_PAD }} />
          <div ref={playheadCapRef} style={{ position: 'absolute', top: -4, width: 9, height: 9, background: 'rgba(255,255,255,0.9)', clipPath: 'polygon(50% 100%, 0 0, 100% 0)', pointerEvents: 'none', zIndex: 6 }} />

          {/* Badge */}
          {(phase === 'generating' || phase === 'resolving') && (
            <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', display: 'flex', alignItems: 'center', gap: 8, padding: '7px 14px', borderRadius: 20, background: `rgba(${S2_GLOW},0.15)`, backdropFilter: 'blur(20px)', border: `1px solid rgba(${S2_GLOW},0.3)`, zIndex: 10, color: '#fff', fontSize: 11, fontWeight: 600 }}>
              <span style={{ color: `rgba(${S2_GLOW},0.9)` }}>{Icons.wand}</span>
              Generating...
              <div ref={spinnerRef} className={styles.spinner} style={{ borderTopColor: `rgba(${S2_GLOW},0.8)` }} />
            </div>
          )}
        </div>
      </div>
    </SlideFrame>
  );
}

// ═══════════════════════════════════════════════════════════════
// Slide 3 — "Shape every detail."
// from /home/ani/slide3 — morphing waveform canvas + knobs/sliders
// GLOW: 139,92,246
// ═══════════════════════════════════════════════════════════════
const S3_GLOW = '139,92,246';
const S3_BAR_W = 4, S3_BAR_GAP = 3;
const S3_WAVE_W = 620, S3_WAVE_H = 110;
const S3_BAR_COUNT = Math.floor(S3_WAVE_W / (S3_BAR_W + S3_BAR_GAP));

function buildAmps(seed) {
  const rng = seededRandom(seed);
  const raw = [];
  for (let i = 0; i < S3_BAR_COUNT; i++) {
    const pos = i / S3_BAR_COUNT;
    const env = 0.3 + 0.7 * Math.sin(pos * Math.PI);
    raw.push(rng() * env * (rng() > 0.88 ? 1.3 : 1));
  }
  const smoothed = [];
  for (let i = 0; i < raw.length; i++) {
    let sum = 0, cnt = 0;
    for (let j = Math.max(0, i - 3); j <= Math.min(raw.length - 1, i + 3); j++) { sum += raw[j]; cnt++; }
    smoothed.push(sum / cnt);
  }
  return smoothed;
}
const phaseAmps = [buildAmps(777), buildAmps(877), buildAmps(977), buildAmps(1077)];

function KnobCanvas({ val, color, size = 44 }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr; canvas.height = size * dpr;
    canvas.style.width = size + 'px'; canvas.style.height = size + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    const cx = size / 2, cy = size / 2, r = size / 2 - 4;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0.75 * Math.PI, 2.25 * Math.PI);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 3; ctx.lineCap = 'round'; ctx.stroke();
    ctx.beginPath(); ctx.arc(cx, cy, r, 0.75 * Math.PI, 0.75 * Math.PI + val * 1.5 * Math.PI);
    ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.stroke();
  }, [val, color, size]);
  return <canvas ref={canvasRef} style={{ width: size, height: size }} />;
}

function Slide3({ active }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(0);
  const intervalRef = useRef(null);
  const currentAmps = useRef([...phaseAmps[0]]);
  const targetAmps = useRef([...phaseAmps[1]]);
  const phaseIdx = useRef(0);
  const [knobs, setKnobs] = useState([0.6, 0.4, 0.75, 0.3]);
  const [sliders, setSliders] = useState([0.55, 0.7, 0.45]);

  useEffect(() => {
    if (!active) {
      cancelAnimationFrame(rafRef.current);
      clearInterval(intervalRef.current);
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = S3_WAVE_W * dpr; canvas.height = S3_WAVE_H * dpr;
    canvas.style.width = S3_WAVE_W + 'px'; canvas.style.height = S3_WAVE_H + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    const mid = S3_WAVE_H / 2, maxAmp = S3_WAVE_H * 0.46;
    const barStep = S3_BAR_W + S3_BAR_GAP;

    const drawFrame = () => {
      ctx.clearRect(0, 0, S3_WAVE_W, S3_WAVE_H);
      ctx.fillStyle = `rgba(${S3_GLOW},0.85)`;
      for (let b = 0; b < S3_BAR_COUNT; b++) {
        const x = b * barStep;
        const h = currentAmps.current[b] * maxAmp;
        if (h < 1) continue;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x, mid - h, S3_BAR_W, h * 2, S3_BAR_W / 2);
        else ctx.rect(x, mid - h, S3_BAR_W, h * 2);
        ctx.fill();
      }
    };

    const morphStep = () => {
      let done = true;
      for (let b = 0; b < S3_BAR_COUNT; b++) {
        const diff = targetAmps.current[b] - currentAmps.current[b];
        if (Math.abs(diff) > 0.001) { currentAmps.current[b] += diff * 0.04; done = false; }
        else currentAmps.current[b] = targetAmps.current[b];
      }
      drawFrame();
      if (!done) rafRef.current = requestAnimationFrame(morphStep);
    };
    rafRef.current = requestAnimationFrame(morphStep);

    intervalRef.current = setInterval(() => {
      phaseIdx.current = (phaseIdx.current + 1) % phaseAmps.length;
      for (let b = 0; b < S3_BAR_COUNT; b++) targetAmps.current[b] = phaseAmps[phaseIdx.current][b];
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(morphStep);
      setKnobs([0.3 + Math.random() * 0.6, 0.3 + Math.random() * 0.6, 0.3 + Math.random() * 0.6, 0.3 + Math.random() * 0.6]);
      setSliders([0.25 + Math.random() * 0.65, 0.25 + Math.random() * 0.65, 0.25 + Math.random() * 0.65]);
    }, 2200);

    return () => {
      cancelAnimationFrame(rafRef.current);
      clearInterval(intervalRef.current);
    };
  }, [active]);

  const knobDefs = [
    { label: 'Density', color: `rgba(${S3_GLOW},0.8)` },
    { label: 'Energy',  color: 'rgba(72,202,228,0.8)' },
    { label: 'Timbre',  color: 'rgba(16,185,129,0.8)' },
    { label: 'Reverb',  color: 'rgba(245,158,11,0.8)' },
  ];
  const sliderColors = ['rgba(255,100,100,0.8)', `rgba(${S3_GLOW},0.8)`, 'rgba(72,202,228,0.8)'];
  const sliderLabels = ['Low', 'Mid', 'Hi'];

  return (
    <SlideFrame glowRGB={S3_GLOW} headline="Shape every detail." copy="Fine-tune timbre, energy, density and more with per-track parameter controls. Every adjustment morphs the audio in real time.">
      <div style={{ position: 'absolute', inset: 0, ...glassPanelStyle(S3_GLOW), display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16, padding: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, alignSelf: 'flex-start' }}>
          <span style={{ color: `rgba(${S3_GLOW},0.9)` }}>{Icons.sliders}</span>
          <span style={{ fontSize: 12, color: '#ccc', fontWeight: 600 }}>Mix Parameters</span>
        </div>
        <canvas ref={canvasRef} style={{ display: 'block', maxWidth: '100%' }} />
        <div style={{ display: 'flex', gap: 20, alignItems: 'center' }}>
          {knobDefs.map((k, i) => (
            <div key={k.label} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
              <KnobCanvas val={knobs[i]} color={k.color} />
              <span style={{ fontSize: 9, color: '#888' }}>{k.label}</span>
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 24, alignItems: 'flex-end' }}>
          {sliders.map((v, i) => (
            <div key={sliderLabels[i]} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
              <span style={{ fontSize: 9, color: '#888' }}>{sliderLabels[i]}</span>
              <div style={{ width: 6, height: 60, background: 'rgba(255,255,255,0.06)', borderRadius: 3, position: 'relative' }}>
                <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, borderRadius: 3, background: sliderColors[i], height: v * 60, transition: 'height 0.8s ease' }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </SlideFrame>
  );
}

// ═══════════════════════════════════════════════════════════════
// Slide 4 — "Personalized, adaptive music."
// from /home/ani/slide4 — video + scene markers + track reveal
// GLOW: 30,60,180  VIDEO: storage.googleapis.com/.../sitedemo.mp4
// ═══════════════════════════════════════════════════════════════
const S4_GLOW = '30,60,180';
const scoreTracks = [
  { ...instrumentPool[1], name: 'Strings', seed: 510, row: 0, startFrac: 0, widthFrac: 1, isPlaceholder: false },
  { ...instrumentPool[2], name: 'Piano',   seed: 620, row: 1, startFrac: 0, widthFrac: 1, isPlaceholder: false },
  { ...instrumentPool[4], name: 'Drums',   seed: 730, row: 2, startFrac: 0, widthFrac: 1, isPlaceholder: false },
  { ...instrumentPool[5], name: 'Bass',    seed: 840, row: 3, startFrac: 0, widthFrac: 1, isPlaceholder: false },
];
const sceneMarkers = [
  { id: 1, pos: 0.05, color: 'rgba(100,180,255,0.8)', label: 'Sc 1' },
  { id: 2, pos: 0.28, color: 'rgba(255,160,60,0.8)',  label: 'Sc 2' },
  { id: 3, pos: 0.52, color: 'rgba(120,220,120,0.8)', label: 'Sc 3' },
  { id: 4, pos: 0.76, color: 'rgba(255,100,160,0.8)', label: 'Sc 4' },
];

function Slide4({ active }) {
  const [phase, setPhase] = useState('video');
  const [revealedTracks, setRevealedTracks] = useState([]);
  const videoRef = useRef(null);
  const timersRef = useRef([]);

  const clearAll = useCallback(() => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
  }, []);

  useEffect(() => {
    if (!active) { clearAll(); if (videoRef.current) videoRef.current.pause(); return; }

    const t = (ms, fn) => { const id = setTimeout(fn, ms); timersRef.current.push(id); };
    const run = () => {
      clearAll();
      setPhase('video');
      setRevealedTracks([]);
      if (videoRef.current) { videoRef.current.currentTime = 0; videoRef.current.play().catch(() => {}); }
      t(1500, () => setPhase('analyzing'));
      t(3200, () => setPhase('generating'));
      t(3800, () => setPhase('scoring'));
      scoreTracks.forEach((_, i) => t(4200 + i * 400, () => setRevealedTracks(p => [...p, i])));
      t(4200 + scoreTracks.length * 400 + 400, () => setPhase('done'));
      timersRef.current.push(setTimeout(run, 9000));
    };
    run();
    return clearAll;
  }, [active, clearAll]);

  const showMarkers = phase === 'analyzing' || phase === 'generating';
  const showTracks = phase === 'scoring' || phase === 'done';
  const S4_TH = 42, S4_LW = 90;

  return (
    <SlideFrame glowRGB={S4_GLOW} headline="Personalized, adaptive music." copy="Adapt existing music to picture, or generate new scores tailored for your visual media. With track-level key framing, polish to perfection.">
      <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
        {/* Video panel */}
        <div style={{ flex: 1, ...glassPanelStyle(S4_GLOW), overflow: 'hidden', position: 'relative', borderRadius: 14 }}>
          <div style={{ display: 'flex', borderBottom: '1px solid rgba(255,255,255,0.06)', background: 'rgba(6,6,12,0.25)', position: 'relative', zIndex: 2 }}>
            {['Image', 'FX', 'Video', 'MIDI', 'Audio'].map((tab, i) => (
              <div key={tab} style={{ padding: '6px 12px', fontSize: 9, fontWeight: 500, color: i === 2 ? `rgba(${S4_GLOW},1)` : '#888', borderBottom: i === 2 ? `2px solid rgba(${S4_GLOW},1)` : '2px solid transparent', background: i === 2 ? `rgba(${S4_GLOW},0.06)` : 'transparent' }}>{tab}</div>
            ))}
          </div>
          <video ref={videoRef} src="https://storage.googleapis.com/audiocraft-411005.appspot.com/assets/sitedemo.mp4" playsInline muted loop preload="auto" style={{ position: 'absolute', top: 28, left: 0, right: 0, bottom: 0, width: '100%', height: 'calc(100% - 28px)', objectFit: 'cover', opacity: 0.7 }} />
          <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: 60, background: 'linear-gradient(to top,rgba(5,5,8,0.9),transparent)', zIndex: 1 }} />
          {/* Scene markers */}
          <div style={{ position: 'absolute', bottom: 8, left: 12, right: 12, display: 'flex', justifyContent: 'space-around', zIndex: 4, opacity: showMarkers ? 1 : 0, transition: 'opacity 0.4s ease' }}>
            {sceneMarkers.map((m, mi) => (
              <div key={m.id} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3, opacity: showMarkers ? 1 : 0, transform: showMarkers ? 'translateY(0)' : 'translateY(16px)', transition: `opacity 0.35s ease ${mi * 0.12}s, transform 0.35s ease ${mi * 0.12}s` }}>
                <div style={{ width: 48, height: 32, borderRadius: 5, background: `linear-gradient(135deg,${m.color.replace('0.8', '0.25')},${m.color.replace('0.8', '0.08')})`, border: `1px solid ${m.color.replace('0.8', '0.5')}`, boxShadow: `0 2px 10px rgba(0,0,0,0.5)`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <svg width={14} height={14} viewBox="0 0 24 24" fill="none"><polygon points="5 3 19 12 5 21" fill={m.color} /></svg>
                </div>
                <span style={{ fontSize: 8, color: m.color, fontWeight: 600 }}>{m.label}</span>
              </div>
            ))}
          </div>
        </div>
        {/* DAW panel */}
        <div style={{ flex: 1, ...glassPanelStyle(S4_GLOW), overflow: 'hidden', display: 'flex', flexDirection: 'column', borderRadius: 14 }}>
          <div style={{ height: 32, display: 'flex', alignItems: 'center', padding: '0 10px', gap: 6, background: 'rgba(8,8,14,0.3)', borderBottom: '1px solid rgba(255,255,255,0.08)', borderRadius: '14px 14px 0 0', flexShrink: 0 }}>
            <div style={{ width: 24, height: 24, borderRadius: 6, border: `1px solid rgba(${S4_GLOW},0.5)`, background: `rgba(${S4_GLOW},0.2)`, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{Icons.pause}</div>
            <div style={{ width: 24, height: 24, borderRadius: 6, border: '1px solid rgba(255,255,255,0.1)', background: 'rgba(255,255,255,0.06)', color: '#ccc', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{Icons.stop}</div>
            <div style={{ flex: 1 }} />
            <div style={{ padding: '3px 8px', borderRadius: 6, fontSize: 10, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: '#888' }}>120 BPM</div>
          </div>
          <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
            {scoreTracks.map((t, i) => {
              const isRev = revealedTracks.includes(i);
              return (
                <div key={t.name} style={{ position: 'absolute', top: i * S4_TH, left: 0, right: 0, height: S4_TH, opacity: showTracks ? 1 : 0, transition: `opacity 0.4s ease ${i * 0.1}s`, borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                  <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: S4_LW, display: 'flex', alignItems: 'center', padding: '0 6px', gap: 6, background: 'rgba(8,8,14,0.2)', backdropFilter: 'blur(20px)', borderRight: '1px solid rgba(255,255,255,0.06)' }}>
                    <div style={{ color: t.color, opacity: 0.7 }}>{t.icon}</div>
                    <span style={{ fontSize: 10, color: '#ccc', fontWeight: 500 }}>{t.name}</span>
                  </div>
                  <div style={{ position: 'absolute', top: 3, left: S4_LW + 6, right: 6, height: S4_TH - 6, borderRadius: 8, overflow: 'hidden' }}>
                    {!isRev && showTracks && (
                      <div style={{ position: 'absolute', inset: 0, borderRadius: 8, background: `rgba(${S4_GLOW},0.04)`, border: `1px solid rgba(${S4_GLOW},0.12)` }}>
                        <NoiseWaveform width={600} height={S4_TH - 10} color={t.color} settling={isRev} />
                      </div>
                    )}
                    {isRev && (
                      <div style={{ position: 'absolute', inset: 0, borderRadius: 8, background: `linear-gradient(180deg,${t.colorBg},${t.colorBg.replace('0.06', '0.02')})`, backdropFilter: 'blur(14px)', borderLeft: `2px solid ${t.color}`, borderRight: `1px solid ${t.color.replace('0.7', '0.12')}`, borderTop: `1px solid ${t.color.replace('0.7', '0.15')}`, boxShadow: '0 4px 16px rgba(0,0,0,0.3),inset 0 1px 0 rgba(255,255,255,0.08)' }}>
                        <div style={{ position: 'absolute', left: 6, top: '50%', transform: 'translateY(-50%)' }}>
                          <WaveformSVG color={t.color} seed={t.seed} width={590} height={S4_TH - 16} />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </SlideFrame>
  );
}

// ═══════════════════════════════════════════════════════════════
// Slide 5 — "Everything stays editable."
// from /home/ani/slide5 — tracks play, bass track waveform morphs
// GLOW: 100,160,255  editTracks: Vocals, Drums, Bass, Synth
// ═══════════════════════════════════════════════════════════════
const S5_GLOW = '100,160,255';
const editTracks = [
  { ...instrumentPool[0], name: 'Vocals', seed: 111, row: 0, startFrac: 0, widthFrac: 1, isPlaceholder: false },
  { ...instrumentPool[4], name: 'Drums',  seed: 222, row: 1, startFrac: 0, widthFrac: 1, isPlaceholder: false },
  { ...instrumentPool[5], name: 'Bass',   seed: 333, row: 2, startFrac: 0, widthFrac: 1, isPlaceholder: false },
  { ...instrumentPool[6], name: 'Synth',  seed: 444, row: 3, startFrac: 0, widthFrac: 1, isPlaceholder: false },
];
const EDIT_IDX = 2;

function buildEditAmps(seed) {
  const rng = seededRandom(seed);
  const a = [];
  for (let x = 0; x < 660; x += 3) {
    const env = 0.4 + 0.6 * Math.sin((x / 480) * Math.PI);
    a.push(rng() > 0.87 ? 1.3 : 1, rng() * env);
  }
  return a;
}
const editAmpsA = buildEditAmps(333);
const editAmpsB = buildEditAmps(888);

function Slide5({ active }) {
  const [phase, setPhase] = useState('playing');
  const [editSeed, setEditSeed] = useState(333);
  const playheadRef = useRef(null);
  const playheadCapRef = useRef(null);
  const timeRef = useRef(null);
  const editCanvasRef = useRef(null);
  const rafRef = useRef(0);
  const timersRef = useRef([]);
  const playStart = useRef(Date.now());
  const curAmps = useRef([...editAmpsA]);
  const targetAmps = useRef(null);
  const morphing = useRef(false);
  const lastNow = useRef(0);
  const LW = 100, TH = 52;

  const clearAll = useCallback(() => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
    cancelAnimationFrame(rafRef.current);
  }, []);

  const drawEditWave = useCallback(() => {
    const canvas = editCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = 640, H = TH - 12;
    ctx.clearRect(0, 0, W, H);
    const mid = H / 2;
    ctx.strokeStyle = editTracks[EDIT_IDX].color;
    ctx.lineWidth = 2; ctx.lineCap = 'round';
    let d = '';
    const amps = curAmps.current;
    for (let xi = 0, idx = 0; xi < W; xi += 3, idx += 2) {
      const burst = amps[idx] || 1;
      const amp = (amps[idx + 1] || 0) * (H * 0.42) * burst;
      d += `M${xi},${mid - amp}V${mid + amp}`;
    }
    const p = new Path2D(d);
    ctx.stroke(p);
  }, []);

  useEffect(() => {
    if (!active) { clearAll(); return; }

    const canvas = editCanvasRef.current;
    if (canvas) {
      const dpr = window.devicePixelRatio || 1;
      const W = 640, H = TH - 12;
      canvas.width = W * dpr; canvas.height = H * dpr;
      canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
      canvas.getContext('2d').scale(dpr, dpr);
    }

    const rafTick = (now) => {
      const dt = now - (lastNow.current || now);
      lastNow.current = now;
      const elapsed = (now - playStart.current) / 1000;
      const ph = (elapsed % 30) / 30;
      const phPx = LW + ph * (600 - LW);
      if (playheadRef.current) playheadRef.current.style.left = phPx + 'px';
      if (playheadCapRef.current) playheadCapRef.current.style.left = (phPx - 4.5) + 'px';
      if (timeRef.current) timeRef.current.textContent = `0:${String(Math.floor(ph * 30)).padStart(2, '0')}`;
      if (morphing.current && targetAmps.current) {
        let done = true;
        for (let i = 0; i < curAmps.current.length; i++) {
          const diff = targetAmps.current[i] - curAmps.current[i];
          if (Math.abs(diff) > 0.001) { curAmps.current[i] += diff * 0.05; done = false; }
          else curAmps.current[i] = targetAmps.current[i];
        }
        drawEditWave();
        if (done) morphing.current = false;
      }
      rafRef.current = requestAnimationFrame(rafTick);
    };

    const t = (ms, fn) => { const id = setTimeout(fn, ms); timersRef.current.push(id); };
    const run = () => {
      clearAll();
      playStart.current = Date.now();
      lastNow.current = 0;
      setPhase('playing');
      setEditSeed(333);
      morphing.current = false;
      targetAmps.current = null;
      for (let i = 0; i < editAmpsA.length; i++) curAmps.current[i] = editAmpsA[i];
      drawEditWave();
      rafRef.current = requestAnimationFrame(rafTick);

      t(2000, () => setPhase('selecting'));
      t(3500, () => {
        setPhase('editing');
        setEditSeed(888);
        targetAmps.current = [...editAmpsB];
        morphing.current = true;
      });
      t(5000, () => setPhase('reshaped'));
      t(7000, () => setPhase('playing'));
      timersRef.current.push(setTimeout(run, 9000));
    };
    run();
    return clearAll;
  }, [active, clearAll, drawEditWave]);

  return (
    <SlideFrame glowRGB={S5_GLOW} headline="Everything stays editable." copy="All generated tracks are fully non-destructive. Regenerate any section, adjust parameters mid-playback, and the audio reshapes in real time.">
      <div style={{ position: 'absolute', inset: 0, ...glassPanelStyle(S5_GLOW), overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <TransportBar glowRGB={S5_GLOW} timeRef={timeRef} />
        <Timeline labelW={LW} />
        <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          {editTracks.map((t, i) => {
            const clipW = 660;
            return (
              <div key={t.name} style={{ position: 'absolute', top: i * TH, left: 0, right: 0, height: TH, borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: LW, display: 'flex', alignItems: 'center', padding: '0 6px', gap: 6, background: 'rgba(8,8,14,0.2)', backdropFilter: 'blur(20px)', borderRight: '1px solid rgba(255,255,255,0.06)' }}>
                  <div style={{ color: t.color, opacity: 0.7 }}>{t.icon}</div>
                  <span style={{ fontSize: 10, color: '#ccc', fontWeight: 500 }}>{t.name}</span>
                </div>
                {i !== EDIT_IDX ? (
                  <div style={{ position: 'absolute', top: 3, left: LW + 4, width: clipW, height: TH - 6, borderRadius: 8, background: `linear-gradient(180deg,${t.colorBg},${t.colorBg.replace('0.06', '0.02')})`, backdropFilter: 'blur(14px)', borderLeft: `2px solid ${t.color}`, borderRight: `1px solid ${t.color.replace('0.7', '0.12')}`, borderTop: `1px solid ${t.color.replace('0.7', '0.15')}`, borderBottom: `1px solid ${t.color.replace('0.7', '0.05')}`, boxShadow: '0 4px 16px rgba(0,0,0,0.3),inset 0 1px 0 rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                    <div style={{ position: 'absolute', left: 6, top: '50%', transform: 'translateY(-50%)' }}>
                      <WaveformSVG color={t.color} seed={t.seed} width={clipW - 12} height={TH - 18} />
                    </div>
                  </div>
                ) : (
                  <>
                    {/* Edit highlight */}
                    <div style={{ position: 'absolute', top: 0, left: LW, right: 0, bottom: 0, background: `rgba(${S5_GLOW},0.08)`, border: `1px solid rgba(${S5_GLOW},0.25)`, borderRadius: '0 6px 6px 0', opacity: (phase === 'selecting' || phase === 'editing' || phase === 'reshaped') ? 1 : 0, transition: 'opacity 0.4s ease', zIndex: 3, pointerEvents: 'none' }} />
                    {/* Morphing canvas clip */}
                    <div style={{ position: 'absolute', top: 3, left: LW + 4, width: clipW, height: TH - 6, borderRadius: 8, background: `linear-gradient(180deg,${t.colorBg},${t.colorBg.replace('0.06', '0.02')})`, backdropFilter: 'blur(14px)', borderLeft: `2px solid ${t.color}`, borderRight: `1px solid ${t.color.replace('0.7', '0.12')}`, borderTop: `1px solid ${t.color.replace('0.7', '0.15')}`, borderBottom: `1px solid ${t.color.replace('0.7', '0.05')}`, boxShadow: '0 4px 16px rgba(0,0,0,0.3),inset 0 1px 0 rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                      <div style={{ position: 'absolute', left: 6, top: '50%', transform: 'translateY(-50%)' }}>
                        <canvas ref={editCanvasRef} style={{ display: 'block' }} />
                      </div>
                    </div>
                  </>
                )}
              </div>
            );
          })}

          {/* Playhead */}
          <div ref={playheadRef} style={{ position: 'absolute', top: 0, bottom: 0, width: 1, background: 'rgba(255,255,255,0.7)', pointerEvents: 'none', zIndex: 5, left: LW }} />
          <div ref={playheadCapRef} style={{ position: 'absolute', top: -4, width: 9, height: 9, background: 'rgba(255,255,255,0.9)', clipPath: 'polygon(50% 100%, 0 0, 100% 0)', pointerEvents: 'none', zIndex: 6 }} />
        </div>
      </div>
    </SlideFrame>
  );
}

// ═══════════════════════════════════════════════════════════════
// FeatureSlideshow — wrapper with navigation
// ═══════════════════════════════════════════════════════════════
const SLIDE_COMPONENTS = [Slide1, Slide2, Slide3, Slide4, Slide5];
const SLIDE_DURATION = 9500;

export default function FeatureSlideshow() {
  const [activeIdx, setActiveIdx] = useState(0);
  const intervalRef = useRef(null);

  const resetInterval = useCallback((idx) => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => {
      setActiveIdx(i => (i + 1) % SLIDE_COMPONENTS.length);
    }, SLIDE_DURATION);
  }, []);

  useEffect(() => {
    resetInterval(0);
    return () => clearInterval(intervalRef.current);
  }, [resetInterval]);

  const goTo = useCallback((idx) => {
    setActiveIdx(idx);
    resetInterval(idx);
  }, [resetInterval]);

  return (
    <div className={styles.container}>
      {/* Slides */}
      <div className={styles.slidesWrapper}>
        {SLIDE_COMPONENTS.map((SlideComp, i) => (
          <div key={i} className={`${styles.slide} ${i === activeIdx ? styles.slideActive : ''}`}>
            <SlideComp active={i === activeIdx} />
          </div>
        ))}
      </div>

      {/* Prev / Next */}
      <button className={`${styles.navBtn} ${styles.prevBtn}`} onClick={() => goTo((activeIdx - 1 + SLIDE_COMPONENTS.length) % SLIDE_COMPONENTS.length)} aria-label="Previous">&#8249;</button>
      <button className={`${styles.navBtn} ${styles.nextBtn}`} onClick={() => goTo((activeIdx + 1) % SLIDE_COMPONENTS.length)} aria-label="Next">&#8250;</button>

      {/* Dots */}
      <div className={styles.dots}>
        {SLIDE_COMPONENTS.map((_, i) => (
          <button key={i} className={`${styles.dot} ${i === activeIdx ? styles.dotActive : ''}`} onClick={() => goTo(i)} aria-label={`Slide ${i + 1}`} />
        ))}
      </div>
    </div>
  );
}
