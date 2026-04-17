import React, { useState, useRef, useCallback, useEffect } from 'react';
import styles from './DO1.module.css';

const PRESETS = [
  { name: 'Exact Copy', mask: 1.0, cfg: 0.0, desc: 'Reproduce x_cond exactly' },
  { name: 'Natural Variation', mask: 0.95, cfg: 0.0, desc: 'Micro-variations' },
  { name: 'Timbre Transfer', mask: 1.0, cfg: 1.0, desc: 'x_cond notes + x_ref sound' },
  { name: 'Gentle Transfer', mask: 1.0, cfg: 0.3, desc: 'Subtle timbre shift' },
  { name: 'Style Transfer', mask: 0.85, cfg: 0.8, desc: 'Content + x_ref feel' },
  { name: 'Separation', mask: 0.0, cfg: 1.0, desc: 'Extract from mix' },
  { name: 'MIDI Render', mask: 0.0, cfg: 1.0, desc: 'MIDI to realistic audio' },
  { name: 'Expressive MIDI', mask: 0.85, cfg: 1.0, desc: 'MIDI with natural feel' },
  { name: 'Inpaint', mask: null, cfg: 1.0, desc: 'Fill selected gap' },
  { name: 'Continue', mask: null, cfg: 1.0, desc: 'Extend from end' },
  { name: 'Free Generate', mask: 0.0, cfg: 1.0, desc: 'Generate from reference' },
  { name: 'Accompaniment', mask: 0.0, cfg: 1.0, desc: 'Generate compatible part' },
];

const LIBRARY_CATEGORIES = ['All', 'Instruments', 'Drums', 'Vocals', 'FX', 'User'];

const LIBRARY_ITEMS = [
  { name: 'Acoustic Guitar', category: 'Instruments', duration: '2.8s' },
  { name: 'Violin Sustain', category: 'Instruments', duration: '3.1s' },
  { name: 'Piano Chord', category: 'Instruments', duration: '2.4s' },
  { name: 'Electric Bass', category: 'Instruments', duration: '1.9s' },
  { name: 'Trumpet Legato', category: 'Instruments', duration: '2.6s' },
  { name: 'Rock Kit', category: 'Drums', duration: '2.0s' },
  { name: 'Jazz Brushes', category: 'Drums', duration: '3.2s' },
  { name: 'Electronic Kit', category: 'Drums', duration: '1.5s' },
  { name: 'Female Vocal', category: 'Vocals', duration: '3.0s' },
  { name: 'Male Vocal', category: 'Vocals', duration: '2.7s' },
  { name: 'Choir Pad', category: 'Vocals', duration: '4.1s' },
  { name: 'Reverb Tail', category: 'FX', duration: '2.2s' },
  { name: 'Tape Saturation', category: 'FX', duration: '1.8s' },
];

const LANE_HEIGHT = 64;

// Draw a single curve (line + fill + points) on a canvas context
function drawCurve(ctx, points, w, h, maxVal, color, totalDuration, selectedIdx) {
  const sorted = [...points].sort((a, b) => a.time - b.time);
  if (sorted.length === 0) return;

  const tx = (t) => (t / totalDuration) * w;
  const vy = (v) => h - (v / maxVal) * h;

  // Line
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  sorted.forEach((pt, i) => {
    const x = tx(pt.time);
    const y = vy(pt.value);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Fill under
  const last = sorted[sorted.length - 1];
  ctx.lineTo(tx(last.time), h);
  ctx.lineTo(tx(sorted[0].time), h);
  ctx.closePath();
  ctx.globalAlpha = 0.05;
  ctx.fillStyle = color;
  ctx.fill();
  ctx.globalAlpha = 1.0;

  // Points (skip edge)
  sorted.forEach((pt) => {
    if (pt.isEdge) return;
    const origIdx = points.indexOf(pt);
    const x = tx(pt.time);
    const y = vy(pt.value);
    ctx.fillStyle = origIdx === selectedIdx ? '#ffffff' : color;
    ctx.strokeStyle = 'rgba(0,0,0,0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });
}

// Smooth and downsample a freehand path aggressively
function smoothPath(rawPath, targetPoints) {
  if (rawPath.length < 3) return rawPath;
  const target = targetPoints || 12;
  // Downsample to target number of points
  const step = Math.max(1, Math.floor(rawPath.length / target));
  const sampled = [];
  for (let i = 0; i < rawPath.length; i += step) sampled.push(rawPath[i]);
  if (sampled[sampled.length - 1] !== rawPath[rawPath.length - 1]) sampled.push(rawPath[rawPath.length - 1]);
  // Three passes of moving-average smoothing
  let result = sampled;
  for (let pass = 0; pass < 3; pass++) {
    result = result.map((pt, i) => {
      if (i === 0 || i === result.length - 1) return pt;
      const prev = result[i - 1];
      const next = result[i + 1];
      return { time: (prev.time + pt.time + next.time) / 3, value: (prev.value + pt.value + next.value) / 3 };
    });
  }
  return result;
}

// ─── Single-curve Automation Lane ───────────────────────────────
const AutomationLane = ({ points, setPoints, maxVal, color, label, totalDuration, tool }) => {
  const canvasRef = useRef(null);
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const isDrawingRef = useRef(false);
  const drawPathRef = useRef([]);

  const resizeAndDraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = '100%';
    canvas.style.height = LANE_HEIGHT + 'px';
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0) return;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    const divs = maxVal <= 1 ? 4 : 6;
    for (let i = 1; i < divs; i++) {
      const y = (rect.height / divs) * i;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(rect.width, y); ctx.stroke();
    }

    drawCurve(ctx, points, rect.width, rect.height, maxVal, color, totalDuration, selectedIdx);
  }, [points, maxVal, color, totalDuration, selectedIdx]);

  useEffect(() => {
    requestAnimationFrame(() => requestAnimationFrame(resizeAndDraw));
  }, [resizeAndDraw]);

  useEffect(() => {
    window.addEventListener('resize', resizeAndDraw);
    return () => window.removeEventListener('resize', resizeAndDraw);
  }, [resizeAndDraw]);

  const getCoords = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const getSize = () => {
    const c = canvasRef.current;
    return { w: c.offsetWidth, h: c.offsetHeight };
  };

  const findPoint = (mx, my) => {
    const { w, h } = getSize();
    for (let i = 0; i < points.length; i++) {
      if (points[i].isEdge) continue;
      const px = (points[i].time / totalDuration) * w;
      const py = h - (points[i].value / maxVal) * h;
      if (Math.sqrt((mx - px) ** 2 + (my - py) ** 2) <= 15) return i;
    }
    return -1;
  };

  const handleMouseDown = (e) => {
    if (e.button === 2) return;
    const { x, y } = getCoords(e);
    const { w, h } = getSize();
    if (w === 0) return;

    if (tool === 'draw') {
      isDrawingRef.current = true;
      drawPathRef.current = [{ time: (x / w) * totalDuration, value: Math.max(0, Math.min(maxVal, (1 - y / h) * maxVal)) }];
      return;
    }

    const idx = findPoint(x, y);
    if (idx !== -1) {
      setSelectedIdx(idx);
      setIsDragging(true);
      canvasRef.current.style.cursor = 'grabbing';
    } else {
      const time = (x / w) * totalDuration;
      const value = Math.max(0, Math.min(maxVal, (1 - y / h) * maxVal));
      setPoints([...points, { time, value, isEdge: false }].sort((a, b) => a.time - b.time));
    }
  };

  const handleMouseMove = (e) => {
    const { x, y } = getCoords(e);
    const { w, h } = getSize();
    if (w === 0) return;

    if (tool === 'draw' && isDrawingRef.current) {
      drawPathRef.current.push({ time: (x / w) * totalDuration, value: Math.max(0, Math.min(maxVal, (1 - y / h) * maxVal)) });
      // Live preview with aggressive downsampling
      const path = drawPathRef.current;
      const minT = Math.min(path[0].time, path[path.length - 1].time);
      const maxT = Math.max(path[0].time, path[path.length - 1].time);
      const kept = points.filter(p => p.isEdge || p.time < minT || p.time > maxT);
      // Show smoothed preview
      const smoothed = smoothPath(path, 20).map(p => ({ ...p, isEdge: false }));
      setPoints([...kept, ...smoothed].sort((a, b) => a.time - b.time));
      return;
    }

    if (isDragging && selectedIdx !== null) {
      const time = Math.max(0, Math.min(totalDuration, (x / w) * totalDuration));
      const value = Math.max(0, Math.min(maxVal, (1 - y / h) * maxVal));
      const updated = points.map((pt, i) => i === selectedIdx ? { ...pt, time, value } : pt);
      const sorted = [...updated].sort((a, b) => a.time - b.time);
      const draggedPt = updated[selectedIdx];
      const newIdx = sorted.findIndex(p => p.time === draggedPt.time && p.value === draggedPt.value);
      setSelectedIdx(newIdx);
      setPoints(sorted);
    } else if (!isDragging) {
      const idx = findPoint(x, y);
      canvasRef.current.style.cursor = idx !== -1 ? 'grab' : 'crosshair';
    }
  };

  const handleMouseUp = () => {
    if (tool === 'draw' && isDrawingRef.current) {
      isDrawingRef.current = false;
      const path = drawPathRef.current;
      if (path.length > 2) {
        const minT = Math.min(path[0].time, path[path.length - 1].time);
        const maxT = Math.max(path[0].time, path[path.length - 1].time);
        const kept = points.filter(p => p.isEdge || p.time < minT || p.time > maxT);
        const smoothed = smoothPath(path, 10).map(p => ({ ...p, isEdge: false }));
        setPoints([...kept, ...smoothed].sort((a, b) => a.time - b.time));
      }
      drawPathRef.current = [];
      return;
    }
    setIsDragging(false);
    setSelectedIdx(null);
    if (canvasRef.current) canvasRef.current.style.cursor = 'crosshair';
  };

  const handleRightClick = (e) => {
    e.preventDefault();
    const { x, y } = getCoords(e);
    const idx = findPoint(x, y);
    if (idx !== -1 && !points[idx].isEdge) {
      setPoints(points.filter((_, i) => i !== idx));
    }
  };

  useEffect(() => {
    const up = () => { setIsDragging(false); setSelectedIdx(null); isDrawingRef.current = false; };
    window.addEventListener('mouseup', up);
    return () => window.removeEventListener('mouseup', up);
  }, []);

  return (
    <div style={{ flex: 1, minWidth: 0, height: LANE_HEIGHT, position: 'relative', background: '#12121A' }}>
      <canvas
        ref={canvasRef}
        style={{ cursor: 'crosshair' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onContextMenu={handleRightClick}
      />
      {label && <span className={styles.automationValue}>{label}</span>}
    </div>
  );
};

// ─── Unified dual-curve Automation Lane (mask + cfg on one canvas) ──
const UnifiedAutomationLane = ({
  maskPoints, setMaskPoints, cfgPoints, setCfgPoints, totalDuration, tool,
}) => {
  const canvasRef = useRef(null);
  // Which curve is active: 'mask' or 'cfg'
  const [activeCurve, setActiveCurve] = useState('mask');
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const isDrawingRef = useRef(false);
  const drawPathRef = useRef([]);

  const activePoints = activeCurve === 'mask' ? maskPoints : cfgPoints;
  const setActivePoints = activeCurve === 'mask' ? setMaskPoints : setCfgPoints;
  const activeMaxVal = activeCurve === 'mask' ? 1.0 : 3.0;
  const activeColor = activeCurve === 'mask' ? '#D4A843' : '#2E75B6';
  const inactivePoints = activeCurve === 'mask' ? cfgPoints : maskPoints;
  const inactiveMaxVal = activeCurve === 'mask' ? 3.0 : 1.0;
  const inactiveColor = activeCurve === 'mask' ? '#2E75B6' : '#D4A843';

  const resizeAndDraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = '100%';
    canvas.style.height = LANE_HEIGHT + 'px';
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0) return;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const y = (rect.height / 4) * i;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(rect.width, y); ctx.stroke();
    }

    // Draw inactive curve first (dimmer)
    ctx.globalAlpha = 0.4;
    drawCurve(ctx, inactivePoints, rect.width, rect.height, inactiveMaxVal, inactiveColor, totalDuration, -1);
    ctx.globalAlpha = 1.0;

    // Draw active curve on top
    drawCurve(ctx, activePoints, rect.width, rect.height, activeMaxVal, activeColor, totalDuration, selectedIdx);
  }, [activePoints, inactivePoints, activeMaxVal, inactiveMaxVal, activeColor, inactiveColor, totalDuration, selectedIdx]);

  useEffect(() => {
    requestAnimationFrame(() => requestAnimationFrame(resizeAndDraw));
  }, [resizeAndDraw]);

  useEffect(() => {
    window.addEventListener('resize', resizeAndDraw);
    return () => window.removeEventListener('resize', resizeAndDraw);
  }, [resizeAndDraw]);

  const getCoords = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const getSize = () => {
    const c = canvasRef.current;
    return { w: c.offsetWidth, h: c.offsetHeight };
  };

  // Find point in a specific curve
  const findPointIn = (pts, maxVal, mx, my) => {
    const { w, h } = getSize();
    for (let i = 0; i < pts.length; i++) {
      if (pts[i].isEdge) continue;
      const px = (pts[i].time / totalDuration) * w;
      const py = h - (pts[i].value / maxVal) * h;
      if (Math.sqrt((mx - px) ** 2 + (my - py) ** 2) <= 15) return i;
    }
    return -1;
  };

  const handleMouseDown = (e) => {
    if (e.button === 2) return;
    const { x, y } = getCoords(e);
    const { w, h } = getSize();
    if (w === 0) return;

    if (tool === 'draw') {
      isDrawingRef.current = true;
      drawPathRef.current = [{ time: (x / w) * totalDuration, value: Math.max(0, Math.min(activeMaxVal, (1 - y / h) * activeMaxVal)) }];
      return;
    }

    // Check active curve first, then inactive
    let idx = findPointIn(activePoints, activeMaxVal, x, y);
    if (idx !== -1) {
      setSelectedIdx(idx);
      setIsDragging(true);
      canvasRef.current.style.cursor = 'grabbing';
      return;
    }

    // Check if clicking near the inactive curve's points - switch active curve
    const inIdx = findPointIn(inactivePoints, inactiveMaxVal, x, y);
    if (inIdx !== -1) {
      setActiveCurve(activeCurve === 'mask' ? 'cfg' : 'mask');
      setSelectedIdx(inIdx);
      setIsDragging(true);
      canvasRef.current.style.cursor = 'grabbing';
      return;
    }

    // Add new point to active curve
    const time = (x / w) * totalDuration;
    const value = Math.max(0, Math.min(activeMaxVal, (1 - y / h) * activeMaxVal));
    setActivePoints([...activePoints, { time, value, isEdge: false }].sort((a, b) => a.time - b.time));
  };

  const handleMouseMove = (e) => {
    const { x, y } = getCoords(e);
    const { w, h } = getSize();
    if (w === 0) return;

    const pts = activeCurve === 'mask' ? maskPoints : cfgPoints;
    const setPts = activeCurve === 'mask' ? setMaskPoints : setCfgPoints;
    const mv = activeCurve === 'mask' ? 1.0 : 3.0;

    if (tool === 'draw' && isDrawingRef.current) {
      drawPathRef.current.push({ time: (x / w) * totalDuration, value: Math.max(0, Math.min(mv, (1 - y / h) * mv)) });
      const path = drawPathRef.current;
      const minT = Math.min(path[0].time, path[path.length - 1].time);
      const maxT = Math.max(path[0].time, path[path.length - 1].time);
      const kept = pts.filter(p => p.isEdge || p.time < minT || p.time > maxT);
      const smoothed = smoothPath(path, 20).map(p => ({ ...p, isEdge: false }));
      setPts([...kept, ...smoothed].sort((a, b) => a.time - b.time));
      return;
    }

    if (isDragging && selectedIdx !== null) {
      const time = Math.max(0, Math.min(totalDuration, (x / w) * totalDuration));
      const value = Math.max(0, Math.min(mv, (1 - y / h) * mv));
      const updated = pts.map((pt, i) => i === selectedIdx ? { ...pt, time, value } : pt);
      const sorted = [...updated].sort((a, b) => a.time - b.time);
      const draggedPt = updated[selectedIdx];
      const newIdx = sorted.findIndex(p => p.time === draggedPt.time && p.value === draggedPt.value);
      setSelectedIdx(newIdx);
      setPts(sorted);
    } else if (!isDragging) {
      const idx = findPointIn(pts, mv, x, y);
      canvasRef.current.style.cursor = idx !== -1 ? 'grab' : 'crosshair';
    }
  };

  const handleMouseUp = () => {
    const pts = activeCurve === 'mask' ? maskPoints : cfgPoints;
    const setPts = activeCurve === 'mask' ? setMaskPoints : setCfgPoints;
    const mv = activeCurve === 'mask' ? 1.0 : 3.0;

    if (tool === 'draw' && isDrawingRef.current) {
      isDrawingRef.current = false;
      const path = drawPathRef.current;
      if (path.length > 2) {
        const minT = Math.min(path[0].time, path[path.length - 1].time);
        const maxT = Math.max(path[0].time, path[path.length - 1].time);
        const kept = pts.filter(p => p.isEdge || p.time < minT || p.time > maxT);
        const smoothed = smoothPath(path, 10).map(p => ({ ...p, isEdge: false }));
        setPts([...kept, ...smoothed].sort((a, b) => a.time - b.time));
      }
      drawPathRef.current = [];
      return;
    }
    setIsDragging(false);
    setSelectedIdx(null);
    if (canvasRef.current) canvasRef.current.style.cursor = 'crosshair';
  };

  const handleRightClick = (e) => {
    e.preventDefault();
    const { x, y } = getCoords(e);
    const pts = activeCurve === 'mask' ? maskPoints : cfgPoints;
    const setPts = activeCurve === 'mask' ? setMaskPoints : setCfgPoints;
    const mv = activeCurve === 'mask' ? 1.0 : 3.0;
    const idx = findPointIn(pts, mv, x, y);
    if (idx !== -1 && !pts[idx].isEdge) {
      setPts(pts.filter((_, i) => i !== idx));
    }
  };

  useEffect(() => {
    const up = () => { setIsDragging(false); setSelectedIdx(null); isDrawingRef.current = false; };
    window.addEventListener('mouseup', up);
    return () => window.removeEventListener('mouseup', up);
  }, []);

  return (
    <div style={{ flex: 1, minWidth: 0, height: LANE_HEIGHT, position: 'relative', background: '#12121A' }}>
      <canvas
        ref={canvasRef}
        style={{ cursor: 'crosshair' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onContextMenu={handleRightClick}
      />
      {/* Toggle which curve is active */}
      <div style={{ position: 'absolute', top: 4, right: 8, display: 'flex', gap: 6, pointerEvents: 'auto' }}>
        <button
          onClick={() => setActiveCurve('mask')}
          style={{
            fontSize: 10, padding: '2px 6px', borderRadius: 3, border: 'none', cursor: 'pointer',
            background: activeCurve === 'mask' ? 'rgba(212,168,67,0.3)' : 'rgba(255,255,255,0.06)',
            color: activeCurve === 'mask' ? '#D4A843' : 'rgba(200,200,208,0.5)',
          }}
        >Mask</button>
        <button
          onClick={() => setActiveCurve('cfg')}
          style={{
            fontSize: 10, padding: '2px 6px', borderRadius: 3, border: 'none', cursor: 'pointer',
            background: activeCurve === 'cfg' ? 'rgba(46,117,182,0.3)' : 'rgba(255,255,255,0.06)',
            color: activeCurve === 'cfg' ? '#2E75B6' : 'rgba(200,200,208,0.5)',
          }}
        >CFG</button>
      </div>
    </div>
  );
};

// ─── Main DO1 Component ───────────────────────────────────────
const DO1 = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(false);
  const [bpm, setBpm] = useState(120);
  const [currentTime, setCurrentTime] = useState(0);
  const [xCondFile, setXCondFile] = useState(null);
  const [xCondType, setXCondType] = useState('audio');
  const [xRefFile, setXRefFile] = useState(null);
  const [outputReady, setOutputReady] = useState(false);
  const defaultDuration = 30;
  const [splitLanes, setSplitLanes] = useState(false);
  const [automationTool, setAutomationTool] = useState('point');
  const [maskPoints, setMaskPoints] = useState([
    { time: 0, value: 1.0, isEdge: true },
    { time: defaultDuration, value: 1.0, isEdge: true },
  ]);
  const [cfgPoints, setCfgPoints] = useState([
    { time: 0, value: 1.0, isEdge: true },
    { time: defaultDuration, value: 1.0, isEdge: true },
  ]);
  const [seed, setSeed] = useState(42);
  const [seedLocked, setSeedLocked] = useState(false);
  const [steps, setSteps] = useState(50);
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [selectedPreset, setSelectedPreset] = useState(null);
  const [activeTab, setActiveTab] = useState('presets');
  const [libraryCategory, setLibraryCategory] = useState('All');
  const [librarySearch, setLibrarySearch] = useState('');
  const xCondCanvasRef = useRef(null);
  const outputCanvasRef = useRef(null);

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${m}:${String(s).padStart(2, '0')}.${String(ms).padStart(2, '0')}`;
  };

  const drawWaveform = useCallback((canvas, color, hasData) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!hasData) return;
    ctx.fillStyle = color;
    const centerY = h / 2;
    for (let x = 0; x < w; x += 3) {
      const amp = (Math.sin(x * 0.02) * 0.3 + Math.sin(x * 0.05) * 0.2 + Math.random() * 0.3) * centerY * 0.8;
      ctx.fillRect(x, centerY - amp, 2, amp * 2);
    }
  }, []);

  useEffect(() => { drawWaveform(xCondCanvasRef.current, 'rgba(74, 155, 142, 0.6)', !!xCondFile); }, [xCondFile, drawWaveform]);
  useEffect(() => { drawWaveform(outputCanvasRef.current, 'rgba(212, 168, 67, 0.6)', outputReady); }, [outputReady, drawWaveform]);
  useEffect(() => {
    const handleResize = () => {
      drawWaveform(xCondCanvasRef.current, 'rgba(74, 155, 142, 0.6)', !!xCondFile);
      drawWaveform(outputCanvasRef.current, 'rgba(212, 168, 67, 0.6)', outputReady);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [xCondFile, outputReady, drawWaveform]);

  const handleDrop = useCallback((e, target) => {
    e.preventDefault();
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (target === 'xcond') {
        setXCondFile(file);
        setXCondType(file.name.endsWith('.mid') || file.name.endsWith('.midi') ? 'midi' : 'audio');
      } else {
        setXRefFile(file);
      }
    }
  }, []);

  const handleFileClick = useCallback((target) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = target === 'xcond' ? 'audio/*,.mid,.midi' : 'audio/*';
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        if (target === 'xcond') {
          setXCondFile(file);
          setXCondType(file.name.endsWith('.mid') || file.name.endsWith('.midi') ? 'midi' : 'audio');
        } else {
          setXRefFile(file);
        }
      }
    };
    input.click();
  }, []);

  const applyPreset = useCallback((preset) => {
    setSelectedPreset(preset.name);
    if (preset.mask !== null) {
      setMaskPoints([
        { time: 0, value: preset.mask, isEdge: true },
        { time: defaultDuration, value: preset.mask, isEdge: true },
      ]);
    }
    setCfgPoints([
      { time: 0, value: preset.cfg, isEdge: true },
      { time: defaultDuration, value: preset.cfg, isEdge: true },
    ]);
  }, [defaultDuration]);

  const handleGenerate = useCallback(async () => {
    if (generating) return;
    setGenerating(true);
    setProgress(0);
    setOutputReady(false);
    for (let i = 0; i <= 100; i += 2) {
      await new Promise(r => setTimeout(r, 100));
      setProgress(i);
    }
    setGenerating(false);
    setOutputReady(true);
    if (!seedLocked) setSeed(Math.floor(Math.random() * 999999));
  }, [generating, seedLocked]);

  const canGenerate = xCondFile && !generating;
  const generateBtnClass = [
    styles.generateBtn,
    canGenerate && !generating ? styles.generateBtnReady : '',
    !canGenerate && !generating ? styles.generateBtnDisabled : '',
    generating ? styles.generateBtnGenerating : '',
  ].filter(Boolean).join(' ');

  const filteredLibrary = LIBRARY_ITEMS.filter(item => {
    const matchCat = libraryCategory === 'All' || item.category === libraryCategory;
    const matchSearch = !librarySearch || item.name.toLowerCase().includes(librarySearch.toLowerCase());
    return matchCat && matchSearch;
  });

  return (
    <div className={styles.do1}>
      <div className={styles.header}>
        <h1 className={styles.title}>DO<span className={styles.titleAccent}>1</span></h1>
        <div className={styles.headerControls}>
          <button className={generateBtnClass} onClick={handleGenerate} disabled={!canGenerate} title={!xCondFile ? 'Load audio or MIDI into Input track' : 'Generate'}>
            {generating ? (<><i className="fa-solid fa-spinner fa-spin" /> {Math.round(progress)}%</>) : (<><i className="fa-solid fa-bolt" /> Generate</>)}
          </button>
        </div>
      </div>

      <div className={styles.transport}>
        <button className={`${styles.transportBtn} ${isPlaying ? styles.transportBtnActive : ''}`} onClick={() => setIsPlaying(!isPlaying)}>
          <i className={`fa-solid ${isPlaying ? 'fa-pause' : 'fa-play'}`} />
        </button>
        <button className={styles.transportBtn} onClick={() => { setIsPlaying(false); setCurrentTime(0); }}>
          <i className="fa-solid fa-stop" />
        </button>
        <button className={`${styles.transportBtn} ${isLooping ? styles.transportBtnActive : ''}`} onClick={() => setIsLooping(!isLooping)}>
          <i className="fa-solid fa-repeat" />
        </button>
        <span className={styles.transportTime}>{formatTime(currentTime)}</span>
        <div className={styles.transportBpm}>
          <input type="number" className={styles.transportBpmInput} value={bpm} onChange={(e) => setBpm(Number(e.target.value))} min={20} max={300} />
          <span>BPM</span>
        </div>
        <div className={styles.transportSpacer} />
        <div className={styles.toolSelector}>
          <button className={`${styles.transportBtn} ${automationTool === 'point' ? styles.transportBtnActive : ''}`} onClick={() => setAutomationTool('point')} title="Point tool: click to add, drag to move, right-click to delete">
            <i className="fa-solid fa-circle-dot" />
          </button>
          <button className={`${styles.transportBtn} ${automationTool === 'draw' ? styles.transportBtnActive : ''}`} onClick={() => setAutomationTool('draw')} title="Draw tool: freehand draw (smoothed)">
            <i className="fa-solid fa-pencil" />
          </button>
        </div>
        <button className={`${styles.transportBtn} ${splitLanes ? styles.transportBtnActive : ''}`} onClick={() => setSplitLanes(!splitLanes)} title={splitLanes ? 'Merge Mask & CFG lanes' : 'Split Mask & CFG into separate lanes'}>
          <i className={`fa-solid ${splitLanes ? 'fa-compress' : 'fa-expand'}`} />
        </button>
        {generating && (
          <div className={styles.progressBar} style={{ width: 200 }}>
            <div className={styles.progressFill} style={{ width: `${progress}%` }} />
          </div>
        )}
      </div>

      <div className={styles.workspace}>
        {/* Input Track */}
        <div className={styles.trackRow}>
          <div className={styles.trackHeader}>
            <div className={styles.trackLabel}>Input</div>
            <div className={styles.trackSublabel}>{xCondFile ? xCondFile.name : xCondType === 'midi' ? 'MIDI' : 'Audio'}</div>
            <div className={styles.trackHeaderControls}>
              <button className={`${styles.trackSmallBtn} ${xCondType === 'audio' ? styles.trackSmallBtnActive : ''}`} onClick={() => setXCondType('audio')} title="Audio mode"><i className="fa-solid fa-wave-square" /></button>
              <button className={`${styles.trackSmallBtn} ${xCondType === 'midi' ? styles.trackSmallBtnActive : ''}`} onClick={() => setXCondType('midi')} title="MIDI mode"><i className="fa-solid fa-music" /></button>
              <button className={styles.trackSmallBtn} title="Mute">M</button>
              <button className={styles.trackSmallBtn} title="Solo">S</button>
            </div>
          </div>
          <div className={`${styles.trackContent} ${styles.trackContentTall}`}>
            {xCondFile ? (
              <div className={styles.waveformContainer}>
                <canvas ref={xCondCanvasRef} className={styles.waveformCanvas} />
                <span className={styles.waveformLabel}>{xCondFile.name}</span>
              </div>
            ) : (
              <div className={styles.dropZone} onDrop={(e) => handleDrop(e, 'xcond')} onDragOver={(e) => e.preventDefault()} onClick={() => handleFileClick('xcond')}>
                <span className={styles.dropZoneText}><i className="fa-solid fa-cloud-arrow-up" /> Drop audio or MIDI file here</span>
              </div>
            )}
          </div>
        </div>

        {/* Reference Track */}
        <div className={styles.trackRow}>
          <div className={styles.trackHeader}>
            <div className={styles.trackLabel}>Reference</div>
            <div className={styles.trackSublabel}>{xRefFile ? xRefFile.name : 'None'}</div>
            <div className={styles.trackHeaderControls}>
              <button className={styles.trackSmallBtn} title="From Selection"><i className="fa-solid fa-crop" /></button>
            </div>
          </div>
          <div className={styles.trackContent}>
            {xRefFile ? (
              <div className={styles.waveformContainer}><span className={styles.waveformLabel}>{xRefFile.name}</span></div>
            ) : (
              <div className={styles.dropZone} onDrop={(e) => handleDrop(e, 'xref')} onDragOver={(e) => e.preventDefault()} onClick={() => handleFileClick('xref')}>
                <span className={styles.dropZoneText}><i className="fa-solid fa-music" /> Drop reference audio or drag from library</span>
              </div>
            )}
          </div>
        </div>

        {/* Automation */}
        {splitLanes ? (
          <>
            <div className={styles.trackRow}>
              <div className={styles.trackHeader}>
                <div className={styles.trackLabel}>Mask</div>
                <div className={styles.trackSublabel}>0.0 - 1.0</div>
              </div>
              <AutomationLane points={maskPoints} setPoints={setMaskPoints} maxVal={1.0} color="#D4A843" label="Mask" totalDuration={defaultDuration} tool={automationTool} />
            </div>
            <div className={styles.trackRow}>
              <div className={styles.trackHeader}>
                <div className={styles.trackLabel}>CFG Scale</div>
                <div className={styles.trackSublabel}>0.0 - 3.0</div>
              </div>
              <AutomationLane points={cfgPoints} setPoints={setCfgPoints} maxVal={3.0} color="#2E75B6" label="CFG" totalDuration={defaultDuration} tool={automationTool} />
            </div>
          </>
        ) : (
          <div className={styles.trackRow}>
            <div className={styles.trackHeader}>
              <div className={styles.trackLabel}>Automation</div>
              <div className={styles.trackSublabel}>Mask + CFG</div>
              <div className={styles.automationLegend}>
                <span className={styles.legendDot} style={{ background: '#D4A843' }} /> Mask
                <span className={styles.legendDot} style={{ background: '#2E75B6', marginLeft: 8 }} /> CFG
              </div>
            </div>
            <UnifiedAutomationLane
              maskPoints={maskPoints} setMaskPoints={setMaskPoints}
              cfgPoints={cfgPoints} setCfgPoints={setCfgPoints}
              totalDuration={defaultDuration} tool={automationTool}
            />
          </div>
        )}

        {/* Output Track */}
        <div className={styles.trackRow}>
          <div className={styles.trackHeader}>
            <div className={styles.trackLabel}>Output</div>
            <div className={styles.trackSublabel}>{outputReady ? 'Ready' : 'Waiting'}</div>
            <div className={styles.trackHeaderControls}>
              <button className={styles.trackSmallBtn} title="Mute">M</button>
              <button className={styles.trackSmallBtn} title="Solo">S</button>
              {outputReady && (
                <button className={styles.trackSmallBtn} onClick={handleGenerate} title="Re-generate (new seed)"><i className="fa-solid fa-rotate" /></button>
              )}
            </div>
          </div>
          <div className={`${styles.trackContent} ${styles.trackContentTall}`}>
            {outputReady ? (
              <div className={styles.waveformContainer}>
                <canvas ref={outputCanvasRef} className={styles.waveformCanvas} />
                <span className={styles.waveformLabel}>Output</span>
              </div>
            ) : (
              <div className={styles.waveformContainer} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span className={styles.dropZoneText} style={{ border: 'none' }}>
                  <i className="fa-solid fa-wave-square" style={{ opacity: 0.3 }} />
                  {generating ? 'Generating...' : 'Output will appear here'}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom Panel */}
      <div className={styles.bottomPanel}>
        <div className={styles.bottomTabs}>
          {['presets', 'properties', 'library'].map(tab => (
            <button key={tab} className={`${styles.bottomTab} ${activeTab === tab ? styles.bottomTabActive : ''}`} onClick={() => setActiveTab(tab)}>
              {tab === 'presets' ? 'Presets' : tab === 'properties' ? 'Properties' : 'Reference Library'}
            </button>
          ))}
        </div>
        <div className={styles.bottomContent}>
          {activeTab === 'presets' && (
            <div className={styles.presetsGrid}>
              {PRESETS.map(preset => (
                <div key={preset.name} className={`${styles.presetCard} ${selectedPreset === preset.name ? styles.presetCardActive : ''}`} onClick={() => applyPreset(preset)}>
                  <div className={styles.presetName}>{preset.name}</div>
                  <div className={styles.presetDesc}>{preset.desc}</div>
                  <div className={styles.presetValues}>
                    <span className={styles.presetValueMask}>M: {preset.mask !== null ? preset.mask.toFixed(2) : 'auto'}</span>
                    <span className={styles.presetValueCfg}>CFG: {preset.cfg.toFixed(1)}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
          {activeTab === 'properties' && (
            <div className={styles.propertiesGrid}>
              <div className={styles.propertyItem}>
                <span className={styles.propertyLabel}>Seed</span>
                <div className={styles.propertyRow}>
                  <input type="number" className={styles.propertyInput} value={seed} onChange={(e) => setSeed(Number(e.target.value))} style={{ flex: 1 }} />
                  <button className={`${styles.lockBtn} ${seedLocked ? styles.lockBtnActive : ''}`} onClick={() => setSeedLocked(!seedLocked)} title={seedLocked ? 'Unlock seed' : 'Lock seed'}>
                    <i className={`fa-solid ${seedLocked ? 'fa-lock' : 'fa-lock-open'}`} />
                  </button>
                  <button className={styles.randomBtn} onClick={() => setSeed(Math.floor(Math.random() * 999999))}>Random</button>
                </div>
              </div>
              <div className={styles.propertyItem}>
                <span className={styles.propertyLabel}>Steps</span>
                <input type="number" className={styles.propertyInput} value={steps} onChange={(e) => setSteps(Number(e.target.value))} min={1} max={200} />
              </div>
              <div className={styles.propertyItem}>
                <span className={styles.propertyLabel}>Output Duration</span>
                <input type="text" className={styles.propertyInput} value="Auto (match input)" readOnly />
              </div>
              <div className={styles.propertyItem}>
                <span className={styles.propertyLabel}>Sample Rate</span>
                <input type="text" className={styles.propertyInput} value="48,000 Hz" readOnly style={{ opacity: 0.5 }} />
              </div>
            </div>
          )}
          {activeTab === 'library' && (
            <>
              <div className={styles.librarySearch}>
                <input type="text" className={styles.librarySearchInput} placeholder="Search references..." value={librarySearch} onChange={(e) => setLibrarySearch(e.target.value)} />
                {LIBRARY_CATEGORIES.map(cat => (
                  <button key={cat} className={`${styles.libraryCategoryBtn} ${libraryCategory === cat ? styles.libraryCategoryBtnActive : ''}`} onClick={() => setLibraryCategory(cat)}>{cat}</button>
                ))}
              </div>
              <div className={styles.libraryGrid}>
                {filteredLibrary.map(item => (
                  <div key={item.name} className={styles.libraryCard} onClick={() => setXRefFile({ name: item.name })} title="Click to load as reference">
                    <div className={styles.libraryCardWaveform} />
                    <div className={styles.libraryCardName}>{item.name}</div>
                    <div className={styles.libraryCardMeta}>{item.category} - {item.duration}</div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default DO1;
