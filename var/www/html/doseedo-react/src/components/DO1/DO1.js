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

// ─── Point-based Automation Lane (rewritten from AutomationWindow.js pattern) ───
// Click to add point, drag to move, right-click to delete.
// "draw" tool: freehand with smoothing.
const LANE_HEIGHT = 64;

const AutomationLane = ({ points, setPoints, maxVal, color, label, totalDuration, tool }) => {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const isDrawingRef = useRef(false);
  const drawPathRef = useRef([]);
  const widthRef = useRef(400);

  const POINT_RADIUS = 6;
  const GRAB_THRESHOLD = 15;

  // Coordinate conversions
  const valToY = useCallback((val) => {
    const h = LANE_HEIGHT;
    return h - (val / maxVal) * h;
  }, [maxVal]);

  const yToVal = useCallback((y) => {
    const h = LANE_HEIGHT;
    return Math.max(0, Math.min(maxVal, (1 - y / h) * maxVal));
  }, [maxVal]);

  const timeToX = useCallback((t) => {
    return (t / totalDuration) * widthRef.current;
  }, [totalDuration]);

  const xToTime = useCallback((x) => {
    return Math.max(0, Math.min(totalDuration, (x / widthRef.current) * totalDuration));
  }, [totalDuration]);

  // Resize canvas to match container
  const resizeCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;

    // Set display size explicitly via inline style
    canvas.style.width = '100%';
    canvas.style.height = LANE_HEIGHT + 'px';

    // Measure actual rendered size
    const rect = canvas.getBoundingClientRect();
    widthRef.current = rect.width;

    // Set canvas buffer resolution
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    // Scale context
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }, []);

  // Draw the automation lane
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;

    if (w === 0 || h === 0) return;

    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    const divs = maxVal <= 1 ? 4 : 6;
    for (let i = 1; i < divs; i++) {
      const y = (h / divs) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Sort points by time
    const sorted = [...points].sort((a, b) => a.time - b.time);
    if (sorted.length === 0) return;

    // Draw automation line
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    sorted.forEach((pt, i) => {
      const x = (pt.time / totalDuration) * w;
      const y = h - (pt.value / maxVal) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Fill under curve (very subtle)
    const lastPt = sorted[sorted.length - 1];
    ctx.lineTo((lastPt.time / totalDuration) * w, h);
    ctx.lineTo((sorted[0].time / totalDuration) * w, h);
    ctx.closePath();
    ctx.globalAlpha = 0.05;
    ctx.fillStyle = color;
    ctx.fill();
    ctx.globalAlpha = 1.0;

    // Draw points (skip edge points)
    sorted.forEach((pt) => {
      if (pt.isEdge) return;
      const origIdx = points.indexOf(pt);
      const x = (pt.time / totalDuration) * w;
      const y = h - (pt.value / maxVal) * h;
      const isActive = origIdx === selectedIdx;

      ctx.fillStyle = isActive ? '#ffffff' : color;
      ctx.strokeStyle = 'rgba(0,0,0,0.5)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, POINT_RADIUS, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  }, [points, color, maxVal, totalDuration, selectedIdx]);

  // Initialize: resize then draw
  useEffect(() => {
    // Use double rAF to ensure layout is complete (same as AutomationWindow.js)
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        resizeCanvas();
        draw();
      });
    });
  }, [resizeCanvas, draw]);

  // Redraw when points or selection change
  useEffect(() => {
    draw();
  }, [draw]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      resizeCanvas();
      draw();
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [resizeCanvas, draw]);

  // Get mouse coords relative to canvas
  const getCoords = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  // Find a point near the cursor
  const findPoint = (mx, my) => {
    const canvas = canvasRef.current;
    if (!canvas) return -1;
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;

    for (let i = 0; i < points.length; i++) {
      const pt = points[i];
      if (pt.isEdge) continue;
      const px = (pt.time / totalDuration) * w;
      const py = h - (pt.value / maxVal) * h;
      const dist = Math.sqrt((mx - px) ** 2 + (my - py) ** 2);
      if (dist <= GRAB_THRESHOLD) return i;
    }
    return -1;
  };

  // Smooth freehand path
  const smoothPath = (rawPath) => {
    if (rawPath.length < 3) return rawPath;
    const step = Math.max(1, Math.floor(rawPath.length / Math.max(5, rawPath.length / 3)));
    const sampled = [];
    for (let i = 0; i < rawPath.length; i += step) sampled.push(rawPath[i]);
    if (sampled[sampled.length - 1] !== rawPath[rawPath.length - 1]) sampled.push(rawPath[rawPath.length - 1]);
    return sampled.map((pt, i) => {
      if (i === 0 || i === sampled.length - 1) return pt;
      const prev = sampled[i - 1];
      const next = sampled[i + 1];
      return { time: pt.time, value: (prev.value + pt.value + next.value) / 3 };
    });
  };

  const handleMouseDown = (e) => {
    if (e.button === 2) return;
    const { x, y } = getCoords(e);
    const w = canvasRef.current.offsetWidth;
    const h = canvasRef.current.offsetHeight;

    if (tool === 'draw') {
      isDrawingRef.current = true;
      drawPathRef.current = [{ time: xToTime(x), value: yToVal(y) }];
      return;
    }

    // Point tool
    const idx = findPoint(x, y);
    if (idx !== -1) {
      setSelectedIdx(idx);
      setIsDragging(true);
      canvasRef.current.style.cursor = 'grabbing';
    } else {
      const time = (x / w) * totalDuration;
      const value = Math.max(0, Math.min(maxVal, (1 - y / h) * maxVal));
      const newPoints = [...points, { time, value, isEdge: false }].sort((a, b) => a.time - b.time);
      setPoints(newPoints);
    }
  };

  const handleMouseMove = (e) => {
    const { x, y } = getCoords(e);
    const canvas = canvasRef.current;
    if (!canvas) return;
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;

    if (tool === 'draw' && isDrawingRef.current) {
      drawPathRef.current.push({ time: xToTime(x), value: yToVal(y) });
      const path = drawPathRef.current;
      const minT = Math.min(path[0].time, path[path.length - 1].time);
      const maxT = Math.max(path[0].time, path[path.length - 1].time);
      const kept = points.filter(p => p.isEdge || p.time < minT || p.time > maxT);
      const preview = [...kept, ...path.map(p => ({ ...p, isEdge: false }))].sort((a, b) => a.time - b.time);
      setPoints(preview);
      return;
    }

    if (isDragging && selectedIdx !== null) {
      const time = Math.max(0, Math.min(totalDuration, (x / w) * totalDuration));
      const value = Math.max(0, Math.min(maxVal, (1 - y / h) * maxVal));

      const updatedPoints = points.map((pt, i) =>
        i === selectedIdx ? { ...pt, time, value } : pt
      );
      const sorted = [...updatedPoints].sort((a, b) => a.time - b.time);
      const draggedPt = updatedPoints[selectedIdx];
      const newIdx = sorted.findIndex(p => p.time === draggedPt.time && p.value === draggedPt.value);
      setSelectedIdx(newIdx);
      setPoints(sorted);
    } else if (!isDragging) {
      // Hover cursor
      const idx = findPoint(x, y);
      canvas.style.cursor = idx !== -1 ? 'grab' : (tool === 'draw' ? 'crosshair' : 'crosshair');
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
        const smoothed = smoothPath(path).map(p => ({ ...p, isEdge: false }));
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

  // Global mouseup to catch drags that leave the canvas
  useEffect(() => {
    const up = () => {
      setIsDragging(false);
      setSelectedIdx(null);
      isDrawingRef.current = false;
    };
    window.addEventListener('mouseup', up);
    return () => window.removeEventListener('mouseup', up);
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        flex: 1,
        minWidth: 0,
        height: LANE_HEIGHT + 'px',
        position: 'relative',
        background: '#12121A',
        cursor: 'crosshair',
      }}
    >
      <canvas
        ref={canvasRef}
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

// ─── Main DO1 Component ───────────────────────────────────────
const DO1 = () => {
  // Transport
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(false);
  const [bpm, setBpm] = useState(120);
  const [currentTime, setCurrentTime] = useState(0);

  // Tracks
  const [xCondFile, setXCondFile] = useState(null);
  const [xCondType, setXCondType] = useState('audio');
  const [xRefFile, setXRefFile] = useState(null);
  const [outputReady, setOutputReady] = useState(false);

  // Automation: point-based
  const defaultDuration = 30;
  const [splitLanes, setSplitLanes] = useState(false);
  const [automationTool, setAutomationTool] = useState('point'); // 'point' | 'draw'

  // Unified lane: mask points (value 0-1) and cfg points (value 0-3)
  const [maskPoints, setMaskPoints] = useState([
    { time: 0, value: 1.0, isEdge: true },
    { time: defaultDuration, value: 1.0, isEdge: true },
  ]);
  const [cfgPoints, setCfgPoints] = useState([
    { time: 0, value: 1.0, isEdge: true },
    { time: defaultDuration, value: 1.0, isEdge: true },
  ]);

  // Generation
  const [seed, setSeed] = useState(42);
  const [seedLocked, setSeedLocked] = useState(false);
  const [steps, setSteps] = useState(50);
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [selectedPreset, setSelectedPreset] = useState(null);

  // UI
  const [activeTab, setActiveTab] = useState('presets');
  const [libraryCategory, setLibraryCategory] = useState('All');
  const [librarySearch, setLibrarySearch] = useState('');

  // Waveform canvases
  const xCondCanvasRef = useRef(null);
  const outputCanvasRef = useRef(null);

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${m}:${String(s).padStart(2, '0')}.${String(ms).padStart(2, '0')}`;
  };

  // Draw waveform placeholder
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

  // File handling
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

  // Apply preset
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

  // Generate (mock)
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
      {/* Header */}
      <div className={styles.header}>
        <h1 className={styles.title}>DO<span className={styles.titleAccent}>1</span></h1>
        <div className={styles.headerControls}>
          <button
            className={generateBtnClass}
            onClick={handleGenerate}
            disabled={!canGenerate}
            title={!xCondFile ? 'Load audio or MIDI into Input track' : 'Generate'}
          >
            {generating ? (
              <><i className="fa-solid fa-spinner fa-spin" /> {Math.round(progress)}%</>
            ) : (
              <><i className="fa-solid fa-bolt" /> Generate</>
            )}
          </button>
        </div>
      </div>

      {/* Transport Bar */}
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

        {/* Automation tool selector */}
        <div className={styles.toolSelector}>
          <button
            className={`${styles.transportBtn} ${automationTool === 'point' ? styles.transportBtnActive : ''}`}
            onClick={() => setAutomationTool('point')}
            title="Point tool: click to add, drag to move, right-click to delete"
          >
            <i className="fa-solid fa-circle-dot" />
          </button>
          <button
            className={`${styles.transportBtn} ${automationTool === 'draw' ? styles.transportBtnActive : ''}`}
            onClick={() => setAutomationTool('draw')}
            title="Draw tool: freehand draw (smoothed)"
          >
            <i className="fa-solid fa-pencil" />
          </button>
        </div>

        {/* Split/unified toggle */}
        <button
          className={`${styles.transportBtn} ${splitLanes ? styles.transportBtnActive : ''}`}
          onClick={() => setSplitLanes(!splitLanes)}
          title={splitLanes ? 'Merge Mask & CFG lanes' : 'Split Mask & CFG into separate lanes'}
        >
          <i className={`fa-solid ${splitLanes ? 'fa-compress' : 'fa-expand'}`} />
        </button>

        {generating && (
          <div className={styles.progressBar} style={{ width: 200 }}>
            <div className={styles.progressFill} style={{ width: `${progress}%` }} />
          </div>
        )}
      </div>

      {/* Track Workspace */}
      <div className={styles.workspace}>
        {/* x_cond Track */}
        <div className={styles.trackRow}>
          <div className={styles.trackHeader}>
            <div className={styles.trackLabel}>Input</div>
            <div className={styles.trackSublabel}>{xCondFile ? xCondFile.name : xCondType === 'midi' ? 'MIDI' : 'Audio'}</div>
            <div className={styles.trackHeaderControls}>
              <button className={`${styles.trackSmallBtn} ${xCondType === 'audio' ? styles.trackSmallBtnActive : ''}`} onClick={() => setXCondType('audio')} title="Audio mode">
                <i className="fa-solid fa-wave-square" />
              </button>
              <button className={`${styles.trackSmallBtn} ${xCondType === 'midi' ? styles.trackSmallBtnActive : ''}`} onClick={() => setXCondType('midi')} title="MIDI mode">
                <i className="fa-solid fa-music" />
              </button>
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

        {/* x_ref Track */}
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

        {/* Automation Lanes */}
        {splitLanes ? (
          <>
            {/* Separate Mask Lane */}
            <div className={styles.trackRow}>
              <div className={styles.trackHeader}>
                <div className={styles.trackLabel}>Mask</div>
                <div className={styles.trackSublabel}>0.0 - 1.0</div>
              </div>
              <AutomationLane
                points={maskPoints}
                setPoints={setMaskPoints}
                maxVal={1.0}
                color="#D4A843"
                label="Mask"
                totalDuration={defaultDuration}
                tool={automationTool}
              />
            </div>
            {/* Separate CFG Lane */}
            <div className={styles.trackRow}>
              <div className={styles.trackHeader}>
                <div className={styles.trackLabel}>CFG Scale</div>
                <div className={styles.trackSublabel}>0.0 - 3.0</div>
              </div>
              <AutomationLane
                points={cfgPoints}
                setPoints={setCfgPoints}
                maxVal={3.0}
                color="#2E75B6"
                label="CFG"
                totalDuration={defaultDuration}
                tool={automationTool}
              />
            </div>
          </>
        ) : (
          /* Unified Mask + CFG Lane */
          <div className={styles.trackRow}>
            <div className={styles.trackHeader}>
              <div className={styles.trackLabel}>Automation</div>
              <div className={styles.trackSublabel}>Mask + CFG</div>
              <div className={styles.automationLegend}>
                <span className={styles.legendDot} style={{ background: '#D4A843' }} /> Mask
                <span className={styles.legendDot} style={{ background: '#2E75B6', marginLeft: 8 }} /> CFG
              </div>
            </div>
            <div style={{ flex: 1, position: 'relative', height: LANE_HEIGHT + 'px', minWidth: 0 }}>
              <AutomationLane
                points={maskPoints}
                setPoints={setMaskPoints}
                maxVal={1.0}
                color="#D4A843"
                label=""
                totalDuration={defaultDuration}
                tool={automationTool}
              />
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, pointerEvents: 'none' }}>
                <AutomationLane
                  points={cfgPoints}
                  setPoints={setCfgPoints}
                  maxVal={3.0}
                  color="#2E75B6"
                  label=""
                  totalDuration={defaultDuration}
                  tool={automationTool}
                />
              </div>
            </div>
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
                <button className={styles.trackSmallBtn} onClick={handleGenerate} title="Re-generate (new seed)">
                  <i className="fa-solid fa-rotate" />
                </button>
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
