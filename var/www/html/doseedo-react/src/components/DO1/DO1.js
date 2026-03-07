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

const FRAME_RATE = 25; // Oobleck latent framerate

/**
 * DO1 - Latent-to-latent audio generation interface
 * Unified workspace controlling x_cond, x_ref, mask, and CFG automation
 */
const DO1 = () => {
  // Transport state
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(false);
  const [bpm, setBpm] = useState(120);
  const [currentTime, setCurrentTime] = useState(0);

  // Track state
  const [xCondFile, setXCondFile] = useState(null);
  const [xCondType, setXCondType] = useState('audio'); // 'audio' | 'midi'
  const [xRefFile, setXRefFile] = useState(null);
  const [outputReady, setOutputReady] = useState(false);

  // Automation state (default arrays for 30 seconds at 25Hz = 750 frames)
  const defaultDuration = 30;
  const frameCount = defaultDuration * FRAME_RATE;
  const [maskAutomation, setMaskAutomation] = useState(() => new Float32Array(frameCount).fill(1.0));
  const [cfgAutomation, setCfgAutomation] = useState(() => new Float32Array(frameCount).fill(1.0));
  const [maskDrawTool, setMaskDrawTool] = useState('pencil');
  const [cfgDrawTool, setCfgDrawTool] = useState('pencil');

  // Generation state
  const [seed, setSeed] = useState(42);
  const [seedLocked, setSeedLocked] = useState(false);
  const [steps, setSteps] = useState(50);
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [selectedPreset, setSelectedPreset] = useState(null);

  // UI state
  const [activeTab, setActiveTab] = useState('presets');
  const [libraryCategory, setLibraryCategory] = useState('All');
  const [librarySearch, setLibrarySearch] = useState('');

  // Refs for canvases
  const maskCanvasRef = useRef(null);
  const cfgCanvasRef = useRef(null);
  const xCondCanvasRef = useRef(null);
  const outputCanvasRef = useRef(null);
  const isDrawingRef = useRef(false);

  // Waveform data refs
  const xCondWaveformRef = useRef(null);
  const outputWaveformRef = useRef(null);

  // Format time display
  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${m}:${String(s).padStart(2, '0')}.${String(ms).padStart(2, '0')}`;
  };

  // Draw automation lane
  const drawAutomation = useCallback((canvas, data, type) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    const divisions = type === 'mask' ? 5 : 6;
    for (let i = 1; i < divisions; i++) {
      const y = (h / divisions) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Draw the automation curve
    const maxVal = type === 'mask' ? 1.0 : 3.0;
    ctx.beginPath();
    ctx.strokeStyle = type === 'mask' ? '#D4A843' : '#2E75B6';
    ctx.lineWidth = 2;

    for (let i = 0; i < data.length; i++) {
      const x = (i / data.length) * w;
      const y = h - (data[i] / maxVal) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fill under curve
    const lastX = ((data.length - 1) / data.length) * w;
    const lastY = h - (data[data.length - 1] / maxVal) * h;
    ctx.lineTo(lastX, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = type === 'mask'
      ? 'rgba(212, 168, 67, 0.08)'
      : 'rgba(46, 117, 182, 0.08)';
    ctx.fill();
  }, []);

  // Draw waveform placeholder
  const drawWaveform = useCallback((canvas, color, hasData) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (!hasData) return;

    // Generate a fake waveform for display
    ctx.fillStyle = color;
    const centerY = h / 2;
    const barWidth = 2;
    const gap = 1;
    for (let x = 0; x < w; x += barWidth + gap) {
      const amplitude = (Math.sin(x * 0.02) * 0.3 + Math.sin(x * 0.05) * 0.2 + Math.random() * 0.3) * centerY * 0.8;
      ctx.fillRect(x, centerY - amplitude, barWidth, amplitude * 2);
    }
  }, []);

  // Redraw canvases on state change
  useEffect(() => {
    drawAutomation(maskCanvasRef.current, maskAutomation, 'mask');
  }, [maskAutomation, drawAutomation]);

  useEffect(() => {
    drawAutomation(cfgCanvasRef.current, cfgAutomation, 'cfg');
  }, [cfgAutomation, drawAutomation]);

  useEffect(() => {
    drawWaveform(xCondCanvasRef.current, 'rgba(74, 155, 142, 0.6)', !!xCondFile);
  }, [xCondFile, drawWaveform]);

  useEffect(() => {
    drawWaveform(outputCanvasRef.current, 'rgba(212, 168, 67, 0.6)', outputReady);
  }, [outputReady, drawWaveform]);

  // Resize canvases to match container
  useEffect(() => {
    const resizeCanvas = (canvas) => {
      if (!canvas) return;
      const parent = canvas.parentElement;
      if (parent) {
        canvas.width = parent.clientWidth;
        canvas.height = parent.clientHeight;
      }
    };

    const handleResize = () => {
      resizeCanvas(maskCanvasRef.current);
      resizeCanvas(cfgCanvasRef.current);
      resizeCanvas(xCondCanvasRef.current);
      resizeCanvas(outputCanvasRef.current);
      drawAutomation(maskCanvasRef.current, maskAutomation, 'mask');
      drawAutomation(cfgCanvasRef.current, cfgAutomation, 'cfg');
      drawWaveform(xCondCanvasRef.current, 'rgba(74, 155, 142, 0.6)', !!xCondFile);
      drawWaveform(outputCanvasRef.current, 'rgba(212, 168, 67, 0.6)', outputReady);
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [maskAutomation, cfgAutomation, xCondFile, outputReady, drawAutomation, drawWaveform]);

  // Handle automation drawing
  const handleAutomationDraw = useCallback((e, canvasRef, setAutomation, maxVal) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const frameIndex = Math.floor((x / rect.width) * frameCount);
    const value = Math.max(0, Math.min(maxVal, (1 - y / rect.height) * maxVal));

    if (frameIndex >= 0 && frameIndex < frameCount) {
      setAutomation(prev => {
        const next = new Float32Array(prev);
        // Draw a small region around the cursor for smoother drawing
        const radius = 3;
        for (let i = Math.max(0, frameIndex - radius); i <= Math.min(frameCount - 1, frameIndex + radius); i++) {
          const blend = 1 - Math.abs(i - frameIndex) / (radius + 1);
          next[i] = prev[i] + (value - prev[i]) * blend;
        }
        return next;
      });
    }
  }, [frameCount]);

  const handleMouseDown = useCallback((e, canvasRef, setAutomation, maxVal) => {
    isDrawingRef.current = true;
    handleAutomationDraw(e, canvasRef, setAutomation, maxVal);
  }, [handleAutomationDraw]);

  const handleMouseMove = useCallback((e, canvasRef, setAutomation, maxVal) => {
    if (!isDrawingRef.current) return;
    handleAutomationDraw(e, canvasRef, setAutomation, maxVal);
  }, [handleAutomationDraw]);

  const handleMouseUp = useCallback(() => {
    isDrawingRef.current = false;
  }, []);

  useEffect(() => {
    window.addEventListener('mouseup', handleMouseUp);
    return () => window.removeEventListener('mouseup', handleMouseUp);
  }, [handleMouseUp]);

  // Handle file drops
  const handleDrop = useCallback((e, target) => {
    e.preventDefault();
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (target === 'xcond') {
        setXCondFile(file);
        setXCondType(file.name.endsWith('.mid') || file.name.endsWith('.midi') ? 'midi' : 'audio');
      } else if (target === 'xref') {
        setXRefFile(file);
      }
    }
  }, []);

  const handleDragOver = (e) => e.preventDefault();

  // Handle file input click
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
      setMaskAutomation(new Float32Array(frameCount).fill(preset.mask));
    }
    setCfgAutomation(new Float32Array(frameCount).fill(preset.cfg));
  }, [frameCount]);

  // Generate (mock)
  const handleGenerate = useCallback(async () => {
    if (generating) return;
    setGenerating(true);
    setProgress(0);
    setOutputReady(false);

    // Mock generation progress
    for (let i = 0; i <= 100; i += 2) {
      await new Promise(r => setTimeout(r, 100));
      setProgress(i);
    }

    setGenerating(false);
    setOutputReady(true);
    if (!seedLocked) {
      setSeed(Math.floor(Math.random() * 999999));
    }
  }, [generating, seedLocked]);

  // Determine generate button state
  const canGenerate = xCondFile && !generating;
  const generateBtnClass = [
    styles.generateBtn,
    canGenerate && !generating ? styles.generateBtnReady : '',
    !canGenerate && !generating ? styles.generateBtnDisabled : '',
    generating ? styles.generateBtnGenerating : '',
  ].filter(Boolean).join(' ');

  // Filtered library items
  const filteredLibrary = LIBRARY_ITEMS.filter(item => {
    const matchCategory = libraryCategory === 'All' || item.category === libraryCategory;
    const matchSearch = !librarySearch || item.name.toLowerCase().includes(librarySearch.toLowerCase());
    return matchCategory && matchSearch;
  });

  return (
    <div className={styles.do1}>
      {/* Header */}
      <div className={styles.header}>
        <h1 className={styles.title}>
          DO<span className={styles.titleAccent}>1</span>
        </h1>
        <div className={styles.headerControls}>
          <button
            className={generateBtnClass}
            onClick={handleGenerate}
            disabled={!canGenerate}
            title={!xCondFile ? 'Load audio or MIDI into Input track' : 'Generate'}
          >
            {generating ? (
              <>
                <i className="fa-solid fa-spinner fa-spin" />
                {Math.round(progress)}%
              </>
            ) : (
              <>
                <i className="fa-solid fa-bolt" />
                Generate
              </>
            )}
          </button>
        </div>
      </div>

      {/* Transport Bar */}
      <div className={styles.transport}>
        <button
          className={`${styles.transportBtn} ${isPlaying ? styles.transportBtnActive : ''}`}
          onClick={() => setIsPlaying(!isPlaying)}
        >
          <i className={`fa-solid ${isPlaying ? 'fa-pause' : 'fa-play'}`} />
        </button>
        <button className={styles.transportBtn} onClick={() => { setIsPlaying(false); setCurrentTime(0); }}>
          <i className="fa-solid fa-stop" />
        </button>
        <button
          className={`${styles.transportBtn} ${isLooping ? styles.transportBtnActive : ''}`}
          onClick={() => setIsLooping(!isLooping)}
        >
          <i className="fa-solid fa-repeat" />
        </button>

        <span className={styles.transportTime}>{formatTime(currentTime)}</span>

        <div className={styles.transportBpm}>
          <input
            type="number"
            className={styles.transportBpmInput}
            value={bpm}
            onChange={(e) => setBpm(Number(e.target.value))}
            min={20}
            max={300}
          />
          <span>BPM</span>
        </div>

        <div className={styles.transportSpacer} />

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
            <div className={styles.trackSublabel}>
              {xCondFile ? xCondFile.name : xCondType === 'midi' ? 'MIDI' : 'Audio'}
            </div>
            <div className={styles.trackHeaderControls}>
              <button
                className={`${styles.trackSmallBtn} ${xCondType === 'audio' ? styles.trackSmallBtnActive : ''}`}
                onClick={() => setXCondType('audio')}
                title="Audio mode"
              >
                <i className="fa-solid fa-waveform-lines" />
              </button>
              <button
                className={`${styles.trackSmallBtn} ${xCondType === 'midi' ? styles.trackSmallBtnActive : ''}`}
                onClick={() => setXCondType('midi')}
                title="MIDI mode"
              >
                <i className="fa-solid fa-piano-keyboard" />
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
              <div
                className={styles.dropZone}
                onDrop={(e) => handleDrop(e, 'xcond')}
                onDragOver={handleDragOver}
                onClick={() => handleFileClick('xcond')}
              >
                <span className={styles.dropZoneText}>
                  <i className="fa-solid fa-cloud-arrow-up" />
                  Drop audio or MIDI file here
                </span>
              </div>
            )}
          </div>
        </div>

        {/* x_ref Track */}
        <div className={styles.trackRow}>
          <div className={styles.trackHeader}>
            <div className={styles.trackLabel}>Reference</div>
            <div className={styles.trackSublabel}>
              {xRefFile ? xRefFile.name : 'None'}
            </div>
            <div className={styles.trackHeaderControls}>
              <button className={styles.trackSmallBtn} title="From Selection">
                <i className="fa-solid fa-crop" />
              </button>
            </div>
          </div>
          <div className={styles.trackContent}>
            {xRefFile ? (
              <div className={styles.waveformContainer}>
                <span className={styles.waveformLabel}>{xRefFile.name}</span>
              </div>
            ) : (
              <div
                className={styles.dropZone}
                onDrop={(e) => handleDrop(e, 'xref')}
                onDragOver={handleDragOver}
                onClick={() => handleFileClick('xref')}
              >
                <span className={styles.dropZoneText}>
                  <i className="fa-solid fa-music" />
                  Drop reference audio or drag from library
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Mask Automation Lane */}
        <div className={styles.trackRow}>
          <div className={styles.trackHeader}>
            <div className={styles.trackLabel}>Mask</div>
            <div className={styles.trackSublabel}>0.0 - 1.0</div>
            <div className={styles.drawingTools}>
              {['pencil', 'line', 'square'].map(tool => (
                <button
                  key={tool}
                  className={`${styles.drawToolBtn} ${maskDrawTool === tool ? styles.drawToolBtnActive : ''}`}
                  onClick={() => setMaskDrawTool(tool)}
                  title={tool.charAt(0).toUpperCase() + tool.slice(1)}
                >
                  <i className={`fa-solid ${tool === 'pencil' ? 'fa-pencil' : tool === 'line' ? 'fa-minus' : 'fa-square'}`} />
                </button>
              ))}
            </div>
          </div>
          <div className={`${styles.trackContent} ${styles.automationMask}`}>
            <div className={styles.automationLane}>
              <canvas
                ref={maskCanvasRef}
                className={styles.automationCanvas}
                onMouseDown={(e) => handleMouseDown(e, maskCanvasRef, setMaskAutomation, 1.0)}
                onMouseMove={(e) => handleMouseMove(e, maskCanvasRef, setMaskAutomation, 1.0)}
              />
              <span className={styles.automationValue}>Mask</span>
            </div>
          </div>
        </div>

        {/* CFG Automation Lane */}
        <div className={styles.trackRow}>
          <div className={styles.trackHeader}>
            <div className={styles.trackLabel}>CFG Scale</div>
            <div className={styles.trackSublabel}>0.0 - 3.0</div>
            <div className={styles.drawingTools}>
              {['pencil', 'line', 'square'].map(tool => (
                <button
                  key={tool}
                  className={`${styles.drawToolBtn} ${cfgDrawTool === tool ? styles.drawToolBtnActive : ''}`}
                  onClick={() => setCfgDrawTool(tool)}
                  title={tool.charAt(0).toUpperCase() + tool.slice(1)}
                >
                  <i className={`fa-solid ${tool === 'pencil' ? 'fa-pencil' : tool === 'line' ? 'fa-minus' : 'fa-square'}`} />
                </button>
              ))}
            </div>
          </div>
          <div className={`${styles.trackContent} ${styles.automationCfg}`}>
            <div className={styles.automationLane}>
              <canvas
                ref={cfgCanvasRef}
                className={styles.automationCanvas}
                onMouseDown={(e) => handleMouseDown(e, cfgCanvasRef, setCfgAutomation, 3.0)}
                onMouseMove={(e) => handleMouseMove(e, cfgCanvasRef, setCfgAutomation, 3.0)}
              />
              <span className={styles.automationValue}>CFG</span>
            </div>
          </div>
        </div>

        {/* Output Track */}
        <div className={styles.trackRow}>
          <div className={styles.trackHeader}>
            <div className={styles.trackLabel}>Output</div>
            <div className={styles.trackSublabel}>{outputReady ? 'Ready' : 'Waiting'}</div>
            <div className={styles.trackHeaderControls}>
              <button className={styles.trackSmallBtn} title="Mute">M</button>
              <button className={styles.trackSmallBtn} title="Solo">S</button>
              {outputReady && (
                <button
                  className={styles.trackSmallBtn}
                  onClick={handleGenerate}
                  title="Re-generate (new seed)"
                >
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
                  <i className="fa-solid fa-waveform-lines" style={{ opacity: 0.3 }} />
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
            <button
              key={tab}
              className={`${styles.bottomTab} ${activeTab === tab ? styles.bottomTabActive : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab === 'presets' ? 'Presets' : tab === 'properties' ? 'Properties' : 'Reference Library'}
            </button>
          ))}
        </div>

        <div className={styles.bottomContent}>
          {/* Presets Tab */}
          {activeTab === 'presets' && (
            <div className={styles.presetsGrid}>
              {PRESETS.map(preset => (
                <div
                  key={preset.name}
                  className={`${styles.presetCard} ${selectedPreset === preset.name ? styles.presetCardActive : ''}`}
                  onClick={() => applyPreset(preset)}
                >
                  <div className={styles.presetName}>{preset.name}</div>
                  <div className={styles.presetDesc}>{preset.desc}</div>
                  <div className={styles.presetValues}>
                    <span className={styles.presetValueMask}>
                      M: {preset.mask !== null ? preset.mask.toFixed(2) : 'auto'}
                    </span>
                    <span className={styles.presetValueCfg}>
                      CFG: {preset.cfg.toFixed(1)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Properties Tab */}
          {activeTab === 'properties' && (
            <div className={styles.propertiesGrid}>
              <div className={styles.propertyItem}>
                <span className={styles.propertyLabel}>Seed</span>
                <div className={styles.propertyRow}>
                  <input
                    type="number"
                    className={styles.propertyInput}
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value))}
                    style={{ flex: 1 }}
                  />
                  <button
                    className={`${styles.lockBtn} ${seedLocked ? styles.lockBtnActive : ''}`}
                    onClick={() => setSeedLocked(!seedLocked)}
                    title={seedLocked ? 'Unlock seed' : 'Lock seed'}
                  >
                    <i className={`fa-solid ${seedLocked ? 'fa-lock' : 'fa-lock-open'}`} />
                  </button>
                  <button
                    className={styles.randomBtn}
                    onClick={() => setSeed(Math.floor(Math.random() * 999999))}
                  >
                    Random
                  </button>
                </div>
              </div>

              <div className={styles.propertyItem}>
                <span className={styles.propertyLabel}>Steps</span>
                <input
                  type="number"
                  className={styles.propertyInput}
                  value={steps}
                  onChange={(e) => setSteps(Number(e.target.value))}
                  min={1}
                  max={200}
                />
              </div>

              <div className={styles.propertyItem}>
                <span className={styles.propertyLabel}>Output Duration</span>
                <input
                  type="text"
                  className={styles.propertyInput}
                  value="Auto (match input)"
                  readOnly
                />
              </div>

              <div className={styles.propertyItem}>
                <span className={styles.propertyLabel}>Sample Rate</span>
                <input
                  type="text"
                  className={styles.propertyInput}
                  value="48,000 Hz"
                  readOnly
                  style={{ opacity: 0.5 }}
                />
              </div>
            </div>
          )}

          {/* Reference Library Tab */}
          {activeTab === 'library' && (
            <>
              <div className={styles.librarySearch}>
                <input
                  type="text"
                  className={styles.librarySearchInput}
                  placeholder="Search references..."
                  value={librarySearch}
                  onChange={(e) => setLibrarySearch(e.target.value)}
                />
                {LIBRARY_CATEGORIES.map(cat => (
                  <button
                    key={cat}
                    className={`${styles.libraryCategoryBtn} ${libraryCategory === cat ? styles.libraryCategoryBtnActive : ''}`}
                    onClick={() => setLibraryCategory(cat)}
                  >
                    {cat}
                  </button>
                ))}
              </div>
              <div className={styles.libraryGrid}>
                {filteredLibrary.map(item => (
                  <div
                    key={item.name}
                    className={styles.libraryCard}
                    onClick={() => setXRefFile({ name: item.name })}
                    title={`Click to load as reference`}
                  >
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
