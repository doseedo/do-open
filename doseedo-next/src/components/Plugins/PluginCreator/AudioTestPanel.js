import React, { useState, useRef, useCallback, useEffect } from 'react';
import styles from './PluginCreator.module.css';

const TEST_TONES = [
  { id: 'drums', label: 'Drums', icon: 'fa-drum' },
  { id: 'sine', label: 'Sine 440Hz', icon: 'fa-wave-square' },
  { id: 'sweep', label: 'Sweep', icon: 'fa-chart-line' },
  { id: 'noise', label: 'Noise', icon: 'fa-signal' },
];

const AudioTestPanel = ({ engine, dspConfig, paramValues, paramMapping, components, onParamChange }) => {
  const [playing, setPlaying] = useState(false);
  const [volume, setVolume] = useState(0.8);
  const [loop, setLoop] = useState(true);
  const [audioSource, setAudioSource] = useState('drums');
  const [audioFileName, setAudioFileName] = useState('');
  const [showMapping, setShowMapping] = useState(false);
  const [micActive, setMicActive] = useState(false);
  const [micError, setMicError] = useState('');
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);
  const animRef = useRef(null);

  const isInstrument = engine?.isInstrument;

  // Analyser visualization — run for both instruments (when notes are active) and effects
  useEffect(() => {
    const shouldAnimate = playing || micActive || (isInstrument && engine?.playing);
    if (!shouldAnimate || !engine || !canvasRef.current) {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      return;
    }
    const canvas = canvasRef.current;
    const ctxC = canvas.getContext('2d');
    const draw = () => {
      const data = engine.getAnalyserData();
      if (!data) { animRef.current = requestAnimationFrame(draw); return; }

      ctxC.clearRect(0, 0, canvas.width, canvas.height);
      const barW = canvas.width / data.length;
      for (let i = 0; i < data.length; i++) {
        const h = (data[i] / 255) * canvas.height;
        const hue = 250 + (i / data.length) * 60;
        ctxC.fillStyle = `hsla(${hue}, 70%, 60%, 0.8)`;
        ctxC.fillRect(i * barW, canvas.height - h, barW - 1, h);
      }
      animRef.current = requestAnimationFrame(draw);
    };
    animRef.current = requestAnimationFrame(draw);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [playing, micActive, isInstrument, engine]);

  const handlePlay = useCallback(() => {
    if (!engine) return;
    if (micActive) {
      engine.stop();
      setMicActive(false);
    }
    if (playing) {
      engine.stop();
      setPlaying(false);
    } else {
      engine.setLoop(loop);
      engine.setMasterVolume(volume);
      engine.play();
      setPlaying(true);
    }
  }, [engine, playing, micActive, loop, volume]);

  const handleVolumeChange = useCallback((e) => {
    const v = parseFloat(e.target.value);
    setVolume(v);
    if (engine) engine.setMasterVolume(v);
  }, [engine]);

  const handleLoopToggle = useCallback(() => {
    const next = !loop;
    setLoop(next);
    if (engine) engine.setLoop(next);
  }, [engine, loop]);

  const handleToneSelect = useCallback((toneId) => {
    if (micActive && engine) { engine.stopMicInput(); setMicActive(false); }
    setAudioSource(toneId);
    setAudioFileName('');
    setMicError('');
    if (engine) {
      const wasPlaying = engine.playing;
      if (wasPlaying) engine.stop();
      engine.loadTestTone(toneId);
      if (wasPlaying) {
        engine.play();
        setPlaying(true);
      }
    }
  }, [engine, micActive]);

  const handleFileUpload = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file || !engine) return;
    if (micActive) { engine.stopMicInput(); setMicActive(false); }
    setAudioFileName(file.name);
    setAudioSource('file');
    setMicError('');
    const wasPlaying = engine.playing;
    if (wasPlaying) engine.stop();
    await engine.loadAudioFile(file);
    if (wasPlaying) {
      engine.play();
      setPlaying(true);
    }
  }, [engine, micActive]);

  const handleMicToggle = useCallback(async () => {
    if (!engine) return;
    if (micActive) {
      engine.stop();
      setMicActive(false);
      setPlaying(false);
      setMicError('');
    } else {
      // Stop any current playback
      if (playing) { engine.stop(); setPlaying(false); }
      try {
        setMicError('');
        await engine.startMicInput();
        setMicActive(true);
        setAudioSource('mic');
      } catch (err) {
        setMicError('Mic access denied');
        console.error('Mic error:', err);
      }
    }
  }, [engine, micActive, playing]);

  const hasDsp = dspConfig && dspConfig.dspChain && dspConfig.dspChain.length > 0;
  const dspParams = dspConfig?.parameters || [];

  return (
    <div className={styles.audioTestPanel}>
      {/* For instruments: simplified controls */}
      {isInstrument ? (
        <>
          <div className={styles.audioTransport}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, flex: 1 }}>
              <i className="fa-solid fa-piano-keyboard" style={{ color: '#667eea', fontSize: 12 }} />
              <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.6)' }}>Instrument Mode</span>
            </div>
            <div className={styles.audioVolumeWrap}>
              <i className="fa-solid fa-volume-high" style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)' }} />
              <input
                type="range"
                min={0} max={1} step={0.01}
                value={volume}
                onChange={handleVolumeChange}
                className={styles.audioVolumeSlider}
              />
            </div>
            <canvas ref={canvasRef} className={styles.audioAnalyser} width={120} height={28} />
          </div>
        </>
      ) : (
        <>
          {/* Transport for effects */}
          <div className={styles.audioTransport}>
            <button
              className={`${styles.audioPlayBtn} ${playing ? styles.audioPlayBtnActive : ''}`}
              onClick={handlePlay}
              disabled={!engine}
              title={playing ? 'Stop' : 'Play'}
            >
              <i className={`fa-solid ${playing ? 'fa-stop' : 'fa-play'}`} />
            </button>
            <button
              className={`${styles.audioLoopBtn} ${loop ? styles.audioLoopBtnActive : ''}`}
              onClick={handleLoopToggle}
              title="Loop"
            >
              <i className="fa-solid fa-repeat" />
            </button>
            <div className={styles.audioVolumeWrap}>
              <i className="fa-solid fa-volume-high" style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)' }} />
              <input
                type="range"
                min={0} max={1} step={0.01}
                value={volume}
                onChange={handleVolumeChange}
                className={styles.audioVolumeSlider}
              />
            </div>
            <canvas ref={canvasRef} className={styles.audioAnalyser} width={120} height={28} />
          </div>

          {/* Audio Source — effects only */}
          <div className={styles.audioSourceRow}>
            <span className={styles.audioSourceLabel}>Source:</span>
            {TEST_TONES.map(t => (
              <button
                key={t.id}
                className={`${styles.audioSourceBtn} ${audioSource === t.id ? styles.audioSourceBtnActive : ''}`}
                onClick={() => handleToneSelect(t.id)}
                title={t.label}
              >
                <i className={`fa-solid ${t.icon}`} /> {t.label}
              </button>
            ))}
            <button
              className={`${styles.audioSourceBtn} ${audioSource === 'mic' ? styles.audioSourceBtnActive : ''}`}
              onClick={handleMicToggle}
              title={micActive ? 'Stop microphone' : 'Use microphone input'}
              style={micActive ? { color: '#ef4444', borderColor: 'rgba(239,68,68,0.4)' } : undefined}
            >
              <i className={`fa-solid ${micActive ? 'fa-microphone-slash' : 'fa-microphone'}`} />
              {micActive ? 'Stop Mic' : 'Mic'}
            </button>
            <button
              className={`${styles.audioSourceBtn} ${audioSource === 'file' ? styles.audioSourceBtnActive : ''}`}
              onClick={() => fileInputRef.current?.click()}
              title="Upload audio file"
            >
              <i className="fa-solid fa-upload" /> {audioFileName || 'Upload'}
            </button>
            <input ref={fileInputRef} type="file" accept="audio/*" onChange={handleFileUpload} style={{ display: 'none' }} />
          </div>
          {micError && (
            <div style={{ padding: '3px 8px', fontSize: 10, color: '#ef4444', background: 'rgba(239,68,68,0.08)' }}>
              <i className="fa-solid fa-triangle-exclamation" /> {micError}
            </div>
          )}
        </>
      )}

      {/* DSP Status */}
      {hasDsp ? (
        <div className={styles.audioDspStatus}>
          <div className={styles.audioDspChain}>
            <i className="fa-solid fa-microchip" style={{ color: '#00e5ff' }} />
            <span>DSP: </span>
            {dspConfig.dspChain.map((n, i) => (
              <span key={i} className={styles.audioDspNode}>
                {n.type}{i < dspConfig.dspChain.length - 1 ? ' → ' : ''}
              </span>
            ))}
          </div>
          {dspParams.length > 0 && (
            <button
              className={styles.audioMappingToggle}
              onClick={() => setShowMapping(!showMapping)}
            >
              <i className={`fa-solid fa-${showMapping ? 'chevron-up' : 'chevron-down'}`} />
              {dspParams.length} params
            </button>
          )}
        </div>
      ) : (
        <div className={styles.audioDspEmpty}>
          <i className="fa-solid fa-circle-info" /> No DSP config — audio plays dry. Use the Backend Coder tab to design a DSP chain.
        </div>
      )}

      {/* Parameter Mapping (expanded) */}
      {showMapping && dspParams.length > 0 && (
        <div className={styles.audioParamList}>
          {dspParams.map(p => {
            const mapped = paramMapping?.[p.id];
            const comp = mapped ? components.find(c => c.id === mapped) : null;
            const val = paramValues?.[mapped] ?? 0.5;
            const actual = p.min != null ? (p.min + (p.max - p.min) * Math.pow(val, 1 / (p.skew || 1))).toFixed(1) : (val * 100).toFixed(0);
            return (
              <div key={p.id} className={styles.audioParamRow}>
                <span className={styles.audioParamName}>{p.name || p.id}</span>
                <span className={styles.audioParamValue}>
                  {actual}{p.unit ? ` ${p.unit}` : ''}
                </span>
                <span className={styles.audioParamBound}>
                  {comp ? (
                    <><i className="fa-solid fa-link" /> {comp.label}</>
                  ) : (
                    <span style={{ opacity: 0.4 }}>unbound</span>
                  )}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {/* Keyboard hint (instrument mode — full keyboard is on the plugin canvas) */}
      {isInstrument && (
        <div style={{
          padding: '4px 8px', background: 'rgba(0,0,0,0.15)',
          borderTop: '1px solid rgba(255,255,255,0.06)',
          fontSize: 9, color: 'rgba(255,255,255,0.35)',
          display: 'flex', alignItems: 'center', gap: 4,
        }}>
          <i className="fa-solid fa-keyboard" />
          <span>Press A-L keys or click the keyboard below the plugin to play</span>
        </div>
      )}
    </div>
  );
};

export default AudioTestPanel;
