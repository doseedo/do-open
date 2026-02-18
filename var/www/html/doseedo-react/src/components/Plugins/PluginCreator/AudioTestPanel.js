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
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);
  const animRef = useRef(null);

  // Analyser visualization
  useEffect(() => {
    if (!playing || !engine || !canvasRef.current) {
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
  }, [playing, engine]);

  const handlePlay = useCallback(() => {
    if (!engine) return;
    if (playing) {
      engine.stop();
      setPlaying(false);
    } else {
      engine.setLoop(loop);
      engine.setMasterVolume(volume);
      engine.play();
      setPlaying(true);
    }
  }, [engine, playing, loop, volume]);

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
    setAudioSource(toneId);
    setAudioFileName('');
    if (engine) {
      const wasPlaying = engine.playing;
      if (wasPlaying) engine.stop();
      engine.loadTestTone(toneId);
      if (wasPlaying) {
        engine.play();
        setPlaying(true);
      }
    }
  }, [engine]);

  const handleFileUpload = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file || !engine) return;
    setAudioFileName(file.name);
    setAudioSource('file');
    const wasPlaying = engine.playing;
    if (wasPlaying) engine.stop();
    await engine.loadAudioFile(file);
    if (wasPlaying) {
      engine.play();
      setPlaying(true);
    }
  }, [engine]);

  const hasDsp = dspConfig && dspConfig.dspChain && dspConfig.dspChain.length > 0;
  const dspParams = dspConfig?.parameters || [];

  return (
    <div className={styles.audioTestPanel}>
      {/* Transport */}
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

      {/* Audio Source */}
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
          className={`${styles.audioSourceBtn} ${audioSource === 'file' ? styles.audioSourceBtnActive : ''}`}
          onClick={() => fileInputRef.current?.click()}
          title="Upload audio file"
        >
          <i className="fa-solid fa-upload" /> {audioFileName || 'Upload'}
        </button>
        <input ref={fileInputRef} type="file" accept="audio/*" onChange={handleFileUpload} style={{ display: 'none' }} />
      </div>

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
    </div>
  );
};

export default AudioTestPanel;
