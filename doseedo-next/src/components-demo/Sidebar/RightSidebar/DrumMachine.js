import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useApp } from '../../../context/AppContext';
import styles from './DrumMachine.module.css';

/**
 * DrumMachine - Minimalist drum pad interface
 * Double-click knobs/sliders to enter automation mode
 * Slider values animate based on automation shapes during playback
 */
const DrumMachine = React.memo(() => {
  const { state, dispatch } = useApp();
  const [currentStep, setCurrentStep] = useState(0);
  const [activePads, setActivePads] = useState([]);
  const lastBeatRef = useRef(-1);

  // Base control states (user-set values)
  const [baseCutoff, setBaseCutoff] = useState(75);
  const [baseFxAmount, setBaseFxAmount] = useState(30);
  const [baseVolume, setBaseVolume] = useState(80);
  const [baseAttack, setBaseAttack] = useState(10);
  const [baseDecay, setBaseDecay] = useState(40);
  const [baseTone, setBaseTone] = useState(50);

  const automationMode = state.drumAutomationMode;
  const drumAutomation = state.drumAutomation || {};
  const bpm = state.bpm || 120;
  const secondsPerBeat = 60 / bpm;
  const secondsPerBar = secondsPerBeat * 4;

  // Pattern - 16 steps (eighth notes)
  const pattern = {
    kick: [true, false, false, false, true, false, false, false, true, false, false, false, true, false, false, false],
    snare: [false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false],
    hihat: [true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false],
    clap: [false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false]
  };

  // Pads - pastel blue, purple, cyan shades
  const pads = [
    { id: 'kick', label: 'KICK', color: '#a78bfa' },      // pastel purple
    { id: 'snare', label: 'SNARE', color: '#7dd3fc' },    // pastel cyan
    { id: 'hihat', label: 'HH', color: '#c4b5fd' },       // light lavender
    { id: 'clap', label: 'CLAP', color: '#67e8f9' },      // bright cyan
    { id: 'tom1', label: 'TOM', color: '#818cf8' },       // pastel indigo
    { id: 'tom2', label: 'RIM', color: '#5eead4' },       // pastel teal
    { id: 'crash', label: 'CRS', color: '#c084fc' },      // pastel violet
    { id: 'ride', label: 'RIDE', color: '#93c5fd' }       // pastel blue
  ];

  // Form sections for automation lookup
  const formSections = [
    { id: 'intro', startBar: 0, endBar: 4 },
    { id: 'verse', startBar: 4, endBar: 12 },
    { id: 'chorus', startBar: 12, endBar: 22 },
    { id: 'outro', startBar: 22, endBar: 100 }
  ];

  // Get current section based on playhead
  const getCurrentSection = useCallback((playheadPos) => {
    const currentBar = Math.floor(playheadPos / secondsPerBar);
    for (const section of formSections) {
      if (currentBar >= section.startBar && currentBar < section.endBar) {
        return section;
      }
    }
    return null;
  }, [secondsPerBar]);

  // Calculate automated value based on shape and position within section
  const getAutomatedValue = useCallback((baseValue, parameter, playheadPos) => {
    const section = getCurrentSection(playheadPos);
    if (!section) return baseValue;

    const shape = drumAutomation[section.id]?.[parameter];
    if (!shape) return baseValue;

    // Calculate position within section (0-1)
    const sectionStartTime = section.startBar * secondsPerBar;
    const sectionEndTime = section.endBar * secondsPerBar;
    const sectionDuration = sectionEndTime - sectionStartTime;
    const positionInSection = (playheadPos - sectionStartTime) / sectionDuration;
    // Scale t so it reaches 1 at 95% of section (ensures max is hit before section ends)
    const t = Math.max(0, Math.min(1, positionInSection / 0.95));

    // Apply shape
    switch (shape) {
      case 'ramp-up':
        return Math.round(baseValue * t);
      case 'ramp-down':
        return Math.round(baseValue * (1 - t));
      case 'flat':
      default:
        return baseValue;
    }
  }, [getCurrentSection, drumAutomation, secondsPerBar]);

  // Get display values (automated if playing, base otherwise)
  const playheadPos = state.playheadPosition || 0;
  const isPlaying = state.isPlaying;

  const cutoff = isPlaying ? getAutomatedValue(baseCutoff, 'cutoff', playheadPos) : baseCutoff;
  const fxAmount = isPlaying ? getAutomatedValue(baseFxAmount, 'fx', playheadPos) : baseFxAmount;
  const volume = isPlaying ? getAutomatedValue(baseVolume, 'volume', playheadPos) : baseVolume;
  const attack = isPlaying ? getAutomatedValue(baseAttack, 'attack', playheadPos) : baseAttack;
  const decay = isPlaying ? getAutomatedValue(baseDecay, 'decay', playheadPos) : baseDecay;
  const tone = isPlaying ? getAutomatedValue(baseTone, 'tone', playheadPos) : baseTone;

  // Enter automation mode
  const enterAutomationMode = useCallback((parameter) => {
    dispatch({
      type: 'SET_DRUM_AUTOMATION_MODE',
      payload: { enabled: true, parameter }
    });
  }, [dispatch]);

  // Sync sequencer with playhead position (not internal timer)
  // Double time: each step is an eighth note
  const secondsPerEighth = secondsPerBeat / 2;

  useEffect(() => {
    if (!isPlaying) {
      setCurrentStep(0);
      setActivePads([]);
      lastBeatRef.current = -1;
      return;
    }

    // Calculate current step from playhead position (eighth notes, 16 steps)
    const eighthsElapsed = playheadPos / secondsPerEighth;
    const currentEighth = Math.floor(eighthsElapsed) % 16;

    // Only update if step changed
    if (currentEighth !== lastBeatRef.current) {
      lastBeatRef.current = currentEighth;
      setCurrentStep(currentEighth);

      // Trigger pads based on pattern
      const active = [];
      if (pattern.kick[currentEighth]) active.push('kick');
      if (pattern.snare[currentEighth]) active.push('snare');
      if (pattern.hihat[currentEighth]) active.push('hihat');
      if (pattern.clap[currentEighth]) active.push('clap');
      setActivePads(active);
    }
  }, [isPlaying, playheadPos, secondsPerEighth]);

  // Pad click
  const handlePadClick = useCallback((padId) => {
    setActivePads(prev =>
      prev.includes(padId) ? prev.filter(p => p !== padId) : [...prev, padId]
    );
    setTimeout(() => {
      if (!isPlaying) setActivePads([]);
    }, 150);
  }, [isPlaying]);

  return (
    <div className={styles.drumMachine}>
      {/* Step Sequencer - 16 steps (eighth notes) */}
      <div className={styles.sequencer}>
        {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].map(step => (
          <div
            key={step}
            className={`${styles.step} ${currentStep === step ? styles.stepActive : ''}`}
          />
        ))}
      </div>

      {/* Pads */}
      <div className={styles.pads}>
        {pads.map(pad => (
          <button
            key={pad.id}
            className={`${styles.pad} ${activePads.includes(pad.id) ? styles.padActive : ''}`}
            onClick={() => handlePadClick(pad.id)}
            style={{ '--pad-color': pad.color }}
          >
            {pad.label}
          </button>
        ))}
      </div>

      {/* Controls */}
      <div className={styles.controls}>
        <div className={styles.sliders}>
          <Slider label="CUT" param="cutoff" value={cutoff} baseValue={baseCutoff} onChange={setBaseCutoff} onDoubleClick={enterAutomationMode} isAutomating={automationMode?.parameter === 'cutoff'} />
          <Slider label="FX" param="fx" value={fxAmount} baseValue={baseFxAmount} onChange={setBaseFxAmount} onDoubleClick={enterAutomationMode} isAutomating={automationMode?.parameter === 'fx'} />
          <Slider label="VOL" param="volume" value={volume} baseValue={baseVolume} onChange={setBaseVolume} onDoubleClick={enterAutomationMode} isAutomating={automationMode?.parameter === 'volume'} />
        </div>
        <div className={styles.knobs}>
          <Knob label="ATK" param="attack" value={attack} baseValue={baseAttack} onChange={setBaseAttack} onDoubleClick={enterAutomationMode} isAutomating={automationMode?.parameter === 'attack'} />
          <Knob label="DCY" param="decay" value={decay} baseValue={baseDecay} onChange={setBaseDecay} onDoubleClick={enterAutomationMode} isAutomating={automationMode?.parameter === 'decay'} />
          <Knob label="TONE" param="tone" value={tone} baseValue={baseTone} onChange={setBaseTone} onDoubleClick={enterAutomationMode} isAutomating={automationMode?.parameter === 'tone'} />
        </div>
      </div>
    </div>
  );
});

// Simple Slider with double-click for automation
const Slider = ({ label, param, value, baseValue, onChange, onDoubleClick, isAutomating }) => {
  const trackRef = useRef(null);
  const dragging = useRef(false);

  const updateValue = useCallback((clientY) => {
    if (!trackRef.current) return;
    const rect = trackRef.current.getBoundingClientRect();
    const y = clientY - rect.top;
    const pct = Math.max(0, Math.min(100, 100 - (y / rect.height) * 100));
    onChange(Math.round(pct));
  }, [onChange]);

  const onPointerDown = (e) => {
    dragging.current = true;
    trackRef.current.setPointerCapture(e.pointerId);
    updateValue(e.clientY);
  };

  const onPointerMove = (e) => {
    if (dragging.current) updateValue(e.clientY);
  };

  const onPointerUp = (e) => {
    dragging.current = false;
    trackRef.current.releasePointerCapture(e.pointerId);
  };

  const handleDoubleClick = () => {
    onDoubleClick?.(param);
  };

  return (
    <div className={`${styles.slider} ${isAutomating ? styles.automating : ''}`}>
      <div
        ref={trackRef}
        className={styles.sliderTrack}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onDoubleClick={handleDoubleClick}
      >
        <div className={styles.sliderFill} style={{ height: `${value}%` }} />
      </div>
      <span className={styles.sliderLabel}>{label}</span>
    </div>
  );
};

// Simple Knob with double-click for automation
const Knob = ({ label, param, value, baseValue, onChange, onDoubleClick, isAutomating }) => {
  const knobRef = useRef(null);
  const dragging = useRef(false);
  const startY = useRef(0);
  const startVal = useRef(0);

  const onPointerDown = (e) => {
    dragging.current = true;
    startY.current = e.clientY;
    startVal.current = baseValue;
    knobRef.current.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e) => {
    if (!dragging.current) return;
    const delta = startY.current - e.clientY;
    const newVal = Math.max(0, Math.min(100, startVal.current + delta));
    onChange(Math.round(newVal));
  };

  const onPointerUp = (e) => {
    dragging.current = false;
    knobRef.current.releasePointerCapture(e.pointerId);
  };

  const handleDoubleClick = () => {
    onDoubleClick?.(param);
  };

  const rotation = (value / 100) * 270 - 135;

  return (
    <div className={`${styles.knob} ${isAutomating ? styles.automating : ''}`}>
      <div
        ref={knobRef}
        className={styles.knobDial}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onDoubleClick={handleDoubleClick}
      >
        <div
          className={styles.knobIndicator}
          style={{ transform: `rotate(${rotation}deg)` }}
        />
      </div>
      <span className={styles.knobLabel}>{label}</span>
    </div>
  );
};

DrumMachine.displayName = 'DrumMachine';
export default DrumMachine;
