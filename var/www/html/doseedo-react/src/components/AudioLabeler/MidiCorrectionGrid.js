import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { parseMIDI } from '../../utils/midiParser';
import midiPlayer from '../../utils/midiPlayer';

const API_BASE = 'https://doseedo.com/api';

// GM Drum Map for channel 10 (MIDI notes 35-81)
const DRUM_MAP = {
  35: 'Acoustic Bass Drum', 36: 'Bass Drum 1', 37: 'Side Stick', 38: 'Acoustic Snare',
  39: 'Hand Clap', 40: 'Electric Snare', 41: 'Low Floor Tom', 42: 'Closed Hi-Hat',
  43: 'High Floor Tom', 44: 'Pedal Hi-Hat', 45: 'Low Tom', 46: 'Open Hi-Hat',
  47: 'Low-Mid Tom', 48: 'Hi-Mid Tom', 49: 'Crash Cymbal 1', 50: 'High Tom',
  51: 'Ride Cymbal 1', 52: 'Chinese Cymbal', 53: 'Ride Bell', 54: 'Tambourine',
  55: 'Splash Cymbal', 56: 'Cowbell', 57: 'Crash Cymbal 2', 59: 'Ride Cymbal 2',
  60: 'Hi Bongo', 61: 'Low Bongo', 62: 'Mute Hi Conga', 63: 'Open Hi Conga',
  64: 'Low Conga', 65: 'High Timbale', 66: 'Low Timbale', 69: 'Cabasa',
  70: 'Maracas', 75: 'Claves', 76: 'Hi Wood Block', 77: 'Low Wood Block',
};

// Common drum notes to always show in drum mode
const COMMON_DRUM_NOTES = [36, 38, 40, 42, 44, 46, 41, 43, 45, 47, 48, 50, 49, 51, 57];

// Map drum stem names to GM MIDI note rows for onset display
const ONSET_STEM_NOTES = {
  kick: 36,     // Bass Drum 1
  snare: 38,    // Acoustic Snare
  toms: 47,     // Low-Mid Tom
  hh: 42,       // Closed Hi-Hat
  cymbals: 49,  // Crash Cymbal 1
};

const ONSET_STEM_COLORS = {
  kick:    'rgba(255, 100, 60, 0.85)',
  snare:   'rgba(60, 180, 255, 0.85)',
  toms:    'rgba(255, 200, 50, 0.85)',
  hh:      'rgba(80, 220, 120, 0.85)',
  cymbals: 'rgba(200, 100, 255, 0.85)',
};

const MidiCorrectionGrid = ({ audioPath, isDrumMode, onSave, currentTime = 0, duration = 0, isPlaying = false, onsets = null, isOnsetMode = false }) => {
  const canvasRef = useRef(null);
  const canvasWrapperRef = useRef(null);
  const [notes, setNotes] = useState([]);
  const [originalNotes, setOriginalNotes] = useState([]);
  const [correctedNotes, setCorrectedNotes] = useState(null); // From corrections.json
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [midiPath, setMidiPath] = useState(null);
  const [hasEdits, setHasEdits] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null);
  const [playerInitialized, setPlayerInitialized] = useState(false);

  // Refined extraction state
  const [refinedNotes, setRefinedNotes] = useState(null);
  const [refinedInfo, setRefinedInfo] = useState(null); // {strategy, stats, comparison, data_available}
  const [isRefining, setIsRefining] = useState(false);
  const [showRefined, setShowRefined] = useState(true);
  const [refinedError, setRefinedError] = useState(null);

  // Interaction state
  const [hoveredNote, setHoveredNote] = useState(null);
  const [selectedNotes, setSelectedNotes] = useState([]);
  const [dragState, setDragState] = useState(null);
  const [toolMode, setToolMode] = useState('select'); // 'select' or 'draw'
  const [zoomX, setZoomX] = useState(1.0);
  const [zoomY, setZoomY] = useState(1.0);
  const [showOriginal, setShowOriginal] = useState(true); // Show ghost of original MIDI
  const [showCorrected, setShowCorrected] = useState(true); // Show previous corrections overlay

  // Initialize MIDI player
  useEffect(() => {
    const init = async () => {
      try {
        await midiPlayer.initialize();
        setPlayerInitialized(true);
      } catch (e) { console.error('MIDI player init failed:', e); }
    };
    init();
    return () => midiPlayer.stopAll();
  }, []);

  // Derive MIDI path and load when audioPath changes
  useEffect(() => {
    if (!audioPath) {
      setNotes([]);
      setOriginalNotes([]);
      setCorrectedNotes(null);
      setMidiPath(null);
      setHasEdits(false);
      setRefinedNotes(null);
      setRefinedInfo(null);
      setRefinedError(null);
      return;
    }

    // In onset mode, skip MIDI loading entirely
    if (isOnsetMode) {
      setNotes([]);
      setOriginalNotes([]);
      setCorrectedNotes(null);
      setMidiPath(null);
      setHasEdits(false);
      setIsLoading(false);
      setError(null);
      setRefinedNotes(null);
      setRefinedInfo(null);
      setRefinedError(null);
      return;
    }

    const loadMidi = async () => {
      setIsLoading(true);
      setError(null);
      setHasEdits(false);
      setSelectedNotes([]);
      setRefinedNotes(null);
      setRefinedInfo(null);
      setRefinedError(null);

      try {
        // 1. Check for existing corrections first
        const corrRes = await fetch(`${API_BASE}/midi-corrections?path=${encodeURIComponent(audioPath)}`);
        const corrData = await corrRes.json();

        // 2. Get MIDI path from audio path
        const pathRes = await fetch(`${API_BASE}/midi-path?audio_path=${encodeURIComponent(audioPath)}`);
        const pathData = await pathRes.json();

        if (!pathData.exists) {
          setError('No BasicPitch MIDI available for this entry');
          setMidiPath(null);
          setNotes([]);
          setOriginalNotes([]);

          // But if we have corrections, show those
          if (corrData.correction) {
            setCorrectedNotes(corrData.correction);
            setNotes(corrData.correction.notes || []);
            setOriginalNotes(corrData.correction.original_notes || []);
          }
          setIsLoading(false);
          return;
        }

        setMidiPath(pathData.midi_path);

        // 3. Fetch and parse the MIDI file
        const midiRes = await fetch(`${API_BASE}/midi?path=${encodeURIComponent(pathData.midi_path)}`);
        if (!midiRes.ok) throw new Error(`MIDI fetch failed: ${midiRes.status}`);
        const midiBuffer = await midiRes.arrayBuffer();
        const parsed = parseMIDI(midiBuffer);

        const midiNotes = (parsed.notes || []).map(n => ({
          note: n.note,
          time: n.time,
          duration: n.duration,
          velocity: n.velocity || 100,
        }));

        setOriginalNotes(midiNotes);

        // If we have corrections, use corrected notes; otherwise use original
        if (corrData.correction && corrData.correction.notes) {
          setCorrectedNotes(corrData.correction);
          setNotes(corrData.correction.notes);
        } else {
          setCorrectedNotes(null);
          setNotes(JSON.parse(JSON.stringify(midiNotes))); // Deep copy
        }
      } catch (e) {
        console.error('Failed to load MIDI:', e);
        setError(e.message);
      }

      setIsLoading(false);
    };

    loadMidi();
  }, [audioPath, isOnsetMode]);

  // Determine if drum mode based on group or explicit prop
  const effectiveDrumMode = isDrumMode;

  // Note range calculation
  const noteRange = useMemo(() => {
    // In onset mode, use fixed range covering all onset stem rows
    if (isOnsetMode && onsets) {
      return { min: 34, max: 51 };
    }

    const allNotes = [...notes, ...originalNotes, ...(refinedNotes || [])];
    if (allNotes.length === 0) {
      return effectiveDrumMode ? { min: 35, max: 57 } : { min: 48, max: 84 };
    }

    const noteNumbers = allNotes.map(n => n.note);
    let min = Math.min(...noteNumbers);
    let max = Math.max(...noteNumbers);

    // Pad
    min = Math.max(0, min - 2);
    max = Math.min(127, max + 2);

    // Ensure minimum range
    const minRange = effectiveDrumMode ? 15 : 24;
    if (max - min + 1 < minRange) {
      const expand = minRange - (max - min + 1);
      min = Math.max(0, min - Math.floor(expand / 2));
      max = Math.min(127, max + Math.ceil(expand / 2));
    }

    return { min, max };
  }, [notes, originalNotes, refinedNotes, effectiveDrumMode, isOnsetMode, onsets]);

  // Config
  const config = useMemo(() => ({
    minNote: noteRange.min,
    maxNote: noteRange.max,
    gridWidth: 60 * zoomX, // pixels per second
    noteHeight: (effectiveDrumMode ? 24 : 16) * zoomY,
    leftMargin: effectiveDrumMode ? 140 : 60,
    topMargin: 25,
    bottomMargin: 10,
    resizeHandleWidth: 6,
  }), [zoomX, zoomY, noteRange, effectiveDrumMode]);

  const totalNotes = config.maxNote - config.minNote + 1;
  const gridDuration = Math.max(duration || 10, notes.length > 0 ? Math.max(...notes.map(n => n.time + n.duration)) + 2 : 10);
  const canvasWidth = config.leftMargin + (config.gridWidth * gridDuration) + 20;
  const canvasHeight = config.topMargin + (config.noteHeight * totalNotes) + config.bottomMargin;

  // Note name helpers
  const getNoteNameFromMidi = (midiNote) => {
    if (effectiveDrumMode && DRUM_MAP[midiNote]) {
      return DRUM_MAP[midiNote];
    }
    const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const octave = Math.floor(midiNote / 12) - 1;
    return `${noteNames[midiNote % 12]}${octave}`;
  };

  const isBlackKey = (midiNote) => [1, 3, 6, 8, 10].includes(midiNote % 12);

  // Draw the grid
  const drawGrid = useCallback((ctx) => {
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Piano keys / drum labels on the left
    for (let i = 0; i < totalNotes; i++) {
      const midiNote = config.maxNote - i;
      const y = config.topMargin + (i * config.noteHeight);

      if (effectiveDrumMode) {
        // Drum mode: labeled rows with alternating bg
        const isDrumNote = DRUM_MAP[midiNote];
        ctx.fillStyle = i % 2 === 0 ? '#14142e' : '#0f0f20';
        ctx.fillRect(0, y, config.leftMargin, config.noteHeight);

        // Label
        ctx.fillStyle = isDrumNote ? '#c5cae9' : '#555';
        ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        const label = isDrumNote ? DRUM_MAP[midiNote] : `N${midiNote}`;
        ctx.fillText(label, config.leftMargin - 6, y + config.noteHeight / 2);
      } else {
        // Piano mode
        const isBlack = isBlackKey(midiNote);
        ctx.fillStyle = isBlack ? '#1a1a2e' : '#e8e8f0';
        ctx.fillRect(0, y, config.leftMargin, config.noteHeight);
        ctx.strokeStyle = '#0a0a0f';
        ctx.lineWidth = 1;
        ctx.strokeRect(0, y, config.leftMargin, config.noteHeight);

        if (midiNote % 12 === 0) {
          ctx.fillStyle = '#888';
          ctx.font = '11px -apple-system, sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(getNoteNameFromMidi(midiNote), config.leftMargin / 2, y + config.noteHeight / 2);
        }
      }
    }

    // Horizontal grid lines
    for (let i = 0; i <= totalNotes; i++) {
      const y = config.topMargin + (i * config.noteHeight);
      const midiNote = config.maxNote - i;
      ctx.beginPath();
      ctx.moveTo(config.leftMargin, y);
      ctx.lineTo(canvasWidth, y);
      ctx.strokeStyle = (midiNote % 12 === 0 && !effectiveDrumMode) ? '#4a4a6e' : '#1e1e30';
      ctx.lineWidth = (midiNote % 12 === 0 && !effectiveDrumMode) ? 2 : 1;
      ctx.stroke();
    }

    // Vertical grid lines (every second, with beat subdivisions)
    for (let t = 0; t <= gridDuration; t += 0.5) {
      const x = config.leftMargin + (t * config.gridWidth);
      const isWholeSecond = t === Math.floor(t);
      ctx.beginPath();
      ctx.moveTo(x, config.topMargin);
      ctx.lineTo(x, canvasHeight - config.bottomMargin);
      ctx.strokeStyle = isWholeSecond ? '#3a3a5e' : '#1a1a2e';
      ctx.lineWidth = isWholeSecond ? 1.5 : 0.5;
      ctx.stroke();

      // Time labels every second
      if (isWholeSecond && t > 0) {
        ctx.fillStyle = '#666';
        ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${t}s`, x, config.topMargin - 8);
      }
    }
  }, [canvasWidth, canvasHeight, config, totalNotes, gridDuration, effectiveDrumMode]);

  // Draw notes (editable layer)
  const drawNotes = useCallback((ctx) => {
    notes.forEach((note, index) => {
      if (!Number.isFinite(note.time) || !Number.isFinite(note.duration) || !Number.isFinite(note.note)) return;

      const x = config.leftMargin + (note.time * config.gridWidth);
      const y = config.topMargin + ((config.maxNote - note.note) * config.noteHeight);
      const width = Math.max(2, note.duration * config.gridWidth);
      const height = config.noteHeight - 2;

      if (width <= 0 || !Number.isFinite(x) || !Number.isFinite(y)) return;

      const isHovered = hoveredNote === index;
      const isSelected = selectedNotes.includes(index);

      // Note color
      if (isSelected) {
        ctx.fillStyle = '#88a3f7';
      } else if (isHovered) {
        ctx.fillStyle = '#7090e0';
      } else {
        ctx.fillStyle = effectiveDrumMode ? '#e07050' : '#667eea';
      }
      ctx.fillRect(x + 1, y + 1, width - 2, height);

      // Resize handles when hovered/selected
      if (isHovered || isSelected) {
        ctx.fillStyle = '#c5d0ff';
        ctx.fillRect(x + 1, y + 1, 3, height);
        ctx.fillRect(x + width - 4, y + 1, 3, height);
      }
    });
  }, [notes, config, hoveredNote, selectedNotes, effectiveDrumMode]);

  // Draw ghost notes (original MIDI in transparent gray)
  const drawGhostNotes = useCallback((ctx) => {
    if (!showOriginal || originalNotes.length === 0) return;

    originalNotes.forEach((note) => {
      if (!Number.isFinite(note.time) || !Number.isFinite(note.duration) || !Number.isFinite(note.note)) return;
      const x = config.leftMargin + (note.time * config.gridWidth);
      const y = config.topMargin + ((config.maxNote - note.note) * config.noteHeight);
      const width = Math.max(2, note.duration * config.gridWidth);
      const height = config.noteHeight - 2;
      if (width <= 0 || !Number.isFinite(x) || !Number.isFinite(y)) return;

      ctx.fillStyle = 'rgba(120, 120, 120, 0.15)';
      ctx.fillRect(x + 1, y + 1, width - 2, height);
      ctx.strokeStyle = 'rgba(100, 100, 100, 0.2)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 1, y + 1, width - 2, height);
    });
  }, [originalNotes, config, showOriginal]);

  // Draw corrected notes overlay (from previous corrections - green tint)
  const drawCorrectedOverlay = useCallback((ctx) => {
    if (!showCorrected || !correctedNotes || !correctedNotes.notes) return;

    correctedNotes.notes.forEach((note) => {
      if (!Number.isFinite(note.time) || !Number.isFinite(note.duration) || !Number.isFinite(note.note)) return;
      const x = config.leftMargin + (note.time * config.gridWidth);
      const y = config.topMargin + ((config.maxNote - note.note) * config.noteHeight);
      const width = Math.max(2, note.duration * config.gridWidth);
      const height = config.noteHeight - 2;
      if (width <= 0 || !Number.isFinite(x) || !Number.isFinite(y)) return;

      ctx.strokeStyle = 'rgba(34, 197, 94, 0.5)';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 2]);
      ctx.strokeRect(x + 1, y + 1, width - 2, height);
      ctx.setLineDash([]);
    });
  }, [correctedNotes, config, showCorrected]);

  // Draw refined notes overlay (amber)
  const drawRefinedOverlay = useCallback((ctx) => {
    if (!showRefined || !refinedNotes || refinedNotes.length === 0) return;

    refinedNotes.forEach((note) => {
      if (!Number.isFinite(note.time) || !Number.isFinite(note.duration) || !Number.isFinite(note.note)) return;
      const x = config.leftMargin + (note.time * config.gridWidth);
      const y = config.topMargin + ((config.maxNote - note.note) * config.noteHeight);
      const width = Math.max(2, note.duration * config.gridWidth);
      const height = config.noteHeight - 2;
      if (width <= 0 || !Number.isFinite(x) || !Number.isFinite(y)) return;

      ctx.fillStyle = 'rgba(245, 158, 11, 0.25)';
      ctx.fillRect(x + 1, y + 1, width - 2, height);
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.7)';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.strokeRect(x + 1, y + 1, width - 2, height);
      ctx.setLineDash([]);
    });
  }, [refinedNotes, config, showRefined]);

  // Draw onset ticks (drum onset review mode)
  const drawOnsets = useCallback((ctx) => {
    if (!onsets || !isOnsetMode) return;

    const totalNotes = config.maxNote - config.minNote + 1;
    const TICK_WIDTH = 4;

    Object.entries(ONSET_STEM_NOTES).forEach(([stem, midiNote]) => {
      const stemData = onsets[stem];
      if (!stemData || !stemData.times || stemData.times.length === 0) return;

      const color = ONSET_STEM_COLORS[stem];
      const noteIndex = config.maxNote - midiNote;
      if (noteIndex < 0 || noteIndex >= totalNotes) return;

      const y = config.topMargin + (noteIndex * config.noteHeight);
      const height = config.noteHeight - 2;

      ctx.fillStyle = color;

      for (let i = 0; i < stemData.times.length; i++) {
        const t = stemData.times[i];
        const x = config.leftMargin + (t * config.gridWidth);
        if (x < config.leftMargin) continue;
        ctx.fillRect(x, y + 1, TICK_WIDTH, height);
      }
    });
  }, [onsets, isOnsetMode, config]);

  // Draw playhead
  const drawPlayhead = useCallback((ctx) => {
    if (!currentTime || currentTime <= 0) return;
    const x = config.leftMargin + (currentTime * config.gridWidth);
    if (x < config.leftMargin || x > canvasWidth) return;

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, config.topMargin);
    ctx.lineTo(x, canvasHeight - config.bottomMargin);
    ctx.stroke();

    // Triangle at top
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.moveTo(x, config.topMargin - 6);
    ctx.lineTo(x - 5, config.topMargin);
    ctx.lineTo(x + 5, config.topMargin);
    ctx.closePath();
    ctx.fill();
  }, [currentTime, config, canvasWidth, canvasHeight]);

  // Render canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    drawGrid(ctx);
    drawGhostNotes(ctx);
    drawCorrectedOverlay(ctx);
    drawRefinedOverlay(ctx);
    drawNotes(ctx);
    drawOnsets(ctx);
    drawPlayhead(ctx);
  }, [drawGrid, drawGhostNotes, drawCorrectedOverlay, drawRefinedOverlay, drawNotes, drawOnsets, drawPlayhead]);

  // Auto-scroll to follow playhead
  useEffect(() => {
    if (!isPlaying || !canvasWrapperRef.current) return;
    const wrapper = canvasWrapperRef.current;
    const playheadX = config.leftMargin + (currentTime * config.gridWidth);
    const rightThreshold = wrapper.scrollLeft + wrapper.clientWidth * 0.8;
    if (playheadX > rightThreshold) {
      wrapper.scrollLeft = playheadX - wrapper.clientWidth * 0.5;
    }
  }, [currentTime, isPlaying, config]);

  // Mouse helpers
  const getMousePos = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }, []);

  const findNoteAtPos = useCallback((x, y) => {
    for (let i = notes.length - 1; i >= 0; i--) {
      const note = notes[i];
      const noteX = config.leftMargin + (note.time * config.gridWidth);
      const noteY = config.topMargin + ((config.maxNote - note.note) * config.noteHeight);
      const noteWidth = Math.max(2, note.duration * config.gridWidth);
      const noteHeight = config.noteHeight - 2;
      if (x >= noteX && x <= noteX + noteWidth && y >= noteY && y <= noteY + noteHeight) {
        const isLeftHandle = x <= noteX + config.resizeHandleWidth;
        const isRightHandle = x >= noteX + noteWidth - config.resizeHandleWidth;
        return { index: i, isResizeHandle: isLeftHandle || isRightHandle, isLeftHandle };
      }
    }
    return null;
  }, [notes, config]);

  // Mouse down
  const handleMouseDown = useCallback(async (e) => {
    const pos = getMousePos(e);
    const result = findNoteAtPos(pos.x, pos.y);

    if (toolMode === 'select') {
      if (result) {
        // Play note preview
        if (playerInitialized) {
          await midiPlayer.resume();
          midiPlayer.playNote(notes[result.index].note, 0.7, 0.3);
        }

        if (result.isResizeHandle) {
          setDragState({
            type: 'resize',
            noteIndex: result.index,
            startX: pos.x,
            startDuration: notes[result.index].duration,
            startTime: notes[result.index].time,
            isLeftHandle: result.isLeftHandle,
          });
        } else {
          setSelectedNotes([result.index]);
          setDragState({
            type: 'move',
            noteIndex: result.index,
            startX: pos.x,
            startY: pos.y,
            startTime: notes[result.index].time,
            startNote: notes[result.index].note,
          });
        }
      } else {
        setSelectedNotes([]);
      }
    } else if (toolMode === 'draw') {
      if (result) {
        // Double-click on existing note = delete
        return;
      }
      // Add new note
      if (pos.x >= config.leftMargin && pos.y >= config.topMargin && pos.y <= canvasHeight - config.bottomMargin) {
        const time = (pos.x - config.leftMargin) / config.gridWidth;
        const noteIndex = Math.floor((pos.y - config.topMargin) / config.noteHeight);
        const midiNote = config.maxNote - noteIndex;

        const newNote = {
          note: midiNote,
          time: Math.max(0, time),
          duration: effectiveDrumMode ? 0.1 : 0.25,
          velocity: 100,
        };
        setNotes(prev => [...prev, newNote]);
        setHasEdits(true);

        if (playerInitialized) {
          await midiPlayer.resume();
          midiPlayer.playNote(midiNote, 0.7, 0.3);
        }
      }
    }
    e.preventDefault();
  }, [notes, config, canvasHeight, toolMode, getMousePos, findNoteAtPos, playerInitialized, effectiveDrumMode]);

  // Mouse move
  const handleMouseMove = useCallback((e) => {
    const pos = getMousePos(e);

    if (dragState) {
      if (dragState.type === 'move') {
        const deltaX = pos.x - dragState.startX;
        const deltaY = pos.y - dragState.startY;
        const deltaTime = deltaX / config.gridWidth;
        const deltaNoteIndex = Math.round(deltaY / config.noteHeight);

        setNotes(prev => {
          const newNotes = [...prev];
          newNotes[dragState.noteIndex] = {
            ...newNotes[dragState.noteIndex],
            time: Math.max(0, dragState.startTime + deltaTime),
            note: Math.max(config.minNote, Math.min(config.maxNote, dragState.startNote - deltaNoteIndex)),
          };
          return newNotes;
        });
        setHasEdits(true);
      } else if (dragState.type === 'resize') {
        const deltaX = pos.x - dragState.startX;
        const deltaDuration = deltaX / config.gridWidth;

        if (dragState.isLeftHandle) {
          const newTime = Math.max(0, dragState.startTime + deltaDuration);
          const newDuration = Math.max(0.02, dragState.startDuration - deltaDuration);
          setNotes(prev => {
            const newNotes = [...prev];
            newNotes[dragState.noteIndex] = {
              ...newNotes[dragState.noteIndex],
              time: newTime,
              duration: newDuration,
            };
            return newNotes;
          });
        } else {
          const newDuration = Math.max(0.02, dragState.startDuration + deltaDuration);
          setNotes(prev => {
            const newNotes = [...prev];
            newNotes[dragState.noteIndex] = {
              ...newNotes[dragState.noteIndex],
              duration: newDuration,
            };
            return newNotes;
          });
        }
        setHasEdits(true);
      }
    } else {
      // Hover
      const result = findNoteAtPos(pos.x, pos.y);
      setHoveredNote(result ? result.index : null);

      const canvas = canvasRef.current;
      if (canvas) {
        if (toolMode === 'draw') {
          canvas.style.cursor = result ? 'pointer' : 'crosshair';
        } else {
          canvas.style.cursor = result ? (result.isResizeHandle ? 'ew-resize' : 'move') : 'default';
        }
      }
    }
  }, [dragState, config, getMousePos, findNoteAtPos, toolMode]);

  // Mouse up
  const handleMouseUp = useCallback(() => {
    setDragState(null);
  }, []);

  // Double click to delete
  const handleDoubleClick = useCallback((e) => {
    const pos = getMousePos(e);
    const result = findNoteAtPos(pos.x, pos.y);
    if (result) {
      setNotes(prev => prev.filter((_, i) => i !== result.index));
      setSelectedNotes([]);
      setHasEdits(true);
    }
  }, [getMousePos, findNoteAtPos]);

  // Delete key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedNotes.length > 0) {
        e.preventDefault();
        e.stopImmediatePropagation();
        setNotes(prev => prev.filter((_, i) => !selectedNotes.includes(i)));
        setSelectedNotes([]);
        setHasEdits(true);
      }
    };
    window.addEventListener('keydown', handleKeyDown, { capture: true });
    return () => window.removeEventListener('keydown', handleKeyDown, { capture: true });
  }, [selectedNotes]);

  // Zoom
  const handleZoomIn = () => setZoomX(prev => Math.min(prev + 0.25, 5.0));
  const handleZoomOut = () => setZoomX(prev => Math.max(prev - 0.25, 0.5));

  // Save corrections
  const handleSave = async () => {
    if (!audioPath) return;
    setSaveStatus('saving');
    try {
      const res = await fetch(`${API_BASE}/midi-corrections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          path: audioPath,
          notes: notes,
          original_notes: originalNotes,
          is_drum: effectiveDrumMode,
        }),
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setSaveStatus('saved');
        setHasEdits(false);
        setCorrectedNotes({ notes, original_notes: originalNotes, is_drum: effectiveDrumMode });
        onSave?.();
      } else {
        setSaveStatus('error');
      }
    } catch (e) {
      console.error('Save failed:', e);
      setSaveStatus('error');
    }
    setTimeout(() => setSaveStatus(null), 2000);
  };

  // Reset to original
  const handleReset = () => {
    if (originalNotes.length > 0) {
      setNotes(JSON.parse(JSON.stringify(originalNotes)));
      setHasEdits(true);
    }
  };

  // Run refined extraction
  const handleRunRefined = async () => {
    if (!audioPath) return;
    setIsRefining(true);
    setRefinedError(null);
    try {
      const res = await fetch(`${API_BASE}/midi-refined?audio_path=${encodeURIComponent(audioPath)}`);
      const data = await res.json();
      if (data.status === 'ok') {
        setRefinedNotes(data.notes || []);
        setRefinedInfo({
          strategy: data.strategy,
          group: data.group,
          subgroup: data.subgroup,
          stats: data.stats,
          comparison: data.comparison,
          data_available: data.data_available,
        });
        setShowRefined(true);
      } else {
        setRefinedError(data.error || 'Extraction failed');
        setRefinedNotes(null);
        setRefinedInfo(null);
      }
    } catch (e) {
      console.error('Refined extraction failed:', e);
      setRefinedError(e.message);
    }
    setIsRefining(false);
  };

  // Use refined notes as the current editable notes
  const handleUseRefined = () => {
    if (refinedNotes && refinedNotes.length > 0) {
      setNotes(JSON.parse(JSON.stringify(refinedNotes)));
      setHasEdits(true);
    }
  };

  // Wheel zoom
  const handleWheel = useCallback((e) => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.15 : 0.15;
    setZoomX(prev => Math.max(0.5, Math.min(5.0, prev + delta)));
  }, []);

  // Attach native wheel to prevent page zoom
  useEffect(() => {
    const wrapper = canvasWrapperRef.current;
    if (!wrapper) return;
    const handleNativeWheel = (e) => {
      if (e.ctrlKey || e.metaKey) { e.preventDefault(); e.stopPropagation(); }
    };
    wrapper.addEventListener('wheel', handleNativeWheel, { passive: false });
    return () => wrapper.removeEventListener('wheel', handleNativeWheel);
  }, []);

  if (isLoading && !isOnsetMode) {
    return <div className="midi-grid-loading">Loading MIDI...</div>;
  }

  if (!audioPath) {
    return <div className="midi-grid-empty">Select an entry to view its MIDI</div>;
  }

  return (
    <div className="midi-correction-grid">
      <div className="midi-grid-toolbar">
        <div className="midi-grid-tools">
          {!isOnsetMode && (<>
          <button
            className={`midi-tool-btn ${toolMode === 'select' ? 'active' : ''}`}
            onClick={() => setToolMode('select')}
            title="Select & Edit"
          >
            Select
          </button>
          <button
            className={`midi-tool-btn ${toolMode === 'draw' ? 'active' : ''}`}
            onClick={() => setToolMode('draw')}
            title="Draw Notes"
          >
            Draw
          </button>
          <span className="midi-tool-sep">|</span>
          </>)}
          <button className="midi-tool-btn" onClick={handleZoomOut} disabled={zoomX <= 0.5}>-</button>
          <span className="midi-zoom-label">{Math.round(zoomX * 100)}%</span>
          <button className="midi-tool-btn" onClick={handleZoomIn} disabled={zoomX >= 5.0}>+</button>
        </div>
        <div className="midi-grid-toggles">
          {!isOnsetMode && (
            <>
            <label className="midi-toggle">
              <input type="checkbox" checked={showOriginal} onChange={e => setShowOriginal(e.target.checked)} />
              <span>Original</span>
            </label>
            {correctedNotes && (
              <label className="midi-toggle">
                <input type="checkbox" checked={showCorrected} onChange={e => setShowCorrected(e.target.checked)} />
                <span>Prev Corrections</span>
              </label>
            )}
            {refinedNotes && (
              <label className="midi-toggle">
                <input type="checkbox" checked={showRefined} onChange={e => setShowRefined(e.target.checked)} />
                <span style={{color: '#f59e0b'}}>Refined</span>
              </label>
            )}
            </>
          )}
          {effectiveDrumMode && !isOnsetMode && <span className="midi-drum-badge">DRUM MODE</span>}
          {isOnsetMode && <span className="midi-drum-badge" style={{ background: '#e07050' }}>ONSET REVIEW</span>}
        </div>
        {!isOnsetMode && (
        <div className="midi-grid-actions">
          <button
            className={`midi-tool-btn midi-refined-btn ${isRefining ? 'running' : ''}`}
            onClick={handleRunRefined}
            disabled={!audioPath || isRefining}
            title="Run instrument-aware refined MIDI extraction"
          >
            {isRefining ? 'Extracting...' : 'Run Refined'}
          </button>
          {refinedNotes && refinedNotes.length > 0 && (
            <button
              className="midi-tool-btn midi-use-refined-btn"
              onClick={handleUseRefined}
              title="Replace current notes with refined extraction"
            >
              Use Refined
            </button>
          )}
          <button className="midi-tool-btn" onClick={handleReset} disabled={originalNotes.length === 0}>
            Reset
          </button>
          <button
            className={`midi-save-btn ${hasEdits ? 'has-edits' : ''} ${saveStatus || ''}`}
            onClick={handleSave}
            disabled={!hasEdits || saveStatus === 'saving'}
          >
            {saveStatus === 'saving' ? 'Saving...' : saveStatus === 'saved' ? 'Saved!' : saveStatus === 'error' ? 'Error' : 'Save Correction'}
          </button>
        </div>
        )}
      </div>

      {error && !notes.length && !isOnsetMode && (
        <div className="midi-grid-error">{error}</div>
      )}

      <div className="midi-grid-info">
        {isOnsetMode && onsets ? (
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
            {Object.entries(ONSET_STEM_COLORS).map(([stem, color]) => {
              const count = onsets[stem]?.count ?? 0;
              return (
                <span key={stem} style={{ display: 'flex', alignItems: 'center', gap: '3px', fontSize: '12px' }}>
                  <span style={{ width: '10px', height: '10px', background: color, display: 'inline-block', borderRadius: '2px' }} />
                  {stem} ({count})
                </span>
              );
            })}
            <span style={{ color: '#888', fontSize: '11px', marginLeft: '4px' }}>
              total: {Object.values(onsets).reduce((s, v) => s + (v?.count ?? 0), 0)}
            </span>
          </div>
        ) : (
          <>
            <span>{notes.length} notes</span>
            {originalNotes.length > 0 && notes.length !== originalNotes.length && (
              <span className="midi-diff">({notes.length > originalNotes.length ? '+' : ''}{notes.length - originalNotes.length} vs original)</span>
            )}
            {correctedNotes && <span className="midi-has-correction">Has previous correction</span>}
            {midiPath && <span className="midi-path-info" title={midiPath}>BasicPitch</span>}
          </>
        )}
      </div>

      {/* Refined extraction info */}
      {refinedInfo && (
        <div className="midi-refined-info">
          <span className="midi-refined-strategy" title="Extraction strategy used">
            Strategy: <strong>{refinedInfo.strategy}</strong>
          </span>
          <span>Group: {refinedInfo.group}/{refinedInfo.subgroup}</span>
          {refinedInfo.stats?.note_count != null && (
            <span>Refined: {refinedInfo.stats.note_count} notes</span>
          )}
          {refinedInfo.comparison && (
            <>
              <span>BasicPitch: {refinedInfo.comparison.baseline_count} notes</span>
              <span className={`midi-refined-reduction ${refinedInfo.comparison.reduction_pct > 0 ? 'positive' : 'negative'}`}>
                {refinedInfo.comparison.reduction_pct > 0 ? '' : '+'}{(-refinedInfo.comparison.reduction_pct).toFixed(1)}% change
              </span>
            </>
          )}
          {refinedInfo.data_available && (
            <span className="midi-data-sources" title="Available data sources">
              {Object.entries(refinedInfo.data_available)
                .filter(([, v]) => v)
                .map(([k]) => k)
                .join(', ')}
            </span>
          )}
        </div>
      )}
      {refinedError && (
        <div className="midi-refined-error">Refined extraction error: {refinedError}</div>
      )}

      <div ref={canvasWrapperRef} className="midi-grid-canvas-wrapper" onWheel={handleWheel}>
        <canvas
          ref={canvasRef}
          width={canvasWidth}
          height={canvasHeight}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onDoubleClick={handleDoubleClick}
          className="midi-grid-canvas"
        />
      </div>
    </div>
  );
};

export default MidiCorrectionGrid;
