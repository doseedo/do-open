/*
 * StudioDevMidi — themed piano-roll MIDI window for /studio-dev.
 *
 * Canvas-based, hi-fi purple palette. Reads the selected track's
 * midiData.notes and dispatches UPDATE_TRACK_MIDI_DATA on edits so the
 * production reducer, waveform view, and play-engine all stay in sync.
 *
 * Features:
 *   • keyboard rail (pitch labels, white/black stripes)
 *   • velocity-coloured note blocks in the track's color
 *   • click-empty → add note, drag → move, drag-right-edge → resize
 *   • double-click → delete, shift-click → multi-select
 *   • playhead that tracks state.playheadPosition
 *   • zoom: Ctrl/⌘+scroll (horiz), Alt+scroll (vert), buttons on the toolbar
 *   • pan: shift+scroll horizontal, plain scroll vertical
 *   • drum-roll mode for drum tracks (GM drum labels on y-axis)
 *   • empty-state prompt when no track is selected
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useApp } from '../../context/AppContext';

// ---- Pitch helpers ----
const NOTE_NAMES_SHARP = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const BLACK_KEYS = new Set(['C#','D#','F#','G#','A#']);
function midiToName(m) {
  const n = NOTE_NAMES_SHARP[((m % 12) + 12) % 12];
  const oct = Math.floor(m / 12) - 1;
  return `${n}${oct}`;
}
function isBlackKey(m) { return BLACK_KEYS.has(NOTE_NAMES_SHARP[((m % 12) + 12) % 12]); }

// GM drum labels for drum-roll mode. Subset matching the stemphonic
// backend's drum-class whitelist.
const GM_DRUM_LABEL = {
  35: 'Kick',  36: 'Kick',
  37: 'Rim',   39: 'Clap',
  38: 'Snare', 40: 'Snare',
  41: 'LowT',  43: 'LowT',
  42: 'HH',    44: 'HH',
  45: 'MidT',  47: 'MidT',
  46: 'OpenHH',
  48: 'HiT',   50: 'HiT',
  49: 'Crash', 57: 'Crash',
  51: 'Ride',  59: 'Ride',
  52: 'China', 53: 'Bell', 54: 'Tamb',
  55: 'Splash',56: 'Cowbell', 58: 'Vibra',
};
const DRUM_PITCH_MIN = 35, DRUM_PITCH_MAX = 59;

// Track-color palette from the hi-fi spec.
const TRACK_COLORS = {
  vocals: '#a88adc', lead: '#a88adc',
  rhodes: '#e8c88a', piano: '#e8c88a', keys: '#e8c88a',
  bass:   '#8ac8a0',
  drums:  '#e07556',
  strings:'#6aa8e8', guitar: '#6aa8e8',
  other:  '#a88adc',
};
function colorFor(type = '') {
  const t = type.toLowerCase();
  for (const [k, v] of Object.entries(TRACK_COLORS)) if (t.includes(k)) return v;
  return TRACK_COLORS.other;
}

// Palette — cream piano roll matching the rest of the workbench theme.
// Mirrors the --wb-* tokens in StudioDev.css so canvas + CSS surfaces
// agree exactly.
const C = {
  bg:        '#e8e6e1',                  // --wb-bg (canvas paper)
  surf:      '#f2f0ea',                  // --wb-surface (toolbar, ruler, keyboard rail)
  surf2:     '#dcd9d1',                  // --wb-surface-2 (even lanes)
  ink:       '#15181c',                  // --wb-ink
  inkSoft:   'rgba(21, 24, 28, 0.62)',
  inkMute:   'rgba(21, 24, 28, 0.38)',
  rule:      'rgba(21, 24, 28, 0.14)',
  ruleStrong:'rgba(21, 24, 28, 0.30)',
  accent:    '#c94f2c',                  // --wb-accent-warm (playhead)
  accentDeep:'#1d4c7a',                  // --wb-accent (selection)
};

const RULER_H  = 24;
const KEYS_W   = 54;

function hitTestNote(notes, mx, my, cfg) {
  for (let i = notes.length - 1; i >= 0; i--) {
    const n = notes[i];
    const x = KEYS_W + (n.time - cfg.scrollX) * cfg.pxPerSec;
    const y = RULER_H + (cfg.maxPitch - n.note) * cfg.rowH - cfg.scrollY;
    const w = Math.max(6, n.duration * cfg.pxPerSec);
    const h = cfg.rowH - 1;
    if (mx >= x && mx <= x + w && my >= y && my <= y + h) {
      // Edge grab: right 6 px → resize end (duration), left 6 px →
      // resize start (shifts time while keeping the far end pinned).
      // Notes under ~14 px wide skip left-edge resize so both grabs
      // don't overlap the centre.
      const edge =
        mx >= x + w - 6 ? 'right'
        : (w > 14 && mx <= x + 6) ? 'left'
        : null;
      return { idx: i, note: n, edge };
    }
  }
  return null;
}

export default function StudioDevMidi() {
  const { state, dispatch } = useApp();
  const selectedTrack = state.selectedTrack;
  const canvasRef  = useRef(null);
  const wrapRef    = useRef(null);
  const [size,    setSize]    = useState({ w: 800, h: 400 });
  const [pxPerSec,setPxPerSec]= useState(96);    // horizontal zoom
  const [rowH,    setRowH]    = useState(14);    // vertical zoom
  const [scrollX, setScrollX] = useState(0);
  const [scrollY, setScrollY] = useState(0);
  const [selected,setSelected]= useState(new Set());
  const [drag, setDrag] = useState(null);        // {mode:'move'|'resize-right'|'resize-left'|'marquee'|'new', ...}
  const [hoverTime, setHoverTime] = useState(null);
  // Cell currently under the cursor: { row, beatIdx } in piano-roll grid
  // coords. Drawn as a faint rect during draw() so users can see where a
  // click will land. Null when the cursor is over the ruler / key rail.
  const [hoverCell, setHoverCell] = useState(null);
  // Duration of the most-recently edited note, used as the default when
  // the user clicks an empty cell to add a new note. Starts at a half-beat.
  const lastNoteDurRef = useRef(null);

  const busIdForSelected = useMemo(() => {
    if (!selectedTrack) return null;
    for (const bus of state.buses || []) {
      if ((bus.tracks || []).some((t) => t.id === selectedTrack.id)) return bus.id;
    }
    return null;
  }, [state.buses, selectedTrack]);

  const notes = useMemo(() => {
    const md = selectedTrack?.midiData || selectedTrack?.metadata?.midiData;
    const raw = md?.notes || [];
    return raw
      .filter((n) => Number.isFinite(n.note) && Number.isFinite(n.time) && Number.isFinite(n.duration) && n.duration > 0)
      .map((n) => ({
        note: n.note | 0,
        time: n.time,
        duration: n.duration,
        velocity: Math.max(1, Math.min(127, n.velocity ?? 100)),
        lyric: n.lyric,
      }));
  }, [selectedTrack]);

  const type = (selectedTrack?.metadata?.stemType
             || selectedTrack?.metadata?.instrument
             || selectedTrack?.name || '').toLowerCase();
  const trackColor = colorFor(type);
  const isDrum = type.includes('drum') || type.includes('kick')
               || type.includes('snare') || type.includes('hat') || type.includes('perc');

  // Pitch viewport — HARD-LOCKED to the full keyboard range (or the GM
  // drum range for drum tracks). Previously we tried to "fit" the view
  // to the note extent, but every commit mutated the extent, which
  // shifted minPitch/maxPitch and translated the whole canvas. The
  // coord space now depends ONLY on isDrum, which depends on the
  // track's metadata — not on the notes array — so adding/moving notes
  // can never shift the window. Users who want a tighter view can
  // Alt-scroll to change rowH.
  const minPitch = isDrum ? DRUM_PITCH_MIN : 21;   // A0, bottom of 88-key
  const maxPitch = isDrum ? DRUM_PITCH_MAX : 108;  // C8, top of 88-key

  const totalSec = useMemo(() => {
    if (!notes.length) return 8;
    let end = 0;
    for (const n of notes) end = Math.max(end, n.time + n.duration);
    return Math.max(end + 1, 8);
  }, [notes]);

  // ---- Reactive canvas sizing ----
  // Observe the canvas ELEMENT (not the wrap) — that way the size we
  // read is the exact same number getBoundingClientRect returns to
  // mouse handlers. Observing the wrap meant a CSS-sized canvas could
  // disagree with the `size` state during layout changes, which caused
  // draws + hit tests to use different coordinate spaces (notes landed
  // above the cursor, hover cell was offset, canvas looked squished).
  useEffect(() => {
    const c = canvasRef.current; if (!c) return;
    const measure = () => setSize({ w: c.clientWidth, h: c.clientHeight });
    const ro = new ResizeObserver(measure);
    ro.observe(c);
    measure();
    return () => ro.disconnect();
  }, []);

  // ---- Writeback ----
  const commit = useCallback((nextNotes) => {
    if (!selectedTrack || !busIdForSelected) return;
    const md = selectedTrack?.midiData || selectedTrack?.metadata?.midiData || {};
    const duration = Math.max(md.duration || 0, ...nextNotes.map((n) => n.time + n.duration), 0);
    dispatch({
      type: 'UPDATE_TRACK_MIDI_DATA',
      payload: {
        trackId: selectedTrack.id,
        busId: busIdForSelected,
        midiData: { ...md, notes: nextNotes, duration, tempo: md.tempo || (state.bpm || 120) },
      },
    });
  }, [dispatch, selectedTrack, busIdForSelected, state.bpm]);

  // ---- Coords ----
  const cfg = { pxPerSec, rowH, maxPitch, minPitch, scrollX, scrollY };
  const timeAtX = (x) => (x - KEYS_W) / pxPerSec + scrollX;
  const pitchAtY = (y) => {
    const p = maxPitch - Math.floor((y - RULER_H + scrollY) / rowH);
    return Math.max(minPitch, Math.min(maxPitch, p));
  };

  // Mouse handlers. Drag uses window-level move/up so React re-renders
  // during commit() can't drop the drag session mid-flight.
  const onMouseDown = (e) => {
    if (!selectedTrack) return;
    const r = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    if (mx < KEYS_W || my < RULER_H) return;
    e.preventDefault();
    const hit = hitTestNote(notes, mx, my, cfg);
    if (hit) {
      let nextSel;
      if (e.shiftKey) {
        nextSel = new Set(selected);
        if (nextSel.has(hit.idx)) nextSel.delete(hit.idx); else nextSel.add(hit.idx);
      } else {
        nextSel = selected.has(hit.idx) ? selected : new Set([hit.idx]);
      }
      setSelected(nextSel);
      const mode =
        hit.edge === 'right' ? 'resize-right'
        : hit.edge === 'left' ? 'resize-left'
        : 'move';
      setDrag({
        mode,
        startClientX: e.clientX, startClientY: e.clientY,
        origNotes: notes.map((n) => ({ ...n })),
        indices: [...nextSel],
        moved: false,
      });
    } else {
      // Click empty → add a note at clicked position using the user's
      // last-edited note duration (falls back to a half-beat for the
      // very first note of a session). Continues into a resize drag so
      // the user can extend the duration in the same gesture if they
      // want a different length.
      const t = Math.max(0, timeAtX(mx));
      const p = pitchAtY(my);
      const beatSec = 60 / Math.max(40, state.bpm || 120);
      const defDur = lastNoteDurRef.current ?? (beatSec / 2);
      const newNote = {
        note: p,
        time: +(Math.round(t / (beatSec / 4)) * (beatSec / 4)).toFixed(4),
        duration: defDur, velocity: 100,
      };
      const nxt = [...notes, newNote];
      commit(nxt);
      setSelected(new Set([nxt.length - 1]));
      setDrag({
        mode: 'resize-right',
        startClientX: e.clientX, startClientY: e.clientY,
        origNotes: nxt.map((n) => ({ ...n })),
        indices: [nxt.length - 1],
        moved: false,
      });
    }
  };

  // Window-level move/up for the active drag. Re-attached when drag,
  // pxPerSec, rowH, minPitch, maxPitch change (any of which affect math).
  useEffect(() => {
    if (!drag) return;
    // Threshold (px) below which a mousedown-up is treated as a plain
    // click, not a drag — prevents tiny 1-2px jitters after selecting a
    // note from shifting it unexpectedly (user's 'note moves when I
    // click' complaint).
    const DRAG_THRESHOLD_PX = 3;
    const onMove = (e) => {
      const dxPx = e.clientX - drag.startClientX;
      const dyPx = e.clientY - drag.startClientY;
      if (!drag.moved && Math.abs(dxPx) < DRAG_THRESHOLD_PX && Math.abs(dyPx) < DRAG_THRESHOLD_PX) return;
      drag.moved = true;  // mutate in place — next tick takes this path
      const dt = dxPx / pxPerSec;
      const dp = -Math.round(dyPx / rowH);
      const nxt = drag.origNotes.map((n) => ({ ...n }));
      if (drag.mode === 'move') {
        for (const i of drag.indices) {
          nxt[i].time = Math.max(0, drag.origNotes[i].time + dt);
          nxt[i].note = Math.max(minPitch, Math.min(maxPitch, drag.origNotes[i].note + dp));
        }
      } else if (drag.mode === 'resize-right') {
        for (const i of drag.indices) {
          nxt[i].duration = Math.max(0.03, drag.origNotes[i].duration + dt);
        }
      } else if (drag.mode === 'resize-left') {
        // Drag the start edge: shift time forward, shrink duration by
        // the same amount so the right edge stays pinned. Don't let
        // the note get shorter than 0.03s or earlier than t=0.
        for (const i of drag.indices) {
          const orig = drag.origNotes[i];
          const maxShift = orig.duration - 0.03;
          const clampedDt = Math.max(-orig.time, Math.min(maxShift, dt));
          nxt[i].time = orig.time + clampedDt;
          nxt[i].duration = orig.duration - clampedDt;
        }
      }
      commit(nxt);
    };
    const onUp = () => {
      // On release, remember the last-edited note's duration so the
      // next empty-cell click creates a note the same size.
      if (drag.moved && drag.indices.length) {
        const last = drag.indices[drag.indices.length - 1];
        // Read from current notes; after commit the track state carries
        // the new duration. Fall back to origNotes if index shifted.
        const lastDur = (notes[last] && notes[last].duration) ?? drag.origNotes[last]?.duration;
        if (lastDur && lastDur > 0) lastNoteDurRef.current = lastDur;
      }
      setDrag(null);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [drag, pxPerSec, rowH, minPitch, maxPitch, commit, notes]);

  const onMouseMoveCanvas = (e) => {
    // Lightweight — cursor read-out + cell highlight + resize cursor.
    // Heavy drag math runs on window.
    const r = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    setHoverTime(timeAtX(mx));
    // Cell under cursor (grid-quantized to a quarter-beat column).
    if (mx >= KEYS_W && my >= RULER_H) {
      const beatSec = 60 / Math.max(40, state.bpm || 120);
      const cellSec = beatSec / 4;
      const t = timeAtX(mx);
      const beatIdx = Math.floor(Math.max(0, t) / cellSec);
      const row = Math.floor((my - RULER_H + scrollY) / rowH);
      setHoverCell({ row, beatIdx });
    } else {
      setHoverCell(null);
    }
    // Swap mouse cursor to ew-resize when hovering an edge so users see
    // the grab affordance before clicking. Falls back to crosshair on
    // empty cells and default 'grab' on note bodies.
    if (!drag) {
      const c = canvasRef.current;
      const hit = hitTestNote(notes, mx, my, cfg);
      c.style.cursor = hit
        ? (hit.edge ? 'ew-resize' : 'grab')
        : (mx < KEYS_W || my < RULER_H ? 'default' : 'crosshair');
    }
  };

  const onDoubleClick = (e) => {
    if (!selectedTrack) return;
    const r = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const hit = hitTestNote(notes, mx, my, cfg);
    if (hit) {
      const nxt = notes.filter((_, i) => i !== hit.idx);
      commit(nxt);
      setSelected(new Set());
    }
  };

  const onWheel = (e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      setPxPerSec((v) => Math.max(20, Math.min(800, v * (e.deltaY < 0 ? 1.15 : 1 / 1.15))));
    } else if (e.altKey) {
      e.preventDefault();
      setRowH((v) => Math.max(8, Math.min(36, v + (e.deltaY < 0 ? 1 : -1))));
    } else if (e.shiftKey) {
      e.preventDefault();
      setScrollX((v) => Math.max(0, v + e.deltaY / pxPerSec));
    } else {
      // vertical scroll
      setScrollY((v) => {
        const contentH = (maxPitch - minPitch + 1) * rowH;
        const maxS = Math.max(0, contentH - (size.h - RULER_H));
        return Math.max(0, Math.min(maxS, v + e.deltaY));
      });
    }
  };

  // Delete / escape key
  useEffect(() => {
    const h = (e) => {
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (!selected.size) return;
        const nxt = notes.filter((_, i) => !selected.has(i));
        commit(nxt);
        setSelected(new Set());
      } else if (e.key === 'Escape') {
        setSelected(new Set());
      }
    };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [notes, selected, commit]);

  // ---- DRAW ----
  useEffect(() => {
    const c = canvasRef.current; if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    // Measure the canvas directly at paint time. CSS sets the visible
    // dimensions to 100% of the wrap, so clientWidth/clientHeight is
    // the authoritative source — mouse handlers use the same values
    // via getBoundingClientRect, so there's zero chance of coord
    // space drift between where the user clicks and where we draw.
    const W = c.clientWidth;
    const H = c.clientHeight;
    if (W <= 0 || H <= 0) return;
    c.width  = W * dpr;
    c.height = H * dpr;
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Background
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);

    // Visible pitch range (with scroll)
    const firstPitchRow = Math.floor(scrollY / rowH);          // rows from top
    const lastPitchRow  = Math.ceil((scrollY + H - RULER_H) / rowH);
    const nPitches = maxPitch - minPitch + 1;

    // Lane backgrounds (zebra by white/black key)
    for (let row = firstPitchRow; row <= lastPitchRow; row++) {
      if (row < 0 || row >= nPitches) continue;
      const pitch = maxPitch - row;
      const y = RULER_H + row * rowH - scrollY;
      if (isDrum) {
        ctx.fillStyle = row % 2 === 0 ? C.surf : C.surf2;
      } else {
        ctx.fillStyle = isBlackKey(pitch) ? C.surf2 : C.surf;
      }
      ctx.fillRect(KEYS_W, y, W - KEYS_W, rowH);
      // Lane divider
      ctx.fillStyle = C.rule;
      ctx.fillRect(KEYS_W, y + rowH - 0.5, W - KEYS_W, 0.5);
      // Octave C highlight
      if (!isDrum && (pitch % 12) === 0) {
        ctx.fillStyle = C.ruleStrong;
        ctx.fillRect(KEYS_W, y, W - KEYS_W, 1);
      }
    }

    // Time-grid (beats + bars)
    const bpm = state.bpm || 120;
    const beatSec = 60 / bpm;
    const barSec  = beatSec * 4;
    const startSec = scrollX;
    const endSec = scrollX + (W - KEYS_W) / pxPerSec;
    // Bar lines
    for (let b = Math.floor(startSec / barSec); b * barSec < endSec; b++) {
      const x = KEYS_W + (b * barSec - scrollX) * pxPerSec;
      ctx.fillStyle = C.ruleStrong;
      ctx.fillRect(x, RULER_H, 1, H - RULER_H);
    }
    // Beat lines
    ctx.fillStyle = C.rule;
    for (let i = Math.floor(startSec / beatSec); i * beatSec < endSec; i++) {
      if ((i * beatSec) % barSec < 1e-6) continue;
      const x = KEYS_W + (i * beatSec - scrollX) * pxPerSec;
      ctx.fillRect(x, RULER_H, 1, H - RULER_H);
    }

    // Ruler
    ctx.fillStyle = C.surf;
    ctx.fillRect(0, 0, W, RULER_H);
    ctx.fillStyle = C.rule;
    ctx.fillRect(0, RULER_H - 1, W, 1);
    ctx.font = '10px "JetBrains Mono", ui-monospace, monospace';
    ctx.fillStyle = C.inkSoft;
    for (let b = Math.floor(startSec / barSec); b * barSec < endSec; b++) {
      const x = KEYS_W + (b * barSec - scrollX) * pxPerSec;
      ctx.fillText(String(b + 1), x + 4, 15);
    }

    // Keyboard rail
    ctx.fillStyle = C.surf;
    ctx.fillRect(0, RULER_H, KEYS_W, H - RULER_H);
    ctx.fillStyle = C.rule;
    ctx.fillRect(KEYS_W - 1, RULER_H, 1, H - RULER_H);
    ctx.font = '10px "JetBrains Mono", ui-monospace, monospace';
    for (let row = firstPitchRow; row <= lastPitchRow; row++) {
      if (row < 0 || row >= nPitches) continue;
      const pitch = maxPitch - row;
      const y = RULER_H + row * rowH - scrollY;
      // Black-key lane mark on rail
      if (!isDrum && isBlackKey(pitch)) {
        ctx.fillStyle = C.bg;
        ctx.fillRect(4, y + 1, KEYS_W - 10, rowH - 2);
      }
      if (rowH >= 10 && (isDrum || (pitch % 12) === 0 || rowH >= 14)) {
        ctx.fillStyle = C.inkMute;
        const label = isDrum ? (GM_DRUM_LABEL[pitch] || pitch) : midiToName(pitch);
        ctx.fillText(label, 6, y + rowH - 3);
      }
    }

    // Hover-cell highlight — faint rectangle showing where a click
    // would land. Skip while dragging (the note itself is the feedback).
    if (hoverCell && !drag) {
      const cellSec = beatSec / 4;
      const hx = KEYS_W + (hoverCell.beatIdx * cellSec - scrollX) * pxPerSec;
      const hy = RULER_H + hoverCell.row * rowH - scrollY;
      const hw = Math.max(2, cellSec * pxPerSec);
      if (hx + hw > KEYS_W && hx < W && hy + rowH > RULER_H && hy < H) {
        ctx.fillStyle = C.ink + '14';  // ~8% alpha
        ctx.fillRect(Math.max(KEYS_W, hx), hy, hw, rowH);
      }
    }

    // Notes
    for (let i = 0; i < notes.length; i++) {
      const n = notes[i];
      const x = KEYS_W + (n.time - scrollX) * pxPerSec;
      const y = RULER_H + (maxPitch - n.note) * rowH - scrollY;
      const w = Math.max(3, n.duration * pxPerSec);
      const h = rowH - 2;
      if (x + w < KEYS_W || x > W || y + h < RULER_H || y > H) continue;
      const alpha = 0.6 + (n.velocity / 127) * 0.4;
      ctx.fillStyle = trackColor + Math.floor(alpha * 255).toString(16).padStart(2, '0');
      ctx.beginPath();
      ctx.roundRect(x, y + 1, w, h, 2);
      ctx.fill();
      // Border
      ctx.strokeStyle = selected.has(i) ? C.ink : trackColor;
      ctx.lineWidth = selected.has(i) ? 1.4 : 1;
      ctx.stroke();
      // Lyric
      if (n.lyric && w > 24) {
        ctx.fillStyle = C.bg;
        ctx.font = '9px "Inter", system-ui, sans-serif';
        ctx.fillText(String(n.lyric).slice(0, 8), x + 4, y + h);
      }
    }

    // Playhead
    const px = KEYS_W + ((state.playheadPosition || 0) - scrollX) * pxPerSec;
    if (px >= KEYS_W && px < W) {
      ctx.fillStyle = C.accent;
      ctx.fillRect(px, 0, 1, H);
    }
  }, [size, notes, selected, trackColor, isDrum, pxPerSec, rowH, scrollX, scrollY, maxPitch, minPitch, state.bpm, state.playheadPosition, hoverCell, drag]);

  // Empty state — uses the workbench .wb-canvas / .wb-empty block verbatim
  // so the font stack (Inter 28/600 title, Inter 13 body, mono status pill)
  // and layout match the rest of the theme exactly.
  if (!selectedTrack) {
    return (
      <div className="wb-canvas">
        <div className="wb-canvas__grid" aria-hidden="true" />
        <div className="wb-empty">
          <div className="wb-empty__status">
            <div className="wb-empty__dot" />
            STATUS · NO TRACK SELECTED
          </div>
          <h1 className="wb-empty__title">Piano roll idle.</h1>
          <p className="wb-empty__body">
            Select a track in the timeline or pick an instrument in the sidebar to start writing notes.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="sd-midi">
      {/* Toolbar */}
      <div className="sd-midi-toolbar">
        <div className="sd-midi-title">
          <span className="sd-midi-color" style={{ background: trackColor }} />
          <span className="sd-midi-name">{selectedTrack.name || selectedTrack.id}</span>
          <span className="sd-midi-meta">
            {notes.length} notes · {selected.size ? `${selected.size} selected · ` : ''}{isDrum ? 'drum roll' : 'piano roll'}
          </span>
        </div>
        <div className="sd-midi-spacer" />
        <div className="sd-midi-group">
          <span className="sd-midi-kv-k">Zoom</span>
          <button className="sd-midi-btn" onClick={() => setPxPerSec((v) => Math.max(20, v / 1.25))}>−</button>
          <button className="sd-midi-btn" onClick={() => setPxPerSec((v) => Math.min(800, v * 1.25))}>+</button>
          <button className="sd-midi-btn" onClick={() => setRowH((v) => Math.max(8, v - 2))} title="Row shorter">⇡</button>
          <button className="sd-midi-btn" onClick={() => setRowH((v) => Math.min(36, v + 2))} title="Row taller">⇣</button>
        </div>
        <div className="sd-midi-group">
          <button className="sd-midi-btn" onClick={() => {
            // Quantize: snap note.time + duration to the nearest 16th note.
            const q = (60 / (state.bpm || 120)) / 4;
            const target = selected.size ? [...selected] : notes.map((_, i) => i);
            const nxt = notes.map((n, i) => target.includes(i)
              ? { ...n, time: Math.round(n.time / q) * q,
                  duration: Math.max(q, Math.round(n.duration / q) * q) }
              : n);
            commit(nxt);
          }} title="Snap note onsets+lengths to 16th grid">Quantize</button>
          <button className="sd-midi-btn" onClick={() => {
            // Humanize: jitter time ±1/64th, velocity ±10.
            const q = (60 / (state.bpm || 120)) / 16;
            const target = selected.size ? [...selected] : notes.map((_, i) => i);
            const nxt = notes.map((n, i) => {
              if (!target.includes(i)) return n;
              const j = (Math.random() - 0.5) * 2 * q;
              const v = Math.max(1, Math.min(127, (n.velocity || 100) + Math.round((Math.random() - 0.5) * 20)));
              return { ...n, time: Math.max(0, n.time + j), velocity: v };
            });
            commit(nxt);
          }}>Humanize</button>
        </div>
        <div className="sd-midi-group">
          <button className="sd-midi-btn" onClick={() => { setScrollX(0); setScrollY(0); }}>Reset view</button>
          <button className="sd-midi-btn sd-midi-danger" onClick={() => {
            if (!selected.size) { if (window.confirm('Clear all notes?')) commit([]); return; }
            const nxt = notes.filter((_, i) => !selected.has(i));
            commit(nxt); setSelected(new Set());
          }}>{selected.size ? `Delete (${selected.size})` : 'Clear'}</button>
        </div>
      </div>

      {/* Canvas area */}
      <div
        ref={wrapRef}
        className="sd-midi-canvas-wrap"
        onWheel={onWheel}
        onMouseLeave={() => setHoverCell(null)}
      >
        <canvas
          ref={canvasRef}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMoveCanvas}
          onDoubleClick={onDoubleClick}
          className="sd-midi-canvas"
        />
      </div>

      {/* Footer readout */}
      <div className="sd-midi-footer">
        <span className="sd-midi-kv-k">Cursor</span>
        <span className="sd-midi-kv-v">
          {hoverTime != null ? `${hoverTime.toFixed(2)}s` : '—'}
        </span>
        <div className="sd-midi-spacer" />
        <span className="sd-midi-kv-k">Bars</span>
        <span className="sd-midi-kv-v">{totalSec.toFixed(1)}s · {(state.bpm || 120).toFixed(0)} bpm</span>
      </div>
    </div>
  );
}
