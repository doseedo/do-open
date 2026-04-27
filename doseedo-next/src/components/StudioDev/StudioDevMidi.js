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
import midiPlayer from '../../utils/midiPlayer';
import ScoreView from '../MIDIChart/ScoreView';

// Safe expression evaluator for the axis settings. Allows arithmetic +
// `BPM` / `SR` variables, converts `^` to `**`, returns NaN on parse
// error. Kept small — no access to globals, just numeric eval.
function evalAxisExpr(expr, vars = { BPM: 120, SR: 48000 }) {
  if (typeof expr !== 'string') return NaN;
  try {
    const js = expr.replace(/\^/g, '**');
    const fn = new Function(...Object.keys(vars), `"use strict"; return (${js});`);
    const v = fn(...Object.values(vars));
    return Number.isFinite(v) ? v : NaN;
  } catch (_) { return NaN; }
}

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
    const span = Math.max(1, n.pitchSpan || 1);
    const topPitch = n.note + span - 1;
    const x = KEYS_W + (n.time - cfg.scrollX) * cfg.pxPerSec;
    const y = RULER_H + (cfg.maxPitch - topPitch) * cfg.rowH - cfg.scrollY;
    const w = Math.max(6, n.duration * cfg.pxPerSec);
    const h = span * cfg.rowH - 1;
    if (mx >= x && mx <= x + w && my >= y && my <= y + h) {
      // Edge grab bands scale with the note size so the zones are always
      // reachable whether the note is narrow or wide:
      //   left/right: min(8, w/3)
      //   top/bottom: min(8, h/3)
      // Vertical edges win over horizontal in the CENTRE of the side —
      // if the cursor is in the top 8 px of the note but not in a
      // horizontal edge's corner, it's a top grab.
      const hEdge = Math.min(8, Math.max(3, w / 3));
      const vEdge = Math.min(8, Math.max(3, h / 3));
      const onLeft   = mx <= x + hEdge;
      const onRight  = mx >= x + w - hEdge;
      const onTop    = my <= y + vEdge;
      const onBottom = my >= y + h - vEdge;
      // Priority: vertical edge if the cursor is NOT in a horizontal
      // corner zone. Corners fall through to the side (L/R) grab.
      let edge = null;
      if (onTop && !onLeft && !onRight) edge = 'top';
      else if (onBottom && !onLeft && !onRight) edge = 'bottom';
      else if (onRight) edge = 'right';
      else if (onLeft) edge = 'left';
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
  const [pxPerSec,setPxPerSec]= useState(96);    // X zoom (px per second). Cell width.
  // Independent Y zoom multiplier on top of cellSec*pxPerSec. Tuned so
  // the canvas opens with roughly an octave in view: at the default
  // 1/8-note cell (cellSec≈0.25 s, pxPerSec=96) this gives
  // 0.25*96*1.25 ≈ 30 px rows, ~13 pitches in a 400 px canvas. Arrow-
  // key Up/Down still adjusts in place.
  const [rowZoom, setRowZoom] = useState(1.25);
  const [scrollX, setScrollX] = useState(0);
  const [scrollY, setScrollY] = useState(0);
  // Hover flag for the arrow-key zoom shortcuts. Set on mouseenter /
  // unset on mouseleave on the canvas wrap. The keyboard listener
  // is window-level (canvas isn't focusable) and gates on this flag
  // so arrow keys outside the MIDI window still scroll the page.
  const [hovering, setHovering] = useState(false);
  // Axis settings — placeholder UI for future custom quantization.
  // Defaults mirror the built-in behavior: Y = 2^(1/12) (semitone
  // spacing), X = 60/BPM (one beat per unit). Expressions are stored
  // as strings now; we'll wire them to actual grid math in a follow-up.
  const [axisOpen, setAxisOpen] = useState(false);
  // Default cell base = one eighth note (30/BPM seconds), so the X grid
  // shows 2 cells per beat. New-note default snaps to one cell wide
  // (defDur logic in onMouseDown), giving the user 1/8-note placement
  // out of the box. Musical beat / bar lines below derive from BPM
  // directly so changing the cell base doesn't move the bar markers
  // off the actual downbeats.
  const [xAxisExpr, setXAxisExpr] = useState('30/BPM');
  const [yAxisExpr, setYAxisExpr] = useState('2^(1/12)');
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
  // Live cursor position in canvas-pixel coords. Updated by the canvas
  // mousemove handler and consumed by the arrow-key zoom-to-cursor
  // logic so keyboard zoom anchors to the same point that ctrl-scroll
  // does. Null when the cursor isn't over the grid region.
  const cursorPxRef = useRef({ x: null, y: null });

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
        // Preserve vertical stretch (default 1 = normal single-row
        // note). Dropping this was why Y-resize appeared to do nothing.
        pitchSpan: Math.max(1, n.pitchSpan | 0 || 1),
        // Pitch-trajectory params for multi-row notes:
        //   bend  ∈ [0, 1] — how far from the span midpoint the pitch
        //                     travels. 0 = static midpoint.
        //   curve ∈ [-1, 1] — shape: -1 log, 0 linear, +1 exp.
        bend: Number.isFinite(n.bend) ? Math.max(0, Math.min(1, n.bend)) : 0,
        curve: Number.isFinite(n.curve) ? Math.max(-1, Math.min(1, n.curve)) : 0,
        // Per-note override color (bus composite view tags each note with
        // its source track's palette color). Falls through to trackColor
        // in the note draw loop when absent.
        color: typeof n.color === 'string' ? n.color : null,
      }));
  }, [selectedTrack]);

  const type = (selectedTrack?.metadata?.stemType
             || selectedTrack?.metadata?.instrument
             || selectedTrack?.name || '').toLowerCase();
  const trackColor = colorFor(type);
  const trackIsDrum = type.includes('drum') || type.includes('kick')
               || type.includes('snare') || type.includes('hat') || type.includes('perc');
  const trackIsVocal = type.includes('vocal') || type.includes('vox') || type.includes('lead')
               || type.includes('lyric');

  // ---- View mode tabs (4-up): piano | drum | lyrics | score ----
  // Mirrors the legacy MIDIChart's chart-mode + roll-mode + vox toggles
  // collapsed into a single mutually-exclusive switch:
  //   piano  → standard 88-key piano roll
  //   drum   → GM percussion lanes (35..59) with drum labels
  //   lyrics → piano roll with per-note lyric overlay (writes n.lyric)
  //   score  → VexFlow sheet music (read-only render of the same notes)
  // Initial pick auto-derives from track type when the selected track
  // changes, then stays sticky so manual tab clicks aren't reverted on
  // re-render.
  const [viewMode, setViewMode] = useState('piano');
  const lastTrackIdRef = useRef(null);
  useEffect(() => {
    if (!selectedTrack) { lastTrackIdRef.current = null; return; }
    if (lastTrackIdRef.current === selectedTrack.id) return;
    lastTrackIdRef.current = selectedTrack.id;
    if (trackIsDrum) setViewMode('drum');
    else if (trackIsVocal) setViewMode('lyrics');
    else setViewMode('piano');
  }, [selectedTrack, trackIsDrum, trackIsVocal]);

  // Drum-roll lane constraints follow the explicit tab choice now —
  // drum tracks no longer force the GM-percussion grid unless the user
  // is on the drum tab. This lets users pull up a piano roll on a drum
  // track when they want to write tonal hits.
  const isDrum = viewMode === 'drum';
  const isLyrics = viewMode === 'lyrics';
  const isScore = viewMode === 'score';

  // ---- Cursor tools ----
  // arrow → select-only (no new-note placement, no F0 drawing)
  // block → default piano-roll behavior (place notes, drag, resize)
  // pen   → F0 pitch-contour draw (continuous pitch, persisted to
  //         midiData.f0Contour). When pen is engaged, mouse drags trace
  //         a polyline that gets committed on release; the existing
  //         contour stays visible on the canvas at all times.
  const [tool, setTool] = useState('block');
  const [f0Draft, setF0Draft] = useState(null);  // array while drawing, null otherwise

  const f0Contour = useMemo(() => {
    const md = selectedTrack?.midiData || selectedTrack?.metadata?.midiData;
    const raw = md?.f0Contour;
    return Array.isArray(raw)
      ? raw.filter((p) => Number.isFinite(p?.time) && Number.isFinite(p?.note))
      : [];
  }, [selectedTrack]);

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
    // selectedTrack?.id is included so the observer re-attaches the
    // first time a track gets picked. The component returns an empty-
    // state <div> with no canvas when no track is selected, so the
    // initial mount sees canvasRef.current === null and bails — and
    // without this dep nothing would trigger a second attach.
  }, [isScore, selectedTrack?.id]);

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

  // F0 contour writeback. Reuses UPDATE_TRACK_MIDI_DATA so the contour
  // travels with the rest of the MIDI payload (one persistence path,
  // one reducer). Notes are preserved verbatim — only f0Contour
  // changes. Pass [] to clear.
  const commitF0 = useCallback((nextContour) => {
    if (!selectedTrack || !busIdForSelected) return;
    const md = selectedTrack?.midiData || selectedTrack?.metadata?.midiData || {};
    dispatch({
      type: 'UPDATE_TRACK_MIDI_DATA',
      payload: {
        trackId: selectedTrack.id,
        busId: busIdForSelected,
        midiData: { ...md, notes: md.notes || [], f0Contour: nextContour },
      },
    });
  }, [dispatch, selectedTrack, busIdForSelected]);

  // ---- Grid timing ----
  // Must come BEFORE cfg/coord helpers below — cfg reads rowH, which
  // is derived from cellSec, which is derived from pxPerSec + bpm.
  // Putting the cfg literal first caused a TDZ (Cannot access 'W'/'rowH'
  // before initialization) because const rowH hoisted the binding but
  // left it unreadable until this block ran.
  //
  // cellSec is the single source of truth for:
  //   • note snapping on click/drag
  //   • hover-cell highlight width
  //   • subdivision lines drawn in the ruler + grid
  // beatSec is the user-configurable seconds-per-beat value. Default
  // expr is `60/BPM`; evaluating any other expression (e.g. `60/BPM/2`,
  // `1`) replaces it cleanly. Falls back to the standard formula if
  // parsing fails.
  const bpmVal = Math.max(40, state.bpm || 120);
  const xEval = evalAxisExpr(xAxisExpr, { BPM: bpmVal });
  const beatSec = Number.isFinite(xEval) && xEval > 0 ? xEval : 60 / bpmVal;

  // Y-axis ratio — frequency multiplier per MIDI-pitch row. Default is
  // 12-EDO (2^(1/12)); 2^(1/24) gives quarter-tones; 3/2 gives just-
  // intonation fifths. Evaluated + pushed to the sine synth below.
  const yAxisRatio = useMemo(() => {
    const v = evalAxisExpr(yAxisExpr, { BPM: bpmVal });
    return Number.isFinite(v) && v > 1 ? v : Math.pow(2, 1 / 12);
  }, [yAxisExpr, bpmVal]);

  // Push the current axis values into the shared midiPlayer so it uses
  // them for pitch frequency + time scaling on every playNote /
  // scheduleNotes. Default beatSec is 60/BPM; timeScale = beatSec /
  // default captures any user override of the X-axis expression.
  useEffect(() => {
    midiPlayer.initialize().catch(() => {});
    midiPlayer.setYAxisRatio(yAxisRatio);
    midiPlayer.setTimeScale(beatSec / (60 / bpmVal));
  }, [yAxisRatio, beatSec, bpmVal]);
  // Subdivide as soon as a beat is wider than ~48 px (threshold for
  // halving into eighths). The loop halves again whenever the CURRENT
  // cell is still wider than that, so the grid keeps up with zoom
  // through 8th → 16th → 32nd.
  const SUBDIVIDE_AT_PX = 48;
  let subdivision = 1;
  while (subdivision < 8 && (beatSec / subdivision) * pxPerSec > SUBDIVIDE_AT_PX * 2) {
    subdivision *= 2;
  }
  const cellSec = beatSec / subdivision;
  const snapTime = (t) => Math.round(t / cellSec) * cellSec;

  // Row height derives from the cell width (square cells) and is then
  // scaled by rowZoom — the user's independent Y multiplier. Default
  // rowZoom = 1 keeps the original square-cell behavior; arrow-key
  // Up/Down adjusts rowZoom so the keyboard rail can stretch/squeeze
  // without touching the X grid.
  const rowH = Math.max(12, Math.round(cellSec * pxPerSec * rowZoom));

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

    // Pen tool — start an F0 contour. Continuous pitch (not snapped to
    // semitones) is read off the y-coord using the same maxPitch/rowH
    // axis the notes use, so the polyline lands exactly where the cursor
    // is. The draft is committed to midiData.f0Contour on mouseup.
    if (tool === 'pen') {
      const t = Math.max(0, timeAtX(mx));
      const pitchF = maxPitch - (my - RULER_H + scrollY) / rowH;
      const seed = [{ time: t, note: pitchF }];
      setF0Draft(seed);
      setDrag({
        mode: 'f0',
        startClientX: e.clientX, startClientY: e.clientY,
        moved: true, // pen always counts as a draw, even single click
      });
      return;
    }

    const hit = hitTestNote(notes, mx, my, cfg);

    // Arrow tool — select-only. Hit a note → select it; click empty →
    // clear the selection. No new notes get placed, no resize/move
    // is started. Selection still routes through the same Set + drag
    // contract so Shift-add and per-note auditioning stay consistent.
    if (tool === 'arrow') {
      if (hit) {
        let nextSel;
        if (e.shiftKey) {
          nextSel = new Set(selected);
          if (nextSel.has(hit.idx)) nextSel.delete(hit.idx); else nextSel.add(hit.idx);
        } else {
          nextSel = new Set([hit.idx]);
        }
        setSelected(nextSel);
        midiPlayer.resume().then(() => {
          if (isDrum) midiPlayer.playDrum(hit.note.note, 0.7);
          else midiPlayer.playNote(hit.note.note, 0.7, Math.min(0.6, hit.note.duration || 0.3), 0, {
            span: hit.note.pitchSpan || 1,
            bend: hit.note.bend || 0,
            curve: hit.note.curve || 0,
          });
        }).catch(() => {});
      } else {
        setSelected(new Set());
      }
      return;
    }

    if (hit) {
      let nextSel;
      if (e.shiftKey) {
        nextSel = new Set(selected);
        if (nextSel.has(hit.idx)) nextSel.delete(hit.idx); else nextSel.add(hit.idx);
      } else {
        nextSel = selected.has(hit.idx) ? selected : new Set([hit.idx]);
      }
      setSelected(nextSel);
      // Audition the clicked note through the sine synth at the
      // current axis ratio + per-note bend/curve/span.
      midiPlayer.resume().then(() => {
        if (isDrum) midiPlayer.playDrum(hit.note.note, 0.7);
        else midiPlayer.playNote(hit.note.note, 0.7, Math.min(0.6, hit.note.duration || 0.3), 0, {
          span: hit.note.pitchSpan || 1,
          bend: hit.note.bend || 0,
          curve: hit.note.curve || 0,
        });
      }).catch(() => {});
      const mode =
        hit.edge === 'right' ? 'resize-right'
        : hit.edge === 'left' ? 'resize-left'
        : hit.edge === 'top' ? 'stretch-up'
        : hit.edge === 'bottom' ? 'stretch-down'
        : 'move';
      setDrag({
        mode,
        startClientX: e.clientX, startClientY: e.clientY,
        origNotes: notes.map((n) => ({ ...n })),
        indices: [...nextSel],
        // For stretch modes: snapshot of the note being stretched so
        // onMove can sync "extra" notes at adjacent pitches with the
        // same time/duration. Extra note indices are tracked so they
        // can be removed if the user drags back.
        anchorIdx: hit.idx,
        stretchExtras: [],
        moved: false,
      });
    } else {
      // Click empty → add a note at the snapped grid cell, with the
      // users last-edited duration (falls back to one cell). Continues
      // into a resize drag so the user can extend duration in the same
      // gesture if they want a different length.
      const t = Math.max(0, timeAtX(mx));
      const p = pitchAtY(my);
      // Default new-note duration = one cell at the active zoom. The
      // last-placed duration sticks ONLY when it was longer than a
      // cell — so a fresh load always lands on 1/8 note (the default
      // cell base) and after a 4-cell drag every subsequent click
      // produces a 4-cell note, but a tiny note from a deeply zoomed-in
      // session never bleeds back to a coarser zoom.
      const defDur = lastNoteDurRef.current && lastNoteDurRef.current > cellSec
        ? lastNoteDurRef.current
        : cellSec;
      const newNote = {
        note: p,
        time: +snapTime(t).toFixed(4),
        duration: defDur, velocity: 100,
      };
      const nxt = [...notes, newNote];
      commit(nxt);
      setSelected(new Set([nxt.length - 1]));
      // Audition the freshly placed note.
      midiPlayer.resume().then(() => {
        if (isDrum) midiPlayer.playDrum(p, 0.7);
        else midiPlayer.playNote(p, 0.7, 0.3);
      }).catch(() => {});
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
      // Pen tool — append continuous-pitch points to the f0 draft.
      // Throttled by the same heuristic the legacy MIDIChart used:
      // skip points that are too close to the previous one in either
      // axis (<5 ms / <0.1 semitones) so the polyline stays light.
      if (drag.mode === 'f0') {
        const c = canvasRef.current; if (!c) return;
        const r = c.getBoundingClientRect();
        const mx = e.clientX - r.left, my = e.clientY - r.top;
        if (mx < KEYS_W || my < RULER_H) return;
        const t = Math.max(0, (mx - KEYS_W) / pxPerSec + scrollX);
        const pitchF = maxPitch - (my - RULER_H + scrollY) / rowH;
        setF0Draft((prev) => {
          if (!prev) return [{ time: t, note: pitchF }];
          const last = prev[prev.length - 1];
          if (last && Math.abs(last.time - t) < 0.005 && Math.abs(last.note - pitchF) < 0.1) {
            return prev;
          }
          return [...prev, { time: t, note: pitchF }];
        });
        return;
      }
      const dxPx = e.clientX - drag.startClientX;
      const dyPx = e.clientY - drag.startClientY;
      if (!drag.moved && Math.abs(dxPx) < DRAG_THRESHOLD_PX && Math.abs(dyPx) < DRAG_THRESHOLD_PX) return;
      drag.moved = true;  // mutate in place — next tick takes this path
      // Snap drag delta to the active grid cell so notes always land on
      // cell boundaries. Pitch is already quantized (row-height integer).
      const dtSnapped = Math.round((dxPx / pxPerSec) / cellSec) * cellSec;
      const dp = -Math.round(dyPx / rowH);
      const nxt = drag.origNotes.map((n) => ({ ...n }));
      if (drag.mode === 'move') {
        for (const i of drag.indices) {
          nxt[i].time = Math.max(0, drag.origNotes[i].time + dtSnapped);
          nxt[i].note = Math.max(minPitch, Math.min(maxPitch, drag.origNotes[i].note + dp));
        }
      } else if (drag.mode === 'resize-right') {
        for (const i of drag.indices) {
          nxt[i].duration = Math.max(cellSec, drag.origNotes[i].duration + dtSnapped);
        }
      } else if (drag.mode === 'resize-left') {
        for (const i of drag.indices) {
          const orig = drag.origNotes[i];
          const maxShift = orig.duration - cellSec;
          const clampedDt = Math.max(-orig.time, Math.min(maxShift, dtSnapped));
          nxt[i].time = orig.time + clampedDt;
          nxt[i].duration = orig.duration - clampedDt;
        }
      } else if (drag.mode === 'stretch-up') {
        // Drag the top edge up → grow pitchSpan; base.note pinned.
        // pitchSpan=1 is a normal single-row note. The anchor's base
        // note stays and the note visually covers multiple rows.
        for (const i of drag.indices) {
          const orig = drag.origNotes[i];
          const baseSpan = orig.pitchSpan || 1;
          const newSpan = Math.max(1, Math.min(maxPitch - orig.note + 1, baseSpan + dp));
          nxt[i].pitchSpan = newSpan;
        }
      } else if (drag.mode === 'stretch-down') {
        // Drag the bottom edge down → grow pitchSpan downward. base.note
        // decreases so the TOP stays pinned at (origNote + origSpan - 1).
        for (const i of drag.indices) {
          const orig = drag.origNotes[i];
          const baseSpan = orig.pitchSpan || 1;
          const origTop = orig.note + baseSpan - 1;
          // dp is positive when dragging up; bottom-edge wants the
          // inverse — dragging DOWN (dp < 0) should extend the note.
          const grow = -dp;
          const newSpan = Math.max(1, Math.min(origTop - minPitch + 1, baseSpan + grow));
          nxt[i].pitchSpan = newSpan;
          nxt[i].note = origTop - newSpan + 1;
        }
      }
      commit(nxt);
    };
    const onUp = () => {
      // Pen tool — finalize the F0 draft into the persisted contour.
      // Empty / 1-point drafts are dropped so accidental clicks don't
      // overwrite an existing contour with a stub.
      if (drag.mode === 'f0') {
        setF0Draft((draft) => {
          if (draft && draft.length > 1) commitF0(draft);
          return null;
        });
        setDrag(null);
        return;
      }
      // On release, remember the last-edited note's duration so the
      // next empty-cell click creates a note the same size.
      if (drag.moved && drag.indices && drag.indices.length) {
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
  }, [drag, pxPerSec, rowH, minPitch, maxPitch, scrollX, scrollY, commit, commitF0, notes]);

  const onMouseMoveCanvas = (e) => {
    // Lightweight — cursor read-out + cell highlight + resize cursor.
    // Heavy drag math runs on window.
    const r = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    setHoverTime(timeAtX(mx));
    // Cell under cursor — store raw time + row; snap to the active
    // cell at DRAW time so when the user zooms in mid-hover the
    // highlight re-quantizes automatically (previously beatIdx was
    // frozen to old cellSec, making the hover rect misalign after a
    // zoom change without a fresh mouse move).
    if (mx >= KEYS_W && my >= RULER_H) {
      const t = Math.max(0, timeAtX(mx));
      const row = Math.floor((my - RULER_H + scrollY) / rowH);
      setHoverCell({ row, time: t });
      cursorPxRef.current = { x: mx, y: my };
    } else {
      setHoverCell(null);
      cursorPxRef.current = { x: null, y: null };
    }
    // Swap mouse cursor based on the active tool. Block keeps the
    // edge/grab/crosshair affordances; arrow forces a pointer (no draw
    // intent); pen forces a crosshair so users feel the F0-draw mode.
    if (!drag) {
      const c = canvasRef.current;
      const overGutter = mx < KEYS_W || my < RULER_H;
      if (tool === 'pen') {
        c.style.cursor = overGutter ? 'default' : 'crosshair';
      } else if (tool === 'arrow') {
        const hit = hitTestNote(notes, mx, my, cfg);
        c.style.cursor = overGutter ? 'default' : (hit ? 'pointer' : 'default');
      } else {
        const hit = hitTestNote(notes, mx, my, cfg);
        c.style.cursor = hit
          ? (hit.edge === 'left' || hit.edge === 'right' ? 'ew-resize'
            : hit.edge === 'top' || hit.edge === 'bottom' ? 'ns-resize'
            : 'grab')
          : (overGutter ? 'default' : 'crosshair');
      }
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

  // Wheel handler attached as a NATIVE non-passive listener. React's
  // synthetic onWheel is always passive in recent Chrome/Safari, so
  // preventDefault() inside it throws 'Unable to preventDefault inside
  // passive event listener invocation' and zoom still scrolls the
  // page underneath. Using addEventListener('wheel', ..., { passive:
  // false }) is the supported way to both stop the page scroll and
  // drive our own X/Y zoom + pan.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    // Score view scrolls natively (it's an SVG inside a scrollable div);
    // suppressing wheel here would steal that scroll. Skip the listener
    // entirely and let the browser handle the score pane.
    if (isScore) return;
    const onWheelNative = (e) => {
      if (e.ctrlKey || e.metaKey) {
        // Zoom-to-cursor: the point under the cursor stays anchored
        // so users can dial in a region. We:
        //   1. compute the logical (time, rowIdx) under the cursor
        //      using the CURRENT pxPerSec / rowH
        //   2. scale pxPerSec by 1.15 and derive the new rowH (same
        //      square-cell rule as the render body)
        //   3. pick new scrollX/scrollY so that same (time, rowIdx)
        //      still lands under the same (mx, my)
        e.preventDefault();
        const c = canvasRef.current; if (!c) return;
        const r = c.getBoundingClientRect();
        const mx = e.clientX - r.left;
        const my = e.clientY - r.top;
        // Only zoom when cursor is over the grid region.
        if (mx < KEYS_W || my < RULER_H) return;

        // Normalize deltaY across deltaMode (0=pixels, 1=lines, 2=pages)
        // then clamp per-event magnitude so a huge deltaY from a fast
        // scroll doesnt saturate the zoom in one tick. Linear zoom
        // sensitivity feels predictable across mouse-wheel and trackpad:
        //   mouse wheel notch: deltaPx ≈ 100 → step 0.15 → factor 1.15
        //   trackpad pinch:    deltaPx ≈ 10  → step 0.015 → factor 1.015
        const deltaPx = e.deltaMode === 1 ? e.deltaY * 16
                       : e.deltaMode === 2 ? e.deltaY * 400
                       : e.deltaY;
        // Sensitivity 0.008: mouse wheel notch (100 px) → step 0.60
        // (clamped), factor 1.60 — punchy. Trackpad pinch ticks
        // (~10 px) → step 0.08, factor 1.08 — still smooth, but a
        // single deliberate gesture moves the zoom meaningfully.
        const step = Math.max(-0.6, Math.min(0.6, -deltaPx * 0.008));
        const factor = 1 + step;
        const newPxPerSec = Math.max(20, Math.min(800, pxPerSec * factor));
        // If the clamped factor rounded to a no-op, bail early so we
        // dont schedule a state update that re-runs this useEffect for
        // nothing.
        if (Math.abs(newPxPerSec - pxPerSec) < 0.01) return;
        // Derive the new cellSec + rowH the same way the render body does.
        let newSub = 1;
        while (newSub < 8 && (beatSec / newSub) * newPxPerSec > SUBDIVIDE_AT_PX * 2) newSub *= 2;
        const newCellSec = beatSec / newSub;
        const newRowH = Math.max(12, Math.round(newCellSec * newPxPerSec * rowZoom));

        // Logical anchor under the cursor BEFORE zoom.
        const t = (mx - KEYS_W) / pxPerSec + scrollX;
        const rowIdxF = (my - RULER_H + scrollY) / rowH;
        // New scroll so (t, rowIdxF) projects back to (mx, my).
        const nextScrollX = Math.max(0, t - (mx - KEYS_W) / newPxPerSec);
        const contentH = (maxPitch - minPitch + 1) * newRowH;
        const maxScrollY = Math.max(0, contentH - (size.h - RULER_H));
        const nextScrollY = Math.max(0, Math.min(maxScrollY,
          rowIdxF * newRowH - (my - RULER_H)));

        setPxPerSec(newPxPerSec);
        setScrollX(nextScrollX);
        setScrollY(nextScrollY);
      } else if (e.shiftKey) {
        // Shift+wheel always pans X (single-axis mice without native
        // horizontal scroll).
        e.preventDefault();
        setScrollX((v) => Math.max(0, v + e.deltaY / pxPerSec));
      } else {
        // Trackpads + horizontal-capable mice report BOTH deltaX and
        // deltaY on a single event, so we pan each axis independently:
        // deltaX → scrollX, deltaY → scrollY. Previously deltaX was
        // dropped, which is why users couldn't horizontal-scroll the
        // piano roll at all on trackpads.
        e.preventDefault();
        if (e.deltaX) {
          setScrollX((v) => Math.max(0, v + e.deltaX / pxPerSec));
        }
        if (e.deltaY) {
          setScrollY((v) => {
            const contentH = (maxPitch - minPitch + 1) * rowH;
            const maxS = Math.max(0, contentH - (size.h - RULER_H));
            return Math.max(0, Math.min(maxS, v + e.deltaY));
          });
        }
      }
    };
    el.addEventListener('wheel', onWheelNative, { passive: false });
    return () => el.removeEventListener('wheel', onWheelNative);
    // selectedTrack?.id pulls this effect back into life when the
    // user picks the first track — until then the empty-state branch
    // returns a different DOM tree with no wrapRef target, so the
    // initial mount attaches nothing and silent-bails. (Symptom: scroll
    // zoom only worked after clicking the toolbar zoom buttons, which
    // changed pxPerSec and re-ran this effect.)
  }, [pxPerSec, rowH, maxPitch, minPitch, size.h, scrollX, scrollY, beatSec, rowZoom, isScore, selectedTrack?.id]);

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

  // Arrow-key zoom — only fires while the mouse is over the MIDI wrap
  // and no notes are selected (so arrow keys don't fight any future
  // note-nudge shortcut, and don't hijack page scroll when the user
  // isn't looking at the editor). Up/Down adjust the Y zoom (rowZoom),
  // Left/Right adjust the X zoom (pxPerSec). Anchored at the current
  // cursor position the same way ctrl-wheel does — point under the
  // cursor stays put as the grid grows/shrinks. When the cursor is
  // outside the grid, the canvas centre is used as the anchor.
  useEffect(() => {
    if (!hovering || isScore) return;
    if (selected.size > 0) return;
    const STEP = 1.15;
    const onKey = (e) => {
      // Ignore when the user is typing into a field (axis input,
      // lyric input, etc.) so editing remains undisturbed.
      const tag = (e.target?.tagName || '').toUpperCase();
      if (tag === 'INPUT' || tag === 'TEXTAREA' || e.target?.isContentEditable) return;
      const isLR = e.key === 'ArrowLeft' || e.key === 'ArrowRight';
      const isUD = e.key === 'ArrowUp'   || e.key === 'ArrowDown';
      if (!isLR && !isUD) return;
      e.preventDefault();

      // Pick the anchor point in canvas-pixel space. Prefer the live
      // cursor (set in onMouseMoveCanvas), fall back to the visible
      // grid centre so the user still gets a sensible zoom even when
      // they tap a key without moving the mouse first.
      const cur = cursorPxRef.current;
      const haveCursor = Number.isFinite(cur.x) && Number.isFinite(cur.y);
      const mx = haveCursor ? cur.x : KEYS_W + Math.max(0, (size.w - KEYS_W) / 2);
      const my = haveCursor ? cur.y : RULER_H + Math.max(0, (size.h - RULER_H) / 2);

      // Logical anchor under the cursor BEFORE the zoom change — same
      // formulas the wheel handler uses.
      const t = (mx - KEYS_W) / pxPerSec + scrollX;
      const rowIdxF = (my - RULER_H + scrollY) / rowH;

      if (isLR) {
        const newPxPerSec = e.key === 'ArrowRight'
          ? Math.min(800, pxPerSec * STEP)
          : Math.max(20, pxPerSec / STEP);
        if (Math.abs(newPxPerSec - pxPerSec) < 0.01) return;
        // Recompute the X subdivision at the new pxPerSec.
        let newSub = 1;
        while (newSub < 8 && (beatSec / newSub) * newPxPerSec > SUBDIVIDE_AT_PX * 2) newSub *= 2;
        const newCellSec = beatSec / newSub;
        // Back-solve rowZoom so rowH stays exactly where it was — L/R
        // is X-only zoom. Without this, rowH = cellSec*pxPerSec*rowZoom
        // grew with pxPerSec inside a subdivision band, dragging the Y
        // axis along for the ride.
        const targetRowH = rowH;
        const newRowZoom = Math.max(0.05, Math.min(32,
          targetRowH / Math.max(1, newCellSec * newPxPerSec)));
        const newRowH = Math.max(12, Math.round(newCellSec * newPxPerSec * newRowZoom));
        const nextScrollX = Math.max(0, t - (mx - KEYS_W) / newPxPerSec);
        const contentH = (maxPitch - minPitch + 1) * newRowH;
        const maxScrollY = Math.max(0, contentH - (size.h - RULER_H));
        // Y anchor stays at the cursor; with rowH unchanged this is a no-op
        // when the cursor was over the grid, but keeps things clean if the
        // round() above shifted rowH by a pixel.
        const nextScrollY = Math.max(0, Math.min(maxScrollY, rowIdxF * newRowH - (my - RULER_H)));
        setPxPerSec(newPxPerSec);
        setRowZoom(newRowZoom);
        setScrollX(nextScrollX);
        setScrollY(nextScrollY);
      } else {
        const newRowZoom = e.key === 'ArrowUp'
          ? Math.min(8, rowZoom * STEP)
          : Math.max(0.25, rowZoom / STEP);
        if (Math.abs(newRowZoom - rowZoom) < 1e-4) return;
        const newRowH = Math.max(12, Math.round(cellSec * pxPerSec * newRowZoom));
        const contentH = (maxPitch - minPitch + 1) * newRowH;
        const maxScrollY = Math.max(0, contentH - (size.h - RULER_H));
        const nextScrollY = Math.max(0, Math.min(maxScrollY, rowIdxF * newRowH - (my - RULER_H)));
        setRowZoom(newRowZoom);
        setScrollY(nextScrollY);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [hovering, isScore, selected.size, pxPerSec, rowH, rowZoom, scrollX, scrollY,
      beatSec, cellSec, maxPitch, minPitch, size.w, size.h]);

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

    // Time-grid — three tiers:
    //   bar   → strong rule (musical bar boundary)
    //   beat  → medium rule (musical beat — independent of cell base)
    //   cell  → very faint (cellSec, derived from xAxisExpr+subdivision)
    // The musical beat/bar math reads BPM directly so changing the cell
    // base via xAxisExpr (default 120/BPM = 2 beats/cell) doesn't shift
    // bar markers off the actual downbeats.
    const musicalBeatSec = 60 / bpmVal;
    const beatsPerBar = state.beatsPerBar || 4;
    const barSec = musicalBeatSec * beatsPerBar;
    const startSec = scrollX;
    const endSec = scrollX + (W - KEYS_W) / pxPerSec;

    // Cell lines (faintest) — drawn first so beat/bar lines paint on top.
    // Drawn whenever cellSec ≠ musicalBeatSec OR when subdivision > 1
    // (zoomed in so subdivided cells exist). At default 2-beat cells the
    // cell lines coincide with every-other-beat markers; we still draw
    // them so the user sees the active grid the snap is using.
    ctx.fillStyle = 'rgba(21, 24, 28, 0.12)';
    const firstCell = Math.floor(startSec / cellSec);
    for (let i = firstCell; i * cellSec < endSec; i++) {
      const tCell = i * cellSec;
      // skip cells coincident with a musical beat — beat lines below
      // will draw a stronger rule there.
      if (Math.abs(tCell / musicalBeatSec - Math.round(tCell / musicalBeatSec)) < 1e-6) continue;
      const x = KEYS_W + (tCell - scrollX) * pxPerSec;
      ctx.fillRect(x, RULER_H, 1, H - RULER_H);
    }

    // Bar lines (strongest)
    for (let b = Math.floor(startSec / barSec); b * barSec < endSec; b++) {
      const x = KEYS_W + (b * barSec - scrollX) * pxPerSec;
      ctx.fillStyle = C.ruleStrong;
      ctx.fillRect(x, RULER_H, 1, H - RULER_H);
    }
    // Beat lines — every musical beat, skipping ones that coincide with
    // a bar (already drawn stronger above).
    ctx.fillStyle = C.rule;
    for (let i = Math.floor(startSec / musicalBeatSec); i * musicalBeatSec < endSec; i++) {
      if ((i * musicalBeatSec) % barSec < 1e-6) continue;
      const x = KEYS_W + (i * musicalBeatSec - scrollX) * pxPerSec;
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

    // Hover-cell highlight — rectangle exactly matching the active
    // grid cell. We floor the hover time to the nearest cell AT DRAW
    // TIME using the current cellSec, then left/right edges come from
    // the same pxPerSec math the grid lines use → zero-gap fit.
    if (hoverCell && !drag) {
      const cellStartSec = Math.floor(hoverCell.time / cellSec) * cellSec;
      const hxRaw = KEYS_W + (cellStartSec - scrollX) * pxPerSec;
      const hxRight = hxRaw + cellSec * pxPerSec;
      const hy = RULER_H + hoverCell.row * rowH - scrollY;
      const hx = Math.max(KEYS_W, hxRaw);
      const hw = Math.max(2, hxRight - hx);
      if (hxRight > KEYS_W && hxRaw < W && hy + rowH > RULER_H && hy < H) {
        ctx.fillStyle = C.ink + '1f';  // ~12% alpha
        ctx.fillRect(hx, hy, hw, rowH);
      }
    }

    // Notes — pitchSpan (default 1) stretches the note over multiple
    // rows; the rect's top sits at (maxPitch - topPitch) and the
    // height is span*rowH.
    for (let i = 0; i < notes.length; i++) {
      const n = notes[i];
      const span = Math.max(1, n.pitchSpan || 1);
      const topPitch = n.note + span - 1;
      const x = KEYS_W + (n.time - scrollX) * pxPerSec;
      const y = RULER_H + (maxPitch - topPitch) * rowH - scrollY;
      const w = Math.max(3, n.duration * pxPerSec);
      const h = span * rowH - 2;
      if (x + w < KEYS_W || x > W || y + h < RULER_H || y > H) continue;
      const alpha = 0.6 + (n.velocity / 127) * 0.4;
      const col = n.color || trackColor;
      ctx.fillStyle = col + Math.floor(alpha * 255).toString(16).padStart(2, '0');
      ctx.beginPath();
      ctx.roundRect(x, y + 1, w, h, 2);
      ctx.fill();
      // Border
      ctx.strokeStyle = selected.has(i) ? C.ink : col;
      ctx.lineWidth = selected.has(i) ? 1.4 : 1;
      ctx.stroke();
      // Lyric
      if (n.lyric && w > 24) {
        ctx.fillStyle = C.bg;
        ctx.font = '9px "Inter", system-ui, sans-serif';
        ctx.fillText(String(n.lyric).slice(0, 8), x + 4, y + h);
      }
    }

    // F0 contour — draw the polyline + control dots OVER the notes so
    // the pitch trajectory stays readable. Draws f0Draft while the
    // user is mid-pen-stroke, otherwise the persisted contour. Pitch
    // is continuous so the y math uses fractional rows directly.
    const contourToDraw = f0Draft || f0Contour;
    if (contourToDraw && contourToDraw.length) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(KEYS_W, RULER_H, W - KEYS_W, H - RULER_H);
      ctx.clip();
      ctx.strokeStyle = C.accentDeep;
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      contourToDraw.forEach((pt, i) => {
        const cx = KEYS_W + (pt.time - scrollX) * pxPerSec;
        const cy = RULER_H + (maxPitch - pt.note) * rowH - scrollY;
        if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
      });
      ctx.stroke();
      // Dots only shown for persisted contour (not the live draft) so
      // dragging stays visually quiet.
      if (!f0Draft) {
        ctx.fillStyle = C.accentDeep;
        for (let i = 0; i < contourToDraw.length; i += Math.max(1, Math.floor(contourToDraw.length / 200))) {
          const pt = contourToDraw[i];
          const cx = KEYS_W + (pt.time - scrollX) * pxPerSec;
          const cy = RULER_H + (maxPitch - pt.note) * rowH - scrollY;
          ctx.beginPath();
          ctx.arc(cx, cy, 2.2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore();
    }

    // Playhead
    const px = KEYS_W + ((state.playheadPosition || 0) - scrollX) * pxPerSec;
    if (px >= KEYS_W && px < W) {
      ctx.fillStyle = C.accent;
      ctx.fillRect(px, 0, 1, H);
    }
  }, [size, notes, selected, trackColor, isDrum, pxPerSec, rowH, scrollX, scrollY, maxPitch, minPitch, state.bpm, state.beatsPerBar, beatSec, cellSec, state.playheadPosition, hoverCell, drag, f0Contour, f0Draft]);

  // Empty state — uses the workbench .wb-canvas / .wb-empty block verbatim
  // so the font stack (Inter 28/600 title, Inter 13 body, mono status pill)
  // and layout match the rest of the theme exactly. Includes a "New
  // MIDI track" button that creates a fresh INSTRUMENT bus + empty
  // type:'midi' track and selects it — same payload shape StudioDev's
  // addInstrumentTrack uses, replicated here so the empty MIDI window
  // can self-unlock without the user reaching for the left sidebar.
  if (!selectedTrack) {
    const selBus = state.selectedBus;
    const busVisibleTracks = selBus
      ? (selBus.tracks || []).filter((t) => !t.metadata?.isBusMaster)
      : [];

    // Empty-bus path — adds a fresh type:'midi' track INTO the selected
    // bus (vs creating a brand new bus). The piano roll then opens on
    // that new track because SELECT_TRACK fires straight after.
    const addTrackToBus = () => {
      if (!selBus) return;
      const trackId = `t-${Date.now()}`;
      dispatch({
        type: 'ADD_TRACK',
        payload: {
          busId: selBus.id,
          track: {
            id: trackId,
            name: selBus.tracks?.length ? `Track ${(selBus.tracks?.length || 0) + 1}` : 'New track',
            duration: 4, startPosition: 0,
            gain: 1.0, isMuted: false, isSolo: false,
            fx: { reverb: 0, fadeIn: 0, fadeOut: 0 },
            metadata: { type: 'midi', instrument: 'other', icon: 'synth' },
            midiData: { notes: [], duration: 4, tempo: state.bpm || 120 },
          },
        },
      });
      dispatch({ type: 'SELECT_TRACK', payload: { trackId, busId: selBus.id } });
    };

    // Composite-open path — synthesizes the master MIDI from every
    // child track in the bus, tagging each note with the source track's
    // palette colour so the StudioDevMidi canvas paints multitrack
    // notes in different colours. The composite carries
    // metadata.isComposite + sourceTrackIds so a future commit handler
    // can route writes to a primary track in the bus.
    const COMPOSITE_PALETTE = ['#c94f2c', '#7a5d3a', '#2f6b4e', '#4a3d6b', '#1d4c7a', '#a88adc', '#6aa8e8', '#e07556'];
    const openBusMaster = () => {
      if (!selBus || busVisibleTracks.length === 0) return;
      if (busVisibleTracks.length === 1) {
        dispatch({ type: 'SELECT_TRACK', payload: { trackId: busVisibleTracks[0].id, busId: selBus.id } });
        return;
      }
      const mergedNotes = [];
      let maxDur = 0;
      busVisibleTracks.forEach((t, i) => {
        const colorTag = COMPOSITE_PALETTE[i % COMPOSITE_PALETTE.length];
        const md = t.midiData || t.metadata?.midiData;
        const ns = md?.notes || [];
        for (const n of ns) mergedNotes.push({ ...n, color: colorTag, __sourceTrackId: t.id });
        maxDur = Math.max(maxDur, md?.duration || 0, t.duration || 0);
      });
      const compositeTrack = {
        id: `__composite-${selBus.id}`,
        name: `${selBus.name || selBus.type || 'Bus'} · ${busVisibleTracks.length} tracks`,
        midiData: { notes: mergedNotes, duration: maxDur || 8, tempo: state.bpm || 120 },
        metadata: {
          type: 'composite',
          isComposite: true,
          sourceTrackIds: busVisibleTracks.map((t) => t.id),
        },
      };
      dispatch({ type: 'SELECT_TRACK', payload: { compositeTrack, busId: selBus.id } });
    };

    // Fall-through — no bus + no track. Adds a fresh bus AND track,
    // matching addInstrumentTrack's payload shape so the rest of the
    // app (right sidebar, playback bus filter, etc.) treats it the
    // same as any other studio-created MIDI track.
    const createNewMidiTrackAndBus = () => {
      const busId = `bus-${Date.now()}`;
      const trackId = `t-${Date.now()}`;
      dispatch({
        type: 'CREATE_BUS',
        payload: { id: busId, type: 'INSTRUMENT', name: 'New track', expanded: false },
      });
      dispatch({
        type: 'ADD_TRACK',
        payload: {
          busId,
          track: {
            id: trackId, name: 'New track', duration: 4, startPosition: 0,
            gain: 1.0, isMuted: false, isSolo: false,
            fx: { reverb: 0, fadeIn: 0, fadeOut: 0 },
            metadata: { type: 'midi', instrument: 'other', icon: 'synth' },
            midiData: { notes: [], duration: 4, tempo: state.bpm || 120 },
          },
        },
      });
      dispatch({ type: 'SELECT_TRACK', payload: { trackId, busId } });
    };

    // Bus selected, but no tracks yet — tailored empty state.
    if (selBus && busVisibleTracks.length === 0) {
      return (
        <div className="wb-canvas">
          <div className="wb-canvas__grid" aria-hidden="true" />
          <div className="wb-empty">
            <div className="wb-empty__status">
              <div className="wb-empty__dot" />
              STATUS · BUS EMPTY
            </div>
            <h1 className="wb-empty__title">{selBus.name || selBus.type || 'Bus'}</h1>
            <p className="wb-empty__body">
              No tracks in this bus yet. Add a MIDI track and the master piano roll will open on it.
            </p>
            <div className="wb-empty__actions">
              <button
                type="button"
                className="wb-btn wb-btn--primary"
                onClick={addTrackToBus}
              >
                <i className="fa-solid fa-plus" /> Add MIDI track to {selBus.name || 'this bus'}
              </button>
            </div>
          </div>
        </div>
      );
    }

    // Bus selected and it has tracks — offer the composite master view
    // (multi-track notes painted in distinct colours via the existing
    // canvas n.color path). One click and the canvas above takes over.
    if (selBus && busVisibleTracks.length > 0) {
      return (
        <div className="wb-canvas">
          <div className="wb-canvas__grid" aria-hidden="true" />
          <div className="wb-empty">
            <div className="wb-empty__status">
              <div className="wb-empty__dot" />
              STATUS · BUS · {busVisibleTracks.length} TRACK{busVisibleTracks.length === 1 ? '' : 'S'}
            </div>
            <h1 className="wb-empty__title">{selBus.name || selBus.type || 'Bus'}</h1>
            <p className="wb-empty__body">
              Open the master piano roll to see every track's notes overlaid in colour, or add a new track.
            </p>
            <div className="wb-empty__actions">
              <button
                type="button"
                className="wb-btn wb-btn--primary"
                onClick={openBusMaster}
              >
                <i className="fa-solid fa-music" /> Open master piano roll
              </button>
              <button
                type="button"
                className="wb-btn"
                onClick={addTrackToBus}
              >
                <i className="fa-solid fa-plus" /> Add MIDI track
              </button>
            </div>
          </div>
        </div>
      );
    }

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
          <div className="wb-empty__actions">
            <button
              type="button"
              className="wb-btn wb-btn--primary"
              onClick={createNewMidiTrackAndBus}
            >
              <i className="fa-solid fa-plus" /> New MIDI track
            </button>
          </div>
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
            {notes.length} notes{selected.size ? ` · ${selected.size} selected` : ''}
          </span>
        </div>
        <div className="sd-midi-tabs" role="tablist" aria-label="MIDI view mode">
          {[
            { id: 'piano',  label: 'Piano',  icon: 'fa-keyboard' },
            { id: 'drum',   label: 'Drum',   icon: 'fa-drum' },
            { id: 'lyrics', label: 'Lyrics', icon: 'fa-microphone' },
            { id: 'score',  label: 'Score',  icon: 'fa-music' },
          ].map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={viewMode === t.id}
              className={`sd-midi-tab ${viewMode === t.id ? 'sd-midi-tab--active' : ''}`}
              onClick={() => setViewMode(t.id)}
              title={`${t.label} view`}
            >
              <i className={`fa-solid ${t.icon}`} />
              <span>{t.label}</span>
            </button>
          ))}
        </div>
        {/* Cursor tools — arrow / block / pen. Hidden in score mode
         * (read-only sheet music). Pen draws an F0 pitch contour over
         * the piano roll; arrow disables note placement so users can
         * marquee/select without accidentally creating notes. */}
        {!isScore && (
          <div className="sd-midi-tools" role="toolbar" aria-label="Cursor tool">
            {[
              { id: 'arrow', icon: 'fa-arrow-pointer', label: 'Arrow', hint: 'Select only' },
              { id: 'block', icon: 'fa-square',        label: 'Block', hint: 'Place + edit notes' },
              { id: 'pen',   icon: 'fa-pen',           label: 'Pen',   hint: 'Draw F0 pitch contour' },
            ].map((t) => (
              <button
                key={t.id}
                className={`sd-midi-tool ${tool === t.id ? 'sd-midi-tool--active' : ''}`}
                onClick={() => setTool(t.id)}
                title={`${t.label} — ${t.hint}`}
                aria-pressed={tool === t.id}
              >
                <i className={`fa-solid ${t.icon}`} />
              </button>
            ))}
          </div>
        )}
        <div className="sd-midi-spacer" />
        <div className="sd-midi-group">
          <span className="sd-midi-kv-k">Zoom</span>
          <button className="sd-midi-btn" onClick={() => setPxPerSec((v) => Math.max(20, v / 1.25))}>−</button>
          <button className="sd-midi-btn" onClick={() => setPxPerSec((v) => Math.min(800, v * 1.25))}>+</button>
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
          <button className="sd-midi-btn" onClick={() => { setScrollX(0); setScrollY(0); setRowZoom(1.25); setPxPerSec(96); }}>Reset view</button>
          {f0Contour.length > 0 && !isScore && (
            <button
              className="sd-midi-btn sd-midi-danger"
              onClick={() => { if (window.confirm('Clear F0 contour?')) commitF0([]); }}
              title={`Clear F0 contour (${f0Contour.length} pts)`}
            >Clear F0</button>
          )}
          <button className="sd-midi-btn sd-midi-danger" onClick={() => {
            if (!selected.size) { if (window.confirm('Clear all notes?')) commit([]); return; }
            const nxt = notes.filter((_, i) => !selected.has(i));
            commit(nxt); setSelected(new Set());
          }}>{selected.size ? `Delete (${selected.size})` : 'Clear'}</button>
        </div>
      </div>

      {/* Canvas area — wheel is attached natively via useEffect above
       * with { passive: false } so ctrl-scroll zoom can preventDefault
       * the page's own scroll. In score mode the canvas + overlays are
       * replaced by a VexFlow render of the same notes; the wrap stays
       * so wheel/zoom state survives switching tabs. */}
      <div
        ref={wrapRef}
        className="sd-midi-canvas-wrap"
        onMouseEnter={() => setHovering(true)}
        onMouseLeave={() => {
          setHoverCell(null);
          setHovering(false);
          cursorPxRef.current = { x: null, y: null };
        }}
      >
        {isScore && (
          <ScoreViewPane
            notes={notes}
            bpm={state.bpm || 120}
          />
        )}
        {!isScore && (
        <canvas
          ref={canvasRef}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMoveCanvas}
          onDoubleClick={onDoubleClick}
          className="sd-midi-canvas"
        />
        )}
        {/* Per-note bend + curve sliders — appear when the selected
         * note spans multiple rows. Vertical slider (right of note):
         * bend amount 0..1. Horizontal slider (under note): curve
         * shape -1 (log) .. +1 (exp), 0 = linear. Together they
         * describe the pitch trajectory over the notes duration:
         *   pitchOffset(t) = bend * (span-1)/2 * (2 * (t/dur)^p - 1)
         *                    where p = 4^curve
         */}
        {!isScore && (() => {
          let idx = -1;
          for (const i of selected) {
            const n = notes[i];
            if (n && (n.pitchSpan || 1) > 1) { idx = i; break; }
          }
          if (idx < 0) return null;
          const n = notes[idx];
          const span = n.pitchSpan || 1;
          const topPitch = n.note + span - 1;
          const x = KEYS_W + (n.time - scrollX) * pxPerSec;
          const y = RULER_H + (maxPitch - topPitch) * rowH - scrollY;
          const w = Math.max(12, n.duration * pxPerSec);
          const h = span * rowH - 2;
          // Skip when offscreen.
          if (x + w < KEYS_W - 40 || x > size.w + 40 || y + h < RULER_H - 40 || y > size.h + 40) return null;
          const bend = n.bend || 0;
          const curve = n.curve || 0;
          const updateNote = (patch) => {
            const nxt = notes.map((m, i) => i === idx ? { ...m, ...patch } : m);
            commit(nxt);
          };
          return (
            <>
              <input
                type="range"
                className="sd-midi-note-bend"
                orient="vertical"
                style={{ left: x + w + 2, top: y, height: h }}
                min={0} max={1} step={0.01}
                value={bend}
                title={`Bend ${(bend * 100).toFixed(0)}%`}
                onChange={(e) => updateNote({ bend: parseFloat(e.target.value) })}
              />
              <input
                type="range"
                className="sd-midi-note-curve"
                style={{ left: x, top: y + h + 2, width: w }}
                min={-1} max={1} step={0.01}
                value={curve}
                title={`Curve ${curve.toFixed(2)} (${curve === 0 ? 'linear' : curve > 0 ? 'exp' : 'log'})`}
                onChange={(e) => updateNote({ curve: parseFloat(e.target.value) })}
              />
            </>
          );
        })()}
        {/* Lyrics overlay — when in `lyrics` mode, every selected note
         * gets a small text input anchored to its on-canvas rect. Typing
         * commits to `n.lyric` on blur (also drawn back into the note
         * body by the canvas paint loop, so the input + canvas always
         * agree). Only shown for the piano/drum/lyrics tabs (canvas-
         * coord based — no canvas, no overlay). */}
        {isLyrics && !isScore && [...selected].slice(0, 16).map((idx) => {
          const n = notes[idx];
          if (!n) return null;
          const span = Math.max(1, n.pitchSpan || 1);
          const topPitch = n.note + span - 1;
          const x = KEYS_W + (n.time - scrollX) * pxPerSec;
          const y = RULER_H + (maxPitch - topPitch) * rowH - scrollY;
          const w = Math.max(40, n.duration * pxPerSec);
          const h = span * rowH - 2;
          if (x + w < KEYS_W || x > size.w || y + h < RULER_H || y > size.h) return null;
          return (
            <input
              key={`lyric-${idx}`}
              className="sd-midi-lyric-input"
              style={{ left: x + 2, top: y + 1, width: w - 4, height: h - 2 }}
              defaultValue={n.lyric || ''}
              placeholder="lyric"
              onMouseDown={(e) => e.stopPropagation()}
              onDoubleClick={(e) => e.stopPropagation()}
              onBlur={(e) => {
                const v = e.target.value;
                if ((n.lyric || '') === v) return;
                const nxt = notes.map((m, i) => i === idx ? { ...m, lyric: v || undefined } : m);
                commit(nxt);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') e.target.blur();
                if (e.key === 'Escape') { e.target.value = n.lyric || ''; e.target.blur(); }
              }}
            />
          );
        })}
        {/* Axis-settings gear — top-left of the canvas. Opens a panel
         * where the user edits X / Y axis expressions. Expression
         * evaluation is a follow-up; for now the fields just store
         * strings and the panel is purely UI. */}
        {!isScore && (
        <button
          className="sd-midi-axis-btn"
          onClick={() => setAxisOpen((v) => !v)}
          title="Axis settings"
        >
          <i className="fa-solid fa-sliders" />
        </button>
        )}
        {!isScore && axisOpen && (
          <div className="sd-midi-axis-panel" onClick={(e) => e.stopPropagation()}>
            <div className="sd-midi-axis-head">
              <span className="sd-midi-kv-k">Axis settings</span>
              <div className="sd-midi-spacer" />
              <button className="sd-midi-btn" onClick={() => setAxisOpen(false)}>Close</button>
            </div>
            <div className="sd-midi-axis-row">
              <label className="sd-midi-kv-k">X axis</label>
              <input
                type="text"
                className="sd-midi-axis-input"
                value={xAxisExpr}
                onChange={(e) => setXAxisExpr(e.target.value)}
                placeholder="30/BPM"
              />
              <span className="sd-midi-axis-hint">units / cell</span>
            </div>
            <div className="sd-midi-axis-row">
              <label className="sd-midi-kv-k">Y axis</label>
              <input
                type="text"
                className="sd-midi-axis-input"
                value={yAxisExpr}
                onChange={(e) => setYAxisExpr(e.target.value)}
                placeholder="2^(1/12)"
              />
              <span className="sd-midi-axis-hint">ratio / row</span>
            </div>
            <div className="sd-midi-axis-row">
              <button
                className="sd-midi-btn"
                onClick={() => { setXAxisExpr('30/BPM'); setYAxisExpr('2^(1/12)'); }}
              >
                Reset defaults
              </button>
            </div>
          </div>
        )}
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

// ScoreViewPane — measures the wrap and renders the shared ScoreView at
// the right size. ScoreView wants beats; our notes carry seconds, so we
// convert with the active bpm. Drum tracks would render as percussion
// hits on the b/4 ledger line — fine for now; a follow-up could swap to
// a one-line drum staff when the user picks a drum track.
function ScoreViewPane({ notes, bpm }) {
  const wrapRef = useRef(null);
  const [size, setSize] = useState({ w: 900, h: 480 });
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => setSize({
      w: Math.max(320, el.clientWidth - 24),
      h: Math.max(240, el.clientHeight - 24),
    });
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, []);
  const beatsPerSec = (bpm || 120) / 60;
  const scoreNotes = useMemo(() => (notes || []).map((n) => ({
    note: n.note,
    time: n.time * beatsPerSec,
    duration: Math.max(0.0625, n.duration * beatsPerSec),
    velocity: n.velocity,
  })), [notes, beatsPerSec]);
  return (
    <div ref={wrapRef} className="sd-midi-score-wrap">
      <ScoreView notes={scoreNotes} tempo={bpm} width={size.w} height={size.h} />
    </div>
  );
}
