/*
 * StudioDev — /studio-dev route. Port of home/newtheme/hifi-purple-studio.html
 * with the full /studio feature set wired in:
 *   • sidebar tabs filter the instrument/source palette
 *   • mode rail (Video / MIDI / Audio / FX) switches the canvas content to
 *     the existing production components (MIDIChart, AudioWaveform, …)
 *   • right track-info sidebar (themed) — gain, pan, M/S, reverb, delete
 *   • browse tile opens the MIDIBrowser in an overlay
 *   • clicking an instrument tile creates a new track of that type and jumps
 *     to MIDI mode
 *   • transport, timeline, tempo all dispatch into the production
 *     AppContext reducer and the shared useAudioPlayback hook, so /studio-dev
 *     drives the same audio engine as /studio
 */
import React, { useCallback, useMemo, useRef, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../../context/AppContext';
import { useAudioPlayback } from '../../hooks/useAudioPlayback';
import { useMetronome } from '../../hooks/useMetronome';
import { useAudioRecorder } from '../../hooks/useAudioRecorder';
import useAutoRepaintMeter from '../../hooks/useAutoRepaintMeter';
import useTrackActions from '../../hooks/useTrackActions';
import {
  separateStemsAuto, iconForType,
  analyzeAudio, analyzeRhythm,
} from '../../services/trackAnalysisAPI';
import { formatTempoMap } from '../../services/tempoMap';
import * as saveService from '../../services/saveService';

// Existing production components reused for mode content + overlays.
// They read/dispatch AppContext directly, so dropping them in Just Works.
import StudioDevMidi from './StudioDevMidi';
import PipelineStatus from './PipelineStatus';
import { logPipeline, clearPipelineLog } from '../../services/pipelineStatus';
import StudioDevMidiBrowser from './StudioDevMidiBrowser';
import StudioDevWaveform from './StudioDevWaveform';
import StudioDevFX from './StudioDevFX';
import StudioDevVideo from './StudioDevVideo';
import StudioDevChords from './StudioDevChords';
import { applyChordChange as polypitchApplyChordChange } from '../../services/polypitchChordSync';
import StudioDevGenerate from './StudioDevGenerate';
import StudioDevChat from './StudioDevChat';
import StudioDevNav from './StudioDevNav';
import StudioDevFileMenu from './StudioDevFileMenu';

import './StudioDev.css';

/* ---------- Icons (ported from hifi-purple-studio.html) ---------- */
const ICON_PATHS = {
  piano:  'M3 8h18v8H3z M7 8v5 M11 8v5 M15 8v5',
  mic:    'M12 2a3 3 0 013 3v5a3 3 0 01-6 0V5a3 3 0 013-3z M5 10a7 7 0 0014 0 M12 17v4',
  drums:  'M4 9l2-2h12l2 2v6l-2 2H6l-2-2z',
  upload: 'M12 14V4 M6 10l6-6 6 6 M4 20h16',
  play:   'M6 4l13 8-13 8z',
  pause:  'M7 4h4v16H7z M13 4h4v16h-4z',
  stop:   'M6 6h12v12H6z',
  rec:    'M12 7a5 5 0 100 10 5 5 0 000-10z',
  wave:   'M2 12h2l2-6 2 12 2-8 2 4 2-2 2 2 2-2 2 2h2',
  plus:   'M12 5v14 M5 12h14',
  gear:   'M12 3l8 4.5v9L12 21l-8-4.5v-9z M12 9a3 3 0 110 6 3 3 0 010-6z',
  search: 'M10 3a7 7 0 105 12l4 4 M10 3a7 7 0 017 7',
  folder: 'M3 6h6l2 2h10v10H3z',
  video:  'M3 6h11v12H3z M14 10l6-3v10l-6-3z',
  midi:   'M4 6v12 M9 6v12 M14 6v12 M19 6v12',
  fx:     'M4 12h4l2-6 2 12 2-8 2 6h4',
  guitar: 'M14 4l4 4 M12 6l6 6 M10 8l6 6 M6 12l-3 3 3 3 3-3 6-6-6-6z',
  synth:  'M3 7h18v10H3z M6 17v2 M10 17v2 M14 17v2 M18 17v2',
  close:  'M6 6l12 12 M18 6l-12 12',
  trash:  'M4 6h16 M8 6V4h8v2 M6 6l1 14h10l1-14',
};
function Icon({ k, size = 16, stroke = 1.4, color = 'currentColor', fill = 'none' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke={color}
         strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round">
      <path d={ICON_PATHS[k]} />
    </svg>
  );
}

/* ---------- Static waveform bars (from spec) ---------- */
const BAR_SIG = [0.10,0.14,0.10,0.30,0.60,0.55,0.20,0.38,0.85,0.70,0.55,0.78,0.45,0.35,0.28,0.22,0.48,0.80,0.95,0.82,0.58,0.38,0.30,0.45,0.62,0.78,0.55,0.38,0.28,0.20,0.30,0.42,0.30,0.22,0.58,0.90,0.82,0.70,0.52,0.40,0.28,0.20,0.20,0.30,0.42,0.50,0.40,0.30,0.22,0.18,0.16,0.28,0.45,0.60,0.72,0.60,0.45,0.35,0.52,0.70,0.82,0.75,0.60,0.45,0.35,0.25,0.20,0.14,0.18,0.26,0.40,0.58,0.70,0.60,0.45,0.30,0.22,0.18,0.14,0.12,0.20,0.32,0.48,0.62,0.58,0.45,0.32,0.22,0.18,0.14,0.16,0.26,0.42,0.58,0.72,0.80,0.78,0.62];
/**
 * Waveform — fixed-density clip-internal visualization.
 *
 * Caller passes `bars` proportional to the clip's duration (≈5 bars/sec,
 * see BARS_PER_SEC below). The SVG still scales with the container via
 * preserveAspectRatio="none", but because bars-count scales with width,
 * each bar stays visually ~the same width when a clip is stretched. The
 * `seed` selects a rotating starting offset in BAR_SIG so two clips of
 * the same length don't look identical.
 */
function Waveform({ height = 16, color = '#fff', seed = 0, bars = 60, bw = 2, gap = 1, opacity = 0.85, silent = false }) {
  const n = Math.max(4, Math.floor(bars));
  // Silent = placeholder track (no audio / MIDI-only). Render a flat
  // minimal bar row — preserves the waveform layout so sizes stay
  // consistent across all clips, but reads as "no audio yet".
  const samples = new Array(n);
  for (let i = 0; i < n; i++) {
    samples[i] = silent ? 0 : BAR_SIG[(seed + i) % BAR_SIG.length];
  }
  const w = n * (bw + gap) - gap;
  const silentOp = silent ? Math.min(opacity, 0.35) : opacity;
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" style={{ display: 'block' }}>
      {samples.map((a, i) => {
        const h = silent ? 1.5 : Math.max(1.5, a * height);
        return <rect key={i} x={i * (bw + gap)} y={(height - h) / 2} width={bw} height={h} rx={bw / 2} fill={color} opacity={silentOp} />;
      })}
    </svg>
  );
}
// Bars-per-second target for clip waveforms. At zoom=1 a bar is ~5 per
// timeline-second; when the clip is stretched (duration ↑) the bar count
// scales with it, so the pixel density remains ≈constant.
const CLIP_BARS_PER_SEC = 5;

/**
 * SummedWaveform — bus master waveform.
 *
 * Given a list of tracks (each with `start`, `end` times and a `seed`
 * offset into BAR_SIG), compute a per-bar composite amplitude across the
 * bus's active range: sum each track's amplitude at every bar, normalized
 * so the peak hits ~1. That's the visual analogue of a stereo mixdown
 * — overlapping tracks at the same moment make the waveform taller.
 */
function SummedWaveform({
  tracks,                // [{ start, end, seed }]
  busStart, busEnd,      // seconds — envelope of rendered bar range
  width100 = 100,        // cosmetic number of bars to render
  height = 16, color = '#fff', opacity = 0.9, bw = 2, gap = 1,
  silent = false,
}) {
  const range = Math.max(0.0001, busEnd - busStart);
  // Cap bars at 600 so a very long / zoomed-in bus doesn't generate
  // thousands of SVG rects — visually indistinguishable past this point.
  // Snap to multiples of 10 so tiny drag movements don't change the bar
  // count and trigger a full SVG rebuild each frame.
  const rawBars = Math.max(20, Math.min(width100, 600));
  const bars = Math.max(20, Math.round(rawBars / 10) * 10);
  // Silent bus (all children are MIDI placeholders / no audioUrl yet) —
  // render a flat minimal bar row at reduced opacity so the aggregate
  // reads as 'no audio' while still showing the bus envelope.
  if (silent) {
    const w = bars * (bw + gap) - gap;
    return (
      <svg width="100%" height={height} viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" style={{ display: 'block' }}>
        {Array.from({ length: bars }).map((_, i) => (
          <rect key={i} x={i * (bw + gap)} y={(height - 1.5) / 2} width={bw} height={1.5} rx={bw / 2} fill={color} opacity={Math.min(opacity, 0.35)} />
        ))}
      </svg>
    );
  }

  // Sample by absolute TIME, not bar index. Each bar b maps to tSec;
  // amplitude = BAR_SIG[(seed + floor(tSec × density)) % len]. This
  // means when a clip drags, its coverage at tSec changes (bar flips
  // from "covered" to "uncovered" or vice versa) but the amplitude of
  // any already-covered bar stays the same. Without this, shifting
  // busStart re-phased every sample index by δ and the whole waveform
  // flickered every mousemove.
  const SAMPLE_DENSITY = 5;   // samples per second — matches CLIP_BARS_PER_SEC
  const sums = new Array(bars).fill(0);
  for (const t of tracks) {
    const seed = (t.seed || 0);
    for (let b = 0; b < bars; b++) {
      const tSec = busStart + (b / bars) * range;
      if (tSec < t.start || tSec > t.end) continue;
      const idx = ((Math.floor(tSec * SAMPLE_DENSITY) + seed) % BAR_SIG.length + BAR_SIG.length) % BAR_SIG.length;
      sums[b] += BAR_SIG[idx];
    }
  }
  const peak = Math.max(...sums, 0.0001);
  const norm = sums.map((v) => v / peak);
  const w = bars * (bw + gap) - gap;
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" style={{ display: 'block' }}>
      {norm.map((a, i) => {
        const h = Math.max(1.5, a * height);
        return <rect key={i} x={i * (bw + gap)} y={(height - h) / 2} width={bw} height={h} rx={bw / 2} fill={color} opacity={opacity} />;
      })}
    </svg>
  );
}
// Memoize so identical prop sets don't re-render. React.memo compares
// shallowly, so `tracks` (the envelopes array) still triggers a re-render
// when its identity changes. Callers should pass a useMemo'd envelopes
// array keyed on track ids + starts/ends.
const SummedWaveformMemo = React.memo(SummedWaveform);

/* ---------- Sidebar palette ----------
 * Matches the production GenerationPanel: each tab has top-level GROUPS;
 * clicking a group reveals its SUBGROUPS. IDs mirror the backend vocabulary
 * (stemphonic_server.py → APPROVED_SUBGROUPS / DRUM_GROUPS / VOCAL_GROUPS)
 * so new tracks land in the right training-program lookup. */
const INSTRUMENT_GROUPS = [
  { id: 'piano',   label: 'Piano',   img: '/assets/icons/piano.png',        type: 'piano' },
  { id: 'guitar',  label: 'Guitar',  img: '/assets/icons/acguitar.png',     type: 'guitar' },
  { id: 'bass',    label: 'Bass',    img: '/assets/icons/elecbass.png',     type: 'bass' },
  { id: 'strings', label: 'Strings', img: '/assets/icons/violin.png',       type: 'strings' },
  { id: 'brass',   label: 'Brass',   img: '/assets/icons/trumpetens.png',   type: 'brass' },
  { id: 'winds',   label: 'Winds',   img: '/assets/icons/sax.png',          type: 'winds' },
];
const INSTRUMENT_SUBGROUPS = {
  piano:   [
    { id: 'acoustic_piano', label: 'Acoustic',  sub: 'Grand',    img: '/assets/icons/piano.png' },
    { id: 'keys',           label: 'Keys',      sub: 'Rhodes',   img: '/assets/icons/keyboard.png' },
  ],
  guitar:  [
    { id: 'acoustic_guitar', label: 'Acoustic', sub: 'Steel',    img: '/assets/icons/acguitar.png' },
    { id: 'electric_guitar', label: 'Electric', sub: 'Strat',    img: '/assets/icons/elecgtr.png' },
  ],
  bass:    [
    { id: 'electric_bass', label: 'Electric',   sub: 'P-Bass',   img: '/assets/icons/elecbass.png' },
    { id: 'upright_bass',  label: 'Upright',    sub: 'Jazz',     img: '/assets/icons/elecbass.png' },
  ],
  strings: [
    { id: 'ensemble_strings', label: 'Ensemble', sub: 'Section', img: '/assets/icons/viollinensemble.png' },
    { id: 'violin',           label: 'Violin',   sub: 'Solo',    img: '/assets/icons/violin.png' },
    { id: 'cello',            label: 'Cello',    sub: 'Solo',    img: '/assets/icons/cello.png' },
  ],
  brass:   [
    { id: 'ensemble_brass', label: 'Ensemble', sub: 'Section',   img: '/assets/icons/trumpetens.png' },
    { id: 'trumpet',        label: 'Trumpet',  sub: 'Bb',        img: '/assets/icons/tpt.png' },
    { id: 'trombone',       label: 'Trombone', sub: 'Tenor',     img: '/assets/icons/tbn.png' },
  ],
  winds:   [
    { id: 'ensemble_winds', label: 'Ensemble', sub: 'Section',   img: '/assets/icons/sax.png' },
    { id: 'flute',          label: 'Flute',    sub: 'Concert',   img: '/assets/icons/flute.png' },
    { id: 'sax',            label: 'Sax',      sub: 'Tenor',     img: '/assets/icons/sax.png' },
  ],
};
const DRUM_GROUPS = [
  { id: 'drum_kit',   label: 'Drum Kit',   img: '/assets/icons/drumkit.png' },
  { id: 'electronic', label: 'Electronic', img: '/assets/icons/elecdrums.png' },
  { id: 'percussion', label: 'Percussion', img: '/assets/icons/drumkit.png' },
];
const VOCAL_GROUPS = [
  { id: 'lead_vox',  label: 'Lead',      img: '/assets/icons/microphone.png' },
  { id: 'bg_vox',    label: 'BGVs',      img: '/assets/icons/microphone.png' },
  { id: 'choir',     label: 'Choir',     img: '/assets/icons/microphone.png' },
  { id: 'synth_vox', label: 'Synth Vox', img: '/assets/icons/microphone.png' },
];
const MODES = [
  { icon: 'video', label: 'Video',   key: 'video' },
  { icon: 'midi',  label: 'MIDI',    key: 'midi'  },
  { icon: 'wave',  label: 'Audio',   key: 'audio' },
  { icon: 'fx',    label: 'FX',      key: 'fx'    },
  { icon: 'wave',  label: 'Routing', key: 'routing' },
  { icon: 'wave',  label: 'Mixer',   key: 'mixer'   },
];

// Track palette — mirrors --wb-track-* in workbench.css exactly so clips,
// waveforms, and swatches all read as one theme. Keys match the lookup
// vocabulary colorFor() uses (instrument type, stemType, or track name).
const TRACK_COLORS = {
  vocals:  '#c94f2c', lead:  '#c94f2c',                       // --wb-track-vox
  rhodes:  '#7a5d3a', piano: '#7a5d3a', keys: '#7a5d3a',       // --wb-track-key
  bass:    '#2f6b4e',                                          // --wb-track-bass
  drums:   '#4a3d6b', drum: '#4a3d6b', kick: '#4a3d6b',        // --wb-track-drum
  strings: '#1d4c7a', guitar: '#1d4c7a', pad: '#1d4c7a', str: '#1d4c7a',  // --wb-track-pad
  other:   '#1d4c7a',
};
function colorFor(type = '') {
  const t = type.toLowerCase();
  for (const [k, v] of Object.entries(TRACK_COLORS)) if (t.includes(k)) return v;
  return TRACK_COLORS.other;
}

function fmtTime(ms) {
  const s = Math.floor(ms / 1000), t = Math.floor((ms % 1000) / 100);
  const mm = String(Math.floor(s / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return `${mm}:${ss}.${t}`;
}

const TIMELINE_SECONDS = 32;

/** Minimal WAV encoder. Converts an AudioBuffer → 16-bit PCM WAV bytes. */
function encodeWavFromBuffer(buffer) {
  const numCh = buffer.numberOfChannels;
  const sr = buffer.sampleRate;
  const nFrames = buffer.length;
  const bytesPerSample = 2;
  const blockAlign = numCh * bytesPerSample;
  const dataSize = nFrames * blockAlign;
  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);
  let o = 0;
  const wStr = (s) => { for (const c of s) view.setUint8(o++, c.charCodeAt(0)); };
  const w32 = (n) => { view.setUint32(o, n, true); o += 4; };
  const w16 = (n) => { view.setUint16(o, n, true); o += 2; };
  wStr('RIFF'); w32(36 + dataSize); wStr('WAVE');
  wStr('fmt '); w32(16); w16(1); w16(numCh);
  w32(sr); w32(sr * blockAlign); w16(blockAlign); w16(16);
  wStr('data'); w32(dataSize);
  // Interleave + convert float→int16
  const chans = Array.from({ length: numCh }, (_, c) => buffer.getChannelData(c));
  for (let i = 0; i < nFrames; i++) {
    for (let c = 0; c < numCh; c++) {
      const s = Math.max(-1, Math.min(1, chans[c][i]));
      view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      o += 2;
    }
  }
  return new Uint8Array(buf);
}

export default function StudioDev() {
  const { state, dispatch } = useApp();
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  const [activeTab, setActiveTab] = useState('Instruments');
  // For the Instruments tab: null → show group tiles, group-id → show its subgroups.
  const [activeInstGroup, setActiveInstGroup] = useState(null);
  // Palette source tab: 'live' shows the default instrument / drum / vocal
  // tree; 'custom' is a user-curated collection (empty placeholder for now
  // — 'Create new' button is wired to nothing).
  const [paletteSource, setPaletteSource] = useState('live');
  // Instrument picked from the left-sidebar palette. It's just a
  // selection marker now — clicking a row no longer creates a track.
  // StudioDevGenerate reads this as the target instrument for stemphonic.
  // Shape: { id, label, group, subgroup, sub } — same keys addInstrumentTrack
  // used to take. Null means generate uses its own fallback.
  const [selectedInstrument, setSelectedInstrument] = useState(null);
  // Palette layout: 'list' (workbench-style dense rows) | 'grid' (icon tiles).
  const [activeMode, setActiveMode] = useState('midi');
  // Which content is showing in the 300px left panel.
  //   'instruments' (default) · 'chat' · 'browse' · 'generate'
  const [sidebarPanel, setSidebarPanel] = useState('instruments');
  const [showChords,  setShowChords]  = useState(true);
  const [navExpanded, setNavExpanded] = useState(false);
  // Loop region is local UI — the production reducer doesn't persist it yet.
  // { start, end } in seconds; null = no loop.
  const [loopRegion, setLoopRegion] = useState(null);
  const [loopEnabled, setLoopEnabled] = useState(false);
  const [loopDrag, setLoopDrag] = useState(null);
  // Layout — resizable widths/heights for the sidebar + timeline split.
  const [sidebarWidth, setSidebarWidth] = useState(260);
  const [timelineHeight, setTimelineHeight] = useState(260);  // resizer defaults low — more canvas room for piano-roll / waveform
  const [sbResizing, setSbResizing] = useState(false);
  const [tlResizing, setTlResizing] = useState(false);
  const [snapMode, setSnapMode] = useState('beat');           // 'off' · 'sixteenth' · 'beat' · 'bar'
  const tapTempoTimes = useRef([]);
  const [colorPickerFor, setColorPickerFor] = useState(null); // trackId whose color picker is open
  const [markers, setMarkers] = useState([]);                 // [{ time, name, color }]
  const [reorderDrag, setReorderDrag] = useState(null);       // { trackId, busId, startY }
  const [meterByTrack, setMeterByTrack] = useState({});       // trackId → 0..1 pseudo-level
  const recorder = useAudioRecorder();
  const clipboardRef = useRef(null);  // for Cmd-C/X/V fallback when reducer copy paths aren't enough

  // Track-level VU meters. The engine doesn't expose RMS per source, so we
  // synthesise a plausible level from gain × masterGain × !muted plus some
  // natural jitter. Updates at 30 fps while playing, falls to zero when
  // stopped. Good enough for at-a-glance "is this track hot?" feedback.
  useEffect(() => {
    if (!state.isPlaying) {
      setMeterByTrack({});
      return;
    }
    const id = setInterval(() => {
      const out = {};
      const hasSolo = (state.buses || []).some((b) => (b.tracks || []).some((t) => t.isSolo));
      for (const bus of state.buses || []) {
        for (const t of bus.tracks || []) {
          if (t.isMuted || (hasSolo && !t.isSolo) || bus.mute) { out[t.id] = 0; continue; }
          const base = (t.gain ?? 1) * (bus.gain ?? 1) * (state.masterGain ?? 0.8);
          const jitter = 0.4 + Math.random() * 0.6;
          out[t.id] = Math.max(0, Math.min(1, base * jitter));
        }
      }
      setMeterByTrack(out);
    }, 66);
    return () => clearInterval(id);
  }, [state.isPlaying, state.buses, state.masterGain]);
  const [autosaveStatus, setAutosaveStatus] = useState(null);
  const [detectingChords, setDetectingChords] = useState(false);
  const [stemSepRunning, setStemSepRunning] = useState(false);
  const [timelineZoom, setTimelineZoom] = useState(1);  // 1 = 32s window, 2 = 16s, 0.5 = 64s
  const [dragClip, setDragClip] = useState(null);        // {trackId, busId, origStart, startPx}
  const [laneRowZoom, setLaneRowZoom] = useState(1);      // Y-axis zoom multiplier for lane row heights

  /* ---------- Real tracks from AppContext (flattened out of buses) ---------- */
  const tracks = useMemo(() => {
    const out = [];
    for (const bus of state.buses || []) {
      for (const t of bus.tracks || []) {
        const type = (t.metadata?.stemType || t.metadata?.instrument || t.name || '').toLowerCase();
        out.push({
          ...t,
          _busId: bus.id,
          _color: colorFor(type),
          _displayName: t.name || t.metadata?.stemType || t.id,
          _type: type,
        });
      }
    }
    return out;
  }, [state.buses]);

  const selectedTrack = state.selectedTrack;
  const selectedBusId = useMemo(() => {
    if (!selectedTrack) return null;
    for (const bus of state.buses || []) {
      if ((bus.tracks || []).some((t) => t.id === selectedTrack.id)) return bus.id;
    }
    return null;
  }, [state.buses, selectedTrack]);

  // Shared backend action wrappers for the selected track (clarify,
  // trumpet-mute, audio→midi, per-chord regen). Same hook /studio's
  // TrackInfoSidebar uses, so both routes mutate AppContext identically.
  const trackActions = useTrackActions({ track: selectedTrack, busId: selectedBusId });

  /* ---------- Audio engine (same hook /studio uses).
       useAudioPlayback expects tracks keyed by bus type
       ({ vo, music, sfx, drums, midi, audio }) with _busId / _bus* fields
       stamped on every track, not a flat array — without this the hook's
       play() finds zero tracks and never starts the RAF playhead loop. */
  const tracksForPlayback = useMemo(() => {
    const out = { vo: [], music: [], sfx: [], drums: [], midi: [], audio: [] };
    for (const bus of state.buses || []) {
      const key = (bus.type || '').toLowerCase();
      const bucket = out[key] || out.music;
      for (const t of bus.tracks || []) {
        bucket.push({
          ...t,
          _busId: bus.id,
          _busGain: bus.gain,
          _busPan: bus.pan,
          _busReverbSend: bus.reverbSend,
          _busMuted: bus.mute,
          _busSolo: bus.solo,
        });
      }
    }
    return out;
  }, [state.buses]);

  const { seek } = useAudioPlayback(
    tracksForPlayback,
    state.isPlaying,
    dispatch,
    state.totalDuration || TIMELINE_SECONDS,
    state.playheadPosition || 0,
    state.bpm || 120,
    state.masterGain ?? 0.8,
    state.beatsPerBar || 4,
    state.meterDenominator || 4,
    state.tempoMap || null,
  );

  /* ---------- Auto-switch mode based on selected track type ---------- */
  useEffect(() => {
    if (!selectedTrack) return;
    const m = selectedTrack.metadata || {};
    if (m.midiData || selectedTrack.midiData || m.type === 'midi') {
      setActiveMode('midi');
    } else if (m.type === 'video_audio' || m.type === 'video') {
      setActiveMode('video');
    } else if (selectedTrack.audioUrl || m.type === 'stem' || m.type === 'audio') {
      setActiveMode('audio');
    }
  }, [selectedTrack?.id]);  // eslint-disable-line react-hooks/exhaustive-deps

  /* ---------- Actions ---------- */
  const togglePlay = useCallback(() => dispatch({ type: 'TOGGLE_PLAY' }), [dispatch]);
  const stopPlay = useCallback(() => {
    dispatch({ type: 'SET_PLAYING', payload: false });
    dispatch({ type: 'RESET_PLAYHEAD' });
    seek?.(0);
  }, [dispatch, seek]);
  const triggerUpload = useCallback(() => fileInputRef.current?.click(), []);
  // Same upload flow /studio's Timeline.js uses:
  //   1. Create bus + parent track immediately (fast UI feedback).
  //   2. In parallel: analyzeAudio (basic-pitch MIDI + PANNs instrument
  //      classification + VAE latent encode) and separateStemsAuto (demucs
  //      stems → whisper lyrics + drum teacher).
  //   3. When analyze returns, patch the parent track's metadata.
  //   4. When stem-sep returns, bulk-add stem child tracks under the same bus
  //      and collapse the bus (so timeline stays tidy).
  // ingestFile runs the full /studio upload pipeline (create bus + parent
  // track, then analyze + stem-separate in parallel). Called from both the
  // hidden <input type=file> change handler AND the timeline drag-drop
  // zone so both entry points do exactly the same thing.
  const ingestFile = useCallback((file) => {
    if (!file) return;
    clearPipelineLog();
    logPipeline('upload', `${file.name} (${(file.size / 1024).toFixed(0)} KB)`);
    const busId = `bus-${Date.now()}`;
    const trackId = `t-${Date.now()}`;
    const baseName = file.name.replace(/\.[^.]+$/, '');
    // ── MIDI transcription cascade (three quality tiers, converging) ──
    //   Tier 1 — master BasicPitch on-device (this IIFE, ~5–10s after upload).
    //            Writes high-quality mix MIDI onto the PARENT track and
    //            fires a first chord pass. The masterNotesPromise below
    //            resolves with this result so downstream tiers can refine.
    //   Tier 2 — per-stem latentPitch (inside the WebGPU pipeline). When
    //            it completes it awaits masterNotesPromise and runs
    //            combineForStems(master, stems) to drop stem false
    //            positives and attribute master notes back to the
    //            right stem. Drums stay on the latentDrumTranscribe path.
    //   Tier 3 — backend BasicPitch per stem (swapInBasicPitch). When
    //            the server finishes a stem, that tier overrides the
    //            refined MIDI on that stem with ground-truth server MIDI
    //            + re-runs chord detection.
    let masterNotesResolve;
    const masterNotesPromise = new Promise((r) => { masterNotesResolve = r; });
    dispatch({
      type: 'CREATE_BUS',
      payload: { id: busId, type: 'INSTRUMENT', name: baseName, expanded: false },
    });
    dispatch({
      type: 'ADD_TRACK',
      payload: {
        busId,
        track: {
          id: trackId, name: baseName,
          audioFile: file, audioUrl: URL.createObjectURL(file),
          duration: 0, startPosition: 0, gain: 1.0, isMuted: false, isSolo: false,
          fx: { reverb: 0, fadeIn: 0, fadeOut: 0 },
          metadata: { type: 'uploaded', originalFilename: file.name },
        },
      },
    });

    // Rich rhythm analysis (per-bar tempoMap + beat_map). Fires in parallel
    // with analyze-audio so the UI sees bar lines + correct meter ASAP.
    logPipeline('rhythm', 'analyzing tempo + meter…');
    analyzeRhythm(file).then((ra) => {
      if (!ra || !Array.isArray(ra.tempoMap) || ra.tempoMap.length === 0) {
        logPipeline('rhythm', 'no tempoMap extracted', 'warn');
        return;
      }
      const first = ra.tempoMap[0];
      logPipeline('rhythm', `${Math.round(ra.bpm)} BPM · ${first.meter[0]}/${first.meter[1]}${ra.grouping ? ' (' + ra.grouping + ')' : ''}`, 'ok');
      dispatch({ type: 'SET_PROJECT_TEMPO_MAP', payload: ra.tempoMap });
      if (ra.beat_map) dispatch({ type: 'SET_BEAT_MAP', payload: ra.beat_map });
      if (typeof ra.downbeat_offset === 'number') {
        dispatch({ type: 'SET_TIMELINE_OFFSET', payload: ra.downbeat_offset });
      }
      if (ra.bpm) dispatch({ type: 'UPDATE_BPM', payload: Math.round(ra.bpm) });
      if (first?.meter) dispatch({ type: 'SET_METER', payload: `${first.meter[0]}/${first.meter[1]}` });
      dispatch({
        type: 'UPDATE_TRACK',
        payload: {
          busId, trackId,
          updates: {
            metadata: {
              tempoMap: ra.tempoMap,
              barStarts: (ra.beat_map || []).filter((b) => b.pos === 1).map((b) => b.t),
              detectedBpm: ra.bpm,
              detectedMeter: ra.beatsPerBar,
              detectedMeterDenominator: ra.meterDenominator,
              detectedGrouping: ra.grouping,
              downbeatOffset: ra.downbeat_offset,
            },
          },
        },
      });
      console.log(`🎶 tempoMap extracted (${ra.tempoMap.length} entries, duration ${ra.duration?.toFixed(2)}s):`);
      console.log(formatTempoMap(ra.tempoMap));
    }).catch((err) => logPipeline('rhythm', `failed: ${err?.message || err}`, 'error'));

    // ── Tier 1: master BasicPitch on-device (in parallel) ─────────────
    // Decodes the file once to stereo@48k (same call the WebGPU pipeline
    // uses below — our toMono22050 will resample inside BasicPitch), mixes
    // to mono, feeds BasicPitch ONNX. When done:
    //   • parent track gets midiData for the midi window
    //   • first-pass chord detection fires (master-only pool)
    //   • masterNotesPromise resolves so the latentPitch refinement (Tier 2)
    //     downstream can call combineForStems(master, stems).
    (async () => {
      try {
        const { audioFileToStereo48k } = await import('../../services/latentEncoder');
        const src = await audioFileToStereo48k(file);
        const N = src.numFrames;
        const mono = new Float32Array(N);
        for (let i = 0; i < N; i++) mono[i] = (src.flat[i] + src.flat[N + i]) * 0.5;
        const { transcribeAudio } = await import('../../services/basicPitchOnnx');
        const t0 = performance.now();
        logPipeline('basicPitch', 'transcribing master audio…');
        const { notes, duration } = await transcribeAudio(mono, 48000);
        if (!notes || notes.length === 0) {
          logPipeline('basicPitch', 'no notes extracted (model disabled or silent audio)', 'warn');
          masterNotesResolve([]);
          return;
        }
        logPipeline('basicPitch', `master: ${notes.length} notes in ${(performance.now() - t0).toFixed(0)}ms`, 'ok');
        const s0 = stateRef.current;
        dispatch({
          type: 'UPDATE_TRACK',
          payload: {
            busId, trackId,
            updates: { metadata: {
              midiData: { notes, duration, tempo: s0.bpm || 120 },
              masterBasicPitch: { notes, duration },
            } },
          },
        });
        // First-pass chord detection from master BasicPitch alone. Master
        // BasicPitch is good enough for chord labels (vs per-stem pool).
        try {
          const { detectChordsFromNotes } = await import('../../services/detectChordsFromMIDI');
          const s = stateRef.current;
          const pool = notes.map((n) => ({
            start: n.time, end: n.time + (n.duration || 0.25),
            pitch: n.note,
          }));
          const beatTimes = (Array.isArray(s.beatMap) && s.beatMap.length > 0)
            ? s.beatMap.map((b) => b.t)
            : (() => {
                const spb = 60 / (s.bpm || 120);
                const nb = Math.ceil(duration / spb) + 1;
                return Array.from({ length: nb }, (_, i) => i * spb);
              })();
          const chords = detectChordsFromNotes(pool, beatTimes);
          if (Object.keys(chords).length > 0) {
            const num = {};
            Object.entries(chords).forEach(([k, v]) => { num[parseInt(k, 10)] = v; });
            dispatch({ type: 'SET_CHORDS', payload: num });
            logPipeline('chords', `${Object.keys(chords).length} chords from master (tier 1)`, 'ok');
          }
        } catch (_) {}
        masterNotesResolve(notes);
      } catch (err) {
        logPipeline('basicPitch', `master failed: ${err?.message || err}`, 'error');
        masterNotesResolve([]);   // unblock downstream refinement
      }
    })();

    analyzeAudio(file).then((res) => {
      const cls = res.classification;
      const midi = res.midi;
      const latent = res.latent;
      const instType = cls?.type || 'other';
      dispatch({
        type: 'UPDATE_TRACK',
        payload: {
          busId, trackId,
          updates: {
            metadata: {
              type: 'uploaded',
              originalFilename: file.name,
              instrument: instType,
              instrumentLabel: cls?.label || null,
              icon: iconForType(instType),
              midi: midi?.midi_url || null,
              latent: latent?.latent_url || null,
              latentId: latent?.latent_id || null,
              inputFiles: midi?.midi_url ? { midiPath: midi.midi_url } : {},
            },
          },
        },
      });
    }).catch((err) => console.warn('analyze failed:', err?.message || err));

    // BasicPitch (server-side, higher quality than LatentPitch) delivers
    // per-stem MIDI incrementally after demucs finishes. Each delivery
    // replaces the latent_pitch placeholder on the matching stem track
    // AND triggers a chord-row re-detection so the ChordTrack upgrades
    // to the cleaner transcription. Debounced so a burst of stem
    // deliveries produces one chord pass.
    const bpMidiByStem = {};
    let chordRerunTimer = null;
    const swapInBasicPitch = async ({ midi_urls }) => {
      const tempo = stateRef.current.bpm || 120;
      for (const [stemName, midiUrl] of Object.entries(midi_urls || {})) {
        try {
          const r = await fetch(midiUrl);
          if (!r.ok) {
            console.warn(`[basicPitch] ${stemName} fetch ${r.status}`);
            logPipeline('basicPitch', `${stemName} fetch ${r.status}`, 'warn');
            continue;
          }
          const ab = await r.arrayBuffer();
          const { Midi } = await import('@tonejs/midi');
          const midi = new Midi(ab);
          const notes = [];
          let duration = 0;
          for (const tr of midi.tracks) {
            for (const n of tr.notes) {
              notes.push({
                note: n.midi,
                time: n.time,
                duration: n.duration,
                velocity: Math.max(1, Math.min(127, Math.round((n.velocity ?? 0.7) * 127))),
              });
              duration = Math.max(duration, n.time + n.duration);
            }
          }
          notes.sort((a, b) => a.time - b.time);
          console.log(`[basicPitch] ${stemName}: ${notes.length} notes (replacing latent_pitch placeholder)`);
          logPipeline('basicPitch', `${stemName}: ${notes.length} notes (tier 3)`, 'ok');
          dispatch({
            type: 'UPDATE_TRACK',
            payload: {
              busId, trackId: `stem-${trackId}-${stemName}`,
              updates: { metadata: { midiData: { notes, duration, tempo } } },
            },
          });
          bpMidiByStem[stemName] = notes;
        } catch (e) {
          console.warn(`[basicPitch] ${stemName} parse failed:`, e?.message || e);
        }
      }
      if (chordRerunTimer) clearTimeout(chordRerunTimer);
      chordRerunTimer = setTimeout(async () => {
        try {
          const { rerunChordDetection } = await import('../../services/detectChordsFromMIDI');
          const s = stateRef.current;
          const chordsNum = rerunChordDetection(bpMidiByStem, {
            beatMap: s.beatMap, bpm: s.bpm || tempo,
          });
          if (Object.keys(chordsNum).length > 0) {
            dispatch({ type: 'SET_CHORDS', payload: chordsNum });
            console.log(`[basicPitch] chord row rebuilt from ${Object.keys(bpMidiByStem).length} upgraded stem(s): ${Object.keys(chordsNum).length} chord changes`);
            logPipeline('chords', `${Object.keys(chordsNum).length} chords from basicPitch stems (tier 3)`, 'ok');
          }
        } catch (err) {
          console.warn('[basicPitch] chord rerun failed:', err?.message || err);
          logPipeline('chords', `tier-3 rerun failed: ${err?.message || 'error'}`, 'warn');
        }
      }, 400);
    };

    logPipeline('separate', 'posting to backend demucs…');
    separateStemsAuto(file, {
      onMidiReady: swapInBasicPitch,
      onDrumTeacher: ({ drum_substem_urls, drum_substem_onsets, drum_substem_onset_strengths }) => {
        logPipeline('drumTeacher', `kick/snare/hh/toms/ride/crash ready`, 'ok');
        // Attach per-substem WAV URLs + onset times + per-onset strengths
        // to the drums STEM TRACK's metadata. virtualTrackEdit uses the
        // strengths to weight accent vs ghost when re-quantizing triplets,
        // and the onsets to do per-substem hit-snap on the new meter grid.
        // Sustain substems (hh/ride/crash) stay on bar-rearrange. All mix
        // through one shared per-track gain so solo/mute keeps working.
        const names = Object.keys(drum_substem_urls || {});
        if (!names.length) return;
        console.log('[studio-dev] drum teacher ready:', names);
        dispatch({
          type: 'UPDATE_TRACK',
          payload: {
            busId, trackId: `stem-${trackId}-drums`,
            updates: { metadata: {
              drumSubstems: drum_substem_urls,
              drumSubstemOnsets: drum_substem_onsets || {},
              drumSubstemOnsetStrengths: drum_substem_onset_strengths || {},
            } },
          },
        });
      },
      onLyrics: ({ vocals_lyrics, vocals_lyrics_language }) => {
        // Word-level timing [{word, start, end, probability}] is required
        // by virtualTrackEdit.buildVocalProtectedSchedule to avoid cutting
        // a word during meter changes. Attach to the vocals STEM TRACK so
        // the schedule builder can read it directly from track.metadata.
        if (!Array.isArray(vocals_lyrics) || vocals_lyrics.length === 0) return;
        console.log(`[studio-dev] whisper: ${vocals_lyrics.length} words (${vocals_lyrics_language})`);
        dispatch({
          type: 'UPDATE_TRACK',
          payload: {
            busId, trackId: `stem-${trackId}-vocals`,
            updates: { metadata: {
              vocalsLyrics: vocals_lyrics,
              vocalsLyricsLanguage: vocals_lyrics_language || null,
            } },
          },
        });
      },
    }).then((sep) => {
      if (!sep?.stems) {
        logPipeline('separate', 'no stems returned', 'warn');
        return;
      }
      logPipeline('separate', `${Object.keys(sep.stems).length} stems ready`, 'ok');
      const stemOnsets = sep.stem_onsets || {};
      const stemTracks = Object.entries(sep.stems).map(([stemName, audioUrl]) => ({
        id: `stem-${trackId}-${stemName}`,
        name: `${baseName} — ${stemName}`,
        audioUrl, duration: 0, startPosition: 0,
        gain: 1.0, isMuted: false, isSolo: false, cropStart: 0, cropEnd: 0,
        fx: { reverb: 0, fadeIn: 0.2, fadeOut: 1.0 },
        metadata: {
          type: 'stem', stemType: stemName,
          parentTrackId: trackId, instrument: stemName,
          icon: iconForType(stemName),
          // Per-stem librosa onsets — consumed by
          // virtualTrackEdit.buildMelodicProtectedSchedule to gate bar
          // drops around sustained notes. Empty array = silent/no hits.
          stemOnsets: Array.isArray(stemOnsets[stemName]) ? stemOnsets[stemName] : [],
        },
      }));
      dispatch({ type: 'ADD_TRACKS_BULK', payload: { busId, tracks: stemTracks } });
      // Promote the parent uploaded-mix track to the bus MASTER: it
      // keeps all its audio + analysis metadata (lyrics, latents, MIDI)
      // so playback + regen still have it, but the render code filters
      // isBusMaster tracks out of the per-track list — so users see 6
      // stems under the bus header instead of 6+1. The master's
      // waveform already lives in the bus summary lane (driven by the
      // stems composite), matching user intent.
      dispatch({
        type: 'UPDATE_TRACK',
        payload: {
          busId, trackId,
          updates: { metadata: { isBusMaster: true } },
        },
      });
      dispatch({ type: 'SET_BUS_EXPANDED', payload: { busId, expanded: false } });

      // -- WEBGPU LATENT EXTRACTION + MIDI TRANSCRIPTION --------------
      // Parallels /studio (DAWOptimized): decode the master audio once,
      // run sem4Decoder's streamPreviewSeparation to get per-stem oobleck
      // latents [64, T], then run latentPitch on pitched stems and
      // latentDrumTranscribe on the drum stem. Each stem gets a latent
      // reference + a MIDI note list attached to its metadata, which
      // the piano roll picks up automatically.
      (async () => {
        try {
          logPipeline('webgpu', 'loading demucs latents…');
          const { audioFileToStereo48k } = await import('../../services/latentEncoder');
          const { streamPreviewSeparation, SEM4_LATENT_CHANS } = await import('../../services/sem4Decoder');
          const src = await audioFileToStereo48k(file);
          await streamPreviewSeparation(src.flat, src.numFrames, {
            onAllLatentsReady: async ({ stemLatents, stemNames, fps }) => {
              logPipeline('webgpu', `${stemNames.length} stem latents ready`, 'ok');
              for (let i = 0; i < stemNames.length; i++) {
                const stemName = stemNames[i];
                dispatch({
                  type: 'UPDATE_TRACK',
                  payload: {
                    busId, trackId: `stem-${trackId}-${stemName}`,
                    updates: { metadata: {
                      latent: stemLatents[i],
                      latentChans: SEM4_LATENT_CHANS,
                      latentFps: fps,
                    } },
                  },
                });
              }
              console.log(`[sem4Decoder] cached ${stemNames.length} stem latents on track metadata`);

              const tempo = state.bpm || 120;
              const T = Math.floor(stemLatents[0].length / SEM4_LATENT_CHANS);
              const pitchedIdx = [];
              const pitchedLatents = [];
              for (let i = 0; i < stemNames.length; i++) {
                if (stemNames[i] !== 'drums') {
                  pitchedIdx.push(i);
                  pitchedLatents.push(stemLatents[i]);
                }
              }

              const pitchedPromise = (async () => {
                if (!pitchedLatents.length) return [];
                const { extractPitchFromLatentsBatch } = await import('../../services/latentPitch');
                const t0 = performance.now();
                const results = await extractPitchFromLatentsBatch(pitchedLatents, T);
                console.log(`[latentPitch] batched ${pitchedLatents.length} pitched stems in ${(performance.now() - t0).toFixed(0)}ms`);
                return results;
              })();

              const drumsIdx = stemNames.indexOf('drums');
              const drumsPromise = (async () => {
                if (drumsIdx < 0) return null;
                const { extractDrumMIDI } = await import('../../services/latentDrumTranscribe');
                const t0 = performance.now();
                const out = await extractDrumMIDI(stemLatents[drumsIdx], T);
                const dt = (performance.now() - t0).toFixed(0);
                console.log(`[latentDrumTranscribe] drums: ${out.notes.length} hits in ${dt}ms`);
                logPipeline('drums', `${out.notes.length} drum hits (${dt}ms)`, 'ok');
                return out;
              })();

              const [pitchedResults, drumResult] = await Promise.all([pitchedPromise, drumsPromise]);

              // ── Tier 2: refine per-stem latentPitch with master BasicPitch ──
              // Wait for master BasicPitch (may have already resolved). If
              // it returned empty (disabled / failed), fall through and use
              // raw latentPitch for each stem. Drums stay on the
              // latentDrumTranscribe path — BasicPitch doesn't do drums.
              let masterNotes = [];
              try { masterNotes = await masterNotesPromise; } catch (_) { masterNotes = []; }
              const stemNotesByName = {};
              for (let k = 0; k < pitchedIdx.length; k++) {
                const stemName = stemNames[pitchedIdx[k]];
                stemNotesByName[stemName] = pitchedResults[k].notes;
              }
              let refinedByStem = stemNotesByName;
              if (masterNotes.length > 0) {
                try {
                  const { combineForStems } = await import('../../services/midiRefine');
                  refinedByStem = combineForStems(masterNotes, stemNotesByName);
                  const before = Object.values(stemNotesByName).reduce((s, a) => s + a.length, 0);
                  const after = Object.values(refinedByStem).reduce((s, a) => s + a.length, 0);
                  console.log(`[basicPitch] refined ${Object.keys(stemNotesByName).length} stems with master: ${before} → ${after} notes`);
                  logPipeline('refine', `${Object.keys(stemNotesByName).length} stems: ${before} → ${after} notes`, 'ok');
                } catch (err) {
                  console.warn('[basicPitch] refinement failed, using raw latentPitch:', err?.message || err);
                  logPipeline('refine', `fallback to raw latentPitch (${err?.message || 'error'})`, 'warn');
                  refinedByStem = stemNotesByName;
                }
              }
              for (let k = 0; k < pitchedIdx.length; k++) {
                const i = pitchedIdx[k];
                const stemName = stemNames[i];
                const { duration } = pitchedResults[k];
                const notes = refinedByStem[stemName] || pitchedResults[k].notes;
                dispatch({
                  type: 'UPDATE_TRACK',
                  payload: {
                    busId, trackId: `stem-${trackId}-${stemName}`,
                    updates: { metadata: { midiData: { notes, duration, tempo } } },
                  },
                });
              }
              if (drumResult && drumsIdx >= 0) {
                dispatch({
                  type: 'UPDATE_TRACK',
                  payload: {
                    busId, trackId: `stem-${trackId}-drums`,
                    updates: { metadata: { midiData: { notes: drumResult.notes, duration: drumResult.duration, tempo } } },
                  },
                });
              }
              console.log(`[latentPitch] applied MIDI to all ${stemNames.length} stems`);

              // Chord pass from REFINED stems (supersedes the master-only
              // first pass). Same pool-then-detect pipeline; the notes
              // are now timing-snapped + false-positive-filtered.
              try {
                const { rerunChordDetection } = await import('../../services/detectChordsFromMIDI');
                const s = stateRef.current;
                const chordsNum = rerunChordDetection(refinedByStem, {
                  beatMap: s.beatMap, bpm: s.bpm || tempo,
                });
                if (Object.keys(chordsNum).length > 0) {
                  dispatch({ type: 'SET_CHORDS', payload: chordsNum });
                  console.log(`[chords] Tier-2 refined-stem pass: ${Object.keys(chordsNum).length} chord changes`);
                  logPipeline('chords', `${Object.keys(chordsNum).length} chords from refined stems (tier 2)`, 'ok');
                }
              } catch (err) {
                console.warn('[chords] Tier-2 pass failed:', err?.message || err);
                logPipeline('chords', `tier-2 pass failed: ${err?.message || 'error'}`, 'warn');
              }
            },
          });
        } catch (err) {
          console.warn('[webgpu-pipeline] failed (non-fatal):', err?.message || err);
        }
      })();
    }).catch((err) => console.warn('stem-sep failed:', err?.message || err));
  }, [dispatch, state.bpm]);

  const onFilePick = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';
    ingestFile(file);
  }, [ingestFile]);

  // Drag-and-drop on the timeline (empty state or over existing lanes).
  // Matches the original /studio behavior — drop an audio file anywhere
  // on the DAW and it gets ingested + auto-separated.
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const onTimelineDragOver = useCallback((e) => {
    if (!e.dataTransfer?.types?.includes('Files')) return;
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingFile(true);
  }, []);
  const onTimelineDragLeave = useCallback((e) => {
    if (e.currentTarget.contains(e.relatedTarget)) return;
    setIsDraggingFile(false);
  }, []);
  const onTimelineDrop = useCallback((e) => {
    if (!e.dataTransfer?.files?.length) return;
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingFile(false);
    const file = e.dataTransfer.files[0];
    if (!file.type.startsWith('audio/') && !/\.(mid|midi|wav|mp3|ogg|flac|aac|m4a)$/i.test(file.name)) return;
    ingestFile(file);
  }, [ingestFile]);

  const addInstrumentTrack = useCallback((inst) => {
    const busId = `bus-${Date.now()}`;
    const trackId = `t-${Date.now()}`;
    const busType = (inst.group === 'vocals') ? 'VO'
                  : (inst.group === 'drums')  ? 'DRUMS' : 'INSTRUMENT';
    dispatch({
      type: 'CREATE_BUS',
      payload: { id: busId, type: busType, name: inst.label, expanded: false },
    });
    const newTrack = {
      id: trackId, name: inst.label, duration: 4, startPosition: 0,
      gain: 1.0, isMuted: false, isSolo: false,
      fx: { reverb: 0, fadeIn: 0, fadeOut: 0 },
      metadata: {
        type: 'midi',
        instrument: inst.type,
        instrumentGroup: inst.group,
        instrumentSubgroup: inst.subgroup,
        icon: inst.icon,
      },
      midiData: { notes: [], duration: 4, tempo: state.bpm || 120 },
    };
    dispatch({ type: 'ADD_TRACK', payload: { busId, track: newTrack } });
    dispatch({ type: 'SELECT_TRACK', payload: { trackId, busId } });
    setActiveMode('midi');
  }, [dispatch, state.bpm]);

  const addEmptyTrack = useCallback(() => {
    addInstrumentTrack({ label: 'New track', type: 'other', icon: 'synth' });
  }, [addInstrumentTrack]);

  const toggleMute = (trackId, busId) => dispatch({ type: 'TOGGLE_TRACK_MUTE', payload: { trackId, busId } });
  const toggleSolo = (trackId, busId) => dispatch({ type: 'TOGGLE_TRACK_SOLO', payload: { trackId, busId } });
  const setBpm = (bpm) => dispatch({ type: 'UPDATE_BPM', payload: Math.max(40, Math.min(240, bpm)) });

  // Tap-tempo: record click timestamps, average the interval across the
  // last N taps, convert → BPM. Resets if more than 2 s between taps.
  const tapTempo = useCallback(() => {
    const now = performance.now();
    const prev = tapTempoTimes.current;
    if (prev.length && now - prev[prev.length - 1] > 2000) tapTempoTimes.current = [];
    tapTempoTimes.current.push(now);
    const ts = tapTempoTimes.current.slice(-4);
    if (ts.length < 2) return;
    const intervals = [];
    for (let i = 1; i < ts.length; i++) intervals.push(ts[i] - ts[i - 1]);
    const avg = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const bpm = Math.round(60_000 / avg);
    if (bpm >= 40 && bpm <= 240) dispatch({ type: 'UPDATE_BPM', payload: bpm });
  }, [dispatch]);

  // Snap-quantize helper — used by clip-drag + MIDI commits.
  const snapSec = useCallback((t) => {
    if (snapMode === 'off') return t;
    const beat = 60 / (state.bpm || 120);
    const step = snapMode === 'bar'       ? beat * (state.beatsPerBar || 4)
              : snapMode === 'sixteenth' ? beat / 4
              : beat;
    return Math.round(t / step) * step;
  }, [snapMode, state.bpm, state.beatsPerBar]);
  const selectTrack = (trackId, busId) => dispatch({ type: 'SELECT_TRACK', payload: { trackId, busId } });
  const setTrackGain = (trackId, busId, gain) => dispatch({ type: 'UPDATE_TRACK_GAIN', payload: { trackId, busId, gain } });
  const setTrackPan  = (trackId, busId, pan)  => dispatch({ type: 'UPDATE_TRACK_PAN',  payload: { trackId, busId, pan } });
  const setTrackReverb = (trackId, busId, reverb) => dispatch({ type: 'UPDATE_TRACK_REVERB', payload: { trackId, busId, reverb } });
  const deleteTrack = (trackId, busId) => dispatch({ type: 'REMOVE_TRACK', payload: { trackId, busId } });
  const toggleMetronome = () => dispatch({ type: 'TOGGLE_METRONOME' });
  const setMasterGain = (g) => dispatch({ type: 'UPDATE_MASTER_GAIN', payload: Math.max(0, Math.min(1, g)) });
  const undo = useCallback(() => dispatch({ type: 'UNDO' }), [dispatch]);
  const redo = useCallback(() => dispatch({ type: 'REDO' }), [dispatch]);
  const toggleCinema = useCallback(() => dispatch({ type: 'TOGGLE_CINEMA_MODE' }), [dispatch]);
  const setTimeSig = useCallback((sig) => {
    // AppContext reducer handles SET_METER ("N/D" string), not SET_TIME_SIGNATURE
    // — the previous SET_TIME_SIGNATURE dispatch silently no-op'd, so /studio's
    // meter select never moved state. With virtual-edit playback, this dispatch
    // is also what makes meter change instant: useAudioPlayback watches
    // state.beatsPerBar/meterDenominator and live-reschedules in place.
    if (!sig) return;
    const [n, d] = sig.split('/').map((v) => parseInt(v, 10));
    if (!n || !d) return;
    dispatch({ type: 'SET_METER', payload: sig });
  }, [dispatch]);

  // Wire real metronome audio — the hook schedules WebAudio clicks against
  // the transport clock. `state.isMetronomeOn` is the reducer's key name
  // (not `metronomeEnabled` as I was reading earlier).
  useMetronome(
    state.isPlaying,
    state.playheadPosition || 0,
    !!state.isMetronomeOn,
    state.bpm || 120,
    null, null,
    state.beatsPerBar || 4,
    state.meterDenominator || 4,
  );

  // Auto-repaint: when bpm/meter changes, re-stemphonic every track with a
  // cached VAE latent. Shared hook — /studio's TempoControls calls the same
  // one, so both routes stay in lockstep.
  useAutoRepaintMeter();

  // Prewarm ALL WebGPU/WASM ORT sessions so none lazy-loads during the
  // first file drop. Matches the /studio (DAWOptimized.js) boot block —
  // order smallest → largest so any failure short-circuits cheaper work
  // first. Each model owns its own init log (e.g. `[latentPitch] model
  // loaded…`, `[latentVisual] ready`) so the console shows each as it
  // warms up.
  useEffect(() => {
    fetch('/health').catch(() => {});
    (async () => {
      try {
        const { initRmsDemucs } = await import('../../services/rmsDemucs');
        await initRmsDemucs();
      } catch (_) {}
      try {
        const { initSem4Decoder } = await import('../../services/sem4Decoder');
        await initSem4Decoder();
      } catch (_) {}
      try {
        const { initLatentEncoder } = await import('../../services/latentEncoder');
        await initLatentEncoder();
      } catch (_) {}
      try {
        const { initLatentPitch } = await import('../../services/latentPitch');
        await initLatentPitch();
      } catch (_) {}
      try {
        const { initDrumSep } = await import('../../services/latentDrumSep');
        await initDrumSep();
      } catch (_) {}
      try {
        const { initLatentVisual } = await import('../../services/latentVisual');
        await initLatentVisual();
      } catch (_) {}
    })();
    console.log('[prewarm] studio opened — warming rmsDemucs + sem4Decoder + latentEncoder + latentPitch + latentDrumSep + latentVisual');
  }, []);

  // Chord-row diff: when a user edits a chord cell (or a re-detection
  // pass rewrites the map), find what changed vs. the previous render and
  // drive polypitch to pitch-shift the affected stem audio so it matches.
  // Tier-1/2/3 detection passes ALSO hit SET_CHORDS but tend to replace
  // wholesale — we only resynth for single-cell changes (user edits).
  const prevChordsRef = useRef({});
  useEffect(() => {
    const prev = prevChordsRef.current || {};
    const next = state.chordTrack?.chords || {};
    prevChordsRef.current = next;

    const changedKeys = [];
    const seen = new Set();
    for (const k of Object.keys(next)) {
      seen.add(k);
      if (prev[k] !== next[k]) changedKeys.push(k);
    }
    for (const k of Object.keys(prev)) {
      if (!seen.has(k) && prev[k]) changedKeys.push(k);
    }
    if (changedKeys.length === 0) return;

    // Bulk rewrites (detection passes) touch tens of cells at once; skip
    // those, only resynth on single-cell user edits. If we later want to
    // re-voice on full-song key changes, lift this threshold.
    if (changedKeys.length > 3) return;

    const pitchedStemTracks = [];
    for (const bus of state.buses || []) {
      for (const tr of bus.tracks || []) {
        const isDrum = tr.metadata?.stemType === 'drums' || tr.metadata?.instrumentGroup === 'drums';
        if (isDrum) continue;
        if (!tr.audioUrl || !tr.metadata?.midiData?.notes?.length) continue;
        pitchedStemTracks.push({ ...tr, busId: bus.id, midiData: tr.metadata.midiData });
      }
    }
    if (pitchedStemTracks.length === 0) return;

    for (const k of changedKeys) {
      const beatIndex = parseInt(k, 10);
      polypitchApplyChordChange({
        beatIndex,
        oldChord: prev[k] || null,
        newChord: next[k] || null,
        pitchedStemTracks,
        tempo: { beatMap: state.beatMap, bpm: state.bpm || 120 },
        onTrackAudioReady: (trackId, busId, newUrl) => {
          dispatch({
            type: 'UPDATE_TRACK',
            payload: { busId, trackId, updates: { audioUrl: newUrl } },
          });
        },
      });
    }
  }, [state.chordTrack?.chords, state.buses, state.beatMap, state.bpm, dispatch]);

  // Autosave loop: quickSave every 8 s. We read state out of a ref so the
  // interval is installed once and doesn't get torn down on every mutation
  // (which would prevent it from ever firing).
  const stateRef = useRef(state);
  useEffect(() => { stateRef.current = state; }, [state]);
  useEffect(() => {
    const id = setInterval(async () => {
      const s = stateRef.current;
      const name = s.projectName || 'Untitled Session';
      try {
        setAutosaveStatus('saving');
        const r = await saveService.quickSave(name, s);
        setAutosaveStatus(r?.success ? 'saved' : 'failed');
      } catch (_) { setAutosaveStatus('failed'); }
    }, 8000);
    return () => clearInterval(id);
  }, []);

  // Keyboard shortcuts. Cmd-S is owned by StudioDevFileMenu.
  //
  // Critical: effect deps are stable ([dispatch, undo, redo]) so the
  // listener mounts once and stays. Previously this effect listed
  // state.playheadPosition, which changes ~60 fps during playback —
  // tearing down + re-adding the listener every frame meant space
  // presses that landed inside the cleanup window were dropped, so
  // play/pause felt unreliable. All mutable state used by the handler
  // (selectedTrack, selectedBusId, playheadPosition) now comes from a
  // ref that always points at the latest value.
  const keyDepsRef = useRef({ selectedTrack, selectedBusId, playheadPosition: state.playheadPosition });
  useEffect(() => {
    keyDepsRef.current = { selectedTrack, selectedBusId, playheadPosition: state.playheadPosition };
  }, [selectedTrack, selectedBusId, state.playheadPosition]);

  useEffect(() => {
    const onKey = (e) => {
      // Ignore when typing in an input / textarea / contenteditable.
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
      const { selectedTrack: sel, selectedBusId: selBus, playheadPosition: pp } = keyDepsRef.current;
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) redo(); else undo();
      } else if (mod && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        redo();
      } else if (e.code === 'Space') {
        // preventDefault cancels the keyup-click the browser would
        // otherwise fire on whichever button is currently focused
        // (e.g. the transport play button right after a click). Without
        // this, space would fire toggle-play twice per press.
        e.preventDefault();
        dispatch({ type: 'TOGGLE_PLAY' });
      } else if (mod && e.key.toLowerCase() === 'd') {
        e.preventDefault();
        duplicateSelectedRef.current?.();
      } else if (mod && e.key.toLowerCase() === 'c') {
        if (!sel) return;
        e.preventDefault();
        dispatch({ type: 'COPY_TRACK', payload: { track: sel } });
      } else if (mod && e.key.toLowerCase() === 'x') {
        if (!sel || !selBus) return;
        e.preventDefault();
        dispatch({ type: 'COPY_TRACK', payload: { track: sel } });
        dispatch({ type: 'REMOVE_TRACK', payload: { trackId: sel.id, busId: selBus } });
      } else if (mod && e.key.toLowerCase() === 'v') {
        if (!selBus) return;
        e.preventDefault();
        dispatch({
          type: 'PASTE_TRACK',
          payload: { targetBusId: selBus, playheadPosition: pp || 0 },
        });
      } else if (!mod && e.key >= '1' && e.key <= '4') {
        // Mode-rail quick switch: 1=Video, 2=MIDI, 3=Audio, 4=FX
        const modeMap = { '1': 'video', '2': 'midi', '3': 'audio', '4': 'fx' };
        setActiveMode(modeMap[e.key]);
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (sel && selBus && !e.repeat) {
          deleteTrack(sel.id, selBus);
        }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dispatch, undo, redo]);

  // Keep the ref pointing at the current duplicateSelected callback so
  // the keyboard effect (which runs before it in source order) can invoke
  // it without needing it in the dep array (would cause a TDZ error).
  const duplicateSelectedRef = useRef(null);

  // Auto-stop-on-loop-end. If loopEnabled and playhead crosses loopEnd,
  // snap back to loopStart without pausing.
  useEffect(() => {
    if (!loopEnabled || !loopRegion || !state.isPlaying) return;
    if ((state.playheadPosition || 0) >= loopRegion.end) {
      dispatch({ type: 'SEEK_TO', payload: loopRegion.start });
      seek?.(loopRegion.start);
    }
  }, [state.playheadPosition, loopEnabled, loopRegion, state.isPlaying, dispatch, seek]);

  // Detect-chords action on selected track. Pools MIDI from the selected
  // bus's non-drum stems (already produced by the latent-pitch pass on
  // upload) and runs the client-side symbolic chord detector. Zero
  // backend — the old /api/detect-chords chord path is 503-gated and
  // was replaced by detectChordsFromMIDI.
  const runDetectChords = useCallback(async () => {
    if (!selectedBusId) return;
    setDetectingChords(true);
    try {
      const bus = state.buses.find((b) => b.id === selectedBusId);
      const stemsWithMidi = (bus?.tracks || []).filter((t) => t?.metadata?.midiData?.notes?.length);
      if (stemsWithMidi.length === 0) {
        console.warn('[studio-dev] detect-chords: no stem MIDI available yet — wait for latent-pitch pass to finish');
        return;
      }
      const { poolMidiFromStems, detectChordsFromNotes } = await import('../../services/detectChordsFromMIDI');
      const notes = poolMidiFromStems(stemsWithMidi);
      if (notes.length === 0) {
        console.warn('[studio-dev] detect-chords: no pitched notes pooled (all drum stems?)');
        return;
      }
      const bpm = state.bpm || 120;
      const spb = 60 / bpm;
      const beatMap = state.beatMap;
      let beatTimes;
      if (Array.isArray(beatMap) && beatMap.length > 0) {
        beatTimes = beatMap.map((b) => b.t);
      } else {
        const maxEnd = notes.reduce((m, n) => Math.max(m, n.end), 0);
        const nBeats = Math.ceil(maxEnd / spb) + 1;
        beatTimes = Array.from({ length: nBeats }, (_, i) => i * spb);
      }
      const chords = detectChordsFromNotes(notes, beatTimes);
      const chordsNum = {};
      Object.entries(chords).forEach(([k, v]) => { chordsNum[parseInt(k, 10)] = v; });
      dispatch({ type: 'SET_CHORDS', payload: chordsNum });
      console.log(`[studio-dev] detect-chords: ${Object.keys(chords).length} chord changes from ${stemsWithMidi.length} stem(s)`);
    } catch (e) {
      console.warn('[studio-dev] detect-chords failed:', e?.message || e);
    } finally {
      setDetectingChords(false);
    }
  }, [selectedBusId, state.buses, state.bpm, state.beatMap, dispatch]);

  // Stem-separate the selected (existing) track.
  const runStemSep = useCallback(async () => {
    if (!selectedTrack?.audioFile && !selectedTrack?.audioUrl) return;
    setStemSepRunning(true);
    try {
      const file = selectedTrack.audioFile instanceof File
        ? selectedTrack.audioFile
        : await (async () => {
            const r = await fetch(selectedTrack.audioUrl);
            const b = await r.blob();
            return new File([b], selectedTrack.name || 'track.wav', { type: b.type || 'audio/wav' });
          })();
      const sep = await separateStemsAuto(file);
      if (!sep?.stems) return;
      // Put new stem tracks in the same bus as the source track.
      const busId = selectedBusId;
      const stemTracks = Object.entries(sep.stems).map(([stemName, audioUrl]) => ({
        id: `stem-${selectedTrack.id}-${stemName}`,
        name: stemName,
        audioUrl, duration: 0, startPosition: 0,
        gain: 1.0, isMuted: false, isSolo: false,
        fx: { reverb: 0, fadeIn: 0, fadeOut: 0 },
        metadata: { type: 'stem', stemType: stemName, parentTrackId: selectedTrack.id, instrument: stemName, icon: iconForType(stemName) },
      }));
      dispatch({ type: 'ADD_TRACKS_BULK', payload: { busId, tracks: stemTracks } });
    } catch (e) {
      console.warn('[studio-dev] stem-sep failed:', e?.message || e);
    } finally {
      setStemSepRunning(false);
    }
  }, [selectedTrack, selectedBusId, dispatch]);

  // Duplicate currently-selected track via COPY_TRACK + PASTE_TRACK.
  const duplicateSelected = useCallback(() => {
    if (!selectedTrack || !selectedBusId) return;
    dispatch({ type: 'COPY_TRACK', payload: { track: selectedTrack } });
    dispatch({
      type: 'PASTE_TRACK',
      payload: {
        targetBusId: selectedBusId,
        playheadPosition: (selectedTrack.startPosition || 0) + (selectedTrack.duration || 4) + 0.1,
      },
    });
  }, [selectedTrack, selectedBusId, dispatch]);
  useEffect(() => { duplicateSelectedRef.current = duplicateSelected; }, [duplicateSelected]);

  // Version cycling — walks metadata.versions, swapping audioUrl/midiData
  // onto the track each time. Keeps the original version at index 0.
  const cycleVersion = useCallback((dir) => {
    if (!selectedTrack || !selectedBusId) return;
    const versions = selectedTrack.metadata?.versions || [];
    if (!versions.length) return;
    const curIdx = selectedTrack.metadata?.currentVersionIndex ?? 0;
    const nextIdx = (curIdx + dir + versions.length) % versions.length;
    const v = versions[nextIdx] || {};
    dispatch({
      type: 'UPDATE_TRACK_PROPS',
      payload: {
        trackId: selectedTrack.id,
        audioUrl: v.audioUrl || selectedTrack.audioUrl,
        duration: v.duration ?? selectedTrack.duration,
        midiData: v.midiData || selectedTrack.midiData,
        metadata: { ...(selectedTrack.metadata || {}), currentVersionIndex: nextIdx },
      },
    });
  }, [selectedTrack, selectedBusId, dispatch]);

  const submitFeedback = useCallback(async (rating) => {
    if (!selectedTrack) return;
    try {
      await fetch('/api/feedback', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          trackId: selectedTrack.id,
          projectName: state.projectName,
          rating,   // 'like' | 'dislike'
          timestamp: Date.now(),
        }),
      });
      dispatch({
        type: 'UPDATE_TRACK_PROPS',
        payload: {
          trackId: selectedTrack.id,
          metadata: { ...(selectedTrack.metadata || {}), feedback: rating },
        },
      });
    } catch (_) { /* best effort */ }
  }, [selectedTrack, state.projectName, dispatch]);

  const setTrackColor = useCallback((trackId, color) => {
    dispatch({ type: 'UPDATE_TRACK_PROPS', payload: {
      trackId,
      metadata: { ...((state.buses || []).flatMap(b => b.tracks || []).find((t) => t.id === trackId)?.metadata || {}), customColor: color },
    } });
  }, [state.buses, dispatch]);

  const toggleMasterReverb = () => dispatch({ type: 'TOGGLE_MASTER_REVERB_PANEL' });
  const toggleMasterEq     = () => dispatch({ type: 'TOGGLE_MASTER_EQ_PANEL' });
  const setBusReverbSend   = (busId, send) =>
    dispatch({ type: 'UPDATE_BUS_REVERB', payload: { busId, reverbSend: send } });

  // Ruler markers — simple local state for now (Intro / Verse / Chorus …).
  // Shift-click on the ruler adds one at that time.
  const addMarker = useCallback((timeSec) => {
    const name = window.prompt('Marker name?', `Section ${markers.length + 1}`);
    if (!name) return;
    setMarkers((m) => [...m, { time: timeSec, name, color: '#a88adc' }].sort((a, b) => a.time - b.time));
  }, [markers.length]);
  const removeMarker = (idx) => setMarkers((m) => m.filter((_, i) => i !== idx));

  const exportAudio = useCallback(async () => {
    // Render the live audio graph through an OfflineAudioContext and
    // download as .wav. Uses the same `tracksForPlayback` shape the engine
    // already consumes, so every mute/solo/gain state is honored.
    try {
      const dur = Math.max(state.totalDuration || TIMELINE_SECONDS, 4);
      const sr = 48000;
      const ctx = new OfflineAudioContext(2, Math.ceil(dur * sr), sr);
      const master = ctx.createGain();
      master.gain.value = state.masterGain ?? 0.8;
      master.connect(ctx.destination);
      // Collect decoded buffers off the live engine's cache where possible,
      // else decode now.
      const allTracks = [].concat(
        tracksForPlayback.vo, tracksForPlayback.music, tracksForPlayback.sfx,
        tracksForPlayback.drums, tracksForPlayback.audio,
      ).filter((t) => t.audioUrl && !t.isMuted);
      const anySolo = allTracks.some((t) => t.isSolo);
      for (const t of allTracks) {
        if (anySolo && !t.isSolo) continue;
        try {
          const r = await fetch(t.audioUrl);
          const ab = await r.arrayBuffer();
          const buf = await ctx.decodeAudioData(ab.slice(0));
          const src = ctx.createBufferSource();
          src.buffer = buf;
          const g = ctx.createGain();
          g.gain.value = (t.gain ?? 1) * (t._busGain ?? 1);
          src.connect(g).connect(master);
          src.start(t.startPosition || 0);
        } catch (e) { console.warn(`[export] skip ${t.id}:`, e?.message); }
      }
      const rendered = await ctx.startRendering();
      const wav = encodeWavFromBuffer(rendered);
      const blob = new Blob([wav], { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${(state.projectName || 'Untitled').replace(/\s+/g, '_')}.wav`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert(`Export failed: ${e.message}`);
    }
  }, [state, tracksForPlayback]);

  // Clip drag on the timeline. Grid-snaps to beats (≈quarter-note res).
  // mode = 'move' (drag anywhere on clip body) | 'resize' (drag right edge).
  //
  // IMPORTANT: we need the .sd-lane bounding rect for pixel→seconds math.
  // In move mode e.currentTarget is the clip → parent is the lane.
  // In resize mode e.currentTarget is the handle → parent is the clip, so
  //   walking up once lands on the clip, not the lane. Use closest() for
  //   both paths so the math is the lane width regardless of entry point.
  const onClipMouseDown = (e, real, mode = 'move') => {
    if (!real) return;
    e.stopPropagation();
    e.preventDefault();
    const laneEl = e.currentTarget.closest('.sd-lane');
    if (!laneEl) return;
    const r = laneEl.getBoundingClientRect();
    setDragClip({
      mode,
      trackId: real.id, busId: real._busId,
      origStart:    real.startPosition || 0,
      origDuration: real.duration || 4,
      startPx: e.clientX, laneW: r.width,
    });
  };
  // Sidebar horizontal resize (drag the 1 px bar between sidebar + canvas).
  useEffect(() => {
    if (!sbResizing) return;
    const onMove = (e) => setSidebarWidth((_w) => Math.max(220, Math.min(560, e.clientX - sbResizing.offset)));
    const onUp = () => setSbResizing(false);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    document.body.style.cursor = 'ew-resize';
    document.body.style.userSelect = 'none';
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [sbResizing]);

  // Timeline/canvas vertical split (drag the 1 px bar above the chord row).
  // Delta-based so the chord row height between the handle and .sd-bottom
  // doesn't skew the math: moving the mouse up by N pixels adds N to
  // timelineHeight (and the chord row keeps its auto height).
  useEffect(() => {
    if (!tlResizing) return;
    const onMove = (e) => {
      const delta = tlResizing.startY - e.clientY;   // up = positive
      const viewportH = window.innerHeight;
      const next = Math.max(140, Math.min(viewportH - 200, tlResizing.startH + delta));
      setTimelineHeight(next);
    };
    const onUp = () => setTlResizing(false);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    document.body.style.cursor = 'ns-resize';
    document.body.style.userSelect = 'none';
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [tlResizing]);

  useEffect(() => {
    if (!loopDrag) return;
    const onMove = (e) => {
      const pct = (e.clientX - loopDrag.laneLeft) / loopDrag.laneW;
      const t = Math.max(0, pct * (TIMELINE_SECONDS / timelineZoom));
      setLoopRegion({
        start: Math.min(loopDrag.startSec, t),
        end:   Math.max(loopDrag.startSec, t),
      });
    };
    const onUp = () => { setLoopDrag(null); setLoopEnabled(true); };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [loopDrag, timelineZoom]);

  useEffect(() => {
    if (!dragClip) return;
    const onMove = (e) => {
      const dx = e.clientX - dragClip.startPx;
      // Lane shows `TIMELINE_SECONDS / timelineZoom` seconds of content.
      const laneSec = TIMELINE_SECONDS / timelineZoom;
      const dt = (dx / dragClip.laneW) * laneSec;
      const beat = 60 / (state.bpm || 120);
      if (dragClip.mode === 'move') {
        const snapped = Math.max(0, snapSec(dragClip.origStart + dt));
        // UPDATE_TRACK_PROPS spreads payload flat — do NOT wrap in `updates`.
        dispatch({
          type: 'UPDATE_TRACK_PROPS',
          payload: { trackId: dragClip.trackId, startPosition: snapped },
        });
      } else if (dragClip.mode === 'resize') {
        const raw = dragClip.origDuration + dt;
        const newDur = Math.max(beat / 2, snapMode === 'off' ? raw : snapSec(raw));
        dispatch({
          type: 'UPDATE_TRACK_PROPS',
          payload: { trackId: dragClip.trackId, duration: newDur },
        });
      }
    };
    const onUp = () => setDragClip(null);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [dragClip, timelineZoom, state.bpm, dispatch]);

  const timeMs = (state.playheadPosition || 0) * 1000;
  // playheadPct is relative to TIMELINE_SECONDS. When timelineZoom > 1 the
  // visible slice is TIMELINE_SECONDS / zoom, so we multiply in the render
  // (same scaling we already use for clip `left`).
  const playheadPct = Math.max(0, Math.min(100,
    ((state.playheadPosition || 0) / TIMELINE_SECONDS) * 100));

  // Scrub on lane/ruler click. Must divide by zoom so a click at 50 % of
  // the lane seeks to 50 % of the *visible* window, not the full 32 s.
  const onLaneClick = useCallback((e) => {
    // Ignore clicks that bubbled up from a clip — those are track selects.
    if (e.target.closest && e.target.closest('.sd-clip')) return;
    const r = e.currentTarget.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    const t = pct * (TIMELINE_SECONDS / timelineZoom);
    dispatch({ type: 'SEEK_TO', payload: t });
    seek?.(t);
  }, [dispatch, seek, timelineZoom]);

  const playIcon = state.isPlaying ? 'pause' : 'play';
  const ticks = Array.from({ length: 16 }, (_, i) => i * 2 + 1);

  /* ---------- Timeline buses: real if present, spec demo otherwise ----------
     Each bus is a collapsible group: header row (name + M/S + gain) plus, when
     expanded, its child track rows. Dispatches route to UPDATE_BUS_*, TOGGLE_
     BUS_*, and TOGGLE_TRACK_* so every change persists through the production
     reducer. */
  const busColorFor = (bus) => {
    const t = (bus?.type || bus?.name || '').toLowerCase();
    if (t.includes('vocal') || t === 'vo') return TRACK_COLORS.vocals;
    if (t.includes('drum')) return TRACK_COLORS.drums;
    if (t.includes('bass')) return TRACK_COLORS.bass;
    if (t.includes('string') || t.includes('guitar')) return TRACK_COLORS.strings;
    if (t.includes('music') || t.includes('piano') || t.includes('keys')) return TRACK_COLORS.rhodes;
    return TRACK_COLORS.lead;
  };

  const realBuses = (state.buses || []).map((bus) => {
    const color = busColorFor(bus);
    return {
      _real: bus,
      id: bus.id, name: bus.name || bus.type || 'Bus',
      color, expanded: bus.expanded !== false,
      mute: !!bus.mute, solo: !!bus.solo,
      gain: bus.gain ?? 1,
      tracks: (bus.tracks || [])
        // Hide the original uploaded mix once stems have been extracted
        // (metadata.isBusMaster flagged in ingestFile). It stays in
        // state for playback / regen but the timeline shows only the
        // 6 stems so the tracklist doesn't double-count.
        .filter((t) => !t.metadata?.isBusMaster)
        .map((t) => {
          const type = (t.metadata?.stemType || t.metadata?.instrument || t.name || '').toLowerCase();
          return {
            _real: { ...t, _busId: bus.id },
            name: t.name || t.metadata?.stemType || t.id,
            color: colorFor(type),
            clips: [{
              s: t.startPosition || 0,
              e: Math.max((t.startPosition || 0) + 1, (t.startPosition || 0) + (t.duration || 4)),
            }],
          };
        }),
    };
  });

  const timelineBuses = realBuses;

  // Bus-level actions (no-op on demo buses, where _real is absent).
  const toggleBusExpanded = (busId) => busId && dispatch({ type: 'TOGGLE_BUS_EXPANDED', payload: { busId } });
  const toggleBusMute     = (busId) => busId && dispatch({ type: 'TOGGLE_BUS_MUTE',     payload: { busId } });
  const toggleBusSolo     = (busId) => busId && dispatch({ type: 'TOGGLE_BUS_SOLO',     payload: { busId } });
  const setBusGain        = (busId, gain) => busId && dispatch({ type: 'UPDATE_BUS_GAIN', payload: { busId, gain } });
  const setBusPan         = (busId, pan)  => busId && dispatch({ type: 'UPDATE_BUS_PAN',  payload: { busId, pan } });
  const renameBus         = (busId, name) => busId && dispatch({ type: 'UPDATE_BUS_NAME', payload: { busId, name } });
  const clearBus          = (busId) => busId && dispatch({ type: 'CLEAR_BUS', payload: { busId } });

  /* ---------- Canvas content ---------- */
  // Always render the active mode's component. Each mode handles its own
  // empty state internally, so a never-used /studio-dev still looks right
  // (the piano roll shows a "Pick a track" prompt, etc.). The hero is only
  // visible when the user explicitly switches to `welcome` — currently
  // reached by clicking the brand dot.
  const renderCanvas = () => {
    if (activeMode === 'welcome') {
      return (
        <div className="wb-canvas">
          <div className="wb-canvas__grid" aria-hidden="true" />
          <div className="wb-empty">
            <div className="wb-empty__status">
              <div className="wb-empty__dot" />
              STATUS · NO SOURCE LOADED
            </div>
            <h1 className="wb-empty__title">Session is empty.</h1>
            <p className="wb-empty__body">
              Load a reference track, start recording, or route MIDI from the studio rail. Workbench is ready.
            </p>
            <div className="wb-empty__actions">
              <button className="wb-btn wb-btn--primary" onClick={triggerUpload}>&gt; Load file</button>
              <button className="wb-btn" onClick={togglePlay}>
                &gt; {state.isPlaying ? 'Pause' : 'Record'}
              </button>
              <button className="wb-btn wb-btn--muted" onClick={() => { setActiveMode('midi'); addEmptyTrack(); }}>&gt; New MIDI</button>
            </div>
          </div>
        </div>
      );
    }
    switch (activeMode) {
      case 'midi':  return <StudioDevMidi />;
      case 'audio': return <StudioDevWaveform />;
      case 'video': return <StudioDevVideo />;
      case 'fx':    return <StudioDevFX />;
      default:      return null;
    }
  };

  return (
    <div className="studio-dev">
      {/* ================== HEADER ================== */}
      <header className="wb-menubar">
        <div className="wb-brand" style={{ cursor: 'pointer' }} onClick={() => setActiveMode('welcome')}>
          <div className="wb-brand__mark">d</div>
          <div className="wb-brand__name">doseedo</div>
          <div className="wb-brand__version">v0.42.1</div>
        </div>
        <div className="wb-divider">│</div>
        <nav className="wb-menu">
          <button className="wb-menu__item" onClick={() => undo()}>Undo</button>
          <button className="wb-menu__item" onClick={() => redo()}>Redo</button>
          <button className="wb-menu__item" onClick={exportAudio}>Export</button>
          <button className="wb-menu__item" onClick={toggleCinema}>Cinema</button>
          <button className="wb-menu__item" onClick={() => navigate('/search')}>Search</button>
          <button className="wb-menu__item" onClick={() => navigate('/studio')}>Settings</button>
        </nav>
        <div className="wb-menubar__meta">
          PROJECT: {(state.projectName || 'untitled').toLowerCase().replace(/\s+/g, '_')}.dsd · SR 48kHz · 24bit ·
          {' '}{autosaveStatus === 'saving' ? 'saving' : autosaveStatus === 'saved' ? 'saved' : autosaveStatus === 'failed' ? 'save failed' : 'idle'}
        </div>
        {/* Retain the File dropdown as a tiny pinned slot at the far right so
            New/Open/Save/Save As/Export shortcuts still exist. */}
        <div className="wb-brand" style={{ marginLeft: 6 }}>
          <StudioDevFileMenu />
        </div>
      </header>

      {/* ================== MAIN SPLIT ================== */}
      <div className="sd-main">
        {/* -------- LEFT NAV RAIL (collapsed by default) -------- */}
        <StudioDevNav
          expanded={navExpanded}
          onToggleExpanded={() => setNavExpanded((v) => !v)}
          // The wand ("generate") icon is the home/default view — it maps to
          // the instrument palette (which IS the generation tab). Chat and
          // Browse swap the sidebar content and toggle back to instruments
          // when re-clicked.
          activePanel={sidebarPanel === 'instruments' ? 'generate' : sidebarPanel}
          onOpenGenerate={() => setSidebarPanel('instruments')}
          onOpenBrowse={()   => setSidebarPanel((p) => p === 'browse' ? 'instruments' : 'browse')}
          onOpenChat={()     => setSidebarPanel((p) => p === 'chat'   ? 'instruments' : 'chat')}
        />

        {/* -------- LEFT SIDEBAR (swaps content by panel) -------- */}
        <aside className="sd-sidebar" style={{ width: sidebarWidth, flex: `0 0 ${sidebarWidth}px` }}>
          {sidebarPanel === 'instruments' && (
            <>
              <div className="sd-side-title">Studio</div>
              <div className="sd-side-sub">Your instruments and sources</div>

              <div className="sd-tabs" role="tablist">
                {['Instruments', 'Vocals', 'Drums'].map((t) => (
                  <button key={t} className={`sd-tab ${t === activeTab ? 'active' : ''}`}
                          role="tab" onClick={() => setActiveTab(t)}>{t}</button>
                ))}
              </div>

              <div className="sd-label" data-count="3"><span>Sources</span></div>
              <div className="sd-grid-3">
                <button className="sd-tile" onClick={triggerUpload}>
                  <Icon k="upload" size={14} color="var(--wb-ink)" stroke={1.3} />
                  <div className="sd-tile-label">UPL</div>
                </button>
                <button className={`sd-tile ${recorder?.isRecording ? 'sd-tile-rec' : ''}`}
                        onClick={async () => {
                          if (!recorder) return;
                          if (recorder.isRecording) {
                            const blob = await recorder.stopRecording();
                            if (!blob) return;
                            const file = new File([blob], `take-${Date.now()}.webm`, { type: blob.type });
                            const busId = `bus-rec-${Date.now()}`;
                            const trackId = `t-rec-${Date.now()}`;
                            dispatch({ type: 'CREATE_BUS', payload: { id: busId, type: 'AUDIO', name: 'Recording', expanded: false } });
                            dispatch({ type: 'ADD_TRACK', payload: { busId, track: {
                              id: trackId, name: file.name, audioFile: file, audioUrl: URL.createObjectURL(blob),
                              duration: 0, startPosition: state.playheadPosition || 0,
                              gain: 1, isMuted: false, isSolo: false, pan: 0,
                              fx: { reverb: 0, fadeIn: 0, fadeOut: 0 },
                              metadata: { type: 'recording' },
                            } } });
                            dispatch({ type: 'SELECT_TRACK', payload: { trackId, busId } });

                            // Same auto-pipeline as the upload flow: analyze +
                            // stem-separate in parallel. When analysis returns
                            // the parent track gets midi/classification/latent;
                            // when stems return they drop as children and the
                            // bus collapses.
                            analyzeAudio(file).then((res) => {
                              const cls = res.classification;
                              const midi = res.midi;
                              const latent = res.latent;
                              const instType = cls?.type || 'other';
                              dispatch({
                                type: 'UPDATE_TRACK',
                                payload: {
                                  busId, trackId,
                                  updates: {
                                    metadata: {
                                      type: 'recording',
                                      instrument: instType,
                                      instrumentLabel: cls?.label || null,
                                      icon: iconForType(instType),
                                      midi: midi?.midi_url || null,
                                      latent: latent?.latent_url || null,
                                      latentId: latent?.latent_id || null,
                                    },
                                  },
                                },
                              });
                            }).catch(() => {});
                            separateStemsAuto(file).then((sep) => {
                              if (!sep?.stems) return;
                              const stemTracks = Object.entries(sep.stems).map(([stemName, audioUrl]) => ({
                                id: `stem-${trackId}-${stemName}`,
                                name: `take — ${stemName}`,
                                audioUrl, duration: 0,
                                startPosition: state.playheadPosition || 0,
                                gain: 1, isMuted: false, isSolo: false,
                                fx: { reverb: 0, fadeIn: 0.2, fadeOut: 1.0 },
                                metadata: { type: 'stem', stemType: stemName, parentTrackId: trackId, instrument: stemName, icon: iconForType(stemName) },
                              }));
                              dispatch({ type: 'ADD_TRACKS_BULK', payload: { busId, tracks: stemTracks } });
                              // Promote the recorded master to isBusMaster
                              // (hidden from the per-track list, stays in
                              // bus.tracks for playback). Same rationale
                              // as the upload flow.
                              dispatch({
                                type: 'UPDATE_TRACK',
                                payload: {
                                  busId, trackId,
                                  updates: { metadata: { isBusMaster: true } },
                                },
                              });
                              dispatch({ type: 'SET_BUS_EXPANDED', payload: { busId, expanded: false } });
                            }).catch(() => {});
                          } else {
                            try { await recorder.startRecording(); }
                            catch (e) { alert(`Recording failed: ${e.message || e}`); }
                          }
                        }}
                        title={recorder?.isRecording ? 'Stop recording' : 'Record from mic'}>
                  <Icon k="mic" size={14}
                        color={recorder?.isRecording ? 'var(--wb-accent-warm)' : 'var(--wb-ink)'} stroke={1.3} />
                  <div className="sd-tile-label">{recorder?.isRecording ? 'STP' : 'REC'}</div>
                </button>
                <button className="sd-tile" onClick={() => setSidebarPanel('browse')}>
                  <Icon k="folder" size={14} color="var(--wb-ink)" stroke={1.3} />
                  <div className="sd-tile-label">BRW</div>
                </button>
              </div>
              <input ref={fileInputRef} type="file" accept="audio/*"
                     style={{ display: 'none' }} onChange={onFilePick} />

              {/* Currently selected track — auto-loaded as the generation
               * input so Generate acts on whatever the user has selected
               * in the timeline. */}
              {selectedTrack && (
                <div className="sd-selected-source">
                  <div className="sd-selected-source-head">
                    <span className="sd-selected-source-dot"
                          style={{ background: colorFor(
                            (selectedTrack.metadata?.stemType
                              || selectedTrack.metadata?.instrument
                              || selectedTrack.name || '').toLowerCase()
                          ) }} />
                    <span className="sd-selected-source-label">Selected</span>
                  </div>
                  <div className="sd-selected-source-name" title={selectedTrack.name || selectedTrack.id}>
                    {selectedTrack.name || selectedTrack.id}
                  </div>
                  <div className="sd-selected-source-meta">
                    {selectedTrack.metadata?.instrumentLabel
                      || selectedTrack.metadata?.stemType
                      || selectedTrack.metadata?.instrument
                      || selectedTrack.metadata?.type
                      || 'track'}
                    {selectedTrack.duration
                      ? ` · ${selectedTrack.duration.toFixed(1)}s`
                      : ''}
                  </div>
                </div>
              )}

              <div className="sd-inst-palette sd-inst-list">
                {/* Source sub-tabs — Live = built-in instrument list,
                 * Custom = user-curated collection (empty placeholder). */}
                <div className="sd-tabs sd-palette-src-tabs" role="tablist" aria-label="Palette source">
                  {[
                    { key: 'live',   label: 'Live' },
                    { key: 'custom', label: 'Custom' },
                  ].map((t) => (
                    <button key={t.key}
                            role="tab"
                            className={`sd-tab ${paletteSource === t.key ? 'active' : ''}`}
                            onClick={() => setPaletteSource(t.key)}>
                      {t.label}
                    </button>
                  ))}
                </div>

                {paletteSource === 'custom' && (
                  <div className="sd-palette-empty">
                    <div className="sd-palette-empty-body">No custom instruments yet.</div>
                    <button className="wb-btn wb-btn--primary sd-palette-create"
                            onClick={() => { /* no-op placeholder */ }}>
                      &gt; Create new
                    </button>
                  </div>
                )}

                {/* Workbench-style list palette — Live source. */}
                {paletteSource === 'live' && <>

                {/* ---- Instruments: expandable group → subgroup tree ---- */}
                {activeTab === 'Instruments' && INSTRUMENT_GROUPS.map((g, idx) => {
                  const open = activeInstGroup === g.id;
                  const subs = INSTRUMENT_SUBGROUPS[g.id] || [];
                  return (
                    <React.Fragment key={g.id}>
                      <button
                        className={`sd-inst-row ${open ? 'open' : ''}`}
                        onClick={() => setActiveInstGroup(open ? null : g.id)}
                      >
                        <span className="sd-inst-row-num">{String(idx + 1).padStart(2, '0')}</span>
                        <span className="sd-inst-row-name">{g.label}</span>
                        <span className="sd-inst-row-meta">{subs.length}</span>
                        <i className={`fa-solid fa-chevron-${open ? 'down' : 'right'} sd-inst-row-caret`} />
                      </button>
                      {open && subs.map((s, si) => (
                        <button
                          key={s.id}
                          className={`sd-inst-row sd-inst-row-sub ${selectedInstrument?.subgroup === s.id ? 'on' : ''}`}
                          onClick={() => setSelectedInstrument({
                            id: s.id, label: s.label, sub: s.sub,
                            group: g.id, subgroup: s.id,
                          })}
                        >
                          <span className="sd-inst-row-num">
                            {String(idx + 1).padStart(2, '0')}.{String(si + 1).padStart(2, '0')}
                          </span>
                          <span className="sd-inst-row-name">{s.label}</span>
                          <span className="sd-inst-row-meta">{s.sub}</span>
                        </button>
                      ))}
                    </React.Fragment>
                  );
                })}

                {/* ---- Drums: flat list ---- */}
                {activeTab === 'Drums' && DRUM_GROUPS.map((g, idx) => (
                  <button key={g.id}
                          className={`sd-inst-row ${selectedInstrument?.subgroup === g.id ? 'on' : ''}`}
                          onClick={() => setSelectedInstrument({
                            id: g.id, label: g.label,
                            group: 'drums', subgroup: g.id,
                          })}>
                    <span className="sd-inst-row-num">{String(idx + 1).padStart(2, '0')}</span>
                    <span className="sd-inst-row-name">{g.label}</span>
                  </button>
                ))}

                {/* ---- Vocals: flat list ---- */}
                {activeTab === 'Vocals' && VOCAL_GROUPS.map((g, idx) => (
                  <button key={g.id}
                          className={`sd-inst-row ${selectedInstrument?.subgroup === g.id ? 'on' : ''}`}
                          onClick={() => setSelectedInstrument({
                            id: g.id, label: g.label,
                            group: 'vocals', subgroup: g.id,
                          })}>
                    <span className="sd-inst-row-num">{String(idx + 1).padStart(2, '0')}</span>
                    <span className="sd-inst-row-name">{g.label}</span>
                  </button>
                ))}
                </>}
              </div>

              {/* Generate form — visible below the palette in every tab
                  (Instruments / Vocals / Drums) so the advanced-params
                  dropdown and the Generate button are always reachable. */}
              <StudioDevGenerate embedded selectedInstrument={selectedInstrument} />
            </>
          )}

          {sidebarPanel === 'chat' && (
            <StudioDevChat onClose={() => setSidebarPanel('instruments')} />
          )}

          {sidebarPanel === 'browse' && (
            <StudioDevMidiBrowser onClose={() => setSidebarPanel('instruments')} />
          )}

          {sidebarPanel === 'generate' && (
            <StudioDevGenerate onClose={() => setSidebarPanel('instruments')} selectedInstrument={selectedInstrument} />
          )}
        </aside>

        {/* Vertical drag handle — resizes the sidebar horizontally. */}
        <div
          className={`sd-resize sd-resize-v ${sbResizing ? 'active' : ''}`}
          onMouseDown={(e) => {
            e.preventDefault();
            // Record the offset between viewport-left and sidebar-left so
            // nav-rail width doesn't skew the resize math.
            const sidebarLeft = e.currentTarget.getBoundingClientRect().left - sidebarWidth;
            setSbResizing({ offset: sidebarLeft });
          }}
          title="Drag to resize sidebar"
        />

        {/* -------- CANVAS COLUMN -------- */}
        <div className="sd-canvas-col">
          {/* Workbench tab strip. */}
          <div className="wb-tabs">
            {MODES.map((m) => (
              <button key={m.key}
                      className={`wb-tab ${m.key === activeMode ? 'wb-tab--active' : ''}`}
                      onClick={() => setActiveMode(m.key)}>
                {m.label}
              </button>
            ))}
          </div>
          <section className="sd-canvas">
            <div className="sd-canvas-main">
              {renderCanvas()}
            </div>
          </section>
          {/* Horizontal drag handle — resizes the bottom (chords+transport+timeline)
              vs the canvas above. Sits above the chord row so dragging it
              treats the chords as part of the bottom panel. */}
          <div
            className={`sd-resize sd-resize-h ${tlResizing ? 'active' : ''}`}
            onMouseDown={(e) => {
              e.preventDefault();
              setTlResizing({ startY: e.clientY, startH: timelineHeight });
            }}
            title="Drag to resize timeline"
          />

          {/* Chord row — hidden in FX mode to avoid cluttering the knobs. */}
          {activeMode !== 'fx' && <StudioDevChords />}

          {/* -------- BOTTOM: TRANSPORT + TIMELINE -------- */}
          <div className="sd-bottom" style={{ height: timelineHeight, flex: `0 0 ${timelineHeight}px` }}>
            <div className="sd-transport">
              <div className="sd-transport-group">
                <button className="sd-tbtn" aria-label="Play/Pause" onClick={togglePlay}>
                  <Icon k={playIcon} size={12} color="var(--hifi-ink)" />
                </button>
                <button className="sd-tbtn" aria-label="Stop" onClick={stopPlay}>
                  <Icon k="stop" size={12} color="var(--hifi-ink)" />
                </button>
                <button className="sd-tbtn rec" aria-label="Record" onClick={togglePlay}>
                  <Icon k="rec" size={12} color="var(--hifi-accent)" fill="var(--hifi-accent)" />
                </button>
              </div>
              <div className="sd-time">{fmtTime(timeMs)}</div>
              <div className="sd-divider" />
              <div className="sd-kv">
                <span className="sd-kv-k">Tempo</span>
                <input type="number" className="sd-kv-v sd-kv-input"
                       value={Math.round(state.bpm || 120)}
                       onChange={(e) => setBpm(parseFloat(e.target.value))} />
                <span className="sd-kv-u">bpm</span>
                <button className="sd-btn ghost" style={{ padding: '2px 6px', fontSize: 10 }}
                        onClick={tapTempo} title="Tap 4× to set tempo">Tap</button>
              </div>
              <div className="sd-divider" />
              <div className="sd-kv">
                <span className="sd-kv-k">Meter</span>
                <select
                  className="sd-kv-input"
                  style={{ width: 60 }}
                  value={`${state.beatsPerBar || 4}/${state.meterDenominator || 4}`}
                  onChange={(e) => setTimeSig(e.target.value)}
                >
                  {['3/4','4/4','5/4','6/4','6/8','7/8','9/8','12/8'].map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </div>
              <div className="sd-divider" />
              {/* Metronome toggle */}
              <button
                className={`sd-btn ghost ${state.metronomeEnabled ? 'on' : ''}`}
                onClick={toggleMetronome}
                title="Metronome"
              >
                <span style={{ fontFamily: 'var(--hifi-mono)', fontSize: 10 }}>♩</span>
                {state.metronomeEnabled ? 'Click on' : 'Click'}
              </button>
              {/* Timeline zoom — buttons route to X or Y depending on direction.
                  X scales the timeline horizontally (clip widths); Y scales
                  lane row heights. Direction is stored in state.zoomMode so
                  other panels (MIDI, waveform) can pick it up too. */}
              <div className="sd-kv">
                <span className="sd-kv-k">Zoom</span>
                <button
                  className="sd-btn ghost"
                  style={{
                    padding: 0, width: 28, height: 24,
                    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                    fontFamily: 'var(--hifi-mono)', fontSize: 13, lineHeight: 1,
                  }}
                  onClick={() => dispatch({
                    type: 'SET_ZOOM_MODE',
                    payload: (state.zoomMode === 'x') ? 'y' : 'x',
                  })}
                  title={`Zoom direction: ${state.zoomMode === 'x' ? 'Horizontal (X)' : 'Vertical (Y)'} — click to switch`}
                >
                  {state.zoomMode === 'x' ? '↔' : '↕'}
                </button>
                <button className="sd-btn ghost" style={{ padding: '2px 8px' }}
                        onClick={() => {
                          if (state.zoomMode === 'y') setLaneRowZoom((z) => Math.max(0.6, z / 1.25));
                          else setTimelineZoom((z) => Math.max(0.25, z / 1.5));
                        }}>−</button>
                <button className="sd-btn ghost" style={{ padding: '2px 8px' }}
                        onClick={() => {
                          if (state.zoomMode === 'y') setLaneRowZoom((z) => Math.min(3, z * 1.25));
                          else setTimelineZoom((z) => Math.min(8, z * 1.5));
                        }}>+</button>
              </div>
              <div className="sd-spacer" />
              <button className="sd-btn ghost" onClick={addEmptyTrack}>
                <Icon k="plus" size={12} /> Add track
              </button>
            </div>

            <div
              className={`sd-timeline ${isDraggingFile ? 'sd-drag-over' : ''}`}
              style={{ '--row-h': `${Math.round(38 * laneRowZoom)}px` }}
              onDragOver={onTimelineDragOver}
              onDragLeave={onTimelineDragLeave}
              onDrop={onTimelineDrop}
            >
              <div className="sd-tracks-col">
                <div className="sd-tracks-header">Buses · Tracks</div>
                {timelineBuses.map((bus) => (
                  <React.Fragment key={bus.id}>
                    {/* ---- Bus header row ---- */}
                    <div className={`sd-bus-row ${state.selectedBus?.id === bus.id ? 'selected' : ''}`}
                         onClick={() => dispatch({ type: 'SELECT_BUS', payload: { busId: bus.id } })}
                         title="Click to select this bus (controls move to the right sidebar). Click the caret to collapse/expand.">
                      <button className="sd-bus-caret"
                              onClick={(e) => { e.stopPropagation(); toggleBusExpanded(bus.id); }}>
                        <i className={`fa-solid fa-${bus.expanded ? 'caret-down' : 'caret-right'}`} />
                      </button>
                      <div className="sd-bus-color" style={{ background: bus.color }} />
                      <input
                        className="sd-bus-name"
                        value={bus.name}
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => renameBus(bus.id, e.target.value)}
                      />
                      <div className="sd-track-ms" onClick={(e) => e.stopPropagation()}>
                        <button
                          className="sd-ms-btn"
                          style={bus.mute ? { color: 'var(--hifi-accent)', borderColor: 'var(--hifi-accent)' } : {}}
                          onClick={() => toggleBusMute(bus.id)}
                        >M</button>
                        <button
                          className="sd-ms-btn"
                          style={bus.solo ? { background: 'var(--hifi-accent)', color: 'var(--hifi-bg)', borderColor: 'var(--hifi-accent)' } : {}}
                          onClick={() => toggleBusSolo(bus.id)}
                        >S</button>
                      </div>
                    </div>

                    {/* ---- Track rows (only when expanded) ---- */}
                    {bus.expanded && bus.tracks.map((tr, i) => {
                      const real = tr._real;
                      const isSel = real && selectedTrack?.id === real.id;
                      const level = real ? (meterByTrack[real.id] ?? 0) : 0;
                      return (
                        <div key={real?.id || `${bus.id}-t-${i}`}
                             className={`sd-track-row indented ${isSel ? 'selected' : ''}`}
                             draggable={!!real}
                             onDragStart={(e) => {
                               if (!real) return;
                               e.dataTransfer.effectAllowed = 'move';
                               e.dataTransfer.setData('text/plain', JSON.stringify({ trackId: real.id, busId: real._busId }));
                             }}
                             onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; }}
                             onDrop={(e) => {
                               e.preventDefault();
                               try {
                                 const { trackId, busId: srcBus } = JSON.parse(e.dataTransfer.getData('text/plain'));
                                 if (srcBus !== bus.id) return;   // cross-bus moves: not here yet
                                 dispatch({ type: 'REORDER_TRACK_IN_BUS', payload: { busId: bus.id, trackId, toIndex: i } });
                               } catch (_) {}
                             }}
                             onClick={() => real && selectTrack(real.id, real._busId)}>
                          <div className="sd-track-color"
                               style={{ background: real?.metadata?.customColor || tr.color, cursor: real ? 'pointer' : 'default' }}
                               onClick={(e) => { e.stopPropagation(); if (real) setColorPickerFor((x) => x === real.id ? null : real.id); }} />
                          {colorPickerFor && real?.id === colorPickerFor && (
                            <div className="sd-color-picker" onClick={(e) => e.stopPropagation()}>
                              {['#a88adc','#e8c88a','#8ac8a0','#e07556','#6aa8e8','#c1abe8','#6a4e9e'].map((c) => (
                                <button key={c} style={{ background: c }} className="sd-color-swatch"
                                        onClick={() => { setTrackColor(real.id, c); setColorPickerFor(null); }} />
                              ))}
                            </div>
                          )}
                          <div className="sd-track-name">{tr.name}</div>
                          <div className="sd-track-meter" title={`level ${Math.round(level * 100)}%`}>
                            <div className="sd-track-meter-fill" style={{ width: `${Math.round(level * 100)}%` }} />
                          </div>
                          <div className="sd-track-ms">
                            <button
                              className="sd-ms-btn"
                              style={real?.isMuted ? { color: 'var(--hifi-accent)', borderColor: 'var(--hifi-accent)' } : {}}
                              onClick={(e) => { e.stopPropagation(); real && toggleMute(real.id, real._busId); }}
                            >M</button>
                            <button
                              className="sd-ms-btn"
                              style={real?.isSolo ? { background: 'var(--hifi-accent)', color: 'var(--hifi-bg)', borderColor: 'var(--hifi-accent)' } : {}}
                              onClick={(e) => { e.stopPropagation(); real && toggleSolo(real.id, real._busId); }}
                            >S</button>
                          </div>
                        </div>
                      );
                    })}
                  </React.Fragment>
                ))}
              </div>
              <div className="sd-lanes" onClick={onLaneClick}>
                <div
                  className="sd-ruler"
                  onClick={(e) => {
                    // Shift-click the ruler adds a section marker.
                    if (!e.shiftKey) return;
                    e.preventDefault();
                    e.stopPropagation();
                    const r = e.currentTarget.getBoundingClientRect();
                    const pct = (e.clientX - r.left) / r.width;
                    addMarker(pct * (TIMELINE_SECONDS / timelineZoom));
                  }}
                  onMouseDown={(e) => {
                    // Alt-drag (Option on mac) paints the loop region.
                    if (!e.altKey) return;
                    e.preventDefault();
                    e.stopPropagation();
                    const r = e.currentTarget.getBoundingClientRect();
                    const pct = (e.clientX - r.left) / r.width;
                    const t = pct * (TIMELINE_SECONDS / timelineZoom);
                    setLoopDrag({ startSec: t, laneW: r.width, laneLeft: r.left });
                    setLoopRegion({ start: t, end: t });
                  }}
                >
                  {ticks.map((n, i) => <div key={i} className="sd-tick">{Math.round(n * timelineZoom)}</div>)}
                  {loopRegion && (loopRegion.end > loopRegion.start) && (
                    <div className={`sd-loop-region ${loopEnabled ? 'active' : ''}`} style={{
                      left: `${(loopRegion.start / TIMELINE_SECONDS) * 100 * timelineZoom}%`,
                      width: `${((loopRegion.end - loopRegion.start) / TIMELINE_SECONDS) * 100 * timelineZoom}%`,
                    }} />
                  )}
                  {markers.map((m, i) => (
                    <div key={i} className="sd-marker" style={{
                      left: `${(m.time / TIMELINE_SECONDS) * 100 * timelineZoom}%`,
                      borderColor: m.color,
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (e.altKey) { removeMarker(i); return; }
                      dispatch({ type: 'SEEK_TO', payload: m.time });
                    }}
                    title={`${m.name} — Alt-click to remove`}>
                      <span className="sd-marker-label" style={{ color: m.color }}>{m.name}</span>
                    </div>
                  ))}
                </div>
                {timelineBuses.length === 0 && (
                  <div className="sd-empty-upload">
                    <button type="button" className="sd-empty-upload-inner" onClick={triggerUpload}>
                      <div className="sd-empty-upload-title">Drop an audio file here</div>
                      <div className="sd-empty-upload-body">WAV · MP3 · MIDI · FLAC · M4A</div>
                    </button>
                  </div>
                )}
                {timelineBuses.map((bus, bi) => {
                  const rows = [];
                  // Bus-level lane — empty when expanded (the tracks below
                  // show each clip). When collapsed, render a summary of all
                  // child clips in the bus color so you still see content.
                  // Compute the bus's active range (earliest start / latest end)
                  // so the collapsed view can render ONE merged clip across it.
                  let busStart = Infinity, busEnd = -Infinity;
                  for (const tr of bus.tracks) for (const c of tr.clips) {
                    if (c.s < busStart) busStart = c.s;
                    if (c.e > busEnd) busEnd = c.e;
                  }
                  const hasRange = isFinite(busStart) && busEnd > busStart;
                  const masterLeft  = hasRange ? (busStart / TIMELINE_SECONDS) * 100 * timelineZoom : 0;
                  const masterWidth = hasRange ? Math.max(1, ((busEnd - busStart) / TIMELINE_SECONDS) * 100 * timelineZoom) : 0;

                  // Bus master row. Always ONE clip spanning the bus's range,
                  // with a waveform whose amplitude at each bar is the SUM of
                  // the active tracks' amplitudes at that moment — so stacked
                  // voices visibly add up in height. Clicking it toggles the
                  // bus's expansion.
                  const busLaneChildren = [];
                  if (hasRange) {
                    const trackEnvelopes = bus.tracks.flatMap((tr, ti) =>
                      tr.clips.map((c) => ({ start: c.s, end: c.e, seed: (bi * 7 + ti * 13) })));
                    busLaneChildren.push(
                      <div key="master" className={`sd-clip sd-summary sd-master ${bus.expanded ? 'sd-expanded' : ''}`}
                           style={{
                             left: `${masterLeft}%`, width: `${masterWidth}%`,
                             background: `${bus.color}22`,
                             border: `1px solid ${bus.color}66`,
                           }}
                           onClick={(e) => { e.stopPropagation(); toggleBusExpanded(bus.id); }}
                           title={bus.expanded ? 'Click to collapse' : 'Click to expand'}>
                        <div className="sd-clip-label" style={{ color: bus.color }}>
                          {bus.name} · {bus.tracks.length} track{bus.tracks.length === 1 ? '' : 's'}
                        </div>
                        <SummedWaveformMemo
                          tracks={trackEnvelopes}
                          busStart={busStart} busEnd={busEnd}
                          width100={Math.max(40, Math.floor((busEnd - busStart) * CLIP_BARS_PER_SEC * timelineZoom))}
                          height={20} color={bus.color}
                          silent={!bus.tracks.some((tr) => tr._real?.audioUrl)}
                        />
                      </div>
                    );
                  }
                  rows.push(
                    <div key={`${bus.id}-bus`} className="sd-lane bus-lane">
                      {busLaneChildren}
                    </div>
                  );
                  if (bus.expanded) {
                    bus.tracks.forEach((tr, i) => {
                      rows.push(
                        <div key={tr._real?.id || `${bus.id}-tl-${i}`}
                             className={`sd-lane ${i % 2 === 0 ? 'even' : 'odd'}`}>
                          {tr.clips.map((c, j) => {
                            const left = (c.s / TIMELINE_SECONDS) * 100 * timelineZoom;
                            const width = Math.max(1, ((c.e - c.s) / TIMELINE_SECONDS) * 100 * timelineZoom);
                            return (
                              <div key={j} className="sd-clip" style={{
                                left: `${left}%`, width: `${width}%`,
                                background: `${tr.color}22`, border: `1px solid ${tr.color}55`,
                                cursor: tr._real ? 'grab' : 'default',
                              }}
                              onMouseDown={(e) => onClipMouseDown(e, tr._real, 'move')}
                              onClick={(e) => { e.stopPropagation(); tr._real && selectTrack(tr._real.id, tr._real._busId); }}
                              >
                                <div className="sd-clip-label" style={{ color: tr.color }}>{tr.name}</div>
                                <Waveform
                                  height={16}
                                  color={tr.color}
                                  seed={(bi * 8 + i * 3) % BAR_SIG.length}
                                  bars={Math.max(8, Math.round((c.e - c.s) * CLIP_BARS_PER_SEC * timelineZoom))}
                                  silent={!tr._real?.audioUrl}
                                />
                                {tr._real && (
                                  <div className="sd-clip-resize"
                                       onMouseDown={(e) => onClipMouseDown(e, tr._real, 'resize')}
                                       title="Drag to resize" />
                                )}
                              </div>
                            );
                          })}
                        </div>
                      );
                    });
                  }
                  return rows;
                })}
                {/* Single playhead spans ruler + all lanes. */}
                <div className="sd-playhead" style={{ left: `${playheadPct * timelineZoom}%` }} />
              </div>
            </div>
          </div>
        </div>

        {/* -------- RIGHT: TRACK INFO -------- */}
        <aside className="sd-right">
          {/* BUS view — shown when a bus is selected and no track is selected. */}
          {!selectedTrack && state.selectedBus && (() => {
            const b = state.selectedBus;
            return (
              <>
                <div className="sd-side-title">Bus</div>
                <div className="sd-side-sub">{b.name || b.type || b.id}</div>

                <div className="sd-label">Levels</div>
                <div className="sd-ctrl-block">
                  <div className="sd-ctrl-row">
                    <span className="sd-ctrl-k">Gain</span>
                    <input type="range" min={0} max={1} step={0.01}
                           value={b.gain ?? 1}
                           onChange={(e) => setBusGain(b.id, parseFloat(e.target.value))} />
                    <span className="sd-ctrl-v">{Math.round((b.gain ?? 1) * 100)}</span>
                  </div>
                  <div className="sd-ctrl-row">
                    <span className="sd-ctrl-k">Pan</span>
                    <input type="range" min={-1} max={1} step={0.01}
                           value={b.pan ?? 0}
                           onChange={(e) => setBusPan(b.id, parseFloat(e.target.value))} />
                    <span className="sd-ctrl-v">{((b.pan ?? 0) >= 0 ? '+' : '') + (b.pan ?? 0).toFixed(2)}</span>
                  </div>
                  <div className="sd-ctrl-row">
                    <span className="sd-ctrl-k">Reverb</span>
                    <input type="range" min={0} max={1} step={0.01}
                           value={b.reverbSend ?? 0}
                           onChange={(e) => setBusReverbSend(b.id, parseFloat(e.target.value))} />
                    <span className="sd-ctrl-v">{Math.round((b.reverbSend ?? 0) * 100)}</span>
                  </div>
                  <div className="sd-ctrl-row">
                    <span className="sd-ctrl-k">M / S</span>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button className="sd-ms-btn sd-ms-lg"
                              style={b.mute ? { color: 'var(--hifi-accent)', borderColor: 'var(--hifi-accent)' } : {}}
                              onClick={() => toggleBusMute(b.id)}>M</button>
                      <button className="sd-ms-btn sd-ms-lg"
                              style={b.solo ? { background: 'var(--hifi-accent)', color: 'var(--hifi-bg)', borderColor: 'var(--hifi-accent)' } : {}}
                              onClick={() => toggleBusSolo(b.id)}>S</button>
                    </div>
                  </div>
                </div>

                <div className="sd-right-actions">
                  <button className="sd-btn ghost" onClick={() => clearBus(b.id)}>
                    <i className="fa-solid fa-eraser" style={{ fontSize: 10 }} /> Clear tracks
                  </button>
                </div>
              </>
            );
          })()}

          {!selectedTrack && !state.selectedBus && (
            <>
              <div className="sd-side-title">Track</div>
              <div className="sd-side-sub">Nothing selected</div>
            </>
          )}

          {selectedTrack && (
            <>
              <div className="sd-side-title">Track</div>
              <div className="sd-side-sub">
                {selectedTrack._displayName || selectedTrack.name || selectedTrack.id}
              </div>
            </>
          )}

          {selectedTrack && (
            <>
              <div className="sd-label">Info</div>
              <div className="sd-info-rows">
                <div className="sd-info-row">
                  <span className="sd-info-k">Type</span>
                  <span className="sd-info-v">
                    {selectedTrack.metadata?.stemType
                      || selectedTrack.metadata?.instrument
                      || selectedTrack.metadata?.type
                      || '—'}
                  </span>
                </div>
                <div className="sd-info-row">
                  <span className="sd-info-k">ID</span>
                  <span className="sd-info-v sd-info-mono">{selectedTrack.id}</span>
                </div>
                <div className="sd-info-row">
                  <span className="sd-info-k">Start</span>
                  <span className="sd-info-v">{(selectedTrack.startPosition ?? 0).toFixed(2)}s</span>
                </div>
                <div className="sd-info-row">
                  <span className="sd-info-k">Dur</span>
                  <span className="sd-info-v">{(selectedTrack.duration ?? 0).toFixed(2)}s</span>
                </div>
              </div>

              <div className="sd-label">Levels</div>
              <div className="sd-ctrl-block">
                <div className="sd-ctrl-row">
                  <span className="sd-ctrl-k">Gain</span>
                  <input type="range" min={0} max={1} step={0.01}
                         value={selectedTrack.gain ?? 1}
                         onChange={(e) => selectedBusId && setTrackGain(selectedTrack.id, selectedBusId, parseFloat(e.target.value))} />
                  <span className="sd-ctrl-v">{Math.round((selectedTrack.gain ?? 1) * 100)}</span>
                </div>
                <div className="sd-ctrl-row">
                  <span className="sd-ctrl-k">Pan</span>
                  <input type="range" min={-1} max={1} step={0.01}
                         value={selectedTrack.pan ?? 0}
                         onChange={(e) => selectedBusId && setTrackPan(selectedTrack.id, selectedBusId, parseFloat(e.target.value))} />
                  <span className="sd-ctrl-v">{((selectedTrack.pan ?? 0) >= 0 ? '+' : '') + (selectedTrack.pan ?? 0).toFixed(2)}</span>
                </div>
              </div>

              <div className="sd-label">FX</div>
              <div className="sd-ctrl-block">
                <div className="sd-ctrl-row">
                  <span className="sd-ctrl-k">Reverb</span>
                  <input type="range" min={0} max={1} step={0.01}
                         value={selectedTrack.fx?.reverb ?? 0}
                         onChange={(e) => selectedBusId && setTrackReverb(selectedTrack.id, selectedBusId, parseFloat(e.target.value))} />
                  <span className="sd-ctrl-v">{Math.round((selectedTrack.fx?.reverb ?? 0) * 100)}</span>
                </div>
              </div>

              {/* Version cycling — visible when this track has prior generations */}
              {(selectedTrack.metadata?.versions?.length || 0) > 1 && (
                <>
                  <div className="sd-label">Version</div>
                  <div className="sd-ctrl-block">
                    <div className="sd-ctrl-row">
                      <button className="sd-btn ghost" style={{ padding: '2px 8px' }}
                              onClick={() => cycleVersion(-1)}>◀</button>
                      <span className="sd-ctrl-v" style={{ textAlign: 'center' }}>
                        {(selectedTrack.metadata?.currentVersionIndex ?? 0) + 1}
                        {' / '}
                        {selectedTrack.metadata.versions.length}
                      </span>
                      <button className="sd-btn ghost" style={{ padding: '2px 8px' }}
                              onClick={() => cycleVersion(+1)}>▶</button>
                    </div>
                  </div>
                </>
              )}

              {/* Feedback thumbs — improves the training signal for generations */}
              {selectedTrack.metadata?.type === 'generated' && (
                <>
                  <div className="sd-label">Feedback</div>
                  <div className="sd-ctrl-block">
                    <div className="sd-ctrl-row">
                      <button className={`sd-btn ghost ${selectedTrack.metadata?.feedback === 'like' ? 'on' : ''}`}
                              onClick={() => submitFeedback('like')}>
                        <i className="fa-solid fa-thumbs-up" style={{ fontSize: 10 }} /> Like
                      </button>
                      <button className={`sd-btn ghost ${selectedTrack.metadata?.feedback === 'dislike' ? 'on' : ''}`}
                              onClick={() => submitFeedback('dislike')}>
                        <i className="fa-solid fa-thumbs-down" style={{ fontSize: 10 }} /> No
                      </button>
                    </div>
                  </div>
                </>
              )}

              <div className="sd-right-actions">
                <button className="sd-btn ghost" onClick={() => setSidebarPanel('generate')}>
                  <i className="fa-solid fa-wand-magic-sparkles" style={{ fontSize: 10 }} /> Regenerate…
                </button>
                <button className="sd-btn ghost" onClick={duplicateSelected}>
                  <i className="fa-solid fa-clone" style={{ fontSize: 10 }} /> Duplicate (⌘D)
                </button>
                <button className="sd-btn ghost" onClick={() => {
                  // Use the current state.inpaintSelection if set (from waveform),
                  // else mark this track ready to receive one in audio mode.
                  if (state.inpaintSelection?.trackId === selectedTrack.id) {
                    setSidebarPanel('generate');
                  } else {
                    dispatch({ type: 'SET_INPAINT_MODE', payload: { enabled: true, trackId: selectedTrack.id } });
                    setActiveMode('audio');
                  }
                }}>
                  <i className="fa-solid fa-eraser" style={{ fontSize: 10 }} /> Inpaint region
                </button>
                <button className="sd-btn ghost" onClick={() => {
                  dispatch({ type: 'DOWNLOAD_TRACK', payload: { trackId: selectedTrack.id } });
                  if (selectedTrack.audioUrl) {
                    const a = document.createElement('a');
                    a.href = selectedTrack.audioUrl;
                    a.download = `${selectedTrack.name || 'track'}.wav`;
                    a.click();
                  }
                }}>
                  <i className="fa-solid fa-arrow-down" style={{ fontSize: 10 }} /> Download
                </button>
                <button className="sd-btn ghost sd-btn-danger" onClick={() => {
                  if (selectedBusId) deleteTrack(selectedTrack.id, selectedBusId);
                }}>
                  <Icon k="trash" size={12} /> Delete
                </button>
              </div>
            </>
          )}
          <PipelineStatus />
        </aside>
      </div>

    </div>
  );
}
