/*
 * StudioDevWaveform — themed audio-waveform view for /studio-dev.
 *
 * Replaces the original components/AudioWaveform. Renders the selected
 * track's audio (audioUrl or audioFile) as a canvas-drawn hi-fi waveform
 * in the track color. Click anywhere to seek; click-and-drag to select a
 * region (used for inpaint / loop / crop). Scrollwheel + modifier for
 * zoom/pan.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useApp } from '../../context/AppContext';

const C = {
  bg: '#1f1a15', surf: '#2a241d', surf2: '#36302a',
  ink: '#e8dfc8', inkSoft: 'rgba(232,223,200,0.68)', inkMute: 'rgba(232,223,200,0.42)',
  rule: 'rgba(232,223,200,0.10)', ruleStrong: 'rgba(232,223,200,0.24)',
  accent: '#a88adc',
};

const TRACK_COLORS = {
  vocals: '#a88adc', lead: '#a88adc',
  rhodes: '#e8c88a', piano: '#e8c88a', keys: '#e8c88a',
  bass: '#8ac8a0', drums: '#e07556',
  strings: '#6aa8e8', guitar: '#6aa8e8', other: '#a88adc',
};
function colorFor(type = '') {
  const t = type.toLowerCase();
  for (const [k, v] of Object.entries(TRACK_COLORS)) if (t.includes(k)) return v;
  return TRACK_COLORS.other;
}

// Shared AudioContext + decoded-buffer cache so hopping between tracks
// doesn't re-download+decode the same stem every time.
const BUFFER_CACHE = new Map(); // audioUrl → AudioBuffer
function getCtx() {
  if (!window.__sdWaveformCtx) {
    window.__sdWaveformCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  return window.__sdWaveformCtx;
}
async function loadBuffer(track) {
  const url = track.audioUrl;
  if (!url) return null;
  if (BUFFER_CACHE.has(url)) return BUFFER_CACHE.get(url);
  const ctx = getCtx();
  let arrayBuf;
  if (track.audioFile instanceof File) {
    arrayBuf = await track.audioFile.arrayBuffer();
  } else {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetch ${url} → ${r.status}`);
    arrayBuf = await r.arrayBuffer();
  }
  const buf = await ctx.decodeAudioData(arrayBuf);
  BUFFER_CACHE.set(url, buf);
  return buf;
}

// Peak pairs (min/max) per pixel for fast waveform rendering at any zoom.
function buildPeaks(buffer, targetBins) {
  if (!buffer) return null;
  const ch0 = buffer.getChannelData(0);
  const ch1 = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : ch0;
  const n = ch0.length;
  const step = Math.max(1, Math.floor(n / targetBins));
  const mins = new Float32Array(targetBins);
  const maxs = new Float32Array(targetBins);
  for (let i = 0; i < targetBins; i++) {
    const s = i * step;
    const e = Math.min(n, s + step);
    let lo = Infinity, hi = -Infinity;
    for (let j = s; j < e; j++) {
      const v = (ch0[j] + ch1[j]) * 0.5;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    mins[i] = lo === Infinity ? 0 : lo;
    maxs[i] = hi === -Infinity ? 0 : hi;
  }
  return { mins, maxs, duration: buffer.duration, sr: buffer.sampleRate };
}

export default function StudioDevWaveform() {
  const { state, dispatch } = useApp();
  const selectedTrack = state.selectedTrack;
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const [size, setSize] = useState({ w: 800, h: 220 });
  const [peaks, setPeaks] = useState(null);     // {mins, maxs, duration, sr}
  const [loadErr, setLoadErr] = useState(null);
  const [zoom, setZoom] = useState(1);           // 1 = fit-to-window
  const [scrollX, setScrollX] = useState(0);     // seconds
  const [selection, setSelection] = useState(null); // {startSec, endSec}
  const [drag, setDrag] = useState(null);
  const [hoverSec, setHoverSec] = useState(null);

  const type = (selectedTrack?.metadata?.stemType
             || selectedTrack?.metadata?.instrument
             || selectedTrack?.name || '').toLowerCase();
  const trackColor = colorFor(type);

  const busIdForSelected = useMemo(() => {
    if (!selectedTrack) return null;
    for (const bus of state.buses || []) {
      if ((bus.tracks || []).some((t) => t.id === selectedTrack.id)) return bus.id;
    }
    return null;
  }, [state.buses, selectedTrack]);

  // Load + decode on track change.
  useEffect(() => {
    setPeaks(null); setLoadErr(null); setSelection(null); setScrollX(0); setZoom(1);
    if (!selectedTrack?.audioUrl) return;
    let alive = true;
    (async () => {
      try {
        const buf = await loadBuffer(selectedTrack);
        if (!alive || !buf) return;
        // 4× the canvas width → crisp at 4× zoom, still cheap to recompute.
        const bins = Math.max(800, (wrapRef.current?.clientWidth || 800) * 4);
        setPeaks(buildPeaks(buf, bins));
      } catch (e) {
        if (alive) setLoadErr(e?.message || String(e));
      }
    })();
    return () => { alive = false; };
  }, [selectedTrack?.audioUrl, selectedTrack?.id]);   // eslint-disable-line

  // Sizing.
  useEffect(() => {
    const el = wrapRef.current; if (!el) return;
    const ro = new ResizeObserver(() => {
      setSize({ w: el.clientWidth, h: el.clientHeight });
    });
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  const duration = peaks?.duration || 0;
  const visibleSec = duration / zoom;
  const pxPerSec = visibleSec > 0 ? size.w / visibleSec : 0;
  const secAtX = (x) => scrollX + x / pxPerSec;

  // Draw.
  useEffect(() => {
    const c = canvasRef.current; if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = size.w * dpr; c.height = size.h * dpr;
    c.style.width = size.w + 'px'; c.style.height = size.h + 'px';
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, size.w, size.h);

    // Center line
    const midY = size.h / 2;
    ctx.fillStyle = C.rule;
    ctx.fillRect(0, midY, size.w, 1);

    // Waveform
    if (peaks) {
      const startBin = Math.floor((scrollX / duration) * peaks.mins.length);
      const endBin   = Math.min(peaks.mins.length,
                        Math.ceil(((scrollX + visibleSec) / duration) * peaks.mins.length));
      const binsShown = Math.max(1, endBin - startBin);
      const pxPerBin = size.w / binsShown;
      ctx.fillStyle = trackColor + '66';   // soft fill
      ctx.strokeStyle = trackColor;
      ctx.lineWidth = 1;
      // Draw as solid blocks from min→max
      for (let i = 0; i < binsShown; i++) {
        const x = i * pxPerBin;
        const lo = peaks.mins[startBin + i] || 0;
        const hi = peaks.maxs[startBin + i] || 0;
        const y1 = midY - hi * (size.h * 0.48);
        const y2 = midY - lo * (size.h * 0.48);
        const h = Math.max(1, y2 - y1);
        ctx.fillRect(x, y1, Math.max(1, pxPerBin - 0.5), h);
      }
      // Top/bottom outline for crispness
      ctx.strokeStyle = trackColor;
      ctx.beginPath();
      for (let i = 0; i < binsShown; i++) {
        const x = i * pxPerBin;
        const hi = peaks.maxs[startBin + i] || 0;
        const y1 = midY - hi * (size.h * 0.48);
        if (i === 0) ctx.moveTo(x, y1); else ctx.lineTo(x, y1);
      }
      ctx.stroke();
    }

    // Persisted inpaint region (set via track-info sidebar's "Inpaint region")
    const ip = state.inpaintSelection;
    if (ip && ip.trackId === selectedTrack?.id && duration > 0) {
      const a = Math.min(ip.startTime, ip.endTime);
      const b = Math.max(ip.startTime, ip.endTime);
      const x1 = Math.max(0, (a - scrollX) * pxPerSec);
      const x2 = Math.min(size.w, (b - scrollX) * pxPerSec);
      if (x2 > x1) {
        ctx.fillStyle = 'rgba(224,117,86,0.12)';                // rust-tinted
        ctx.fillRect(x1, 0, x2 - x1, size.h);
        ctx.strokeStyle = 'rgba(224,117,86,0.85)';
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 1.2;
        ctx.strokeRect(x1, 0, x2 - x1, size.h - 1);
        ctx.setLineDash([]);
        ctx.fillStyle = '#e07556';
        ctx.font = '10px "JetBrains Mono", ui-monospace, monospace';
        ctx.fillText('INPAINT', x1 + 4, size.h - 6);
      }
    }

    // Selection region
    if (selection && duration > 0) {
      const a = Math.min(selection.startSec, selection.endSec);
      const b = Math.max(selection.startSec, selection.endSec);
      const x1 = Math.max(0, (a - scrollX) * pxPerSec);
      const x2 = Math.min(size.w, (b - scrollX) * pxPerSec);
      if (x2 > x1) {
        ctx.fillStyle = 'rgba(168,138,220,0.15)';
        ctx.fillRect(x1, 0, x2 - x1, size.h);
        ctx.fillStyle = C.accent;
        ctx.fillRect(x1, 0, 1, size.h);
        ctx.fillRect(x2 - 1, 0, 1, size.h);
      }
    }

    // Ruler ticks (every second, labels at bigger intervals)
    ctx.fillStyle = C.inkMute;
    ctx.font = '10px "JetBrains Mono", ui-monospace, monospace';
    const tickStep = visibleSec < 4 ? 0.25 : visibleSec < 20 ? 1 : visibleSec < 60 ? 5 : 10;
    const labelStep = visibleSec < 4 ? 1 : visibleSec < 20 ? 5 : visibleSec < 60 ? 10 : 30;
    for (let t = Math.floor(scrollX / tickStep) * tickStep; t < scrollX + visibleSec; t += tickStep) {
      if (t < 0) continue;
      const x = (t - scrollX) * pxPerSec;
      ctx.fillStyle = C.rule;
      ctx.fillRect(x, 0, 1, 6);
      ctx.fillRect(x, size.h - 6, 1, 6);
      if (t % labelStep === 0) {
        ctx.fillStyle = C.inkMute;
        ctx.fillText(`${t.toFixed(0)}s`, x + 3, 12);
      }
    }

    // Playhead
    const ph = (state.playheadPosition || 0);
    if (ph >= scrollX && ph <= scrollX + visibleSec) {
      const x = (ph - scrollX) * pxPerSec;
      ctx.fillStyle = C.accent;
      ctx.fillRect(x, 0, 1, size.h);
    }
  }, [size, peaks, trackColor, selection, scrollX, visibleSec, pxPerSec, duration, state.playheadPosition, state.inpaintSelection, selectedTrack?.id]);

  // --- Mouse: click-to-seek, drag-to-select region. ---
  const onMouseDown = (e) => {
    if (!duration) return;
    e.preventDefault();
    const r = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const t0 = Math.max(0, Math.min(duration, secAtX(mx)));
    setDrag({ startClientX: e.clientX, startSec: t0, r });
    // click-to-seek: update playhead immediately on click.
    dispatch({ type: 'SEEK_TO', payload: t0 });
  };
  useEffect(() => {
    if (!drag) return;
    const onMove = (e) => {
      const mx = e.clientX - drag.r.left;
      const tNow = Math.max(0, Math.min(duration, secAtX(mx)));
      if (Math.abs(e.clientX - drag.startClientX) > 3) {
        setSelection({ startSec: drag.startSec, endSec: tNow });
      }
    };
    const onUp = () => setDrag(null);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [drag, duration, pxPerSec, scrollX]);   // eslint-disable-line

  const onMouseMove = (e) => {
    const r = canvasRef.current.getBoundingClientRect();
    setHoverSec(Math.max(0, Math.min(duration, secAtX(e.clientX - r.left))));
  };

  const onWheel = (e) => {
    if (!duration) return;
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const pointerT = secAtX(e.clientX - canvasRef.current.getBoundingClientRect().left);
      setZoom((z) => {
        const nz = Math.max(1, Math.min(64, z * (e.deltaY < 0 ? 1.2 : 1 / 1.2)));
        // Keep the pointer's time fixed in place.
        const newVis = duration / nz;
        setScrollX(Math.max(0, Math.min(duration - newVis, pointerT - (e.clientX - canvasRef.current.getBoundingClientRect().left) / (size.w / newVis))));
        return nz;
      });
    } else if (e.shiftKey || Math.abs(e.deltaX) > 0) {
      e.preventDefault();
      setScrollX((v) => Math.max(0, Math.min(duration - visibleSec, v + (e.deltaX || e.deltaY) / pxPerSec)));
    }
  };

  // --- Action buttons. ---
  const setTrackVolume = (g) => {
    if (!selectedTrack || !busIdForSelected) return;
    dispatch({ type: 'UPDATE_TRACK_GAIN', payload: { trackId: selectedTrack.id, busId: busIdForSelected, gain: g } });
  };
  const onInpaint = () => {
    if (!selection || !selectedTrack) return;
    const a = Math.min(selection.startSec, selection.endSec);
    const b = Math.max(selection.startSec, selection.endSec);
    dispatch({
      type: 'SET_INPAINT_SELECTION',
      payload: { trackId: selectedTrack.id, startTime: a, endTime: b },
    });
    console.log(`[studio-dev] inpaint region set ${a.toFixed(2)}→${b.toFixed(2)}s`);
  };
  const onCrop = () => {
    // Simple crop stub — writes new cropStart/cropEnd on the track so the
    // production audio engine trims playback on reload.
    if (!selection || !selectedTrack || !busIdForSelected) return;
    const a = Math.min(selection.startSec, selection.endSec);
    const b = Math.max(selection.startSec, selection.endSec);
    dispatch({
      type: 'UPDATE_TRACK_PROPS',
      payload: { trackId: selectedTrack.id, cropStart: a, cropEnd: b },
    });
  };
  const onLoop = () => {
    // Toggle loop flag on track metadata.
    if (!selectedTrack || !busIdForSelected) return;
    dispatch({
      type: 'UPDATE_TRACK_PROPS',
      payload: { trackId: selectedTrack.id, loop: !selectedTrack.loop },
    });
  };

  if (!selectedTrack) {
    return (
      <div className="sd-wave-empty">
        <div className="sd-wave-empty-eyebrow">— the waveform —</div>
        <div className="sd-wave-empty-title">Pick a track.</div>
        <div className="sd-wave-empty-body">
          Select an audio track from the timeline or upload a file to see its waveform.
        </div>
      </div>
    );
  }
  if (!selectedTrack.audioUrl) {
    return (
      <div className="sd-wave-empty">
        <div className="sd-wave-empty-eyebrow">— no audio on this track —</div>
        <div className="sd-wave-empty-title">Record or import audio.</div>
        <div className="sd-wave-empty-body">
          This is a MIDI-only track. Switch to the MIDI mode to edit notes, or drop audio onto it.
        </div>
      </div>
    );
  }

  return (
    <div className="sd-wave">
      <div className="sd-midi-toolbar">
        <div className="sd-midi-title">
          <span className="sd-midi-color" style={{ background: trackColor }} />
          <span className="sd-midi-name">{selectedTrack.name || selectedTrack.id}</span>
          <span className="sd-midi-meta">
            {peaks ? `${duration.toFixed(2)}s · ${peaks.sr/1000}kHz` : loadErr ? 'failed' : 'decoding…'}
            {selection ? ` · selection ${(Math.abs(selection.endSec - selection.startSec)).toFixed(2)}s` : ''}
          </span>
        </div>
        <div className="sd-midi-spacer" />
        <div className="sd-midi-group">
          <span className="sd-midi-kv-k">Zoom</span>
          <button className="sd-midi-btn" onClick={() => setZoom((z) => Math.max(1, z / 1.5))}>−</button>
          <button className="sd-midi-btn" onClick={() => setZoom((z) => Math.min(64, z * 1.5))}>+</button>
        </div>
        <div className="sd-midi-group">
          <button className="sd-midi-btn" onClick={onInpaint} disabled={!selection}>Inpaint region</button>
          <button className="sd-midi-btn" onClick={onCrop} disabled={!selection}>Crop</button>
          <button className="sd-midi-btn" onClick={onLoop}>{selectedTrack.loop ? 'Loop on' : 'Loop'}</button>
          <button className="sd-midi-btn sd-midi-danger" onClick={() => setSelection(null)} disabled={!selection}>Clear sel</button>
        </div>
      </div>

      <div
        ref={wrapRef}
        className="sd-midi-canvas-wrap"
        onWheel={onWheel}
      >
        <canvas
          ref={canvasRef}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          className="sd-wave-canvas"
        />
      </div>

      <div className="sd-midi-footer">
        <span className="sd-midi-kv-k">Cursor</span>
        <span className="sd-midi-kv-v">{hoverSec != null ? `${hoverSec.toFixed(2)}s` : '—'}</span>
        <div className="sd-midi-spacer" />
        <span className="sd-midi-kv-k">Gain</span>
        <input type="range" min={0} max={1} step={0.01}
               value={selectedTrack.gain ?? 1}
               onChange={(e) => setTrackVolume(parseFloat(e.target.value))}
               style={{ width: 120, accentColor: C.accent }} />
        <span className="sd-midi-kv-v">{Math.round((selectedTrack.gain ?? 1) * 100)}</span>
      </div>
    </div>
  );
}
