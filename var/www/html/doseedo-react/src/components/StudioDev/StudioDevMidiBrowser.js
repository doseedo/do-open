/*
 * StudioDevMidiBrowser — themed MIDI file browser for /studio-dev.
 *
 * Functional replacement for components/MIDIBrowser. Fetches the server's
 * MIDI library (/api/list-midi-files), searches locally, and on load
 * drops the file onto the timeline as either a single track or (for
 * multitrack MIDI) one track per part — matching the original behaviour.
 *
 * Styled in the hi-fi purple palette so it fits the sidebar / overlay.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { parseMIDIFile } from '../../utils/midiParser';

const TRACK_COLORS_CYCLE = ['#a88adc', '#e8c88a', '#8ac8a0', '#e07556', '#6aa8e8', '#c1abe8'];

export default function StudioDevMidiBrowser({ onClose }) {
  const { state, dispatch } = useApp();
  const [files, setFiles] = useState([]);
  const [term, setTerm] = useState('');
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [loadingFile, setLoadingFile] = useState(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      setLoading(true); setError(null);
      try {
        const r = await fetch('/api/list-midi-files');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        if (!alive) return;
        setFiles(Array.isArray(data.files) ? data.files : []);
      } catch (e) {
        if (alive) setError('MIDI library unavailable. Upload a .mid manually in the sidebar.');
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, []);

  const filtered = useMemo(() => {
    const q = term.trim().toLowerCase();
    if (!q) return files;
    return files.filter((f) => f.toLowerCase().includes(q));
  }, [term, files]);

  const loadFile = useCallback(async (filename) => {
    setLoadingFile(filename);
    try {
      const r = await fetch(`/api/get-midi-file/${encodeURIComponent(filename)}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const blob = await r.blob();
      const file = new File([blob], filename, { type: 'audio/midi' });
      const midiData = await parseMIDIFile(file);

      // Find a Music bus or auto-create one.
      const musicBus = state.buses?.find((b) => (b.type || '').toLowerCase() === 'music');
      let busId = musicBus?.id;
      if (!busId) {
        busId = `music-${Date.now()}`;
        dispatch({
          type: 'CREATE_BUS',
          payload: { id: busId, type: 'MUSIC', name: 'Music', expanded: true },
        });
      }

      const base = filename.replace(/\.midi?$/i, '');
      const parts = (midiData.tracks || []).filter((t) => t.notes?.length > 0);

      if (parts.length > 1) {
        parts.forEach((midiTrack, i) => {
          const trackId = `track-${Date.now()}-${i}`;
          dispatch({
            type: 'ADD_TRACK',
            payload: {
              busId,
              track: {
                id: trackId,
                name: `${base} · ${i + 1}`,
                type: 'midi',
                midiData: {
                  ...midiData,
                  notes: midiTrack.notes,
                  trackIndex: i,
                  isMultitrack: true,
                  allTracks: midiData.tracks,
                },
                file, duration: midiData.duration, startPosition: 0,
                gain: 1, isMuted: false, isSolo: false, pan: 0,
                cropStart: 0, cropEnd: 0,
                metadata: { type: 'midi' },
              },
            },
          });
        });
      } else {
        const trackId = `track-${Date.now()}`;
        dispatch({
          type: 'ADD_TRACK',
          payload: {
            busId,
            track: {
              id: trackId,
              name: base,
              type: 'midi',
              midiData,
              file, duration: midiData.duration, startPosition: 0,
              gain: 1, isMuted: false, isSolo: false, pan: 0,
              cropStart: 0, cropEnd: 0,
              metadata: { type: 'midi' },
            },
          },
        });
        dispatch({ type: 'SELECT_TRACK', payload: { trackId, busId } });
      }

      onClose?.();
    } catch (e) {
      alert(`Failed to load MIDI: ${e.message}`);
    } finally {
      setLoadingFile(null);
    }
  }, [dispatch, state.buses, onClose]);

  return (
    <div className="sd-midibrowser">
      <div className="sd-midi-toolbar">
        <div className="sd-midi-title">
          <span className="sd-midi-meta">BROWSE</span>
          <span className="sd-midi-name" style={{ marginLeft: 6 }}>MIDI library</span>
          <span className="sd-midi-meta" style={{ marginLeft: 8 }}>
            {loading ? '…' : error ? 'offline' : `${filtered.length} of ${files.length}`}
          </span>
        </div>
        <div className="sd-midi-spacer" />
        {onClose && <button className="sd-midi-btn" onClick={onClose}>Close</button>}
      </div>

      <div className="sd-midibrowser-search">
        <i className="fa-solid fa-magnifying-glass" />
        <input
          className="sd-midibrowser-search-input"
          placeholder="Search MIDI files…"
          value={term}
          onChange={(e) => setTerm(e.target.value)}
          autoFocus
        />
        {term && (
          <button className="sd-midibrowser-search-clear" onClick={() => setTerm('')}>
            <i className="fa-solid fa-xmark" />
          </button>
        )}
      </div>

      <div className="sd-midibrowser-list">
        {loading && (
          <div className="sd-midibrowser-state">
            <i className="fa-solid fa-spinner fa-spin" /> Loading MIDI files…
          </div>
        )}
        {error && (
          <div className="sd-midibrowser-state sd-midibrowser-error">
            <i className="fa-solid fa-triangle-exclamation" /> {error}
          </div>
        )}
        {!loading && !error && filtered.length === 0 && (
          <div className="sd-midibrowser-state">No files match "{term}".</div>
        )}
        {!loading && !error && filtered.map((f, i) => {
          const accent = TRACK_COLORS_CYCLE[i % TRACK_COLORS_CYCLE.length];
          const isSel = selected === f;
          const isLoadingThis = loadingFile === f;
          return (
            <button
              key={f}
              className={`sd-midibrowser-row ${isSel ? 'selected' : ''}`}
              onClick={() => setSelected(f)}
              onDoubleClick={() => loadFile(f)}
              disabled={isLoadingThis}
            >
              <span className="sd-midibrowser-chip" style={{ background: accent }} />
              <span className="sd-midibrowser-file">{f.replace(/\.midi?$/i, '')}</span>
              <span className="sd-midibrowser-ext">{f.match(/\.(midi?)$/i)?.[1] || 'mid'}</span>
              {isLoadingThis && <i className="fa-solid fa-spinner fa-spin" style={{ marginLeft: 8 }} />}
            </button>
          );
        })}
      </div>

      {selected && (
        <div className="sd-midibrowser-footer">
          <div className="sd-midibrowser-selected">
            <span className="sd-midi-kv-k">Selected</span>
            <span className="sd-midi-kv-v">{selected}</span>
          </div>
          <button className="sd-btn" onClick={() => loadFile(selected)} disabled={!!loadingFile}>
            {loadingFile ? 'Loading…' : 'Use this MIDI'}
          </button>
        </div>
      )}

      <div className="sd-midibrowser-hint">
        Click to select · double-click to load into the timeline.
      </div>
    </div>
  );
}
