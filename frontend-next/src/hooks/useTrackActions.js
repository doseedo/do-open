import { useCallback, useState } from 'react';
import { useApp } from '../context/AppContext';
import {
  convertAudioToMidi as apiConvertAudioToMidi,
  clarifyAudio as apiClarifyAudio,
  applyFx as apiApplyFx,
  pollFxStatus,
  pollClarifyStatus,
} from '../services/generationAPI';
import { regenStemForChord as apiRegenStemForChord } from '../services/trackAnalysisAPI';

/**
 * useTrackActions — shared wrappers for per-track backend calls. Matches
 * the handler pattern in Sidebar/RightSidebar/TrackInfoSidebar.js so that
 * /studio and /studio-dev hit the same endpoints and mutate AppContext in
 * the same way.
 *
 * Returns async functions:
 *   clarify()              — /clarify-audio, swaps audioUrl + pushes version.
 *   audioToMidi()          — /api/audio-to-midi, attaches midiData to track.
 *   applyTrumpetMute(on)   — /api/apply-fx trumpet_mute toggle; caches
 *                            originalAudioUrl + mutedAudioUrl on metadata.
 *   regenForChord(opts)    — /api/regen-stem-for-chord; returns task_id.
 *
 * Each accepts the target (trackId, busId) up front. Callers supply
 * loading-state UI if they want.
 */
export default function useTrackActions({ track, busId } = {}) {
  const { state, dispatch } = useApp();
  const [busy, setBusy] = useState({ clarify: false, mute: false, a2m: false, regen: false });

  // Resolve bus id if caller didn't pass one.
  const resolveBusId = useCallback((t) => {
    if (busId) return busId;
    if (!t) return null;
    for (const b of state.buses || []) {
      if ((b.tracks || []).some((x) => x.id === t.id)) return b.id;
    }
    return null;
  }, [state.buses, busId]);

  // Utility: audioUrl → File (for endpoints that want multipart).
  const fetchAsFile = useCallback(async (t) => {
    if (t.audioFile instanceof File) return t.audioFile;
    if (!t.audioUrl) throw new Error('track has no audioUrl');
    const r = await fetch(t.audioUrl);
    const b = await r.blob();
    return new File([b], t.name || 'track.wav', { type: b.type || 'audio/wav' });
  }, []);

  const clarify = useCallback(async () => {
    const t = track;
    if (!t) return;
    const group    = t.metadata?.params?.instrumentGroup    || t.instrumentGroup;
    const subgroup = t.metadata?.params?.instrumentSubgroup || t.instrumentSubgroup;
    if (!group || !subgroup) throw new Error('track has no instrument group/subgroup');
    const trackBusId = resolveBusId(t);
    if (!trackBusId) throw new Error('bus not found');

    setBusy((b) => ({ ...b, clarify: true }));
    try {
      const initial = await apiClarifyAudio(t.audioUrl, group, subgroup);
      let result = initial;
      if (result.status === 'processing' && result.task_id) {
        for (let i = 0; i < 60; i++) {
          await new Promise((r) => setTimeout(r, 2000));
          const s = await pollClarifyStatus(result.task_id);
          if (s.status === 'completed') { result = s; break; }
          if (s.status === 'failed') throw new Error(s.error || 'clarification failed');
        }
      }
      if (!result.clarified_url) throw new Error('no clarified_url returned');
      const clarifiedUrl = result.clarified_url.startsWith('http') || result.clarified_url.startsWith('blob:')
        ? result.clarified_url
        : `https://doseedo.com${result.clarified_url}`;

      // Push into version history (same shape as TrackInfoSidebar).
      let versions = t.metadata?.versions || [];
      if (versions.length === 0) {
        versions = [{
          audioUrl: t.audioUrl, timestamp: Date.now(),
          type: 'original', name: 'Original', params: t.metadata?.params || {},
        }];
      }
      versions = [...versions, {
        audioUrl: clarifiedUrl, timestamp: Date.now(),
        type: 'clarified',
        name: `Clarified ${versions.filter((v) => v.type === 'clarified').length + 1}`,
        params: t.metadata?.params || {},
      }];

      dispatch({
        type: 'UPDATE_TRACK',
        payload: {
          busId: trackBusId, trackId: t.id,
          updates: {
            audioUrl: clarifiedUrl,
            metadata: { ...t.metadata, versions, currentVersionIndex: versions.length - 1 },
          },
        },
      });
      return clarifiedUrl;
    } finally {
      setBusy((b) => ({ ...b, clarify: false }));
    }
  }, [track, resolveBusId, dispatch]);

  const audioToMidi = useCallback(async ({ detailedMode = false } = {}) => {
    const t = track;
    if (!t) return;
    const trackBusId = resolveBusId(t);
    if (!trackBusId) throw new Error('bus not found');

    setBusy((b) => ({ ...b, a2m: true }));
    try {
      const file = await fetchAsFile(t);
      const bpm = state.bpm || 120;
      const res = await apiConvertAudioToMidi(file, bpm, detailedMode);
      // res has { midi_url, notes, metadata } — attach to the track so
      // the MIDI canvas can pick it up.
      dispatch({
        type: 'UPDATE_TRACK',
        payload: {
          busId: trackBusId, trackId: t.id,
          updates: {
            metadata: {
              ...(t.metadata || {}),
              midi: res.midi_url || null,
              inputFiles: { ...(t.metadata?.inputFiles || {}), midiPath: res.midi_url || null },
            },
            ...(res.notes ? { midiData: { notes: res.notes, duration: res.metadata?.duration || t.duration || 4, tempo: bpm } } : {}),
          },
        },
      });
      return res;
    } finally {
      setBusy((b) => ({ ...b, a2m: false }));
    }
  }, [track, resolveBusId, fetchAsFile, state.bpm, dispatch]);

  // Trumpet-mute FX toggle. Caches the muted/original URLs on metadata so
  // subsequent toggles are instant (no re-render roundtrip).
  const applyTrumpetMute = useCallback(async (enable) => {
    const t = track;
    if (!t) return;
    const trackBusId = resolveBusId(t);
    if (!trackBusId) throw new Error('bus not found');

    if (enable) {
      if (t.metadata?.mutedAudioUrl) {
        dispatch({
          type: 'UPDATE_TRACK',
          payload: {
            busId: trackBusId, trackId: t.id,
            updates: {
              audioUrl: t.metadata.mutedAudioUrl,
              metadata: {
                ...t.metadata,
                isMuted: true,
                originalAudioUrl: t.metadata.originalAudioUrl || t.audioUrl,
              },
            },
          },
        });
        return t.metadata.mutedAudioUrl;
      }
      setBusy((b) => ({ ...b, mute: true }));
      try {
        const originalUrl = t.audioUrl;
        const initial = await apiApplyFx(originalUrl, 'trumpet_mute');
        let fxResult = initial;
        if (initial.status === 'processing' && initial.task_id) {
          for (let i = 0; i < 60; i++) {
            await new Promise((r) => setTimeout(r, 2000));
            const s = await pollFxStatus(initial.task_id);
            if (s.status === 'completed') { fxResult = s; break; }
            if (s.status === 'failed') throw new Error(s.error || 'mute failed');
          }
        }
        if (!fxResult.fx_url) throw new Error('no fx_url returned');
        const mutedUrl = fxResult.fx_url.startsWith('http') || fxResult.fx_url.startsWith('blob:')
          ? fxResult.fx_url
          : `https://doseedo.com${fxResult.fx_url}`;
        dispatch({
          type: 'UPDATE_TRACK',
          payload: {
            busId: trackBusId, trackId: t.id,
            updates: {
              audioUrl: mutedUrl,
              metadata: {
                ...t.metadata,
                isMuted: true,
                originalAudioUrl: originalUrl,
                mutedAudioUrl: mutedUrl,
              },
            },
          },
        });
        return mutedUrl;
      } finally {
        setBusy((b) => ({ ...b, mute: false }));
      }
    } else {
      const originalUrl = t.metadata?.originalAudioUrl;
      if (!originalUrl) return null;
      dispatch({
        type: 'UPDATE_TRACK',
        payload: {
          busId: trackBusId, trackId: t.id,
          updates: {
            audioUrl: originalUrl,
            metadata: { ...t.metadata, isMuted: false },
          },
        },
      });
      return originalUrl;
    }
  }, [track, resolveBusId, dispatch]);

  // Regen a stem for a chord change. Returns { task_id } — callers should
  // poll /api/generate-stemphonic/task/<task_id> and swap audioUrl on
  // success (same pattern as useAutoRepaintMeter).
  const regenForChord = useCallback(async ({ oldChord, newChord, regionStart = 0, regionEnd = null, coverNoise = 0.7, prompt } = {}) => {
    const t = track;
    if (!t) return;
    const trackBusId = resolveBusId(t);
    if (!trackBusId) throw new Error('bus not found');
    const role = t.metadata?.stemType || t.metadata?.instrument || 'harmony';

    setBusy((b) => ({ ...b, regen: true }));
    try {
      const file = await fetchAsFile(t);
      const midiFile = null;
      return await apiRegenStemForChord({
        audioFile: file, midiFile, role,
        oldChord, newChord, regionStart, regionEnd, coverNoise, prompt,
        duration: t.duration,
      });
    } finally {
      setBusy((b) => ({ ...b, regen: false }));
    }
  }, [track, resolveBusId, fetchAsFile]);

  return { clarify, audioToMidi, applyTrumpetMute, regenForChord, busy };
}
