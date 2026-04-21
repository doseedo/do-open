import { useEffect, useRef, useState } from 'react';
import { useApp } from '../context/AppContext';
import { repaintMeter } from '../services/trackAnalysisAPI';

// Flip to true to re-enable the backend stemphonic repaint path. Virtual-
// edit playback in useAudioPlayback.js handles meter change instantly
// against the original source buffer with zero pitch shift. The backend
// repaint is only useful as a diffusion-smoothed sweetener.
const ENABLE_BACKEND_REPAINT = false;

/**
 * useAutoRepaintMeter — when bpm / time-signature changes, re-stemphonic
 * every track that has a cached VAE latent, then swap its audioUrl once
 * the backend task completes.
 *
 * Debounced 1.2s so scrubbing the BPM input doesn't fire a flood of jobs.
 * The hook reads state.bpm / beatsPerBar / meterDenominator and dispatches
 * UPDATE_TRACK per completed task — no caller input needed.
 *
 * Both /studio (TempoControls.js) and /studio-dev (StudioDev.js) call this
 * hook so the two routes stay in lockstep. When ENABLE_BACKEND_REPAINT is
 * false the hook is inert — virtual-edit playback owns the meter change.
 */
export default function useAutoRepaintMeter() {
  const { state, dispatch } = useApp();
  const bpm = state.bpm;
  const beatsPerBar = state.beatsPerBar || 4;
  const meterDen = state.meterDenominator || 4;

  const snapRef = useRef({ bpm, beatsPerBar, meterDen });
  const debounceRef = useRef(null);
  const inFlightRef = useRef(false);
  const [applying, setApplying] = useState(false);

  useEffect(() => {
    if (!ENABLE_BACKEND_REPAINT) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    const snap = snapRef.current;
    if (bpm === snap.bpm && beatsPerBar === snap.beatsPerBar && meterDen === snap.meterDen) return;

    debounceRef.current = setTimeout(async () => {
      if (inFlightRef.current) return;

      const stems = [];
      const trackRefs = [];
      (state.buses || []).forEach((bus) => {
        (bus.tracks || []).forEach((track) => {
          const latentId = track.metadata?.latentId;
          if (!latentId) return;
          stems.push({
            latent_id: latentId,
            stem_type: track.metadata?.instrument || track.metadata?.stemType || 'other',
          });
          trackRefs.push({ busId: bus.id, trackId: track.id });
        });
      });

      if (!stems.length) {
        snapRef.current = { bpm, beatsPerBar, meterDen };
        return;
      }

      inFlightRef.current = true;
      setApplying(true);
      try {
        const result = await repaintMeter({
          stems,
          srcMeter: [snap.beatsPerBar, snap.meterDen],
          tgtMeter: [beatsPerBar, meterDen],
          srcBpm: snap.bpm,
          tgtBpm: bpm,
          coverNoise: 0.55,
          prompt: 'preserve original style and instrument timbre',
        });

        (result.results || []).forEach((r, i) => {
          if (r.error || !r.task_id) return;
          const ref = trackRefs[i];
          (async () => {
            for (let p = 0; p < 120; p++) {
              await new Promise((res) => setTimeout(res, 2500));
              const tr = await fetch(`/api/generate-stemphonic/task/${r.task_id}`);
              if (!tr.ok) continue;
              const td = await tr.json();
              if (td.state === 'SUCCESS' && td.result?.file_paths?.[0]) {
                dispatch({
                  type: 'UPDATE_TRACK',
                  payload: {
                    busId: ref.busId,
                    trackId: ref.trackId,
                    updates: {
                      audioUrl: td.result.file_paths[0],
                      metadata: { latentId: r.new_latent_id },
                    },
                  },
                });
                return;
              }
              if (td.state === 'FAILURE') return;
            }
          })();
        });

        snapRef.current = { bpm, beatsPerBar, meterDen };
      } catch (err) {
        console.warn('[auto-repaint-meter] failed:', err?.message || err);
      } finally {
        inFlightRef.current = false;
        setApplying(false);
      }
    }, 1200);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bpm, beatsPerBar, meterDen]);

  return { applying };
}
