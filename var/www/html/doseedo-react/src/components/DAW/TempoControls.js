import React, { useState, useRef, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { repaintMeter } from '../../services/trackAnalysisAPI';
import styles from './TempoControls.module.css';

/**
 * TempoControls Component - master tempo + meter for the studio.
 *
 * Layout (left → right):
 *   [metronome] [BPM input] [Meter select] [repaint spinner]
 *
 * The timeline now defaults to bars/beats mode (no toggle button).
 *
 * BPM/meter changes auto-trigger a stemphonic repaint of every track
 * with a cached VAE latent, after a 1.2s debounce so the user can
 * scrub the BPM input without firing 100 jobs.
 */
function TempoControls() {
  const { state, dispatch } = useApp();

  const bpm = state.bpm;
  const beatsPerBar = state.beatsPerBar || 4;
  const meterDen = state.meterDenominator || 4;
  const meterStr = `${beatsPerBar}/${meterDen}`;
  const isMetronomeOn = state.isMetronomeOn;

  // Snapshot of the last bpm/meter we successfully repainted from.
  // The next auto-trigger will use these as the source meter.
  const lastAppliedRef = useRef({ bpm, beatsPerBar, meterDen });
  const debounceRef = useRef(null);
  const inFlightRef = useRef(false);
  const [applying, setApplying] = useState(false);
  const [lastResult, setLastResult] = useState(null);

  const handleBPMChange = (e) => {
    dispatch({ type: 'UPDATE_BPM', payload: parseInt(e.target.value, 10) });
  };

  const handleMeterChange = (e) => {
    dispatch({ type: 'SET_METER', payload: e.target.value });
  };

  const toggleMetronome = () => dispatch({ type: 'TOGGLE_METRONOME' });

  // Auto-repaint dispatcher
  const runRepaint = async () => {
    if (inFlightRef.current) return;
    const { bpm: srcBpm, beatsPerBar: srcBeats, meterDen: srcDen } = lastAppliedRef.current;
    const tgtBpm = bpm;
    const tgtBeats = beatsPerBar;
    const tgtDen = meterDen;
    if (srcBpm === tgtBpm && srcBeats === tgtBeats && srcDen === tgtDen) return;

    // Collect every track with a cached latent
    const stems = [];
    const trackRefs = [];
    state.buses?.forEach((bus) => {
      bus.tracks?.forEach((track) => {
        const latentId = track.metadata?.latentId;
        if (!latentId) return;
        stems.push({
          latent_id: latentId,
          stem_type: track.metadata?.instrument || track.metadata?.stemType || 'other',
        });
        trackRefs.push({ busId: bus.id, trackId: track.id });
      });
    });

    if (stems.length === 0) {
      // No latents yet — silently bump the snapshot so we don't keep retrying
      lastAppliedRef.current = { bpm: tgtBpm, beatsPerBar: tgtBeats };
      return;
    }

    inFlightRef.current = true;
    setApplying(true);
    try {
      const result = await repaintMeter({
        stems,
        srcMeter: [srcBeats, srcDen],
        tgtMeter: [tgtBeats, tgtDen],
        srcBpm,
        tgtBpm,
        coverNoise: 0.55,
        prompt: 'preserve original style and instrument timbre',
      });
      console.log(`🎚️ auto-repaint ${srcBeats}/${srcDen} ${srcBpm} → ${tgtBeats}/${tgtDen} ${tgtBpm} for ${stems.length} stems`, result);
      setLastResult(`${stems.length} stems → ${tgtBeats}/${tgtDen} ${tgtBpm} BPM`);

      // Poll each task and swap track audioUrl when done
      (result.results || []).forEach((r, i) => {
        if (r.error || !r.task_id) return;
        const ref = trackRefs[i];
        const taskId = r.task_id;
        (async () => {
          for (let p = 0; p < 120; p++) {
            await new Promise((res) => setTimeout(res, 2500));
            const tr = await fetch(`/api/generate-stemphonic/task/${taskId}`);
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

      lastAppliedRef.current = { bpm: tgtBpm, beatsPerBar: tgtBeats, meterDen: tgtDen };
    } catch (err) {
      console.error('auto-repaint failed', err);
      setLastResult(`error: ${err.message}`);
    } finally {
      inFlightRef.current = false;
      setApplying(false);
    }
  };

  // Debounce: when BPM or meter changes, schedule a repaint after 1.2s
  // of inactivity so dragging the BPM input doesn't fire 100 jobs.
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    const { bpm: srcBpm, beatsPerBar: srcBeats, meterDen: srcDen } = lastAppliedRef.current;
    if (bpm === srcBpm && beatsPerBar === srcBeats && meterDen === srcDen) return;
    debounceRef.current = setTimeout(runRepaint, 1200);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bpm, beatsPerBar, meterDen]);

  return (
    <div className={styles.container} style={{ float: 'right', display: 'flex', alignItems: 'center', gap: 6 }}>
      <button
        id="metronome-btn"
        onClick={toggleMetronome}
        title="Toggle Metronome"
        className={`${styles.button} ${isMetronomeOn ? styles.active : ''}`}
      >
        <i className="fa-solid fa-drum"></i>
      </button>

      <div className={styles.bpmInputContainer}>
        <label htmlFor="bpm-input">BPM</label>
        <input
          type="number"
          id="bpm-input"
          className={styles.bpmInput}
          min="40"
          max="240"
          value={bpm}
          onChange={handleBPMChange}
        />
      </div>

      <div className={styles.bpmInputContainer} title="Time signature">
        <label htmlFor="meter-input">Meter</label>
        <select
          id="meter-input"
          className={styles.bpmInput}
          value={meterStr}
          onChange={handleMeterChange}
          style={{ minWidth: 60 }}
        >
          <option value="3/4">3/4</option>
          <option value="4/4">4/4</option>
          <option value="5/4">5/4</option>
          <option value="6/8">6/8</option>
          <option value="7/8">7/8</option>
        </select>
      </div>

      {applying && (
        <span
          className={styles.button}
          title="Repainting tracks via stemphonic stage2d-130k"
          style={{ pointerEvents: 'none' }}
        >
          <i className="fa-solid fa-wand-magic-sparkles fa-spin" style={{ color: '#8B7FF0' }}></i>
        </span>
      )}
    </div>
  );
}

export default TempoControls;
