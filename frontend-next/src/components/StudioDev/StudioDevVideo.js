/*
 * StudioDevVideo — themed video/reference viewer for /studio.
 * Drop a video file or MP4 URL on the track to use it as a scoring
 * reference. Playhead syncs to state.playheadPosition so audio + video stay
 * in lockstep.
 *
 * On drop, the file is also POSTed to /api/video-score (proxied to the
 * Modal video-scoring app). The response carries scene_changes,
 * per-scene VLM features, and a generated MIDI score, all of which
 * useVideoProcessing dispatches into the studio's bus/track state.
 */
import React, { useEffect, useRef } from 'react';
import { useApp } from '../../context/AppContext';
import { useVideoProcessing } from '../../hooks/useVideoProcessing';

const STATUS_COPY = {
  uploading: 'Uploading video…',
  detecting_scenes: 'Detecting scenes…',
  analyzing_scenes: 'Analyzing scenes…',
  writing_midi: 'Writing MIDI…',
  extracting_audio: 'Extracting audio…',
  cache_hit: 'Reusing previous score',
  completed: 'Score ready',
  completed_no_scenes: 'No scenes detected',
  failed: 'Scoring failed',
};

export default function StudioDevVideo() {
  const { state, dispatch } = useApp();
  const selectedTrack = state.selectedTrack;
  const videoRef = useRef(null);
  const fileInputRef = useRef(null);

  const { isProcessing, processingStatus, progress, processingError, processVideo } =
    useVideoProcessing();

  const videoUrl = selectedTrack?.metadata?.videoUrl
                || (selectedTrack?.metadata?.type === 'video_audio' ? selectedTrack?.audioUrl : null);

  // Sync video time to playhead — 50 ms tolerance keeps us frame-accurate
  // while still absorbing the natural drift between AudioContext clock and
  // HTMLVideoElement playback.
  useEffect(() => {
    const v = videoRef.current; if (!v || !videoUrl) return;
    if (Math.abs(v.currentTime - (state.playheadPosition || 0)) > 0.05) {
      try { v.currentTime = state.playheadPosition || 0; } catch (_) {}
    }
    if (state.isPlaying) v.play().catch(() => {});
    else v.pause();
  }, [state.playheadPosition, state.isPlaying, videoUrl]);

  const handleVideoFile = async (f) => {
    if (!f || !f.type.startsWith('video/')) return;

    // Local preview track first (so the picture appears instantly while
    // the backend works). The hook will additionally create MIDI/VO bus
    // tracks once /api/video-score responds.
    const url = URL.createObjectURL(f);
    const busId = `bus-video-${Date.now()}`;
    const trackId = `t-${Date.now()}`;
    dispatch({ type: 'CREATE_BUS', payload: { id: busId, type: 'VIDEO', name: f.name, expanded: true } });
    dispatch({
      type: 'ADD_TRACK',
      payload: {
        busId,
        track: {
          id: trackId, name: f.name, audioUrl: url, videoUrl: url,
          duration: 0, startPosition: 0, gain: 1.0,
          metadata: { type: 'video_audio', videoUrl: url },
        },
      },
    });
    dispatch({ type: 'SELECT_TRACK', payload: { trackId, busId } });

    // Fire the scoring pipeline. Errors surface via processingError;
    // we don't await before returning so the UI stays responsive.
    try {
      await processVideo(f);
    } catch (_) {
      // already logged inside the hook
    }
  };

  const onDrop = async (e) => {
    e.preventDefault(); e.stopPropagation();
    const f = e.dataTransfer.files?.[0];
    await handleVideoFile(f);
  };

  const baseStatusLine = processingError
    ? `Error: ${processingError}`
    : (processingStatus ? (STATUS_COPY[processingStatus] || processingStatus) : null);

  // Append the per-scene counter when the worker is mid-analysis.
  const statusLine = (() => {
    if (!baseStatusLine) return null;
    if (progress && progress.stage === 'scene' && progress.of) {
      return `${baseStatusLine} (${(progress.i ?? 0) + 1}/${progress.of})`;
    }
    if (progress && progress.stage === 'scene_done' && progress.of) {
      return `${baseStatusLine} (${(progress.i ?? 0) + 1}/${progress.of})`;
    }
    if (progress && progress.stage === 'shots' && progress.count) {
      return `${baseStatusLine} (${progress.count} shots)`;
    }
    return baseStatusLine;
  })();

  if (!videoUrl) {
    return (
      <div
        className="sd-video-empty"
        onDragOver={(e) => { e.preventDefault(); }}
        onDrop={onDrop}
      >
        <div className="sd-wave-empty-eyebrow">— score to picture —</div>
        <div className="sd-wave-empty-title">Drop a video.</div>
        <div className="sd-wave-empty-body">
          Drag an MP4 anywhere on this panel to use it as a reference. The video's playhead will lock to the audio transport, and a draft MIDI score will be generated from its scenes.
        </div>
        <div style={{ marginTop: 16 }}>
          <button className="sd-btn" onClick={() => fileInputRef.current?.click()} disabled={isProcessing}>
            {isProcessing ? 'Working…' : 'Choose video'}
          </button>
          <input
            ref={fileInputRef} type="file" accept="video/*" style={{ display: 'none' }}
            onChange={(e) => {
              const f = e.target.files?.[0]; e.target.value = '';
              if (f) handleVideoFile(f);
            }}
          />
        </div>
        {statusLine && (
          <div className="sd-video-status" style={{ marginTop: 12, opacity: 0.8 }}>
            {statusLine}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="sd-video" onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
      <video
        ref={videoRef} src={videoUrl} className="sd-video-el"
        playsInline muted={false}
      />
      <div className="sd-video-bar">
        <span className="sd-midi-kv-k">Playing</span>
        <span className="sd-midi-name" style={{ fontSize: 13 }}>{selectedTrack?.name}</span>
        <div className="sd-midi-spacer" />
        {statusLine && (
          <>
            <span className="sd-midi-kv-k">{processingError ? '⚠' : '●'}</span>
            <span className="sd-midi-kv-v" style={{ marginRight: 12 }}>{statusLine}</span>
          </>
        )}
        <span className="sd-midi-kv-k">Time</span>
        <span className="sd-midi-kv-v">{(state.playheadPosition || 0).toFixed(2)}s</span>
      </div>
    </div>
  );
}
