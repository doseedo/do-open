/*
 * StudioDevVideo — themed video/reference viewer for /studio-dev.
 * Drop a video file or MP4 URL on the track to use it as a scoring
 * reference. Playhead syncs to state.playheadPosition so audio + video stay
 * in lockstep.
 */
import React, { useEffect, useRef } from 'react';
import { useApp } from '../../context/AppContext';

export default function StudioDevVideo() {
  const { state, dispatch } = useApp();
  const selectedTrack = state.selectedTrack;
  const videoRef = useRef(null);
  const fileInputRef = useRef(null);

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

  const onDrop = async (e) => {
    e.preventDefault(); e.stopPropagation();
    const f = e.dataTransfer.files?.[0];
    if (!f || !f.type.startsWith('video/')) return;
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
  };

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
          Drag an MP4 anywhere on this panel to use it as a reference. The video's playhead will lock to the audio transport.
        </div>
        <div style={{ marginTop: 16 }}>
          <button className="sd-btn" onClick={() => fileInputRef.current?.click()}>Choose video</button>
          <input
            ref={fileInputRef} type="file" accept="video/*" style={{ display: 'none' }}
            onChange={(e) => {
              const f = e.target.files?.[0]; e.target.value = '';
              if (!f) return;
              onDrop({ preventDefault(){}, stopPropagation(){}, dataTransfer: { files: [f] } });
            }}
          />
        </div>
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
        <span className="sd-midi-kv-k">Time</span>
        <span className="sd-midi-kv-v">{(state.playheadPosition || 0).toFixed(2)}s</span>
      </div>
    </div>
  );
}
