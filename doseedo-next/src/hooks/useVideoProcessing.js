import { useState, useCallback, useRef } from 'react';
import { useApp } from '../context/AppContext';
import * as videoAPI from '../services/videoAPI';

/**
 * Custom hook for managing video upload + scoring on the studio drop zone.
 *
 * Flow:
 *   1. POST the file to /api/video-score (SSE; sessionStorage cache by
 *      file fingerprint short-circuits re-uploads of the same clip).
 *   2. Stream events into `processingStatus` so the UI can show
 *      "Detecting scenes (3/12)…" as the worker progresses.
 *   3. On completion: collapse short scenes, compute per-scene tempos,
 *      dispatch SET_VIDEO_INFO + SET_SCENE_CHANGES + UPDATE_TOTAL_DURATION.
 *   4. Decode the returned MIDI base64 into a Blob URL and add it to a
 *      MIDI bus track (so the user immediately sees a generated score
 *      laid under the picture).
 *   5. Optionally extract audio client-side (MediaRecorder) and add to a
 *      VO bus — replaces the GCV-era audio_url returned by the legacy
 *      ScoreAI Celery worker.
 */
export function useVideoProcessing() {
  const { state, dispatch } = useApp();
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [progress, setProgress] = useState(null); // { stage, i?, of?, count? }
  const [processingError, setProcessingError] = useState(null);
  const [videoInfo, setVideoInfo] = useState(null);

  // Latest progress payload, kept in a ref so we don't re-render the
  // calling component on every event in addition to the state set.
  const progressRef = useRef(null);

  const processVideo = useCallback(async (videoFile, opts = {}) => {
    if (!videoFile) return;

    const { extractAudio = true, bpm, baseProgression, framesPerScene } = opts;

    setIsProcessing(true);
    setProcessingError(null);
    setProcessingStatus('uploading');
    setProgress(null);

    try {
      const onProgress = (stage, payload) => {
        progressRef.current = { stage, ...(payload || {}) };
        setProgress(progressRef.current);
        // Map stages to a coarse status string for cheap consumers.
        if (stage === 'shots') setProcessingStatus('detecting_scenes');
        else if (stage === 'scene') setProcessingStatus('analyzing_scenes');
        else if (stage === 'midi') setProcessingStatus('writing_midi');
        else if (stage === 'cache_hit') setProcessingStatus('cache_hit');
      };

      const result = await videoAPI.uploadVideo(videoFile, {
        bpm, baseProgression, framesPerScene, onProgress,
      });
      const { scene_data, scene_changes, duration, midi_base64 } = result;
      const videoId = `vid-${Date.now()}`;

      setVideoInfo({ video_id: videoId, videoFile, scene_data, duration });
      dispatch({
        type: 'SET_VIDEO_INFO',
        payload: { videoId, fileName: videoFile.name, videoFile },
      });

      if (!Array.isArray(scene_changes) || scene_changes.length === 0) {
        setProcessingStatus('completed_no_scenes');
        return result;
      }

      const collapsed = videoAPI.collapseSceneChanges(scene_changes, 3);
      const tempos = videoAPI.computeBestTempos(collapsed);

      dispatch({
        type: 'SET_SCENE_CHANGES',
        payload: {
          sceneChanges: collapsed,
          sceneTempos: tempos,
          videoDuration: duration || collapsed[collapsed.length - 1],
        },
      });

      if (duration) {
        dispatch({ type: 'UPDATE_TOTAL_DURATION', payload: duration });
      } else if (collapsed.length > 0) {
        dispatch({
          type: 'UPDATE_TOTAL_DURATION',
          payload: collapsed[collapsed.length - 1],
        });
      }

      // Add the generated MIDI score to a MIDI bus track.
      if (midi_base64) {
        const midiBlob = videoAPI.decodeMidiBase64(midi_base64);
        const midiUrl = URL.createObjectURL(midiBlob);

        let midiBusId = state.buses?.find(b => b.type === 'MIDI')?.id;
        if (!midiBusId) {
          midiBusId = `midi-${Date.now()}`;
          dispatch({
            type: 'CREATE_BUS',
            payload: { id: midiBusId, type: 'MIDI', name: 'MIDI 1' },
          });
        }

        dispatch({
          type: 'ADD_TRACK',
          payload: {
            busId: midiBusId,
            track: {
              id: `video-score-${Date.now()}`,
              name: `${videoFile.name.replace(/\.[^/.]+$/, '')} (Score)`,
              midiUrl,
              duration: duration || collapsed[collapsed.length - 1],
              startPosition: 0,
              gain: 1.0,
              metadata: {
                type: 'midi',
                source: 'video_score',
                sceneData: scene_data,
              },
            },
          },
        });
      }

      // Best-effort client-side audio extraction → VO bus.
      if (extractAudio) {
        try {
          setProcessingStatus('extracting_audio');
          const audioBlob = await videoAPI.extractAudioFromVideo(videoFile);
          const audioUrl = URL.createObjectURL(audioBlob);

          let voBusId = state.buses?.find(b => b.type === 'VO')?.id;
          if (!voBusId) {
            voBusId = `vo-${Date.now()}`;
            dispatch({
              type: 'CREATE_BUS',
              payload: { id: voBusId, type: 'VO', name: 'VO 1' },
            });
          }
          dispatch({
            type: 'ADD_TRACK',
            payload: {
              busId: voBusId,
              track: {
                id: `video-audio-${Date.now()}`,
                name: videoFile.name.replace(/\.[^/.]+$/, '') + ' (Audio)',
                audioUrl,
                duration: duration || collapsed[collapsed.length - 1],
                startPosition: 0,
                gain: 1.0,
                isMuted: false,
                isSolo: false,
                cropStart: 0,
                cropEnd: 0,
                fx: { reverb: 0, fadeIn: 0.2, fadeOut: 1.0 },
                metadata: { type: 'video_audio', videoFileName: videoFile.name },
              },
            },
          });
        } catch (audioErr) {
          // eslint-disable-next-line no-console
          console.warn('[useVideoProcessing] audio extract failed:', audioErr);
        }
      }

      setProcessingStatus('completed');
      return result;
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('[useVideoProcessing] error:', error);
      setProcessingError(error.message || 'Video processing failed');
      setProcessingStatus('failed');
      throw error;
    } finally {
      setIsProcessing(false);
    }
  }, [dispatch, state.buses]);

  const clearVideo = useCallback(() => {
    setIsProcessing(false);
    setProcessingStatus(null);
    setProcessingError(null);
    setProgress(null);
    setVideoInfo(null);
    dispatch({ type: 'CLEAR_VIDEO' });
  }, [dispatch]);

  return {
    isProcessing,
    processingStatus,
    progress,
    processingError,
    videoInfo,
    processVideo,
    clearVideo,
  };
}
