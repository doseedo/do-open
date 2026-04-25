import { useState, useCallback } from 'react';
import { useApp } from '../context/AppContext';
import * as videoAPI from '../services/videoAPI';

/**
 * Custom hook for managing video upload and processing
 * Handles: upload, scene detection, audio extraction, and state management
 */
export function useVideoProcessing() {
  const { state, dispatch } = useApp();
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [processingError, setProcessingError] = useState(null);
  const [videoInfo, setVideoInfo] = useState(null);

  /**
   * Process uploaded video file
   * @param {File} videoFile - Video file to process
   * @param {boolean} useWhisper - Whether to use Whisper for transcription
   * @returns {Promise<void>}
   */
  const processVideo = useCallback(async (videoFile, useWhisper = false) => {
    console.log('🎬 Starting video processing...');

    // Reset state
    setIsProcessing(true);
    setProcessingError(null);
    setProcessingStatus('uploading');

    try {
      // 1. Upload video to backend
      const uploadResult = await videoAPI.uploadVideo(videoFile, useWhisper);
      const { video_id, task_id } = uploadResult;

      console.log(`✅ Video uploaded: ${video_id}`);
      setVideoInfo({ video_id, videoFile });

      // Store video ID and file in state
      dispatch({
        type: 'SET_VIDEO_INFO',
        payload: { videoId: video_id, fileName: videoFile.name, videoFile: videoFile }
      });

      // 2. Poll for scene detection results
      setProcessingStatus('detecting_scenes');

      const result = await videoAPI.pollVideoUntilComplete(
        task_id,
        (progress) => {
          setProcessingStatus(`processing (${progress.attempts} attempts)`);
        }
      );

      console.log('✅ Scene detection complete!');

      // 4. Process scene changes and audio
      const { scene_changes, video_duration, audio_url } = result;

      if (!scene_changes || scene_changes.length === 0) {
        console.warn('⚠️ No scene changes detected');
        setProcessingStatus('completed_no_scenes');
        return;
      }

      // 5. Collapse short scenes (merge scenes closer than 3 seconds)
      const collapsed = videoAPI.collapseSceneChanges(scene_changes, 3);
      console.log(`🎬 Scene changes: ${scene_changes.length} → ${collapsed.length} (after collapse)`);

      // 6. Compute optimal tempos for each scene
      const tempos = videoAPI.computeBestTempos(collapsed);
      console.log('🎵 Optimal tempos computed:', tempos);

      // Log detailed breakdown
      for (let i = 0; i < tempos.length; i++) {
        const start = collapsed[i];
        const end = collapsed[i + 1];
        const duration = end - start;
        console.log(`   Scene ${i + 1}: ${start.toFixed(2)}s - ${end.toFixed(2)}s (${duration.toFixed(2)}s) → ${tempos[i]} BPM`);
      }

      // 7. Store in app state
      dispatch({
        type: 'SET_SCENE_CHANGES',
        payload: {
          sceneChanges: collapsed,
          sceneTempos: tempos,
          videoDuration: video_duration || collapsed[collapsed.length - 1]
        }
      });

      // 8. Create VO bus track from backend audio (if available)
      if (audio_url) {
        console.log('🎵 Video audio URL from backend:', audio_url);

        // Get or create VO bus
        let voBusId = state.buses?.find(b => b.type === 'VO')?.id;

        if (!voBusId) {
          voBusId = `vo-${Date.now()}`;
          dispatch({
            type: 'CREATE_BUS',
            payload: { id: voBusId, type: 'VO', name: 'VO 1' }
          });
          console.log('🆕 Created VO bus:', voBusId);
        }

        // Create track from video audio
        const track = {
          id: `video-audio-${Date.now()}`,
          name: videoFile.name.replace(/\.[^/.]+$/, '') + ' (Audio)',
          audioUrl: audio_url,
          duration: video_duration || collapsed[collapsed.length - 1],
          startPosition: 0,
          gain: 1.0,
          isMuted: false,
          isSolo: false,
          cropStart: 0,
          cropEnd: 0,
          fx: {
            reverb: 0,
            fadeIn: 0.2,
            fadeOut: 1.0
          },
          // Mark as video audio for sidebar detection
          metadata: {
            type: 'video_audio',
            videoFileName: videoFile.name
          }
        };

        dispatch({
          type: 'ADD_TRACK',
          payload: { busId: voBusId, track }
        });

        console.log('✅ Added video audio track to VO bus');
      } else {
        console.warn('⚠️ No audio URL returned from backend');
      }

      // 9. Update total duration based on video
      if (video_duration) {
        console.log(`⏱️  Updating total duration to ${video_duration}s`);
        dispatch({
          type: 'UPDATE_TOTAL_DURATION',
          payload: video_duration
        });
      } else if (collapsed.length > 0) {
        // Use last scene change time if no explicit duration
        const lastSceneTime = collapsed[collapsed.length - 1];
        console.log(`⏱️  Updating total duration to last scene time: ${lastSceneTime}s`);
        dispatch({
          type: 'UPDATE_TOTAL_DURATION',
          payload: lastSceneTime
        });
      }

      setProcessingStatus('completed');
      console.log('✅ Video processing complete!');

    } catch (error) {
      console.error('❌ Video processing error:', error);
      setProcessingError(error.message || 'Video processing failed');
      setProcessingStatus('failed');
    } finally {
      setIsProcessing(false);
    }
  }, [dispatch, state.buses]);

  /**
   * Clear video processing state
   */
  const clearVideo = useCallback(() => {
    setIsProcessing(false);
    setProcessingStatus(null);
    setProcessingError(null);
    setVideoInfo(null);

    dispatch({
      type: 'CLEAR_VIDEO'
    });
  }, [dispatch]);

  return {
    // State
    isProcessing,
    processingStatus,
    processingError,
    videoInfo,

    // Actions
    processVideo,
    clearVideo
  };
}
